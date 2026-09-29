package usecase

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"slices"
	"strconv"
	"strings"
	"sync"
	"time"

	"distillery/internal/domain"
)

// MaxTableRows caps imported rows in the MVP per the plan's "very large
// tables" risk item; larger uploads are rejected with a clear error.
const MaxTableRows = 200000

// minTrainRows is the smallest table a tabular/forecast job accepts.
const minTrainRows = 30

const (
	defaultCVFolds       = 5
	defaultTimeBudgetSec = 60
	maxTimeBudgetSec     = 3600
	maxForecastHorizon   = 10000
	rowCacheEntries      = 3
)

// TabularUsecase implements the plan-08 surface: table upload with column
// type inference and preview, tabular/forecast training dispatch, and the
// shared row source used by inference engines. Rows live in the blob store
// (one JSON-lines blob per table); the repository holds metadata only.
type TabularUsecase struct {
	tasks  domain.TaskRepository
	tables domain.TableDatasetRepository
	jobs   domain.TrainingJobRepository
	blobs  domain.BlobStore
	tuner  domain.FineTuner
	idGen  IDGenerator

	// jobMu guards mutation of jobs that training goroutines update while
	// HTTP handlers read snapshots of them.
	jobMu sync.Mutex

	cacheMu    sync.Mutex
	cache      map[string]*rowCacheEntry
	cacheOrder []string
}

type rowCacheEntry struct {
	blobKey string
	rows    []tableRow
}

// NewTabularUsecase wires the tabular/time-series usecase. The tuner must
// handle KindTabular and KindTimeSeries jobs (see infra/tabular.Router).
func NewTabularUsecase(
	tasks domain.TaskRepository,
	tables domain.TableDatasetRepository,
	jobs domain.TrainingJobRepository,
	blobs domain.BlobStore,
	tuner domain.FineTuner,
	idGen IDGenerator,
) *TabularUsecase {
	return &TabularUsecase{
		tasks: tasks, tables: tables, jobs: jobs, blobs: blobs, tuner: tuner, idGen: idGen,
		cache: map[string]*rowCacheEntry{},
	}
}

// ---------- Upload + mapping ----------.

// UploadTableRequest is the parsed CSV upload. Optional fields prefill the
// mapping; everything can be changed afterwards with UpdateTableMapping.
type UploadTableRequest struct {
	Filename  string
	Content   []byte
	Target    string
	Timestamp string
	ItemIDCol string
	Frequency string
	Horizon   int
}

func tableKindOK(k domain.ModelKind) bool {
	return k == domain.KindTabular || k == domain.KindTimeSeries
}

// UploadTable parses a CSV, infers column types, applies leakage warnings,
// stores the rows in the blob store, and returns the TableDataset for the
// mapper UI. The table kind always follows the task's kind.
func (u *TabularUsecase) UploadTable(taskID string, req UploadTableRequest) (*domain.TableDataset, error) {
	task, err := u.tasks.Get(taskID)
	if err != nil {
		return nil, err
	}

	if !tableKindOK(task.Kind) {
		return nil, fmt.Errorf("%w: task kind %q cannot upload a table", domain.ErrInvalidInput, task.Kind)
	}

	rows, headers, err := parseCSV(req.Content)
	if err != nil {
		return nil, err
	}

	if len(rows) == 0 {
		return nil, fmt.Errorf("%w: table has no data rows", domain.ErrInvalidInput)
	}

	if len(rows) > MaxTableRows {
		return nil, fmt.Errorf("%w: table has %d rows; the MVP caps uploads at %d — sample or split the file",
			domain.ErrInvalidInput, len(rows), MaxTableRows)
	}

	td := &domain.TableDataset{
		ID:        u.idGen.NewID("table"),
		TaskID:    taskID,
		Filename:  req.Filename,
		Kind:      task.Kind,
		RowCount:  len(rows),
		Columns:   inferColumns(headers, rows),
		Horizon:   req.Horizon,
		CreatedAt: time.Now().UTC(),
	}

	// Normalise every datetime column to ISO so all consumers parse alike.
	for i := range td.Columns {
		if td.Columns[i].Type != domain.ColDatetime {
			continue
		}

		if !normalizeDatetime(rows, td.Columns[i].Name) {
			td.Columns[i].Type = domain.ColCategorical
		}
	}

	m := MappingUpdate{
		Target: req.Target, Timestamp: req.Timestamp, ItemID: req.ItemIDCol,
		Frequency: req.Frequency, Horizon: req.Horizon,
	}

	err = u.applyMapping(td, rows, &m)
	if err != nil {
		return nil, err
	}

	if td.Kind == domain.KindTabular {
		td.Extra.Split = string(u.defaultSplit(td))
	}

	err = u.storeRows(td, rows)
	if err != nil {
		return nil, err
	}

	err = u.tables.Create(td)
	if err != nil {
		_ = u.blobs.Delete(taskID, td.BlobKey)

		return nil, err
	}

	return td, nil
}

// MappingUpdate is the column-mapper payload. Nil pointers / empty strings
// leave the current value untouched.
type MappingUpdate struct {
	Target      string
	Types       map[string]domain.TableColumnType
	Excluded    *[]string
	Timestamp   string
	ItemID      string
	Split       domain.SplitStrategy
	GroupColumn *string
	Task        string
	Frequency   string
	Horizon     int
}

// UpdateTableMapping applies the user's column-mapper edits and recomputes
// leakage warnings. Retyping a column to datetime normalises its values.
func (u *TabularUsecase) UpdateTableMapping(taskID, tableID string, m MappingUpdate) (*domain.TableDataset, error) {
	td, err := u.GetTable(taskID, tableID)
	if err != nil {
		return nil, err
	}

	rows, err := u.loadRows(td)
	if err != nil {
		return nil, err
	}

	// Work on copies so a validation failure leaves the stored dataset as is.
	cp := *td
	cp.Columns = slices.Clone(td.Columns)
	cp.Excluded = slices.Clone(td.Excluded)

	rewrite := false

	for name := range m.Types {
		if !td.HasColumn(name) {
			return nil, fmt.Errorf("%w: unknown column %q", domain.ErrInvalidInput, name)
		}
	}

	for i := range cp.Columns {
		t, ok := m.Types[cp.Columns[i].Name]
		if !ok || t == cp.Columns[i].Type {
			continue
		}

		if !domain.IsValidTableColumnType(t) {
			return nil, fmt.Errorf("%w: unknown column type %q", domain.ErrInvalidInput, t)
		}

		rows, err = u.retypeColumn(&cp.Columns[i], t, rows, &rewrite)
		if err != nil {
			return nil, err
		}
	}

	err = u.applyMapping(&cp, rows, &m)
	if err != nil {
		return nil, err
	}

	if rewrite {
		err = u.storeRows(&cp, rows)
		if err != nil {
			return nil, err
		}
	}

	err = u.tables.Update(&cp)
	if err != nil {
		return nil, err
	}

	return &cp, nil
}

func validFrequency(f string) bool {
	switch f {
	case "H", "D", "W", "M", "Q", "Y":
		return true
	}

	return false
}

// validateMappingConsistency checks the cross-field rules of a mapping.
func validateMappingConsistency(td *domain.TableDataset) error {
	bad := func(format string, a ...interface{}) error {
		return fmt.Errorf("%w: %s", domain.ErrInvalidInput, fmt.Sprintf(format, a...))
	}

	if td.Target != "" && td.ColumnType(td.Target) == domain.ColDatetime {
		return bad("target %q is a datetime column; use a time_series task to forecast over time", td.Target)
	}

	if td.Target != "" && slices.Contains(td.Excluded, td.Target) {
		return bad("target %q cannot also be excluded", td.Target)
	}

	if td.Kind == domain.KindTimeSeries {
		if td.Target != "" && td.ColumnType(td.Target) != domain.ColNumeric {
			return bad("forecast target %q must be numeric", td.Target)
		}

		if t := td.Extra.TimestampColumn; t != "" && td.ColumnType(t) != domain.ColDatetime {
			return bad("timestamp column %q is not a datetime column", t)
		}
	}

	switch domain.SplitStrategy(td.Extra.Split) {
	case domain.SplitGroup:
		if td.Extra.GroupColumn == "" {
			return bad("group split needs a group column")
		}
	case domain.SplitTime:
		if !hasDatetime(td) {
			return bad("time split needs a datetime column")
		}
	case domain.SplitRandom, domain.SplitStratified:
	}

	return nil
}

func hasDatetime(td *domain.TableDataset) bool {
	for _, c := range td.Columns {
		if c.Type == domain.ColDatetime && c.Name != td.Target {
			return true
		}
	}

	return false
}

// GetTable returns a table dataset (mapper/preview).
func (u *TabularUsecase) GetTable(taskID, tableID string) (*domain.TableDataset, error) {
	td, err := u.tables.Get(tableID)
	if err != nil {
		return nil, err
	}

	if td.TaskID != taskID {
		return nil, domain.ErrNotFound
	}

	return td, nil
}

// ListTables lists a task's table datasets.
func (u *TabularUsecase) ListTables(taskID string) ([]*domain.TableDataset, error) {
	return u.tables.ListByTask(taskID)
}

// DeleteTable removes a table and its rows unless a job is training on it.
func (u *TabularUsecase) DeleteTable(taskID, tableID string) error {
	td, err := u.GetTable(taskID, tableID)
	if err != nil {
		return err
	}

	jobs, _ := u.jobs.ListByTask(taskID)
	for _, j := range jobs {
		if j.TableDatasetID == tableID && j.Status == domain.TrainingRunning {
			return domain.ErrAlreadyRunning
		}
	}

	_ = u.blobs.Delete(taskID, td.BlobKey)

	u.cacheMu.Lock()
	delete(u.cache, tableID)
	u.cacheMu.Unlock()

	return u.tables.Delete(tableID)
}

// TablePreview returns the first N parsed rows for the preview UI, in table
// column order metadata alongside.
func (u *TabularUsecase) TablePreview(taskID, tableID string, limit int) ([]map[string]interface{}, error) {
	td, err := u.GetTable(taskID, tableID)
	if err != nil {
		return nil, err
	}

	data, err := u.blobs.Get(taskID, td.BlobKey)
	if err != nil {
		return nil, err
	}

	return decodeRows(data, limit)
}

// ---------- Row storage (blob store, JSON lines) ----------.

func decodeRows(data []byte, limit int) ([]tableRow, error) {
	sc := bufio.NewScanner(bytes.NewReader(data))
	sc.Buffer(make([]byte, 0, 1<<20), 16<<20)

	var rows []tableRow

	for sc.Scan() {
		line := bytes.TrimSpace(sc.Bytes())
		if len(line) == 0 {
			continue
		}

		var r tableRow

		err := json.Unmarshal(line, &r)
		if err != nil {
			return nil, fmt.Errorf("corrupt table blob: %w", err)
		}

		rows = append(rows, r)

		if limit > 0 && len(rows) >= limit {
			break
		}
	}

	return rows, sc.Err()
}

// JobRows implements domain.TableRowSource: the training rows of the table a
// job was trained on.
func (u *TabularUsecase) JobRows(job *domain.TrainingJob) ([]map[string]interface{}, error) {
	td, err := u.tables.Get(job.TableDatasetID)
	if err != nil {
		return nil, err
	}

	return u.loadRows(td)
}

// ---------- Training dispatch ----------.

var (
	classificationMetrics = []string{"roc_auc", "pr_auc", "f1", "accuracy", "logloss"}
	regressionMetrics     = []string{"rmse", "mae", "r2"}
	tabularModels         = []string{"", "auto", "lightgbm", "xgboost", "catboost"}
	forecastModels        = []string{"", "auto", "seasonal_naive", "stats", "gbm", "chronos"}
	forecastMetrics       = []string{"", "mase", "smape", "wql"}
)

// tabularBaseModel is the synthetic "base model" recorded on tabular jobs:
// model families are chosen by cross-validation, not by parameter count.
var tabularBaseModel = domain.BaseModel{
	Name: "gbm-cv", Family: "gbm", Kind: domain.KindTabular,
	Capabilities: domain.Capabilities{RunsOnCPU: true},
}

var forecastBaseModel = domain.BaseModel{
	Name: "forecast-backtest", Family: "forecast", Kind: domain.KindTimeSeries,
	Capabilities: domain.Capabilities{RunsOnCPU: true},
}

// inferTableTask guesses classification vs regression from the target.
func inferTableTask(td *domain.TableDataset) string {
	if td.Extra.Task != "" {
		return td.Extra.Task
	}

	for _, c := range td.Columns {
		if c.Name != td.Target {
			continue
		}

		// Numeric targets with few integer levels (0/1 churn flags, ratings)
		// are class labels, everything else numeric is a regression.
		if c.Type == domain.ColNumeric && (!c.IntegerLike || c.Unique > 10) {
			return domain.TabularRegression
		}

		return domain.TabularClassification
	}

	return domain.TabularClassification
}

func oneOf(v string, allowed []string) bool { return slices.Contains(allowed, v) }

// StartTabularTraining validates the table and launches a tabular training
// run (classification or regression). Target, exclusions and column types
// always come from the stored mapping, never from the request.
func (u *TabularUsecase) StartTabularTraining(
	taskID, tableID string, cfg domain.TabularConfig,
) (*domain.TrainingJob, error) {
	td, rows, err := u.prepareTraining(taskID, tableID, domain.KindTabular)
	if err != nil {
		return nil, err
	}

	bad := func(format string, a ...interface{}) error {
		return fmt.Errorf("%w: %s", domain.ErrInvalidInput, fmt.Sprintf(format, a...))
	}

	if cfg.Task == "" {
		cfg.Task = inferTableTask(td)
	}

	tcol := td.Columns[indexOfColumn(td, td.Target)]
	if cfg.Task == domain.TabularRegression && tcol.Type != domain.ColNumeric {
		return nil, bad("regression needs a numeric target; %q is %s", td.Target, tcol.Type)
	}

	if cfg.Task != domain.TabularRegression && cfg.Task != domain.TabularClassification {
		return nil, bad("task must be %q or %q", domain.TabularClassification, domain.TabularRegression)
	}

	if cfg.Task == domain.TabularClassification && (tcol.Unique < 2 || tcol.Unique > 200) {
		return nil, bad("classification target %q has %d distinct values (need 2–200)", td.Target, tcol.Unique)
	}

	metrics := classificationMetrics
	if cfg.Task == domain.TabularRegression {
		metrics = regressionMetrics
	}

	if cfg.Metric != "" && !oneOf(cfg.Metric, metrics) {
		return nil, bad("metric %q is not valid for %s (use %s)", cfg.Metric, cfg.Task, strings.Join(metrics, ", "))
	}

	if !oneOf(cfg.Model, tabularModels) {
		return nil, bad("model must be auto, lightgbm, xgboost or catboost")
	}

	if cfg.Model == "" {
		cfg.Model = "lightgbm"
	}

	cfg.SplitStrategy, cfg.GroupColumn, err = resolveSplit(td, &cfg)
	if err != nil {
		return nil, err
	}

	cfg.CVFolds = clampInt(cfg.CVFolds, defaultCVFolds, 2, 10)
	cfg.TimeBudgetSec = clampInt(cfg.TimeBudgetSec, defaultTimeBudgetSec, 5, maxTimeBudgetSec)
	cfg.Target = td.Target
	cfg.Columns = map[string]domain.TableColumnType{}
	cfg.ExcludeColumns = nil

	for _, c := range td.Columns {
		cfg.Columns[c.Name] = c.Type

		if c.Name != td.Target && (td.IsExcluded(c.Name) || c.Name == td.Extra.GroupColumn) {
			cfg.ExcludeColumns = append(cfg.ExcludeColumns, c.Name)
		}
	}

	job := &domain.TrainingJob{
		ID:             u.idGen.NewID("job"),
		TaskID:         taskID,
		Version:        u.nextVersion(taskID),
		Kind:           domain.KindTabular,
		BaseModel:      tabularBaseModel,
		Status:         domain.TrainingRunning,
		Tabular:        &cfg,
		TableDatasetID: tableID,
		CreatedAt:      time.Now().UTC(),
		StartedAt:      ptrTime(time.Now().UTC()),
	}

	return u.launch(job, td, rows)
}

func indexOfColumn(td *domain.TableDataset, name string) int {
	for i, c := range td.Columns {
		if c.Name == name {
			return i
		}
	}

	return 0
}

func clampInt(v, def, lo, hi int) int {
	if v <= 0 {
		v = def
	}

	return max(lo, min(hi, v))
}

// resolveSplit picks and validates the validation split strategy.
func resolveSplit(td *domain.TableDataset, cfg *domain.TabularConfig) (domain.SplitStrategy, string, error) {
	s := cfg.SplitStrategy
	if s == "" {
		s = domain.SplitStrategy(td.Extra.Split)
	}

	group := cfg.GroupColumn
	if group == "" {
		group = td.Extra.GroupColumn
	}

	if s == "" {
		switch {
		case hasDatetime(td):
			s = domain.SplitTime
		case cfg.Task == domain.TabularClassification:
			s = domain.SplitStratified
		default:
			s = domain.SplitRandom
		}
	}

	bad := func(msg string) error { return fmt.Errorf("%w: %s", domain.ErrInvalidInput, msg) }

	switch s {
	case domain.SplitGroup:
		if group == "" || !td.HasColumn(group) {
			return "", "", bad("group split needs an existing group column")
		}
	case domain.SplitTime:
		if !hasDatetime(td) {
			return "", "", bad("time split needs a datetime column")
		}
	case domain.SplitStratified:
		if cfg.Task != domain.TabularClassification {
			return "", "", bad("stratified split is only for classification")
		}
	case domain.SplitRandom:
	default:
		return "", "", bad(fmt.Sprintf("unknown split strategy %q", s))
	}

	return s, group, nil
}

// StartForecastTraining validates the series and launches a forecasting run.
func (u *TabularUsecase) StartForecastTraining(
	taskID, tableID string, cfg domain.ForecastConfig,
) (*domain.TrainingJob, error) {
	td, rows, err := u.prepareTraining(taskID, tableID, domain.KindTimeSeries)
	if err != nil {
		return nil, err
	}

	bad := func(format string, a ...interface{}) error {
		return fmt.Errorf("%w: %s", domain.ErrInvalidInput, fmt.Sprintf(format, a...))
	}

	if td.Extra.TimestampColumn == "" {
		return nil, bad("pick the timestamp column first")
	}

	if cfg.Horizon <= 0 {
		cfg.Horizon = td.Horizon
	}

	if cfg.Horizon <= 0 || cfg.Horizon > maxForecastHorizon {
		return nil, bad("choose a forecast horizon between 1 and %d", maxForecastHorizon)
	}

	cfg.Frequency = strings.ToUpper(strings.TrimSpace(cfg.Frequency))

	switch {
	case cfg.Frequency != "":
		if !validFrequency(cfg.Frequency) {
			return nil, bad("frequency must be one of H, D, W, M, Q, Y")
		}
	case td.Frequency != "" && td.FrequencyConfirmed:
		cfg.Frequency = td.Frequency
	default:
		return nil, bad("confirm the series frequency first (detected: %q)", td.Frequency)
	}

	if !oneOf(cfg.Model, forecastModels) {
		return nil, bad("model must be auto, seasonal_naive, stats, gbm or chronos")
	}

	if !oneOf(cfg.Metric, forecastMetrics) {
		return nil, bad("metric must be mase, smape or wql")
	}

	if cfg.SeasonLength < 0 || cfg.SeasonLength > 400 {
		return nil, bad("season_length must be between 0 and 400")
	}

	if cfg.Metric == "" {
		cfg.Metric = "mase"
	}

	if cfg.Model == "" {
		cfg.Model = "auto"
	}

	cfg.BacktestWindows = clampInt(cfg.BacktestWindows, 3, 1, 12)
	cfg.TimeBudgetSec = clampInt(cfg.TimeBudgetSec, defaultTimeBudgetSec, 5, maxTimeBudgetSec)
	cfg.Timestamp, cfg.Target, cfg.ItemID = td.Extra.TimestampColumn, td.Target, td.Extra.ItemIDColumn

	job := &domain.TrainingJob{
		ID:             u.idGen.NewID("job"),
		TaskID:         taskID,
		Version:        u.nextVersion(taskID),
		Kind:           domain.KindTimeSeries,
		BaseModel:      forecastBaseModel,
		Status:         domain.TrainingRunning,
		Forecast:       &cfg,
		TableDatasetID: tableID,
		CreatedAt:      time.Now().UTC(),
		StartedAt:      ptrTime(time.Now().UTC()),
	}

	return u.launch(job, td, rows)
}

// retypeColumn switches a column to a new type, validating (and for
// datetime, normalising) its values.
func (u *TabularUsecase) retypeColumn(
	col *domain.TableColumn, t domain.TableColumnType, rows []tableRow, rewrite *bool,
) ([]tableRow, error) {
	switch t {
	case domain.ColDatetime:
		if !normalizeDatetime(rows, col.Name) {
			return nil, fmt.Errorf("%w: column %q does not look like a date/time", domain.ErrInvalidInput, col.Name)
		}

		*rewrite = true
	case domain.ColNumeric:
		n, total := 0, 0

		for _, r := range rows {
			if s, _ := r[col.Name].(string); s != "" {
				total++

				_, err := strconv.ParseFloat(s, 64)
				if err == nil {
					n++
				}
			}
		}

		if total == 0 || n*2 < total {
			return nil, fmt.Errorf("%w: column %q is not numeric", domain.ErrInvalidInput, col.Name)
		}
	case domain.ColCategorical, domain.ColText, domain.ColID, domain.ColIgnored:
	}

	col.Type = t

	return rows, nil
}

// applyMapping validates and applies target/timestamp/item/split/etc. to td
// and recomputes warnings. It is shared by upload and mapping updates.
//
//nolint:cyclop,gocognit // sequential pipeline; splitting adds indirection without clarity
func (u *TabularUsecase) applyMapping(td *domain.TableDataset, rows []tableRow, m *MappingUpdate) error {
	bad := func(format string, a ...interface{}) error {
		return fmt.Errorf("%w: %s", domain.ErrInvalidInput, fmt.Sprintf(format, a...))
	}

	need := func(kind, name string) error {
		if name != "" && !td.HasColumn(name) {
			return bad("%s column %q does not exist", kind, name)
		}

		return nil
	}

	for _, c := range []struct{ kind, name string }{
		{"target", m.Target}, {"timestamp", m.Timestamp}, {"item_id", m.ItemID},
	} {
		err := need(c.kind, c.name)
		if err != nil {
			return err
		}
	}

	if m.GroupColumn != nil {
		err := need("group", *m.GroupColumn)
		if err != nil {
			return err
		}

		td.Extra.GroupColumn = *m.GroupColumn
	}

	if m.Excluded != nil {
		for _, e := range *m.Excluded {
			err := need("excluded", e)
			if err != nil {
				return err
			}
		}

		td.Excluded = slices.Clone(*m.Excluded)
	}

	if m.Target != "" {
		td.Target = m.Target
	}

	if m.Timestamp != "" {
		td.Extra.TimestampColumn = m.Timestamp
	}

	if m.ItemID != "" {
		td.Extra.ItemIDColumn = m.ItemID
	}

	if m.Task != "" {
		if m.Task != domain.TabularClassification && m.Task != domain.TabularRegression {
			return bad("task must be %q or %q", domain.TabularClassification, domain.TabularRegression)
		}

		td.Extra.Task = m.Task
	}

	if m.Split != "" {
		if !domain.IsValidSplitStrategy(m.Split) {
			return bad("unknown split strategy %q", m.Split)
		}

		td.Extra.Split = string(m.Split)
	}

	if m.Horizon > 0 {
		td.Horizon = m.Horizon
	}

	if td.Kind == domain.KindTimeSeries {
		u.defaultSeriesColumns(td)
	}

	if td.Target != "" {
		for i := range td.Columns {
			if td.Columns[i].Name == td.Target {
				retypeTarget(&td.Columns[i], rows)
			}
		}
	}

	err := validateMappingConsistency(td)
	if err != nil {
		return err
	}

	// Frequency: an explicit value is confirmed by definition; otherwise it is
	// inferred and stays unconfirmed until the user re-submits it.
	if f := strings.ToUpper(strings.TrimSpace(m.Frequency)); f != "" {
		if !validFrequency(f) {
			return bad("frequency must be one of H, D, W, M, Q, Y")
		}

		td.Frequency, td.FrequencyConfirmed = f, true
	} else if td.Frequency == "" && td.Kind == domain.KindTimeSeries {
		td.Frequency = inferFrequency(rows, td.Extra.TimestampColumn)
	}

	for _, c := range td.Columns {
		if c.Type == domain.ColDatetime {
			td.Extra.HasDatetime = true

			break
		}
	}

	warnLeakage(td, rows)

	return nil
}

// defaultSeriesColumns fills in the timestamp and target of a time-series
// table when the caller did not choose them.
func (u *TabularUsecase) defaultSeriesColumns(td *domain.TableDataset) {
	if td.Extra.TimestampColumn == "" {
		for _, c := range td.Columns {
			if c.Type == domain.ColDatetime {
				td.Extra.TimestampColumn = c.Name

				break
			}
		}
	}

	if td.Target == "" {
		for _, c := range td.Columns {
			if c.Type == domain.ColNumeric && c.Name != td.Extra.ItemIDColumn {
				td.Target = c.Name

				break
			}
		}
	}
}

// defaultSplit is time-based whenever a datetime column exists, otherwise
// random (stratified is chosen at training time for classification).
func (u *TabularUsecase) defaultSplit(td *domain.TableDataset) domain.SplitStrategy {
	if hasDatetime(td) {
		return domain.SplitTime
	}

	return domain.SplitRandom
}

func (u *TabularUsecase) storeRows(td *domain.TableDataset, rows []tableRow) error {
	var buf bytes.Buffer

	enc := json.NewEncoder(&buf)

	for _, r := range rows {
		err := enc.Encode(r)
		if err != nil {
			return fmt.Errorf("%w: cannot encode row: %w", domain.ErrInvalidInput, err)
		}
	}

	key, err := u.blobs.Put(td.TaskID, buf.Bytes(), ".jsonl")
	if err != nil {
		return err
	}

	old := td.BlobKey
	td.BlobKey = key
	td.RowCount = len(rows)

	if old != "" && old != key {
		_ = u.blobs.Delete(td.TaskID, old)
	}

	u.cacheMu.Lock()
	delete(u.cache, td.ID)
	u.cacheMu.Unlock()

	return nil
}

// loadRows returns all rows of a table (cached; the last few tables stay hot).
func (u *TabularUsecase) loadRows(td *domain.TableDataset) ([]tableRow, error) {
	u.cacheMu.Lock()
	e, ok := u.cache[td.ID]
	u.cacheMu.Unlock()

	if ok && e.blobKey == td.BlobKey {
		return e.rows, nil
	}

	data, err := u.blobs.Get(td.TaskID, td.BlobKey)
	if err != nil {
		return nil, err
	}

	rows, err := decodeRows(data, 0)
	if err != nil {
		return nil, err
	}

	u.cacheMu.Lock()
	defer u.cacheMu.Unlock()

	if _, exists := u.cache[td.ID]; !exists {
		u.cacheOrder = append(u.cacheOrder, td.ID)
	}

	u.cache[td.ID] = &rowCacheEntry{blobKey: td.BlobKey, rows: rows}

	for len(u.cacheOrder) > rowCacheEntries {
		delete(u.cache, u.cacheOrder[0])
		u.cacheOrder = u.cacheOrder[1:]
	}

	return rows, nil
}

// rowsAsExamples wraps rows as ephemeral examples for the FineTuner contract.
// Nothing is stored in the example repository.
func (u *TabularUsecase) rowsAsExamples(td *domain.TableDataset, rows []tableRow) ([]*domain.Example, error) {
	now := time.Now().UTC()
	out := make([]*domain.Example, 0, len(rows))

	for i, r := range rows {
		b, err := json.Marshal(r)
		if err != nil {
			return nil, fmt.Errorf("%w: cannot encode row %d: %w", domain.ErrInvalidInput, i+1, err)
		}

		out = append(out, &domain.Example{
			ID: fmt.Sprintf("%s-%d", td.ID, i), TaskID: td.TaskID, Kind: td.Kind,
			Payload: b, Source: domain.SourceUser, CreatedAt: now,
		})
	}

	return out, nil
}

func (u *TabularUsecase) prepareTraining(taskID, tableID string, kind domain.ModelKind) (*domain.TableDataset, []tableRow, error) {
	task, err := u.tasks.Get(taskID)
	if err != nil {
		return nil, nil, err
	}

	if task.Kind != kind {
		return nil, nil, fmt.Errorf("%w: task kind is %q, this endpoint trains %q", domain.ErrInvalidInput, task.Kind, kind)
	}

	td, err := u.GetTable(taskID, tableID)
	if err != nil {
		return nil, nil, err
	}

	if td.Kind != kind {
		return nil, nil, fmt.Errorf("%w: table %s is a %s table", domain.ErrInvalidInput, tableID, td.Kind)
	}

	if td.Target == "" {
		return nil, nil, fmt.Errorf("%w: pick a target column first", domain.ErrInvalidInput)
	}

	if td.RowCount < minTrainRows {
		return nil, nil, fmt.Errorf("%w: need at least %d rows to train (table has %d)",
			domain.ErrInvalidInput, minTrainRows, td.RowCount)
	}

	jobs, _ := u.jobs.ListByTask(taskID)
	for _, j := range jobs {
		if j.Status == domain.TrainingRunning {
			return nil, nil, domain.ErrAlreadyRunning
		}
	}

	rows, err := u.loadRows(td)
	if err != nil {
		return nil, nil, err
	}

	return td, rows, nil
}

func (u *TabularUsecase) launch(job *domain.TrainingJob, td *domain.TableDataset, rows []tableRow) (*domain.TrainingJob, error) {
	// The repository hands out its stored pointer to every reader, so the
	// training goroutine works on a private job and publishes immutable
	// copies (Create/Update below); readers never see a job being mutated.
	pub := *job

	err := u.jobs.Create(&pub)
	if err != nil {
		return nil, err
	}

	examples, err := u.rowsAsExamples(td, rows)
	if err != nil {
		u.complete(job)(nil, err)

		return nil, err
	}

	snap := u.snapshot(job)

	u.tuner.Start(job, examples, u.progress(job), u.complete(job))

	return snap, nil
}

// snapshot returns a copy of a job that is safe to serialise while the
// training goroutine keeps updating the original.
func (u *TabularUsecase) snapshot(job *domain.TrainingJob) *domain.TrainingJob {
	u.jobMu.Lock()
	defer u.jobMu.Unlock()

	cp := *job

	return &cp
}

func (u *TabularUsecase) nextVersion(taskID string) int {
	jobs, _ := u.jobs.ListByTask(taskID)
	v := 1

	for _, j := range jobs {
		if j.Version >= v {
			v = j.Version + 1
		}
	}

	return v
}

func (u *TabularUsecase) progress(job *domain.TrainingJob) func(int) {
	return func(p int) {
		u.jobMu.Lock()
		defer u.jobMu.Unlock()

		job.Progress = max(0, min(99, p))
		pub := *job
		_ = u.jobs.Update(&pub)
	}
}

func (u *TabularUsecase) complete(job *domain.TrainingJob) func(metrics *domain.TrainingMetrics, err error) {
	return func(metrics *domain.TrainingMetrics, err error) {
		u.jobMu.Lock()
		defer u.jobMu.Unlock()

		now := time.Now().UTC()
		job.CompletedAt = &now

		if err != nil {
			job.Status = domain.TrainingFailed
			job.Error = err.Error()
		} else {
			job.Status = domain.TrainingCompleted
			job.Progress = 100
			job.Metrics = metrics
		}

		pub := *job
		_ = u.jobs.Update(&pub)
	}
}

func ptrTime(t time.Time) *time.Time { return &t }
