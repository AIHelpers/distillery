package domain

import (
	"encoding/json"
	"fmt"
	"math"
	"strconv"
	"strings"
	"time"
)

// Model-kind constants KindTabular / KindTimeSeries live in modelkind.go.

// TableColumnType is the inferred/editor type of one CSV column.
type TableColumnType string

const (
	// ColNumeric is an integer/float feature or target.
	ColNumeric TableColumnType = "numeric"
	// ColCategorical is a small-cardinality string/bool column.
	ColCategorical TableColumnType = "categorical"
	// ColDatetime is a parseable timestamp column.
	ColDatetime TableColumnType = "datetime"
	// ColText is free text (ignored as a feature by default).
	ColText TableColumnType = "text"
	// ColID is a unique-per-row identifier (excluded from features; warned
	// against as a target).
	ColID TableColumnType = "id"
	// ColIgnored is user-marked exclusion.
	ColIgnored TableColumnType = "ignored"
)

// IsValidTableColumnType reports whether t is a known column type.
func IsValidTableColumnType(t TableColumnType) bool {
	switch t {
	case ColNumeric, ColCategorical, ColDatetime, ColText, ColID, ColIgnored:
		return true
	}

	return false
}

// TableColumn is one detected column of a table dataset.
type TableColumn struct {
	Name   string          `json:"name"`
	Type   TableColumnType `json:"type"`
	Unique int             `json:"unique,omitempty"`
	// IntegerLike is set for numeric columns whose values are all integers.
	IntegerLike bool     `json:"integer_like,omitempty"`
	Missing     int      `json:"missing,omitempty"`
	Sample      []string `json:"sample,omitempty"`
	Warnings    []string `json:"warnings,omitempty"`
}

// TableDataset is a structured-data dataset (tabular or time-series) with
// schema metadata only; the row payload lives in blob storage (C12) under
// BlobKey as JSON lines.
type TableDataset struct {
	ID       string        `json:"id"`
	TaskID   string        `json:"task_id"`
	Filename string        `json:"filename"`
	Kind     ModelKind     `json:"kind"`
	RowCount int           `json:"row_count"`
	Columns  []TableColumn `json:"columns"`
	Target   string        `json:"target"`
	// Excluded lists the columns the user explicitly removed from the
	// feature set (ID/ignored/text typed columns are excluded implicitly,
	// see FeatureColumns).
	Excluded  []string `json:"excluded,omitempty"`
	Frequency string   `json:"frequency,omitempty"`
	// FrequencyConfirmed is set once the user confirmed (or supplied) the
	// series frequency; forecast training requires it (plan §2).
	FrequencyConfirmed bool              `json:"frequency_confirmed,omitempty"`
	Horizon            int               `json:"horizon,omitempty"`
	BlobKey            string            `json:"blob_key,omitempty"`
	CreatedAt          time.Time         `json:"created_at"`
	Extra              TableDatasetExtra `json:"extra,omitempty"`
}

// TableDatasetExtra carries per-kind settings: split strategy/group for
// tabular, timestamp/item-id columns for forecasting.
type TableDatasetExtra struct {
	HasDatetime     bool   `json:"has_datetime,omitempty"`
	TimestampColumn string `json:"timestamp_column,omitempty"`
	ItemIDColumn    string `json:"item_id_column,omitempty"`
	Split           string `json:"split,omitempty"`
	GroupColumn     string `json:"group_column,omitempty"`
	// Task is the user's classification/regression override ("" = infer).
	Task string `json:"task,omitempty"`
}

// ColumnType returns the type of a named column ("" when absent).
func (t *TableDataset) ColumnType(name string) TableColumnType {
	for _, c := range t.Columns {
		if c.Name == name {
			return c.Type
		}
	}

	return ""
}

// HasColumn reports whether the table has a column called name.
func (t *TableDataset) HasColumn(name string) bool {
	for _, c := range t.Columns {
		if c.Name == name {
			return true
		}
	}

	return false
}

// IsExcluded reports whether a column is removed from the feature set.
func (t *TableDataset) IsExcluded(name string) bool {
	for _, e := range t.Excluded {
		if e == name {
			return true
		}
	}

	switch t.ColumnType(name) {
	case ColID, ColIgnored, ColText:
		return true
	case ColNumeric, ColCategorical, ColDatetime:
	}

	return false
}

// FeatureColumns lists the columns used as model inputs for a tabular job,
// in table order: everything except the target, explicit/implicit exclusions,
// and the group column.
func (t *TableDataset) FeatureColumns() []TableColumn {
	out := make([]TableColumn, 0, len(t.Columns))

	for _, c := range t.Columns {
		if c.Name == t.Target || t.IsExcluded(c.Name) {
			continue
		}

		if t.Extra.GroupColumn != "" && c.Name == t.Extra.GroupColumn {
			continue
		}

		out = append(out, c)
	}

	return out
}

// SplitStrategy selects how rows are divided for validation.
type SplitStrategy string

const (
	// SplitRandom is a uniform random holdout.
	SplitRandom SplitStrategy = "random"
	// SplitStratified preserves class ratios (classification only).
	SplitStratified SplitStrategy = "stratified"
	// SplitTime is a chronological holdout — mandatory for temporal data to
	// prevent leakage (the default when a datetime column exists).
	SplitTime SplitStrategy = "time"
	// SplitGroup keeps all rows of a group (e.g. one user) on one side.
	SplitGroup SplitStrategy = "group"
)

// IsValidSplitStrategy reports whether s is a known split strategy.
func IsValidSplitStrategy(s SplitStrategy) bool {
	switch s {
	case SplitRandom, SplitStratified, SplitTime, SplitGroup:
		return true
	}

	return false
}

// Tabular task names.
const (
	TabularClassification = "classification"
	TabularRegression     = "regression"
)

// TabularConfig carries the tabular training hyperparameters mirrored by
// trainer/tasks/tabular.py. Target, Columns and the like are filled by the
// usecase from the table's mapping — clients cannot override them.
type TabularConfig struct {
	// Model family: "auto" | "lightgbm" | "xgboost" | "catboost".
	Model string `json:"model,omitempty"`
	// Task is "classification" or "regression" (auto from the target when
	// empty).
	Task string `json:"task,omitempty"`
	// Metric is the CV selection metric (auto by task when empty).
	Metric string `json:"metric,omitempty"`
	// TimeBudgetSec bounds total training/HPO seconds (0 = default).
	TimeBudgetSec int `json:"time_budget_sec,omitempty"`
	// CVFolds is the number of cross-validation folds (default 5).
	CVFolds int `json:"cv_folds,omitempty"`
	// SplitStrategy: random | stratified | time | group.
	SplitStrategy SplitStrategy `json:"split_strategy,omitempty"`
	// GroupColumn names the column used by the group split.
	GroupColumn string `json:"group_column,omitempty"`
	// Target is the label column (set server-side from the mapping).
	Target string `json:"target,omitempty"`
	// ExcludeColumns are removed from the feature set (server-side).
	ExcludeColumns []string `json:"exclude_columns,omitempty"`
	// Columns records the mapper's final column types (server-side) so the
	// trainer does not re-guess them.
	Columns map[string]TableColumnType `json:"columns,omitempty"`
}

// ForecastConfig carries the forecasting hyperparameters mirrored by
// trainer/tasks/forecast.py.
type ForecastConfig struct {
	// Horizon is how many future steps to forecast.
	Horizon int `json:"horizon,omitempty"`
	// Frequency: H/D/W/M/Q/Y.
	Frequency string `json:"frequency,omitempty"`
	// Metric is the backtest selection metric (mase primary).
	Metric string `json:"metric,omitempty"`
	// BacktestWindows is the number of rolling-origin windows (default 3).
	BacktestWindows int `json:"backtest_windows,omitempty"`
	// Model family: "auto" | "seasonal_naive" | "stats" | "gbm" | "chronos".
	Model string `json:"model,omitempty"`
	// SeasonLength overrides the frequency-derived season (0 = default).
	SeasonLength int `json:"season_length,omitempty"`
	// TimeBudgetSec bounds the model search (0 = default).
	TimeBudgetSec int `json:"time_budget_sec,omitempty"`
	// Timestamp/Target/ItemID are set server-side from the table mapping.
	Timestamp string `json:"timestamp,omitempty"`
	Target    string `json:"target,omitempty"`
	ItemID    string `json:"item_id,omitempty"`
}

// TableFeatureType is the per-feature type recorded in the deployment's
// feature schema.
type TableFeatureType string

const (
	FeatureNumeric     TableFeatureType = "numeric"
	FeatureCategorical TableFeatureType = "categorical"
	FeatureDatetime    TableFeatureType = "datetime"
)

// TableFeature is one validated input field of a deployed tabular model.
type TableFeature struct {
	Name       string           `json:"name"`
	Type       TableFeatureType `json:"type"`
	Categories []string         `json:"categories,omitempty"`
	// OpenVocabulary marks a categorical feature whose vocabulary was capped
	// at training time: unseen values are accepted (the model treats them as
	// "other") instead of rejected.
	OpenVocabulary bool `json:"open_vocabulary,omitempty"`
	// Optional features were missing in some training rows, so the model
	// imputes them and callers may omit them.
	Optional bool     `json:"optional,omitempty"`
	Min      *float64 `json:"min,omitempty"`
	Max      *float64 `json:"max,omitempty"`
}

// FeatureSchema is the recorded input contract of a deployed model: the
// ordered feature list, allowed categories, and the output classes for
// classification. Stored on the deployment so /predict can validate inputs
// and reject unknown categories / missing fields with clear errors.
type FeatureSchema struct {
	Features []TableFeature `json:"features"`
	Target   string         `json:"target"`
	Task     string         `json:"task,omitempty"`
	Classes  []string       `json:"classes,omitempty"`
}

// ValidatedRow is a schema-checked input row: canonical categorical values,
// parsed numerics and timestamps, plus the canonical JSON-ready map handed to
// the model server.
type ValidatedRow struct {
	Numerics map[string]float64
	Cats     map[string]string
	Times    map[string]time.Time
	// Canonical holds every provided feature in canonical form (float64,
	// canonical category string, RFC3339 string); missing optional features
	// are absent.
	Canonical map[string]interface{}
}

// ValidateRow checks one input row against the schema: required features
// must be present (optional ones may be omitted or null); categorical values
// must be allowed (matched case-insensitively and mapped to the canonical
// spelling); numerics must be finite numbers; datetimes must parse. Unknown
// extra keys are ignored (the API is additive-friendly).
func (fs *FeatureSchema) ValidateRow(input map[string]interface{}) (*ValidatedRow, error) {
	if fs == nil || len(fs.Features) == 0 {
		return nil, fmt.Errorf("%w: deployment has no feature schema", ErrInvalidInput)
	}

	out := &ValidatedRow{
		Numerics:  make(map[string]float64, len(fs.Features)),
		Cats:      make(map[string]string, len(fs.Features)),
		Times:     map[string]time.Time{},
		Canonical: make(map[string]interface{}, len(fs.Features)),
	}

	var missing []string

	for i := range fs.Features {
		f := &fs.Features[i]

		raw, ok := input[f.Name]
		if !ok || raw == nil || strings.TrimSpace(fmt.Sprintf("%v", raw)) == "" {
			if !f.Optional {
				missing = append(missing, f.Name)
			}

			continue
		}

		err := out.add(f, raw)
		if err != nil {
			return nil, err
		}
	}

	if len(missing) > 0 {
		return nil, fmt.Errorf("%w: missing required fields: %s", ErrInvalidInput, strings.Join(missing, ", "))
	}

	return out, nil
}

func (v *ValidatedRow) add(f *TableFeature, raw interface{}) error {
	switch f.Type {
	case FeatureCategorical:
		str := strings.TrimSpace(fmt.Sprintf("%v", raw))
		canon := str

		if len(f.Categories) > 0 && !f.OpenVocabulary {
			found := false

			for _, c := range f.Categories {
				if strings.EqualFold(str, c) {
					canon, found = c, true

					break
				}
			}

			if !found {
				return fmt.Errorf(
					"%w: unknown category %q for feature %q (allowed: %s)",
					ErrInvalidInput, str, f.Name, strings.Join(f.Categories, ", "))
			}
		} else if len(f.Categories) > 0 {
			for _, c := range f.Categories {
				if strings.EqualFold(str, c) {
					canon = c

					break
				}
			}
		}

		v.Cats[f.Name] = canon
		v.Canonical[f.Name] = canon

	case FeatureDatetime:
		t, ok := ParseTableTime(fmt.Sprintf("%v", raw))
		if !ok {
			return fmt.Errorf("%w: feature %q must be a date/time (got %q)", ErrInvalidInput, f.Name, TableValueString(raw))
		}

		v.Times[f.Name] = t
		v.Canonical[f.Name] = t.Format(time.RFC3339)

	case FeatureNumeric:
		fv, ok := TableToFloat(raw)
		if !ok {
			return fmt.Errorf("%w: feature %q must be numeric (got %q)", ErrInvalidInput, f.Name, TableValueString(raw))
		}

		v.Numerics[f.Name] = fv
		v.Canonical[f.Name] = fv
	}

	return nil
}

// TableValueString renders any value as a short string for error messages.
func TableValueString(v interface{}) string {
	if s, ok := v.(string); ok {
		return s
	}

	return fmt.Sprintf("%v", v)
}

// TableToFloat coerces JSON numbers / numeric strings to a finite float64.
// Strings must be fully numeric ("12abc", "NaN" and "Inf" are rejected).
func TableToFloat(v interface{}) (float64, bool) {
	var f float64

	switch x := v.(type) {
	case float64:
		f = x
	case float32:
		f = float64(x)
	case int:
		f = float64(x)
	case int64:
		f = float64(x)
	case json.Number:
		p, err := x.Float64()
		if err != nil {
			return 0, false
		}

		f = p
	case string:
		p, err := strconv.ParseFloat(strings.TrimSpace(x), 64)
		if err != nil {
			return 0, false
		}

		f = p
	default:
		return 0, false
	}

	if math.IsNaN(f) || math.IsInf(f, 0) {
		return 0, false
	}

	return f, true
}

// tableTimeLayouts are the accepted timestamp layouts, tried in order.
// Ambiguous slash dates are month-first; day-first dates use dots or dashes
// with a leading day. Bare years are deliberately NOT timestamps (a column of
// 4-digit numbers is far more likely a quantity).
var tableTimeLayouts = []string{
	time.RFC3339, "2006-01-02T15:04:05", "2006-01-02 15:04:05", "2006-01-02 15:04",
	"2006-01-02", "2006/01/02", "02.01.2006 15:04:05", "02.01.2006 15:04", "02.01.2006",
	"01/02/2006 15:04:05", "01/02/2006 15:04", "01/02/2006", "2006-01",
}

// ParseTableTime parses s with the shared timestamp layouts.
func ParseTableTime(s string) (time.Time, bool) {
	s = strings.TrimSpace(s)
	if s == "" {
		return time.Time{}, false
	}

	for _, layout := range tableTimeLayouts {
		t, err := time.Parse(layout, s)
		if err == nil {
			return t, true
		}
	}

	return time.Time{}, false
}

// TablePrediction is the structured result of a tabular predict call.
type TablePrediction struct {
	// Prediction is the class label (classification) or numeric value.
	Prediction interface{} `json:"prediction"`
	// Probability is the probability of the predicted class (classification).
	Probability *float64 `json:"probability,omitempty"`
	// Probabilities maps every class to its score (classification).
	Probabilities map[string]float64 `json:"probabilities,omitempty"`
	// TopFactors are the highest-impact features for this prediction. With
	// the trained GBM these are SHAP contributions (signed, in the model's
	// output space: log-odds for classifiers); the simulation backend uses a
	// sensitivity approximation.
	TopFactors []FeatureImpact `json:"top_factors,omitempty"`
	// Method names how TopFactors were computed ("shap", "sensitivity").
	Method string `json:"explanation_method,omitempty"`
}

// FeatureImpact is one feature's contribution to a single prediction.
type FeatureImpact struct {
	Feature string  `json:"feature"`
	Impact  float64 `json:"impact"`
}

// ForecastPoint is one step of a forecast with an interval.
type ForecastPoint struct {
	Timestamp string  `json:"timestamp"`
	Value     float64 `json:"value"`
	Lower     float64 `json:"lower"`
	Upper     float64 `json:"upper"`
}

// ForecastRequest is the input of a forecast call.
type ForecastRequest struct {
	// History optionally replaces/extends the stored training history:
	// rows of {timestamp, target, item_id?} (column names as in training).
	History []map[string]interface{}
	Horizon int
	// ItemID selects one series of a multi-series model.
	ItemID string
}

// ForecastResult is the structured forecast response.
type ForecastResult struct {
	// Forecast points in chronological order.
	Forecast []ForecastPoint `json:"forecast"`
	// ItemID is the series this forecast belongs to ("" for single series).
	ItemID string `json:"item_id,omitempty"`
	// Model names the winning family from the backtest comparison.
	Model string `json:"model,omitempty"`
	// Metric reports the backtest score of the chosen model.
	Metric      string  `json:"metric,omitempty"`
	MetricValue float64 `json:"metric_value,omitempty"`
	// Baselines reports the comparison against seasonal-naive baselines so
	// the UI can show the gain.
	Baselines map[string]float64 `json:"baselines,omitempty"`
}

// TableRowSource lazily provides the training rows of a job (loaded from the
// table's blob) so inference engines that need them (the simulation kNN) load
// once and cache, while trained-model engines never touch them.
type TableRowSource interface {
	JobRows(job *TrainingJob) ([]map[string]interface{}, error)
}

// TabularInferenceEngine serves predictions from a deployed tabular model,
// validating inputs against the recorded feature schema.
type TabularInferenceEngine interface {
	PredictTable(
		job *TrainingJob,
		rows TableRowSource,
		schema *FeatureSchema,
		row *ValidatedRow,
	) (TablePrediction, error)
}

// ForecastInferenceEngine serves forecasts from a deployed time-series model.
type ForecastInferenceEngine interface {
	Forecast(job *TrainingJob, rows TableRowSource, req ForecastRequest) (ForecastResult, error)
}

// TableDatasetRepository is the port for persisting table datasets.
type TableDatasetRepository interface {
	Create(t *TableDataset) error
	Get(id string) (*TableDataset, error)
	ListByTask(taskID string) ([]*TableDataset, error)
	Update(t *TableDataset) error
	Delete(id string) error
}

// ParseTablePayload decodes a table row payload (a generic JSON object of
// column -> value).
func ParseTablePayload(payload []byte) (map[string]interface{}, error) {
	var m map[string]interface{}

	err := json.Unmarshal(payload, &m)
	if err != nil {
		return nil, err
	}

	return m, nil
}
