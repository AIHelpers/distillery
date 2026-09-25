package usecase

import (
	"encoding/csv"
	"encoding/json"
	"strings"
	"time"

	"distillery/internal/domain"
)

const (
	minInputLen  = 2
	minOutputLen = 1
)

type DatasetUsecase struct {
	tasks    domain.TaskRepository
	examples domain.ExampleRepository
	synthGen domain.SyntheticGenerator
	idGen    IDGenerator
}

func NewDatasetUsecase(
	tasks domain.TaskRepository,
	examples domain.ExampleRepository,
	synthGen domain.SyntheticGenerator,
	idGen IDGenerator,
) *DatasetUsecase {
	return &DatasetUsecase{
		tasks:    tasks,
		examples: examples,
		synthGen: synthGen,
		idGen:    idGen,
	}
}

type ExamplePair struct {
	Input  string
	Output string
}

// AddExamples adds user-provided input/output pairs, then re-runs curation.
func (u *DatasetUsecase) AddExamples(taskID string, pairs []ExamplePair) (*domain.DatasetStats, error) {
	_, err := u.tasks.Get(taskID)
	if err != nil {
		return nil, err
	}

	now := time.Now().UTC()

	var batch []*domain.Example

	for _, p := range pairs {
		in, out := strings.TrimSpace(p.Input), strings.TrimSpace(p.Output)
		if in == "" || out == "" {
			continue
		}

		batch = append(batch, &domain.Example{
			ID:        u.idGen.NewID("ex"),
			TaskID:    taskID,
			Input:     in,
			Output:    out,
			Source:    domain.SourceUser,
			CreatedAt: now, Flagged: false, FlagNote: "", Duplicate: false,
		})
	}

	if len(batch) == 0 {
		return nil, domain.ErrInvalidInput
	}

	err = u.examples.AddBatch(batch)
	if err != nil {
		return nil, err
	}

	return u.Curate(taskID)
}

// GenerateSynthetic bootstraps additional examples from the existing
// user-provided seed set via the (simulated) frontier-model generator.
func (u *DatasetUsecase) GenerateSynthetic(taskID string, count int) (*domain.DatasetStats, error) {
	task, err := u.tasks.Get(taskID)
	if err != nil {
		return nil, err
	}

	existing, err := u.examples.ListByTask(taskID)
	if err != nil {
		return nil, err
	}

	var seed []*domain.Example

	for _, e := range existing {
		if e.Source == domain.SourceUser {
			seed = append(seed, e)
		}
	}

	if len(seed) == 0 {
		return nil, domain.ErrInvalidInput
	}

	generated := u.synthGen.Generate(task, seed, count)
	now := time.Now().UTC()

	for _, g := range generated {
		g.ID = u.idGen.NewID("ex")
		g.CreatedAt = now
	}

	err = u.examples.AddBatch(generated)
	if err != nil {
		return nil, err
	}

	return u.Curate(taskID)
}

func (u *DatasetUsecase) ListExamples(taskID string) ([]*domain.Example, error) {
	return u.examples.ListByTask(taskID)
}

// ImportCSV bulk-loads example pairs from CSV content.
func (u *DatasetUsecase) ImportCSV(taskID, content string) (*domain.DatasetStats, error) {
	_, err := u.tasks.Get(taskID)
	if err != nil {
		return nil, err
	}

	reader := csv.NewReader(strings.NewReader(content))
	reader.FieldsPerRecord = -1
	reader.TrimLeadingSpace = true

	records, err := reader.ReadAll()
	if err != nil || len(records) == 0 {
		return nil, domain.ErrInvalidInput
	}

	inputIdx, outputIdx, start := 0, 1, 0

	if len(records[0]) >= 2 {
		li, lo := -1, -1

		for i, col := range records[0] {
			switch strings.ToLower(strings.TrimSpace(col)) {
			case "input":
				li = i
			case "output":
				lo = i
			}
		}

		if li >= 0 && lo >= 0 {
			inputIdx, outputIdx, start = li, lo, 1
		}
	}

	var pairs []ExamplePair

	for _, rec := range records[start:] {
		if len(rec) <= inputIdx || len(rec) <= outputIdx {
			continue
		}

		pairs = append(pairs, ExamplePair{Input: rec[inputIdx], Output: rec[outputIdx]})
	}

	if len(pairs) == 0 {
		return nil, domain.ErrInvalidInput
	}

	return u.AddExamples(taskID, pairs)
}

// UpdateExample edits an existing example's input/output and re-curates.
func (u *DatasetUsecase) UpdateExample(taskID, exampleID, input, output string) (*domain.DatasetStats, error) {
	input, output = strings.TrimSpace(input), strings.TrimSpace(output)
	if input == "" || output == "" {
		return nil, domain.ErrInvalidInput
	}

	e, err := u.examples.Get(taskID, exampleID)
	if err != nil {
		return nil, err
	}

	e.Input = input
	e.Output = output

	err = u.examples.Update(e)
	if err != nil {
		return nil, err
	}

	return u.Curate(taskID)
}

// DeleteExample removes a single example and re-curates the dataset.
func (u *DatasetUsecase) DeleteExample(taskID, exampleID string) (*domain.DatasetStats, error) {
	err := u.examples.Delete(taskID, exampleID)
	if err != nil {
		return nil, err
	}

	return u.Curate(taskID)
}

// Curate dedupes, flags low-quality examples, and recomputes label balance
// (for classification tasks), persisting the updated flags back to storage.
func (u *DatasetUsecase) Curate(taskID string) (*domain.DatasetStats, error) {
	task, err := u.tasks.Get(taskID)
	if err != nil {
		return nil, err
	}

	list, err := u.examples.ListByTask(taskID)
	if err != nil {
		return nil, err
	}

	stats := &domain.DatasetStats{
		TaskID:          taskID,
		LabelBalance:    map[string]int{},
		Total:           0,
		Duplicates:      0,
		Flagged:         0,
		Synthetic:       0,
		UserProvided:    0,
		Feedback:        0,
		UsableCount:     0,
		ReadyToTrain:    false,
		ReadinessReason: "",
	}
	seenInputs := map[string]bool{}

	for _, e := range list {
		key := curationKey(e)
		wasDup := seenInputs[key]
		e.Duplicate = wasDup
		seenInputs[key] = true

		holdout := e.FlagNote == "holdout:query_group"

		flagged, note := qualityFlag(e)

		if flagged {
			e.Flagged = true
			e.FlagNote = note
		} else if holdout {
			e.Flagged = false
			e.FlagNote = "holdout:query_group"
		} else {
			e.Flagged = false
			e.FlagNote = ""
		}

		_ = u.examples.Update(e)

		stats.Total++
		if e.Duplicate {
			stats.Duplicates++
		}

		if e.Flagged {
			stats.Flagged++
		}

		switch e.Source {
		case domain.SourceSynthetic:
			stats.Synthetic++
		case domain.SourceUser:
			stats.UserProvided++
		case domain.SourceFeedback:
			stats.Feedback++
		}

		if task.Type == domain.TaskClassification && !e.Duplicate && !e.Flagged {
			stats.LabelBalance[e.Output]++
		}
	}

	stats.UsableCount = stats.Total - stats.Duplicates - stats.Flagged
	stats.ReadyToTrain, stats.ReadinessReason = readiness(task, stats)

	// Track B: when the task declares a JSON schema, report the fraction of
	// usable examples whose output satisfies it.
	if schema := strings.TrimSpace(task.JSONSchema); schema != "" {
		stats.JSONValidRate = jsonValidRate(schema, list)
	}

	return stats, nil
}

// jsonValidRate returns the fraction of usable examples whose output parses
// as JSON and satisfies the task schema.
func jsonValidRate(schema string, examples []*domain.Example) float64 {
	checked, valid := 0, 0

	for _, e := range examples {
		if e.Duplicate || e.Flagged {
			continue
		}

		out := strings.TrimSpace(e.Output)
		if out == "" {
			continue
		}

		checked++

		if domain.ValidateJSONAgainstSchema(schema, out) == nil {
			valid++
		}
	}

	if checked == 0 {
		return 0
	}

	return float64(valid) / float64(checked)
}

// curationKey derives a dedup key from an example. Legacy text-only kinds use
// the Input/Output fields; retrieval kinds (embedding/reranker) use the query
// text plus the positive/document text so identical pairs are collapsed. NER
// kinds key on the example text (spans don't affect identity).
func curationKey(e *domain.Example) string {
	if text, ok := nerPayloadText(e); ok {
		return "ner:" + strings.ToLower(text)
	}

	query, positive, negative, doc := payloadFields(e)

	switch {
	case negative != "":
		return "triplet:" + strings.ToLower(query) + "\x00" + strings.ToLower(positive) + "\x00" + strings.ToLower(negative)
	case doc != "" && query != "":
		return "graded:" + strings.ToLower(query) + "\x00" + strings.ToLower(doc)
	case doc != "":
		return "docs:" + strings.ToLower(doc)
	case query != "" && positive != "":
		return "pair:" + strings.ToLower(query) + "\x00" + strings.ToLower(positive)
	default:
		return strings.ToLower(strings.TrimSpace(e.Input))
	}
}

// nerPayloadText extracts the "text" field from a token_classifier payload.
// It returns ok=false for non-NER payloads.
func nerPayloadText(e *domain.Example) (string, bool) {
	if len(e.Payload) == 0 {
		return "", false
	}

	var probe struct {
		Text string `json:"text"`
	}

	if json.Unmarshal(e.Payload, &probe) != nil || probe.Text == "" {
		return "", false
	}

	// Distinguish NER payloads from other shapes that happen to carry "text"
	// by requiring at least an entities array (possibly empty).
	var spans struct {
		Entities []json.RawMessage `json:"entities"`
	}

	if json.Unmarshal(e.Payload, &spans) != nil {
		return "", false
	}

	return probe.Text, true
}

func payloadFields(e *domain.Example) (query, positive, negative, doc string) {
	if len(e.Payload) == 0 {
		return e.Input, e.Output, "", ""
	}

	var p struct {
		Query    string `json:"query"`
		Positive string `json:"positive"`
		Negative string `json:"negative"`
		Document string `json:"document"`
	}

	_ = json.Unmarshal(e.Payload, &p)

	return p.Query, p.Positive, p.Negative, p.Document
}

// qualityFlag flags low-quality examples. Retrieval kinds (embedding/reranker)
// store their structured text (query/positive/document) in the typed Payload,
// so those rows are checked against each other for duplicate text. Legacy
// text-only kinds use the Input/Output fields and are checked for length and
// input==output noise.
func qualityFlag(e *domain.Example) (flagged bool, note string) {
	if len(e.Payload) == 0 {
		// Text-only kind: check Input/Output directly.
		if len(e.Input) < minInputLen {
			return true, "input too short"
		}

		if len(e.Output) < minOutputLen {
			return true, "output too short"
		}

		if strings.EqualFold(strings.TrimSpace(e.Input), strings.TrimSpace(e.Output)) {
			return true, "input and output are identical"
		}

		return false, ""
	}

	// NER kind (typed Payload with text/entities present).
	if text, ok := nerPayloadText(e); ok {
		if len(strings.TrimSpace(text)) < minInputLen {
			return true, "text too short"
		}

		return false, ""
	}

	// Retrieval kind (typed Payload present).
	query, positive, _, doc := payloadFields(e)

	if query != "" && positive != "" && strings.EqualFold(strings.TrimSpace(query), strings.TrimSpace(positive)) {
		return true, "query and positive document are identical"
	}

	if query != "" && doc != "" && strings.EqualFold(strings.TrimSpace(query), strings.TrimSpace(doc)) {
		return true, "query and document are identical"
	}

	if query == "" && positive == "" && doc == "" {
		return true, "missing query and document text"
	}

	return false, ""
}

func readiness(task *domain.Task, stats *domain.DatasetStats) (ready bool, reason string) {
	if stats.UsableCount < 3 {
		return false, "need at least 3 usable (non-duplicate, non-flagged) examples"
	}

	if task.Type == domain.TaskClassification && len(stats.LabelBalance) < 2 {
		return false, "classification tasks need at least 2 distinct labels represented"
	}

	if task.Type == domain.TaskClassification {
		minCount, maxCount := -1, -1
		for _, c := range stats.LabelBalance {
			if minCount == -1 || c < minCount {
				minCount = c
			}

			if maxCount == -1 || c > maxCount {
				maxCount = c
			}
		}

		if minCount > 0 && maxCount > minCount*5 {
			return true, "dataset is usable but label classes are imbalanced (>5x); " +
				"consider adding more examples of the rarer labels"
		}
	}

	return true, ""
}
