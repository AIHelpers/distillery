package usecase

import (
	"encoding/csv"
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
	if _, err := u.tasks.Get(taskID); err != nil {
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
			CreatedAt: now,
		})
	}
	if len(batch) == 0 {
		return nil, domain.ErrInvalidInput
	}
	if err := u.examples.AddBatch(batch); err != nil {
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
	if err := u.examples.AddBatch(generated); err != nil {
		return nil, err
	}
	return u.Curate(taskID)
}

func (u *DatasetUsecase) ListExamples(taskID string) ([]*domain.Example, error) {
	return u.examples.ListByTask(taskID)
}

// ImportCSV bulk-loads example pairs from CSV content. If the first row has
// "input"/"output" column headers (case-insensitive) those columns are used;
// otherwise the first two columns of every row are treated as input/output.
func (u *DatasetUsecase) ImportCSV(taskID, content string) (*domain.DatasetStats, error) {
	if _, err := u.tasks.Get(taskID); err != nil {
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

// UpdateExample edits an existing example's input/output and re-curates the dataset.
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
	if err := u.examples.Update(e); err != nil {
		return nil, err
	}
	return u.Curate(taskID)
}

// DeleteExample removes a single example and re-curates the dataset.
func (u *DatasetUsecase) DeleteExample(taskID, exampleID string) (*domain.DatasetStats, error) {
	if err := u.examples.Delete(taskID, exampleID); err != nil {
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

	stats := &domain.DatasetStats{TaskID: taskID, LabelBalance: map[string]int{}}
	seenInputs := map[string]bool{}

	for _, e := range list {
		key := strings.ToLower(strings.TrimSpace(e.Input))
		wasDup := seenInputs[key]
		e.Duplicate = wasDup
		seenInputs[key] = true

		e.Flagged, e.FlagNote = qualityFlag(e)

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
	return stats, nil
}

func qualityFlag(e *domain.Example) (flagged bool, note string) {
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
