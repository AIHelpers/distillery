package usecase

import (
	"encoding/json"
	"strings"
	"time"

	"distillery/internal/domain"
)

type FeedbackUsecase struct {
	tasks    domain.TaskRepository
	feedback domain.FeedbackRepository
	examples domain.ExampleRepository
	idGen    IDGenerator
}

func NewFeedbackUsecase(tasks domain.TaskRepository,
	feedback domain.FeedbackRepository,
	examples domain.ExampleRepository,
	idGen IDGenerator,
) *FeedbackUsecase {
	return &FeedbackUsecase{
		tasks:    tasks,
		feedback: feedback,
		examples: examples,
		idGen:    idGen,
	}
}

// FoldTarget selects what a piece of feedback becomes once folded: a
// corrected SFT example (the original behavior) or a DPO/ORPO preference
// pair (rejected = the model's actual output, chosen = the human
// correction).
type FoldTarget string

const (
	FoldTargetSFT         FoldTarget = "sft"
	FoldTargetPreferences FoldTarget = "preferences"
)

// SubmitMisprediction records a production error for the continuous
// improvement loop described in the product spec.
func (u *FeedbackUsecase) SubmitMisprediction(taskID, input, actual, expected string) (*domain.Misprediction, error) {
	_, err := u.tasks.Get(taskID)
	if err != nil {
		return nil, err
	}

	input, actual, expected = strings.TrimSpace(input), strings.TrimSpace(actual), strings.TrimSpace(expected)
	if input == "" || expected == "" {
		return nil, domain.ErrInvalidInput
	}

	m := &domain.Misprediction{
		ID:             u.idGen.NewID("fb"),
		TaskID:         taskID,
		Input:          input,
		ActualOutput:   actual,
		ExpectedOutput: expected,
		CreatedAt:      time.Now().UTC(), Resolved: false,
	}

	err = u.feedback.Add(m)
	if err != nil {
		return nil, err
	}

	return m, nil
}

func (u *FeedbackUsecase) ListFeedback(taskID string) ([]*domain.Misprediction, error) {
	return u.feedback.ListByTask(taskID)
}

// FoldIntoDataset converts all unresolved mispredictions for a task into
// corrected training examples and marks them resolved, ready for the next
// retrain. Returns the number of examples added.
func (u *FeedbackUsecase) FoldIntoDataset(taskID string) (int, error) {
	unresolved, err := u.feedback.ListUnresolved(taskID)
	if err != nil {
		return 0, err
	}

	if len(unresolved) == 0 {
		return 0, nil
	}

	now := time.Now().UTC()

	var (
		batch []*domain.Example
		ids   []string
	)

	for _, m := range unresolved {
		batch = append(batch, &domain.Example{
			ID:        u.idGen.NewID("ex"),
			TaskID:    taskID,
			Input:     m.Input,
			Output:    m.ExpectedOutput,
			Source:    domain.SourceFeedback,
			CreatedAt: now, Flagged: false, FlagNote: "", Duplicate: false,
		})
		ids = append(ids, m.ID)
	}

	err = u.examples.AddBatch(batch)
	if err != nil {
		return 0, err
	}

	err = u.feedback.MarkResolved(ids)
	if err != nil {
		return 0, err
	}

	return len(batch), nil
}

// FoldFeedbackAsPreferences converts unresolved mispredictions into DPO/ORPO
// preference pairs — rejected is the model's actual (wrong) output, chosen
// is the human-provided correction — and marks them resolved. This is the
// preference-tuning counterpart to FoldIntoDataset: the plan lets the user
// choose, per feedback batch, whether it becomes another SFT example or a
// preference pair. Mispredictions whose actual output is empty or equals
// the correction (nothing to contrast) are skipped rather than failing the
// whole batch.
func (u *FeedbackUsecase) FoldFeedbackAsPreferences(taskID string) (int, error) {
	unresolved, err := u.feedback.ListUnresolved(taskID)
	if err != nil {
		return 0, err
	}

	if len(unresolved) == 0 {
		return 0, nil
	}

	now := time.Now().UTC()

	var (
		batch []*domain.Example
		ids   []string
	)

	for _, m := range unresolved {
		actual, expected := strings.TrimSpace(m.ActualOutput), strings.TrimSpace(m.ExpectedOutput)
		ids = append(ids, m.ID) // this misprediction is handled either way.

		if actual == "" || strings.EqualFold(actual, expected) {
			continue // nothing to contrast — not usable as a preference pair.
		}

		payload, err := json.Marshal(preferencePayload{Prompt: m.Input, Chosen: expected, Rejected: actual})
		if err != nil {
			continue
		}

		if preferenceSchema.Validate(payload) != nil {
			continue
		}

		batch = append(batch, &domain.Example{
			ID:        u.idGen.NewID("ex"),
			TaskID:    taskID,
			Kind:      domain.KindPreferenceLM,
			Payload:   payload,
			Source:    domain.SourceFeedback,
			CreatedAt: now,
		})
	}

	if len(batch) > 0 {
		err = u.examples.AddBatch(batch)
		if err != nil {
			return 0, err
		}
	}

	err = u.feedback.MarkResolved(ids)
	if err != nil {
		return 0, err
	}

	return len(batch), nil
}
