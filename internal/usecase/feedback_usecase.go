package usecase

import (
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

// SubmitMisprediction records a production error for the continuous
// improvement loop described in the product spec.
func (u *FeedbackUsecase) SubmitMisprediction(taskID, input, actual, expected string) (*domain.Misprediction, error) {
	if _, err := u.tasks.Get(taskID); err != nil {
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
		CreatedAt:      time.Now().UTC(),
	}
	if err := u.feedback.Add(m); err != nil {
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
	var batch []*domain.Example
	var ids []string
	for _, m := range unresolved {
		batch = append(batch, &domain.Example{
			ID:        u.idGen.NewID("ex"),
			TaskID:    taskID,
			Input:     m.Input,
			Output:    m.ExpectedOutput,
			Source:    domain.SourceFeedback,
			CreatedAt: now,
		})
		ids = append(ids, m.ID)
	}
	if err := u.examples.AddBatch(batch); err != nil {
		return 0, err
	}
	if err := u.feedback.MarkResolved(ids); err != nil {
		return 0, err
	}
	return len(batch), nil
}
