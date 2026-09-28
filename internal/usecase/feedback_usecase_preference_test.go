package usecase_test

import (
	"encoding/json"
	"testing"

	"distillery/internal/domain"
	"distillery/internal/usecase"
)

func TestFoldFeedbackAsPreferences(t *testing.T) {
	t.Parallel()

	unresolved := []*domain.Misprediction{
		{ID: "fb_1", TaskID: "task_1", Input: "What's 2+2?", ActualOutput: "5", ExpectedOutput: "4"},
		{ID: "fb_2", TaskID: "task_1", Input: "no contrast", ActualOutput: "", ExpectedOutput: "something"},
		{ID: "fb_3", TaskID: "task_1", Input: "identical", ActualOutput: "same", ExpectedOutput: "same"},
	}

	feedback := &mockFeedbackRepo{unresolved: unresolved}
	examples := &mockExampleRepo{}
	uc := usecase.NewFeedbackUsecase(&mockTaskRepo{task: &domain.Task{ID: "task_1"}}, feedback, examples, &mockIDGen{})

	n, err := uc.FoldFeedbackAsPreferences("task_1")
	if err != nil {
		t.Fatalf("expected success, got %v", err)
	}

	if n != 1 {
		t.Fatalf("expected 1 preference pair added (only fb_1 has a real contrast), got %d", n)
	}

	if len(examples.addBatch) != 1 {
		t.Fatalf("expected 1 example in AddBatch, got %d", len(examples.addBatch))
	}

	ex := examples.addBatch[0]
	if ex.Kind != domain.KindPreferenceLM {
		t.Fatalf("expected Kind == preference_lm, got %v", ex.Kind)
	}

	if ex.Source != domain.SourceFeedback {
		t.Fatalf("expected Source == feedback, got %v", ex.Source)
	}

	var payload struct {
		Prompt   string `json:"prompt"`
		Chosen   string `json:"chosen"`
		Rejected string `json:"rejected"`
	}

	err = json.Unmarshal(ex.Payload, &payload)
	if err != nil {
		t.Fatalf("malformed payload: %v", err)
	}

	if payload.Prompt != "What's 2+2?" || payload.Chosen != "4" || payload.Rejected != "5" {
		t.Fatalf("unexpected payload: %+v", payload)
	}

	if len(feedback.resolvedIDs) != 3 {
		t.Fatalf("expected all 3 mispredictions marked resolved, got %d", len(feedback.resolvedIDs))
	}
}

func TestFoldFeedbackAsPreferences_NoUnresolved(t *testing.T) {
	t.Parallel()

	feedback := &mockFeedbackRepo{}
	examples := &mockExampleRepo{}
	uc := usecase.NewFeedbackUsecase(&mockTaskRepo{task: &domain.Task{ID: "task_1"}}, feedback, examples, &mockIDGen{})

	n, err := uc.FoldFeedbackAsPreferences("task_1")
	if err != nil {
		t.Fatalf("expected success, got %v", err)
	}

	if n != 0 {
		t.Fatalf("expected 0, got %d", n)
	}
}
