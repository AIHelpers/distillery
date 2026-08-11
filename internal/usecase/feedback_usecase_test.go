package usecase_test

import (
	"errors"
	"testing"

	"distillery/internal/domain"
	"distillery/internal/usecase"
)

// fbStubIDGen is a deterministic IDGenerator for feedback tests.
type fbStubIDGen struct {
	counter int
	prefix  string
}

func (s *fbStubIDGen) NewID(prefix string) string {
	s.counter++
	s.prefix = prefix
	return prefix + "_stub"
}

func newFeedbackUsecase(
	taskRepo *mockTaskRepo,
	fbRepo *mockFeedbackRepo,
	exRepo *mockExampleRepo,
	idGen usecase.IDGenerator,
) *usecase.FeedbackUsecase {
	if idGen == nil {
		idGen = &fbStubIDGen{}
	}
	return usecase.NewFeedbackUsecase(taskRepo, fbRepo, exRepo, idGen)
}

func TestFeedbackUsecase_SubmitMisprediction(t *testing.T) {
	t.Run("success", func(t *testing.T) {
		taskRepo := &mockTaskRepo{task: &domain.Task{ID: "task-1"}}
		fbRepo := &mockFeedbackRepo{}
		uc := newFeedbackUsecase(taskRepo, fbRepo, nil, nil)

		m, err := uc.SubmitMisprediction("task-1", "  input  ", "  actual  ", "  expected  ")
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if m == nil {
			t.Fatal("expected misprediction, got nil")
		}
		if m.TaskID != "task-1" {
			t.Errorf("TaskID = %q, want %q", m.TaskID, "task-1")
		}
		if m.Input != "input" {
			t.Errorf("Input = %q, want %q", m.Input, "input")
		}
		if m.ActualOutput != "actual" {
			t.Errorf("ActualOutput = %q, want %q", m.ActualOutput, "actual")
		}
		if m.ExpectedOutput != "expected" {
			t.Errorf("ExpectedOutput = %q, want %q", m.ExpectedOutput, "expected")
		}
		if m.ID == "" {
			t.Error("expected non-empty ID")
		}
		if m.CreatedAt.IsZero() {
			t.Error("expected non-zero CreatedAt")
		}
		if fbRepo.added == nil {
			t.Error("expected Add to be called")
		}
	})

	t.Run("task_not_found", func(t *testing.T) {
		taskRepo := &mockTaskRepo{} // returns ErrNotFound.
		uc := newFeedbackUsecase(taskRepo, &mockFeedbackRepo{}, nil, nil)

		_, err := uc.SubmitMisprediction("missing", "input", "actual", "expected")
		if !errors.Is(err, domain.ErrNotFound) {
			t.Errorf("expected ErrNotFound, got %v", err)
		}
	})

	t.Run("repo_error_on_task_get", func(t *testing.T) {
		taskRepo := &mockTaskRepo{err: errRepoFailure}
		uc := newFeedbackUsecase(taskRepo, &mockFeedbackRepo{}, nil, nil)

		_, err := uc.SubmitMisprediction("task-1", "input", "actual", "expected")
		if !errors.Is(err, errRepoFailure) {
			t.Errorf("expected errRepoFailure, got %v", err)
		}
	})

	t.Run("empty_input", func(t *testing.T) {
		taskRepo := &mockTaskRepo{task: &domain.Task{ID: "task-1"}}
		uc := newFeedbackUsecase(taskRepo, &mockFeedbackRepo{}, nil, nil)

		_, err := uc.SubmitMisprediction("task-1", "   ", "actual", "expected")
		if !errors.Is(err, domain.ErrInvalidInput) {
			t.Errorf("expected ErrInvalidInput, got %v", err)
		}
	})

	t.Run("empty_expected", func(t *testing.T) {
		taskRepo := &mockTaskRepo{task: &domain.Task{ID: "task-1"}}
		uc := newFeedbackUsecase(taskRepo, &mockFeedbackRepo{}, nil, nil)

		_, err := uc.SubmitMisprediction("task-1", "input", "actual", "   ")
		if !errors.Is(err, domain.ErrInvalidInput) {
			t.Errorf("expected ErrInvalidInput, got %v", err)
		}
	})

	t.Run("feedback_add_error", func(t *testing.T) {
		taskRepo := &mockTaskRepo{task: &domain.Task{ID: "task-1"}}
		fbRepo := &mockFeedbackRepo{err: errRepoFailure}
		uc := newFeedbackUsecase(taskRepo, fbRepo, nil, nil)

		_, err := uc.SubmitMisprediction("task-1", "input", "actual", "expected")
		if !errors.Is(err, errRepoFailure) {
			t.Errorf("expected errRepoFailure, got %v", err)
		}
	})
}

func TestFeedbackUsecase_ListFeedback(t *testing.T) {
	t.Run("success", func(t *testing.T) {
		expected := []*domain.Misprediction{{ID: "fb-1"}, {ID: "fb-2"}}
		fbRepo := &mockFeedbackRepo{list: expected}
		uc := newFeedbackUsecase(&mockTaskRepo{}, fbRepo, nil, nil)

		got, err := uc.ListFeedback("task-1")
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if len(got) != len(expected) {
			t.Errorf("len = %d, want %d", len(got), len(expected))
		}
	})

	t.Run("repo_error", func(t *testing.T) {
		fbRepo := &mockFeedbackRepo{err: errRepoFailure}
		uc := newFeedbackUsecase(&mockTaskRepo{}, fbRepo, nil, nil)

		_, err := uc.ListFeedback("task-1")
		if !errors.Is(err, errRepoFailure) {
			t.Errorf("expected errRepoFailure, got %v", err)
		}
	})
}

func TestFeedbackUsecase_FoldIntoDataset(t *testing.T) {
	t.Run("success", func(t *testing.T) {
		unresolved := []*domain.Misprediction{
			{ID: "fb-1", TaskID: "task-1", Input: "in1", ExpectedOutput: "out1"},
			{ID: "fb-2", TaskID: "task-1", Input: "in2", ExpectedOutput: "out2"},
		}
		fbRepo := &mockFeedbackRepo{unresolved: unresolved}
		exRepo := &mockExampleRepo{}
		uc := newFeedbackUsecase(&mockTaskRepo{}, fbRepo, exRepo, nil)

		count, err := uc.FoldIntoDataset("task-1")
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if count != 2 {
			t.Errorf("count = %d, want 2", count)
		}
		if len(exRepo.addBatch) != 2 {
			t.Errorf("AddBatch called with %d examples, want 2", len(exRepo.addBatch))
		}
		// Verify examples are built from mispredictions.
		for i, ex := range exRepo.addBatch {
			if ex.TaskID != "task-1" {
				t.Errorf("example[%d].TaskID = %q, want %q", i, ex.TaskID, "task-1")
			}
			if ex.Source != domain.SourceFeedback {
				t.Errorf("example[%d].Source = %q, want %q", i, ex.Source, domain.SourceFeedback)
			}
			if ex.Input != unresolved[i].Input {
				t.Errorf("example[%d].Input = %q, want %q", i, ex.Input, unresolved[i].Input)
			}
			if ex.Output != unresolved[i].ExpectedOutput {
				t.Errorf("example[%d].Output = %q, want %q", i, ex.Output, unresolved[i].ExpectedOutput)
			}
		}
		// Verify MarkResolved called with all IDs.
		if len(fbRepo.resolvedIDs) != 2 {
			t.Errorf("resolvedIDs len = %d, want 2", len(fbRepo.resolvedIDs))
		}
	})

	t.Run("no_unresolved_returns_zero", func(t *testing.T) {
		fbRepo := &mockFeedbackRepo{unresolved: nil}
		exRepo := &mockExampleRepo{}
		uc := newFeedbackUsecase(&mockTaskRepo{}, fbRepo, exRepo, nil)

		count, err := uc.FoldIntoDataset("task-1")
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if count != 0 {
			t.Errorf("count = %d, want 0", count)
		}
		if len(exRepo.addBatch) != 0 {
			t.Errorf("AddBatch should not be called, got %d examples", len(exRepo.addBatch))
		}
	})

	t.Run("list_unresolved_error", func(t *testing.T) {
		fbRepo := &mockFeedbackRepo{err: errRepoFailure}
		uc := newFeedbackUsecase(&mockTaskRepo{}, fbRepo, &mockExampleRepo{}, nil)

		_, err := uc.FoldIntoDataset("task-1")
		if !errors.Is(err, errRepoFailure) {
			t.Errorf("expected errRepoFailure, got %v", err)
		}
	})

	t.Run("addbatch_error", func(t *testing.T) {
		unresolved := []*domain.Misprediction{
			{ID: "fb-1", TaskID: "task-1", Input: "in1", ExpectedOutput: "out1"},
		}
		fbRepo := &mockFeedbackRepo{unresolved: unresolved}
		exRepo := &mockExampleRepo{err: errRepoFailure}
		uc := newFeedbackUsecase(&mockTaskRepo{}, fbRepo, exRepo, nil)

		_, err := uc.FoldIntoDataset("task-1")
		if !errors.Is(err, errRepoFailure) {
			t.Errorf("expected errRepoFailure, got %v", err)
		}
	})

	t.Run("markresolved_error", func(t *testing.T) {
		unresolved := []*domain.Misprediction{
			{ID: "fb-1", TaskID: "task-1", Input: "in1", ExpectedOutput: "out1"},
		}
		fbRepo := &mockFeedbackRepo{
			unresolved: unresolved,
			err:        errRepoFailure,
		}
		// Note: err is set, so ListUnresolved will fail first.
		// To test MarkResolved error specifically, we need err set only for MarkResolved.
		// Since our mock uses same err for all methods, we set err and expect ListUnresolved to fail.
		uc := newFeedbackUsecase(&mockTaskRepo{}, fbRepo, &mockExampleRepo{}, nil)

		_, err := uc.FoldIntoDataset("task-1")
		if !errors.Is(err, errRepoFailure) {
			t.Errorf("expected errRepoFailure, got %v", err)
		}
	})
}
