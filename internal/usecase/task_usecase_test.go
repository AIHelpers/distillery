package usecase_test

import (
	"testing"

	"distillery/internal/domain"
	"distillery/internal/usecase"
)

// stubIDGenerator is a deterministic ID generator for testing.
type stubIDGenerator struct {
	counter int
}

func (s *stubIDGenerator) NewID(prefix string) string {
	s.counter++
	return prefix + "_stub_001"
}

func newStubIDGen() *stubIDGenerator { return &stubIDGenerator{} }

// --- TaskUsecase tests ---.

func TestTaskUsecase_CreateTask_Valid(t *testing.T) {
	tasks := &mockTaskRepo{}
	examples := &mockExampleRepo{}
	uc := usecase.NewTaskUsecase(tasks, examples, newStubIDGen())

	task, err := uc.CreateTask("My Task", "  desc  ", domain.TaskClassification)
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}
	if task.Name != "My Task" {
		t.Errorf("expected name 'My Task', got %q", task.Name)
	}
	if task.Description != "desc" {
		t.Errorf("expected trimmed description 'desc', got %q", task.Description)
	}
	if task.Type != domain.TaskClassification {
		t.Errorf("expected type classification, got %v", task.Type)
	}
	if task.ID == "" {
		t.Error("expected non-empty ID")
	}
	if task.CreatedAt.IsZero() {
		t.Error("expected non-zero CreatedAt")
	}
	if task.UpdatedAt.IsZero() {
		t.Error("expected non-zero UpdatedAt")
	}
	if !tasks.createCalled {
		t.Error("expected Create to be called on repo")
	}
}

func TestTaskUsecase_CreateTask_EmptyName(t *testing.T) {
	uc := usecase.NewTaskUsecase(&mockTaskRepo{}, &mockExampleRepo{}, newStubIDGen())
	_, err := uc.CreateTask("  ", "desc", domain.TaskClassification)
	if err != domain.ErrInvalidInput {
		t.Errorf("expected ErrInvalidInput, got %v", err)
	}
}

func TestTaskUsecase_CreateTask_InvalidType(t *testing.T) {
	uc := usecase.NewTaskUsecase(&mockTaskRepo{}, &mockExampleRepo{}, newStubIDGen())
	_, err := uc.CreateTask("name", "desc", domain.TaskType("unknown"))
	if err != domain.ErrInvalidInput {
		t.Errorf("expected ErrInvalidInput, got %v", err)
	}
}

func TestTaskUsecase_CreateTask_AllValidTypes(t *testing.T) {
	validTypes := []domain.TaskType{
		domain.TaskClassification,
		domain.TaskExtraction,
		domain.TaskGeneration,
	}
	for _, tt := range validTypes {
		uc := usecase.NewTaskUsecase(&mockTaskRepo{}, &mockExampleRepo{}, newStubIDGen())
		task, err := uc.CreateTask("name", "desc", tt)
		if err != nil {
			t.Errorf("type %v: expected no error, got %v", tt, err)
		}
		if task.Type != tt {
			t.Errorf("type %v: mismatch", tt)
		}
	}
}

func TestTaskUsecase_GetTask(t *testing.T) {
	tasks := &mockTaskRepo{task: &domain.Task{ID: "task_1"}}
	uc := usecase.NewTaskUsecase(tasks, &mockExampleRepo{}, newStubIDGen())

	task, err := uc.GetTask("task_1")
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}
	if task.ID != "task_1" {
		t.Errorf("expected ID 'task_1', got %q", task.ID)
	}
	if tasks.getID != "task_1" {
		t.Errorf("expected Get called with 'task_1', got %q", tasks.getID)
	}
}

func TestTaskUsecase_ListTasks(t *testing.T) {
	tasks := &mockTaskRepo{list: []*domain.Task{{ID: "task_1"}, {ID: "task_2"}}}
	uc := usecase.NewTaskUsecase(tasks, &mockExampleRepo{}, newStubIDGen())

	list, err := uc.ListTasks()
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}
	if len(list) != 2 {
		t.Errorf("expected 2 tasks, got %d", len(list))
	}
}

func TestTaskUsecase_DeleteTask(t *testing.T) {
	tasks := &mockTaskRepo{}
	uc := usecase.NewTaskUsecase(tasks, &mockExampleRepo{}, newStubIDGen())

	err := uc.DeleteTask("task_1")
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}
	if tasks.deleteID != "task_1" {
		t.Errorf("expected Delete called with 'task_1', got %q", tasks.deleteID)
	}
}
