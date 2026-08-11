package memory_test

import (
	"testing"

	"distillery/internal/domain"
	"distillery/internal/repository/memory"
)

func newTaskRepo() (*memory.TaskRepo, *memory.Store) {
	s := memory.NewStore("")
	return memory.NewTaskRepo(s), s
}

func TestTaskRepo_Create(t *testing.T) {
	r, s := newTaskRepo()
	task := &domain.Task{ID: "task_1", Name: "My Task"}

	if err := r.Create(task); err != nil {
		t.Fatalf("expected no error, got %v", err)
	}
	if s.Tasks["task_1"] != task {
		t.Error("expected task to be stored")
	}
}

func TestTaskRepo_Get_Found(t *testing.T) {
	r, _ := newTaskRepo()
	task := &domain.Task{ID: "task_1", Name: "My Task"}
	_ = r.Create(task)

	got, err := r.Get("task_1")
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}
	if got.ID != "task_1" {
		t.Errorf("expected ID 'task_1', got %q", got.ID)
	}
}

func TestTaskRepo_Get_NotFound(t *testing.T) {
	r, _ := newTaskRepo()

	_, err := r.Get("missing")
	if err != domain.ErrNotFound {
		t.Errorf("expected ErrNotFound, got %v", err)
	}
}

func TestTaskRepo_List(t *testing.T) {
	r, _ := newTaskRepo()
	_ = r.Create(&domain.Task{ID: "task_1"})
	_ = r.Create(&domain.Task{ID: "task_2"})

	list, err := r.List()
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}
	if len(list) != 2 {
		t.Errorf("expected 2 tasks, got %d", len(list))
	}
}

func TestTaskRepo_List_Empty(t *testing.T) {
	r, _ := newTaskRepo()

	list, err := r.List()
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}
	if len(list) != 0 {
		t.Errorf("expected 0 tasks, got %d", len(list))
	}
}

func TestTaskRepo_Update_Found(t *testing.T) {
	r, s := newTaskRepo()
	_ = r.Create(&domain.Task{ID: "task_1", Name: "old"})

	updated := &domain.Task{ID: "task_1", Name: "new"}
	if err := r.Update(updated); err != nil {
		t.Fatalf("expected no error, got %v", err)
	}
	if s.Tasks["task_1"].Name != "new" {
		t.Errorf("expected name 'new', got %q", s.Tasks["task_1"].Name)
	}
}

func TestTaskRepo_Update_NotFound(t *testing.T) {
	r, _ := newTaskRepo()

	err := r.Update(&domain.Task{ID: "missing"})
	if err != domain.ErrNotFound {
		t.Errorf("expected ErrNotFound, got %v", err)
	}
}

func TestTaskRepo_Delete_RemovesTaskAndRelatedData(t *testing.T) {
	r, s := newTaskRepo()
	exampleRepo := memory.NewExampleRepo(s)

	_ = r.Create(&domain.Task{ID: "task_1"})
	_ = exampleRepo.Add(&domain.Example{ID: "ex_1", TaskID: "task_1"})

	if err := r.Delete("task_1"); err != nil {
		t.Fatalf("expected no error, got %v", err)
	}
	if _, ok := s.Tasks["task_1"]; ok {
		t.Error("expected task to be deleted")
	}
	if _, ok := s.Examples["task_1"]; ok {
		t.Error("expected examples for task to be deleted")
	}
}

func TestTaskRepo_Delete_NonExistent(t *testing.T) {
	r, _ := newTaskRepo()

	// Deleting non-existent task should not error (idempotent).
	if err := r.Delete("missing"); err != nil {
		t.Fatalf("expected no error, got %v", err)
	}
}
