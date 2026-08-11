package memory_test

import (
	"testing"

	"distillery/internal/domain"
	"distillery/internal/repository/memory"
)

func newExampleRepo() (*memory.ExampleRepo, *memory.Store) {
	s := memory.NewStore("")
	return memory.NewExampleRepo(s), s
}

func TestExampleRepo_Add(t *testing.T) {
	r, s := newExampleRepo()
	e := &domain.Example{ID: "ex_1", TaskID: "task_1", Input: "a", Output: "b"}

	if err := r.Add(e); err != nil {
		t.Fatalf("expected no error, got %v", err)
	}
	if len(s.Examples["task_1"]) != 1 {
		t.Errorf("expected 1 example, got %d", len(s.Examples["task_1"]))
	}
}

func TestExampleRepo_AddBatch(t *testing.T) {
	r, s := newExampleRepo()
	es := []*domain.Example{
		{ID: "ex_1", TaskID: "task_1"},
		{ID: "ex_2", TaskID: "task_1"},
	}

	if err := r.AddBatch(es); err != nil {
		t.Fatalf("expected no error, got %v", err)
	}
	if len(s.Examples["task_1"]) != 2 {
		t.Errorf("expected 2 examples, got %d", len(s.Examples["task_1"]))
	}
}

func TestExampleRepo_AddBatch_Empty(t *testing.T) {
	r, _ := newExampleRepo()

	if err := r.AddBatch(nil); err != nil {
		t.Errorf("expected no error for empty batch, got %v", err)
	}
}

func TestExampleRepo_ListByTask(t *testing.T) {
	r, _ := newExampleRepo()
	_ = r.Add(&domain.Example{ID: "ex_1", TaskID: "task_1"})
	_ = r.Add(&domain.Example{ID: "ex_2", TaskID: "task_1"})

	list, err := r.ListByTask("task_1")
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}
	if len(list) != 2 {
		t.Errorf("expected 2 examples, got %d", len(list))
	}
}

func TestExampleRepo_ListByTask_Empty(t *testing.T) {
	r, _ := newExampleRepo()

	list, err := r.ListByTask("task_1")
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}
	if len(list) != 0 {
		t.Errorf("expected 0 examples, got %d", len(list))
	}
}

func TestExampleRepo_ListByTask_ReturnsCopy(t *testing.T) {
	r, _ := newExampleRepo()
	_ = r.Add(&domain.Example{ID: "ex_1", TaskID: "task_1"})

	list1, _ := r.ListByTask("task_1")
	list1[0] = nil // mutate returned slice.

	list2, _ := r.ListByTask("task_1")
	if list2[0] == nil {
		t.Error("expected ListByTask to return a copy, not internal slice")
	}
}

func TestExampleRepo_Get_Found(t *testing.T) {
	r, _ := newExampleRepo()
	_ = r.Add(&domain.Example{ID: "ex_1", TaskID: "task_1", Input: "a"})

	got, err := r.Get("task_1", "ex_1")
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}
	if got.ID != "ex_1" {
		t.Errorf("expected ID 'ex_1', got %q", got.ID)
	}
}

func TestExampleRepo_Get_NotFound(t *testing.T) {
	r, _ := newExampleRepo()

	_, err := r.Get("task_1", "missing")
	if err != domain.ErrNotFound {
		t.Errorf("expected ErrNotFound, got %v", err)
	}
}

func TestExampleRepo_Update_Found(t *testing.T) {
	r, _ := newExampleRepo()
	_ = r.Add(&domain.Example{ID: "ex_1", TaskID: "task_1", Input: "old"})

	updated := &domain.Example{ID: "ex_1", TaskID: "task_1", Input: "new"}
	if err := r.Update(updated); err != nil {
		t.Fatalf("expected no error, got %v", err)
	}
	got, _ := r.Get("task_1", "ex_1")
	if got.Input != "new" {
		t.Errorf("expected input 'new', got %q", got.Input)
	}
}

func TestExampleRepo_Update_NotFound(t *testing.T) {
	r, _ := newExampleRepo()

	err := r.Update(&domain.Example{ID: "missing", TaskID: "task_1"})
	if err != domain.ErrNotFound {
		t.Errorf("expected ErrNotFound, got %v", err)
	}
}

func TestExampleRepo_Delete_Found(t *testing.T) {
	r, _ := newExampleRepo()
	_ = r.Add(&domain.Example{ID: "ex_1", TaskID: "task_1"})
	_ = r.Add(&domain.Example{ID: "ex_2", TaskID: "task_1"})

	if err := r.Delete("task_1", "ex_1"); err != nil {
		t.Fatalf("expected no error, got %v", err)
	}
	list, _ := r.ListByTask("task_1")
	if len(list) != 1 {
		t.Errorf("expected 1 example after delete, got %d", len(list))
	}
	if list[0].ID != "ex_2" {
		t.Errorf("expected remaining ID 'ex_2', got %q", list[0].ID)
	}
}

func TestExampleRepo_Delete_NotFound(t *testing.T) {
	r, _ := newExampleRepo()

	err := r.Delete("task_1", "missing")
	if err != domain.ErrNotFound {
		t.Errorf("expected ErrNotFound, got %v", err)
	}
}

func TestExampleRepo_DeleteByTask(t *testing.T) {
	r, s := newExampleRepo()
	_ = r.Add(&domain.Example{ID: "ex_1", TaskID: "task_1"})
	_ = r.Add(&domain.Example{ID: "ex_2", TaskID: "task_1"})

	if err := r.DeleteByTask("task_1"); err != nil {
		t.Fatalf("expected no error, got %v", err)
	}
	if _, ok := s.Examples["task_1"]; ok {
		t.Error("expected examples map entry to be deleted")
	}
}
