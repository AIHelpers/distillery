package memory_test

import (
	"testing"

	"distillery/internal/domain"
	"distillery/internal/repository/memory"
)

func newFeedbackRepo() (*memory.FeedbackRepo, *memory.Store) {
	s := memory.NewStore("")
	return memory.NewFeedbackRepo(s), s
}

func TestFeedbackRepo_Add(t *testing.T) {
	r, s := newFeedbackRepo()
	f := &domain.Misprediction{ID: "fb_1", TaskID: "task_1", Input: "x", ExpectedOutput: "y"}

	if err := r.Add(f); err != nil {
		t.Fatalf("expected no error, got %v", err)
	}
	if len(s.Feedback["task_1"]) != 1 {
		t.Errorf("expected 1 feedback entry, got %d", len(s.Feedback["task_1"]))
	}
}

func TestFeedbackRepo_Add_Multiple(t *testing.T) {
	r, s := newFeedbackRepo()
	_ = r.Add(&domain.Misprediction{ID: "fb_1", TaskID: "task_1"})
	_ = r.Add(&domain.Misprediction{ID: "fb_2", TaskID: "task_1"})

	if len(s.Feedback["task_1"]) != 2 {
		t.Errorf("expected 2 feedback entries, got %d", len(s.Feedback["task_1"]))
	}
}

func TestFeedbackRepo_Add_AcrossTasks(t *testing.T) {
	r, s := newFeedbackRepo()
	_ = r.Add(&domain.Misprediction{ID: "fb_1", TaskID: "task_1"})
	_ = r.Add(&domain.Misprediction{ID: "fb_2", TaskID: "task_2"})

	if len(s.Feedback["task_1"]) != 1 {
		t.Errorf("expected 1 feedback for task_1, got %d", len(s.Feedback["task_1"]))
	}
	if len(s.Feedback["task_2"]) != 1 {
		t.Errorf("expected 1 feedback for task_2, got %d", len(s.Feedback["task_2"]))
	}
}

func TestFeedbackRepo_ListByTask(t *testing.T) {
	r, _ := newFeedbackRepo()
	_ = r.Add(&domain.Misprediction{ID: "fb_1", TaskID: "task_1"})
	_ = r.Add(&domain.Misprediction{ID: "fb_2", TaskID: "task_1"})

	list, err := r.ListByTask("task_1")
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}
	if len(list) != 2 {
		t.Errorf("expected 2 entries, got %d", len(list))
	}
}

func TestFeedbackRepo_ListByTask_Empty(t *testing.T) {
	r, _ := newFeedbackRepo()

	list, err := r.ListByTask("task_1")
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}
	if len(list) != 0 {
		t.Errorf("expected 0 entries, got %d", len(list))
	}
}

func TestFeedbackRepo_ListByTask_ReturnsCopy(t *testing.T) {
	r, _ := newFeedbackRepo()
	_ = r.Add(&domain.Misprediction{ID: "fb_1", TaskID: "task_1"})

	list1, _ := r.ListByTask("task_1")
	list1[0] = nil

	list2, _ := r.ListByTask("task_1")
	if list2[0] == nil {
		t.Error("expected ListByTask to return a copy, not internal slice")
	}
}

func TestFeedbackRepo_ListUnresolved(t *testing.T) {
	r, _ := newFeedbackRepo()
	_ = r.Add(&domain.Misprediction{ID: "fb_1", TaskID: "task_1", Resolved: false})
	_ = r.Add(&domain.Misprediction{ID: "fb_2", TaskID: "task_1", Resolved: true})
	_ = r.Add(&domain.Misprediction{ID: "fb_3", TaskID: "task_1", Resolved: false})

	list, err := r.ListUnresolved("task_1")
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}
	if len(list) != 2 {
		t.Errorf("expected 2 unresolved entries, got %d", len(list))
	}
	for _, f := range list {
		if f.Resolved {
			t.Errorf("expected only unresolved entries, found %q resolved", f.ID)
		}
	}
}

func TestFeedbackRepo_ListUnresolved_Empty(t *testing.T) {
	r, _ := newFeedbackRepo()

	list, err := r.ListUnresolved("task_1")
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}
	if len(list) != 0 {
		t.Errorf("expected 0 unresolved entries, got %d", len(list))
	}
}

func TestFeedbackRepo_ListUnresolved_AllResolved(t *testing.T) {
	r, _ := newFeedbackRepo()
	_ = r.Add(&domain.Misprediction{ID: "fb_1", TaskID: "task_1", Resolved: true})

	list, err := r.ListUnresolved("task_1")
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}
	if len(list) != 0 {
		t.Errorf("expected 0 unresolved entries, got %d", len(list))
	}
}

func TestFeedbackRepo_MarkResolved(t *testing.T) {
	r, _ := newFeedbackRepo()
	_ = r.Add(&domain.Misprediction{ID: "fb_1", TaskID: "task_1", Resolved: false})
	_ = r.Add(&domain.Misprediction{ID: "fb_2", TaskID: "task_1", Resolved: false})
	_ = r.Add(&domain.Misprediction{ID: "fb_3", TaskID: "task_1", Resolved: false})

	if err := r.MarkResolved([]string{"fb_1", "fb_3"}); err != nil {
		t.Fatalf("expected no error, got %v", err)
	}

	unresolved, _ := r.ListUnresolved("task_1")
	if len(unresolved) != 1 || unresolved[0].ID != "fb_2" {
		t.Errorf("expected only fb_2 to remain unresolved, got %v", unresolved)
	}
}

func TestFeedbackRepo_MarkResolved_AcrossTasks(t *testing.T) {
	r, _ := newFeedbackRepo()
	_ = r.Add(&domain.Misprediction{ID: "fb_1", TaskID: "task_1", Resolved: false})
	_ = r.Add(&domain.Misprediction{ID: "fb_2", TaskID: "task_2", Resolved: false})

	if err := r.MarkResolved([]string{"fb_1", "fb_2"}); err != nil {
		t.Fatalf("expected no error, got %v", err)
	}

	u1, _ := r.ListUnresolved("task_1")
	u2, _ := r.ListUnresolved("task_2")
	if len(u1) != 0 {
		t.Errorf("expected 0 unresolved for task_1, got %d", len(u1))
	}
	if len(u2) != 0 {
		t.Errorf("expected 0 unresolved for task_2, got %d", len(u2))
	}
}

func TestFeedbackRepo_MarkResolved_EmptyIDs(t *testing.T) {
	r, _ := newFeedbackRepo()
	_ = r.Add(&domain.Misprediction{ID: "fb_1", TaskID: "task_1", Resolved: false})

	if err := r.MarkResolved([]string{}); err != nil {
		t.Fatalf("expected no error for empty ids, got %v", err)
	}

	unresolved, _ := r.ListUnresolved("task_1")
	if len(unresolved) != 1 {
		t.Errorf("expected entry to remain unresolved, got %d", len(unresolved))
	}
}

func TestFeedbackRepo_MarkResolved_NonexistentID(t *testing.T) {
	r, _ := newFeedbackRepo()
	_ = r.Add(&domain.Misprediction{ID: "fb_1", TaskID: "task_1", Resolved: false})

	if err := r.MarkResolved([]string{"nonexistent"}); err != nil {
		t.Fatalf("expected no error, got %v", err)
	}

	unresolved, _ := r.ListUnresolved("task_1")
	if len(unresolved) != 1 {
		t.Errorf("expected entry to remain unresolved, got %d", len(unresolved))
	}
}
