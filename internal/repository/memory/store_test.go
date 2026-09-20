package memory_test

import (
	"os"
	"path/filepath"
	"testing"
	time "time"

	"distillery/internal/domain"
	"distillery/internal/repository/memory"
)

func TestNewStore_EmptyPath(t *testing.T) {
	t.Parallel()

	s := memory.NewStore("")
	if s == nil {
		t.Fatal("expected non-nil store")
	}

	if len(s.Tasks) != 0 {
		t.Errorf("expected empty Tasks, got %d", len(s.Tasks))
	}
}

func TestNewStore_LoadsSnapshot(t *testing.T) {
	t.Parallel()

	dir := t.TempDir()
	path := filepath.Join(dir, "snapshot.json")

	// Seed a snapshot file on disk via the repos (which call persist internally).
	original := memory.NewStore(path)
	taskRepo := memory.NewTaskRepo(original)
	exampleRepo := memory.NewExampleRepo(original)

	err := taskRepo.Create(&domain.Task{ID: "task_1", Name: "My Task", Description: "", Type: "", CreatedAt: time.Time{}, UpdatedAt: time.Time{}})
	if err != nil {
		t.Fatalf("failed to seed task: %v", err)
	}

	err = exampleRepo.Add(&domain.Example{ID: "ex_1", TaskID: "task_1", Input: "", Output: "", Source: "", Flagged: false, FlagNote: "", Duplicate: false, CreatedAt: time.Time{}})
	if err != nil {
		t.Fatalf("failed to seed example: %v", err)
	}

	loaded := memory.NewStore(path)

	got, ok := loaded.Tasks["task_1"]
	if !ok {
		t.Fatal("expected task_1 to be loaded from snapshot")
	}

	if got.Name != "My Task" {
		t.Errorf("expected name 'My Task', got %q", got.Name)
	}

	if len(loaded.Examples["task_1"]) != 1 {
		t.Errorf("expected 1 example loaded, got %d", len(loaded.Examples["task_1"]))
	}
}

func TestNewStore_LoadInvalidJSON(t *testing.T) {
	t.Parallel()

	dir := t.TempDir()

	path := filepath.Join(dir, "snapshot.json")

	err := os.WriteFile(path, []byte("{invalid json"), 0o644)
	if err != nil {
		t.Fatalf("failed to write invalid snapshot: %v", err)
	}

	s := memory.NewStore(path)
	if len(s.Tasks) != 0 {
		t.Errorf("expected empty state on invalid JSON, got %d tasks", len(s.Tasks))
	}
}

func TestNewStore_LoadMissingFile(t *testing.T) {
	t.Parallel()

	dir := t.TempDir()
	path := filepath.Join(dir, "does_not_exist.json")

	s := memory.NewStore(path)
	if s == nil {
		t.Fatal("expected non-nil store")
	}

	if len(s.Tasks) != 0 {
		t.Errorf("expected empty Tasks, got %d", len(s.Tasks))
	}
}
