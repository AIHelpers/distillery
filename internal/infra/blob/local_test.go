package blob_test

import (
	"errors"
	"testing"

	"distillery/internal/infra/blob"
)

func TestLocalStore_PutGetDelete(t *testing.T) {
	t.Parallel()

	store := blob.NewLocalStore(t.TempDir())

	key, err := store.Put("task_1", []byte("hello"), ".png")
	if err != nil {
		t.Fatalf("Put: %v", err)
	}

	if key == "" {
		t.Fatal("expected a non-empty key")
	}

	got, err := store.Get("task_1", key)
	if err != nil {
		t.Fatalf("Get: %v", err)
	}

	if string(got) != "hello" {
		t.Errorf("expected %q, got %q", "hello", got)
	}

	_, ok := store.Stat("task_1", key)
	if !ok {
		t.Error("expected Stat to find the blob")
	}

	err = store.Delete("task_1", key)
	if err != nil {
		t.Fatalf("Delete: %v", err)
	}

	_, err = store.Get("task_1", key)
	if err == nil {
		t.Error("expected an error reading a deleted blob")
	}

	// Deleting a missing key is not an error.
	err = store.Delete("task_1", key)
	if err != nil {
		t.Errorf("expected deleting a missing key to be a no-op, got %v", err)
	}
}

func TestLocalStore_ListKeysAndDeleteTask(t *testing.T) {
	t.Parallel()

	store := blob.NewLocalStore(t.TempDir())

	k1, _ := store.Put("task_1", []byte("a"), ".png")
	k2, _ := store.Put("task_1", []byte("b"), ".jpg")
	_, _ = store.Put("task_2", []byte("c"), ".png")

	keys, err := store.ListKeys("task_1")
	if err != nil {
		t.Fatalf("ListKeys: %v", err)
	}

	if len(keys) != 2 {
		t.Fatalf("expected 2 keys for task_1, got %d (%v)", len(keys), keys)
	}

	found := map[string]bool{}
	for _, k := range keys {
		found[k] = true
	}

	if !found[k1] || !found[k2] {
		t.Errorf("expected keys %v to include %q and %q", keys, k1, k2)
	}

	err = store.DeleteTask("task_1")
	if err != nil {
		t.Fatalf("DeleteTask: %v", err)
	}

	keys, err = store.ListKeys("task_1")
	if err != nil {
		t.Fatalf("ListKeys after delete: %v", err)
	}

	if len(keys) != 0 {
		t.Errorf("expected 0 keys after DeleteTask, got %d", len(keys))
	}

	// task_2's blob is untouched.
	keys, _ = store.ListKeys("task_2")
	if len(keys) != 1 {
		t.Errorf("expected task_2's blob to survive task_1's deletion, got %d keys", len(keys))
	}
}

func TestLocalStore_ListKeysUnknownTask(t *testing.T) {
	t.Parallel()

	store := blob.NewLocalStore(t.TempDir())

	keys, err := store.ListKeys("never_created")
	if err != nil {
		t.Fatalf("expected no error for an unknown task dir, got %v", err)
	}

	if len(keys) != 0 {
		t.Errorf("expected no keys, got %v", keys)
	}
}

func TestLocalStore_RejectsUnsafeIDs(t *testing.T) {
	t.Parallel()

	store := blob.NewLocalStore(t.TempDir())

	_, err := store.Put("../escape", []byte("x"), ".png")
	if err == nil {
		t.Error("expected an error for a path-traversal task ID")
	}

	_, err = store.Get("task_1", "../../etc/passwd")
	if err == nil {
		t.Error("expected an error for a path-traversal blob key")
	}
}

func TestLocalStore_PutRejectsOversizedBlob(t *testing.T) {
	t.Parallel()

	store := blob.NewLocalStore(t.TempDir())

	huge := make([]byte, 26<<20) // over the 25 MiB cap.

	_, err := store.Put("task_1", huge, ".png")
	if !errors.Is(err, blob.ErrBlobTooLarge) {
		t.Fatalf("expected ErrBlobTooLarge, got %v", err)
	}
}
