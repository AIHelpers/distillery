package memory_test

import (
	"errors"
	"testing"
	"time"

	"distillery/internal/domain"
	"distillery/internal/repository/memory"
)

func newTestModelStore(id, name string, kind domain.ModelStoreKind) *domain.ModelStore {
	return &domain.ModelStore{
		ID:        id,
		Name:      name,
		Type:      domain.ModelStoreLocal,
		Kind:      kind,
		Config:    domain.ModelStoreConfig{Path: "/tmp/models", RepoID: "", Token: "", CacheDir: "", Endpoint: "", Bucket: "", Region: "", AccessKey: "", SecretKey: ""},
		Enabled:   true,
		CreatedAt: time.Now().UTC(),
	}
}

func TestModelStoreRepo_CRUD(t *testing.T) {
	t.Parallel()

	repo := memory.NewModelStoreRepo(memory.NewStore("")) // in-memory, no persistence.

	// Create.
	s := newTestModelStore("store_1", "Local Base", domain.ModelStoreBase)

	err := repo.Create(s)
	if err != nil {
		t.Fatalf("Create: unexpected error %v", err)
	}

	// Get.
	got, err := repo.Get("store_1")
	if err != nil {
		t.Fatalf("Get: unexpected error %v", err)
	}

	if got.Name != "Local Base" {
		t.Errorf("Get: expected name %q, got %q", "Local Base", got.Name)
	}

	// Get missing.
	_, err = repo.Get("missing")
	if !errors.Is(err, domain.ErrNotFound) {
		t.Errorf("Get missing: expected ErrNotFound, got %v", err)
	}

	// List.
	list, err := repo.List()
	if err != nil {
		t.Fatalf("List: unexpected error %v", err)
	}

	if len(list) != 1 {
		t.Fatalf("List: expected 1 store, got %d", len(list))
	}

	// Update.
	s2 := newTestModelStore("store_1", "Renamed", domain.ModelStoreBase)

	s2.Enabled = false

	err = repo.Update(s2)
	if err != nil {
		t.Fatalf("Update: unexpected error %v", err)
	}

	got, _ = repo.Get("store_1")
	if got.Name != "Renamed" || got.Enabled {
		t.Errorf("Update: expected name 'Renamed' and disabled, got %q enabled=%v", got.Name, got.Enabled)
	}

	// Update missing.
	err = repo.Update(newTestModelStore("missing", "X", domain.ModelStoreBase))
	if !errors.Is(err, domain.ErrNotFound) {
		t.Errorf("Update missing: expected ErrNotFound, got %v", err)
	}

	// Delete.
	err = repo.Delete("store_1")
	if err != nil {
		t.Fatalf("Delete: unexpected error %v", err)
	}

	_, err = repo.Get("store_1")
	if !errors.Is(err, domain.ErrNotFound) {
		t.Errorf("Delete: expected ErrNotFound after delete, got %v", err)
	}

	// Delete missing.
	err = repo.Delete("missing")
	if !errors.Is(err, domain.ErrNotFound) {
		t.Errorf("Delete missing: expected ErrNotFound, got %v", err)
	}
}

func TestModelStoreRepo_ListByKind(t *testing.T) {
	t.Parallel()

	repo := memory.NewModelStoreRepo(memory.NewStore(""))

	_ = repo.Create(newTestModelStore("base_1", "B1", domain.ModelStoreBase))
	_ = repo.Create(newTestModelStore("base_2", "B2", domain.ModelStoreBase))
	_ = repo.Create(newTestModelStore("trained_1", "T1", domain.ModelStoreTrained))

	base, err := repo.ListByKind(domain.ModelStoreBase)
	if err != nil {
		t.Fatalf("ListByKind base: unexpected error %v", err)
	}

	if len(base) != 2 {
		t.Errorf("ListByKind base: expected 2, got %d", len(base))
	}

	trained, err := repo.ListByKind(domain.ModelStoreTrained)
	if err != nil {
		t.Fatalf("ListByKind trained: unexpected error %v", err)
	}

	if len(trained) != 1 || trained[0].ID != "trained_1" {
		t.Errorf("ListByKind trained: unexpected result %+v", trained)
	}
}

func TestModelStoreRepo_PersistsToDisk(t *testing.T) {
	t.Parallel()

	dir := t.TempDir()
	path := dir + "/state.json"

	repo := memory.NewModelStoreRepo(memory.NewStore(path))
	_ = repo.Create(newTestModelStore("base_1", "P1", domain.ModelStoreBase))

	// A second store wired to the same path should reload the snapshot.
	repo2 := memory.NewModelStoreRepo(memory.NewStore(path))

	list, err := repo2.List()
	if err != nil {
		t.Fatalf("List after reload: unexpected error %v", err)
	}

	if len(list) != 1 || list[0].ID != "base_1" {
		t.Errorf("expected persisted store 'base_1', got %+v", list)
	}
}
