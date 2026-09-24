package modelstore_test

import (
	"errors"
	"os"
	"path/filepath"
	"testing"
	time "time"

	"distillery/internal/domain"
	"distillery/internal/infra/modelstore"
)

func localStore(root string) *domain.ModelStore {
	return &domain.ModelStore{
		ID:      "store_local",
		Name:    "Local Base",
		Type:    domain.ModelStoreLocal,
		Kind:    domain.ModelStoreBase,
		Config:  domain.ModelStoreConfig{Path: root, RepoID: "", Token: "", CacheDir: "", Endpoint: "", Bucket: "", Region: "", AccessKey: "", SecretKey: ""},
		Enabled: true, CreatedAt: time.Time{},
	}
}

func TestTransfer_Local_List(t *testing.T) {
	t.Parallel()

	root := t.TempDir()

	err := os.WriteFile(filepath.Join(root, "model.bin"), []byte("hello"), 0o644)
	if err != nil {
		t.Fatalf("write file: %v", err)
	}

	err = os.MkdirAll(filepath.Join(root, "subdir"), 0o755)
	if err != nil {
		t.Fatalf("mkdir: %v", err)
	}

	tr := modelstore.New("")

	files, err := tr.List(localStore(root))
	if err != nil {
		t.Fatalf("List: unexpected error %v", err)
	}

	if len(files) != 2 {
		t.Fatalf("List: expected 2 entries, got %d", len(files))
	}

	var found bool

	for _, f := range files {
		if f.Name == "model.bin" {
			found = true

			if f.IsDir {
				t.Error("expected model.bin to not be a dir")
			}

			if f.SizeBytes != 5 {
				t.Errorf("expected size 5, got %d", f.SizeBytes)
			}
		}
	}

	if !found {
		t.Error("expected to find model.bin in listing")
	}
}

func TestTransfer_Local_List_MissingPath(t *testing.T) {
	t.Parallel()

	tr := modelstore.New("")

	// Non-existent dir returns empty list (not error).
	files, err := tr.List(localStore(filepath.Join(t.TempDir(), "nope")))
	if err != nil {
		t.Fatalf("List missing path: expected no error, got %v", err)
	}

	if len(files) != 0 {
		t.Errorf("expected empty list, got %d", len(files))
	}
}

func TestTransfer_Local_List_NoPath(t *testing.T) {
	t.Parallel()

	tr := modelstore.New("")
	s := localStore("")
	s.Config.Path = ""

	_, err := tr.List(s)
	if !errors.Is(err, domain.ErrInvalidInput) {
		t.Errorf("expected ErrInvalidInput, got %v", err)
	}
}

func TestTransfer_Local_Download(t *testing.T) {
	t.Parallel()

	root := t.TempDir()

	file := filepath.Join(root, "model.bin")

	err := os.WriteFile(file, []byte("data"), 0o644)
	if err != nil {
		t.Fatalf("write file: %v", err)
	}

	tr := modelstore.New("")

	localPath, err := tr.Download(localStore(root), &domain.ModelFile{Name: "model.bin", Path: file})
	if err != nil {
		t.Fatalf("Download: unexpected error %v", err)
	}

	if localPath != file {
		t.Errorf("expected local path %q, got %q", file, localPath)
	}
}

func TestTransfer_Local_Download_MissingFile(t *testing.T) {
	t.Parallel()

	tr := modelstore.New("")

	_, err := tr.Download(localStore(t.TempDir()), &domain.ModelFile{Name: "nope", Path: filepath.Join(t.TempDir(), "nope")})
	if err == nil {
		t.Fatal("expected error for missing file")
	}
}

func TestTransfer_Local_Download_NoPath(t *testing.T) {
	t.Parallel()

	tr := modelstore.New("")

	_, err := tr.Download(localStore(t.TempDir()), &domain.ModelFile{})
	if !errors.Is(err, domain.ErrInvalidInput) {
		t.Errorf("expected ErrInvalidInput, got %v", err)
	}
}

func TestTransfer_Local_Upload(t *testing.T) {
	t.Parallel()

	root := t.TempDir()
	store := localStore(root)

	// Source file tree.
	src := t.TempDir()

	err := os.WriteFile(filepath.Join(src, "pytorch_model.bin"), []byte("weights"), 0o644)
	if err != nil {
		t.Fatalf("write: %v", err)
	}

	tr := modelstore.New("")

	f, err := tr.Upload(store, src, "my-model", "")
	if err != nil {
		t.Fatalf("Upload: unexpected error %v", err)
	}

	if f.Name != "my-model" {
		t.Errorf("expected name 'my-model', got %q", f.Name)
	}

	// Verify the artifact landed in the store dir.
	_, err = os.Stat(filepath.Join(root, "my-model", "pytorch_model.bin"))
	if err != nil {
		t.Errorf("expected uploaded file to exist: %v", err)
	}
}

func TestTransfer_Local_Upload_NoPath(t *testing.T) {
	t.Parallel()

	tr := modelstore.New("")
	s := localStore("")
	s.Config.Path = ""

	_, err := tr.Upload(s, t.TempDir(), "x", "")
	if !errors.Is(err, domain.ErrInvalidInput) {
		t.Errorf("expected ErrInvalidInput, got %v", err)
	}
}

func TestTransfer_UnsupportedBackend(t *testing.T) {
	t.Parallel()

	tr := modelstore.New("")
	s := &domain.ModelStore{ID: "s", Type: "unknown", Kind: domain.ModelStoreBase, Name: "", Config: domain.ModelStoreConfig{Path: "", RepoID: "", Token: "", CacheDir: "", Endpoint: "", Bucket: "", Region: "", AccessKey: "", SecretKey: ""}, Enabled: false, CreatedAt: time.Time{}}

	_, err := tr.List(s)
	if !errors.Is(err, domain.ErrStoreUnsupported) {
		t.Errorf("List: expected ErrStoreUnsupported, got %v", err)
	}

	_, err = tr.Download(s, &domain.ModelFile{})
	if !errors.Is(err, domain.ErrStoreUnsupported) {
		t.Errorf("Download: expected ErrStoreUnsupported, got %v", err)
	}

	_, err = tr.Upload(s, "", "", "")
	if !errors.Is(err, domain.ErrStoreUnsupported) {
		t.Errorf("Upload: expected ErrStoreUnsupported, got %v", err)
	}
}

func TestTransfer_NewDefaultsCacheDir(t *testing.T) {
	t.Parallel()

	tr := modelstore.New("")

	if tr.LocalCacheDir == "" {
		t.Error("expected default cache dir to be set")
	}
}
