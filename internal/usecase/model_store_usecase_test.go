package usecase_test

import (
	"errors"
	"testing"

	"distillery/internal/domain"
	"distillery/internal/usecase"
)

// --- Mock ModelStoreRepository ---.

type mockModelStoreRepo struct {
	stored   []*domain.ModelStore
	err      error
	created  *domain.ModelStore
	updated  *domain.ModelStore
	deleted  string
	getStore *domain.ModelStore
}

func (m *mockModelStoreRepo) Create(s *domain.ModelStore) error {
	if m.err != nil {
		return m.err
	}

	m.created = s
	m.stored = append(m.stored, s)

	return nil
}

func (m *mockModelStoreRepo) Get(id string) (*domain.ModelStore, error) {
	if m.err != nil {
		return nil, m.err
	}

	if m.getStore != nil {
		return m.getStore, nil
	}

	for _, s := range m.stored {
		if s.ID == id {
			return s, nil
		}
	}

	return nil, domain.ErrNotFound
}

func (m *mockModelStoreRepo) List() ([]*domain.ModelStore, error) {
	if m.err != nil {
		return nil, m.err
	}

	return m.stored, nil
}

func (m *mockModelStoreRepo) ListByKind(kind domain.ModelStoreKind) ([]*domain.ModelStore, error) {
	if m.err != nil {
		return nil, m.err
	}

	var out []*domain.ModelStore

	for _, s := range m.stored {
		if s.Kind == kind {
			out = append(out, s)
		}
	}

	return out, nil
}

func (m *mockModelStoreRepo) Update(s *domain.ModelStore) error {
	if m.err != nil {
		return m.err
	}

	m.updated = s
	for i, ex := range m.stored {
		if ex.ID == s.ID {
			m.stored[i] = s

			return nil
		}
	}

	return domain.ErrNotFound
}

func (m *mockModelStoreRepo) Delete(id string) error {
	if m.err != nil {
		return m.err
	}

	m.deleted = id
	for i, s := range m.stored {
		if s.ID == id {
			m.stored = append(m.stored[:i], m.stored[i+1:]...)

			return nil
		}
	}

	return domain.ErrNotFound
}

// --- Mock ModelTransfer ---.

type mockModelTransfer struct {
	files      []domain.ModelFile
	downloaded string // local path.
	uploaded   domain.ModelFile
	err        error
}

func (m *mockModelTransfer) List(_ *domain.ModelStore) ([]domain.ModelFile, error) {
	if m.err != nil {
		return nil, m.err
	}

	return m.files, nil
}

func (m *mockModelTransfer) Download(_ *domain.ModelStore, _ *domain.ModelFile) (string, error) {
	if m.err != nil {
		return "", m.err
	}

	return m.downloaded, nil
}

func (m *mockModelTransfer) Upload(_ *domain.ModelStore, _, _, _ string) (domain.ModelFile, error) {
	if m.err != nil {
		return domain.ModelFile{}, m.err
	}

	return m.uploaded, nil
}

func newModelStoreUC(repo *mockModelStoreRepo, transfer *mockModelTransfer) *usecase.ModelStoreUsecase {
	return usecase.NewModelStoreUsecase(repo, transfer, newStubIDGen())
}

func enabledStore(id string, kind domain.ModelStoreKind) *domain.ModelStore {
	return &domain.ModelStore{ID: id, Name: "s", Type: domain.ModelStoreLocal, Kind: kind, Enabled: true}
}

// --- CreateStore ---.

func TestModelStoreUsecase_CreateStore_Valid(t *testing.T) {
	t.Parallel()

	repo := &mockModelStoreRepo{}
	uc := newModelStoreUC(repo, &mockModelTransfer{})

	s, err := uc.CreateStore("Base Models", domain.ModelStoreLocal, domain.ModelStoreBase, domain.ModelStoreConfig{Path: "/tmp"})
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}

	if s.Name != "Base Models" {
		t.Errorf("expected name 'Base Models', got %q", s.Name)
	}

	if s.Enabled != true {
		t.Error("expected store to be enabled by default")
	}

	if s.ID == "" {
		t.Error("expected non-empty ID")
	}

	if repo.created == nil {
		t.Error("expected Create to be called on repo")
	}
}

func TestModelStoreUsecase_CreateStore_EmptyName(t *testing.T) {
	t.Parallel()

	uc := newModelStoreUC(&mockModelStoreRepo{}, &mockModelTransfer{})

	_, err := uc.CreateStore("", domain.ModelStoreLocal, domain.ModelStoreBase, domain.ModelStoreConfig{})
	if !errors.Is(err, domain.ErrInvalidInput) {
		t.Errorf("expected ErrInvalidInput, got %v", err)
	}
}

func TestModelStoreUsecase_CreateStore_InvalidType(t *testing.T) {
	t.Parallel()

	uc := newModelStoreUC(&mockModelStoreRepo{}, &mockModelTransfer{})

	_, err := uc.CreateStore("x", "unknown", domain.ModelStoreBase, domain.ModelStoreConfig{})
	if !errors.Is(err, domain.ErrInvalidInput) {
		t.Errorf("expected ErrInvalidInput, got %v", err)
	}
}

func TestModelStoreUsecase_CreateStore_InvalidKind(t *testing.T) {
	t.Parallel()

	uc := newModelStoreUC(&mockModelStoreRepo{}, &mockModelTransfer{})

	_, err := uc.CreateStore("x", domain.ModelStoreLocal, "unknown", domain.ModelStoreConfig{})
	if !errors.Is(err, domain.ErrInvalidInput) {
		t.Errorf("expected ErrInvalidInput, got %v", err)
	}
}

// --- List / Get / Update / Delete ---.

func TestModelStoreUsecase_ListStores_All(t *testing.T) {
	t.Parallel()

	repo := &mockModelStoreRepo{stored: []*domain.ModelStore{enabledStore("s1", domain.ModelStoreBase), enabledStore("s2", domain.ModelStoreTrained)}}
	uc := newModelStoreUC(repo, &mockModelTransfer{})

	list, err := uc.ListStores("")
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}

	if len(list) != 2 {
		t.Errorf("expected 2 stores, got %d", len(list))
	}
}

func TestModelStoreUsecase_ListStores_ByKind(t *testing.T) {
	t.Parallel()

	repo := &mockModelStoreRepo{stored: []*domain.ModelStore{enabledStore("s1", domain.ModelStoreBase), enabledStore("s2", domain.ModelStoreTrained)}}
	uc := newModelStoreUC(repo, &mockModelTransfer{})

	list, err := uc.ListStores(domain.ModelStoreBase)
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}

	if len(list) != 1 || list[0].ID != "s1" {
		t.Errorf("expected only base store, got %+v", list)
	}
}

func TestModelStoreUsecase_GetStore(t *testing.T) {
	t.Parallel()

	repo := &mockModelStoreRepo{getStore: enabledStore("s1", domain.ModelStoreBase)}
	uc := newModelStoreUC(repo, &mockModelTransfer{})

	s, err := uc.GetStore("s1")
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}

	if s.ID != "s1" {
		t.Errorf("expected ID 's1', got %q", s.ID)
	}
}

func TestModelStoreUsecase_UpdateStore(t *testing.T) {
	t.Parallel()

	repo := &mockModelStoreRepo{stored: []*domain.ModelStore{enabledStore("s1", domain.ModelStoreBase)}}
	uc := newModelStoreUC(repo, &mockModelTransfer{})

	s, err := uc.UpdateStore("s1", "Renamed", false, domain.ModelStoreConfig{Path: "/new"})
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}

	if s.Name != "Renamed" || s.Enabled {
		t.Errorf("expected renamed and disabled, got %q enabled=%v", s.Name, s.Enabled)
	}

	if repo.updated == nil {
		t.Error("expected Update to be called on repo")
	}
}

func TestModelStoreUsecase_DeleteStore(t *testing.T) {
	t.Parallel()

	repo := &mockModelStoreRepo{stored: []*domain.ModelStore{enabledStore("s1", domain.ModelStoreBase)}}
	uc := newModelStoreUC(repo, &mockModelTransfer{})

	err := uc.DeleteStore("s1")
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}

	if repo.deleted != "s1" {
		t.Errorf("expected Delete called with 's1', got %q", repo.deleted)
	}
}

// --- List / Download / Upload ---.

func TestModelStoreUsecase_ListModels(t *testing.T) {
	t.Parallel()

	transfer := &mockModelTransfer{files: []domain.ModelFile{{Name: "gpt2"}, {Name: "bert"}}}
	uc := newModelStoreUC(&mockModelStoreRepo{getStore: enabledStore("s1", domain.ModelStoreBase)}, transfer)

	files, err := uc.ListModels("s1")
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}

	if len(files) != 2 {
		t.Errorf("expected 2 files, got %d", len(files))
	}
}

func TestModelStoreUsecase_ListModels_Disabled(t *testing.T) {
	t.Parallel()

	s := enabledStore("s1", domain.ModelStoreBase)
	s.Enabled = false
	uc := newModelStoreUC(&mockModelStoreRepo{getStore: s}, &mockModelTransfer{})

	_, err := uc.ListModels("s1")
	if !errors.Is(err, domain.ErrInvalidInput) {
		t.Errorf("expected ErrInvalidInput for disabled store, got %v", err)
	}
}

func TestModelStoreUsecase_ListTrainedModels_WrongKind(t *testing.T) {
	t.Parallel()

	uc := newModelStoreUC(&mockModelStoreRepo{getStore: enabledStore("s1", domain.ModelStoreBase)}, &mockModelTransfer{})

	_, err := uc.ListTrainedModels("s1")
	if !errors.Is(err, domain.ErrInvalidInput) {
		t.Errorf("expected ErrInvalidInput for base store, got %v", err)
	}
}

func TestModelStoreUsecase_ListTrainedModels_Valid(t *testing.T) {
	t.Parallel()

	transfer := &mockModelTransfer{files: []domain.ModelFile{{Name: "my-model"}}}
	uc := newModelStoreUC(&mockModelStoreRepo{getStore: enabledStore("s1", domain.ModelStoreTrained)}, transfer)

	files, err := uc.ListTrainedModels("s1")
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}

	if len(files) != 1 {
		t.Errorf("expected 1 file, got %d", len(files))
	}
}

func TestModelStoreUsecase_DownloadModel(t *testing.T) {
	t.Parallel()

	transfer := &mockModelTransfer{downloaded: "/cache/gpt2"}
	uc := newModelStoreUC(&mockModelStoreRepo{getStore: enabledStore("s1", domain.ModelStoreBase)}, transfer)

	local, err := uc.DownloadModel("s1", &domain.ModelFile{Name: "gpt2"})
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}

	if local != "/cache/gpt2" {
		t.Errorf("expected local path '/cache/gpt2', got %q", local)
	}
}

func TestModelStoreUsecase_DownloadModel_Disabled(t *testing.T) {
	t.Parallel()

	s := enabledStore("s1", domain.ModelStoreBase)
	s.Enabled = false
	uc := newModelStoreUC(&mockModelStoreRepo{getStore: s}, &mockModelTransfer{})

	_, err := uc.DownloadModel("s1", &domain.ModelFile{})
	if !errors.Is(err, domain.ErrInvalidInput) {
		t.Errorf("expected ErrInvalidInput, got %v", err)
	}
}

func TestModelStoreUsecase_UploadTrainedModel(t *testing.T) {
	t.Parallel()

	transfer := &mockModelTransfer{uploaded: domain.ModelFile{Name: "artifact"}}
	uc := newModelStoreUC(&mockModelStoreRepo{getStore: enabledStore("s1", domain.ModelStoreTrained)}, transfer)

	f, err := uc.UploadTrainedModel("s1", "/src", "artifact", "gpt2")
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}

	if f.Name != "artifact" {
		t.Errorf("expected uploaded name 'artifact', got %q", f.Name)
	}
}

func TestModelStoreUsecase_UploadTrainedModel_BaseStore(t *testing.T) {
	t.Parallel()

	uc := newModelStoreUC(&mockModelStoreRepo{getStore: enabledStore("s1", domain.ModelStoreBase)}, &mockModelTransfer{})

	_, err := uc.UploadTrainedModel("s1", "/src", "a", "")
	if !errors.Is(err, domain.ErrInvalidInput) {
		t.Errorf("expected ErrInvalidInput for base store, got %v", err)
	}
}

func TestModelStoreUsecase_UploadBaseModel(t *testing.T) {
	t.Parallel()

	transfer := &mockModelTransfer{uploaded: domain.ModelFile{Name: "base"}}
	uc := newModelStoreUC(&mockModelStoreRepo{getStore: enabledStore("s1", domain.ModelStoreBase)}, transfer)

	f, err := uc.UploadBaseModel("s1", "/src", "base")
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}

	if f.Name != "base" {
		t.Errorf("expected uploaded name 'base', got %q", f.Name)
	}
}

func TestModelStoreUsecase_UploadBaseModel_TrainedStore(t *testing.T) {
	t.Parallel()

	uc := newModelStoreUC(&mockModelStoreRepo{getStore: enabledStore("s1", domain.ModelStoreTrained)}, &mockModelTransfer{})

	_, err := uc.UploadBaseModel("s1", "/src", "b")
	if !errors.Is(err, domain.ErrInvalidInput) {
		t.Errorf("expected ErrInvalidInput for trained store, got %v", err)
	}
}

func TestModelStoreUsecase_GetMissingStore(t *testing.T) {
	t.Parallel()

	uc := newModelStoreUC(&mockModelStoreRepo{}, &mockModelTransfer{})

	_, err := uc.GetStore("nope")
	if !errors.Is(err, domain.ErrNotFound) {
		t.Errorf("expected ErrNotFound, got %v", err)
	}
}
