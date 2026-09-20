package usecase

import (
	"time"

	"distillery/internal/domain"
)

// ModelStoreUsecase orchestrates model store configuration, listing,
// download and upload operations for both base models and trained models.
type ModelStoreUsecase struct {
	stores   domain.ModelStoreRepository
	transfer domain.ModelTransfer
	idGen    IDGenerator
}

// NewModelStoreUsecase wires the store repo and transfer backend together.
func NewModelStoreUsecase(stores domain.ModelStoreRepository, transfer domain.ModelTransfer, idGen IDGenerator) *ModelStoreUsecase {
	return &ModelStoreUsecase{stores: stores, transfer: transfer, idGen: idGen}
}

// CreateStore registers a new model store backend.
func (u *ModelStoreUsecase) CreateStore(
	name string,
	typ domain.ModelStoreType,
	kind domain.ModelStoreKind,
	cfg domain.ModelStoreConfig,
) (*domain.ModelStore, error) {
	if name == "" {
		return nil, domain.ErrInvalidInput
	}

	if typ != domain.ModelStoreLocal && typ != domain.ModelStoreHuggingFace && typ != domain.ModelStoreS3 {
		return nil, domain.ErrInvalidInput
	}

	if kind != domain.ModelStoreBase && kind != domain.ModelStoreTrained {
		return nil, domain.ErrInvalidInput
	}

	s := &domain.ModelStore{
		ID:        u.idGen.NewID("store"),
		Name:      name,
		Type:      typ,
		Kind:      kind,
		Config:    cfg,
		Enabled:   true,
		CreatedAt: time.Now().UTC(),
	}

	err := u.stores.Create(s)
	if err != nil {
		return nil, err
	}

	return s, nil
}

// ListStores returns all configured stores (optionally filtered by kind).
func (u *ModelStoreUsecase) ListStores(kind domain.ModelStoreKind) ([]*domain.ModelStore, error) {
	if kind != "" {
		return u.stores.ListByKind(kind)
	}

	return u.stores.List()
}

// GetStore returns a single store config.
func (u *ModelStoreUsecase) GetStore(id string) (*domain.ModelStore, error) {
	return u.stores.Get(id)
}

// UpdateStore edits a store's name/config/enabled state.
func (u *ModelStoreUsecase) UpdateStore(id, name string, enabled bool, cfg domain.ModelStoreConfig) (*domain.ModelStore, error) {
	s, err := u.stores.Get(id)
	if err != nil {
		return nil, err
	}

	if name != "" {
		s.Name = name
	}

	s.Enabled = enabled
	s.Config = cfg

	err = u.stores.Update(s)
	if err != nil {
		return nil, err
	}

	return s, nil
}

// DeleteStore removes a store configuration.
func (u *ModelStoreUsecase) DeleteStore(id string) error {
	return u.stores.Delete(id)
}

// ListModels enumerates base models present in a base-model store.
func (u *ModelStoreUsecase) ListModels(storeID string) ([]domain.ModelFile, error) {
	s, err := u.stores.Get(storeID)
	if err != nil {
		return nil, err
	}

	if !s.Enabled {
		return nil, domain.ErrInvalidInput
	}

	return u.transfer.List(s)
}

// ListTrainedModels enumerates trained model artifacts in a trained-model store.
func (u *ModelStoreUsecase) ListTrainedModels(storeID string) ([]domain.ModelFile, error) {
	s, err := u.stores.Get(storeID)
	if err != nil {
		return nil, err
	}

	if s.Kind != domain.ModelStoreTrained {
		return nil, domain.ErrInvalidInput
	}

	if !s.Enabled {
		return nil, domain.ErrInvalidInput
	}

	return u.transfer.List(s)
}

// DownloadModel pulls a base model from a store into the local cache so the
// training worker can use it offline, returning the local path.
func (u *ModelStoreUsecase) DownloadModel(storeID string, file *domain.ModelFile) (string, error) {
	s, err := u.stores.Get(storeID)
	if err != nil {
		return "", err
	}

	if !s.Enabled {
		return "", domain.ErrInvalidInput
	}

	return u.transfer.Download(s, file)
}

// UploadTrainedModel pushes a local trained-model artifact directory into a
// trained-model store so it can be shared or reused later.
func (u *ModelStoreUsecase) UploadTrainedModel(storeID, localPath, artifactName, baseModel string) (domain.ModelFile, error) {
	s, err := u.stores.Get(storeID)
	if err != nil {
		return domain.ModelFile{}, err
	}

	if s.Kind != domain.ModelStoreTrained {
		return domain.ModelFile{}, domain.ErrInvalidInput
	}

	if !s.Enabled {
		return domain.ModelFile{}, domain.ErrInvalidInput
	}

	return u.transfer.Upload(s, localPath, artifactName, baseModel)
}

// UploadBaseModel pushes a local model directory into a base-model store so
// it becomes available for fine-tuning.
func (u *ModelStoreUsecase) UploadBaseModel(storeID, localPath, artifactName string) (domain.ModelFile, error) {
	s, err := u.stores.Get(storeID)
	if err != nil {
		return domain.ModelFile{}, err
	}

	if s.Kind != domain.ModelStoreBase {
		return domain.ModelFile{}, domain.ErrInvalidInput
	}

	if !s.Enabled {
		return domain.ModelFile{}, domain.ErrInvalidInput
	}

	return u.transfer.Upload(s, localPath, artifactName, "")
}
