package domain

import "time"

// ModelStoreType identifies the storage backend for a model store.
type ModelStoreType string

const (
	// ModelStoreLocal is a local or mounted filesystem directory.
	ModelStoreLocal ModelStoreType = "local"
	// ModelStoreHuggingFace is a HuggingFace Hub repo.
	ModelStoreHuggingFace ModelStoreType = "huggingface"
	// ModelStoreS3 is any S3-compatible object store (AWS, MinIO, etc).
	ModelStoreS3 ModelStoreType = "s3"
)

// ModelStoreKind indicates what kind of models a store holds.
type ModelStoreKind string

const (
	// ModelStoreBase stores downloadable base models for fine-tuning.
	ModelStoreBase ModelStoreKind = "base"
	// ModelStoreTrained stores trained/fine-tuned model artifacts.
	ModelStoreTrained ModelStoreKind = "trained"
)

// ModelStoreConfig holds backend-specific connection/settings for a store.
type ModelStoreConfig struct {
	// Local backend.
	Path string `json:"path,omitempty"`

	// HuggingFace backend.
	RepoID   string `json:"repo_id,omitempty"`
	Token    string `json:"token,omitempty"`
	CacheDir string `json:"cache_dir,omitempty"`

	// S3-compatible backend.
	Endpoint  string `json:"endpoint,omitempty"`
	Bucket    string `json:"bucket,omitempty"`
	Region    string `json:"region,omitempty"`
	AccessKey string `json:"access_key,omitempty"`
	SecretKey string `json:"secret_key,omitempty"`
}

// ModelStore is a configured model storage backend.
type ModelStore struct {
	ID        string           `json:"id"`
	Name      string           `json:"name"`
	Type      ModelStoreType   `json:"type"`
	Kind      ModelStoreKind   `json:"kind"`
	Config    ModelStoreConfig `json:"config"`
	Enabled   bool             `json:"enabled"`
	CreatedAt time.Time        `json:"created_at"`
}

// ModelFile describes a single model entry inside a store.
type ModelFile struct {
	ID        string `json:"id"`
	Name      string `json:"name"`
	Path      string `json:"path,omitempty"`
	RepoID    string `json:"repo_id,omitempty"`
	SizeBytes int64  `json:"size_bytes"`
	IsDir     bool   `json:"is_dir"`
	BaseModel string `json:"base_model,omitempty"`
}

// ModelStoreRepository persists store configurations.
type ModelStoreRepository interface {
	Create(s *ModelStore) error
	Get(id string) (*ModelStore, error)
	List() ([]*ModelStore, error)
	ListByKind(kind ModelStoreKind) ([]*ModelStore, error)
	Update(s *ModelStore) error
	Delete(id string) error
}

// ModelTransfer is the port for actually moving model files into/out of
// a configured store backend.
type ModelTransfer interface {
	// List returns the model entries present in the store.
	List(store *ModelStore) ([]ModelFile, error)
	// Download fetches a model (identified by name/id) into the local
	// model cache and returns the resolved local path.
	Download(store *ModelStore, file *ModelFile) (localPath string, err error)
	// Upload pushes a local model artifact into the store, returning the
	// store-side entry for it.
	Upload(store *ModelStore, localPath, artifactName, baseModel string) (ModelFile, error)
}

// ErrStoreUnsupported is returned when a transfer backend is not
// implemented or the underlying tooling (e.g. aws CLI) is missing.
var ErrStoreUnsupported = &storeError{"unsupported model store backend or missing CLI"}

// storeError is a domain-ish descriptive error from the model store layer.
type storeError struct{ msg string }

func (e *storeError) Error() string { return e.msg }
