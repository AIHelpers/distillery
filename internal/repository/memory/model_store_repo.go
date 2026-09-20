package memory

import (
	"distillery/internal/domain"
)

// ModelStoreRepo persists model store configurations in the shared store.
type ModelStoreRepo struct{ store *Store }

// NewModelStoreRepo creates a ModelStoreRepo backed by the shared store.
func NewModelStoreRepo(store *Store) *ModelStoreRepo {
	return &ModelStoreRepo{store: store}
}

func (r *ModelStoreRepo) Create(s *domain.ModelStore) error {
	r.store.mu.Lock()
	defer r.store.mu.Unlock()

	r.store.ModelStores = append(r.store.ModelStores, s)
	r.store.persist()

	return nil
}

func (r *ModelStoreRepo) Get(id string) (*domain.ModelStore, error) {
	r.store.mu.RLock()
	defer r.store.mu.RUnlock()

	for _, s := range r.store.ModelStores {
		if s.ID == id {
			return s, nil
		}
	}

	return nil, domain.ErrNotFound
}

func (r *ModelStoreRepo) List() ([]*domain.ModelStore, error) {
	r.store.mu.RLock()
	defer r.store.mu.RUnlock()

	out := make([]*domain.ModelStore, len(r.store.ModelStores))
	copy(out, r.store.ModelStores)

	return out, nil
}

func (r *ModelStoreRepo) ListByKind(kind domain.ModelStoreKind) ([]*domain.ModelStore, error) {
	r.store.mu.RLock()
	defer r.store.mu.RUnlock()

	var out []*domain.ModelStore

	for _, s := range r.store.ModelStores {
		if s.Kind == kind {
			out = append(out, s)
		}
	}

	return out, nil
}

func (r *ModelStoreRepo) Update(s *domain.ModelStore) error {
	r.store.mu.Lock()
	defer r.store.mu.Unlock()

	for i, ex := range r.store.ModelStores {
		if ex.ID == s.ID {
			r.store.ModelStores[i] = s
			r.store.persist()

			return nil
		}
	}

	return domain.ErrNotFound
}

func (r *ModelStoreRepo) Delete(id string) error {
	r.store.mu.Lock()
	defer r.store.mu.Unlock()

	for i, s := range r.store.ModelStores {
		if s.ID == id {
			r.store.ModelStores = append(r.store.ModelStores[:i], r.store.ModelStores[i+1:]...)
			r.store.persist()

			return nil
		}
	}

	return domain.ErrNotFound
}
