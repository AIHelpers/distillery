package memory

import (
	"distillery/internal/domain"
)

// TableDatasetRepo implements domain.TableDatasetRepository on the shared
// in-memory store (snapshot-persisted like every other repo).
type TableDatasetRepo struct {
	store *Store
}

// NewTableDatasetRepo builds the repo.
func NewTableDatasetRepo(store *Store) *TableDatasetRepo {
	return &TableDatasetRepo{store: store}
}

// Create stores a new table dataset.
func (r *TableDatasetRepo) Create(t *domain.TableDataset) error {
	r.store.mu.Lock()
	defer r.store.mu.Unlock()

	r.store.TableDatasets = append(r.store.TableDatasets, t)
	r.store.persist()

	return nil
}

// Get returns one table dataset by ID.
func (r *TableDatasetRepo) Get(id string) (*domain.TableDataset, error) {
	r.store.mu.RLock()
	defer r.store.mu.RUnlock()

	for _, t := range r.store.TableDatasets {
		if t.ID == id {
			return t, nil
		}
	}

	return nil, domain.ErrNotFound
}

// ListByTask returns the table datasets for a task, newest first.
func (r *TableDatasetRepo) ListByTask(taskID string) ([]*domain.TableDataset, error) {
	r.store.mu.RLock()
	defer r.store.mu.RUnlock()

	var out []*domain.TableDataset

	for i := len(r.store.TableDatasets) - 1; i >= 0; i-- {
		if r.store.TableDatasets[i].TaskID == taskID {
			out = append(out, r.store.TableDatasets[i])
		}
	}

	return out, nil
}

// Update overwrites a stored table dataset.
func (r *TableDatasetRepo) Update(t *domain.TableDataset) error {
	r.store.mu.Lock()
	defer r.store.mu.Unlock()

	for i, existing := range r.store.TableDatasets {
		if existing.ID == t.ID {
			r.store.TableDatasets[i] = t
			r.store.persist()

			return nil
		}
	}

	return domain.ErrNotFound
}

// Delete removes a table dataset.
func (r *TableDatasetRepo) Delete(id string) error {
	r.store.mu.Lock()
	defer r.store.mu.Unlock()

	for i, existing := range r.store.TableDatasets {
		if existing.ID == id {
			r.store.TableDatasets = append(r.store.TableDatasets[:i], r.store.TableDatasets[i+1:]...)
			r.store.persist()

			return nil
		}
	}

	return domain.ErrNotFound
}
