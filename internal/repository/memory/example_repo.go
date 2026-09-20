package memory

import "distillery/internal/domain"

type ExampleRepo struct{ store *Store }

func NewExampleRepo(store *Store) *ExampleRepo { return &ExampleRepo{store: store} }

func (r *ExampleRepo) Add(e *domain.Example) error {
	r.store.mu.Lock()
	defer r.store.mu.Unlock()

	r.store.Examples[e.TaskID] = append(r.store.Examples[e.TaskID], e)
	r.store.persist()

	return nil
}

func (r *ExampleRepo) AddBatch(es []*domain.Example) error {
	if len(es) == 0 {
		return nil
	}

	r.store.mu.Lock()
	defer r.store.mu.Unlock()

	for _, e := range es {
		r.store.Examples[e.TaskID] = append(r.store.Examples[e.TaskID], e)
	}

	r.store.persist()

	return nil
}

func (r *ExampleRepo) ListByTask(taskID string) ([]*domain.Example, error) {
	r.store.mu.RLock()
	defer r.store.mu.RUnlock()

	list := r.store.Examples[taskID]
	out := make([]*domain.Example, len(list))
	copy(out, list)

	return out, nil
}

func (r *ExampleRepo) Get(taskID, id string) (*domain.Example, error) {
	r.store.mu.RLock()
	defer r.store.mu.RUnlock()

	for _, e := range r.store.Examples[taskID] {
		if e.ID == id {
			return e, nil
		}
	}

	return nil, domain.ErrNotFound
}

func (r *ExampleRepo) Delete(taskID, id string) error {
	r.store.mu.Lock()
	defer r.store.mu.Unlock()

	list := r.store.Examples[taskID]
	for i, e := range list {
		if e.ID == id {
			r.store.Examples[taskID] = append(list[:i:i], list[i+1:]...)
			r.store.persist()

			return nil
		}
	}

	return domain.ErrNotFound
}

func (r *ExampleRepo) Update(e *domain.Example) error {
	r.store.mu.Lock()
	defer r.store.mu.Unlock()

	list := r.store.Examples[e.TaskID]
	for i, ex := range list {
		if ex.ID == e.ID {
			list[i] = e

			r.store.persist()

			return nil
		}
	}

	return domain.ErrNotFound
}

func (r *ExampleRepo) DeleteByTask(taskID string) error {
	r.store.mu.Lock()
	defer r.store.mu.Unlock()

	delete(r.store.Examples, taskID)
	r.store.persist()

	return nil
}
