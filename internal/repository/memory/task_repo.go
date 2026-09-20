package memory

import (
	"distillery/internal/domain"
)

type TaskRepo struct{ store *Store }

func NewTaskRepo(store *Store) *TaskRepo { return &TaskRepo{store: store} }

func (r *TaskRepo) Create(t *domain.Task) error {
	r.store.mu.Lock()
	defer r.store.mu.Unlock()

	r.store.Tasks[t.ID] = t
	r.store.persist()

	return nil
}

func (r *TaskRepo) Get(id string) (*domain.Task, error) {
	r.store.mu.RLock()
	defer r.store.mu.RUnlock()

	t, ok := r.store.Tasks[id]
	if !ok {
		return nil, domain.ErrNotFound
	}

	return t, nil
}

func (r *TaskRepo) List() ([]*domain.Task, error) {
	r.store.mu.RLock()
	defer r.store.mu.RUnlock()

	out := make([]*domain.Task, 0, len(r.store.Tasks))
	for _, t := range r.store.Tasks {
		out = append(out, t)
	}

	return out, nil
}

func (r *TaskRepo) Update(t *domain.Task) error {
	r.store.mu.Lock()
	defer r.store.mu.Unlock()

	if _, ok := r.store.Tasks[t.ID]; !ok {
		return domain.ErrNotFound
	}

	r.store.Tasks[t.ID] = t
	r.store.persist()

	return nil
}

func (r *TaskRepo) Delete(id string) error {
	r.store.mu.Lock()
	defer r.store.mu.Unlock()

	delete(r.store.Tasks, id)
	delete(r.store.Examples, id)
	delete(r.store.TrainingJobs, id)
	delete(r.store.Deployments, id)
	delete(r.store.Feedback, id)
	r.store.persist()

	return nil
}
