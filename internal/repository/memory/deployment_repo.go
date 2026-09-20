package memory

import "distillery/internal/domain"

type DeploymentRepo struct{ store *Store }

func NewDeploymentRepo(store *Store) *DeploymentRepo { return &DeploymentRepo{store: store} }

func (r *DeploymentRepo) Create(d *domain.Deployment) error {
	r.store.mu.Lock()
	defer r.store.mu.Unlock()

	r.store.Deployments[d.TaskID] = append(r.store.Deployments[d.TaskID], d)
	r.store.persist()

	return nil
}

func (r *DeploymentRepo) Get(id string) (*domain.Deployment, error) {
	r.store.mu.RLock()
	defer r.store.mu.RUnlock()

	for _, list := range r.store.Deployments {
		for _, d := range list {
			if d.ID == id {
				return d, nil
			}
		}
	}

	return nil, domain.ErrNotFound
}

func (r *DeploymentRepo) GetActiveForTask(taskID string) (*domain.Deployment, error) {
	r.store.mu.RLock()
	defer r.store.mu.RUnlock()

	list := r.store.Deployments[taskID]
	for i := len(list) - 1; i >= 0; i-- {
		if list[i].Status == domain.DeploymentActive {
			return list[i], nil
		}
	}

	return nil, domain.ErrNoDeployment
}

func (r *DeploymentRepo) ListByTask(taskID string) ([]*domain.Deployment, error) {
	r.store.mu.RLock()
	defer r.store.mu.RUnlock()

	list := r.store.Deployments[taskID]
	out := make([]*domain.Deployment, len(list))
	copy(out, list)

	return out, nil
}

func (r *DeploymentRepo) Update(d *domain.Deployment) error {
	r.store.mu.Lock()
	defer r.store.mu.Unlock()

	list := r.store.Deployments[d.TaskID]
	for i, dep := range list {
		if dep.ID == d.ID {
			list[i] = d

			r.store.persist()

			return nil
		}
	}

	return domain.ErrNotFound
}
