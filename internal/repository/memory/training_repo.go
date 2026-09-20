package memory

import "distillery/internal/domain"

type TrainingRepo struct{ store *Store }

func NewTrainingRepo(store *Store) *TrainingRepo { return &TrainingRepo{store: store} }

func (r *TrainingRepo) Create(j *domain.TrainingJob) error {
	r.store.mu.Lock()
	defer r.store.mu.Unlock()

	r.store.TrainingJobs[j.TaskID] = append(r.store.TrainingJobs[j.TaskID], j)
	r.store.persist()

	return nil
}

func (r *TrainingRepo) Get(id string) (*domain.TrainingJob, error) {
	r.store.mu.RLock()
	defer r.store.mu.RUnlock()

	for _, jobs := range r.store.TrainingJobs {
		for _, j := range jobs {
			if j.ID == id {
				return j, nil
			}
		}
	}

	return nil, domain.ErrNotFound
}

func (r *TrainingRepo) ListByTask(taskID string) ([]*domain.TrainingJob, error) {
	r.store.mu.RLock()
	defer r.store.mu.RUnlock()

	list := r.store.TrainingJobs[taskID]
	out := make([]*domain.TrainingJob, len(list))
	copy(out, list)

	return out, nil
}

func (r *TrainingRepo) Update(j *domain.TrainingJob) error {
	r.store.mu.Lock()
	defer r.store.mu.Unlock()

	list := r.store.TrainingJobs[j.TaskID]
	for i, ex := range list {
		if ex.ID == j.ID {
			list[i] = j

			r.store.persist()

			return nil
		}
	}

	return domain.ErrNotFound
}

// Delete removes a training job from the store and stops any deployments
// that were serving that exact job version.
func (r *TrainingRepo) Delete(id string) error {
	r.store.mu.Lock()
	defer r.store.mu.Unlock()

	for taskID, list := range r.store.TrainingJobs {
		for i, j := range list {
			if j.ID != id {
				continue
			}

			// Stop deployments serving this exact job version.
			deployList := r.store.Deployments[taskID]
			for di, d := range deployList {
				if d.TrainingJobID == id && d.Status == domain.DeploymentActive {
					deployList[di].Status = domain.DeploymentStopped
				}
			}

			// Remove the job from the in-memory list.
			r.store.TrainingJobs[taskID] = append(list[:i], list[i+1:]...)
			if len(r.store.TrainingJobs[taskID]) == 0 {
				delete(r.store.TrainingJobs, taskID)
			}

			r.store.persist()

			return nil
		}
	}

	return domain.ErrNotFound
}

func (r *TrainingRepo) LatestCompleted(taskID string) (*domain.TrainingJob, error) {
	r.store.mu.RLock()
	defer r.store.mu.RUnlock()

	list := r.store.TrainingJobs[taskID]

	var latest *domain.TrainingJob

	for _, j := range list {
		if j.Status != domain.TrainingCompleted {
			continue
		}

		if latest == nil || j.Version > latest.Version {
			latest = j
		}
	}

	if latest == nil {
		return nil, domain.ErrNoModel
	}

	return latest, nil
}
