package memory

import (
	"time"

	"distillery/internal/domain"
)

// --- FineTuneRequestRepository ---.

type FineTuneRequestRepo struct{ store *Store }

func NewFineTuneRequestRepo(store *Store) *FineTuneRequestRepo {
	return &FineTuneRequestRepo{store: store}
}

func (r *FineTuneRequestRepo) Create(req *domain.FineTuneRequest) error {
	r.store.mu.Lock()
	defer r.store.mu.Unlock()

	r.store.FineTuneRequests = append(r.store.FineTuneRequests, req)
	r.store.persist()

	return nil
}

func (r *FineTuneRequestRepo) Get(id string) (*domain.FineTuneRequest, error) {
	r.store.mu.RLock()
	defer r.store.mu.RUnlock()

	for _, req := range r.store.FineTuneRequests {
		if req.ID == id {
			return req, nil
		}
	}

	return nil, domain.ErrNotFound
}

func (r *FineTuneRequestRepo) ListByOwner(owner string) ([]*domain.FineTuneRequest, error) {
	r.store.mu.RLock()
	defer r.store.mu.RUnlock()

	var out []*domain.FineTuneRequest

	for _, req := range r.store.FineTuneRequests {
		if req.Owner == owner {
			out = append(out, req)
		}
	}

	return out, nil
}

func (r *FineTuneRequestRepo) List() ([]*domain.FineTuneRequest, error) {
	r.store.mu.RLock()
	defer r.store.mu.RUnlock()

	out := make([]*domain.FineTuneRequest, len(r.store.FineTuneRequests))
	copy(out, r.store.FineTuneRequests)

	return out, nil
}

func (r *FineTuneRequestRepo) Update(req *domain.FineTuneRequest) error {
	r.store.mu.Lock()
	defer r.store.mu.Unlock()

	for i, ex := range r.store.FineTuneRequests {
		if ex.ID == req.ID {
			r.store.FineTuneRequests[i] = req
			r.store.persist()

			return nil
		}
	}

	return domain.ErrNotFound
}

func (r *FineTuneRequestRepo) Delete(id string) error {
	r.store.mu.Lock()
	defer r.store.mu.Unlock()

	for i, req := range r.store.FineTuneRequests {
		if req.ID == id {
			r.store.FineTuneRequests = append(r.store.FineTuneRequests[:i], r.store.FineTuneRequests[i+1:]...)
			r.store.persist()

			return nil
		}
	}

	return domain.ErrNotFound
}

// --- FineTuneJobRepository ---.

type FineTuneJobRepo struct{ store *Store }

func NewFineTuneJobRepo(store *Store) *FineTuneJobRepo {
	return &FineTuneJobRepo{store: store}
}

func (r *FineTuneJobRepo) Create(job *domain.FineTuneJob) error {
	r.store.mu.Lock()
	defer r.store.mu.Unlock()

	r.store.FineTuneJobs = append(r.store.FineTuneJobs, job)
	r.store.persist()

	return nil
}

func (r *FineTuneJobRepo) Get(id string) (*domain.FineTuneJob, error) {
	r.store.mu.RLock()
	defer r.store.mu.RUnlock()

	for _, j := range r.store.FineTuneJobs {
		if j.ID == id {
			return j, nil
		}
	}

	return nil, domain.ErrNotFound
}

func (r *FineTuneJobRepo) GetByRequestID(requestID string) (*domain.FineTuneJob, error) {
	r.store.mu.RLock()
	defer r.store.mu.RUnlock()

	for _, j := range r.store.FineTuneJobs {
		if j.RequestID == requestID {
			return j, nil
		}
	}

	return nil, domain.ErrNotFound
}

func (r *FineTuneJobRepo) List() ([]*domain.FineTuneJob, error) {
	r.store.mu.RLock()
	defer r.store.mu.RUnlock()

	out := make([]*domain.FineTuneJob, len(r.store.FineTuneJobs))
	copy(out, r.store.FineTuneJobs)

	return out, nil
}

func (r *FineTuneJobRepo) ListActive() ([]*domain.FineTuneJob, error) {
	r.store.mu.RLock()
	defer r.store.mu.RUnlock()

	var out []*domain.FineTuneJob

	for _, j := range r.store.FineTuneJobs {
		if j.Status == domain.JobQueued || j.Status == domain.JobPreparing || j.Status == domain.JobRunning {
			out = append(out, j)
		}
	}

	return out, nil
}

func (r *FineTuneJobRepo) Update(job *domain.FineTuneJob) error {
	r.store.mu.Lock()
	defer r.store.mu.Unlock()

	for i, ex := range r.store.FineTuneJobs {
		if ex.ID == job.ID {
			r.store.FineTuneJobs[i] = job
			r.store.persist()

			return nil
		}
	}

	return domain.ErrNotFound
}

func (r *FineTuneJobRepo) UpdateStatus(jobID string, status domain.JobStatus) error {
	r.store.mu.Lock()
	defer r.store.mu.Unlock()

	for _, j := range r.store.FineTuneJobs {
		if j.ID == jobID {
			j.Status = status

			r.store.persist()

			return nil
		}
	}

	return domain.ErrNotFound
}

func (r *FineTuneJobRepo) UpdateProgress(jobID string, progress float64, epoch, step int) error {
	r.store.mu.Lock()
	defer r.store.mu.Unlock()

	for _, j := range r.store.FineTuneJobs {
		if j.ID != jobID {
			continue
		}

		j.Progress = progress
		j.CurrentEpoch = epoch
		j.CurrentStep = step

		r.store.persist()

		return nil
	}

	return domain.ErrNotFound
}

func (r *FineTuneJobRepo) AddMetric(jobID, name string, point domain.MetricPoint) error {
	r.store.mu.Lock()
	defer r.store.mu.Unlock()

	for _, j := range r.store.FineTuneJobs {
		if j.ID != jobID {
			continue
		}

		switch name {
		case string(domain.MetricLoss):
			j.Loss = append(j.Loss, point)
		case "validation_loss":
			j.ValidationLoss = append(j.ValidationLoss, point)
		case "learning_rate":
			j.LearningRate = append(j.LearningRate, point)
		default:
			if j.CustomMetrics == nil {
				j.CustomMetrics = map[string][]domain.MetricPoint{}
			}

			j.CustomMetrics[name] = append(j.CustomMetrics[name], point)
		}

		r.store.persist()

		return nil
	}

	return domain.ErrNotFound
}

func (r *FineTuneJobRepo) SetFinalMetrics(jobID string, metrics map[string]float64) error {
	r.store.mu.Lock()
	defer r.store.mu.Unlock()

	for _, j := range r.store.FineTuneJobs {
		if j.ID == jobID {
			j.FinalMetrics = metrics

			r.store.persist()

			return nil
		}
	}

	return domain.ErrNotFound
}

func (r *FineTuneJobRepo) SetError(jobID, errMsg string) error {
	r.store.mu.Lock()
	defer r.store.mu.Unlock()

	for _, j := range r.store.FineTuneJobs {
		if j.ID != jobID {
			continue
		}

		j.Error = errMsg
		j.Status = domain.JobFailed
		now := time.Now().UTC()
		j.CompletedAt = &now

		r.store.persist()

		return nil
	}

	return domain.ErrNotFound
}

// --- TrainedModelRepository ---.

type TrainedModelRepo struct{ store *Store }

func NewTrainedModelRepo(store *Store) *TrainedModelRepo {
	return &TrainedModelRepo{store: store}
}

func (r *TrainedModelRepo) Create(m *domain.TrainedModel) error {
	r.store.mu.Lock()
	defer r.store.mu.Unlock()

	r.store.TrainedModels = append(r.store.TrainedModels, m)
	r.store.persist()

	return nil
}

func (r *TrainedModelRepo) Get(id string) (*domain.TrainedModel, error) {
	r.store.mu.RLock()
	defer r.store.mu.RUnlock()

	for _, m := range r.store.TrainedModels {
		if m.ID == id {
			return m, nil
		}
	}

	return nil, domain.ErrNotFound
}

func (r *TrainedModelRepo) GetByJobID(jobID string) (*domain.TrainedModel, error) {
	r.store.mu.RLock()
	defer r.store.mu.RUnlock()

	for _, m := range r.store.TrainedModels {
		if m.FineTuneJobID == jobID {
			return m, nil
		}
	}

	return nil, domain.ErrNotFound
}

func (r *TrainedModelRepo) List() ([]*domain.TrainedModel, error) {
	r.store.mu.RLock()
	defer r.store.mu.RUnlock()

	out := make([]*domain.TrainedModel, len(r.store.TrainedModels))
	copy(out, r.store.TrainedModels)

	return out, nil
}

func (r *TrainedModelRepo) ListByLanguage(lang domain.ProgrammingLanguage) ([]*domain.TrainedModel, error) {
	r.store.mu.RLock()
	defer r.store.mu.RUnlock()

	var out []*domain.TrainedModel

	for _, m := range r.store.TrainedModels {
		if m.Language == lang {
			out = append(out, m)
		}
	}

	return out, nil
}

func (r *TrainedModelRepo) ListBySkill(skill domain.SkillCategory) ([]*domain.TrainedModel, error) {
	r.store.mu.RLock()
	defer r.store.mu.RUnlock()

	var out []*domain.TrainedModel

	for _, m := range r.store.TrainedModels {
		if m.Skill == skill {
			out = append(out, m)
		}
	}

	return out, nil
}

func (r *TrainedModelRepo) Update(m *domain.TrainedModel) error {
	r.store.mu.Lock()
	defer r.store.mu.Unlock()

	for i, ex := range r.store.TrainedModels {
		if ex.ID == m.ID {
			r.store.TrainedModels[i] = m
			r.store.persist()

			return nil
		}
	}

	return domain.ErrNotFound
}

// --- DatasetRepository ---.

type DatasetRepo struct{ store *Store }

func NewDatasetRepo(store *Store) *DatasetRepo {
	return &DatasetRepo{store: store}
}

func (r *DatasetRepo) Create(d *domain.DatasetInfo) error {
	r.store.mu.Lock()
	defer r.store.mu.Unlock()

	r.store.Datasets = append(r.store.Datasets, d)
	r.store.persist()

	return nil
}

func (r *DatasetRepo) Get(id string) (*domain.DatasetInfo, error) {
	r.store.mu.RLock()
	defer r.store.mu.RUnlock()

	for _, d := range r.store.Datasets {
		if d.ID == id {
			return d, nil
		}
	}

	return nil, domain.ErrNotFound
}

func (r *DatasetRepo) ListByOwner(owner string) ([]*domain.DatasetInfo, error) {
	r.store.mu.RLock()
	defer r.store.mu.RUnlock()

	var out []*domain.DatasetInfo

	for _, d := range r.store.Datasets {
		if d.Owner == owner {
			out = append(out, d)
		}
	}

	return out, nil
}

func (r *DatasetRepo) List() ([]*domain.DatasetInfo, error) {
	r.store.mu.RLock()
	defer r.store.mu.RUnlock()

	out := make([]*domain.DatasetInfo, len(r.store.Datasets))
	copy(out, r.store.Datasets)

	return out, nil
}

func (r *DatasetRepo) UpdateQuality(datasetID string, q domain.DatasetQuality) error {
	r.store.mu.Lock()
	defer r.store.mu.Unlock()

	for _, d := range r.store.Datasets {
		if d.ID == datasetID {
			d.Quality = q

			r.store.persist()

			return nil
		}
	}

	return domain.ErrNotFound
}

func (r *DatasetRepo) Delete(id string) error {
	r.store.mu.Lock()
	defer r.store.mu.Unlock()

	for i, d := range r.store.Datasets {
		if d.ID == id {
			r.store.Datasets = append(r.store.Datasets[:i], r.store.Datasets[i+1:]...)
			r.store.persist()

			return nil
		}
	}

	return domain.ErrNotFound
}
