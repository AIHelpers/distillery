package usecase

import (
	"time"

	"distillery/internal/domain"
)

type TrainingUsecase struct {
	tasks    domain.TaskRepository
	examples domain.ExampleRepository
	jobs     domain.TrainingJobRepository
	selector domain.ModelSelector
	tuner    domain.FineTuner
	idGen    IDGenerator
}

func NewTrainingUsecase(
	tasks domain.TaskRepository,
	examples domain.ExampleRepository,
	jobs domain.TrainingJobRepository,
	selector domain.ModelSelector,
	tuner domain.FineTuner, idGen IDGenerator,
) *TrainingUsecase {
	return &TrainingUsecase{
		tasks:    tasks,
		examples: examples,
		jobs:     jobs,
		selector: selector,
		tuner:    tuner,
		idGen:    idGen,
	}
}

// StartTraining validates the dataset is ready, selects a base model —
// either the one the user explicitly chose (by name) or the auto-recommended
// default — creates a queued/running TrainingJob, and kicks off the
// (simulated) LoRA/QLoRA fine-tune asynchronously. It returns immediately
// with the job; callers poll GetJob for progress.
func (u *TrainingUsecase) StartTraining(taskID string, baseModelID ...string) (*domain.TrainingJob, error) {
	task, err := u.tasks.Get(taskID)
	if err != nil {
		return nil, err
	}

	existingJobs, err := u.jobs.ListByTask(taskID)
	if err != nil {
		return nil, err
	}

	for _, j := range existingJobs {
		if j.Status == domain.TrainingQueued || j.Status == domain.TrainingRunning {
			return nil, domain.ErrAlreadyRunning
		}
	}

	version := 1
	for _, j := range existingJobs {
		if j.Version >= version {
			version = j.Version + 1
		}
	}

	all, err := u.examples.ListByTask(taskID)
	if err != nil {
		return nil, err
	}

	var usable []*domain.Example

	totalIn, totalOut := 0, 0

	for _, e := range all {
		if e.Duplicate || e.Flagged {
			continue
		}

		usable = append(usable, e)
		totalIn += len(e.Input)
		totalOut += len(e.Output)
	}

	if len(usable) < 3 {
		return nil, domain.ErrNotReady
	}

	avgIn, avgOut := totalIn/len(usable), totalOut/len(usable)

	base := u.selector.SelectBaseModel(task, len(usable), avgIn, avgOut)

	// Prefer the user's explicit model choice over the auto-recommendation.
	if len(baseModelID) > 0 && baseModelID[0] != "" {
		chosen := findBaseModel(u.selector.ListBaseModels(), baseModelID[0])
		if chosen != nil {
			base = *chosen
		}
	}

	now := time.Now().UTC()

	job := &domain.TrainingJob{
		ID:        u.idGen.NewID("job"),
		TaskID:    taskID,
		Version:   version,
		BaseModel: base,
		Status:    domain.TrainingRunning,
		Progress:  0,
		CreatedAt: now,
		StartedAt: &now, Metrics: nil, Error: "", CompletedAt: nil,
	}

	err = u.jobs.Create(job)
	if err != nil {
		return nil, err
	}

	u.tuner.Start(job, usable,
		func(progress int) {
			job.Progress = progress
			_ = u.jobs.Update(job)
		},
		func(metrics *domain.TrainingMetrics, err error) {
			completed := time.Now().UTC()

			job.CompletedAt = &completed
			if err != nil {
				job.Status = domain.TrainingFailed
				job.Error = err.Error()
			} else {
				job.Status = domain.TrainingCompleted
				job.Progress = 100
				job.Metrics = metrics
			}

			_ = u.jobs.Update(job)
		},
	)

	return job, nil
}

// ListBaseModels returns the curated catalog of base models available for
// fine-tuning so the user can pick one explicitly.
func (u *TrainingUsecase) ListBaseModels() []domain.BaseModel {
	return u.selector.ListBaseModels()
}

// findBaseModel looks up a model in the catalog by its name (case-insensitive).
func findBaseModel(catalog []domain.BaseModel, name string) *domain.BaseModel {
	for i := range catalog {
		if catalog[i].Name == name || catalog[i].RepoID == name {
			return &catalog[i]
		}
	}

	return nil
}

func (u *TrainingUsecase) GetJob(id string) (*domain.TrainingJob, error) {
	return u.jobs.Get(id)
}

func (u *TrainingUsecase) ListJobs(taskID string) ([]*domain.TrainingJob, error) {
	return u.jobs.ListByTask(taskID)
}

// DeleteJob removes a fine-tuned model (a training job version). Deleting an
// active run is rejected. Any deployment serving this exact version is
// automatically stopped by the repository.
func (u *TrainingUsecase) DeleteJob(jobID string) error {
	job, err := u.jobs.Get(jobID)
	if err != nil {
		return err
	}

	if job.Status == domain.TrainingQueued || job.Status == domain.TrainingRunning {
		return domain.ErrAlreadyRunning
	}

	return u.jobs.Delete(jobID)
}

func (u *TrainingUsecase) LatestCompleted(taskID string) (*domain.TrainingJob, error) {
	return u.jobs.LatestCompleted(taskID)
}
