package usecase

import (
	"fmt"
	"strings"
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

// TrainingStartOptions carries the optional knobs for starting a training
// run beyond the task ID: an explicit base model choice, and — for a
// DPO/ORPO preference-tuning run — the method and parent SFT job.
type TrainingStartOptions struct {
	// BaseModel optionally names a catalog entry (name or repo ID),
	// overriding the auto-recommended default.
	BaseModel string
	// Method selects "dpo" or "orpo" for a preference-tuning run. Leaving it
	// empty starts (or continues) an ordinary training run for the task's
	// own kind, UNLESS the task's kind is already preference_lm, in which
	// case it defaults to "orpo" (the only method that needs no parent).
	Method string
	// ParentJobID is the completed causal_lm (SFT) job whose adapter is the
	// starting policy for a DPO run (required for method=dpo; optional and
	// unused for method=orpo, which is reference-free).
	ParentJobID string
}

// StartTraining validates the dataset is ready, selects a base model —
// either the one the user explicitly chose (by name) or the auto-recommended
// default — creates a queued/running TrainingJob, and kicks off the
// (simulated) LoRA/QLoRA fine-tune asynchronously. It returns immediately
// with the job; callers poll GetJob for progress.
//
// This is a thin backward-compatible wrapper over StartTrainingWithOptions;
// new callers that need a DPO/ORPO run should call that directly.
func (u *TrainingUsecase) StartTraining(taskID string, baseModelID ...string) (*domain.TrainingJob, error) {
	opts := TrainingStartOptions{}
	if len(baseModelID) > 0 {
		opts.BaseModel = baseModelID[0]
	}

	return u.startTraining(taskID, opts)
}

// StartTrainingWithOptions is StartTraining plus the DPO/ORPO preference-
// tuning options (method, parent job).
func (u *TrainingUsecase) StartTrainingWithOptions(taskID string, opts TrainingStartOptions) (*domain.TrainingJob, error) {
	return u.startTraining(taskID, opts)
}

func (u *TrainingUsecase) startTraining(taskID string, opts TrainingStartOptions) (*domain.TrainingJob, error) {
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

	// The task kind drives the trainer module, dataset schema, and catalog
	// section; pre-kind records default to causal_lm.
	kind := task.Kind
	if !domain.IsValidModelKind(kind) {
		kind = domain.DefaultModelKind
	}

	// A preference-tuning (DPO/ORPO) run is requested either explicitly via
	// opts.Method, or implicitly when the task itself was created with
	// Kind == preference_lm (the from-scratch ORPO case, no parent job).
	// It overrides `kind` for THIS job only — a causal_lm task's own Kind
	// is untouched, so its dataset/readiness view and future SFT reruns
	// keep working exactly as before; the job history simply gains a
	// preference_lm entry (e.g. "SFT v3 -> DPO v4" in version history).
	preferenceRequested := opts.Method != "" || kind == domain.KindPreferenceLM

	var (
		parent  *domain.TrainingJob
		prefCfg domain.PreferenceConfig
	)

	if preferenceRequested {
		if kind != domain.KindCausalLM && kind != domain.KindPreferenceLM {
			return nil, fmt.Errorf("%w: preference tuning is only available for causal_lm or preference_lm tasks", domain.ErrInvalidInput)
		}

		method := domain.PreferenceMethod(strings.ToLower(strings.TrimSpace(opts.Method)))
		if method == "" {
			method = domain.MethodDPO
		}

		if !domain.IsValidPreferenceMethod(method) {
			return nil, fmt.Errorf("%w: method must be %q or %q", domain.ErrInvalidInput, domain.MethodDPO, domain.MethodORPO)
		}

		if strings.TrimSpace(opts.ParentJobID) != "" {
			parent, err = u.jobs.Get(opts.ParentJobID)
			if err != nil {
				return nil, err
			}

			if parent.Kind != domain.KindCausalLM || parent.Status != domain.TrainingCompleted {
				return nil, fmt.Errorf("%w: parent_job_id must reference a completed causal_lm (SFT) job", domain.ErrInvalidInput)
			}
		}

		// Requires a parent SFT job (C2 in the plan): its adapter is the
		// starting policy and, with LoRA, the implicit reference model.
		// Tuning from a base model with no SFT is allowed only with ORPO.
		if method == domain.MethodDPO && parent == nil {
			return nil, fmt.Errorf(
				"%w: dpo requires parent_job_id (a completed SFT job); use method=orpo to tune from a base model directly",
				domain.ErrInvalidInput,
			)
		}

		prefCfg = domain.DefaultPreferenceConfig()
		prefCfg.Method = method
		kind = domain.KindPreferenceLM
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

		// A task can carry examples of more than one kind (preference pairs
		// fed by the Feedback tab live alongside a causal_lm task's SFT
		// examples): only feed the worker examples matching the kind this
		// job actually trains.
		if kind == domain.KindPreferenceLM {
			if e.Kind != domain.KindPreferenceLM {
				continue
			}
		} else if e.Kind != "" && e.Kind != kind {
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

	var base domain.BaseModel
	if kind == domain.KindCausalLM {
		base = u.selector.SelectBaseModel(task, len(usable), avgIn, avgOut)
	} else {
		stats := domain.DatasetStats{TaskID: taskID, Kind: kind, Total: len(usable), UsableCount: len(usable)}
		base = u.selector.Select(kind, stats, domain.SelectionConstraints{})
	}

	// Prefer the user's explicit model choice over the auto-recommendation.
	if opts.BaseModel != "" {
		chosen := findBaseModel(u.selector.ListBaseModels(), opts.BaseModel)
		if chosen != nil {
			base = *chosen
		}
	}

	// A DPO run continues the parent's exact base model — its adapter only
	// loads onto the same architecture it was trained against — so this
	// overrides both the auto-recommendation and any explicit choice above.
	if parent != nil {
		base = parent.BaseModel
	}

	now := time.Now().UTC()

	job := &domain.TrainingJob{
		ID:        u.idGen.NewID("job"),
		TaskID:    taskID,
		Version:   version,
		Kind:      kind,
		BaseModel: base,
		Status:    domain.TrainingRunning,
		Progress:  0,
		CreatedAt: now,
		StartedAt: &now, Metrics: nil, Error: "", CompletedAt: nil,
	}

	// Seed the per-kind hyperparameters so the worker receives explicit
	// values (the trainer falls back to its own defaults for zero fields).
	if kind == domain.KindTokenClassifier {
		job.NER = &domain.NERConfig{MaxLength: 256, Stride: 64, LabelScheme: "BIO"}
		job.LabelSet = task.LabelSet
	}

	if preferenceRequested {
		job.Preference = &prefCfg

		if parent != nil {
			job.ParentJobID = parent.ID
		}
	}

	// Track B: carry the task's JSON schema so the trainer can report a JSON
	// validity rate and inference can use schema-constrained decoding.
	job.JSONSchema = task.JSONSchema

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
