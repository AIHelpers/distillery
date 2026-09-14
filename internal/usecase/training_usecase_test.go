package usecase_test

import (
	"errors"
	"testing"
	"time"

	"distillery/internal/domain"
	"distillery/internal/usecase"
)

func newTrainingUsecase(
	tasks *mockTaskRepo,
	examples *mockExampleRepo,
	jobs *mockTrainingRepo,
	selector *mockModelSelector,
	tuner *mockFineTuner,
) *usecase.TrainingUsecase {
	return usecase.NewTrainingUsecase(tasks, examples, jobs, selector, tuner, newStubIDGen())
}

func TestTrainingUsecase_StartTraining_Success(t *testing.T) {
	t.Parallel()

	tasks := &mockTaskRepo{task: &domain.Task{ID: "task_1", Type: domain.TaskClassification, Name: "", Description: "", CreatedAt: time.Time{}, UpdatedAt: time.Time{}}}
	examples := &mockExampleRepo{
		byTask: []*domain.Example{
			{ID: "e1", Input: "hello world", Output: "yes", Source: domain.SourceUser},
			{ID: "e2", Input: "foo bar baz", Output: "no", Source: domain.SourceUser},
			{ID: "e3", Input: "test input", Output: "yes", Source: domain.SourceUser},
		},
	}
	jobs := &mockTrainingRepo{}
	selector := &mockModelSelector{model: domain.BaseModel{Name: "small", ParamsBillions: 0, Family: ""}}
	tuner := &mockFineTuner{}
	uc := newTrainingUsecase(tasks, examples, jobs, selector, tuner)

	job, err := uc.StartTraining("task_1")
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}

	if job == nil {
		t.Fatal("expected non-nil job")
	}

	if job.TaskID != "task_1" {
		t.Errorf("expected TaskID 'task_1', got %q", job.TaskID)
	}

	if job.Version != 1 {
		t.Errorf("expected Version 1, got %d", job.Version)
	}

	if job.Status != domain.TrainingRunning {
		t.Errorf("expected status running, got %v", job.Status)
	}

	if job.BaseModel.Name != "small" {
		t.Errorf("expected base model small, got %v", job.BaseModel)
	}

	if job.CreatedAt.IsZero() {
		t.Error("expected non-zero CreatedAt")
	}

	if job.StartedAt == nil || job.StartedAt.IsZero() {
		t.Error("expected non-zero StartedAt")
	}

	if !tuner.started {
		t.Error("expected tuner.Start to be called")
	}

	if tuner.job == nil || tuner.job.ID != job.ID {
		t.Errorf("expected tuner to receive the created job")
	}

	if len(tuner.examples) != 3 {
		t.Errorf("expected 3 usable examples passed to tuner, got %d", len(tuner.examples))
	}

	if jobs.createdJob == nil || jobs.createdJob.ID != job.ID {
		t.Error("expected job to be persisted via jobs.Create")
	}
}

func TestTrainingUsecase_StartTraining_TaskNotFound(t *testing.T) {
	t.Parallel()

	uc := newTrainingUsecase(
		&mockTaskRepo{},
		&mockExampleRepo{},
		&mockTrainingRepo{},
		&mockModelSelector{},
		&mockFineTuner{},
	)

	_, err := uc.StartTraining("missing")
	if !errors.Is(err, domain.ErrNotFound) {
		t.Errorf("expected ErrNotFound, got %v", err)
	}
}

func TestTrainingUsecase_StartTraining_AlreadyRunning(t *testing.T) {
	t.Parallel()

	tasks := &mockTaskRepo{task: &domain.Task{ID: "task_1", Name: "", Description: "", Type: "", CreatedAt: time.Time{}, UpdatedAt: time.Time{}}}
	jobs := &mockTrainingRepo{
		jobs: []*domain.TrainingJob{
			{ID: "job_1", TaskID: "task_1", Status: domain.TrainingRunning, Version: 1},
		},
	}
	uc := newTrainingUsecase(tasks, &mockExampleRepo{}, jobs, &mockModelSelector{}, &mockFineTuner{})

	_, err := uc.StartTraining("task_1")
	if !errors.Is(err, domain.ErrAlreadyRunning) {
		t.Errorf("expected ErrAlreadyRunning, got %v", err)
	}
}

func TestTrainingUsecase_StartTraining_AlreadyQueued(t *testing.T) {
	t.Parallel()

	tasks := &mockTaskRepo{task: &domain.Task{ID: "task_1", Name: "", Description: "", Type: "", CreatedAt: time.Time{}, UpdatedAt: time.Time{}}}
	jobs := &mockTrainingRepo{
		jobs: []*domain.TrainingJob{
			{ID: "job_1", TaskID: "task_1", Status: domain.TrainingQueued, Version: 1},
		},
	}
	uc := newTrainingUsecase(tasks, &mockExampleRepo{}, jobs, &mockModelSelector{}, &mockFineTuner{})

	_, err := uc.StartTraining("task_1")
	if !errors.Is(err, domain.ErrAlreadyRunning) {
		t.Errorf("expected ErrAlreadyRunning, got %v", err)
	}
}

func TestTrainingUsecase_StartTraining_IncrementsVersion(t *testing.T) {
	t.Parallel()

	tasks := &mockTaskRepo{task: &domain.Task{ID: "task_1", Name: "", Description: "", Type: "", CreatedAt: time.Time{}, UpdatedAt: time.Time{}}}
	examples := &mockExampleRepo{
		byTask: []*domain.Example{
			{ID: "e1", Input: "hello world", Output: "yes"},
			{ID: "e2", Input: "foo bar baz", Output: "no"},
			{ID: "e3", Input: "test input", Output: "yes"},
		},
	}
	jobs := &mockTrainingRepo{
		jobs: []*domain.TrainingJob{
			{ID: "job_1", TaskID: "task_1", Status: domain.TrainingCompleted, Version: 1},
			{ID: "job_2", TaskID: "task_1", Status: domain.TrainingFailed, Version: 2},
		},
	}
	uc := newTrainingUsecase(tasks, examples, jobs, &mockModelSelector{}, &mockFineTuner{})

	job, err := uc.StartTraining("task_1")
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}

	if job.Version != 3 {
		t.Errorf("expected Version 3, got %d", job.Version)
	}
}

func TestTrainingUsecase_StartTraining_NotReady_TooFewUsable(t *testing.T) {
	t.Parallel()

	tasks := &mockTaskRepo{task: &domain.Task{ID: "task_1", Name: "", Description: "", Type: "", CreatedAt: time.Time{}, UpdatedAt: time.Time{}}}
	examples := &mockExampleRepo{
		byTask: []*domain.Example{
			{ID: "e1", Input: "hello", Output: "world"},
			{ID: "e2", Input: "foo", Output: "bar"},
		},
	}
	uc := newTrainingUsecase(tasks, examples, &mockTrainingRepo{}, &mockModelSelector{}, &mockFineTuner{})

	_, err := uc.StartTraining("task_1")
	if !errors.Is(err, domain.ErrNotReady) {
		t.Errorf("expected ErrNotReady with <3 usable, got %v", err)
	}
}

func TestTrainingUsecase_StartTraining_SkipsFlaggedAndDuplicates(t *testing.T) {
	t.Parallel()

	tasks := &mockTaskRepo{task: &domain.Task{ID: "task_1", Name: "", Description: "", Type: "", CreatedAt: time.Time{}, UpdatedAt: time.Time{}}}
	examples := &mockExampleRepo{
		byTask: []*domain.Example{
			{ID: "e1", Input: "hello world", Output: "yes"},
			{ID: "e2", Input: "foo bar baz", Output: "no"},
			{ID: "e3", Input: "test input", Output: "yes"},
			{ID: "e4", Input: "flagged input", Output: "no", Flagged: true},
			{ID: "e5", Input: "dup input", Output: "out", Duplicate: true},
		},
	}
	jobs := &mockTrainingRepo{}
	tuner := &mockFineTuner{}
	uc := newTrainingUsecase(tasks, examples, jobs, &mockModelSelector{}, tuner)

	_, err := uc.StartTraining("task_1")
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}

	if len(tuner.examples) != 3 {
		t.Errorf("expected 3 usable examples (excluding flagged/duplicate), got %d", len(tuner.examples))
	}
}

func TestTrainingUsecase_StartTraining_JobsListError(t *testing.T) {
	t.Parallel()

	tasks := &mockTaskRepo{task: &domain.Task{ID: "task_1", Name: "", Description: "", Type: "", CreatedAt: time.Time{}, UpdatedAt: time.Time{}}}
	jobs := &mockTrainingRepo{err: errRepoFailure}
	uc := newTrainingUsecase(tasks, &mockExampleRepo{}, jobs, &mockModelSelector{}, &mockFineTuner{})

	_, err := uc.StartTraining("task_1")
	if !errors.Is(err, errRepoFailure) {
		t.Errorf("expected errRepoFailure, got %v", err)
	}
}

func TestTrainingUsecase_StartTraining_ExamplesListError(t *testing.T) {
	t.Parallel()

	tasks := &mockTaskRepo{task: &domain.Task{ID: "task_1", Name: "", Description: "", Type: "", CreatedAt: time.Time{}, UpdatedAt: time.Time{}}}
	examples := &mockExampleRepo{err: errRepoFailure}
	uc := newTrainingUsecase(tasks, examples, &mockTrainingRepo{}, &mockModelSelector{}, &mockFineTuner{})

	_, err := uc.StartTraining("task_1")
	if !errors.Is(err, errRepoFailure) {
		t.Errorf("expected errRepoFailure, got %v", err)
	}
}

func TestTrainingUsecase_StartTraining_JobCreateError(t *testing.T) {
	t.Parallel()

	tasks := &mockTaskRepo{task: &domain.Task{ID: "task_1", Name: "", Description: "", Type: "", CreatedAt: time.Time{}, UpdatedAt: time.Time{}}}
	examples := &mockExampleRepo{
		byTask: []*domain.Example{
			{ID: "e1", Input: "hello world", Output: "yes"},
			{ID: "e2", Input: "foo bar baz", Output: "no"},
			{ID: "e3", Input: "test input", Output: "yes"},
		},
	}
	jobs := &mockTrainingRepo{err: errRepoFailure}
	uc := newTrainingUsecase(tasks, examples, jobs, &mockModelSelector{}, &mockFineTuner{})

	_, err := uc.StartTraining("task_1")
	if !errors.Is(err, errRepoFailure) {
		t.Errorf("expected errRepoFailure, got %v", err)
	}
}

func TestTrainingUsecase_StartTraining_CallbacksUpdateJob(t *testing.T) {
	t.Parallel()

	tasks := &mockTaskRepo{task: &domain.Task{ID: "task_1", Name: "", Description: "", Type: "", CreatedAt: time.Time{}, UpdatedAt: time.Time{}}}
	examples := &mockExampleRepo{
		byTask: []*domain.Example{
			{ID: "e1", Input: "hello world", Output: "yes"},
			{ID: "e2", Input: "foo bar baz", Output: "no"},
			{ID: "e3", Input: "test input", Output: "yes"},
		},
	}
	jobs := &mockTrainingRepo{}
	tuner := &mockFineTuner{}
	uc := newTrainingUsecase(tasks, examples, jobs, &mockModelSelector{}, tuner)

	job, _ := uc.StartTraining("task_1")

	// Simulate progress update.
	tuner.onUpdate(50)

	if job.Progress != 50 {
		t.Errorf("expected progress 50 after onUpdate, got %d", job.Progress)
	}

	if jobs.updatedJob == nil || jobs.updatedJob.Progress != 50 {
		t.Error("expected job to be updated on progress callback")
	}

	// Simulate completion.
	metrics := &domain.TrainingMetrics{FinalLoss: 0.1, EvalAccuracy: 0.95, Epochs: 0, TrainExamples: 0}
	tuner.onDone(metrics, nil)

	if job.Status != domain.TrainingCompleted {
		t.Errorf("expected status completed, got %v", job.Status)
	}

	if job.Progress != 100 {
		t.Errorf("expected progress 100, got %d", job.Progress)
	}

	if job.Metrics == nil || job.Metrics.FinalLoss != 0.1 {
		t.Errorf("expected metrics to be set")
	}

	if job.CompletedAt == nil || job.CompletedAt.IsZero() {
		t.Error("expected non-zero CompletedAt")
	}
}

func TestTrainingUsecase_StartTraining_CallbackFailure(t *testing.T) {
	t.Parallel()

	tasks := &mockTaskRepo{task: &domain.Task{ID: "task_1", Name: "", Description: "", Type: "", CreatedAt: time.Time{}, UpdatedAt: time.Time{}}}
	examples := &mockExampleRepo{
		byTask: []*domain.Example{
			{ID: "e1", Input: "hello world", Output: "yes"},
			{ID: "e2", Input: "foo bar baz", Output: "no"},
			{ID: "e3", Input: "test input", Output: "yes"},
		},
	}
	jobs := &mockTrainingRepo{}
	tuner := &mockFineTuner{}
	uc := newTrainingUsecase(tasks, examples, jobs, &mockModelSelector{}, tuner)

	job, _ := uc.StartTraining("task_1")

	tuner.onDone(nil, errRepoFailure)

	if job.Status != domain.TrainingFailed {
		t.Errorf("expected status failed, got %v", job.Status)
	}

	if job.Error != errRepoFailure.Error() {
		t.Errorf("expected error message set, got %q", job.Error)
	}
}

func TestTrainingUsecase_GetJob_Success(t *testing.T) {
	t.Parallel()

	jobs := &mockTrainingRepo{job: &domain.TrainingJob{ID: "job_1", TaskID: "task_1", Version: 0, BaseModel: domain.BaseModel{Name: "", ParamsBillions: 0, Family: ""}, Status: "", Progress: 0, Metrics: nil, Error: "", CreatedAt: time.Time{}, StartedAt: nil, CompletedAt: nil}}
	uc := newTrainingUsecase(&mockTaskRepo{}, &mockExampleRepo{}, jobs, &mockModelSelector{}, &mockFineTuner{})

	job, err := uc.GetJob("job_1")
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}

	if job.ID != "job_1" {
		t.Errorf("expected ID 'job_1', got %q", job.ID)
	}
}

func TestTrainingUsecase_GetJob_NotFound(t *testing.T) {
	t.Parallel()

	uc := newTrainingUsecase(
		&mockTaskRepo{},
		&mockExampleRepo{},
		&mockTrainingRepo{},
		&mockModelSelector{},
		&mockFineTuner{},
	)

	_, err := uc.GetJob("missing")
	if !errors.Is(err, domain.ErrNotFound) {
		t.Errorf("expected ErrNotFound, got %v", err)
	}
}

func TestTrainingUsecase_GetJob_RepoError(t *testing.T) {
	t.Parallel()

	jobs := &mockTrainingRepo{err: errRepoFailure}
	uc := newTrainingUsecase(&mockTaskRepo{}, &mockExampleRepo{}, jobs, &mockModelSelector{}, &mockFineTuner{})

	_, err := uc.GetJob("job_1")
	if !errors.Is(err, errRepoFailure) {
		t.Errorf("expected errRepoFailure, got %v", err)
	}
}

func TestTrainingUsecase_ListJobs(t *testing.T) {
	t.Parallel()

	jobs := &mockTrainingRepo{
		jobs: []*domain.TrainingJob{
			{ID: "job_1", TaskID: "task_1"},
			{ID: "job_2", TaskID: "task_1"},
		},
	}
	uc := newTrainingUsecase(&mockTaskRepo{}, &mockExampleRepo{}, jobs, &mockModelSelector{}, &mockFineTuner{})

	list, err := uc.ListJobs("task_1")
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}

	if len(list) != 2 {
		t.Errorf("expected 2 jobs, got %d", len(list))
	}
}

func TestTrainingUsecase_ListJobs_RepoError(t *testing.T) {
	t.Parallel()

	jobs := &mockTrainingRepo{err: errRepoFailure}
	uc := newTrainingUsecase(&mockTaskRepo{}, &mockExampleRepo{}, jobs, &mockModelSelector{}, &mockFineTuner{})

	_, err := uc.ListJobs("task_1")
	if !errors.Is(err, errRepoFailure) {
		t.Errorf("expected errRepoFailure, got %v", err)
	}
}

func TestTrainingUsecase_LatestCompleted_Success(t *testing.T) {
	t.Parallel()

	now := time.Now().UTC()
	jobs := &mockTrainingRepo{
		latestCompleted: &domain.TrainingJob{ID: "job_1", Status: domain.TrainingCompleted, CompletedAt: &now, TaskID: "", Version: 0, BaseModel: domain.BaseModel{Name: "", ParamsBillions: 0, Family: ""}, Progress: 0, Metrics: nil, Error: "", CreatedAt: time.Time{}, StartedAt: nil},
	}
	uc := newTrainingUsecase(&mockTaskRepo{}, &mockExampleRepo{}, jobs, &mockModelSelector{}, &mockFineTuner{})

	job, err := uc.LatestCompleted("task_1")
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}

	if job.ID != "job_1" {
		t.Errorf("expected ID 'job_1', got %q", job.ID)
	}
}

func TestTrainingUsecase_LatestCompleted_NoModel(t *testing.T) {
	t.Parallel()

	uc := newTrainingUsecase(
		&mockTaskRepo{},
		&mockExampleRepo{},
		&mockTrainingRepo{},
		&mockModelSelector{},
		&mockFineTuner{},
	)

	_, err := uc.LatestCompleted("task_1")
	if !errors.Is(err, domain.ErrNoModel) {
		t.Errorf("expected ErrNoModel, got %v", err)
	}
}

func TestTrainingUsecase_LatestCompleted_RepoError(t *testing.T) {
	t.Parallel()

	jobs := &mockTrainingRepo{err: errRepoFailure}
	uc := newTrainingUsecase(&mockTaskRepo{}, &mockExampleRepo{}, jobs, &mockModelSelector{}, &mockFineTuner{})

	_, err := uc.LatestCompleted("task_1")
	if !errors.Is(err, errRepoFailure) {
		t.Errorf("expected errRepoFailure, got %v", err)
	}
}
