package usecase_test

import (
	"errors"
	"testing"
	time "time"

	"distillery/internal/domain"
	"distillery/internal/usecase"
)

func newDeploymentUsecase(
	tasks *mockTaskRepo,
	jobs *mockTrainingRepo,
	examples *mockExampleRepo,
	deployments *mockDeploymentRepo,
	engine *mockInferenceEngine,
	exporter *mockExporter,
) *usecase.DeploymentUsecase {
	return usecase.NewDeploymentUsecase(tasks, jobs, examples, deployments, engine, exporter, newStubIDGen())
}

// --- Deploy ---.

func TestDeploymentUsecase_Deploy_Success(t *testing.T) {
	t.Parallel()

	tasks := &mockTaskRepo{task: &domain.Task{ID: "task_1", Name: "", Description: "", Type: "", CreatedAt: time.Time{}, UpdatedAt: time.Time{}}}
	jobs := &mockTrainingRepo{
		latestCompleted: &domain.TrainingJob{ID: "job_1", TaskID: "task_1", Status: domain.TrainingCompleted, Version: 0, BaseModel: domain.BaseModel{Name: "", ParamsBillions: 0, Family: ""}, Progress: 0, Metrics: nil, Error: "", CreatedAt: time.Time{}, StartedAt: nil, CompletedAt: nil},
	}
	deployments := &mockDeploymentRepo{}
	uc := newDeploymentUsecase(tasks, jobs, &mockExampleRepo{}, deployments, &mockInferenceEngine{}, &mockExporter{})

	d, rawKey, err := uc.Deploy("task_1", true)
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}

	if d == nil {
		t.Fatal("expected non-nil deployment")
	}

	if d.TaskID != "task_1" {
		t.Errorf("expected TaskID 'task_1', got %q", d.TaskID)
	}

	if d.TrainingJobID != "job_1" {
		t.Errorf("expected TrainingJobID 'job_1', got %q", d.TrainingJobID)
	}

	if d.Status != domain.DeploymentActive {
		t.Errorf("expected status active, got %v", d.Status)
	}

	if !d.Autoscale {
		t.Error("expected autoscale true")
	}

	if d.Endpoint == "" {
		t.Error("expected non-empty endpoint")
	}

	if d.APIKeyHash == "" {
		t.Error("expected non-empty API key hash")
	}

	if rawKey == "" {
		t.Error("expected non-empty raw API key")
	}

	if d.APIKeyHash == rawKey {
		t.Error("raw key should differ from hash")
	}

	if d.CreatedAt.IsZero() {
		t.Error("expected non-zero CreatedAt")
	}

	if deployments.created == nil || deployments.created.ID != d.ID {
		t.Error("expected deployment to be persisted via Create")
	}
}

func TestDeploymentUsecase_Deploy_TaskNotFound(t *testing.T) {
	t.Parallel()

	uc := newDeploymentUsecase(
		&mockTaskRepo{},
		&mockTrainingRepo{},
		&mockExampleRepo{},
		&mockDeploymentRepo{},
		&mockInferenceEngine{},
		&mockExporter{},
	)

	_, _, err := uc.Deploy("missing", false)
	if !errors.Is(err, domain.ErrNotFound) {
		t.Errorf("expected ErrNotFound, got %v", err)
	}
}

func TestDeploymentUsecase_Deploy_NoCompletedModel(t *testing.T) {
	t.Parallel()

	tasks := &mockTaskRepo{task: &domain.Task{ID: "task_1", Name: "", Description: "", Type: "", CreatedAt: time.Time{}, UpdatedAt: time.Time{}}}
	uc := newDeploymentUsecase(
		tasks,
		&mockTrainingRepo{},
		&mockExampleRepo{},
		&mockDeploymentRepo{},
		&mockInferenceEngine{},
		&mockExporter{},
	)

	_, _, err := uc.Deploy("task_1", false)
	if !errors.Is(err, domain.ErrNoModel) {
		t.Errorf("expected ErrNoModel, got %v", err)
	}
}

func TestDeploymentUsecase_Deploy_TaskRepoError(t *testing.T) {
	t.Parallel()

	tasks := &mockTaskRepo{err: errRepoFailure}
	uc := newDeploymentUsecase(
		tasks,
		&mockTrainingRepo{},
		&mockExampleRepo{},
		&mockDeploymentRepo{},
		&mockInferenceEngine{},
		&mockExporter{},
	)

	_, _, err := uc.Deploy("task_1", false)
	if !errors.Is(err, errRepoFailure) {
		t.Errorf("expected errRepoFailure, got %v", err)
	}
}

func TestDeploymentUsecase_Deploy_LatestCompletedError(t *testing.T) {
	t.Parallel()

	tasks := &mockTaskRepo{task: &domain.Task{ID: "task_1", Name: "", Description: "", Type: "", CreatedAt: time.Time{}, UpdatedAt: time.Time{}}}
	jobs := &mockTrainingRepo{err: errRepoFailure}
	uc := newDeploymentUsecase(
		tasks,
		jobs,
		&mockExampleRepo{},
		&mockDeploymentRepo{},
		&mockInferenceEngine{},
		&mockExporter{},
	)

	_, _, err := uc.Deploy("task_1", false)
	if !errors.Is(err, errRepoFailure) {
		t.Errorf("expected errRepoFailure, got %v", err)
	}
}

func TestDeploymentUsecase_Deploy_CreateError(t *testing.T) {
	t.Parallel()

	tasks := &mockTaskRepo{task: &domain.Task{ID: "task_1", Name: "", Description: "", Type: "", CreatedAt: time.Time{}, UpdatedAt: time.Time{}}}
	jobs := &mockTrainingRepo{
		latestCompleted: &domain.TrainingJob{ID: "job_1", TaskID: "task_1", Status: domain.TrainingCompleted, Version: 0, BaseModel: domain.BaseModel{Name: "", ParamsBillions: 0, Family: ""}, Progress: 0, Metrics: nil, Error: "", CreatedAt: time.Time{}, StartedAt: nil, CompletedAt: nil},
	}
	deployments := &mockDeploymentRepo{err: errRepoFailure}
	uc := newDeploymentUsecase(tasks, jobs, &mockExampleRepo{}, deployments, &mockInferenceEngine{}, &mockExporter{})

	_, _, err := uc.Deploy("task_1", false)
	if !errors.Is(err, errRepoFailure) {
		t.Errorf("expected errRepoFailure, got %v", err)
	}
}

func TestDeploymentUsecase_Deploy_StopsExistingActive(t *testing.T) {
	t.Parallel()

	tasks := &mockTaskRepo{task: &domain.Task{ID: "task_1", Name: "", Description: "", Type: "", CreatedAt: time.Time{}, UpdatedAt: time.Time{}}}
	jobs := &mockTrainingRepo{
		latestCompleted: &domain.TrainingJob{ID: "job_1", TaskID: "task_1", Status: domain.TrainingCompleted, Version: 0, BaseModel: domain.BaseModel{Name: "", ParamsBillions: 0, Family: ""}, Progress: 0, Metrics: nil, Error: "", CreatedAt: time.Time{}, StartedAt: nil, CompletedAt: nil},
	}
	existing := &domain.Deployment{ID: "dep_old", TaskID: "task_1", Status: domain.DeploymentActive, TrainingJobID: "", Endpoint: "", Autoscale: false, RequestCount: 0, APIKeyHash: "", CreatedAt: time.Time{}}
	deployments := &mockDeploymentRepo{active: existing}
	uc := newDeploymentUsecase(tasks, jobs, &mockExampleRepo{}, deployments, &mockInferenceEngine{}, &mockExporter{})

	_, _, err := uc.Deploy("task_1", false)
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}

	if existing.Status != domain.DeploymentStopped {
		t.Errorf("expected existing deployment to be stopped, got %v", existing.Status)
	}

	if deployments.updated != existing {
		t.Error("expected existing deployment to be updated")
	}
}

// --- DeployVersion ---.

func TestDeploymentUsecase_DeployVersion_Success(t *testing.T) {
	t.Parallel()

	tasks := &mockTaskRepo{task: &domain.Task{ID: "task_1", Name: "", Description: "", Type: "", CreatedAt: time.Time{}, UpdatedAt: time.Time{}}}
	jobs := &mockTrainingRepo{
		job: &domain.TrainingJob{ID: "job_1", TaskID: "task_1", Status: domain.TrainingCompleted, Version: 0, BaseModel: domain.BaseModel{Name: "", ParamsBillions: 0, Family: ""}, Progress: 0, Metrics: nil, Error: "", CreatedAt: time.Time{}, StartedAt: nil, CompletedAt: nil},
	}
	deployments := &mockDeploymentRepo{}
	uc := newDeploymentUsecase(tasks, jobs, &mockExampleRepo{}, deployments, &mockInferenceEngine{}, &mockExporter{})

	d, rawKey, err := uc.DeployVersion("task_1", "job_1", false)
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}

	if d.TrainingJobID != "job_1" {
		t.Errorf("expected TrainingJobID 'job_1', got %q", d.TrainingJobID)
	}

	if rawKey == "" {
		t.Error("expected non-empty raw API key")
	}
}

func TestDeploymentUsecase_DeployVersion_TaskNotFound(t *testing.T) {
	t.Parallel()

	uc := newDeploymentUsecase(
		&mockTaskRepo{},
		&mockTrainingRepo{},
		&mockExampleRepo{},
		&mockDeploymentRepo{},
		&mockInferenceEngine{},
		&mockExporter{},
	)

	_, _, err := uc.DeployVersion("missing", "job_1", false)
	if !errors.Is(err, domain.ErrNotFound) {
		t.Errorf("expected ErrNotFound, got %v", err)
	}
}

func TestDeploymentUsecase_DeployVersion_JobNotFound(t *testing.T) {
	t.Parallel()

	tasks := &mockTaskRepo{task: &domain.Task{ID: "task_1", Name: "", Description: "", Type: "", CreatedAt: time.Time{}, UpdatedAt: time.Time{}}}
	uc := newDeploymentUsecase(
		tasks,
		&mockTrainingRepo{},
		&mockExampleRepo{},
		&mockDeploymentRepo{},
		&mockInferenceEngine{},
		&mockExporter{},
	)

	_, _, err := uc.DeployVersion("task_1", "missing", false)
	if !errors.Is(err, domain.ErrNotFound) {
		t.Errorf("expected ErrNotFound, got %v", err)
	}
}

func TestDeploymentUsecase_DeployVersion_NoModelReasons(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name    string
		job     *domain.TrainingJob
		wantMsg string
	}{
		{
			name: "task mismatch",
			job: &domain.TrainingJob{
				ID:        "job_1",
				TaskID:    "other_task",
				Status:    domain.TrainingCompleted,
				BaseModel: domain.BaseModel{Name: "M"},
			},
			wantMsg: "task mismatch",
		},
		{
			name: "not completed",
			job: &domain.TrainingJob{
				ID:        "job_1",
				TaskID:    "task_1",
				Status:    domain.TrainingRunning,
				BaseModel: domain.BaseModel{Name: "M"},
			},
			wantMsg: "non-completed job",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			tasks := &mockTaskRepo{task: &domain.Task{ID: "task_1"}}
			jobs := &mockTrainingRepo{job: tt.job}
			uc := newDeploymentUsecase(
				tasks,
				jobs,
				&mockExampleRepo{},
				&mockDeploymentRepo{},
				&mockInferenceEngine{},
				&mockExporter{},
			)

			_, _, err := uc.DeployVersion("task_1", "job_1", false)
			if !errors.Is(err, domain.ErrNoModel) {
				t.Errorf("expected ErrNoModel for %s, got %v", tt.name, err)
			}

			_ = tt.wantMsg
		})
	}
}

func TestDeploymentUsecase_DeployVersion_JobRepoError(t *testing.T) {
	t.Parallel()

	tasks := &mockTaskRepo{task: &domain.Task{ID: "task_1", Name: "", Description: "", Type: "", CreatedAt: time.Time{}, UpdatedAt: time.Time{}}}
	jobs := &mockTrainingRepo{err: errRepoFailure}
	uc := newDeploymentUsecase(
		tasks,
		jobs,
		&mockExampleRepo{},
		&mockDeploymentRepo{},
		&mockInferenceEngine{},
		&mockExporter{},
	)

	_, _, err := uc.DeployVersion("task_1", "job_1", false)
	if !errors.Is(err, errRepoFailure) {
		t.Errorf("expected errRepoFailure, got %v", err)
	}
}

// --- GetActiveDeployment ---.

func TestDeploymentUsecase_GetActiveDeployment_Success(t *testing.T) {
	t.Parallel()

	deployments := &mockDeploymentRepo{
		active: &domain.Deployment{ID: "dep_1", TaskID: "task_1", Status: domain.DeploymentActive, TrainingJobID: "", Endpoint: "", Autoscale: false, RequestCount: 0, APIKeyHash: "", CreatedAt: time.Time{}},
	}
	uc := newDeploymentUsecase(
		&mockTaskRepo{},
		&mockTrainingRepo{},
		&mockExampleRepo{},
		deployments,
		&mockInferenceEngine{},
		&mockExporter{},
	)

	d, err := uc.GetActiveDeployment("task_1")
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}

	if d.ID != "dep_1" {
		t.Errorf("expected ID 'dep_1', got %q", d.ID)
	}
}

func TestDeploymentUsecase_GetActiveDeployment_None(t *testing.T) {
	t.Parallel()

	uc := newDeploymentUsecase(
		&mockTaskRepo{},
		&mockTrainingRepo{},
		&mockExampleRepo{},
		&mockDeploymentRepo{},
		&mockInferenceEngine{},
		&mockExporter{},
	)

	_, err := uc.GetActiveDeployment("task_1")
	if !errors.Is(err, domain.ErrNotFound) {
		t.Errorf("expected ErrNotFound, got %v", err)
	}
}

func TestDeploymentUsecase_GetActiveDeployment_RepoError(t *testing.T) {
	t.Parallel()

	deployments := &mockDeploymentRepo{err: errRepoFailure}
	uc := newDeploymentUsecase(
		&mockTaskRepo{},
		&mockTrainingRepo{},
		&mockExampleRepo{},
		deployments,
		&mockInferenceEngine{},
		&mockExporter{},
	)

	_, err := uc.GetActiveDeployment("task_1")
	if !errors.Is(err, errRepoFailure) {
		t.Errorf("expected errRepoFailure, got %v", err)
	}
}

// --- ListDeployments ---.

func TestDeploymentUsecase_ListDeployments_Success(t *testing.T) {
	t.Parallel()

	deployments := &mockDeploymentRepo{
		deployments: []*domain.Deployment{
			{ID: "dep_1", TaskID: "task_1"},
			{ID: "dep_2", TaskID: "task_1"},
		},
	}
	uc := newDeploymentUsecase(
		&mockTaskRepo{},
		&mockTrainingRepo{},
		&mockExampleRepo{},
		deployments,
		&mockInferenceEngine{},
		&mockExporter{},
	)

	list, err := uc.ListDeployments("task_1")
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}

	if len(list) != 2 {
		t.Errorf("expected 2 deployments, got %d", len(list))
	}
}

func TestDeploymentUsecase_ListDeployments_RepoError(t *testing.T) {
	t.Parallel()

	deployments := &mockDeploymentRepo{err: errRepoFailure}
	uc := newDeploymentUsecase(
		&mockTaskRepo{},
		&mockTrainingRepo{},
		&mockExampleRepo{},
		deployments,
		&mockInferenceEngine{},
		&mockExporter{},
	)

	_, err := uc.ListDeployments("task_1")
	if !errors.Is(err, errRepoFailure) {
		t.Errorf("expected errRepoFailure, got %v", err)
	}
}

// --- StopDeployment ---.

func TestDeploymentUsecase_StopDeployment_Success(t *testing.T) {
	t.Parallel()

	deployments := &mockDeploymentRepo{
		deployments: []*domain.Deployment{
			{ID: "dep_1", TaskID: "task_1", Status: domain.DeploymentActive},
		},
	}
	uc := newDeploymentUsecase(
		&mockTaskRepo{},
		&mockTrainingRepo{},
		&mockExampleRepo{},
		deployments,
		&mockInferenceEngine{},
		&mockExporter{},
	)

	err := uc.StopDeployment("dep_1")
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}

	if deployments.updated == nil || deployments.updated.Status != domain.DeploymentStopped {
		t.Error("expected deployment to be updated to stopped")
	}
}

func TestDeploymentUsecase_StopDeployment_NotFound(t *testing.T) {
	t.Parallel()

	uc := newDeploymentUsecase(
		&mockTaskRepo{},
		&mockTrainingRepo{},
		&mockExampleRepo{},
		&mockDeploymentRepo{},
		&mockInferenceEngine{},
		&mockExporter{},
	)

	err := uc.StopDeployment("missing")
	if !errors.Is(err, domain.ErrNotFound) {
		t.Errorf("expected ErrNotFound, got %v", err)
	}
}

func TestDeploymentUsecase_StopDeployment_RepoError(t *testing.T) {
	t.Parallel()

	deployments := &mockDeploymentRepo{err: errRepoFailure}
	uc := newDeploymentUsecase(
		&mockTaskRepo{},
		&mockTrainingRepo{},
		&mockExampleRepo{},
		deployments,
		&mockInferenceEngine{},
		&mockExporter{},
	)

	err := uc.StopDeployment("dep_1")
	if !errors.Is(err, errRepoFailure) {
		t.Errorf("expected errRepoFailure, got %v", err)
	}
}

// --- Invoke ---.

func TestDeploymentUsecase_Invoke_Success(t *testing.T) {
	t.Parallel()

	tasks := &mockTaskRepo{task: &domain.Task{ID: "task_1", Name: "", Description: "", Type: "", CreatedAt: time.Time{}, UpdatedAt: time.Time{}}}
	jobs := &mockTrainingRepo{
		latestCompleted: &domain.TrainingJob{ID: "job_1", TaskID: "task_1", Status: domain.TrainingCompleted, Version: 0, BaseModel: domain.BaseModel{Name: "", ParamsBillions: 0, Family: ""}, Progress: 0, Metrics: nil, Error: "", CreatedAt: time.Time{}, StartedAt: nil, CompletedAt: nil},
		job:             &domain.TrainingJob{ID: "job_1", TaskID: "task_1", Status: domain.TrainingCompleted, Version: 0, BaseModel: domain.BaseModel{Name: "", ParamsBillions: 0, Family: ""}, Progress: 0, Metrics: nil, Error: "", CreatedAt: time.Time{}, StartedAt: nil, CompletedAt: nil},
	}
	examples := &mockExampleRepo{
		byTask: []*domain.Example{
			{ID: "e1", Input: "hello", Output: "yes"},
		},
	}
	// Create a real deployment via Deploy to get a valid key hash.
	deployments := &mockDeploymentRepo{}
	engine := &mockInferenceEngine{output: "result", confidence: 0.9}
	uc := newDeploymentUsecase(tasks, jobs, examples, deployments, engine, &mockExporter{})

	d, rawKey, err := uc.Deploy("task_1", false)
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}

	out, conf, err := uc.Invoke(d.ID, rawKey, "hello")
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}

	if out != "result" {
		t.Errorf("expected output 'result', got %q", out)
	}

	if conf != 0.9 {
		t.Errorf("expected confidence 0.9, got %f", conf)
	}

	if engine.predInput != "hello" {
		t.Errorf("expected engine input 'hello', got %q", engine.predInput)
	}

	if engine.predCount != 1 {
		t.Errorf("expected 1 prediction call, got %d", engine.predCount)
	}

	if d.RequestCount != 1 {
		t.Errorf("expected request count 1, got %d", d.RequestCount)
	}
}

func TestDeploymentUsecase_Invoke_DeploymentNotFound(t *testing.T) {
	t.Parallel()

	uc := newDeploymentUsecase(
		&mockTaskRepo{},
		&mockTrainingRepo{},
		&mockExampleRepo{},
		&mockDeploymentRepo{},
		&mockInferenceEngine{},
		&mockExporter{},
	)

	_, _, err := uc.Invoke("missing", "key", "input")
	if !errors.Is(err, domain.ErrNotFound) {
		t.Errorf("expected ErrNotFound, got %v", err)
	}
}

func TestDeploymentUsecase_Invoke_EmptyAPIKey(t *testing.T) {
	t.Parallel()

	deployments := &mockDeploymentRepo{
		getDeployment: &domain.Deployment{ID: "dep_1", Status: domain.DeploymentActive, APIKeyHash: "hash", TaskID: "", TrainingJobID: "", Endpoint: "", Autoscale: false, RequestCount: 0, CreatedAt: time.Time{}},
	}
	uc := newDeploymentUsecase(
		&mockTaskRepo{},
		&mockTrainingRepo{},
		&mockExampleRepo{},
		deployments,
		&mockInferenceEngine{},
		&mockExporter{},
	)

	_, _, err := uc.Invoke("dep_1", "", "input")
	if !errors.Is(err, domain.ErrUnauthorized) {
		t.Errorf("expected ErrUnauthorized, got %v", err)
	}
}

func TestDeploymentUsecaseInvoke_EmptyHash(t *testing.T) {
	t.Parallel()

	deployments := &mockDeploymentRepo{
		getDeployment: &domain.Deployment{ID: "dep_1", Status: domain.DeploymentActive, APIKeyHash: "", TaskID: "", TrainingJobID: "", Endpoint: "", Autoscale: false, RequestCount: 0, CreatedAt: time.Time{}},
	}
	uc := newDeploymentUsecase(
		&mockTaskRepo{},
		&mockTrainingRepo{},
		&mockExampleRepo{},
		deployments,
		&mockInferenceEngine{},
		&mockExporter{},
	)

	_, _, err := uc.Invoke("dep_1", "somekey", "input")
	if !errors.Is(err, domain.ErrUnauthorized) {
		t.Errorf("expected ErrUnauthorized, got %v", err)
	}
}

func TestDeploymentUsecase_Invoke_WrongAPIKey(t *testing.T) {
	t.Parallel()

	deployments := &mockDeploymentRepo{
		getDeployment: &domain.Deployment{ID: "dep_1", Status: domain.DeploymentActive, APIKeyHash: "somehash", TaskID: "", TrainingJobID: "", Endpoint: "", Autoscale: false, RequestCount: 0, CreatedAt: time.Time{}},
	}
	uc := newDeploymentUsecase(
		&mockTaskRepo{},
		&mockTrainingRepo{},
		&mockExampleRepo{},
		deployments,
		&mockInferenceEngine{},
		&mockExporter{},
	)

	_, _, err := uc.Invoke("dep_1", "wrongkey", "input")
	if !errors.Is(err, domain.ErrUnauthorized) {
		t.Errorf("expected ErrUnauthorized, got %v", err)
	}
}

func TestDeploymentUsecase_Invoke_InactiveDeployment(t *testing.T) {
	t.Parallel()

	// Deploy to get a valid key, then stop the deployment.
	tasks := &mockTaskRepo{task: &domain.Task{ID: "task_1", Name: "", Description: "", Type: "", CreatedAt: time.Time{}, UpdatedAt: time.Time{}}}
	jobs := &mockTrainingRepo{
		latestCompleted: &domain.TrainingJob{ID: "job_1", TaskID: "task_1", Status: domain.TrainingCompleted, Version: 0, BaseModel: domain.BaseModel{Name: "", ParamsBillions: 0, Family: ""}, Progress: 0, Metrics: nil, Error: "", CreatedAt: time.Time{}, StartedAt: nil, CompletedAt: nil},
	}
	deployments := &mockDeploymentRepo{}
	uc := newDeploymentUsecase(tasks, jobs, &mockExampleRepo{}, deployments, &mockInferenceEngine{}, &mockExporter{})

	d, rawKey, err := uc.Deploy("task_1", false)
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}

	d.Status = domain.DeploymentStopped

	_, _, err = uc.Invoke(d.ID, rawKey, "input")
	if !errors.Is(err, domain.ErrNoDeployment) {
		t.Errorf("expected ErrNoDeployment, got %v", err)
	}
}

func TestDeploymentUsecase_Invoke_JobNotFound(t *testing.T) {
	t.Parallel()

	tasks := &mockTaskRepo{task: &domain.Task{ID: "task_1", Name: "", Description: "", Type: "", CreatedAt: time.Time{}, UpdatedAt: time.Time{}}}
	jobs := &mockTrainingRepo{
		latestCompleted: &domain.TrainingJob{ID: "job_1", TaskID: "task_1", Status: domain.TrainingCompleted, Version: 0, BaseModel: domain.BaseModel{Name: "", ParamsBillions: 0, Family: ""}, Progress: 0, Metrics: nil, Error: "", CreatedAt: time.Time{}, StartedAt: nil, CompletedAt: nil},
	}
	deployments := &mockDeploymentRepo{}
	uc := newDeploymentUsecase(tasks, jobs, &mockExampleRepo{}, deployments, &mockInferenceEngine{}, &mockExporter{})

	d, rawKey, err := uc.Deploy("task_1", false)
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}

	// Clear the job from the repo so Get fails.
	jobs.job = nil
	jobs.jobs = nil

	_, _, err = uc.Invoke(d.ID, rawKey, "input")
	if !errors.Is(err, domain.ErrNotFound) {
		t.Errorf("expected ErrNotFound, got %v", err)
	}
}

func TestDeploymentUsecase_Invoke_ExamplesError(t *testing.T) {
	t.Parallel()

	tasks := &mockTaskRepo{task: &domain.Task{ID: "task_1", Name: "", Description: "", Type: "", CreatedAt: time.Time{}, UpdatedAt: time.Time{}}}
	jobs := &mockTrainingRepo{
		latestCompleted: &domain.TrainingJob{ID: "job_1", TaskID: "task_1", Status: domain.TrainingCompleted, Version: 0, BaseModel: domain.BaseModel{Name: "", ParamsBillions: 0, Family: ""}, Progress: 0, Metrics: nil, Error: "", CreatedAt: time.Time{}, StartedAt: nil, CompletedAt: nil},
		job:             &domain.TrainingJob{ID: "job_1", TaskID: "task_1", Status: domain.TrainingCompleted, Version: 0, BaseModel: domain.BaseModel{Name: "", ParamsBillions: 0, Family: ""}, Progress: 0, Metrics: nil, Error: "", CreatedAt: time.Time{}, StartedAt: nil, CompletedAt: nil},
	}
	examples := &mockExampleRepo{err: errRepoFailure}
	deployments := &mockDeploymentRepo{}
	uc := newDeploymentUsecase(tasks, jobs, examples, deployments, &mockInferenceEngine{}, &mockExporter{})

	d, rawKey, err := uc.Deploy("task_1", false)
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}

	_, _, err = uc.Invoke(d.ID, rawKey, "input")
	if !errors.Is(err, errRepoFailure) {
		t.Errorf("expected errRepoFailure, got %v", err)
	}
}

func TestDeploymentUsecase_Invoke_DeploymentRepoError(t *testing.T) {
	t.Parallel()

	deployments := &mockDeploymentRepo{err: errRepoFailure}
	uc := newDeploymentUsecase(
		&mockTaskRepo{},
		&mockTrainingRepo{},
		&mockExampleRepo{},
		deployments,
		&mockInferenceEngine{},
		&mockExporter{},
	)

	_, _, err := uc.Invoke("dep_1", "key", "input")
	if !errors.Is(err, errRepoFailure) {
		t.Errorf("expected errRepoFailure, got %v", err)
	}
}

// --- InvokeBatch ---.

func TestDeploymentUsecase_InvokeBatch_Success(t *testing.T) {
	t.Parallel()

	tasks := &mockTaskRepo{task: &domain.Task{ID: "task_1", Name: "", Description: "", Type: "", CreatedAt: time.Time{}, UpdatedAt: time.Time{}}}
	jobs := &mockTrainingRepo{
		latestCompleted: &domain.TrainingJob{ID: "job_1", TaskID: "task_1", Status: domain.TrainingCompleted, Version: 0, BaseModel: domain.BaseModel{Name: "", ParamsBillions: 0, Family: ""}, Progress: 0, Metrics: nil, Error: "", CreatedAt: time.Time{}, StartedAt: nil, CompletedAt: nil},
		job:             &domain.TrainingJob{ID: "job_1", TaskID: "task_1", Status: domain.TrainingCompleted, Version: 0, BaseModel: domain.BaseModel{Name: "", ParamsBillions: 0, Family: ""}, Progress: 0, Metrics: nil, Error: "", CreatedAt: time.Time{}, StartedAt: nil, CompletedAt: nil},
	}
	examples := &mockExampleRepo{
		byTask: []*domain.Example{
			{ID: "e1", Input: "hello", Output: "yes"},
		},
	}
	deployments := &mockDeploymentRepo{}
	engine := &mockInferenceEngine{output: "out", confidence: 0.8}
	uc := newDeploymentUsecase(tasks, jobs, examples, deployments, engine, &mockExporter{})

	d, rawKey, err := uc.Deploy("task_1", false)
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}

	results, err := uc.InvokeBatch(d.ID, rawKey, []string{"  hello  ", "", "  ", "world"})
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}
	// Empty/whitespace inputs are skipped; only "hello" and "world" remain.
	if len(results) != 2 {
		t.Fatalf("expected 2 results (empty skipped), got %d", len(results))
	}

	if results[0].Input != "hello" {
		t.Errorf("expected first input 'hello' (trimmed), got %q", results[0].Input)
	}

	if results[1].Input != "world" {
		t.Errorf("expected second input 'world', got %q", results[1].Input)
	}

	if results[0].Output != "out" {
		t.Errorf("expected output 'out', got %q", results[0].Output)
	}

	if engine.predCount != 2 {
		t.Errorf("expected 2 prediction calls, got %d", engine.predCount)
	}

	if d.RequestCount != 2 {
		t.Errorf("expected request count 2, got %d", d.RequestCount)
	}
}

func TestDeploymentUsecase_InvokeBatch_DeploymentNotFound(t *testing.T) {
	t.Parallel()

	uc := newDeploymentUsecase(
		&mockTaskRepo{},
		&mockTrainingRepo{},
		&mockExampleRepo{},
		&mockDeploymentRepo{},
		&mockInferenceEngine{},
		&mockExporter{},
	)

	_, err := uc.InvokeBatch("missing", "key", []string{"input"})
	if !errors.Is(err, domain.ErrNotFound) {
		t.Errorf("expected ErrNotFound, got %v", err)
	}
}

func TestDeploymentUsecase_InvokeBatch_Unauthorized(t *testing.T) {
	t.Parallel()

	deployments := &mockDeploymentRepo{
		getDeployment: &domain.Deployment{ID: "dep_1", Status: domain.DeploymentActive, APIKeyHash: "hash", TaskID: "", TrainingJobID: "", Endpoint: "", Autoscale: false, RequestCount: 0, CreatedAt: time.Time{}},
	}
	uc := newDeploymentUsecase(
		&mockTaskRepo{},
		&mockTrainingRepo{},
		&mockExampleRepo{},
		deployments,
		&mockInferenceEngine{},
		&mockExporter{},
	)

	_, err := uc.InvokeBatch("dep_1", "wrongkey", []string{"input"})
	if !errors.Is(err, domain.ErrUnauthorized) {
		t.Errorf("expected ErrUnauthorized, got %v", err)
	}
}

func TestDeploymentUsecase_InvokeBatch_Inactive(t *testing.T) {
	t.Parallel()

	tasks := &mockTaskRepo{task: &domain.Task{ID: "task_1", Name: "", Description: "", Type: "", CreatedAt: time.Time{}, UpdatedAt: time.Time{}}}
	jobs := &mockTrainingRepo{
		latestCompleted: &domain.TrainingJob{ID: "job_1", TaskID: "task_1", Status: domain.TrainingCompleted, Version: 0, BaseModel: domain.BaseModel{Name: "", ParamsBillions: 0, Family: ""}, Progress: 0, Metrics: nil, Error: "", CreatedAt: time.Time{}, StartedAt: nil, CompletedAt: nil},
	}
	deployments := &mockDeploymentRepo{}
	uc := newDeploymentUsecase(tasks, jobs, &mockExampleRepo{}, deployments, &mockInferenceEngine{}, &mockExporter{})

	d, rawKey, err := uc.Deploy("task_1", false)
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}

	d.Status = domain.DeploymentStopped

	_, err = uc.InvokeBatch(d.ID, rawKey, []string{"input"})
	if !errors.Is(err, domain.ErrNoDeployment) {
		t.Errorf("expected ErrNoDeployment, got %v", err)
	}
}

func TestDeploymentUsecase_InvokeBatch_JobNotFound(t *testing.T) {
	t.Parallel()

	tasks := &mockTaskRepo{task: &domain.Task{ID: "task_1", Name: "", Description: "", Type: "", CreatedAt: time.Time{}, UpdatedAt: time.Time{}}}
	jobs := &mockTrainingRepo{
		latestCompleted: &domain.TrainingJob{ID: "job_1", TaskID: "task_1", Status: domain.TrainingCompleted, Version: 0, BaseModel: domain.BaseModel{Name: "", ParamsBillions: 0, Family: ""}, Progress: 0, Metrics: nil, Error: "", CreatedAt: time.Time{}, StartedAt: nil, CompletedAt: nil},
	}
	deployments := &mockDeploymentRepo{}
	uc := newDeploymentUsecase(tasks, jobs, &mockExampleRepo{}, deployments, &mockInferenceEngine{}, &mockExporter{})

	d, rawKey, err := uc.Deploy("task_1", false)
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}

	jobs.job = nil
	jobs.jobs = nil

	_, err = uc.InvokeBatch(d.ID, rawKey, []string{"input"})
	if !errors.Is(err, domain.ErrNotFound) {
		t.Errorf("expected ErrNotFound, got %v", err)
	}
}

func TestDeploymentUsecase_InvokeBatch_ExamplesError(t *testing.T) {
	t.Parallel()

	tasks := &mockTaskRepo{task: &domain.Task{ID: "task_1", Name: "", Description: "", Type: "", CreatedAt: time.Time{}, UpdatedAt: time.Time{}}}
	jobs := &mockTrainingRepo{
		latestCompleted: &domain.TrainingJob{ID: "job_1", TaskID: "task_1", Status: domain.TrainingCompleted, Version: 0, BaseModel: domain.BaseModel{Name: "", ParamsBillions: 0, Family: ""}, Progress: 0, Metrics: nil, Error: "", CreatedAt: time.Time{}, StartedAt: nil, CompletedAt: nil},
		job:             &domain.TrainingJob{ID: "job_1", TaskID: "task_1", Status: domain.TrainingCompleted, Version: 0, BaseModel: domain.BaseModel{Name: "", ParamsBillions: 0, Family: ""}, Progress: 0, Metrics: nil, Error: "", CreatedAt: time.Time{}, StartedAt: nil, CompletedAt: nil},
	}
	examples := &mockExampleRepo{err: errRepoFailure}
	deployments := &mockDeploymentRepo{}
	uc := newDeploymentUsecase(tasks, jobs, examples, deployments, &mockInferenceEngine{}, &mockExporter{})

	d, rawKey, err := uc.Deploy("task_1", false)
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}

	_, err = uc.InvokeBatch(d.ID, rawKey, []string{"input"})
	if !errors.Is(err, errRepoFailure) {
		t.Errorf("expected errRepoFailure, got %v", err)
	}
}

func TestDeploymentUsecase_InvokeBatch_EmptyInputs(t *testing.T) {
	t.Parallel()

	tasks := &mockTaskRepo{task: &domain.Task{ID: "task_1", Name: "", Description: "", Type: "", CreatedAt: time.Time{}, UpdatedAt: time.Time{}}}
	jobs := &mockTrainingRepo{
		latestCompleted: &domain.TrainingJob{ID: "job_1", TaskID: "task_1", Status: domain.TrainingCompleted, Version: 0, BaseModel: domain.BaseModel{Name: "", ParamsBillions: 0, Family: ""}, Progress: 0, Metrics: nil, Error: "", CreatedAt: time.Time{}, StartedAt: nil, CompletedAt: nil},
		job:             &domain.TrainingJob{ID: "job_1", TaskID: "task_1", Status: domain.TrainingCompleted, Version: 0, BaseModel: domain.BaseModel{Name: "", ParamsBillions: 0, Family: ""}, Progress: 0, Metrics: nil, Error: "", CreatedAt: time.Time{}, StartedAt: nil, CompletedAt: nil},
	}
	deployments := &mockDeploymentRepo{}
	engine := &mockInferenceEngine{}
	uc := newDeploymentUsecase(tasks, jobs, &mockExampleRepo{}, deployments, engine, &mockExporter{})

	d, rawKey, err := uc.Deploy("task_1", false)
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}

	results, err := uc.InvokeBatch(d.ID, rawKey, []string{"  ", ""})
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}

	if len(results) != 0 {
		t.Errorf("expected 0 results for all-empty inputs, got %d", len(results))
	}

	if engine.predCount != 0 {
		t.Errorf("expected 0 prediction calls, got %d", engine.predCount)
	}
}

// --- Export ---.

func TestDeploymentUsecase_Export_Success(t *testing.T) {
	t.Parallel()

	tasks := &mockTaskRepo{task: &domain.Task{ID: "task_1", Name: "My Task", Description: "", Type: "", CreatedAt: time.Time{}, UpdatedAt: time.Time{}}}
	jobs := &mockTrainingRepo{
		latestCompleted: &domain.TrainingJob{ID: "job_1", TaskID: "task_1", Status: domain.TrainingCompleted, Version: 0, BaseModel: domain.BaseModel{Name: "", ParamsBillions: 0, Family: ""}, Progress: 0, Metrics: nil, Error: "", CreatedAt: time.Time{}, StartedAt: nil, CompletedAt: nil},
	}
	exporter := &mockExporter{bytes: []byte("archive"), filename: "export.zip"}
	uc := newDeploymentUsecase(tasks, jobs, &mockExampleRepo{}, &mockDeploymentRepo{}, &mockInferenceEngine{}, exporter)

	data, filename, err := uc.Export("task_1")
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}

	if string(data) != "archive" {
		t.Errorf("expected data 'archive', got %q", string(data))
	}

	if filename != "export.zip" {
		t.Errorf("expected filename 'export.zip', got %q", filename)
	}

	if !exporter.called {
		t.Error("expected exporter.BuildExport to be called")
	}
}

func TestDeploymentUsecase_Export_TaskNotFound(t *testing.T) {
	t.Parallel()

	uc := newDeploymentUsecase(
		&mockTaskRepo{},
		&mockTrainingRepo{},
		&mockExampleRepo{},
		&mockDeploymentRepo{},
		&mockInferenceEngine{},
		&mockExporter{},
	)

	_, _, err := uc.Export("missing")
	if !errors.Is(err, domain.ErrNotFound) {
		t.Errorf("expected ErrNotFound, got %v", err)
	}
}

func TestDeploymentUsecase_Export_NoCompletedModel(t *testing.T) {
	t.Parallel()

	tasks := &mockTaskRepo{task: &domain.Task{ID: "task_1", Name: "", Description: "", Type: "", CreatedAt: time.Time{}, UpdatedAt: time.Time{}}}
	uc := newDeploymentUsecase(
		tasks,
		&mockTrainingRepo{},
		&mockExampleRepo{},
		&mockDeploymentRepo{},
		&mockInferenceEngine{},
		&mockExporter{},
	)

	_, _, err := uc.Export("task_1")
	if !errors.Is(err, domain.ErrNoModel) {
		t.Errorf("expected ErrNoModel, got %v", err)
	}
}

func TestDeploymentUsecase_Export_TaskRepoError(t *testing.T) {
	t.Parallel()

	tasks := &mockTaskRepo{err: errRepoFailure}
	uc := newDeploymentUsecase(
		tasks,
		&mockTrainingRepo{},
		&mockExampleRepo{},
		&mockDeploymentRepo{},
		&mockInferenceEngine{},
		&mockExporter{},
	)

	_, _, err := uc.Export("task_1")
	if !errors.Is(err, errRepoFailure) {
		t.Errorf("expected errRepoFailure, got %v", err)
	}
}

func TestDeploymentUsecase_Export_LatestCompletedError(t *testing.T) {
	t.Parallel()

	tasks := &mockTaskRepo{task: &domain.Task{ID: "task_1", Name: "", Description: "", Type: "", CreatedAt: time.Time{}, UpdatedAt: time.Time{}}}
	jobs := &mockTrainingRepo{err: errRepoFailure}
	uc := newDeploymentUsecase(
		tasks,
		jobs,
		&mockExampleRepo{},
		&mockDeploymentRepo{},
		&mockInferenceEngine{},
		&mockExporter{},
	)

	_, _, err := uc.Export("task_1")
	if !errors.Is(err, errRepoFailure) {
		t.Errorf("expected errRepoFailure, got %v", err)
	}
}

func TestDeploymentUsecase_Export_ExporterError(t *testing.T) {
	t.Parallel()

	tasks := &mockTaskRepo{task: &domain.Task{ID: "task_1", Name: "", Description: "", Type: "", CreatedAt: time.Time{}, UpdatedAt: time.Time{}}}
	jobs := &mockTrainingRepo{
		latestCompleted: &domain.TrainingJob{ID: "job_1", TaskID: "task_1", Status: domain.TrainingCompleted, Version: 0, BaseModel: domain.BaseModel{Name: "", ParamsBillions: 0, Family: ""}, Progress: 0, Metrics: nil, Error: "", CreatedAt: time.Time{}, StartedAt: nil, CompletedAt: nil},
	}
	exporter := &mockExporter{err: errRepoFailure}
	uc := newDeploymentUsecase(tasks, jobs, &mockExampleRepo{}, &mockDeploymentRepo{}, &mockInferenceEngine{}, exporter)

	_, _, err := uc.Export("task_1")
	if !errors.Is(err, errRepoFailure) {
		t.Errorf("expected errRepoFailure, got %v", err)
	}
}
