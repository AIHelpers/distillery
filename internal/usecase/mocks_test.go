package usecase_test

import (
	"errors"

	"distillery/internal/domain"
)

// errRepoFailure is a generic repository error used for testing error paths.
var errRepoFailure = errors.New("repository failure")

// --- Mock TaskRepository ---.

type mockTaskRepo struct {
	task         *domain.Task
	list         []*domain.Task
	err          error
	createCalled bool
	getID        string
	deleteID     string
	updateTask   *domain.Task
}

func (m *mockTaskRepo) Create(t *domain.Task) error {
	m.createCalled = true
	if m.err != nil {
		return m.err
	}
	m.task = t
	return nil
}

func (m *mockTaskRepo) Get(id string) (*domain.Task, error) {
	m.getID = id
	if m.err != nil {
		return nil, m.err
	}
	if m.task != nil {
		return m.task, nil
	}
	return nil, domain.ErrNotFound
}

func (m *mockTaskRepo) List() ([]*domain.Task, error) {
	if m.err != nil {
		return nil, m.err
	}
	if m.list != nil {
		return m.list, nil
	}
	return []*domain.Task{}, nil
}

func (m *mockTaskRepo) Update(t *domain.Task) error {
	m.updateTask = t
	if m.err != nil {
		return m.err
	}
	return nil
}

func (m *mockTaskRepo) Delete(id string) error {
	m.deleteID = id
	if m.err != nil {
		return m.err
	}
	return nil
}

// --- Mock ExampleRepository ---.

type mockExampleRepo struct {
	examples   []*domain.Example
	byTask     []*domain.Example
	err        error
	addBatch   []*domain.Example
	added      []*domain.Example
	updated    *domain.Example
	deletedID  string
	getExample *domain.Example
}

func (m *mockExampleRepo) Add(e *domain.Example) error {
	if m.err != nil {
		return m.err
	}
	m.added = append(m.added, e)
	return nil
}

func (m *mockExampleRepo) AddBatch(es []*domain.Example) error {
	if m.err != nil {
		return m.err
	}
	m.addBatch = append(m.addBatch, es...)
	return nil
}

func (m *mockExampleRepo) ListByTask(taskID string) ([]*domain.Example, error) {
	if m.err != nil {
		return nil, m.err
	}
	if m.byTask != nil {
		return m.byTask, nil
	}
	return m.examples, nil
}

func (m *mockExampleRepo) Get(taskID, id string) (*domain.Example, error) {
	if m.err != nil {
		return nil, m.err
	}
	if m.getExample != nil {
		return m.getExample, nil
	}
	for _, e := range m.examples {
		if e.ID == id {
			return e, nil
		}
	}
	return nil, domain.ErrNotFound
}

func (m *mockExampleRepo) Update(e *domain.Example) error {
	m.updated = e
	if m.err != nil {
		return m.err
	}
	return nil
}

func (m *mockExampleRepo) Delete(taskID, id string) error {
	m.deletedID = id
	if m.err != nil {
		return m.err
	}
	return nil
}

func (m *mockExampleRepo) DeleteByTask(taskID string) error {
	if m.err != nil {
		return m.err
	}
	return nil
}

// --- Mock TrainingJobRepository ---.

type mockTrainingRepo struct {
	jobs            []*domain.TrainingJob
	job             *domain.TrainingJob
	latestCompleted *domain.TrainingJob
	err             error
	createdJob      *domain.TrainingJob
	updatedJob      *domain.TrainingJob
}

func (m *mockTrainingRepo) Create(j *domain.TrainingJob) error {
	if m.err != nil {
		return m.err
	}
	m.createdJob = j
	m.jobs = append(m.jobs, j)
	return nil
}

func (m *mockTrainingRepo) Get(id string) (*domain.TrainingJob, error) {
	if m.err != nil {
		return nil, m.err
	}
	if m.job != nil {
		return m.job, nil
	}
	for _, j := range m.jobs {
		if j.ID == id {
			return j, nil
		}
	}
	return nil, domain.ErrNotFound
}

func (m *mockTrainingRepo) ListByTask(taskID string) ([]*domain.TrainingJob, error) {
	if m.err != nil {
		return nil, m.err
	}
	return m.jobs, nil
}

func (m *mockTrainingRepo) Update(j *domain.TrainingJob) error {
	m.updatedJob = j
	if m.err != nil {
		return m.err
	}
	return nil
}

func (m *mockTrainingRepo) LatestCompleted(taskID string) (*domain.TrainingJob, error) {
	if m.err != nil {
		return nil, m.err
	}
	if m.latestCompleted != nil {
		return m.latestCompleted, nil
	}
	return nil, domain.ErrNoModel
}

// --- Mock DeploymentRepository ---.

type mockDeploymentRepo struct {
	deployments   []*domain.Deployment
	active        *domain.Deployment
	err           error
	created       *domain.Deployment
	updated       *domain.Deployment
	getDeployment *domain.Deployment
}

func (m *mockDeploymentRepo) Create(d *domain.Deployment) error {
	if m.err != nil {
		return m.err
	}
	m.created = d
	m.deployments = append(m.deployments, d)
	return nil
}

func (m *mockDeploymentRepo) Get(id string) (*domain.Deployment, error) {
	if m.err != nil {
		return nil, m.err
	}
	if m.getDeployment != nil {
		return m.getDeployment, nil
	}
	for _, d := range m.deployments {
		if d.ID == id {
			return d, nil
		}
	}
	return nil, domain.ErrNotFound
}

func (m *mockDeploymentRepo) GetActiveForTask(taskID string) (*domain.Deployment, error) {
	if m.err != nil {
		return nil, m.err
	}
	if m.active != nil {
		return m.active, nil
	}
	return nil, domain.ErrNotFound
}

func (m *mockDeploymentRepo) ListByTask(taskID string) ([]*domain.Deployment, error) {
	if m.err != nil {
		return nil, m.err
	}
	return m.deployments, nil
}

func (m *mockDeploymentRepo) Update(d *domain.Deployment) error {
	m.updated = d
	if m.err != nil {
		return m.err
	}
	return nil
}

// --- Mock FeedbackRepository ---.

type mockFeedbackRepo struct {
	list        []*domain.Misprediction
	unresolved  []*domain.Misprediction
	err         error
	added       *domain.Misprediction
	resolvedIDs []string
}

func (m *mockFeedbackRepo) Add(f *domain.Misprediction) error {
	if m.err != nil {
		return m.err
	}
	m.added = f
	return nil
}

func (m *mockFeedbackRepo) ListByTask(taskID string) ([]*domain.Misprediction, error) {
	if m.err != nil {
		return nil, m.err
	}
	return m.list, nil
}

func (m *mockFeedbackRepo) ListUnresolved(taskID string) ([]*domain.Misprediction, error) {
	if m.err != nil {
		return nil, m.err
	}
	return m.unresolved, nil
}

func (m *mockFeedbackRepo) MarkResolved(ids []string) error {
	if m.err != nil {
		return m.err
	}
	m.resolvedIDs = ids
	return nil
}

// --- Mock InferenceEngine ---.

type mockInferenceEngine struct {
	output     string
	confidence float64
	predInput  string
	predCount  int
}

func (m *mockInferenceEngine) Predict(
	job *domain.TrainingJob,
	examples []*domain.Example,
	input string,
) (output string, confidence float64) {
	m.predInput = input
	m.predCount++
	return m.output, m.confidence
}

// --- Mock Exporter ---.

type mockExporter struct {
	bytes    []byte
	filename string
	err      error
	called   bool
}

func (m *mockExporter) BuildExport(
	task *domain.Task,
	job *domain.TrainingJob,
) (data []byte, filename string, err error) {
	m.called = true
	if m.err != nil {
		return nil, "", m.err
	}
	return m.bytes, m.filename, nil
}

// --- Mock ModelSelector ---.

type mockModelSelector struct {
	model domain.BaseModel
}

func (m *mockModelSelector) SelectBaseModel(
	task *domain.Task,
	exampleCount,
	avgInputLen,
	avgOutputLen int,
) domain.BaseModel {
	return m.model
}

// --- Mock FineTuner ---.

type mockFineTuner struct {
	started  bool
	job      *domain.TrainingJob
	examples []*domain.Example
	onUpdate func(progress int)
	onDone   func(metrics *domain.TrainingMetrics, err error)
}

func (m *mockFineTuner) Start(
	job *domain.TrainingJob,
	examples []*domain.Example,
	onUpdate func(progress int),
	onDone func(metrics *domain.TrainingMetrics, err error),
) {
	m.started = true
	m.job = job
	m.examples = examples
	m.onUpdate = onUpdate
	m.onDone = onDone
}

// --- Mock SyntheticGenerator ---.

type mockSynthGen struct {
	generated []*domain.Example
}

func (m *mockSynthGen) Generate(task *domain.Task, seed []*domain.Example, count int) []*domain.Example {
	return m.generated
}
