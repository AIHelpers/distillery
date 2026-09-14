package usecase_test

import (
	"context"

	"distillery/internal/domain"
)

// --- Mock FineTuneRequestRepository ---.

type mockFTReqRepo struct {
	requests   []*domain.FineTuneRequest
	err        error
	createdReq *domain.FineTuneRequest
	updatedReq *domain.FineTuneRequest
	deletedID  string
	getRequest *domain.FineTuneRequest
}

func (m *mockFTReqRepo) Create(r *domain.FineTuneRequest) error {
	if m.err != nil {
		return m.err
	}

	m.createdReq = r
	m.requests = append(m.requests, r)

	return nil
}

func (m *mockFTReqRepo) Get(id string) (*domain.FineTuneRequest, error) {
	if m.err != nil {
		return nil, m.err
	}

	if m.getRequest != nil {
		return m.getRequest, nil
	}

	for _, r := range m.requests {
		if r.ID == id {
			return r, nil
		}
	}

	return nil, domain.ErrNotFound
}

func (m *mockFTReqRepo) ListByOwner(owner string) ([]*domain.FineTuneRequest, error) {
	if m.err != nil {
		return nil, m.err
	}

	var out []*domain.FineTuneRequest

	for _, r := range m.requests {
		if r.Owner == owner {
			out = append(out, r)
		}
	}

	return out, nil
}

func (m *mockFTReqRepo) List() ([]*domain.FineTuneRequest, error) {
	if m.err != nil {
		return nil, m.err
	}

	return m.requests, nil
}

func (m *mockFTReqRepo) Update(r *domain.FineTuneRequest) error {
	if m.err != nil {
		return m.err
	}

	m.updatedReq = r
	for i, ex := range m.requests {
		if ex.ID == r.ID {
			m.requests[i] = r
			return nil
		}
	}

	return domain.ErrNotFound
}

func (m *mockFTReqRepo) Delete(id string) error {
	if m.err != nil {
		return m.err
	}

	m.deletedID = id
	for i, r := range m.requests {
		if r.ID == id {
			m.requests = append(m.requests[:i], m.requests[i+1:]...)
			return nil
		}
	}

	return domain.ErrNotFound
}

// --- Mock FineTuneJobRepository ---.

type mockFTJobRepo struct {
	jobs        []*domain.FineTuneJob
	err         error
	createdJob  *domain.FineTuneJob
	updatedJob  *domain.FineTuneJob
	jobStatus   domain.JobStatus
	progress    float64
	epoch, step int
}

func (m *mockFTJobRepo) Create(job *domain.FineTuneJob) error {
	if m.err != nil {
		return m.err
	}

	m.createdJob = job
	m.jobs = append(m.jobs, job)

	return nil
}

func (m *mockFTJobRepo) Get(id string) (*domain.FineTuneJob, error) {
	if m.err != nil {
		return nil, m.err
	}

	for _, j := range m.jobs {
		if j.ID == id {
			return j, nil
		}
	}

	return nil, domain.ErrNotFound
}

func (m *mockFTJobRepo) GetByRequestID(requestID string) (*domain.FineTuneJob, error) {
	if m.err != nil {
		return nil, m.err
	}

	for _, j := range m.jobs {
		if j.RequestID == requestID {
			return j, nil
		}
	}

	return nil, domain.ErrNotFound
}

func (m *mockFTJobRepo) List() ([]*domain.FineTuneJob, error) {
	if m.err != nil {
		return nil, m.err
	}

	return m.jobs, nil
}

func (m *mockFTJobRepo) ListActive() ([]*domain.FineTuneJob, error) {
	if m.err != nil {
		return nil, m.err
	}

	var out []*domain.FineTuneJob

	for _, j := range m.jobs {
		if j.Status == domain.JobQueued || j.Status == domain.JobPreparing || j.Status == domain.JobRunning {
			out = append(out, j)
		}
	}

	return out, nil
}

func (m *mockFTJobRepo) Update(job *domain.FineTuneJob) error {
	if m.err != nil {
		return m.err
	}

	m.updatedJob = job
	for i, ex := range m.jobs {
		if ex.ID == job.ID {
			m.jobs[i] = job
			return nil
		}
	}

	return domain.ErrNotFound
}

func (m *mockFTJobRepo) UpdateStatus(_ string, status domain.JobStatus) error {
	if m.err != nil {
		return m.err
	}

	m.jobStatus = status

	return nil
}

func (m *mockFTJobRepo) UpdateProgress(_ string, progress float64, epoch, step int) error {
	if m.err != nil {
		return m.err
	}

	m.progress = progress
	m.epoch = epoch
	m.step = step

	return nil
}

func (m *mockFTJobRepo) AddMetric(_, _ string, _ domain.MetricPoint) error {
	if m.err != nil {
		return m.err
	}

	return nil
}

func (m *mockFTJobRepo) SetFinalMetrics(_ string, _ map[string]float64) error {
	if m.err != nil {
		return m.err
	}

	return nil
}

func (m *mockFTJobRepo) SetError(_, _ string) error {
	if m.err != nil {
		return m.err
	}

	return nil
}

// --- Mock TrainedModelRepository ---.

type mockFTModelRepo struct {
	models       []*domain.TrainedModel
	err          error
	createdModel *domain.TrainedModel
	updatedModel *domain.TrainedModel
}

func (m *mockFTModelRepo) Create(model *domain.TrainedModel) error {
	if m.err != nil {
		return m.err
	}

	m.createdModel = model
	m.models = append(m.models, model)

	return nil
}

func (m *mockFTModelRepo) Get(id string) (*domain.TrainedModel, error) {
	if m.err != nil {
		return nil, m.err
	}

	for _, model := range m.models {
		if model.ID == id {
			return model, nil
		}
	}

	return nil, domain.ErrNotFound
}

func (m *mockFTModelRepo) GetByJobID(jobID string) (*domain.TrainedModel, error) {
	if m.err != nil {
		return nil, m.err
	}

	for _, model := range m.models {
		if model.FineTuneJobID == jobID {
			return model, nil
		}
	}

	return nil, domain.ErrNotFound
}

func (m *mockFTModelRepo) List() ([]*domain.TrainedModel, error) {
	if m.err != nil {
		return nil, m.err
	}

	return m.models, nil
}

func (m *mockFTModelRepo) ListByLanguage(lang domain.ProgrammingLanguage) ([]*domain.TrainedModel, error) {
	if m.err != nil {
		return nil, m.err
	}

	var out []*domain.TrainedModel

	for _, model := range m.models {
		if model.Language == lang {
			out = append(out, model)
		}
	}

	return out, nil
}

func (m *mockFTModelRepo) ListBySkill(skill domain.SkillCategory) ([]*domain.TrainedModel, error) {
	if m.err != nil {
		return nil, m.err
	}

	var out []*domain.TrainedModel

	for _, model := range m.models {
		if model.Skill == skill {
			out = append(out, model)
		}
	}

	return out, nil
}

func (m *mockFTModelRepo) Update(model *domain.TrainedModel) error {
	if m.err != nil {
		return m.err
	}

	m.updatedModel = model
	for i, ex := range m.models {
		if ex.ID == model.ID {
			m.models[i] = model
			return nil
		}
	}

	return domain.ErrNotFound
}

// --- Mock DatasetRepository ---.

type mockFTDatasetRepo struct {
	datasets       []*domain.DatasetInfo
	err            error
	createdDataset *domain.DatasetInfo
	updatedQuality domain.DatasetQuality
	deletedID      string
}

func (m *mockFTDatasetRepo) Create(d *domain.DatasetInfo) error {
	if m.err != nil {
		return m.err
	}

	m.createdDataset = d
	m.datasets = append(m.datasets, d)

	return nil
}

func (m *mockFTDatasetRepo) Get(id string) (*domain.DatasetInfo, error) {
	if m.err != nil {
		return nil, m.err
	}

	for _, d := range m.datasets {
		if d.ID == id {
			return d, nil
		}
	}

	return nil, domain.ErrNotFound
}

func (m *mockFTDatasetRepo) ListByOwner(owner string) ([]*domain.DatasetInfo, error) {
	if m.err != nil {
		return nil, m.err
	}

	var out []*domain.DatasetInfo

	for _, d := range m.datasets {
		if d.Owner == owner {
			out = append(out, d)
		}
	}

	return out, nil
}

func (m *mockFTDatasetRepo) List() ([]*domain.DatasetInfo, error) {
	if m.err != nil {
		return nil, m.err
	}

	return m.datasets, nil
}

func (m *mockFTDatasetRepo) UpdateQuality(datasetID string, q domain.DatasetQuality) error {
	if m.err != nil {
		return m.err
	}

	m.updatedQuality = q
	for _, d := range m.datasets {
		if d.ID == datasetID {
			d.Quality = q
			return nil
		}
	}

	return domain.ErrNotFound
}

func (m *mockFTDatasetRepo) Delete(id string) error {
	if m.err != nil {
		return m.err
	}

	m.deletedID = id
	for i, d := range m.datasets {
		if d.ID == id {
			m.datasets = append(m.datasets[:i], m.datasets[i+1:]...)
			return nil
		}
	}

	return domain.ErrNotFound
}

// --- Mock CodingAIAgent ---.

type mockCodingAgent struct {
	analysis          domain.DatasetAnalysis
	analysisErr       error
	recommendedParams domain.TrainingParameters
	recommendErr      error
	startedJobID      string
	startErr          error
	monitorResult     domain.TrainingOptimization
	monitorErr        error
	qualityReport     domain.QualityReport
	qualityErr        error
	insights          domain.AgentInsights
	insightsErr       error
	analyzeCalled     bool
	recommendCalled   bool
	startCalled       bool
}

func (m *mockCodingAgent) AnalyzeDataset(_ context.Context, _ string) (domain.DatasetAnalysis, error) {
	m.analyzeCalled = true
	if m.analysisErr != nil {
		return domain.DatasetAnalysis{FileCount: 0, TotalSize: 0, TotalTokens: 0, LanguageCoverage: nil, ComplexityRange: [2]int{}, AverageComplexity: 0, SyntaxValidityRate: 0, Recommendations: nil, ReadyForTraining: false, QualityScore: 0}, m.analysisErr
	}

	return m.analysis, nil
}

func (m *mockCodingAgent) RecommendHyperparameters(_ context.Context, _ domain.HyperparameterRecommendationReq) (domain.TrainingParameters, error) {
	m.recommendCalled = true
	if m.recommendErr != nil {
		return domain.TrainingParameters{Epochs: 0, BatchSize: 0, LearningRate: 0, WarmupSteps: 0, MaxSequenceLength: 0, GradientAccumSteps: 0, WeightDecay: 0, SchedulerType: "", OptimizerType: "", PreserveSyntax: false, ContextWindow: 0, BalancedSampling: false}, m.recommendErr
	}

	return m.recommendedParams, nil
}

func (m *mockCodingAgent) StartTraining(_ context.Context, _ *domain.FineTuneRequest) (string, error) {
	m.startCalled = true
	if m.startErr != nil {
		return "", m.startErr
	}

	return m.startedJobID, nil
}

func (m *mockCodingAgent) MonitorTraining(_ context.Context, _ string) (domain.TrainingOptimization, error) {
	if m.monitorErr != nil {
		return domain.TrainingOptimization{CurrentStep: 0, CurrentLoss: 0, LossDirection: "", Suggestion: "", Action: "", Confidence: 0}, m.monitorErr
	}

	return m.monitorResult, nil
}

func (m *mockCodingAgent) ValidateTrainingQuality(_ context.Context, _ string) (domain.QualityReport, error) {
	if m.qualityErr != nil {
		return domain.QualityReport{TrainingComplete: false, FinalLoss: 0, BestMetrics: nil, CodeExecutability: 0, SyntaxValidity: 0, OverallQuality: "", Issues: nil, Recommendations: nil}, m.qualityErr
	}

	return m.qualityReport, nil
}

func (m *mockCodingAgent) GetInsights(_ context.Context, _ string) (domain.AgentInsights, error) {
	if m.insightsErr != nil {
		return domain.AgentInsights{Phase: "", Summary: "", Metrics: nil, Warnings: nil, NextSteps: nil, EstimatedQuality: ""}, m.insightsErr
	}

	return m.insights, nil
}
