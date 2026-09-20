package usecase_test

import (
	"context"
	"errors"
	"testing"
	time "time"

	"distillery/internal/domain"
	"distillery/internal/usecase"
)

func newFinetuneUsecase(
	agent *mockCodingAgent,
	reqs *mockFTReqRepo,
	jobs *mockFTJobRepo,
	models *mockFTModelRepo,
	datasets *mockFTDatasetRepo,
) *usecase.FineTuneUsecase {
	return usecase.NewFineTuneUsecase(agent, reqs, jobs, models, datasets, newStubIDGen())
}

func sampleFineTuneRequest() *domain.FineTuneRequest {
	return &domain.FineTuneRequest{
		Name:      "Python Code Completion",
		Language:  domain.LangPython,
		Skill:     domain.SkillCodeCompletion,
		BaseModel: "gpt2-large",
		DatasetID: "ds_1",
		Owner:     "user_1",
		TrainingParams: domain.TrainingParameters{
			Epochs:       3,
			BatchSize:    8,
			LearningRate: 5e-5, WarmupSteps: 0, MaxSequenceLength: 0, GradientAccumSteps: 0, WeightDecay: 0, SchedulerType: "", OptimizerType: "", PreserveSyntax: false, ContextWindow: 0, BalancedSampling: false,
		}, ID: "", Description: "", ValidationParams: domain.ValidationParameters{}, Status: "", CreatedAt: time.Time{}, UpdatedAt: time.Time{},
	}
}

func TestFineTuneUsecase_CreateRequest_Valid(t *testing.T) {
	t.Parallel()

	agent := &mockCodingAgent{}
	reqs := &mockFTReqRepo{}
	uc := newFinetuneUsecase(agent, reqs, &mockFTJobRepo{}, &mockFTModelRepo{}, &mockFTDatasetRepo{})

	id, err := uc.CreateRequest(sampleFineTuneRequest())
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}

	if id == "" {
		t.Error("expected non-empty ID")
	}

	if reqs.createdReq == nil {
		t.Fatal("expected Create to be called")
	}

	if reqs.createdReq.Status != domain.RequestDraft {
		t.Errorf("expected status %q, got %q", domain.RequestDraft, reqs.createdReq.Status)
	}

	if reqs.createdReq.CreatedAt.IsZero() {
		t.Error("expected non-zero CreatedAt")
	}
}

func TestFineTuneUsecase_CreateRequest_Defaults(t *testing.T) {
	t.Parallel()

	reqs := &mockFTReqRepo{}
	uc := newFinetuneUsecase(&mockCodingAgent{}, reqs, &mockFTJobRepo{}, &mockFTModelRepo{}, &mockFTDatasetRepo{})

	req := sampleFineTuneRequest()
	req.BaseModel = ""
	req.Skill = ""

	_, err := uc.CreateRequest(req)
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}

	if reqs.createdReq.BaseModel != "gpt2-large" {
		t.Errorf("expected default base model 'gpt2-large', got %q", reqs.createdReq.BaseModel)
	}

	if reqs.createdReq.Skill != domain.SkillCodeCompletion {
		t.Errorf("expected default skill %q, got %q", domain.SkillCodeCompletion, reqs.createdReq.Skill)
	}
}

func TestFineTuneUsecase_CreateRequest_EmptyName(t *testing.T) {
	t.Parallel()

	uc := newFinetuneUsecase(&mockCodingAgent{}, &mockFTReqRepo{}, &mockFTJobRepo{}, &mockFTModelRepo{}, &mockFTDatasetRepo{})

	req := sampleFineTuneRequest()
	req.Name = ""

	_, err := uc.CreateRequest(req)
	if !errors.Is(err, domain.ErrInvalidInput) {
		t.Errorf("expected ErrInvalidInput, got %v", err)
	}
}

func TestFineTuneUsecase_CreateRequest_EmptyLanguage(t *testing.T) {
	t.Parallel()

	uc := newFinetuneUsecase(&mockCodingAgent{}, &mockFTReqRepo{}, &mockFTJobRepo{}, &mockFTModelRepo{}, &mockFTDatasetRepo{})

	req := sampleFineTuneRequest()
	req.Language = ""

	_, err := uc.CreateRequest(req)
	if !errors.Is(err, domain.ErrInvalidInput) {
		t.Errorf("expected ErrInvalidInput, got %v", err)
	}
}

func TestFineTuneUsecase_CreateRequest_EmptyDatasetID(t *testing.T) {
	t.Parallel()

	uc := newFinetuneUsecase(&mockCodingAgent{}, &mockFTReqRepo{}, &mockFTJobRepo{}, &mockFTModelRepo{}, &mockFTDatasetRepo{})

	req := sampleFineTuneRequest()
	req.DatasetID = ""

	_, err := uc.CreateRequest(req)
	if !errors.Is(err, domain.ErrInvalidInput) {
		t.Errorf("expected ErrInvalidInput, got %v", err)
	}
}

func TestFineTuneUsecase_CreateRequest_EmptyOwner(t *testing.T) {
	t.Parallel()

	uc := newFinetuneUsecase(&mockCodingAgent{}, &mockFTReqRepo{}, &mockFTJobRepo{}, &mockFTModelRepo{}, &mockFTDatasetRepo{})

	req := sampleFineTuneRequest()
	req.Owner = ""

	_, err := uc.CreateRequest(req)
	if !errors.Is(err, domain.ErrInvalidInput) {
		t.Errorf("expected ErrInvalidInput, got %v", err)
	}
}

func TestFineTuneUsecase_CreateRequest_RepoError(t *testing.T) {
	t.Parallel()

	reqs := &mockFTReqRepo{err: errRepoFailure}
	uc := newFinetuneUsecase(&mockCodingAgent{}, reqs, &mockFTJobRepo{}, &mockFTModelRepo{}, &mockFTDatasetRepo{})

	_, err := uc.CreateRequest(sampleFineTuneRequest())
	if !errors.Is(err, errRepoFailure) {
		t.Errorf("expected errRepoFailure, got %v", err)
	}
}

func TestFineTuneUsecase_StartTraining_Valid(t *testing.T) {
	t.Parallel()

	agent := &mockCodingAgent{startedJobID: "ftj_1"}
	reqs := &mockFTReqRepo{requests: []*domain.FineTuneRequest{{
		ID:     "ft_1",
		Name:   "Test",
		Status: domain.RequestDraft,
	}}}
	uc := newFinetuneUsecase(agent, reqs, &mockFTJobRepo{}, &mockFTModelRepo{}, &mockFTDatasetRepo{})

	jobID, err := uc.StartTraining(context.Background(), "ft_1")
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}

	if jobID != "ftj_1" {
		t.Errorf("expected job ID 'ftj_1', got %q", jobID)
	}

	if !agent.startCalled {
		t.Error("expected agent.StartTraining to be called")
	}

	if reqs.requests[0].Status != domain.RequestQueued {
		t.Errorf("expected status %q, got %q", domain.RequestQueued, reqs.requests[0].Status)
	}
}

func TestFineTuneUsecase_StartTraining_AlreadyRunning(t *testing.T) {
	t.Parallel()

	reqs := &mockFTReqRepo{requests: []*domain.FineTuneRequest{{
		ID:     "ft_1",
		Name:   "Test",
		Status: domain.RequestTraining,
	}}}
	uc := newFinetuneUsecase(&mockCodingAgent{}, reqs, &mockFTJobRepo{}, &mockFTModelRepo{}, &mockFTDatasetRepo{})

	_, err := uc.StartTraining(context.Background(), "ft_1")
	if !errors.Is(err, domain.ErrAlreadyRunning) {
		t.Errorf("expected ErrAlreadyRunning, got %v", err)
	}
}

func TestFineTuneUsecase_StartTraining_NotFound(t *testing.T) {
	t.Parallel()

	uc := newFinetuneUsecase(&mockCodingAgent{}, &mockFTReqRepo{}, &mockFTJobRepo{}, &mockFTModelRepo{}, &mockFTDatasetRepo{})

	_, err := uc.StartTraining(context.Background(), "missing")
	if !errors.Is(err, domain.ErrNotFound) {
		t.Errorf("expected ErrNotFound, got %v", err)
	}
}

func TestFineTuneUsecase_StartTraining_AgentError(t *testing.T) {
	t.Parallel()

	agent := &mockCodingAgent{startErr: errRepoFailure}
	reqs := &mockFTReqRepo{requests: []*domain.FineTuneRequest{{
		ID:     "ft_1",
		Name:   "Test",
		Status: domain.RequestDraft,
	}}}
	uc := newFinetuneUsecase(agent, reqs, &mockFTJobRepo{}, &mockFTModelRepo{}, &mockFTDatasetRepo{})

	_, err := uc.StartTraining(context.Background(), "ft_1")
	if !errors.Is(err, errRepoFailure) {
		t.Errorf("expected errRepoFailure, got %v", err)
	}
}

func TestFineTuneUsecase_JobStatus_Found(t *testing.T) {
	t.Parallel()

	jobs := &mockFTJobRepo{jobs: []*domain.FineTuneJob{{ID: "ftj_1", RequestID: "ft_1"}}}
	uc := newFinetuneUsecase(&mockCodingAgent{}, &mockFTReqRepo{}, jobs, &mockFTModelRepo{}, &mockFTDatasetRepo{})

	job, err := uc.JobStatus("ftj_1")
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}

	if job.ID != "ftj_1" {
		t.Errorf("expected job ID 'ftj_1', got %q", job.ID)
	}
}

func TestFineTuneUsecase_JobStatus_NotFound(t *testing.T) {
	t.Parallel()

	uc := newFinetuneUsecase(&mockCodingAgent{}, &mockFTReqRepo{}, &mockFTJobRepo{}, &mockFTModelRepo{}, &mockFTDatasetRepo{})

	_, err := uc.JobStatus("missing")
	if !errors.Is(err, domain.ErrNotFound) {
		t.Errorf("expected ErrNotFound, got %v", err)
	}
}

func TestFineTuneUsecase_MonitorTraining(t *testing.T) {
	t.Parallel()

	agent := &mockCodingAgent{monitorResult: domain.TrainingOptimization{
		CurrentStep:   100,
		CurrentLoss:   1.5,
		LossDirection: "decreasing",
		Suggestion:    "Keep going",
		Action:        domain.ActionContinue,
		Confidence:    90,
	}}
	uc := newFinetuneUsecase(agent, &mockFTReqRepo{}, &mockFTJobRepo{}, &mockFTModelRepo{}, &mockFTDatasetRepo{})

	opt, err := uc.MonitorTraining(context.Background(), "ftj_1")
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}

	if opt.Action != domain.ActionContinue {
		t.Errorf("expected action %q, got %q", domain.ActionContinue, opt.Action)
	}
}

func TestFineTuneUsecase_MonitorTraining_Error(t *testing.T) {
	t.Parallel()

	agent := &mockCodingAgent{monitorErr: errRepoFailure}
	uc := newFinetuneUsecase(agent, &mockFTReqRepo{}, &mockFTJobRepo{}, &mockFTModelRepo{}, &mockFTDatasetRepo{})

	_, err := uc.MonitorTraining(context.Background(), "ftj_1")
	if !errors.Is(err, errRepoFailure) {
		t.Errorf("expected errRepoFailure, got %v", err)
	}
}

func TestFineTuneUsecase_Insights(t *testing.T) {
	t.Parallel()

	agent := &mockCodingAgent{insights: domain.AgentInsights{
		Phase:   "training",
		Summary: "loss decreasing steadily", Metrics: nil, Warnings: nil, NextSteps: nil, EstimatedQuality: "",
	}}
	uc := newFinetuneUsecase(agent, &mockFTReqRepo{}, &mockFTJobRepo{}, &mockFTModelRepo{}, &mockFTDatasetRepo{})

	ins, err := uc.Insights(context.Background(), "ftj_1")
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}

	if ins.Phase != "training" {
		t.Errorf("expected phase 'training', got %q", ins.Phase)
	}
}

func TestFineTuneUsecase_QualityReport(t *testing.T) {
	t.Parallel()

	agent := &mockCodingAgent{qualityReport: domain.QualityReport{
		TrainingComplete: true,
		FinalLoss:        1.2,
		OverallQuality:   "good", BestMetrics: nil, CodeExecutability: 0, SyntaxValidity: 0, Issues: nil, Recommendations: nil,
	}}
	uc := newFinetuneUsecase(agent, &mockFTReqRepo{}, &mockFTJobRepo{}, &mockFTModelRepo{}, &mockFTDatasetRepo{})

	report, err := uc.QualityReport(context.Background(), "ftj_1")
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}

	if report.OverallQuality != "good" {
		t.Errorf("expected quality 'good', got %q", report.OverallQuality)
	}
}

func TestFineTuneUsecase_ListRequests(t *testing.T) {
	t.Parallel()

	reqs := &mockFTReqRepo{requests: []*domain.FineTuneRequest{
		{ID: "ft_1", Owner: "user_1"},
		{ID: "ft_2", Owner: "user_1"},
		{ID: "ft_3", Owner: "user_2"},
	}}
	uc := newFinetuneUsecase(&mockCodingAgent{}, reqs, &mockFTJobRepo{}, &mockFTModelRepo{}, &mockFTDatasetRepo{})

	list, err := uc.ListRequests("user_1")
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}

	if len(list) != 2 {
		t.Errorf("expected 2 requests, got %d", len(list))
	}
}

func TestFineTuneUsecase_ListJobs(t *testing.T) {
	t.Parallel()

	jobs := &mockFTJobRepo{jobs: []*domain.FineTuneJob{
		{ID: "ftj_1"},
		{ID: "ftj_2"},
	}}
	uc := newFinetuneUsecase(&mockCodingAgent{}, &mockFTReqRepo{}, jobs, &mockFTModelRepo{}, &mockFTDatasetRepo{})

	list, err := uc.ListJobs()
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}

	if len(list) != 2 {
		t.Errorf("expected 2 jobs, got %d", len(list))
	}
}

func TestFineTuneUsecase_ListModels_NoFilter(t *testing.T) {
	t.Parallel()

	models := &mockFTModelRepo{models: []*domain.TrainedModel{
		{ID: "tm_1", Language: domain.LangPython},
		{ID: "tm_2", Language: domain.LangGo},
	}}
	uc := newFinetuneUsecase(&mockCodingAgent{}, &mockFTReqRepo{}, &mockFTJobRepo{}, models, &mockFTDatasetRepo{})

	list, err := uc.ListModels("", "")
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}

	if len(list) != 2 {
		t.Errorf("expected 2 models, got %d", len(list))
	}
}

func TestFineTuneUsecase_ListModels_ByLanguage(t *testing.T) {
	t.Parallel()

	models := &mockFTModelRepo{models: []*domain.TrainedModel{
		{ID: "tm_1", Language: domain.LangPython},
		{ID: "tm_2", Language: domain.LangGo},
	}}
	uc := newFinetuneUsecase(&mockCodingAgent{}, &mockFTReqRepo{}, &mockFTJobRepo{}, models, &mockFTDatasetRepo{})

	list, err := uc.ListModels(string(domain.LangPython), "")
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}

	if len(list) != 1 {
		t.Errorf("expected 1 python model, got %d", len(list))
	}
}

func TestFineTuneUsecase_ListModels_BySkill(t *testing.T) {
	t.Parallel()

	models := &mockFTModelRepo{models: []*domain.TrainedModel{
		{ID: "tm_1", Skill: domain.SkillCodeCompletion},
		{ID: "tm_2", Skill: domain.SkillBugFixing},
	}}
	uc := newFinetuneUsecase(&mockCodingAgent{}, &mockFTReqRepo{}, &mockFTJobRepo{}, models, &mockFTDatasetRepo{})

	list, err := uc.ListModels("", string(domain.SkillCodeCompletion))
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}

	if len(list) != 1 {
		t.Errorf("expected 1 code_completion model, got %d", len(list))
	}
}

func TestFineTuneUsecase_RegisterDataset_Valid(t *testing.T) {
	t.Parallel()

	datasets := &mockFTDatasetRepo{}
	uc := newFinetuneUsecase(&mockCodingAgent{}, &mockFTReqRepo{}, &mockFTJobRepo{}, &mockFTModelRepo{}, datasets)

	d := &domain.DatasetInfo{
		Name:     "Python Corpus",
		Language: domain.LangPython,
		Owner:    "user_1", ID: "", FileCount: 0, TotalSize: 0, TotalLines: 0, TotalTokens: 0, SampleCount: 0, Quality: domain.DatasetQuality{OverallScore: 0, ValidityRate: 0, ComplexityScore: 0, Issues: nil}, Status: "", CreatedAt: time.Time{},
	}

	id, err := uc.RegisterDataset(d)
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}

	if id == "" {
		t.Error("expected non-empty ID")
	}

	if datasets.createdDataset == nil {
		t.Fatal("expected Create to be called")
	}

	if datasets.createdDataset.Status != "validated" {
		t.Errorf("expected status 'validated', got %q", datasets.createdDataset.Status)
	}
}

func TestFineTuneUsecase_RegisterDataset_UnsupportedLanguage(t *testing.T) {
	t.Parallel()

	datasets := &mockFTDatasetRepo{}
	uc := newFinetuneUsecase(&mockCodingAgent{}, &mockFTReqRepo{}, &mockFTJobRepo{}, &mockFTModelRepo{}, datasets)

	d := &domain.DatasetInfo{
		Name:     "Unknown Lang",
		Language: domain.ProgrammingLanguage("lisp"),
		Owner:    "user_1", ID: "", FileCount: 0, TotalSize: 0, TotalLines: 0, TotalTokens: 0, SampleCount: 0, Quality: domain.DatasetQuality{OverallScore: 0, ValidityRate: 0, ComplexityScore: 0, Issues: nil}, Status: "", CreatedAt: time.Time{},
	}

	id, err := uc.RegisterDataset(d)
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}

	if id == "" {
		t.Error("expected non-empty ID")
	}

	if datasets.createdDataset.Status != "invalid" {
		t.Errorf("expected status 'invalid', got %q", datasets.createdDataset.Status)
	}
}

func TestFineTuneUsecase_RegisterDataset_EmptyName(t *testing.T) {
	t.Parallel()

	uc := newFinetuneUsecase(&mockCodingAgent{}, &mockFTReqRepo{}, &mockFTJobRepo{}, &mockFTModelRepo{}, &mockFTDatasetRepo{})

	d := &domain.DatasetInfo{
		Name:     "",
		Language: domain.LangPython,
		Owner:    "user_1", ID: "", FileCount: 0, TotalSize: 0, TotalLines: 0, TotalTokens: 0, SampleCount: 0, Quality: domain.DatasetQuality{OverallScore: 0, ValidityRate: 0, ComplexityScore: 0, Issues: nil}, Status: "", CreatedAt: time.Time{},
	}

	_, err := uc.RegisterDataset(d)
	if !errors.Is(err, domain.ErrInvalidInput) {
		t.Errorf("expected ErrInvalidInput, got %v", err)
	}
}

func TestFineTuneUsecase_RegisterDataset_EmptyLanguage(t *testing.T) {
	t.Parallel()

	uc := newFinetuneUsecase(&mockCodingAgent{}, &mockFTReqRepo{}, &mockFTJobRepo{}, &mockFTModelRepo{}, &mockFTDatasetRepo{})

	d := &domain.DatasetInfo{
		Name:     "Test",
		Language: "",
		Owner:    "user_1", ID: "", FileCount: 0, TotalSize: 0, TotalLines: 0, TotalTokens: 0, SampleCount: 0, Quality: domain.DatasetQuality{OverallScore: 0, ValidityRate: 0, ComplexityScore: 0, Issues: nil}, Status: "", CreatedAt: time.Time{},
	}

	_, err := uc.RegisterDataset(d)
	if !errors.Is(err, domain.ErrInvalidInput) {
		t.Errorf("expected ErrInvalidInput, got %v", err)
	}
}

func TestFineTuneUsecase_RegisterDataset_EmptyOwner(t *testing.T) {
	t.Parallel()

	uc := newFinetuneUsecase(&mockCodingAgent{}, &mockFTReqRepo{}, &mockFTJobRepo{}, &mockFTModelRepo{}, &mockFTDatasetRepo{})

	d := &domain.DatasetInfo{
		Name:     "Test",
		Language: domain.LangPython,
		Owner:    "", ID: "", FileCount: 0, TotalSize: 0, TotalLines: 0, TotalTokens: 0, SampleCount: 0, Quality: domain.DatasetQuality{OverallScore: 0, ValidityRate: 0, ComplexityScore: 0, Issues: nil}, Status: "", CreatedAt: time.Time{},
	}

	_, err := uc.RegisterDataset(d)
	if !errors.Is(err, domain.ErrInvalidInput) {
		t.Errorf("expected ErrInvalidInput, got %v", err)
	}
}

func TestFineTuneUsecase_ListDatasets(t *testing.T) {
	t.Parallel()

	datasets := &mockFTDatasetRepo{datasets: []*domain.DatasetInfo{
		{ID: "ds_1", Owner: "user_1"},
		{ID: "ds_2", Owner: "user_1"},
		{ID: "ds_3", Owner: "user_2"},
	}}
	uc := newFinetuneUsecase(&mockCodingAgent{}, &mockFTReqRepo{}, &mockFTJobRepo{}, &mockFTModelRepo{}, datasets)

	list, err := uc.ListDatasets("user_1")
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}

	if len(list) != 2 {
		t.Errorf("expected 2 datasets, got %d", len(list))
	}
}

func TestFineTuneUsecase_GetDataset_Found(t *testing.T) {
	t.Parallel()

	datasets := &mockFTDatasetRepo{datasets: []*domain.DatasetInfo{{ID: "ds_1"}}}
	uc := newFinetuneUsecase(&mockCodingAgent{}, &mockFTReqRepo{}, &mockFTJobRepo{}, &mockFTModelRepo{}, datasets)

	ds, err := uc.GetDataset("ds_1")
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}

	if ds.ID != "ds_1" {
		t.Errorf("expected ID 'ds_1', got %q", ds.ID)
	}
}

func TestFineTuneUsecase_GetDataset_NotFound(t *testing.T) {
	t.Parallel()

	uc := newFinetuneUsecase(&mockCodingAgent{}, &mockFTReqRepo{}, &mockFTJobRepo{}, &mockFTModelRepo{}, &mockFTDatasetRepo{})

	_, err := uc.GetDataset("missing")
	if !errors.Is(err, domain.ErrNotFound) {
		t.Errorf("expected ErrNotFound, got %v", err)
	}
}

func TestFineTuneUsecase_AnalyseDataset(t *testing.T) {
	t.Parallel()

	agent := &mockCodingAgent{analysis: domain.DatasetAnalysis{
		FileCount:        500,
		QualityScore:     85.0,
		ReadyForTraining: true, TotalSize: 0, TotalTokens: 0, LanguageCoverage: nil, ComplexityRange: [2]int{}, AverageComplexity: 0, SyntaxValidityRate: 0, Recommendations: nil,
	}}
	uc := newFinetuneUsecase(agent, &mockFTReqRepo{}, &mockFTJobRepo{}, &mockFTModelRepo{}, &mockFTDatasetRepo{})

	analysis, err := uc.AnalyseDataset(context.Background(), "ds_1")
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}

	if !agent.analyzeCalled {
		t.Error("expected AnalyzeDataset to be called")
	}

	if !analysis.ReadyForTraining {
		t.Error("expected ready for training")
	}
}

func TestFineTuneUsecase_RecommendHyperparameters(t *testing.T) {
	t.Parallel()

	agent := &mockCodingAgent{recommendedParams: domain.TrainingParameters{
		Epochs:        3,
		BatchSize:     8,
		LearningRate:  3e-5,
		WarmupSteps:   100,
		WeightDecay:   0.01,
		SchedulerType: "cosine", MaxSequenceLength: 0, GradientAccumSteps: 0, OptimizerType: "", PreserveSyntax: false, ContextWindow: 0, BalancedSampling: false,
	}}
	uc := newFinetuneUsecase(agent, &mockFTReqRepo{}, &mockFTJobRepo{}, &mockFTModelRepo{}, &mockFTDatasetRepo{})

	params, err := uc.RecommendHyperparameters(context.Background(), domain.HyperparameterRecommendationReq{
		Language:    domain.LangPython,
		Skill:       domain.SkillCodeCompletion,
		DatasetSize: 100000, AvailableGPU: 0,
	})
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}

	if !agent.recommendCalled {
		t.Error("expected RecommendHyperparameters to be called")
	}

	if params.Epochs != 3 {
		t.Errorf("expected 3 epochs, got %d", params.Epochs)
	}
}

func TestFineTuneUsecase_GetModel_Found(t *testing.T) {
	t.Parallel()

	models := &mockFTModelRepo{models: []*domain.TrainedModel{{ID: "tm_1"}}}
	uc := newFinetuneUsecase(&mockCodingAgent{}, &mockFTReqRepo{}, &mockFTJobRepo{}, models, &mockFTDatasetRepo{})

	model, err := uc.GetModel("tm_1")
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}

	if model.ID != "tm_1" {
		t.Errorf("expected ID 'tm_1', got %q", model.ID)
	}
}

func TestFineTuneUsecase_GetModel_NotFound(t *testing.T) {
	t.Parallel()

	uc := newFinetuneUsecase(&mockCodingAgent{}, &mockFTReqRepo{}, &mockFTJobRepo{}, &mockFTModelRepo{}, &mockFTDatasetRepo{})

	_, err := uc.GetModel("missing")
	if !errors.Is(err, domain.ErrNotFound) {
		t.Errorf("expected ErrNotFound, got %v", err)
	}
}

func TestFineTuneUsecase_ExportModel_Valid(t *testing.T) {
	t.Parallel()

	models := &mockFTModelRepo{models: []*domain.TrainedModel{{
		ID:    "tm_1",
		Name:  "Python Model",
		Owner: "user_1",
	}}}
	uc := newFinetuneUsecase(&mockCodingAgent{}, &mockFTReqRepo{}, &mockFTJobRepo{}, models, &mockFTDatasetRepo{})

	url, err := uc.ExportModel("tm_1")
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}

	if url != "https://huggingface.co/user_1/Python Model" {
		t.Errorf("expected HF URL, got %q", url)
	}

	if models.updatedModel == nil {
		t.Fatal("expected Update to be called")
	}

	if models.updatedModel.Status != "ready" {
		t.Errorf("expected status 'ready', got %q", models.updatedModel.Status)
	}
}

func TestFineTuneUsecase_ExportModel_NotFound(t *testing.T) {
	t.Parallel()

	uc := newFinetuneUsecase(&mockCodingAgent{}, &mockFTReqRepo{}, &mockFTJobRepo{}, &mockFTModelRepo{}, &mockFTDatasetRepo{})

	_, err := uc.ExportModel("missing")
	if !errors.Is(err, domain.ErrNotFound) {
		t.Errorf("expected ErrNotFound, got %v", err)
	}
}
