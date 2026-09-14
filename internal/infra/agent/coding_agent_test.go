package agent_test

import (
	"context"
	"testing"

	"distillery/internal/domain"
	infraagent "distillery/internal/infra/agent"
	"distillery/internal/repository/memory"
)

func newCodingAgentTestRepos() (*memory.FineTuneRequestRepo, *memory.FineTuneJobRepo, *memory.DatasetRepo, *memory.TrainedModelRepo) {
	s := memory.NewStore("")
	return memory.NewFineTuneRequestRepo(s), memory.NewFineTuneJobRepo(s), memory.NewDatasetRepo(s), memory.NewTrainedModelRepo(s)
}

func newTestCodingAgent(
	reqRepo *memory.FineTuneRequestRepo,
	jobRepo *memory.FineTuneJobRepo,
	dsRepo *memory.DatasetRepo,
	modelRepo *memory.TrainedModelRepo,
) *infraagent.CodingAgentImpl {
	return infraagent.NewCodingAgent(reqRepo, jobRepo, dsRepo, modelRepo).(*infraagent.CodingAgentImpl)
}

func sampleAnalysisDataset() *domain.DatasetInfo {
	return &domain.DatasetInfo{
		ID:          "ds_1",
		Name:        "Python Code",
		Language:    domain.LangPython,
		FileCount:   1000,
		TotalSize:   5000000,
		TotalTokens: 200000,
		Status:      "validated",
		Quality: domain.DatasetQuality{
			OverallScore:    85.0,
			ValidityRate:    0.95,
			ComplexityScore: 5.5,
		},
	}
}

func TestCodingAgent_AnalyzeDataset_ReadyForTraining(t *testing.T) {
	reqs, jobs, dsRepo, models := newCodingAgentTestRepos()
	agent := newTestCodingAgent(reqs, jobs, dsRepo, models)
	_ = dsRepo.Create(sampleAnalysisDataset())

	analysis, err := agent.AnalyzeDataset(context.Background(), "ds_1")
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}
	if !analysis.ReadyForTraining {
		t.Error("expected ready for training")
	}
	if analysis.FileCount != 1000 {
		t.Errorf("expected file count 1000, got %d", analysis.FileCount)
	}
	if analysis.QualityScore != 85.0 {
		t.Errorf("expected quality score 85.0, got %f", analysis.QualityScore)
	}
	if analysis.SyntaxValidityRate != 0.95 {
		t.Errorf("expected validity 0.95, got %f", analysis.SyntaxValidityRate)
	}
}

func TestCodingAgent_AnalyzeDataset_LowQuality(t *testing.T) {
	reqs, jobs, dsRepo, models := newCodingAgentTestRepos()
	agent := newTestCodingAgent(reqs, jobs, dsRepo, models)
	ds := sampleAnalysisDataset()
	ds.Quality.OverallScore = 40.0
	_ = dsRepo.Create(ds)

	analysis, err := agent.AnalyzeDataset(context.Background(), "ds_1")
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}
	if analysis.ReadyForTraining {
		t.Error("expected not ready for training")
	}
	if len(analysis.Recommendations) == 0 {
		t.Error("expected recommendations for low quality")
	}
}

func TestCodingAgent_AnalyzeDataset_SmallDataset(t *testing.T) {
	reqs, jobs, dsRepo, models := newCodingAgentTestRepos()
	agent := newTestCodingAgent(reqs, jobs, dsRepo, models)
	ds := sampleAnalysisDataset()
	ds.TotalTokens = 50000
	_ = dsRepo.Create(ds)

	analysis, err := agent.AnalyzeDataset(context.Background(), "ds_1")
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}
	foundSmallRecommendation := false
	for _, rec := range analysis.Recommendations {
		if len(rec) > 0 && rec[0] == 'D' { // "Dataset is small..."
			foundSmallRecommendation = true
		}
	}
	if !foundSmallRecommendation {
		t.Error("expected recommendation about small dataset")
	}
}

func TestCodingAgent_AnalyzeDataset_LowValidity(t *testing.T) {
	reqs, jobs, dsRepo, models := newCodingAgentTestRepos()
	agent := newTestCodingAgent(reqs, jobs, dsRepo, models)
	ds := sampleAnalysisDataset()
	ds.Quality.ValidityRate = 0.8
	_ = dsRepo.Create(ds)

	analysis, err := agent.AnalyzeDataset(context.Background(), "ds_1")
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}
	foundValidityRecommendation := false
	for _, rec := range analysis.Recommendations {
		if len(rec) > 0 && rec[0] == 'C' { // "Code validity rate..."
			foundValidityRecommendation = true
		}
	}
	if !foundValidityRecommendation {
		t.Error("expected recommendation about low validity")
	}
}

func TestCodingAgent_AnalyzeDataset_NotFound(t *testing.T) {
	reqs, jobs, dsRepo, models := newCodingAgentTestRepos()
	agent := newTestCodingAgent(reqs, jobs, dsRepo, models)

	_, err := agent.AnalyzeDataset(context.Background(), "missing")
	if err != domain.ErrNotFound {
		t.Errorf("expected ErrNotFound, got %v", err)
	}
}

func TestCodingAgent_RecommendHyperparameters_Python(t *testing.T) {
	agent := newTestCodingAgent(nil, nil, nil, nil)

	params, err := agent.RecommendHyperparameters(context.Background(), domain.HyperparameterRecommendationReq{
		Language:     domain.LangPython,
		Skill:        domain.SkillCodeCompletion,
		DatasetSize:  1000000,
		AvailableGPU: 1,
	})
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}
	if params.LearningRate != 5e-5 {
		t.Errorf("expected learning rate 5e-5, got %f", params.LearningRate)
	}
	if params.ContextWindow != 256 {
		t.Errorf("expected context window 256, got %d", params.ContextWindow)
	}
	if !params.PreserveSyntax {
		t.Error("expected preserve syntax")
	}
}

func TestCodingAgent_RecommendHyperparameters_Go(t *testing.T) {
	agent := newTestCodingAgent(nil, nil, nil, nil)

	params, err := agent.RecommendHyperparameters(context.Background(), domain.HyperparameterRecommendationReq{
		Language:     domain.LangGo,
		Skill:        domain.SkillCodeCompletion,
		DatasetSize:  1000000,
		AvailableGPU: 1,
	})
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}
	if params.LearningRate != 2e-5 {
		t.Errorf("expected learning rate 2e-5, got %f", params.LearningRate)
	}
	if params.ContextWindow != 200 {
		t.Errorf("expected context window 200, got %d", params.ContextWindow)
	}
}

func TestCodingAgent_RecommendHyperparameters_JavaScript(t *testing.T) {
	agent := newTestCodingAgent(nil, nil, nil, nil)

	params, err := agent.RecommendHyperparameters(context.Background(), domain.HyperparameterRecommendationReq{
		Language:     domain.LangJavaScript,
		Skill:        domain.SkillCodeCompletion,
		DatasetSize:  1000000,
		AvailableGPU: 1,
	})
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}
	if params.ContextWindow != 300 {
		t.Errorf("expected context window 300, got %d", params.ContextWindow)
	}
}

func TestCodingAgent_RecommendHyperparameters_SmallDataset(t *testing.T) {
	agent := newTestCodingAgent(nil, nil, nil, nil)

	params, err := agent.RecommendHyperparameters(context.Background(), domain.HyperparameterRecommendationReq{
		Language:     domain.LangPython,
		Skill:        domain.SkillCodeCompletion,
		DatasetSize:  100000,
		AvailableGPU: 1,
	})
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}
	if params.Epochs != 5 {
		t.Errorf("expected 5 epochs for small dataset, got %d", params.Epochs)
	}
}

func TestCodingAgent_RecommendHyperparameters_LargeDataset(t *testing.T) {
	agent := newTestCodingAgent(nil, nil, nil, nil)

	params, err := agent.RecommendHyperparameters(context.Background(), domain.HyperparameterRecommendationReq{
		Language:     domain.LangPython,
		Skill:        domain.SkillCodeCompletion,
		DatasetSize:  10000000,
		AvailableGPU: 1,
	})
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}
	if params.Epochs != 2 {
		t.Errorf("expected 2 epochs for large dataset, got %d", params.Epochs)
	}
	if params.BatchSize != 16 {
		t.Errorf("expected batch size 16 for large dataset, got %d", params.BatchSize)
	}
}

func TestCodingAgent_RecommendHyperparameters_MultiGPU(t *testing.T) {
	agent := newTestCodingAgent(nil, nil, nil, nil)

	params, err := agent.RecommendHyperparameters(context.Background(), domain.HyperparameterRecommendationReq{
		Language:     domain.LangPython,
		Skill:        domain.SkillCodeCompletion,
		DatasetSize:  1000000,
		AvailableGPU: 4,
	})
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}
	if params.GradientAccumSteps != 2 {
		t.Errorf("expected 2 grad accum steps for multi-GPU, got %d", params.GradientAccumSteps)
	}
}

func TestCodingAgent_StartTraining_Valid(t *testing.T) {
	reqs, jobs, dsRepo, models := newCodingAgentTestRepos()
	agent := newTestCodingAgent(reqs, jobs, dsRepo, models)
	_ = dsRepo.Create(sampleAnalysisDataset())

	req := &domain.FineTuneRequest{
		ID:        "ft_1",
		Name:      "Test",
		Language:  domain.LangPython,
		Skill:     domain.SkillCodeCompletion,
		BaseModel: "gpt2-large",
		DatasetID: "ds_1",
		Owner:     "user_1",
		TrainingParams: domain.TrainingParameters{
			Epochs: 3,
		},
	}
	jobID, err := agent.StartTraining(context.Background(), req)
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}
	if jobID != "" {
		t.Errorf("agent returns empty job ID (usecase generates it), got %q", jobID)
	}
	list, _ := jobs.List()
	if len(list) != 1 {
		t.Errorf("expected 1 job created, got %d", len(list))
	}
	if list[0].RequestID != "ft_1" {
		t.Errorf("expected request ID 'ft_1', got %q", list[0].RequestID)
	}
	if list[0].Status != domain.JobQueued {
		t.Errorf("expected status %q, got %q", domain.JobQueued, list[0].Status)
	}
}

func TestCodingAgent_StartTraining_EmptyEpochs(t *testing.T) {
	reqs, jobs, dsRepo, models := newCodingAgentTestRepos()
	agent := newTestCodingAgent(reqs, jobs, dsRepo, models)
	_ = dsRepo.Create(sampleAnalysisDataset())

	req := &domain.FineTuneRequest{
		ID:        "ft_1",
		Name:      "Test",
		Language:  domain.LangPython,
		Skill:     domain.SkillCodeCompletion,
		BaseModel: "gpt2-large",
		DatasetID: "ds_1",
		Owner:     "user_1",
	}
	jobID, err := agent.StartTraining(context.Background(), req)
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}
	if jobID != "" {
		t.Errorf("agent returns empty job ID (usecase generates it), got %q", jobID)
	}
	if req.TrainingParams.Epochs == 0 {
		t.Error("expected recommended epochs to be set")
	}
}

func TestCodingAgent_StartTraining_NotReady(t *testing.T) {
	reqs, jobs, dsRepo, models := newCodingAgentTestRepos()
	agent := newTestCodingAgent(reqs, jobs, dsRepo, models)
	ds := sampleAnalysisDataset()
	ds.Quality.OverallScore = 30.0
	_ = dsRepo.Create(ds)

	req := &domain.FineTuneRequest{
		ID:        "ft_1",
		Name:      "Test",
		Language:  domain.LangPython,
		DatasetID: "ds_1",
		Owner:     "user_1",
	}
	_, err := agent.StartTraining(context.Background(), req)
	if err == nil {
		t.Fatal("expected error for not-ready dataset")
	}
}

func TestCodingAgent_StartTraining_DatasetNotFound(t *testing.T) {
	reqs, jobs, dsRepo, models := newCodingAgentTestRepos()
	agent := newTestCodingAgent(reqs, jobs, dsRepo, models)

	req := &domain.FineTuneRequest{
		ID:        "ft_1",
		Name:      "Test",
		Language:  domain.LangPython,
		DatasetID: "missing",
		Owner:     "user_1",
	}
	_, err := agent.StartTraining(context.Background(), req)
	if err != domain.ErrNotFound {
		t.Errorf("expected ErrNotFound, got %v", err)
	}
}

func TestCodingAgent_MonitorTraining_NoLossData(t *testing.T) {
	reqs, jobs, dsRepo, models := newCodingAgentTestRepos()
	agent := newTestCodingAgent(reqs, jobs, dsRepo, models)
	job := &domain.FineTuneJob{ID: "ftj_1", RequestID: "ft_1", Status: domain.JobRunning}
	_ = jobs.Create(job)

	opt, err := agent.MonitorTraining(context.Background(), "ftj_1")
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}
	if opt.Action != domain.ActionContinue {
		t.Errorf("expected action %q, got %q", domain.ActionContinue, opt.Action)
	}
	if opt.Suggestion == "" {
		t.Error("expected suggestion")
	}
}

func TestCodingAgent_MonitorTraining_LossDecreasing(t *testing.T) {
	reqs, jobs, dsRepo, models := newCodingAgentTestRepos()
	agent := newTestCodingAgent(reqs, jobs, dsRepo, models)
	job := &domain.FineTuneJob{ID: "ftj_1", RequestID: "ft_1", Status: domain.JobRunning, CurrentStep: 100}
	var loss []domain.MetricPoint
	for i := 0; i < 6; i++ {
		loss = append(loss, domain.MetricPoint{Step: i, Value: 3.0 - float64(i)*0.1})
	}
	job.Loss = loss
	_ = jobs.Create(job)

	opt, err := agent.MonitorTraining(context.Background(), "ftj_1")
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}
	if opt.LossDirection != "decreasing" {
		t.Errorf("expected loss direction 'decreasing', got %q", opt.LossDirection)
	}
	if opt.Action != domain.ActionContinue {
		t.Errorf("expected action %q, got %q", domain.ActionContinue, opt.Action)
	}
}

func TestCodingAgent_MonitorTraining_LossIncreasing(t *testing.T) {
	reqs, jobs, dsRepo, models := newCodingAgentTestRepos()
	agent := newTestCodingAgent(reqs, jobs, dsRepo, models)
	job := &domain.FineTuneJob{ID: "ftj_1", RequestID: "ft_1", Status: domain.JobRunning}
	var loss []domain.MetricPoint
	for i := 0; i < 6; i++ {
		loss = append(loss, domain.MetricPoint{Step: i, Value: 1.0 + float64(i)*0.1})
	}
	job.Loss = loss
	_ = jobs.Create(job)

	opt, err := agent.MonitorTraining(context.Background(), "ftj_1")
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}
	if opt.LossDirection != "increasing" {
		t.Errorf("expected loss direction 'increasing', got %q", opt.LossDirection)
	}
	if opt.Action != domain.ActionReduceLR {
		t.Errorf("expected action %q, got %q", domain.ActionReduceLR, opt.Action)
	}
}

func TestCodingAgent_MonitorTraining_NotFound(t *testing.T) {
	reqs, jobs, dsRepo, models := newCodingAgentTestRepos()
	agent := newTestCodingAgent(reqs, jobs, dsRepo, models)

	_, err := agent.MonitorTraining(context.Background(), "missing")
	if err != domain.ErrNotFound {
		t.Errorf("expected ErrNotFound, got %v", err)
	}
}

func TestCodingAgent_ValidateTrainingQuality_Completed(t *testing.T) {
	reqs, jobs, dsRepo, models := newCodingAgentTestRepos()
	agent := newTestCodingAgent(reqs, jobs, dsRepo, models)
	job := &domain.FineTuneJob{
		ID:        "ftj_1",
		RequestID: "ft_1",
		Status:    domain.JobCompleted,
		FinalMetrics: map[string]float64{
			string(domain.MetricLoss):              1.2,
			string(domain.MetricSyntaxValid):       0.97,
			string(domain.MetricCodeExecutability): 0.95,
		},
	}
	_ = jobs.Create(job)

	report, err := agent.ValidateTrainingQuality(context.Background(), "ftj_1")
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}
	if !report.TrainingComplete {
		t.Error("expected training complete")
	}
	if report.FinalLoss != 1.2 {
		t.Errorf("expected final loss 1.2, got %f", report.FinalLoss)
	}
	if report.OverallQuality != "excellent" {
		t.Errorf("expected quality 'excellent', got %q", report.OverallQuality)
	}
	if report.SyntaxValidity != 0.97 {
		t.Errorf("expected syntax validity 0.97, got %f", report.SyntaxValidity)
	}
}

func TestCodingAgent_ValidateTrainingQuality_Fair(t *testing.T) {
	reqs, jobs, dsRepo, models := newCodingAgentTestRepos()
	agent := newTestCodingAgent(reqs, jobs, dsRepo, models)
	job := &domain.FineTuneJob{
		ID:        "ftj_1",
		RequestID: "ft_1",
		Status:    domain.JobCompleted,
		FinalMetrics: map[string]float64{
			string(domain.MetricLoss):        2.5,
			string(domain.MetricSyntaxValid): 0.8,
		},
	}
	_ = jobs.Create(job)

	report, err := agent.ValidateTrainingQuality(context.Background(), "ftj_1")
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}
	if report.OverallQuality != "fair" {
		t.Errorf("expected quality 'fair', got %q", report.OverallQuality)
	}
}

func TestCodingAgent_ValidateTrainingQuality_NoMetrics(t *testing.T) {
	reqs, jobs, dsRepo, models := newCodingAgentTestRepos()
	agent := newTestCodingAgent(reqs, jobs, dsRepo, models)
	job := &domain.FineTuneJob{ID: "ftj_1", RequestID: "ft_1", Status: domain.JobCompleted}
	_ = jobs.Create(job)

	report, err := agent.ValidateTrainingQuality(context.Background(), "ftj_1")
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}
	if report.OverallQuality != "good" {
		t.Errorf("expected quality 'good', got %q", report.OverallQuality)
	}
}

func TestCodingAgent_ValidateTrainingQuality_NotFound(t *testing.T) {
	reqs, jobs, dsRepo, models := newCodingAgentTestRepos()
	agent := newTestCodingAgent(reqs, jobs, dsRepo, models)

	_, err := agent.ValidateTrainingQuality(context.Background(), "missing")
	if err != domain.ErrNotFound {
		t.Errorf("expected ErrNotFound, got %v", err)
	}
}

func TestCodingAgent_GetInsights_Completed(t *testing.T) {
	reqs, jobs, dsRepo, models := newCodingAgentTestRepos()
	agent := newTestCodingAgent(reqs, jobs, dsRepo, models)
	job := &domain.FineTuneJob{
		ID:           "ftj_1",
		RequestID:    "ft_1",
		Status:       domain.JobCompleted,
		Progress:     100,
		CurrentEpoch: 3,
		TotalEpochs:  3,
		FinalMetrics: map[string]float64{string(domain.MetricLoss): 1.1},
	}
	_ = jobs.Create(job)

	insights, err := agent.GetInsights(context.Background(), "ftj_1")
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}
	if insights.Phase != "completion" {
		t.Errorf("expected phase 'completion', got %q", insights.Phase)
	}
	if insights.EstimatedQuality != "ready" {
		t.Errorf("expected quality 'ready', got %q", insights.EstimatedQuality)
	}
	if len(insights.NextSteps) != 3 {
		t.Errorf("expected 3 next steps, got %d", len(insights.NextSteps))
	}
}

func TestCodingAgent_GetInsights_Running(t *testing.T) {
	reqs, jobs, dsRepo, models := newCodingAgentTestRepos()
	agent := newTestCodingAgent(reqs, jobs, dsRepo, models)
	job := &domain.FineTuneJob{
		ID:           "ftj_1",
		RequestID:    "ft_1",
		Status:       domain.JobRunning,
		CurrentEpoch: 1,
		TotalEpochs:  3,
		CurrentStep:  500,
	}
	_ = jobs.Create(job)

	insights, err := agent.GetInsights(context.Background(), "ftj_1")
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}
	if insights.Phase != "training" {
		t.Errorf("expected phase 'training', got %q", insights.Phase)
	}
	if insights.EstimatedQuality != "progressing" {
		t.Errorf("expected quality 'progressing', got %q", insights.EstimatedQuality)
	}
}

func TestCodingAgent_GetInsights_Failed(t *testing.T) {
	reqs, jobs, dsRepo, models := newCodingAgentTestRepos()
	agent := newTestCodingAgent(reqs, jobs, dsRepo, models)
	job := &domain.FineTuneJob{
		ID:        "ftj_1",
		RequestID: "ft_1",
		Status:    domain.JobFailed,
		Error:     "OOM",
	}
	_ = jobs.Create(job)

	insights, err := agent.GetInsights(context.Background(), "ftj_1")
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}
	if insights.Phase != "completion" {
		t.Errorf("expected phase 'completion', got %q", insights.Phase)
	}
	if insights.EstimatedQuality != "failed" {
		t.Errorf("expected quality 'failed', got %q", insights.EstimatedQuality)
	}
}

func TestCodingAgent_GetInsights_Unknown(t *testing.T) {
	reqs, jobs, dsRepo, models := newCodingAgentTestRepos()
	agent := newTestCodingAgent(reqs, jobs, dsRepo, models)
	job := &domain.FineTuneJob{ID: "ftj_1", RequestID: "ft_1", Status: domain.JobStatus("weird")}
	_ = jobs.Create(job)

	insights, err := agent.GetInsights(context.Background(), "ftj_1")
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}
	if insights.Phase != "unknown" {
		t.Errorf("expected phase 'unknown', got %q", insights.Phase)
	}
}

func TestCodingAgent_GetInsights_NotFound(t *testing.T) {
	reqs, jobs, dsRepo, models := newCodingAgentTestRepos()
	agent := newTestCodingAgent(reqs, jobs, dsRepo, models)

	_, err := agent.GetInsights(context.Background(), "missing")
	if err != domain.ErrNotFound {
		t.Errorf("expected ErrNotFound, got %v", err)
	}
}
