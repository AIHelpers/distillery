package domain_test

import (
	"encoding/json"
	"testing"
	"time"

	"distillery/internal/domain"
)

func TestProgrammingLanguageConstants(t *testing.T) {
	tests := []struct {
		name  string
		value domain.ProgrammingLanguage
		want  string
	}{
		{"LangPython", domain.LangPython, "python"},
		{"LangGo", domain.LangGo, "go"},
		{"LangJavaScript", domain.LangJavaScript, "javascript"},
		{"LangTypeScript", domain.LangTypeScript, "typescript"},
		{"LangJava", domain.LangJava, "java"},
		{"LangCSharp", domain.LangCSharp, "csharp"},
		{"LangRust", domain.LangRust, "rust"},
		{"LangCpp", domain.LangCpp, "cpp"},
	}
	for _, tt := range tests {
		if string(tt.value) != tt.want {
			t.Errorf("%s: expected %q, got %q", tt.name, tt.want, string(tt.value))
		}
	}
}

func TestSkillCategoryConstants(t *testing.T) {
	tests := []struct {
		name  string
		value domain.SkillCategory
		want  string
	}{
		{"SkillCodeGeneration", domain.SkillCodeGeneration, "code_generation"},
		{"SkillCodeCompletion", domain.SkillCodeCompletion, "code_completion"},
		{"SkillBugFixing", domain.SkillBugFixing, "bug_fixing"},
		{"SkillOptimization", domain.SkillOptimization, "optimization"},
		{"SkillDocumentation", domain.SkillDocumentation, "documentation"},
	}
	for _, tt := range tests {
		if string(tt.value) != tt.want {
			t.Errorf("%s: expected %q, got %q", tt.name, tt.want, string(tt.value))
		}
	}
}

func TestRequestStatusConstants(t *testing.T) {
	tests := []struct {
		name  string
		value domain.RequestStatus
		want  string
	}{
		{"RequestDraft", domain.RequestDraft, "draft"},
		{"RequestQueued", domain.RequestQueued, "queued"},
		{"RequestTraining", domain.RequestTraining, "training"},
		{"RequestCompleted", domain.RequestCompleted, "completed"},
		{"RequestFailed", domain.RequestFailed, "failed"},
	}
	for _, tt := range tests {
		if string(tt.value) != tt.want {
			t.Errorf("%s: expected %q, got %q", tt.name, tt.want, string(tt.value))
		}
	}
}

func TestJobStatusConstants(t *testing.T) {
	tests := []struct {
		name  string
		value domain.JobStatus
		want  string
	}{
		{"JobQueued", domain.JobQueued, "queued"},
		{"JobPreparing", domain.JobPreparing, "preparing"},
		{"JobRunning", domain.JobRunning, "running"},
		{"JobCompleted", domain.JobCompleted, "completed"},
		{"JobFailed", domain.JobFailed, "failed"},
	}
	for _, tt := range tests {
		if string(tt.value) != tt.want {
			t.Errorf("%s: expected %q, got %q", tt.name, tt.want, string(tt.value))
		}
	}
}

func TestMetricTypeConstants(t *testing.T) {
	tests := []struct {
		name  string
		value domain.MetricType
		want  string
	}{
		{"MetricLoss", domain.MetricLoss, "loss"},
		{"MetricPerplexity", domain.MetricPerplexity, "perplexity"},
		{"MetricBleuScore", domain.MetricBleuScore, "bleu_score"},
		{"MetricCodeExecutability", domain.MetricCodeExecutability, "code_executability"},
		{"MetricSyntaxValid", domain.MetricSyntaxValid, "syntax_valid"},
	}
	for _, tt := range tests {
		if string(tt.value) != tt.want {
			t.Errorf("%s: expected %q, got %q", tt.name, tt.want, string(tt.value))
		}
	}
}

func TestOptimizationActionConstants(t *testing.T) {
	tests := []struct {
		name  string
		value domain.OptimizationAction
		want  string
	}{
		{"ActionContinue", domain.ActionContinue, "continue"},
		{"ActionReduceLR", domain.ActionReduceLR, "reduce_learning_rate"},
		{"ActionIncreaseBatch", domain.ActionIncreaseBatch, "increase_batch_size"},
		{"ActionEarlyStop", domain.ActionEarlyStop, "early_stop"},
	}
	for _, tt := range tests {
		if string(tt.value) != tt.want {
			t.Errorf("%s: expected %q, got %q", tt.name, tt.want, string(tt.value))
		}
	}
}

func TestFineTuneRequest_JSONRoundTrip(t *testing.T) {
	now := time.Now().UTC()
	req := &domain.FineTuneRequest{
		ID:          "ft_1",
		Name:        "Python Code Completion",
		Description: "Fine-tune for Python code completion",
		Language:    domain.LangPython,
		Skill:       domain.SkillCodeCompletion,
		BaseModel:   "gpt2-large",
		DatasetID:   "ds_1",
		TrainingParams: domain.TrainingParameters{
			Epochs:       3,
			BatchSize:    8,
			LearningRate: 5e-5,
		},
		Owner:     "user_1",
		Status:    domain.RequestDraft,
		CreatedAt: now,
		UpdatedAt: now,
	}

	data, err := json.Marshal(req)
	if err != nil {
		t.Fatalf("expected no marshal error, got %v", err)
	}

	var decoded domain.FineTuneRequest
	if err := json.Unmarshal(data, &decoded); err != nil {
		t.Fatalf("expected no unmarshal error, got %v", err)
	}

	if decoded.ID != req.ID {
		t.Errorf("expected ID %q, got %q", req.ID, decoded.ID)
	}
	if decoded.Language != domain.LangPython {
		t.Errorf("expected language %q, got %q", domain.LangPython, decoded.Language)
	}
	if decoded.Skill != domain.SkillCodeCompletion {
		t.Errorf("expected skill %q, got %q", domain.SkillCodeCompletion, decoded.Skill)
	}
	if decoded.TrainingParams.Epochs != 3 {
		t.Errorf("expected epochs 3, got %d", decoded.TrainingParams.Epochs)
	}
	if decoded.TrainingParams.LearningRate != 5e-5 {
		t.Errorf("expected learning rate 5e-5, got %f", decoded.TrainingParams.LearningRate)
	}
	if decoded.Status != domain.RequestDraft {
		t.Errorf("expected status %q, got %q", domain.RequestDraft, decoded.Status)
	}
	if !decoded.CreatedAt.Equal(now) {
		t.Errorf("expected CreatedAt %v, got %v", now, decoded.CreatedAt)
	}
}

func TestFineTuneJob_JSONRoundTrip(t *testing.T) {
	now := time.Now().UTC()
	job := &domain.FineTuneJob{
		ID:            "ftj_1",
		RequestID:     "ft_1",
		Status:        domain.JobRunning,
		Progress:      50.0,
		CurrentEpoch:  1,
		TotalEpochs:   3,
		CurrentStep:   100,
		TotalSteps:    300,
		Loss:          []domain.MetricPoint{{Step: 100, Epoch: 1, Value: 1.5, Timestamp: now}},
		FinalMetrics:  map[string]float64{string(domain.MetricLoss): 1.2},
		Error:         "",
		OutputModelID: "model_1",
		ComputeCost:   12.5,
		GPUHours:      3.5,
		CreatedAt:     now,
	}

	data, err := json.Marshal(job)
	if err != nil {
		t.Fatalf("expected no marshal error, got %v", err)
	}

	var decoded domain.FineTuneJob
	if err := json.Unmarshal(data, &decoded); err != nil {
		t.Fatalf("expected no unmarshal error, got %v", err)
	}

	if decoded.ID != job.ID {
		t.Errorf("expected ID %q, got %q", job.ID, decoded.ID)
	}
	if decoded.Status != domain.JobRunning {
		t.Errorf("expected status %q, got %q", domain.JobRunning, decoded.Status)
	}
	if decoded.Progress != 50.0 {
		t.Errorf("expected progress 50.0, got %f", decoded.Progress)
	}
	if len(decoded.Loss) != 1 {
		t.Errorf("expected 1 loss point, got %d", len(decoded.Loss))
	}
	if decoded.Loss[0].Value != 1.5 {
		t.Errorf("expected loss value 1.5, got %f", decoded.Loss[0].Value)
	}
	if decoded.FinalMetrics[string(domain.MetricLoss)] != 1.2 {
		t.Errorf("expected final loss 1.2, got %f", decoded.FinalMetrics[string(domain.MetricLoss)])
	}
}

func TestTrainedModel_JSONRoundTrip(t *testing.T) {
	now := time.Now().UTC()
	model := &domain.TrainedModel{
		ID:              "tm_1",
		Name:            "Python Completion v1",
		Language:        domain.LangPython,
		Skill:           domain.SkillCodeCompletion,
		BaseModel:       "gpt2-large",
		FineTuneJobID:   "ftj_1",
		Version:         "1.0.0",
		Status:          "ready",
		Checkpoint:      "checkpoint-300",
		HuggingFaceURL:  "https://huggingface.co/user/model",
		Quantized:       true,
		QuantBits:       8,
		TrainingMetrics: map[string]float64{string(domain.MetricLoss): 1.2},
		BenchmarkScore:  85.5,
		Owner:           "user_1",
		IsPublic:        true,
		Downloads:       100,
		Rating:          4.5,
		CreatedAt:       now,
	}

	data, err := json.Marshal(model)
	if err != nil {
		t.Fatalf("expected no marshal error, got %v", err)
	}

	var decoded domain.TrainedModel
	if err := json.Unmarshal(data, &decoded); err != nil {
		t.Fatalf("expected no unmarshal error, got %v", err)
	}

	if decoded.ID != model.ID {
		t.Errorf("expected ID %q, got %q", model.ID, decoded.ID)
	}
	if decoded.Status != "ready" {
		t.Errorf("expected status 'ready', got %q", decoded.Status)
	}
	if !decoded.Quantized {
		t.Error("expected quantized true")
	}
	if decoded.QuantBits != 8 {
		t.Errorf("expected quant bits 8, got %d", decoded.QuantBits)
	}
	if decoded.BenchmarkScore != 85.5 {
		t.Errorf("expected benchmark score 85.5, got %f", decoded.BenchmarkScore)
	}
	if !decoded.IsPublic {
		t.Error("expected is_public true")
	}
	if decoded.Downloads != 100 {
		t.Errorf("expected downloads 100, got %d", decoded.Downloads)
	}
}

func TestDatasetInfo_JSONRoundTrip(t *testing.T) {
	now := time.Now().UTC()
	dataset := &domain.DatasetInfo{
		ID:          "ds_1",
		Name:        "Python Corpus",
		Language:    domain.LangPython,
		FileCount:   1000,
		TotalSize:   1024 * 1024,
		TotalLines:  50000,
		TotalTokens: 1000000,
		SampleCount: 500,
		Quality: domain.DatasetQuality{
			OverallScore:    85.0,
			ValidityRate:    0.95,
			ComplexityScore: 7.5,
		},
		Status:    "validated",
		Owner:     "user_1",
		CreatedAt: now,
	}

	data, err := json.Marshal(dataset)
	if err != nil {
		t.Fatalf("expected no marshal error, got %v", err)
	}

	var decoded domain.DatasetInfo
	if err := json.Unmarshal(data, &decoded); err != nil {
		t.Fatalf("expected no unmarshal error, got %v", err)
	}

	if decoded.ID != dataset.ID {
		t.Errorf("expected ID %q, got %q", dataset.ID, decoded.ID)
	}
	if decoded.FileCount != 1000 {
		t.Errorf("expected file count 1000, got %d", decoded.FileCount)
	}
	if decoded.Quality.OverallScore != 85.0 {
		t.Errorf("expected quality score 85.0, got %f", decoded.Quality.OverallScore)
	}
	if decoded.Status != "validated" {
		t.Errorf("expected status 'validated', got %q", decoded.Status)
	}
}

func TestMetricPoint_JSONRoundTrip(t *testing.T) {
	now := time.Now().UTC()
	point := domain.MetricPoint{
		Step:      50,
		Epoch:     1,
		Value:     2.5,
		Timestamp: now,
	}

	data, err := json.Marshal(point)
	if err != nil {
		t.Fatalf("expected no marshal error, got %v", err)
	}

	var decoded domain.MetricPoint
	if err := json.Unmarshal(data, &decoded); err != nil {
		t.Fatalf("expected no unmarshal error, got %v", err)
	}

	if decoded.Step != 50 {
		t.Errorf("expected step 50, got %d", decoded.Step)
	}
	if decoded.Value != 2.5 {
		t.Errorf("expected value 2.5, got %f", decoded.Value)
	}
}

func TestCheckpointInfo_JSONRoundTrip(t *testing.T) {
	cp := domain.CheckpointInfo{
		Path:    "checkpoints/step-300",
		Step:    300,
		Epoch:   2,
		Metrics: map[string]float64{"loss": 1.5},
		IsBest:  true,
	}

	data, err := json.Marshal(cp)
	if err != nil {
		t.Fatalf("expected no marshal error, got %v", err)
	}

	var decoded domain.CheckpointInfo
	if err := json.Unmarshal(data, &decoded); err != nil {
		t.Fatalf("expected no unmarshal error, got %v", err)
	}

	if decoded.Path != cp.Path {
		t.Errorf("expected path %q, got %q", cp.Path, decoded.Path)
	}
	if decoded.IsBest != true {
		t.Error("expected is_best true")
	}
}
