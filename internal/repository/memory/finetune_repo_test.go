package memory_test

import (
	"testing"

	"distillery/internal/domain"
	"distillery/internal/repository/memory"
)

func newFinetuneRepos() (*memory.FineTuneRequestRepo, *memory.FineTuneJobRepo, *memory.TrainedModelRepo, *memory.DatasetRepo, *memory.Store) {
	s := memory.NewStore("")
	return memory.NewFineTuneRequestRepo(s), memory.NewFineTuneJobRepo(s), memory.NewTrainedModelRepo(s), memory.NewDatasetRepo(s), s
}

func newSampleRequest(id string) *domain.FineTuneRequest {
	return &domain.FineTuneRequest{
		ID:        id,
		Name:      "Test Request",
		Language:  domain.LangPython,
		Skill:     domain.SkillCodeCompletion,
		BaseModel: "gpt2-large",
		DatasetID: "ds_1",
		Owner:     "user_1",
		Status:    domain.RequestDraft,
	}
}

func sampleJob(id, requestID string) *domain.FineTuneJob {
	return &domain.FineTuneJob{
		ID:        id,
		RequestID: requestID,
		Status:    domain.JobQueued,
	}
}

func sampleModel(id string) *domain.TrainedModel {
	return &domain.TrainedModel{
		ID:            id,
		Name:          "Test Model",
		Language:      domain.LangPython,
		Skill:         domain.SkillCodeCompletion,
		BaseModel:     "gpt2-large",
		FineTuneJobID: "ftj_1",
		Owner:         "user_1",
		Status:        "ready",
	}
}

func sampleDataset(id string) *domain.DatasetInfo {
	return &domain.DatasetInfo{
		ID:       id,
		Name:     "Test Dataset",
		Language: domain.LangPython,
		Owner:    "user_1",
		Status:   "validated",
	}
}

// --- FineTuneRequestRepository ---.

func TestFineTuneRequestRepo_Create(t *testing.T) {
	r, _, _, _, s := newFinetuneRepos()
	req := newSampleRequest("ft_1")

	if err := r.Create(req); err != nil {
		t.Fatalf("expected no error, got %v", err)
	}
	if len(s.FineTuneRequests) != 1 {
		t.Errorf("expected 1 request, got %d", len(s.FineTuneRequests))
	}
	if s.FineTuneRequests[0] != req {
		t.Error("expected request to be stored")
	}
}

func TestFineTuneRequestRepo_Get_Found(t *testing.T) {
	r, _, _, _, _ := newFinetuneRepos()
	req := newSampleRequest("ft_1")
	_ = r.Create(req)

	got, err := r.Get("ft_1")
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}
	if got.ID != "ft_1" {
		t.Errorf("expected ID 'ft_1', got %q", got.ID)
	}
}

func TestFineTuneRequestRepo_Get_NotFound(t *testing.T) {
	r, _, _, _, _ := newFinetuneRepos()

	_, err := r.Get("missing")
	if err != domain.ErrNotFound {
		t.Errorf("expected ErrNotFound, got %v", err)
	}
}

func TestFineTuneRequestRepo_ListByOwner(t *testing.T) {
	r, _, _, _, _ := newFinetuneRepos()
	_ = r.Create(newSampleRequest("ft_1"))
	_ = r.Create(newSampleRequest("ft_2"))
	other := newSampleRequest("ft_3")
	other.Owner = "user_2"
	_ = r.Create(other)

	list, err := r.ListByOwner("user_1")
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}
	if len(list) != 2 {
		t.Errorf("expected 2 requests for user_1, got %d", len(list))
	}
}

func TestFineTuneRequestRepo_ListByOwner_Empty(t *testing.T) {
	r, _, _, _, _ := newFinetuneRepos()

	list, err := r.ListByOwner("nobody")
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}
	if len(list) != 0 {
		t.Errorf("expected 0 requests, got %d", len(list))
	}
}

func TestFineTuneRequestRepo_List(t *testing.T) {
	r, _, _, _, _ := newFinetuneRepos()
	_ = r.Create(newSampleRequest("ft_1"))
	_ = r.Create(newSampleRequest("ft_2"))

	list, err := r.List()
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}
	if len(list) != 2 {
		t.Errorf("expected 2 requests, got %d", len(list))
	}
}

func TestFineTuneRequestRepo_Update_Found(t *testing.T) {
	r, _, _, _, _ := newFinetuneRepos()
	_ = r.Create(newSampleRequest("ft_1"))

	updated := newSampleRequest("ft_1")
	updated.Name = "Updated Name"
	if err := r.Update(updated); err != nil {
		t.Fatalf("expected no error, got %v", err)
	}

	got, _ := r.Get("ft_1")
	if got.Name != "Updated Name" {
		t.Errorf("expected name 'Updated Name', got %q", got.Name)
	}
}

func TestFineTuneRequestRepo_Update_NotFound(t *testing.T) {
	r, _, _, _, _ := newFinetuneRepos()

	err := r.Update(newSampleRequest("missing"))
	if err != domain.ErrNotFound {
		t.Errorf("expected ErrNotFound, got %v", err)
	}
}

func TestFineTuneRequestRepo_Delete_Found(t *testing.T) {
	r, _, _, _, s := newFinetuneRepos()
	_ = r.Create(newSampleRequest("ft_1"))

	if err := r.Delete("ft_1"); err != nil {
		t.Fatalf("expected no error, got %v", err)
	}
	if len(s.FineTuneRequests) != 0 {
		t.Errorf("expected 0 requests after delete, got %d", len(s.FineTuneRequests))
	}
}

func TestFineTuneRequestRepo_Delete_NotFound(t *testing.T) {
	r, _, _, _, _ := newFinetuneRepos()

	err := r.Delete("missing")
	if err != domain.ErrNotFound {
		t.Errorf("expected ErrNotFound, got %v", err)
	}
}

// --- FineTuneJobRepository ---.

func TestFineTuneJobRepo_Create(t *testing.T) {
	_, r, _, _, s := newFinetuneRepos()
	job := sampleJob("ftj_1", "ft_1")

	if err := r.Create(job); err != nil {
		t.Fatalf("expected no error, got %v", err)
	}
	if len(s.FineTuneJobs) != 1 {
		t.Errorf("expected 1 job, got %d", len(s.FineTuneJobs))
	}
	if s.FineTuneJobs[0] != job {
		t.Error("expected job to be stored")
	}
}

func TestFineTuneJobRepo_Get_Found(t *testing.T) {
	_, r, _, _, _ := newFinetuneRepos()
	_ = r.Create(sampleJob("ftj_1", "ft_1"))

	got, err := r.Get("ftj_1")
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}
	if got.ID != "ftj_1" {
		t.Errorf("expected ID 'ftj_1', got %q", got.ID)
	}
	if got.RequestID != "ft_1" {
		t.Errorf("expected RequestID 'ft_1', got %q", got.RequestID)
	}
}

func TestFineTuneJobRepo_Get_NotFound(t *testing.T) {
	_, r, _, _, _ := newFinetuneRepos()

	_, err := r.Get("missing")
	if err != domain.ErrNotFound {
		t.Errorf("expected ErrNotFound, got %v", err)
	}
}

func TestFineTuneJobRepo_GetByRequestID_Found(t *testing.T) {
	_, r, _, _, _ := newFinetuneRepos()
	_ = r.Create(sampleJob("ftj_1", "ft_1"))

	got, err := r.GetByRequestID("ft_1")
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}
	if got.ID != "ftj_1" {
		t.Errorf("expected job ID 'ftj_1', got %q", got.ID)
	}
}

func TestFineTuneJobRepo_GetByRequestID_NotFound(t *testing.T) {
	_, r, _, _, _ := newFinetuneRepos()

	_, err := r.GetByRequestID("missing")
	if err != domain.ErrNotFound {
		t.Errorf("expected ErrNotFound, got %v", err)
	}
}

func TestFineTuneJobRepo_List(t *testing.T) {
	_, r, _, _, _ := newFinetuneRepos()
	_ = r.Create(sampleJob("ftj_1", "ft_1"))
	_ = r.Create(sampleJob("ftj_2", "ft_2"))

	list, err := r.List()
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}
	if len(list) != 2 {
		t.Errorf("expected 2 jobs, got %d", len(list))
	}
}

func TestFineTuneJobRepo_ListActive(t *testing.T) {
	_, r, _, _, _ := newFinetuneRepos()
	_ = r.Create(sampleJob("ftj_1", "ft_1")) // queued = active
	_ = r.Create(sampleJob("ftj_2", "ft_2")) // queued = active

	completed := sampleJob("ftj_3", "ft_3")
	completed.Status = domain.JobCompleted
	_ = r.Create(completed)

	failed := sampleJob("ftj_4", "ft_4")
	failed.Status = domain.JobFailed
	_ = r.Create(failed)

	list, err := r.ListActive()
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}
	if len(list) != 2 {
		t.Errorf("expected 2 active jobs, got %d", len(list))
	}
}

func TestFineTuneJobRepo_UpdateStatus(t *testing.T) {
	_, r, _, _, _ := newFinetuneRepos()
	_ = r.Create(sampleJob("ftj_1", "ft_1"))

	if err := r.UpdateStatus("ftj_1", domain.JobRunning); err != nil {
		t.Fatalf("expected no error, got %v", err)
	}

	got, _ := r.Get("ftj_1")
	if got.Status != domain.JobRunning {
		t.Errorf("expected status %q, got %q", domain.JobRunning, got.Status)
	}
}

func TestFineTuneJobRepo_UpdateStatus_NotFound(t *testing.T) {
	_, r, _, _, _ := newFinetuneRepos()

	err := r.UpdateStatus("missing", domain.JobRunning)
	if err != domain.ErrNotFound {
		t.Errorf("expected ErrNotFound, got %v", err)
	}
}

func TestFineTuneJobRepo_UpdateProgress(t *testing.T) {
	_, r, _, _, _ := newFinetuneRepos()
	_ = r.Create(sampleJob("ftj_1", "ft_1"))

	if err := r.UpdateProgress("ftj_1", 75.5, 2, 250); err != nil {
		t.Fatalf("expected no error, got %v", err)
	}

	got, _ := r.Get("ftj_1")
	if got.Progress != 75.5 {
		t.Errorf("expected progress 75.5, got %f", got.Progress)
	}
	if got.CurrentEpoch != 2 {
		t.Errorf("expected epoch 2, got %d", got.CurrentEpoch)
	}
	if got.CurrentStep != 250 {
		t.Errorf("expected step 250, got %d", got.CurrentStep)
	}
}

func TestFineTuneJobRepo_UpdateProgress_NotFound(t *testing.T) {
	_, r, _, _, _ := newFinetuneRepos()

	err := r.UpdateProgress("missing", 1, 1, 1)
	if err != domain.ErrNotFound {
		t.Errorf("expected ErrNotFound, got %v", err)
	}
}

func TestFineTuneJobRepo_AddMetric_Loss(t *testing.T) {
	_, r, _, _, _ := newFinetuneRepos()
	_ = r.Create(sampleJob("ftj_1", "ft_1"))

	point := domain.MetricPoint{Step: 10, Epoch: 1, Value: 2.5}
	if err := r.AddMetric("ftj_1", string(domain.MetricLoss), point); err != nil {
		t.Fatalf("expected no error, got %v", err)
	}

	got, _ := r.Get("ftj_1")
	if len(got.Loss) != 1 {
		t.Errorf("expected 1 loss point, got %d", len(got.Loss))
	}
	if got.Loss[0].Value != 2.5 {
		t.Errorf("expected loss value 2.5, got %f", got.Loss[0].Value)
	}
}

func TestFineTuneJobRepo_AddMetric_ValidationLoss(t *testing.T) {
	_, r, _, _, _ := newFinetuneRepos()
	_ = r.Create(sampleJob("ftj_1", "ft_1"))

	point := domain.MetricPoint{Step: 10, Epoch: 1, Value: 1.5}
	if err := r.AddMetric("ftj_1", "validation_loss", point); err != nil {
		t.Fatalf("expected no error, got %v", err)
	}

	got, _ := r.Get("ftj_1")
	if len(got.ValidationLoss) != 1 {
		t.Errorf("expected 1 validation loss point, got %d", len(got.ValidationLoss))
	}
}

func TestFineTuneJobRepo_AddMetric_LearningRate(t *testing.T) {
	_, r, _, _, _ := newFinetuneRepos()
	_ = r.Create(sampleJob("ftj_1", "ft_1"))

	point := domain.MetricPoint{Step: 10, Epoch: 1, Value: 3e-5}
	if err := r.AddMetric("ftj_1", "learning_rate", point); err != nil {
		t.Fatalf("expected no error, got %v", err)
	}

	got, _ := r.Get("ftj_1")
	if len(got.LearningRate) != 1 {
		t.Errorf("expected 1 learning rate point, got %d", len(got.LearningRate))
	}
}

func TestFineTuneJobRepo_AddMetric_Custom(t *testing.T) {
	_, r, _, _, _ := newFinetuneRepos()
	_ = r.Create(sampleJob("ftj_1", "ft_1"))

	point := domain.MetricPoint{Step: 10, Epoch: 1, Value: 0.9}
	if err := r.AddMetric("ftj_1", "bleu_score", point); err != nil {
		t.Fatalf("expected no error, got %v", err)
	}

	got, _ := r.Get("ftj_1")
	if got.CustomMetrics == nil {
		t.Fatal("expected custom metrics map to be initialized")
	}
	if len(got.CustomMetrics["bleu_score"]) != 1 {
		t.Errorf("expected 1 bleu_score point, got %d", len(got.CustomMetrics["bleu_score"]))
	}
}

func TestFineTuneJobRepo_AddMetric_NotFound(t *testing.T) {
	_, r, _, _, _ := newFinetuneRepos()

	err := r.AddMetric("missing", string(domain.MetricLoss), domain.MetricPoint{})
	if err != domain.ErrNotFound {
		t.Errorf("expected ErrNotFound, got %v", err)
	}
}

func TestFineTuneJobRepo_SetFinalMetrics(t *testing.T) {
	_, r, _, _, _ := newFinetuneRepos()
	_ = r.Create(sampleJob("ftj_1", "ft_1"))

	metrics := map[string]float64{string(domain.MetricLoss): 1.2, string(domain.MetricSyntaxValid): 0.95}
	if err := r.SetFinalMetrics("ftj_1", metrics); err != nil {
		t.Fatalf("expected no error, got %v", err)
	}

	got, _ := r.Get("ftj_1")
	if got.FinalMetrics[string(domain.MetricLoss)] != 1.2 {
		t.Errorf("expected final loss 1.2, got %f", got.FinalMetrics[string(domain.MetricLoss)])
	}
	if len(got.FinalMetrics) != 2 {
		t.Errorf("expected 2 final metrics, got %d", len(got.FinalMetrics))
	}
}

func TestFineTuneJobRepo_SetFinalMetrics_NotFound(t *testing.T) {
	_, r, _, _, _ := newFinetuneRepos()

	err := r.SetFinalMetrics("missing", map[string]float64{})
	if err != domain.ErrNotFound {
		t.Errorf("expected ErrNotFound, got %v", err)
	}
}

func TestFineTuneJobRepo_SetError(t *testing.T) {
	_, r, _, _, _ := newFinetuneRepos()
	_ = r.Create(sampleJob("ftj_1", "ft_1"))

	if err := r.SetError("ftj_1", "training crashed"); err != nil {
		t.Fatalf("expected no error, got %v", err)
	}

	got, _ := r.Get("ftj_1")
	if got.Error != "training crashed" {
		t.Errorf("expected error message 'training crashed', got %q", got.Error)
	}
	if got.Status != domain.JobFailed {
		t.Errorf("expected status %q, got %q", domain.JobFailed, got.Status)
	}
	if got.CompletedAt == nil {
		t.Error("expected CompletedAt to be set")
	}
}

func TestFineTuneJobRepo_SetError_NotFound(t *testing.T) {
	_, r, _, _, _ := newFinetuneRepos()

	err := r.SetError("missing", "error")
	if err != domain.ErrNotFound {
		t.Errorf("expected ErrNotFound, got %v", err)
	}
}

// --- TrainedModelRepository ---.

func TestTrainedModelRepo_Create(t *testing.T) {
	_, _, r, _, s := newFinetuneRepos()
	model := sampleModel("tm_1")

	if err := r.Create(model); err != nil {
		t.Fatalf("expected no error, got %v", err)
	}
	if len(s.TrainedModels) != 1 {
		t.Errorf("expected 1 model, got %d", len(s.TrainedModels))
	}
}

func TestTrainedModelRepo_Get_Found(t *testing.T) {
	_, _, r, _, _ := newFinetuneRepos()
	_ = r.Create(sampleModel("tm_1"))

	got, err := r.Get("tm_1")
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}
	if got.ID != "tm_1" {
		t.Errorf("expected ID 'tm_1', got %q", got.ID)
	}
}

func TestTrainedModelRepo_Get_NotFound(t *testing.T) {
	_, _, r, _, _ := newFinetuneRepos()

	_, err := r.Get("missing")
	if err != domain.ErrNotFound {
		t.Errorf("expected ErrNotFound, got %v", err)
	}
}

func TestTrainedModelRepo_GetByJobID_Found(t *testing.T) {
	_, _, r, _, _ := newFinetuneRepos()
	_ = r.Create(sampleModel("tm_1"))

	got, err := r.GetByJobID("ftj_1")
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}
	if got.ID != "tm_1" {
		t.Errorf("expected model ID 'tm_1', got %q", got.ID)
	}
}

func TestTrainedModelRepo_GetByJobID_NotFound(t *testing.T) {
	_, _, r, _, _ := newFinetuneRepos()

	_, err := r.GetByJobID("missing")
	if err != domain.ErrNotFound {
		t.Errorf("expected ErrNotFound, got %v", err)
	}
}

func TestTrainedModelRepo_List(t *testing.T) {
	_, _, r, _, _ := newFinetuneRepos()
	_ = r.Create(sampleModel("tm_1"))
	_ = r.Create(sampleModel("tm_2"))

	list, err := r.List()
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}
	if len(list) != 2 {
		t.Errorf("expected 2 models, got %d", len(list))
	}
}

func TestTrainedModelRepo_ListByLanguage(t *testing.T) {
	_, _, r, _, _ := newFinetuneRepos()
	_ = r.Create(sampleModel("tm_1")) // python

	goModel := sampleModel("tm_2")
	goModel.Language = domain.LangGo
	_ = r.Create(goModel)

	list, err := r.ListByLanguage(domain.LangPython)
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}
	if len(list) != 1 {
		t.Errorf("expected 1 python model, got %d", len(list))
	}
}

func TestTrainedModelRepo_ListByLanguage_Empty(t *testing.T) {
	_, _, r, _, _ := newFinetuneRepos()

	list, err := r.ListByLanguage(domain.LangRust)
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}
	if len(list) != 0 {
		t.Errorf("expected 0 models, got %d", len(list))
	}
}

func TestTrainedModelRepo_ListBySkill(t *testing.T) {
	_, _, r, _, _ := newFinetuneRepos()
	_ = r.Create(sampleModel("tm_1")) // code_completion

	bugFixModel := sampleModel("tm_2")
	bugFixModel.Skill = domain.SkillBugFixing
	_ = r.Create(bugFixModel)

	list, err := r.ListBySkill(domain.SkillCodeCompletion)
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}
	if len(list) != 1 {
		t.Errorf("expected 1 code_completion model, got %d", len(list))
	}
}

func TestTrainedModelRepo_ListBySkill_Empty(t *testing.T) {
	_, _, r, _, _ := newFinetuneRepos()

	list, err := r.ListBySkill(domain.SkillDocumentation)
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}
	if len(list) != 0 {
		t.Errorf("expected 0 models, got %d", len(list))
	}
}

func TestTrainedModelRepo_Update_Found(t *testing.T) {
	_, _, r, _, _ := newFinetuneRepos()
	_ = r.Create(sampleModel("tm_1"))

	updated := sampleModel("tm_1")
	updated.Name = "Updated Model"
	if err := r.Update(updated); err != nil {
		t.Fatalf("expected no error, got %v", err)
	}

	got, _ := r.Get("tm_1")
	if got.Name != "Updated Model" {
		t.Errorf("expected name 'Updated Model', got %q", got.Name)
	}
}

func TestTrainedModelRepo_Update_NotFound(t *testing.T) {
	_, _, r, _, _ := newFinetuneRepos()

	err := r.Update(sampleModel("missing"))
	if err != domain.ErrNotFound {
		t.Errorf("expected ErrNotFound, got %v", err)
	}
}

// --- DatasetRepository ---.

func TestDatasetRepo_Create(t *testing.T) {
	_, _, _, r, s := newFinetuneRepos()
	ds := sampleDataset("ds_1")

	if err := r.Create(ds); err != nil {
		t.Fatalf("expected no error, got %v", err)
	}
	if len(s.Datasets) != 1 {
		t.Errorf("expected 1 dataset, got %d", len(s.Datasets))
	}
}

func TestDatasetRepo_Get_Found(t *testing.T) {
	_, _, _, r, _ := newFinetuneRepos()
	_ = r.Create(sampleDataset("ds_1"))

	got, err := r.Get("ds_1")
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}
	if got.ID != "ds_1" {
		t.Errorf("expected ID 'ds_1', got %q", got.ID)
	}
}

func TestDatasetRepo_Get_NotFound(t *testing.T) {
	_, _, _, r, _ := newFinetuneRepos()

	_, err := r.Get("missing")
	if err != domain.ErrNotFound {
		t.Errorf("expected ErrNotFound, got %v", err)
	}
}

func TestDatasetRepo_ListByOwner(t *testing.T) {
	_, _, _, r, _ := newFinetuneRepos()
	_ = r.Create(sampleDataset("ds_1"))
	_ = r.Create(sampleDataset("ds_2"))
	other := sampleDataset("ds_3")
	other.Owner = "user_2"
	_ = r.Create(other)

	list, err := r.ListByOwner("user_1")
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}
	if len(list) != 2 {
		t.Errorf("expected 2 datasets for user_1, got %d", len(list))
	}
}

func TestDatasetRepo_ListByOwner_Empty(t *testing.T) {
	_, _, _, r, _ := newFinetuneRepos()

	list, err := r.ListByOwner("nobody")
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}
	if len(list) != 0 {
		t.Errorf("expected 0 datasets, got %d", len(list))
	}
}

func TestDatasetRepo_List(t *testing.T) {
	_, _, _, r, _ := newFinetuneRepos()
	_ = r.Create(sampleDataset("ds_1"))
	_ = r.Create(sampleDataset("ds_2"))

	list, err := r.List()
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}
	if len(list) != 2 {
		t.Errorf("expected 2 datasets, got %d", len(list))
	}
}

func TestDatasetRepo_UpdateQuality(t *testing.T) {
	_, _, _, r, _ := newFinetuneRepos()
	_ = r.Create(sampleDataset("ds_1"))

	quality := domain.DatasetQuality{
		OverallScore:    90.0,
		ValidityRate:    0.98,
		ComplexityScore: 6.5,
	}
	if err := r.UpdateQuality("ds_1", quality); err != nil {
		t.Fatalf("expected no error, got %v", err)
	}

	got, _ := r.Get("ds_1")
	if got.Quality.OverallScore != 90.0 {
		t.Errorf("expected quality score 90.0, got %f", got.Quality.OverallScore)
	}
	if got.Quality.ValidityRate != 0.98 {
		t.Errorf("expected validity rate 0.98, got %f", got.Quality.ValidityRate)
	}
}

func TestDatasetRepo_UpdateQuality_NotFound(t *testing.T) {
	_, _, _, r, _ := newFinetuneRepos()

	err := r.UpdateQuality("missing", domain.DatasetQuality{})
	if err != domain.ErrNotFound {
		t.Errorf("expected ErrNotFound, got %v", err)
	}
}

func TestDatasetRepo_Delete_Found(t *testing.T) {
	_, _, _, r, s := newFinetuneRepos()
	_ = r.Create(sampleDataset("ds_1"))

	if err := r.Delete("ds_1"); err != nil {
		t.Fatalf("expected no error, got %v", err)
	}
	if len(s.Datasets) != 0 {
		t.Errorf("expected 0 datasets after delete, got %d", len(s.Datasets))
	}
}

func TestDatasetRepo_Delete_NotFound(t *testing.T) {
	_, _, _, r, _ := newFinetuneRepos()

	err := r.Delete("missing")
	if err != domain.ErrNotFound {
		t.Errorf("expected ErrNotFound, got %v", err)
	}
}
