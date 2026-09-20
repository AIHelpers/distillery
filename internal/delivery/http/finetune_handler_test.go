package http_test

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	deliveryhttp "distillery/internal/delivery/http"
	"distillery/internal/domain"
	"distillery/internal/infra/agent"
	"distillery/internal/repository/memory"
	"distillery/internal/usecase"
)

func newFinetuneTestServer(t *testing.T) *httptest.Server {
	t.Helper()

	store := memory.NewStore("")
	reqRepo := memory.NewFineTuneRequestRepo(store)
	jobRepo := memory.NewFineTuneJobRepo(store)
	dsRepo := memory.NewDatasetRepo(store)
	modelRepo := memory.NewTrainedModelRepo(store)

	codingAgent := agent.NewCodingAgent(reqRepo, jobRepo, dsRepo, modelRepo)
	uc := usecase.NewFineTuneUsecase(codingAgent, reqRepo, jobRepo, modelRepo, dsRepo, usecase.NewRandomIDGenerator())
	h := deliveryhttp.NewFineTuneHandler(uc)

	mux := http.NewServeMux()
	mux.HandleFunc("POST /api/v1/fine-tuning/datasets", h.RegisterDataset)
	mux.HandleFunc("GET /api/v1/fine-tuning/datasets/{datasetID}/analyse", func(w http.ResponseWriter, r *http.Request) {
		h.AnalyseDataset(w, r, r.PathValue("datasetID"))
	})
	mux.HandleFunc("POST /api/v1/fine-tuning/hyperparameters", h.RecommendHyperparams)
	mux.HandleFunc("POST /api/v1/fine-tuning/requests", h.CreateRequest)
	mux.HandleFunc("GET /api/v1/fine-tuning/requests", h.ListRequests)
	mux.HandleFunc("POST /api/v1/fine-tuning/requests/{requestID}/start", func(w http.ResponseWriter, r *http.Request) {
		h.StartTraining(w, r, r.PathValue("requestID"))
	})
	mux.HandleFunc("GET /api/v1/fine-tuning/jobs", h.ListJobs)
	mux.HandleFunc("GET /api/v1/fine-tuning/jobs/{jobID}", func(w http.ResponseWriter, r *http.Request) {
		h.JobStatus(w, r, r.PathValue("jobID"))
	})
	mux.HandleFunc("GET /api/v1/fine-tuning/jobs/{jobID}/monitor", func(w http.ResponseWriter, r *http.Request) {
		h.JobMonitor(w, r, r.PathValue("jobID"))
	})
	mux.HandleFunc("GET /api/v1/fine-tuning/jobs/{jobID}/insights", func(w http.ResponseWriter, r *http.Request) {
		h.JobInsights(w, r, r.PathValue("jobID"))
	})
	mux.HandleFunc("GET /api/v1/fine-tuning/jobs/{jobID}/quality", func(w http.ResponseWriter, r *http.Request) {
		h.JobQuality(w, r, r.PathValue("jobID"))
	})
	mux.HandleFunc("GET /api/v1/fine-tuning/models", h.ListModels)
	mux.HandleFunc("GET /api/v1/fine-tuning/models/{modelID}", func(w http.ResponseWriter, r *http.Request) {
		h.GetModel(w, r, r.PathValue("modelID"))
	})
	mux.HandleFunc("POST /api/v1/fine-tuning/models/{modelID}/export", func(w http.ResponseWriter, r *http.Request) {
		h.ExportModel(w, r, r.PathValue("modelID"))
	})

	return httptest.NewServer(mux)
}

func TestFineTuneHandler_CreateRequest_Valid(t *testing.T) {
	t.Parallel()

	srv := newFinetuneTestServer(t)
	defer srv.Close()

	body := `{
		"name": "Test Request",
		"language": "python",
		"skill": "code_completion",
		"base_model": "gpt2-large",
		"dataset_id": "ds_1",
		"training_params": {
			"epochs": 3,
			"batch_size": 8,
			"learning_rate": 0.00005
		}
	}`
	req, _ := http.NewRequest("POST", srv.URL+"/api/v1/fine-tuning/requests", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("X-User-ID", "user_1")

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatalf("request failed: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusCreated {
		t.Errorf("expected status 201, got %d", resp.StatusCode)
	}

	var data map[string]interface{}

	err = json.NewDecoder(resp.Body).Decode(&data)
	if err != nil {
		t.Fatalf("failed to decode response: %v", err)
	}

	if data["request_id"] == "" {
		t.Error("expected non-empty request_id")
	}

	if data["status"] != "draft" {
		t.Errorf("expected status 'draft', got %v", data["status"])
	}
}

func TestFineTuneHandler_CreateRequest_InvalidJSON(t *testing.T) {
	t.Parallel()

	srv := newFinetuneTestServer(t)
	defer srv.Close()

	req, _ := http.NewRequest("POST", srv.URL+"/api/v1/fine-tuning/requests", strings.NewReader("{invalid}"))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("X-User-ID", "user_1")

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatalf("request failed: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusBadRequest {
		t.Errorf("expected status 400, got %d", resp.StatusCode)
	}
}

func TestFineTuneHandler_CreateRequest_NoUserHeader(t *testing.T) {
	t.Parallel()

	srv := newFinetuneTestServer(t)
	defer srv.Close()

	body := `{
		"name": "Test Request",
		"language": "python",
		"skill": "code_completion",
		"dataset_id": "ds_1"
	}`
	req, _ := http.NewRequest("POST", srv.URL+"/api/v1/fine-tuning/requests", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatalf("request failed: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusCreated {
		t.Errorf("expected status 201, got %d", resp.StatusCode)
	}
}

func TestFineTuneHandler_RegisterDataset_Valid(t *testing.T) {
	t.Parallel()

	srv := newFinetuneTestServer(t)
	defer srv.Close()

	body := `{
		"name": "Python DS",
		"language": "python",
		"file_count": 100,
		"total_size": 500000,
		"total_tokens": 100000,
		"status": "uploaded",
		"quality": {
			"overall_score": 75,
			"validity_rate": 0.9,
			"complexity_score": 4
		}
	}`
	req, _ := http.NewRequest("POST", srv.URL+"/api/v1/fine-tuning/datasets", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("X-User-ID", "user_1")

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatalf("request failed: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusCreated {
		t.Errorf("expected status 201, got %d", resp.StatusCode)
	}

	var data map[string]interface{}

	err = json.NewDecoder(resp.Body).Decode(&data)
	if err != nil {
		t.Fatalf("failed to decode response: %v", err)
	}

	if data["dataset_id"] == "" {
		t.Error("expected non-empty dataset_id")
	}
}

func TestFineTuneHandler_AnalyseDataset_Existing(t *testing.T) {
	t.Parallel()

	srv := newFinetuneTestServer(t)
	defer srv.Close()

	// First register a dataset.
	regBody := `{
		"name": "Python DS",
		"language": "python",
		"file_count": 100,
		"total_size": 500000,
		"total_tokens": 100000,
		"status": "uploaded",
		"quality": {
			"overall_score": 85,
			"validity_rate": 0.95,
			"complexity_score": 5
		}
	}`
	req, _ := http.NewRequest("POST", srv.URL+"/api/v1/fine-tuning/datasets", strings.NewReader(regBody))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("X-User-ID", "user_1")

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatalf("register failed: %v", err)
	}

	if resp.StatusCode != http.StatusCreated {
		resp.Body.Close()
		t.Fatalf("expected 201 for register, got %d", resp.StatusCode)
	}

	var regData map[string]interface{}

	err = json.NewDecoder(resp.Body).Decode(&regData)
	if err != nil {
		resp.Body.Close()
		t.Fatalf("failed to decode register response: %v", err)
	}

	resp.Body.Close()

	datasetID, _ := regData["dataset_id"].(string)
	if datasetID == "" {
		t.Fatal("expected non-empty dataset_id from register")
	}

	// Now analyse it.
	resp2, err := http.Get(srv.URL + "/api/v1/fine-tuning/datasets/" + datasetID + "/analyse")
	if err != nil {
		t.Fatalf("analyse failed: %v", err)
	}
	defer resp2.Body.Close()

	if resp2.StatusCode != http.StatusOK {
		t.Errorf("expected status 200, got %d", resp2.StatusCode)
	}

	var analysis domain.DatasetAnalysis

	err = json.NewDecoder(resp2.Body).Decode(&analysis)
	if err != nil {
		t.Fatalf("failed to decode response: %v", err)
	}

	if !analysis.ReadyForTraining {
		t.Error("expected ready for training")
	}
}

func TestFineTuneHandler_AnalyseDataset_NotFound(t *testing.T) {
	t.Parallel()

	srv := newFinetuneTestServer(t)
	defer srv.Close()

	resp, err := http.Get(srv.URL + "/api/v1/fine-tuning/datasets/missing/analyse")
	if err != nil {
		t.Fatalf("request failed: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusNotFound {
		t.Errorf("expected status 404, got %d", resp.StatusCode)
	}
}

func TestFineTuneHandler_RecommendHyperparams_Valid(t *testing.T) {
	t.Parallel()

	srv := newFinetuneTestServer(t)
	defer srv.Close()

	body := `{
		"Language": "python",
		"Skill": "code_completion",
		"DatasetSize": 1000000,
		"AvailableGPU": 1
	}`
	req, _ := http.NewRequest("POST", srv.URL+"/api/v1/fine-tuning/hyperparameters", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatalf("request failed: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		t.Errorf("expected status 200, got %d", resp.StatusCode)
	}

	var params domain.TrainingParameters

	err = json.NewDecoder(resp.Body).Decode(&params)
	if err != nil {
		t.Fatalf("failed to decode response: %v", err)
	}

	if params.LearningRate == 0 {
		t.Error("expected non-zero learning rate")
	}

	if params.BatchSize == 0 {
		t.Error("expected non-zero batch size")
	}
}

func TestFineTuneHandler_ListRequests_Empty(t *testing.T) {
	t.Parallel()

	srv := newFinetuneTestServer(t)
	defer srv.Close()

	req, _ := http.NewRequest("GET", srv.URL+"/api/v1/fine-tuning/requests", http.NoBody)
	req.Header.Set("X-User-ID", "user_1")

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatalf("request failed: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		t.Errorf("expected status 200, got %d", resp.StatusCode)
	}

	var list []*domain.FineTuneRequest

	err = json.NewDecoder(resp.Body).Decode(&list)
	if err != nil {
		t.Fatalf("failed to decode response: %v", err)
	}

	if len(list) != 0 {
		t.Errorf("expected empty list, got %d items", len(list))
	}
}

func TestFineTuneHandler_ListJobs_Empty(t *testing.T) {
	t.Parallel()

	srv := newFinetuneTestServer(t)
	defer srv.Close()

	resp, err := http.Get(srv.URL + "/api/v1/fine-tuning/jobs")
	if err != nil {
		t.Fatalf("request failed: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		t.Errorf("expected status 200, got %d", resp.StatusCode)
	}

	var list []*domain.FineTuneJob

	err = json.NewDecoder(resp.Body).Decode(&list)
	if err != nil {
		t.Fatalf("failed to decode response: %v", err)
	}

	if len(list) != 0 {
		t.Errorf("expected empty list, got %d items", len(list))
	}
}

func TestFineTuneHandler_JobStatus_NotFound(t *testing.T) {
	t.Parallel()

	srv := newFinetuneTestServer(t)
	defer srv.Close()

	resp, err := http.Get(srv.URL + "/api/v1/fine-tuning/jobs/missing")
	if err != nil {
		t.Fatalf("request failed: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusNotFound {
		t.Errorf("expected status 404, got %d", resp.StatusCode)
	}
}

func TestFineTuneHandler_JobMonitor_NotFound(t *testing.T) {
	t.Parallel()

	srv := newFinetuneTestServer(t)
	defer srv.Close()

	resp, err := http.Get(srv.URL + "/api/v1/fine-tuning/jobs/missing/monitor")
	if err != nil {
		t.Fatalf("request failed: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusNotFound {
		t.Errorf("expected status 404, got %d", resp.StatusCode)
	}
}

func TestFineTuneHandler_JobInsights_NotFound(t *testing.T) {
	t.Parallel()

	srv := newFinetuneTestServer(t)
	defer srv.Close()

	resp, err := http.Get(srv.URL + "/api/v1/fine-tuning/jobs/missing/insights")
	if err != nil {
		t.Fatalf("request failed: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusNotFound {
		t.Errorf("expected status 404, got %d", resp.StatusCode)
	}
}

func TestFineTuneHandler_JobQuality_NotFound(t *testing.T) {
	t.Parallel()

	srv := newFinetuneTestServer(t)
	defer srv.Close()

	resp, err := http.Get(srv.URL + "/api/v1/fine-tuning/jobs/missing/quality")
	if err != nil {
		t.Fatalf("request failed: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusNotFound {
		t.Errorf("expected status 404, got %d", resp.StatusCode)
	}
}

func TestFineTuneHandler_StartTraining_NotFound(t *testing.T) {
	t.Parallel()

	srv := newFinetuneTestServer(t)
	defer srv.Close()

	req, _ := http.NewRequest("POST", srv.URL+"/api/v1/fine-tuning/requests/missing/start", http.NoBody)

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatalf("request failed: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusNotFound {
		t.Errorf("expected status 404, got %d", resp.StatusCode)
	}
}

func TestFineTuneHandler_ListModels_Empty(t *testing.T) {
	t.Parallel()

	srv := newFinetuneTestServer(t)
	defer srv.Close()

	resp, err := http.Get(srv.URL + "/api/v1/fine-tuning/models")
	if err != nil {
		t.Fatalf("request failed: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		t.Errorf("expected status 200, got %d", resp.StatusCode)
	}

	var list []*domain.TrainedModel

	err = json.NewDecoder(resp.Body).Decode(&list)
	if err != nil {
		t.Fatalf("failed to decode response: %v", err)
	}

	if len(list) != 0 {
		t.Errorf("expected empty list, got %d items", len(list))
	}
}

func TestFineTuneHandler_GetModel_NotFound(t *testing.T) {
	t.Parallel()

	srv := newFinetuneTestServer(t)
	defer srv.Close()

	resp, err := http.Get(srv.URL + "/api/v1/fine-tuning/models/missing")
	if err != nil {
		t.Fatalf("request failed: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusNotFound {
		t.Errorf("expected status 404, got %d", resp.StatusCode)
	}
}

func TestFineTuneHandler_ExportModel_NotFound(t *testing.T) {
	t.Parallel()

	srv := newFinetuneTestServer(t)
	defer srv.Close()

	req, _ := http.NewRequest("POST", srv.URL+"/api/v1/fine-tuning/models/missing/export", http.NoBody)

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatalf("request failed: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusNotFound {
		t.Errorf("expected status 404, got %d", resp.StatusCode)
	}
}
