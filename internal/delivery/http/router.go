package http

import (
	"io/fs"
	"log"
	"net/http"
	"time"
)

// Handlers bundles all delivery handlers needed to build the router.
type Handlers struct {
	Task       *TaskHandler
	Dataset    *DatasetHandler
	Training   *TrainingHandler
	Deployment *DeploymentHandler
	Feedback   *FeedbackHandler
	Agent      *AgentHandler
	FineTune   *FineTuneHandler
	ModelStore *ModelStoreHandler
}

// NewRouter builds the full HTTP handler: the JSON API under /api/v1 plus
// the embedded single-page web UI served at /.
func NewRouter(h Handlers, webFS fs.FS) http.Handler {
	mux := http.NewServeMux()

	registerTaskRoutes(mux, h)
	registerDeploymentRoutes(mux, h)
	registerModelStoreRoutes(mux, h)
	registerFeedbackRoutes(mux, h)
	registerAgentRoutes(mux, h)
	registerFineTuneRoutes(mux, h.FineTune)

	// --- Health check (useful for container orchestrators) ---.
	mux.HandleFunc("GET /healthz", func(w http.ResponseWriter, _ *http.Request) {
		writeJSON(w, http.StatusOK, map[string]string{"status": "ok"})
	})

	// --- Static web UI ---.
	mux.Handle("/", http.FileServer(http.FS(webFS)))

	return withLogging(mux)
}

func registerTaskRoutes(mux *http.ServeMux, h Handlers) {
	mux.HandleFunc("POST /api/v1/tasks", h.Task.Create)
	mux.HandleFunc("GET /api/v1/tasks", h.Task.List)
	mux.HandleFunc("GET /api/v1/tasks/{taskID}", func(w http.ResponseWriter, r *http.Request) {
		h.Task.Get(w, r, r.PathValue("taskID"))
	})
	mux.HandleFunc("DELETE /api/v1/tasks/{taskID}", func(w http.ResponseWriter, r *http.Request) {
		h.Task.Delete(w, r, r.PathValue("taskID"))
	})

	// --- Dataset ---.
	mux.HandleFunc("POST /api/v1/tasks/{taskID}/examples", func(w http.ResponseWriter, r *http.Request) {
		h.Dataset.AddExamples(w, r, r.PathValue("taskID"))
	})
	mux.HandleFunc("GET /api/v1/tasks/{taskID}/examples", func(w http.ResponseWriter, r *http.Request) {
		h.Dataset.List(w, r, r.PathValue("taskID"))
	})
	mux.HandleFunc("POST /api/v1/tasks/{taskID}/examples/synthetic", func(w http.ResponseWriter, r *http.Request) {
		h.Dataset.GenerateSynthetic(w, r, r.PathValue("taskID"))
	})
	mux.HandleFunc("POST /api/v1/tasks/{taskID}/examples/import", func(w http.ResponseWriter, r *http.Request) {
		h.Dataset.ImportCSV(w, r, r.PathValue("taskID"))
	})
	mux.HandleFunc("POST /api/v1/tasks/{taskID}/examples/import-jsonl", func(w http.ResponseWriter, r *http.Request) {
		h.Dataset.ImportJSONL(w, r, r.PathValue("taskID"))
	})
	mux.HandleFunc("PUT /api/v1/tasks/{taskID}/examples/{exampleID}", func(w http.ResponseWriter, r *http.Request) {
		h.Dataset.UpdateExample(w, r, r.PathValue("taskID"), r.PathValue("exampleID"))
	})
	mux.HandleFunc("DELETE /api/v1/tasks/{taskID}/examples/{exampleID}", func(w http.ResponseWriter, r *http.Request) {
		h.Dataset.DeleteExample(w, r, r.PathValue("taskID"), r.PathValue("exampleID"))
	})
	mux.HandleFunc("GET /api/v1/tasks/{taskID}/dataset/stats", func(w http.ResponseWriter, r *http.Request) {
		h.Dataset.Stats(w, r, r.PathValue("taskID"))
	})

	// --- Training ---.
	mux.HandleFunc("POST /api/v1/tasks/{taskID}/training", func(w http.ResponseWriter, r *http.Request) {
		h.Training.Start(w, r, r.PathValue("taskID"))
	})
	mux.HandleFunc("GET /api/v1/tasks/{taskID}/training", func(w http.ResponseWriter, r *http.Request) {
		h.Training.List(w, r, r.PathValue("taskID"))
	})
	mux.HandleFunc("GET /api/v1/training/{jobID}", func(w http.ResponseWriter, r *http.Request) {
		h.Training.Get(w, r, r.PathValue("jobID"))
	})
	// List the base-model catalog the user can pick from for fine-tuning.
	mux.HandleFunc("GET /api/v1/models", h.Training.Models)
}

func registerDeploymentRoutes(mux *http.ServeMux, h Handlers) {
	mux.HandleFunc("POST /api/v1/tasks/{taskID}/deploy", func(w http.ResponseWriter, r *http.Request) {
		h.Deployment.Deploy(w, r, r.PathValue("taskID"))
	})
	mux.HandleFunc("POST /api/v1/tasks/{taskID}/training/{jobID}/deploy", func(w http.ResponseWriter, r *http.Request) {
		h.Deployment.DeployVersion(w, r, r.PathValue("taskID"), r.PathValue("jobID"))
	})
	mux.HandleFunc("GET /api/v1/tasks/{taskID}/deployments", func(w http.ResponseWriter, r *http.Request) {
		h.Deployment.List(w, r, r.PathValue("taskID"))
	})
	mux.HandleFunc("GET /api/v1/tasks/{taskID}/deployments/active", func(w http.ResponseWriter, r *http.Request) {
		h.Deployment.Active(w, r, r.PathValue("taskID"))
	})
	mux.HandleFunc("POST /api/v1/deployments/{deploymentID}/stop", func(w http.ResponseWriter, r *http.Request) {
		h.Deployment.Stop(w, r, r.PathValue("deploymentID"))
	})
	mux.HandleFunc("POST /api/v1/inference/{deploymentID}/predict", func(w http.ResponseWriter, r *http.Request) {
		h.Deployment.Invoke(w, r, r.PathValue("deploymentID"))
	})
	mux.HandleFunc("POST /api/v1/inference/{deploymentID}/batch", func(w http.ResponseWriter, r *http.Request) {
		h.Deployment.InvokeBatch(w, r, r.PathValue("deploymentID"))
	})
	mux.HandleFunc("GET /api/v1/tasks/{taskID}/export", func(w http.ResponseWriter, r *http.Request) {
		h.Deployment.Export(w, r, r.PathValue("taskID"))
	})
}

func registerModelStoreRoutes(mux *http.ServeMux, h Handlers) {
	if h.ModelStore == nil {
		return
	}

	mux.HandleFunc("POST /api/v1/model-stores", h.ModelStore.CreateStore)
	mux.HandleFunc("GET /api/v1/model-stores", h.ModelStore.ListStores)
	mux.HandleFunc("GET /api/v1/model-stores/{storeID}", func(w http.ResponseWriter, r *http.Request) {
		h.ModelStore.GetStore(w, r, r.PathValue("storeID"))
	})
	mux.HandleFunc("PATCH /api/v1/model-stores/{storeID}", func(w http.ResponseWriter, r *http.Request) {
		h.ModelStore.UpdateStore(w, r, r.PathValue("storeID"))
	})
	mux.HandleFunc("DELETE /api/v1/model-stores/{storeID}", func(w http.ResponseWriter, r *http.Request) {
		h.ModelStore.DeleteStore(w, r, r.PathValue("storeID"))
	})

	// Base models.
	mux.HandleFunc("GET /api/v1/model-stores/{storeID}/models", func(w http.ResponseWriter, r *http.Request) {
		h.ModelStore.ListModels(w, r, r.PathValue("storeID"))
	})
	mux.HandleFunc("POST /api/v1/model-stores/{storeID}/models/download", func(w http.ResponseWriter, r *http.Request) {
		h.ModelStore.DownloadModel(w, r, r.PathValue("storeID"))
	})
	mux.HandleFunc("POST /api/v1/model-stores/{storeID}/models/upload", func(w http.ResponseWriter, r *http.Request) {
		h.ModelStore.UploadBaseModel(w, r, r.PathValue("storeID"))
	})

	// Trained models.
	mux.HandleFunc("GET /api/v1/model-stores/{storeID}/trained", func(w http.ResponseWriter, r *http.Request) {
		h.ModelStore.ListTrainedModels(w, r, r.PathValue("storeID"))
	})
	mux.HandleFunc("POST /api/v1/model-stores/{storeID}/trained/upload", func(w http.ResponseWriter, r *http.Request) {
		h.ModelStore.UploadTrainedModel(w, r, r.PathValue("storeID"))
	})
}

func registerFeedbackRoutes(mux *http.ServeMux, h Handlers) {
	mux.HandleFunc("POST /api/v1/tasks/{taskID}/feedback", func(w http.ResponseWriter, r *http.Request) {
		h.Feedback.Submit(w, r, r.PathValue("taskID"))
	})
	mux.HandleFunc("GET /api/v1/tasks/{taskID}/feedback", func(w http.ResponseWriter, r *http.Request) {
		h.Feedback.List(w, r, r.PathValue("taskID"))
	})
	mux.HandleFunc("POST /api/v1/tasks/{taskID}/feedback/fold", func(w http.ResponseWriter, r *http.Request) {
		h.Feedback.Fold(w, r, r.PathValue("taskID"))
	})
}

func registerAgentRoutes(mux *http.ServeMux, h Handlers) {
	if h.Agent == nil {
		return
	}

	mux.HandleFunc("POST /api/v1/agents/fine-tuning", h.Agent.StartFineTuningAgent)
	mux.HandleFunc("GET /api/v1/agents", h.Agent.ListAgentStates)
	mux.HandleFunc("GET /api/v1/agents/{agentID}", h.Agent.GetAgentState)
	mux.HandleFunc("POST /api/v1/agents/{agentID}/pause", h.Agent.PauseAgent)
	mux.HandleFunc("POST /api/v1/agents/{agentID}/resume", h.Agent.ResumeAgent)
}

func registerFineTuneRoutes(mux *http.ServeMux, h *FineTuneHandler) {
	if h == nil {
		return
	}

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
}

func withLogging(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()

		next.ServeHTTP(w, r)
		log.Printf("%s %s %s", r.Method, r.URL.Path, time.Since(start))
	})
}
