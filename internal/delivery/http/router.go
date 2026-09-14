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
}

// NewRouter builds the full HTTP handler: the JSON API under /api/v1 plus
// the embedded single-page web UI served at /.
func NewRouter(h Handlers, webFS fs.FS) http.Handler {
	mux := http.NewServeMux()

	// --- Tasks ---.
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

	// --- Deployment / inference / export ---.
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

	// --- Feedback / continuous improvement ---.
	mux.HandleFunc("POST /api/v1/tasks/{taskID}/feedback", func(w http.ResponseWriter, r *http.Request) {
		h.Feedback.Submit(w, r, r.PathValue("taskID"))
	})
	mux.HandleFunc("GET /api/v1/tasks/{taskID}/feedback", func(w http.ResponseWriter, r *http.Request) {
		h.Feedback.List(w, r, r.PathValue("taskID"))
	})
	mux.HandleFunc("POST /api/v1/tasks/{taskID}/feedback/fold", func(w http.ResponseWriter, r *http.Request) {
		h.Feedback.Fold(w, r, r.PathValue("taskID"))
	})

	// --- Agents ---.
	if h.Agent != nil {
		mux.HandleFunc("POST /api/v1/agents/fine-tuning", h.Agent.StartFineTuningAgent)
		mux.HandleFunc("GET /api/v1/agents", h.Agent.ListAgentStates)
		mux.HandleFunc("GET /api/v1/agents/{agentID}", h.Agent.GetAgentState)
		mux.HandleFunc("POST /api/v1/agents/{agentID}/pause", h.Agent.PauseAgent)
		mux.HandleFunc("POST /api/v1/agents/{agentID}/resume", h.Agent.ResumeAgent)
	}

	// --- Fine-tuning / Coding AI Agent ---.
	if h.FineTune != nil {
		mux.HandleFunc("POST /api/v1/fine-tuning/datasets", h.FineTune.RegisterDataset)
		mux.HandleFunc("GET /api/v1/fine-tuning/datasets/{datasetID}/analyse", func(w http.ResponseWriter, r *http.Request) {
			h.FineTune.AnalyseDataset(w, r, r.PathValue("datasetID"))
		})
		mux.HandleFunc("POST /api/v1/fine-tuning/hyperparameters", h.FineTune.RecommendHyperparams)
		mux.HandleFunc("POST /api/v1/fine-tuning/requests", h.FineTune.CreateRequest)
		mux.HandleFunc("GET /api/v1/fine-tuning/requests", h.FineTune.ListRequests)
		mux.HandleFunc("POST /api/v1/fine-tuning/requests/{requestID}/start", func(w http.ResponseWriter, r *http.Request) {
			h.FineTune.StartTraining(w, r, r.PathValue("requestID"))
		})
		mux.HandleFunc("GET /api/v1/fine-tuning/jobs", h.FineTune.ListJobs)
		mux.HandleFunc("GET /api/v1/fine-tuning/jobs/{jobID}", func(w http.ResponseWriter, r *http.Request) {
			h.FineTune.JobStatus(w, r, r.PathValue("jobID"))
		})
		mux.HandleFunc("GET /api/v1/fine-tuning/jobs/{jobID}/monitor", func(w http.ResponseWriter, r *http.Request) {
			h.FineTune.JobMonitor(w, r, r.PathValue("jobID"))
		})
		mux.HandleFunc("GET /api/v1/fine-tuning/jobs/{jobID}/insights", func(w http.ResponseWriter, r *http.Request) {
			h.FineTune.JobInsights(w, r, r.PathValue("jobID"))
		})
		mux.HandleFunc("GET /api/v1/fine-tuning/jobs/{jobID}/quality", func(w http.ResponseWriter, r *http.Request) {
			h.FineTune.JobQuality(w, r, r.PathValue("jobID"))
		})
		mux.HandleFunc("GET /api/v1/fine-tuning/models", h.FineTune.ListModels)
		mux.HandleFunc("GET /api/v1/fine-tuning/models/{modelID}", func(w http.ResponseWriter, r *http.Request) {
			h.FineTune.GetModel(w, r, r.PathValue("modelID"))
		})
		mux.HandleFunc("POST /api/v1/fine-tuning/models/{modelID}/export", func(w http.ResponseWriter, r *http.Request) {
			h.FineTune.ExportModel(w, r, r.PathValue("modelID"))
		})
	}

	// --- Health check (useful for container orchestrators) ---.
	mux.HandleFunc("GET /healthz", func(w http.ResponseWriter, r *http.Request) {
		writeJSON(w, http.StatusOK, map[string]string{"status": "ok"})
	})

	// --- Static web UI ---.
	mux.Handle("/", http.FileServer(http.FS(webFS)))

	return withLogging(mux)
}

func withLogging(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()
		next.ServeHTTP(w, r)
		log.Printf("%s %s %s", r.Method, r.URL.Path, time.Since(start))
	})
}
