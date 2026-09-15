package http

import (
	"encoding/json"
	"net/http"

	"distillery/internal/domain"
	"distillery/internal/usecase"
)

type TrainingHandler struct {
	uc *usecase.TrainingUsecase
}

func NewTrainingHandler(uc *usecase.TrainingUsecase) *TrainingHandler {
	return &TrainingHandler{uc: uc}
}

// startTrainingRequest carries an optional user-chosen base model.
type startTrainingRequest struct {
	BaseModel string `json:"base_model"`
}

// Start POST /api/v1/tasks/{taskID}/training
// Body (optional): {"base_model": "Llama-3.2-1B-Instruct"}.
func (h *TrainingHandler) Start(w http.ResponseWriter, r *http.Request, taskID string) {
	var req startTrainingRequest
	if r.Body != nil {
		_ = json.NewDecoder(r.Body).Decode(&req)
	}

	job, err := h.uc.StartTraining(taskID, req.BaseModel)
	if err != nil {
		handleErr(w, err)
		return
	}

	writeJSON(w, http.StatusAccepted, job)
}

// Models GET /api/v1/models returns the curated base-model catalog the
// user can pick from before starting a fine-tuning run.
func (h *TrainingHandler) Models(w http.ResponseWriter, _ *http.Request) {
	models := h.uc.ListBaseModels()
	if models == nil {
		models = []domain.BaseModel{}
	}

	writeJSON(w, http.StatusOK, models)
}

func (h *TrainingHandler) List(w http.ResponseWriter, _ *http.Request, taskID string) {
	jobs, err := h.uc.ListJobs(taskID)
	if err != nil {
		handleErr(w, err)
		return
	}

	writeJSON(w, http.StatusOK, jobs)
}

func (h *TrainingHandler) Get(w http.ResponseWriter, _ *http.Request, jobID string) {
	job, err := h.uc.GetJob(jobID)
	if err != nil {
		handleErr(w, err)
		return
	}

	writeJSON(w, http.StatusOK, job)
}
