package http

import (
	"net/http"

	"distillery/internal/usecase"
)

type TrainingHandler struct {
	uc *usecase.TrainingUsecase
}

func NewTrainingHandler(uc *usecase.TrainingUsecase) *TrainingHandler {
	return &TrainingHandler{uc: uc}
}

func (h *TrainingHandler) Start(w http.ResponseWriter, r *http.Request, taskID string) {
	job, err := h.uc.StartTraining(taskID)
	if err != nil {
		handleErr(w, err)
		return
	}
	writeJSON(w, http.StatusAccepted, job)
}

func (h *TrainingHandler) List(w http.ResponseWriter, r *http.Request, taskID string) {
	jobs, err := h.uc.ListJobs(taskID)
	if err != nil {
		handleErr(w, err)
		return
	}
	writeJSON(w, http.StatusOK, jobs)
}

func (h *TrainingHandler) Get(w http.ResponseWriter, r *http.Request, jobID string) {
	job, err := h.uc.GetJob(jobID)
	if err != nil {
		handleErr(w, err)
		return
	}
	writeJSON(w, http.StatusOK, job)
}
