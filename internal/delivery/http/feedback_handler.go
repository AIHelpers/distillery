package http

import (
	"net/http"

	"distillery/internal/usecase"
)

type FeedbackHandler struct {
	uc *usecase.FeedbackUsecase
}

func NewFeedbackHandler(uc *usecase.FeedbackUsecase) *FeedbackHandler {
	return &FeedbackHandler{uc: uc}
}

func (h *FeedbackHandler) Submit(w http.ResponseWriter, r *http.Request, taskID string) {
	var req mispredictionRequest

	err := decodeJSON(r, &req)
	if err != nil {
		writeError(w, http.StatusBadRequest, "invalid JSON body")
		return
	}

	m, err := h.uc.SubmitMisprediction(taskID, req.Input, req.ActualOutput, req.ExpectedOutput)
	if err != nil {
		handleErr(w, err)
		return
	}

	writeJSON(w, http.StatusCreated, m)
}

func (h *FeedbackHandler) List(w http.ResponseWriter, _ *http.Request, taskID string) {
	list, err := h.uc.ListFeedback(taskID)
	if err != nil {
		handleErr(w, err)
		return
	}

	writeJSON(w, http.StatusOK, list)
}

func (h *FeedbackHandler) Fold(w http.ResponseWriter, _ *http.Request, taskID string) {
	n, err := h.uc.FoldIntoDataset(taskID)
	if err != nil {
		handleErr(w, err)
		return
	}

	writeJSON(w, http.StatusOK, map[string]int{"examples_added": n})
}
