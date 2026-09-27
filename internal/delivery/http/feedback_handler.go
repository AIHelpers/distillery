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

// Fold converts unresolved feedback for a task into new training examples,
// as either corrected SFT examples (default) or DPO/ORPO preference pairs,
// selected by the optional ?target=sft|preferences query parameter.
func (h *FeedbackHandler) Fold(w http.ResponseWriter, r *http.Request, taskID string) {
	target := usecase.FoldTarget(r.URL.Query().Get("target"))
	if target == "" {
		target = usecase.FoldTargetSFT
	}

	var (
		n   int
		err error
	)

	switch target {
	case usecase.FoldTargetSFT:
		n, err = h.uc.FoldIntoDataset(taskID)
	case usecase.FoldTargetPreferences:
		n, err = h.uc.FoldFeedbackAsPreferences(taskID)
	default:
		writeError(w, http.StatusBadRequest, "target must be \"sft\" or \"preferences\"")
		return
	}

	if err != nil {
		handleErr(w, err)
		return
	}

	key := "examples_added"
	if target == usecase.FoldTargetPreferences {
		key = "preferences_added"
	}

	writeJSON(w, http.StatusOK, map[string]int{key: n})
}
