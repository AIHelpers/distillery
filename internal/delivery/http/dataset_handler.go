package http

import (
	"io"
	"net/http"

	"distillery/internal/usecase"
)

type DatasetHandler struct {
	uc *usecase.DatasetUsecase
}

func NewDatasetHandler(uc *usecase.DatasetUsecase) *DatasetHandler { return &DatasetHandler{uc: uc} }

func (h *DatasetHandler) AddExamples(w http.ResponseWriter, r *http.Request, taskID string) {
	var req addExamplesRequest
	if err := decodeJSON(r, &req); err != nil {
		writeError(w, http.StatusBadRequest, "invalid JSON body")
		return
	}
	pairs := make([]usecase.ExamplePair, 0, len(req.Pairs))
	for _, p := range req.Pairs {
		pairs = append(pairs, usecase.ExamplePair{Input: p.Input, Output: p.Output})
	}
	stats, err := h.uc.AddExamples(taskID, pairs)
	if err != nil {
		handleErr(w, err)
		return
	}
	writeJSON(w, http.StatusOK, stats)
}

func (h *DatasetHandler) GenerateSynthetic(w http.ResponseWriter, r *http.Request, taskID string) {
	var req generateSyntheticRequest
	if err := decodeJSON(r, &req); err != nil {
		writeError(w, http.StatusBadRequest, "invalid JSON body")
		return
	}
	if req.Count <= 0 {
		req.Count = 10
	}
	if req.Count > 200 {
		req.Count = 200
	}
	stats, err := h.uc.GenerateSynthetic(taskID, req.Count)
	if err != nil {
		handleErr(w, err)
		return
	}
	writeJSON(w, http.StatusOK, stats)
}

// ImportCSV bulk-loads example pairs from an uploaded CSV file's raw text body.
func (h *DatasetHandler) ImportCSV(w http.ResponseWriter, r *http.Request, taskID string) {
	body, err := io.ReadAll(io.LimitReader(r.Body, 5<<20)) // 5MB cap.
	if err != nil {
		writeError(w, http.StatusBadRequest, "failed to read request body")
		return
	}
	stats, err := h.uc.ImportCSV(taskID, string(body))
	if err != nil {
		handleErr(w, err)
		return
	}
	writeJSON(w, http.StatusOK, stats)
}

func (h *DatasetHandler) UpdateExample(w http.ResponseWriter, r *http.Request, taskID, exampleID string) {
	var req updateExampleRequest
	if err := decodeJSON(r, &req); err != nil {
		writeError(w, http.StatusBadRequest, "invalid JSON body")
		return
	}
	stats, err := h.uc.UpdateExample(taskID, exampleID, req.Input, req.Output)
	if err != nil {
		handleErr(w, err)
		return
	}
	writeJSON(w, http.StatusOK, stats)
}

func (h *DatasetHandler) DeleteExample(w http.ResponseWriter, r *http.Request, taskID, exampleID string) {
	stats, err := h.uc.DeleteExample(taskID, exampleID)
	if err != nil {
		handleErr(w, err)
		return
	}
	writeJSON(w, http.StatusOK, stats)
}

func (h *DatasetHandler) List(w http.ResponseWriter, r *http.Request, taskID string) {
	examples, err := h.uc.ListExamples(taskID)
	if err != nil {
		handleErr(w, err)
		return
	}
	writeJSON(w, http.StatusOK, examples)
}

func (h *DatasetHandler) Stats(w http.ResponseWriter, r *http.Request, taskID string) {
	stats, err := h.uc.Curate(taskID)
	if err != nil {
		handleErr(w, err)
		return
	}
	writeJSON(w, http.StatusOK, stats)
}
