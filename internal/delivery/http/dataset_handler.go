package http

import (
	"encoding/base64"
	"io"
	"net/http"
	"strconv"
	"strings"

	"distillery/internal/usecase"
)

type DatasetHandler struct {
	uc *usecase.DatasetUsecase
}

func NewDatasetHandler(uc *usecase.DatasetUsecase) *DatasetHandler { return &DatasetHandler{uc: uc} }

func (h *DatasetHandler) AddExamples(w http.ResponseWriter, r *http.Request, taskID string) {
	var req addExamplesRequest

	err := decodeJSON(r, &req)
	if err != nil {
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

	err := decodeJSON(r, &req)
	if err != nil {
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

// ImportJSONL bulk-loads Alpaca-style or chat-style JSONL dataset records from
// an uploaded file's raw text body. An optional "format" query parameter
// ("alpaca" or "chat") forces the parser; otherwise it auto-detects from the
// first record.
func (h *DatasetHandler) ImportJSONL(w http.ResponseWriter, r *http.Request, taskID string) {
	body, err := io.ReadAll(io.LimitReader(r.Body, 20<<20)) // 20MB cap.
	if err != nil {
		writeError(w, http.StatusBadRequest, "failed to read request body")
		return
	}

	format := r.URL.Query().Get("format")

	stats, err := h.uc.ImportJSONL(taskID, string(body), format)
	if err != nil {
		handleErr(w, err)
		return
	}

	writeJSON(w, http.StatusOK, stats)
}

func (h *DatasetHandler) UpdateExample(w http.ResponseWriter, r *http.Request, taskID, exampleID string) {
	var req updateExampleRequest

	err := decodeJSON(r, &req)
	if err != nil {
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

func (h *DatasetHandler) DeleteExample(w http.ResponseWriter, _ *http.Request, taskID, exampleID string) {
	stats, err := h.uc.DeleteExample(taskID, exampleID)
	if err != nil {
		handleErr(w, err)
		return
	}

	writeJSON(w, http.StatusOK, stats)
}

// ImportRetrievalJSONL bulk-loads pair/triplet/graded JSONL records with
// query-group holdout splitting.
func (h *DatasetHandler) ImportRetrievalJSONL(w http.ResponseWriter, r *http.Request, taskID string) {
	body, err := io.ReadAll(io.LimitReader(r.Body, 20<<20)) // 20MB cap.
	if err != nil {
		writeError(w, http.StatusBadRequest, "failed to read request body")
		return
	}

	stats, err := h.uc.ImportRetrievalJSONL(taskID, string(body))
	if err != nil {
		handleErr(w, err)
		return
	}

	writeJSON(w, http.StatusOK, stats)
}

// ImportRetrievalCSV bulk-loads pair/triplet/graded CSV records with
// query-group holdout splitting.
func (h *DatasetHandler) ImportRetrievalCSV(w http.ResponseWriter, r *http.Request, taskID string) {
	body, err := io.ReadAll(io.LimitReader(r.Body, 5<<20)) // 5MB cap.
	if err != nil {
		writeError(w, http.StatusBadRequest, "failed to read request body")
		return
	}

	stats, err := h.uc.ImportRetrievalCSV(taskID, string(body))
	if err != nil {
		handleErr(w, err)
		return
	}

	writeJSON(w, http.StatusOK, stats)
}

// ImportNERJSONL bulk-loads span-annotated JSONL records:
//
//	{"text": "Acme paid $4,200 on 3 May.", "entities": [{"start":0,"end":4,"label":"ORG"}]}
func (h *DatasetHandler) ImportNERJSONL(w http.ResponseWriter, r *http.Request, taskID string) {
	body, err := io.ReadAll(io.LimitReader(r.Body, 20<<20)) // 20MB cap.
	if err != nil {
		writeError(w, http.StatusBadRequest, "failed to read request body")
		return
	}

	stats, err := h.uc.ImportNERJSONL(taskID, string(body))
	if err != nil {
		handleErr(w, err)
		return
	}

	writeJSON(w, http.StatusOK, stats)
}

// ImportCoNLL bulk-loads classic token-per-line BIO-tagged CoNLL content.
func (h *DatasetHandler) ImportCoNLL(w http.ResponseWriter, r *http.Request, taskID string) {
	body, err := io.ReadAll(io.LimitReader(r.Body, 20<<20)) // 20MB cap.
	if err != nil {
		writeError(w, http.StatusBadRequest, "failed to read request body")
		return
	}

	stats, err := h.uc.ImportCoNLL(taskID, string(body))
	if err != nil {
		handleErr(w, err)
		return
	}

	writeJSON(w, http.StatusOK, stats)
}

// ImportNERCSV bulk-loads span annotations from CSV (JSON-in-CSV, or flat
// text,start,end,label columns).
func (h *DatasetHandler) ImportNERCSV(w http.ResponseWriter, r *http.Request, taskID string) {
	body, err := io.ReadAll(io.LimitReader(r.Body, 5<<20)) // 5MB cap.
	if err != nil {
		writeError(w, http.StatusBadRequest, "failed to read request body")
		return
	}

	stats, err := h.uc.ImportNERCSV(taskID, string(body))
	if err != nil {
		handleErr(w, err)
		return
	}

	writeJSON(w, http.StatusOK, stats)
}

// GenerateQueriesFromDocs bootstraps pair examples from docs-only examples
// via the LLM adapter, tagging them as synthetic for human review.
func (h *DatasetHandler) GenerateQueriesFromDocs(w http.ResponseWriter, r *http.Request, taskID string) {
	var req generateSyntheticRequest

	err := decodeJSON(r, &req)
	if err != nil {
		writeError(w, http.StatusBadRequest, "invalid JSON body")
		return
	}

	if req.Count <= 0 {
		req.Count = 10
	}

	if req.Count > 200 {
		req.Count = 200
	}

	stats, err := h.uc.GenerateQueriesFromDocs(taskID, req.Count)
	if err != nil {
		handleErr(w, err)
		return
	}

	writeJSON(w, http.StatusOK, stats)
}

func (h *DatasetHandler) List(w http.ResponseWriter, _ *http.Request, taskID string) {
	examples, err := h.uc.ListExamples(taskID)
	if err != nil {
		handleErr(w, err)
		return
	}

	writeJSON(w, http.StatusOK, examples)
}

func (h *DatasetHandler) Stats(w http.ResponseWriter, _ *http.Request, taskID string) {
	stats, err := h.uc.Curate(taskID)
	if err != nil {
		handleErr(w, err)
		return
	}

	writeJSON(w, http.StatusOK, stats)
}

// AddPreference POST /api/v1/tasks/{taskID}/preferences adds a single
// (prompt, chosen, rejected) example to the task's preference dataset.
func (h *DatasetHandler) AddPreference(w http.ResponseWriter, r *http.Request, taskID string) {
	var req preferencePairRequest

	err := decodeJSON(r, &req)
	if err != nil {
		writeError(w, http.StatusBadRequest, "invalid JSON body")
		return
	}

	stats, err := h.uc.AddPreferencePair(taskID, req.Prompt, req.Chosen, req.Rejected)
	if err != nil {
		handleErr(w, err)
		return
	}

	writeJSON(w, http.StatusOK, stats)
}

// ImportPreferences POST /api/v1/tasks/{taskID}/preferences/import bulk-loads
// {"prompt","chosen","rejected"} JSONL records from the uploaded body.
func (h *DatasetHandler) ImportPreferences(w http.ResponseWriter, r *http.Request, taskID string) {
	body, err := io.ReadAll(io.LimitReader(r.Body, 20<<20)) // 20MB cap.
	if err != nil {
		writeError(w, http.StatusBadRequest, "failed to read request body")
		return
	}

	stats, err := h.uc.ImportPreferenceJSONL(taskID, string(body))
	if err != nil {
		handleErr(w, err)
		return
	}

	writeJSON(w, http.StatusOK, stats)
}

// PreferenceStats GET /api/v1/tasks/{taskID}/preferences/stats reports the
// task's preference dataset size and length-bias diagnostic.
func (h *DatasetHandler) PreferenceStats(w http.ResponseWriter, _ *http.Request, taskID string) {
	stats, err := h.uc.PreferenceStats(taskID)
	if err != nil {
		handleErr(w, err)
		return
	}

	writeJSON(w, http.StatusOK, stats)
}

// AddVisionExample POST /api/v1/tasks/{taskID}/examples/vision adds one
// image+prompt(+answer) example. The image is base64-encoded in the JSON
// body (optionally prefixed with a data URL header, e.g.
// "data:image/png;base64,...", which is stripped).
func (h *DatasetHandler) AddVisionExample(w http.ResponseWriter, r *http.Request, taskID string) {
	var req addVisionExampleRequest

	err := decodeJSON(r, &req)
	if err != nil {
		writeError(w, http.StatusBadRequest, "invalid JSON body")
		return
	}

	imgData, err := decodeImageBase64(req.ImageBase64)
	if err != nil {
		writeError(w, http.StatusBadRequest, "image_base64 is not valid base64 image data")
		return
	}

	stats, err := h.uc.AddVisionExample(taskID, imgData, req.Prompt, req.Answer)
	if err != nil {
		handleErr(w, err)
		return
	}

	writeJSON(w, http.StatusOK, stats)
}

// ImportVisionZIP POST /api/v1/tasks/{taskID}/examples/import-vision-zip
// bulk-loads a ZIP archive of page images plus a data.jsonl manifest (see
// docs/vision-language-document-ai.md for the manifest shape).
func (h *DatasetHandler) ImportVisionZIP(w http.ResponseWriter, r *http.Request, taskID string) {
	body, err := io.ReadAll(io.LimitReader(r.Body, 200<<20)) // 200MB cap.
	if err != nil {
		writeError(w, http.StatusBadRequest, "failed to read request body")
		return
	}

	stats, err := h.uc.ImportVisionZIP(taskID, body)
	if err != nil {
		handleErr(w, err)
		return
	}

	writeJSON(w, http.StatusOK, stats)
}

// ImportVisionPDF POST
// /api/v1/tasks/{taskID}/examples/import-vision-pdf?prompt=...&dpi=150
// rasterizes an uploaded PDF (raw body) into one image example per page.
func (h *DatasetHandler) ImportVisionPDF(w http.ResponseWriter, r *http.Request, taskID string) {
	body, err := io.ReadAll(io.LimitReader(r.Body, 50<<20)) // 50MB cap.
	if err != nil {
		writeError(w, http.StatusBadRequest, "failed to read request body")
		return
	}

	prompt := r.URL.Query().Get("prompt")
	dpi := 150

	v := r.URL.Query().Get("dpi")
	if v != "" {
		parsed, convErr := strconv.Atoi(v)
		if convErr == nil && parsed > 0 {
			dpi = parsed
		}
	}

	stats, err := h.uc.ImportPDF(taskID, body, prompt, dpi)
	if err != nil {
		handleErr(w, err)
		return
	}

	writeJSON(w, http.StatusOK, stats)
}

// GetBlob GET /api/v1/tasks/{taskID}/blobs/{key} serves a previously stored
// vision_lm image (or rasterized PDF page) so the dataset gallery and
// test-tab preview can render it directly in an <img> tag.
func (h *DatasetHandler) GetBlob(w http.ResponseWriter, _ *http.Request, taskID, key string) {
	data, contentType, err := h.uc.GetBlob(taskID, key)
	if err != nil {
		handleErr(w, err)
		return
	}

	w.Header().Set("Content-Type", contentType)
	w.Header().Set("Cache-Control", "private, max-age=3600")
	w.WriteHeader(http.StatusOK)
	_, _ = w.Write(data)
}

// decodeImageBase64 decodes a base64 image payload, stripping a leading
// data-URL header ("data:image/png;base64,...") when present.
func decodeImageBase64(s string) ([]byte, error) {
	idx := strings.Index(s, ",")
	if idx >= 0 && strings.HasPrefix(s, "data:") {
		s = s[idx+1:]
	}

	return base64.StdEncoding.DecodeString(strings.TrimSpace(s))
}
