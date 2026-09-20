package http

import (
	"bufio"
	"crypto/rand"
	"encoding/csv"
	"encoding/hex"
	"fmt"
	"io"
	"net/http"
	"strings"

	"distillery/internal/domain"
	"distillery/internal/usecase"
)

type DeploymentHandler struct {
	uc *usecase.DeploymentUsecase
}

func NewDeploymentHandler(uc *usecase.DeploymentUsecase) *DeploymentHandler {
	return &DeploymentHandler{uc: uc}
}

// deploymentResponse wraps a deployment with the one-time raw API key. The
// key is never retrievable again after this response.
type deploymentResponse struct {
	*domain.Deployment

	APIKey string `json:"api_key,omitempty"`
}

func (h *DeploymentHandler) Deploy(w http.ResponseWriter, r *http.Request, taskID string) {
	var req deployRequest

	_ = decodeJSON(r, &req) // autoscale defaults to false if omitted/absent.

	d, apiKey, err := h.uc.Deploy(taskID, req.Autoscale)
	if err != nil {
		handleErr(w, err)
		return
	}

	writeJSON(w, http.StatusCreated, deploymentResponse{Deployment: d, APIKey: apiKey})
}

// DeployVersion deploys a specific completed training job version, enabling
// model comparison and rollback to an earlier fine-tune.
func (h *DeploymentHandler) DeployVersion(w http.ResponseWriter, r *http.Request, taskID, jobID string) {
	var req deployRequest

	_ = decodeJSON(r, &req)

	d, apiKey, err := h.uc.DeployVersion(taskID, jobID, req.Autoscale)
	if err != nil {
		handleErr(w, err)
		return
	}

	writeJSON(w, http.StatusCreated, deploymentResponse{Deployment: d, APIKey: apiKey})
}

func (h *DeploymentHandler) List(w http.ResponseWriter, _ *http.Request, taskID string) {
	list, err := h.uc.ListDeployments(taskID)
	if err != nil {
		handleErr(w, err)
		return
	}

	writeJSON(w, http.StatusOK, list)
}

func (h *DeploymentHandler) Active(w http.ResponseWriter, _ *http.Request, taskID string) {
	d, err := h.uc.GetActiveDeployment(taskID)
	if err != nil {
		handleErr(w, err)
		return
	}

	writeJSON(w, http.StatusOK, d)
}

func (h *DeploymentHandler) Stop(w http.ResponseWriter, _ *http.Request, deploymentID string) {
	err := h.uc.StopDeployment(deploymentID)
	if err != nil {
		handleErr(w, err)
		return
	}

	w.WriteHeader(http.StatusNoContent)
}

func (h *DeploymentHandler) Invoke(w http.ResponseWriter, r *http.Request, deploymentID string) {
	var req invokeRequest

	err := decodeJSON(r, &req)
	if err != nil {
		writeError(w, http.StatusBadRequest, "invalid JSON body")
		return
	}

	output, confidence, err := h.uc.Invoke(deploymentID, apiKeyFromRequest(r), req.Input)
	if err != nil {
		handleErr(w, err)
		return
	}

	writeJSON(w, http.StatusOK, map[string]interface{}{
		"output":     output,
		"confidence": confidence,
	})
}

// InvokeBatch accepts a text body -- either CSV with an "input" column, or
// plain newline-separated inputs -- and returns CSV predictions. This is the
// bulk workload narrow-task deployments exist to serve efficiently.
func (h *DeploymentHandler) InvokeBatch(w http.ResponseWriter, r *http.Request, deploymentID string) {
	body, err := io.ReadAll(io.LimitReader(r.Body, 5<<20)) // 5MB cap.
	if err != nil {
		writeError(w, http.StatusBadRequest, "failed to read request body")
		return
	}

	inputs := parseBatchInputs(string(body))
	if len(inputs) == 0 {
		writeError(w, http.StatusBadRequest, "no inputs found -- provide one per line, or a CSV with an \"input\" column")
		return
	}

	results, err := h.uc.InvokeBatch(deploymentID, apiKeyFromRequest(r), inputs)
	if err != nil {
		handleErr(w, err)
		return
	}

	w.Header().Set("Content-Type", "text/csv")
	w.Header().Set("Content-Disposition", `attachment; filename="predictions.csv"`)
	cw := csv.NewWriter(w)

	_ = cw.Write([]string{"input", "output", "confidence"})
	for _, res := range results {
		_ = cw.Write([]string{res.Input, res.Output, fmt.Sprintf("%.2f", res.Confidence)})
	}

	cw.Flush()
}

func (h *DeploymentHandler) Export(w http.ResponseWriter, _ *http.Request, taskID string) {
	data, filename, err := h.uc.Export(taskID)
	if err != nil {
		handleErr(w, err)
		return
	}

	w.Header().Set("Content-Type", "application/zip")
	w.Header().Set("Content-Disposition", "attachment; filename=\""+filename+"\"")
	_, _ = w.Write(data)
}

// ExportGGUF downloads the task's latest completed model as a GGUF file
// (HomeBred-LLM / llama.cpp compatible).
func (h *DeploymentHandler) ExportGGUF(w http.ResponseWriter, r *http.Request, taskID string) {
	h.serveGGUF(w, taskID, "", ggufQuantization(r))
}

// ExportGGUFVersion downloads a specific training job version's GGUF file.
func (h *DeploymentHandler) ExportGGUFVersion(w http.ResponseWriter, r *http.Request, taskID, jobID string) {
	h.serveGGUF(w, taskID, jobID, ggufQuantization(r))
}

// ggufQuantization reads the ?quantization= query parameter, falling back to
// the default when absent or invalid.
func ggufQuantization(r *http.Request) string {
	if q := strings.TrimSpace(r.URL.Query().Get("quantization")); q != "" {
		// Validate against known quantizations; fall back to default on bad input.
		ok := false

		for _, valid := range []string{"q2_k", "q3_k_s", "q3_k_m", "q3_k_l", "q4_0", "q4_1", "q4_k_s", "q4_k_m", "q5_0", "q5_1", "q5_k_s", "q5_k_m", "q6_k", "q8_0", "f16", "f32"} {
			if strings.EqualFold(q, valid) {
				ok = true
				break
			}
		}

		if ok {
			return strings.ToLower(q)
		}
	}

	return defaultGGUFQuantization
}

// serveGGUF handles the common GGUF attachment response path. When jobID is
// empty, the latest completed job for the task is used.
func (h *DeploymentHandler) serveGGUF(w http.ResponseWriter, taskID, jobID, quantization string) {
	opts := domain.GGUFExportOptions{Quantization: quantization}

	var (
		data     []byte
		filename string
		err      error
	)

	if jobID == "" {
		data, filename, err = h.uc.ExportGGUF(taskID, opts)
	} else {
		data, filename, err = h.uc.ExportGGUFVersion(taskID, jobID, opts)
	}

	if err != nil {
		handleErr(w, err)
		return
	}

	w.Header().Set("Content-Type", "application/octet-stream")
	w.Header().Set("Content-Disposition", "attachment; filename=\""+filename+"\"")
	_, _ = w.Write(data)
}

// StartGGUFAsync starts an async GGUF conversion and returns the session ID
// for progress polling.
func (h *DeploymentHandler) StartGGUFAsync(w http.ResponseWriter, r *http.Request, taskID string) {
	sessionID := generateGGUFSessionID()
	quantization := ggufQuantization(r)

	opts := domain.GGUFExportOptions{Quantization: quantization}
	err := h.uc.StartGGUFAsync(taskID, "", sessionID, opts)
	if err != nil {
		handleErr(w, err)
		return
	}

	writeJSON(w, http.StatusAccepted, map[string]string{
		"session_id":      sessionID,
		"quantization":    quantization,
		"status_endpoint": fmt.Sprintf("/api/v1/tasks/%s/export/gguf/progress/%s", taskID, sessionID),
	})
}

// StartGGUFAsyncVersion starts an async GGUF conversion for a specific job version.
func (h *DeploymentHandler) StartGGUFAsyncVersion(w http.ResponseWriter, r *http.Request, taskID, jobID string) {
	sessionID := generateGGUFSessionID()
	quantization := ggufQuantization(r)

	opts := domain.GGUFExportOptions{Quantization: quantization}
	err := h.uc.StartGGUFAsync(taskID, jobID, sessionID, opts)
	if err != nil {
		handleErr(w, err)
		return
	}

	writeJSON(w, http.StatusAccepted, map[string]string{
		"session_id":      sessionID,
		"quantization":    quantization,
		"status_endpoint": fmt.Sprintf("/api/v1/tasks/%s/export/gguf/progress/%s", taskID, sessionID),
	})
}

// GetGGUFProgress returns the current progress of an async GGUF conversion.
func (h *DeploymentHandler) GetGGUFProgress(w http.ResponseWriter, _ *http.Request, sessionID string) {
	p, err := h.uc.GetGGUFProgress(sessionID)
	if err != nil {
		handleErr(w, err)
		return
	}

	writeJSON(w, http.StatusOK, p)
}

// DownloadGGUF downloads the completed GGUF file for a ready session.
// The file is streamed directly from disk via http.ServeFile -- it is never
// loaded into memory, supporting multi-GB GGUF files without OOM.
// After the file has been fully streamed, the GGUF file and session are
// cleaned up so large model files don't accumulate on disk.
func (h *DeploymentHandler) DownloadGGUF(w http.ResponseWriter, r *http.Request, sessionID string) {
	filePath, filename, err := h.uc.GetGGUFResult(sessionID)
	if err != nil {
		handleErr(w, err)
		return
	}

	w.Header().Set("Content-Disposition", "attachment; filename=\""+filename+"\"")
	http.ServeFile(w, r, filePath)

	// Clean up the GGUF file and session now that it has been streamed.
	h.uc.CleanupGGUFSession(sessionID)
}

// generateGGUFSessionID generates a unique session ID for a GGUF conversion.
func generateGGUFSessionID() string {
	b := make([]byte, 12)
	_, _ = rand.Read(b)

	return "gguf_" + hex.EncodeToString(b)
}

// defaultGGUFQuantization is the default quantization applied when the
// client does not select one explicitly.
const defaultGGUFQuantization = "q4_k_m"

// apiKeyFromRequest reads the deployment API key from either
// "Authorization: Bearer <key>" or "X-API-Key: <key>".
func apiKeyFromRequest(r *http.Request) string {
	if auth := r.Header.Get("Authorization"); auth != "" {
		if strings.HasPrefix(strings.ToLower(auth), "bearer ") {
			return strings.TrimSpace(auth[7:])
		}

		return strings.TrimSpace(auth)
	}

	return strings.TrimSpace(r.Header.Get("X-API-Key"))
}

// parseBatchInputs tries CSV-with-"input"-column first, then falls back to
// treating every non-empty line as one input.
func parseBatchInputs(content string) []string {
	content = strings.TrimSpace(content)
	if content == "" {
		return nil
	}

	reader := csv.NewReader(strings.NewReader(content))
	reader.FieldsPerRecord = -1

	records, err := reader.ReadAll()
	if err == nil && len(records) > 0 && len(records[0]) > 0 {
		inputIdx := -1

		for i, col := range records[0] {
			if strings.EqualFold(strings.TrimSpace(col), "input") {
				inputIdx = i
				break
			}
		}

		if inputIdx >= 0 {
			var out []string

			for _, rec := range records[1:] {
				if len(rec) > inputIdx && strings.TrimSpace(rec[inputIdx]) != "" {
					out = append(out, strings.TrimSpace(rec[inputIdx]))
				}
			}

			if len(out) > 0 {
				return out
			}
		}
	}

	var lines []string

	sc := bufio.NewScanner(strings.NewReader(content))
	sc.Buffer(make([]byte, 1024*1024), 1024*1024)

	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		if line != "" {
			lines = append(lines, line)
		}
	}

	return lines
}
