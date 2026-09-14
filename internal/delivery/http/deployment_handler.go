package http

import (
	"bufio"
	"encoding/csv"
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

// InvokeBatch accepts a text body вЂ” either CSV with an "input" column, or
// plain newline-separated inputs вЂ” and returns CSV predictions. This is the
// bulk workload narrow-task deployments exist to serve efficiently.
func (h *DeploymentHandler) InvokeBatch(w http.ResponseWriter, r *http.Request, deploymentID string) {
	body, err := io.ReadAll(io.LimitReader(r.Body, 5<<20)) // 5MB cap.
	if err != nil {
		writeError(w, http.StatusBadRequest, "failed to read request body")
		return
	}

	inputs := parseBatchInputs(string(body))
	if len(inputs) == 0 {
		writeError(w, http.StatusBadRequest, "no inputs found вЂ” provide one per line, or a CSV with an \"input\" column")
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
