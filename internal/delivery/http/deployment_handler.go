package http

import (
	"bufio"
	"crypto/rand"
	"encoding/base64"
	"encoding/csv"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"mime/multipart"
	"net/http"
	"strconv"
	"strings"

	"distillery/internal/domain"
	"distillery/internal/usecase"
)

// validGGUFQuantizations lists the quantization schemes the GGUF converter
// accepts, matching llama.cpp's naming.
var validGGUFQuantizations = []string{
	"q2_k", "q3_k_s", "q3_k_m", "q3_k_l", "q4_0", "q4_1", "q4_k_s", "q4_k_m",
	"q5_0", "q5_1", "q5_k_s", "q5_k_m", "q6_k", "q8_0", "f16", "f32",
}

type DeploymentHandler struct {
	uc      *usecase.DeploymentUsecase
	tabular *TabularHandler
}

// NewDeploymentHandler builds the deployment handler.
func NewDeploymentHandler(uc *usecase.DeploymentUsecase) *DeploymentHandler {
	return &DeploymentHandler{uc: uc}
}

// WithTabular routes tabular / time_series deployments arriving on the
// generic /predict and /batch endpoints to the table handler.
func (h *DeploymentHandler) WithTabular(t *TabularHandler) *DeploymentHandler {
	h.tabular = t

	return h
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

	d, apiKey, err := h.uc.Deploy(taskID, req.Autoscale, req.Force)
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

	d, apiKey, err := h.uc.DeployVersion(taskID, jobID, req.Autoscale, req.Force)
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
	// Kind-aware dispatch: seq_classifier and token_classifier deployments
	// return structured results, vision_lm accepts a multipart image instead
	// of a JSON body, everything else returns text + confidence.
	kind, err := h.uc.DeploymentKind(deploymentID)
	if err != nil {
		handleErr(w, err)
		return
	}

	// Tabular and time-series deployments have their own JSON contracts.
	if h.tabular != nil {
		switch kind { //nolint:exhaustive // other kinds intentionally fall through to the default
		case domain.KindTabular:
			h.tabular.Predict(w, r, deploymentID)

			return
		case domain.KindTimeSeries:
			h.tabular.Forecast(w, r, deploymentID)

			return
		}
	}

	// Vision deployments are served over multipart (per the plan's
	// `POST /predict (multipart: file=@invoice.png, prompt="...")`
	// contract) rather than the JSON body every other kind uses here.
	if kind == domain.KindVisionLM {
		h.invokeVision(w, r, deploymentID)
		return
	}

	var req invokeRequest

	err = decodeJSON(r, &req)
	if err != nil {
		writeError(w, http.StatusBadRequest, "invalid JSON body")
		return
	}

	switch kind {
	case domain.KindSeqClassifier:
		pred, err := h.uc.InvokeClassify(deploymentID, apiKeyFromRequest(r), req.Input)
		if err != nil {
			handleErr(w, err)
			return
		}

		writeJSON(w, http.StatusOK, map[string]interface{}{
			"kind":   "seq_classifier",
			"result": pred,
		})

		return

	case domain.KindTokenClassifier:
		pred, err := h.uc.InvokeNER(deploymentID, apiKeyFromRequest(r), req.Input)
		if err != nil {
			handleErr(w, err)
			return
		}

		writeJSON(w, http.StatusOK, map[string]interface{}{
			"kind":   "token_classifier",
			"result": pred,
		})

		return

	default:
		// causal_lm, embedding, reranker, preference_lm: all handled by the
		// shared "structured" path below (vision_lm was already dispatched
		// to invokeVision earlier in this function).
	}

	// Track B: for causal_lm/extraction deployments the task may declare a
	// JSON schema; InvokeStructured generates a GBNF grammar from it and
	// constrains decoding so served output is schema-valid. It degrades to
	// the plain path when no schema/engine support is present.
	output, confidence, constrained, err := h.uc.InvokeStructured(deploymentID, apiKeyFromRequest(r), req.Input)
	if err != nil {
		handleErr(w, err)
		return
	}

	resp := map[string]interface{}{
		"output":     output,
		"confidence": confidence,
	}

	if constrained {
		resp["constrained"] = true
		resp["json_valid"] = h.uc.ValidateOutput(deploymentID, output)
	}

	writeJSON(w, http.StatusOK, resp)
}

// maxVisionUploadBytes caps a single /predict image upload (20MB).
const maxVisionUploadBytes = 20 << 20

// maxASRAudioUploadBytes caps a single /transcribe audio upload (50MB,
// comfortably above a 10-minute 16 kHz mono WAV).
const maxASRAudioUploadBytes = 50 << 20

// Transcribe serves a deployed asr model: POST /inference/{id}/transcribe.
// Per plan 07 the primary contract is multipart (`file=@call.wav`, plus
// optional `language`); a JSON body with base64-encoded audio
// ({"audio_base64": "...", "filename": "...", "language": "en"}) is also
// accepted for programmatic clients. The response shape is
// {"kind":"asr","result":{"text":...,"segments":[...],"language":...}}.
func (h *DeploymentHandler) Transcribe(w http.ResponseWriter, r *http.Request, deploymentID string) {
	kind, err := h.uc.DeploymentKind(deploymentID)
	if err != nil {
		handleErr(w, err)
		return
	}

	if kind != domain.KindASR {
		writeError(w, http.StatusBadRequest, "transcribe is only available for asr deployments")
		return
	}

	contentType := r.Header.Get("Content-Type")

	// Multipart path (the plan's documented contract).
	if strings.HasPrefix(contentType, "multipart/") {
		h.transcribeMultipart(w, r, deploymentID)
		return
	}

	// JSON base64 path.
	var req transcribeRequest

	err = decodeJSON(r, &req)
	if err != nil {
		writeError(w, http.StatusBadRequest, "invalid JSON body")
		return
	}

	audioBytes, err := decodeBase64Audio(req.AudioBase64)
	if err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}

	pred, err := h.uc.InvokeTranscribe(deploymentID, apiKeyFromRequest(r), audioBytes, req.Language)
	if err != nil {
		handleErr(w, err)
		return
	}

	writeJSON(w, http.StatusOK, map[string]interface{}{
		"kind":   "asr",
		"result": pred,
	})
}

// decodeBase64Audio decodes a base64 audio payload, stripping an optional
// data-URL prefix ("data:audio/wav;base64,").
var (
	errAudioEmpty   = errors.New("\"audio_base64\" must be a non-empty base64 string")
	errAudioInvalid = errors.New("invalid base64 audio payload")
)

func decodeBase64Audio(s string) ([]byte, error) {
	s = strings.TrimSpace(s)
	if s == "" {
		return nil, errAudioEmpty
	}

	if idx := strings.Index(s, "base64,"); idx >= 0 {
		s = s[idx+len("base64,"):]
	}

	data, err := base64.StdEncoding.DecodeString(s)
	if err != nil {
		return nil, errAudioInvalid
	}

	return data, nil
}

// Embed serves the deployed embedding model. POST /inference/{id}/embed.
func (h *DeploymentHandler) Embed(w http.ResponseWriter, r *http.Request, deploymentID string) {
	var req embedRequest

	err := decodeJSON(r, &req)
	if err != nil {
		writeError(w, http.StatusBadRequest, "invalid JSON body")
		return
	}

	if len(req.Input) == 0 {
		writeError(w, http.StatusBadRequest, "\"input\" must be a non-empty list of texts")
		return
	}

	result, err := h.uc.Embed(deploymentID, apiKeyFromRequest(r), req.Input, req.Type)
	if err != nil {
		handleErr(w, err)
		return
	}

	writeJSON(w, http.StatusOK, map[string]interface{}{
		"kind":   "embedding",
		"result": result,
	})
}

// EmbedCorpus serves the "embed my corpus" batch job. POST /inference/{id}/embed-corpus.
func (h *DeploymentHandler) EmbedCorpus(w http.ResponseWriter, r *http.Request, deploymentID string) {
	job, data, err := h.uc.EmbedCorpus(deploymentID, apiKeyFromRequest(r))
	if err != nil {
		handleErr(w, err)
		return
	}

	w.Header().Set("Content-Type", "text/csv")
	w.Header().Set("Content-Disposition", `attachment; filename="corpus_embeddings.csv"`)
	_, _ = w.Write(data)

	// The job metadata is not streamed in the CSV; expose it in a response
	// header so the client can correlate the run.
	w.Header().Set("X-Corpus-Job-ID", job.ID)
	w.Header().Set("X-Corpus-Docs", strconv.Itoa(job.DoneDocs))
}

// Rerank serves the deployed reranker model. POST /inference/{id}/rerank.
func (h *DeploymentHandler) Rerank(w http.ResponseWriter, r *http.Request, deploymentID string) {
	var req rerankRequest

	err := decodeJSON(r, &req)
	if err != nil {
		writeError(w, http.StatusBadRequest, "invalid JSON body")
		return
	}

	if strings.TrimSpace(req.Query) == "" {
		writeError(w, http.StatusBadRequest, "\"query\" must be a non-empty string")
		return
	}

	if len(req.Documents) == 0 {
		writeError(w, http.StatusBadRequest, "\"documents\" must be a non-empty list")
		return
	}

	result, err := h.uc.Rerank(deploymentID, apiKeyFromRequest(r), req.Query, req.Documents, req.TopK)
	if err != nil {
		handleErr(w, err)
		return
	}

	writeJSON(w, http.StatusOK, map[string]interface{}{
		"kind":   "reranker",
		"result": result,
	})
}

// InvokeBatch accepts a text body -- either CSV with an "input" column, or
// plain newline-separated inputs -- and returns CSV predictions. This is the
// bulk workload narrow-task deployments exist to serve efficiently.
func (h *DeploymentHandler) InvokeBatch(w http.ResponseWriter, r *http.Request, deploymentID string) {
	if h.tabular != nil {
		kind, err := h.uc.DeploymentKind(deploymentID)
		if err == nil && (kind == domain.KindTabular || kind == domain.KindTimeSeries) {
			h.tabular.PredictBatch(w, r, deploymentID)

			return
		}
	}

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

		for _, valid := range validGGUFQuantizations {
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

// transcribeMultipart handles the multipart variant of /transcribe: an audio
// file field ("file", "audio", or "file0") and an optional "language" field.
func (h *DeploymentHandler) transcribeMultipart(w http.ResponseWriter, r *http.Request, deploymentID string) {
	r.Body = http.MaxBytesReader(w, r.Body, maxASRAudioUploadBytes)

	err := r.ParseMultipartForm(maxASRAudioUploadBytes)
	if err != nil {
		writeError(w, http.StatusBadRequest, "expected a multipart/form-data body with an audio file field")
		return
	}

	var file multipart.File

	for _, field := range []string{"file", "audio", "file0"} {
		file, _, err = r.FormFile(field)
		if err == nil {
			break
		}
	}

	if file == nil {
		writeError(w, http.StatusBadRequest, "no audio file found (expected form field \"file\")")
		return
	}

	defer file.Close()

	data, err := io.ReadAll(io.LimitReader(file, maxASRAudioUploadBytes+1))
	if err != nil {
		writeError(w, http.StatusBadRequest, "failed to read uploaded audio")
		return
	}

	pred, err := h.uc.InvokeTranscribe(deploymentID, apiKeyFromRequest(r), data, r.FormValue("language"))
	if err != nil {
		handleErr(w, err)
		return
	}

	writeJSON(w, http.StatusOK, map[string]interface{}{
		"kind":   "asr",
		"result": pred,
	})
}

// invokeVision handles POST /inference/{id}/predict for a vision_lm
// deployment: multipart form with an image file field ("file", "image", or
// "file0") and a "prompt" field, returning
// {"kind":"vision_lm","result":{"text":...,"json":...,"json_valid":...}}.
func (h *DeploymentHandler) invokeVision(w http.ResponseWriter, r *http.Request, deploymentID string) {
	r.Body = http.MaxBytesReader(w, r.Body, maxVisionUploadBytes)

	err := r.ParseMultipartForm(maxVisionUploadBytes)
	if err != nil {
		writeError(w, http.StatusBadRequest, "expected a multipart/form-data body with an image file and a \"prompt\" field")
		return
	}

	var (
		file   multipart.File
		header *multipart.FileHeader
	)

	for _, field := range []string{"file", "image", "file0"} {
		file, header, err = r.FormFile(field)
		if err == nil {
			break
		}
	}

	if file == nil {
		writeError(w, http.StatusBadRequest, "no image file found (expected form field \"file\")")
		return
	}

	defer file.Close()

	_ = header

	data, err := io.ReadAll(io.LimitReader(file, maxVisionUploadBytes+1))
	if err != nil {
		writeError(w, http.StatusBadRequest, "failed to read uploaded image")
		return
	}

	prompt := r.FormValue("prompt")

	pred, err := h.uc.InvokeVision(deploymentID, apiKeyFromRequest(r), data, prompt)
	if err != nil {
		handleErr(w, err)
		return
	}

	writeJSON(w, http.StatusOK, map[string]interface{}{
		"kind":   "vision_lm",
		"result": pred,
	})
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
