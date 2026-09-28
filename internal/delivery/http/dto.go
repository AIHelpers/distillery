package http

import (
	"encoding/json"
	"net/http"
)

func writeJSON(w http.ResponseWriter, status int, v interface{}) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(v)
}

func writeError(w http.ResponseWriter, status int, msg string) {
	writeJSON(w, status, map[string]string{"error": msg})
}

func decodeJSON(r *http.Request, v interface{}) error {
	dec := json.NewDecoder(r.Body)
	dec.DisallowUnknownFields()

	return dec.Decode(v)
}

// --- Request/response payloads ---.

type createTaskRequest struct {
	Name        string `json:"name"`
	Description string `json:"description"`
	Type        string `json:"type"`
	// Kind optionally selects the model architecture (seq_classifier, ...);
	// empty falls back to the causal_lm default.
	Kind string `json:"kind"`
}

// updateTaskRequest is a partial update; omitted (nil) fields are unchanged.
type updateTaskRequest struct {
	Name        *string   `json:"name"`
	Description *string   `json:"description"`
	LabelSet    *[]string `json:"label_set"`
	JSONSchema  *string   `json:"json_schema"`
	// RetentionDays bounds how long a vision_lm task's uploaded images are
	// kept before the automatic retention sweep deletes them.
	RetentionDays *int `json:"retention_days"`
}

type addExamplesRequest struct {
	Pairs []struct {
		Input  string `json:"input"`
		Output string `json:"output"`
	} `json:"pairs"`
}

type generateSyntheticRequest struct {
	Count int `json:"count"`
}

type updateExampleRequest struct {
	Input  string `json:"input"`
	Output string `json:"output"`
}

type deployRequest struct {
	Autoscale bool `json:"autoscale"`
	// Force overrides the DPO/ORPO regression gate (see the training
	// usecase's regression check) and deploys anyway.
	Force bool `json:"force,omitempty"`
}

type invokeRequest struct {
	Input string `json:"input"`
}

// embedRequest is the body of POST /inference/{id}/embed.
type embedRequest struct {
	// Input is the list of texts to embed.
	Input []string `json:"input"`
	// Type selects the encode-time prefix: "query" or "document". Empty
	// defaults to "query".
	Type string `json:"type,omitempty"`
}

// rerankRequest is the body of POST /inference/{id}/rerank.
type rerankRequest struct {
	Query     string   `json:"query"`
	Documents []string `json:"documents"`
	TopK      int      `json:"top_k,omitempty"`
}

type mispredictionRequest struct {
	Input          string `json:"input"`
	ActualOutput   string `json:"actual_output"`
	ExpectedOutput string `json:"expected_output"`
}

// preferencePairRequest is the body of POST /tasks/{id}/preferences.
type preferencePairRequest struct {
	Prompt   string `json:"prompt"`
	Chosen   string `json:"chosen"`
	Rejected string `json:"rejected"`
}

// addASRAudioExampleRequest is the body of POST /tasks/{id}/examples/asr.
// Audio is base64-encoded (data-URL prefixes like "data:audio/wav;base64,"
// are stripped by the handler) so a single-clip add works as plain JSON
// without a multipart round trip.
type addASRAudioExampleRequest struct {
	AudioBase64 string `json:"audio_base64"`
	Filename    string `json:"filename,omitempty"`
	Text        string `json:"text"`
	Speaker     string `json:"speaker,omitempty"`
}

// transcribeRequest is the body of POST /inference/{id}/transcribe.
// Audio is base64-encoded (data-URL prefixes are stripped).
type transcribeRequest struct {
	AudioBase64 string `json:"audio_base64"`
	Filename    string `json:"filename,omitempty"`
	// Language optionally pins the transcription language (ISO 639-1);
	// empty lets the model auto-detect.
	Language string `json:"language,omitempty"`
}

// addVisionExampleRequest is the body of POST /tasks/{id}/examples/vision.
// Image is base64-encoded (data-URL prefixes like "data:image/png;base64,"
// are stripped by the handler) so a single-image add works as plain JSON
// without a multipart round trip.
type addVisionExampleRequest struct {
	ImageBase64 string `json:"image_base64"`
	Prompt      string `json:"prompt"`
	Answer      string `json:"answer,omitempty"`
}
