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
}

type invokeRequest struct {
	Input string `json:"input"`
}

type mispredictionRequest struct {
	Input          string `json:"input"`
	ActualOutput   string `json:"actual_output"`
	ExpectedOutput string `json:"expected_output"`
}
