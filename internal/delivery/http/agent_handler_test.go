package http_test

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	deliveryhttp "distillery/internal/delivery/http"
	"distillery/internal/domain"
	"distillery/internal/infra/agent"
	"distillery/internal/repository/memory"
	"distillery/internal/usecase"
)

func newAgentHandlerTestServer(t *testing.T) *httptest.Server {
	t.Helper()
	agentRepo := memory.NewAgentRepo(memory.NewStore(""))
	toolReg := memory.NewToolRegistry()
	_ = toolReg.Register(agent.NewDatasetValidatorTool(nil))
	llm := agent.NewSimulatedLLMProvider()
	uc := usecase.NewAgentOrchestrationUsecase(agentRepo, toolReg, llm, nil, usecase.NewRandomIDGenerator())
	h := deliveryhttp.NewAgentHandler(uc)

	mux := http.NewServeMux()
	mux.HandleFunc("POST /api/v1/agents/fine-tuning", h.StartFineTuningAgent)
	mux.HandleFunc("GET /api/v1/agents", h.ListAgentStates)
	mux.HandleFunc("GET /api/v1/agents/{agentID}", h.GetAgentState)
	mux.HandleFunc("POST /api/v1/agents/{agentID}/pause", h.PauseAgent)
	mux.HandleFunc("POST /api/v1/agents/{agentID}/resume", h.ResumeAgent)
	return httptest.NewServer(mux)
}

func TestAgentHandler_StartFineTuningAgent(t *testing.T) {
	srv := newAgentHandlerTestServer(t)
	defer srv.Close()

	body := strings.NewReader(`{"taskID":"task_1","baseModel":"gpt2","datasetID":"ds_1"}`)
	resp, err := http.Post(srv.URL+"/api/v1/agents/fine-tuning", "application/json", body)
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusAccepted {
		t.Errorf("expected status 202, got %d", resp.StatusCode)
	}

	var state domain.AgentState
	if err := json.NewDecoder(resp.Body).Decode(&state); err != nil {
		t.Fatalf("expected valid JSON, got %v", err)
	}
	if state.ID == "" {
		t.Error("expected non-empty agent ID")
	}
	if state.Status != "done" {
		t.Errorf("expected status 'done', got %q", state.Status)
	}
}

func TestAgentHandler_StartFineTuningAgent_BadBody(t *testing.T) {
	srv := newAgentHandlerTestServer(t)
	defer srv.Close()

	body := strings.NewReader(`{invalid`)
	resp, err := http.Post(srv.URL+"/api/v1/agents/fine-tuning", "application/json", body)
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusBadRequest {
		t.Errorf("expected status 400, got %d", resp.StatusCode)
	}
}

func TestAgentHandler_GetAgentState_Found(t *testing.T) {
	srv := newAgentHandlerTestServer(t)
	defer srv.Close()

	// Start an agent, then fetch its state using the returned ID.
	body := strings.NewReader(`{"taskID":"task_1","baseModel":"gpt2","datasetID":"ds_1"}`)
	resp, err := http.Post(srv.URL+"/api/v1/agents/fine-tuning", "application/json", body)
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}
	var state domain.AgentState
	_ = json.NewDecoder(resp.Body).Decode(&state)
	resp.Body.Close()

	getResp, err := http.Get(srv.URL + "/api/v1/agents/" + state.ID)
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}
	defer getResp.Body.Close()

	if getResp.StatusCode != http.StatusOK {
		t.Errorf("expected status 200, got %d", getResp.StatusCode)
	}

	var fetched domain.AgentState
	if err := json.NewDecoder(getResp.Body).Decode(&fetched); err != nil {
		t.Fatalf("expected valid JSON, got %v", err)
	}
	if fetched.ID != state.ID {
		t.Errorf("expected ID %q, got %q", state.ID, fetched.ID)
	}
}

func TestAgentHandler_GetAgentState_NotFound(t *testing.T) {
	srv := newAgentHandlerTestServer(t)
	defer srv.Close()

	resp, err := http.Get(srv.URL + "/api/v1/agents/missing_agent")
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusNotFound {
		t.Errorf("expected status 404, got %d", resp.StatusCode)
	}
}

func TestAgentHandler_ListAgentStates(t *testing.T) {
	srv := newAgentHandlerTestServer(t)
	defer srv.Close()

	// Start an agent first.
	body := strings.NewReader(`{"taskID":"task_1","baseModel":"gpt2","datasetID":"ds_1"}`)
	resp, err := http.Post(srv.URL+"/api/v1/agents/fine-tuning", "application/json", body)
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}
	resp.Body.Close()

	listResp, err := http.Get(srv.URL + "/api/v1/agents")
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}
	defer listResp.Body.Close()

	if listResp.StatusCode != http.StatusOK {
		t.Errorf("expected status 200, got %d", listResp.StatusCode)
	}

	var states []domain.AgentState
	if err := json.NewDecoder(listResp.Body).Decode(&states); err != nil {
		t.Fatalf("expected valid JSON, got %v", err)
	}
	if len(states) != 1 {
		t.Errorf("expected 1 agent state, got %d", len(states))
	}
}

func TestAgentHandler_PauseResume_NotFound(t *testing.T) {
	srv := newAgentHandlerTestServer(t)
	defer srv.Close()

	pauseResp, err := http.Post(srv.URL+"/api/v1/agents/missing_agent/pause", "application/json", nil)
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}
	pauseResp.Body.Close()
	if pauseResp.StatusCode != http.StatusNotFound {
		t.Errorf("expected status 404 for pause, got %d", pauseResp.StatusCode)
	}

	resumeResp, err := http.Post(srv.URL+"/api/v1/agents/missing_agent/resume", "application/json", nil)
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}
	resumeResp.Body.Close()
	if resumeResp.StatusCode != http.StatusNotFound {
		t.Errorf("expected status 404 for resume, got %d", resumeResp.StatusCode)
	}
}
