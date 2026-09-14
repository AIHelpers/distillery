package http

import (
	"net/http"

	"distillery/internal/usecase"
)

// AgentHandler exposes agent lifecycle HTTP endpoints.
type AgentHandler struct {
	uc *usecase.AgentOrchestrationUsecase
}

// NewAgentHandler creates an agent handler.
func NewAgentHandler(uc *usecase.AgentOrchestrationUsecase) *AgentHandler {
	return &AgentHandler{uc: uc}
}

type startFineTuningRequest struct {
	TaskID    string `json:"taskID"`
	BaseModel string `json:"baseModel"`
	DatasetID string `json:"datasetID"`
}

// StartFineTuningAgent POST /api/v1/agents/fine-tuning.
func (h *AgentHandler) StartFineTuningAgent(w http.ResponseWriter, r *http.Request) {
	var req startFineTuningRequest
	if err := decodeJSON(r, &req); err != nil {
		writeError(w, http.StatusBadRequest, "invalid JSON body")
		return
	}
	state, err := h.uc.StartFineTuningAgent(r.Context(), req.TaskID, req.BaseModel, req.DatasetID)
	if err != nil {
		handleErr(w, err)
		return
	}
	writeJSON(w, http.StatusAccepted, state)
}

// GetAgentState GET /api/v1/agents/{agentID}.
func (h *AgentHandler) GetAgentState(w http.ResponseWriter, r *http.Request) {
	agentID := r.PathValue("agentID")
	state, err := h.uc.GetAgentState(r.Context(), agentID)
	if err != nil {
		handleErr(w, err)
		return
	}
	writeJSON(w, http.StatusOK, state)
}

// ListAgentStates GET /api/v1/agents.
func (h *AgentHandler) ListAgentStates(w http.ResponseWriter, r *http.Request) {
	states, err := h.uc.ListAgentStates(r.Context())
	if err != nil {
		writeError(w, http.StatusInternalServerError, err.Error())
		return
	}
	writeJSON(w, http.StatusOK, states)
}

// PauseAgent POST /api/v1/agents/{agentID}/pause.
func (h *AgentHandler) PauseAgent(w http.ResponseWriter, r *http.Request) {
	agentID := r.PathValue("agentID")
	if err := h.uc.PauseAgent(r.Context(), agentID); err != nil {
		handleErr(w, err)
		return
	}
	w.WriteHeader(http.StatusNoContent)
}

// ResumeAgent POST /api/v1/agents/{agentID}/resume.
func (h *AgentHandler) ResumeAgent(w http.ResponseWriter, r *http.Request) {
	agentID := r.PathValue("agentID")
	if err := h.uc.ResumeAgent(r.Context(), agentID); err != nil {
		handleErr(w, err)
		return
	}
	w.WriteHeader(http.StatusNoContent)
}
