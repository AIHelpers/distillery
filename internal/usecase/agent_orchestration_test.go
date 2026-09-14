package usecase_test

import (
	"context"
	"errors"
	"testing"
	time "time"

	"distillery/internal/domain"
	"distillery/internal/infra/agent"
	"distillery/internal/repository/memory"
	"distillery/internal/usecase"
)

// --- Mock AgentRepository ---.

type mockAgentRepo struct {
	states   map[string]domain.AgentState
	saved    []domain.AgentState
	err      error
	getAgent string
}

func newMockAgentRepo() *mockAgentRepo {
	return &mockAgentRepo{states: map[string]domain.AgentState{}}
}

func (m *mockAgentRepo) SaveState(_ context.Context, state domain.AgentState) error {
	if m.err != nil {
		return m.err
	}

	m.states[state.ID] = state
	m.saved = append(m.saved, state)

	return nil
}

func (m *mockAgentRepo) GetState(_ context.Context, agentID string) (domain.AgentState, error) {
	m.getAgent = agentID
	if m.err != nil {
		return domain.AgentState{ID: "", Status: "", CurrentTask: "", Progress: 0, ToolCalls: nil, History: nil, Error: "", LastActivity: time.Time{}, Metadata: nil}, m.err
	}

	state, ok := m.states[agentID]
	if !ok {
		return domain.AgentState{ID: "", Status: "", CurrentTask: "", Progress: 0, ToolCalls: nil, History: nil, Error: "", LastActivity: time.Time{}, Metadata: nil}, domain.ErrNotFound
	}

	return state, nil
}

func (m *mockAgentRepo) ListAgents(_ context.Context) ([]domain.AgentState, error) {
	if m.err != nil {
		return nil, m.err
	}

	out := make([]domain.AgentState, 0, len(m.states))
	for _, s := range m.states {
		out = append(out, s)
	}

	return out, nil
}

// --- AgentOrchestration tests ---.

func newOrchestrationUsecase() (*usecase.AgentOrchestrationUsecase, *mockAgentRepo) {
	agentRepo := newMockAgentRepo()
	toolReg := memory.NewToolRegistry()
	_ = toolReg.Register(agent.NewDatasetValidatorTool(&mockTaskRepo{}))
	llm := agent.NewSimulatedLLMProvider()
	uc := usecase.NewAgentOrchestrationUsecase(agentRepo, toolReg, llm, nil, newStubIDGen())

	return uc, agentRepo
}

func TestAgentOrchestration_StartFineTuningAgent(t *testing.T) {
	t.Parallel()

	uc, agentRepo := newOrchestrationUsecase()

	state, err := uc.StartFineTuningAgent(context.Background(), "task_1", "gpt2", "dataset_1")
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}

	if state.ID == "" {
		t.Error("expected non-empty agent ID")
	}

	if state.Status != "done" {
		t.Errorf("expected status 'done', got %q", state.Status)
	}

	if len(agentRepo.saved) == 0 {
		t.Error("expected agent state to be persisted")
	}

	if agentRepo.states[state.ID].ID != state.ID {
		t.Error("expected persisted state to match returned agent")
	}
}

func TestAgentOrchestration_GetAgentState_NotFound(t *testing.T) {
	t.Parallel()

	uc, _ := newOrchestrationUsecase()

	_, err := uc.GetAgentState(context.Background(), "missing_agent")
	if !errors.Is(err, domain.ErrNotFound) {
		t.Errorf("expected ErrNotFound, got %v", err)
	}
}

func TestAgentOrchestration_GetAgentState_FromRepo(t *testing.T) {
	t.Parallel()

	uc, agentRepo := newOrchestrationUsecase()
	agentRepo.states["fta_stub_001"] = domain.AgentState{ID: "fta_stub_001", Status: "done", CurrentTask: "", Progress: 0, ToolCalls: nil, History: nil, Error: "", LastActivity: time.Time{}, Metadata: nil}

	state, err := uc.GetAgentState(context.Background(), "fta_stub_001")
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}

	if state.Status != "done" {
		t.Errorf("expected status 'done', got %q", state.Status)
	}

	if agentRepo.getAgent != "fta_stub_001" {
		t.Errorf("expected GetState called with 'fta_stub_001', got %q", agentRepo.getAgent)
	}
}

func TestAgentOrchestration_ListAgentStates(t *testing.T) {
	t.Parallel()

	uc, agentRepo := newOrchestrationUsecase()
	agentRepo.states["a1"] = domain.AgentState{ID: "a1", Status: "", CurrentTask: "", Progress: 0, ToolCalls: nil, History: nil, Error: "", LastActivity: time.Time{}, Metadata: nil}
	agentRepo.states["a2"] = domain.AgentState{ID: "a2", Status: "", CurrentTask: "", Progress: 0, ToolCalls: nil, History: nil, Error: "", LastActivity: time.Time{}, Metadata: nil}

	states, err := uc.ListAgentStates(context.Background())
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}

	if len(states) != 2 {
		t.Errorf("expected 2 states, got %d", len(states))
	}
}

func TestAgentOrchestration_PauseAgent_NotFound(t *testing.T) {
	t.Parallel()

	uc, _ := newOrchestrationUsecase()

	err := uc.PauseAgent(context.Background(), "missing_agent")
	if !errors.Is(err, domain.ErrNotFound) {
		t.Errorf("expected ErrNotFound, got %v", err)
	}
}

func TestAgentOrchestration_ResumeAgent_NotFound(t *testing.T) {
	t.Parallel()

	uc, _ := newOrchestrationUsecase()

	err := uc.ResumeAgent(context.Background(), "missing_agent")
	if !errors.Is(err, domain.ErrNotFound) {
		t.Errorf("expected ErrNotFound, got %v", err)
	}
}

func TestAgentOrchestration_FineTuning_PersistedState(t *testing.T) {
	t.Parallel()

	uc, _ := newOrchestrationUsecase()

	state, err := uc.StartFineTuningAgent(context.Background(), "task_2", "gpt2-medium", "dataset_2")
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}

	if state.Status != "done" {
		t.Fatalf("expected status 'done', got %q", state.Status)
	}

	// State should be retrievable from the persisted repository.
	actual, err := uc.GetAgentState(context.Background(), state.ID)
	if err != nil {
		t.Fatalf("expected no error retrieving persisted state, got %v", err)
	}

	if actual.ID != state.ID {
		t.Errorf("expected ID %q, got %q", state.ID, actual.ID)
	}
}

func TestAgentOrchestration_SaveStateError_StillReturnsState(t *testing.T) {
	t.Parallel()

	agentRepo := newMockAgentRepo()
	agentRepo.err = errRepoFailure
	toolReg := memory.NewToolRegistry()
	llm := agent.NewSimulatedLLMProvider()
	uc := usecase.NewAgentOrchestrationUsecase(agentRepo, toolReg, llm, nil, newStubIDGen())

	state, err := uc.StartFineTuningAgent(context.Background(), "task_3", "gpt2", "dataset_3")
	if err != nil {
		t.Fatalf("expected no error from agent execution even if persist fails, got %v", err)
	}

	if state.Status != "done" {
		t.Errorf("expected status 'done', got %q", state.Status)
	}
}
