package agent_test

import (
	"context"
	"testing"
	"time"

	"distillery/internal/domain"
	infraagent "distillery/internal/infra/agent"
	"distillery/internal/repository/memory"
)

// mockLLM is a deterministic LLM provider for tests.
type mockLLM struct {
	responses []domain.Message
	tools     []domain.Tool
	callCount int
}

func (m *mockLLM) GenerateResponse(
	_ context.Context,
	_ []domain.Message,
	tools []domain.Tool,
	_ domain.AgentConfig,
) (domain.Message, error) {
	m.tools = tools
	if m.callCount < len(m.responses) {
		msg := m.responses[m.callCount]
		m.callCount++

		return msg, nil
	}

	return domain.Message{Role: "assistant", Content: "done", ToolCalls: nil, Timestamp: time.Time{}}, nil
}

func (m *mockLLM) StreamResponse(
	_ context.Context,
	_ []domain.Message,
	_ []domain.Tool,
	_ domain.AgentConfig,
	callback func(chunk string) error,
) error {
	return callback("mock")
}

func newReflectAgent(t *testing.T, llm domain.LLMProvider, tool *mockTool) *infraagent.ReflectAgent {
	t.Helper()

	toolReg := memory.NewToolRegistry()
	if tool != nil {
		_ = toolReg.Register(tool)
	}

	return infraagent.NewReflectAgent("agent_1", memory.NewAgentMemoryStore(), toolReg, llm, 10)
}

type mockTool struct {
	name     string
	executed bool
}

func (m *mockTool) Name() string { return "mock_tool" }

func (m *mockTool) Description() string { return "mock tool" }

func (m *mockTool) InputSchema() map[string]interface{} {
	return map[string]interface{}{"type": "object"}
}

func (m *mockTool) Execute(_ context.Context, _ domain.ToolInput) (domain.ToolOutput, error) {
	m.executed = true
	return domain.ToolOutput{Result: "executed", Error: ""}, nil
}

func TestReflectAgent_Initialize(t *testing.T) {
	t.Parallel()

	toolReg := memory.NewToolRegistry()
	a := infraagent.NewReflectAgent("agent_1", memory.NewAgentMemoryStore(), toolReg, &mockLLM{}, 10)

	config := domain.AgentConfig{Name: "test", MaxIterations: 5, Timeout: time.Minute, SystemPrompt: "", TemperatureHint: 0, TopPHint: 0}

	err := a.Initialize(context.Background(), config)
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}

	state, err := a.GetState(context.Background())
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}

	if state.Status != "idle" {
		t.Errorf("expected status 'idle', got %q", state.Status)
	}

	if state.Progress != 0 {
		t.Errorf("expected progress 0, got %d", state.Progress)
	}
}

func TestReflectAgent_Execute_NoTools_Done(t *testing.T) {
	t.Parallel()

	llm := &mockLLM{}
	a := newReflectAgent(t, llm, nil)
	_ = a.Initialize(context.Background(), domain.AgentConfig{MaxIterations: 5, Timeout: time.Minute, Name: "", SystemPrompt: "", TemperatureHint: 0, TopPHint: 0})

	state, err := a.Execute(context.Background(), "do something")
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}

	if state.Status != "done" {
		t.Errorf("expected status 'done', got %q", state.Status)
	}
}

func TestReflectAgent_Execute_WithTool(t *testing.T) {
	t.Parallel()

	tool := &mockTool{}
	llm := &mockLLM{responses: []domain.Message{
		{
			Role:    "assistant",
			Content: "using tool",
			ToolCalls: []domain.ToolCall{
				{ID: "call_1", ToolName: "mock_tool", Input: map[string]interface{}{}},
			},
		},
	}}
	a := newReflectAgent(t, llm, tool)
	_ = a.Initialize(context.Background(), domain.AgentConfig{MaxIterations: 5, Timeout: time.Minute, Name: "", SystemPrompt: "", TemperatureHint: 0, TopPHint: 0})

	state, err := a.Execute(context.Background(), "do something")
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}

	if !tool.executed {
		t.Error("expected tool to be executed")
	}

	if state.Status != "done" {
		t.Errorf("expected status 'done', got %q", state.Status)
	}

	if state.ToolCalls[0].Status != "success" {
		t.Errorf("expected tool call success, got %q", state.ToolCalls[0].Status)
	}
}

func TestReflectAgent_Reset_ClearsState(t *testing.T) {
	t.Parallel()

	llm := &mockLLM{}
	a := newReflectAgent(t, llm, nil)
	_ = a.Initialize(context.Background(), domain.AgentConfig{MaxIterations: 5, Timeout: time.Minute, Name: "", SystemPrompt: "", TemperatureHint: 0, TopPHint: 0})
	_, _ = a.Execute(context.Background(), "do something")

	err := a.Reset(context.Background())
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}

	state, err := a.GetState(context.Background())
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}

	if state.Status != "idle" {
		t.Errorf("expected status 'idle', got %q", state.Status)
	}

	if len(state.History) != 0 {
		t.Errorf("expected empty history, got %d messages", len(state.History))
	}
}
