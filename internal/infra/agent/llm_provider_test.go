package agent_test

import (
	"context"
	"strings"
	"testing"

	"distillery/internal/domain"
	infraagent "distillery/internal/infra/agent"
	"distillery/internal/repository/memory"
)

func TestSimulatedLLMProvider_GenerateResponse_SelectsTool(t *testing.T) {
	p := infraagent.NewSimulatedLLMProvider()

	tool := &mockTool{name: "mock_tool"}
	toolReg := memory.NewToolRegistry()
	_ = toolReg.Register(tool)

	msg, err := p.GenerateResponse(
		context.Background(),
		[]domain.Message{{Role: "user", Content: "goal"}},
		toolReg.List(),
		domain.AgentConfig{},
	)
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}
	if len(msg.ToolCalls) != 1 {
		t.Fatalf("expected 1 tool call, got %d", len(msg.ToolCalls))
	}
	if msg.ToolCalls[0].ToolName != "mock_tool" {
		t.Errorf("expected tool 'mock_tool', got %q", msg.ToolCalls[0].ToolName)
	}
}

func TestSimulatedLLMProvider_GenerateResponse_NoTools(t *testing.T) {
	p := infraagent.NewSimulatedLLMProvider()

	msg, err := p.GenerateResponse(
		context.Background(),
		[]domain.Message{{Role: "user", Content: "goal"}},
		nil,
		domain.AgentConfig{},
	)
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}
	if len(msg.ToolCalls) != 0 {
		t.Errorf("expected no tool calls, got %d", len(msg.ToolCalls))
	}
}

func TestSimulatedLLMProvider_StreamResponse(t *testing.T) {
	p := infraagent.NewSimulatedLLMProvider()

	var sb strings.Builder
	err := p.StreamResponse(
		context.Background(),
		nil, nil, domain.AgentConfig{},
		func(chunk string) error {
			sb.WriteString(chunk)
			return nil
		},
	)
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}
	if sb.Len() == 0 {
		t.Error("expected streamed content")
	}
}

var _ domain.LLMProvider = (*infraagent.SimulatedLLMProvider)(nil)
