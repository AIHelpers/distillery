package domain

import (
	"context"
	"testing"
)

// simpleTool is a minimal tool for tests.
type simpleTool struct{}

func (simpleTool) Name() string { return "simple" }

func (simpleTool) Description() string { return "simple tool" }

func (simpleTool) InputSchema() map[string]interface{} {
	return map[string]interface{}{"type": "object"}
}

func (simpleTool) Execute(_ context.Context, _ ToolInput) (ToolOutput, error) {
	return ToolOutput{Result: "ok"}, nil
}

func TestToolSpec_BaseHelpers(t *testing.T) {
	spec := &ToolSpec{toolName: "my_tool", toolDescription: "desc", schema: map[string]interface{}{}}
	if spec.Name() != "my_tool" {
		t.Errorf("expected Name 'my_tool', got %q", spec.Name())
	}
	if spec.Description() != "desc" {
		t.Errorf("expected Description 'desc', got %q", spec.Description())
	}
	if spec.InputSchema() == nil {
		t.Error("expected non-nil InputSchema")
	}
}

func TestToolCall_FailedStatus(t *testing.T) {
	tc := ToolCall{ID: "call_1", ToolName: "simple", Status: "failed", Error: "boom"}
	if tc.Status != "failed" {
		t.Errorf("expected status 'failed', got %q", tc.Status)
	}
	if tc.Error != "boom" {
		t.Errorf("expected error 'boom', got %q", tc.Error)
	}
}

func TestToolCall_Output(t *testing.T) {
	tc := ToolCall{
		ID:       "call_1",
		ToolName: "simple",
		Status:   "success",
		Output:   ToolOutput{Result: "executed"},
	}
	if tc.Status != "success" {
		t.Errorf("expected status 'success', got %q", tc.Status)
	}
	if tc.Output.Result != "executed" {
		t.Errorf("expected Result 'executed', got %v", tc.Output.Result)
	}
}

func TestAgentState_DefaultMetadata(t *testing.T) {
	state := AgentState{ID: "a1", Status: "idle", Progress: 0}
	if state.Status != "idle" {
		t.Errorf("expected status 'idle', got %q", state.Status)
	}
	if state.Progress != 0 {
		t.Errorf("expected progress 0, got %d", state.Progress)
	}
}
