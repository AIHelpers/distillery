package domain_test

import (
	"testing"
	time "time"

	"distillery/internal/domain"
)

func TestToolCall_FailedStatus(t *testing.T) {
	t.Parallel()

	tc := domain.ToolCall{ID: "call_1", ToolName: "simple", Status: "failed", Error: "boom", Input: nil, Output: domain.ToolOutput{Result: nil, Error: ""}}
	if tc.Status != "failed" {
		t.Errorf("expected status 'failed', got %q", tc.Status)
	}

	if tc.Error != "boom" {
		t.Errorf("expected error 'boom', got %q", tc.Error)
	}
}

func TestToolCall_Output(t *testing.T) {
	t.Parallel()

	tc := domain.ToolCall{
		ID:       "call_1",
		ToolName: "simple",
		Status:   "success",
		Output:   domain.ToolOutput{Result: "executed", Error: ""}, Input: nil, Error: "",
	}
	if tc.Status != "success" {
		t.Errorf("expected status 'success', got %q", tc.Status)
	}

	if tc.Output.Result != "executed" {
		t.Errorf("expected Result 'executed', got %v", tc.Output.Result)
	}
}

func TestAgentState_DefaultMetadata(t *testing.T) {
	t.Parallel()

	state := domain.AgentState{ID: "a1", Status: "idle", Progress: 0, CurrentTask: "", ToolCalls: nil, History: nil, Error: "", LastActivity: time.Time{}, Metadata: nil}
	if state.Status != "idle" {
		t.Errorf("expected status 'idle', got %q", state.Status)
	}

	if state.Progress != 0 {
		t.Errorf("expected progress 0, got %d", state.Progress)
	}
}
