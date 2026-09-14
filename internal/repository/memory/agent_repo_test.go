package memory_test

import (
	"context"
	"testing"

	"distillery/internal/domain"
	"distillery/internal/repository/memory"
)

func newAgentRepo() (*memory.AgentRepo, *memory.Store) {
	s := memory.NewStore("")
	return memory.NewAgentRepo(s), s
}

func TestAgentRepo_SaveState(t *testing.T) {
	r, s := newAgentRepo()
	state := domain.AgentState{ID: "fta_1", Status: "thinking"}

	if err := r.SaveState(context.Background(), state); err != nil {
		t.Fatalf("expected no error, got %v", err)
	}
	got, ok := s.Agents["fta_1"]
	if !ok {
		t.Fatal("expected agent state to be stored")
	}
	if got.Status != "thinking" {
		t.Errorf("expected status 'thinking', got %q", got.Status)
	}
}

func TestAgentRepo_GetState_Found(t *testing.T) {
	r, _ := newAgentRepo()
	_ = r.SaveState(context.Background(), domain.AgentState{ID: "fta_1", Status: "done"})

	got, err := r.GetState(context.Background(), "fta_1")
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}
	if got.ID != "fta_1" {
		t.Errorf("expected ID 'fta_1', got %q", got.ID)
	}
}

func TestAgentRepo_GetState_NotFound(t *testing.T) {
	r, _ := newAgentRepo()

	_, err := r.GetState(context.Background(), "missing")
	if err != domain.ErrNotFound {
		t.Errorf("expected ErrNotFound, got %v", err)
	}
}

func TestAgentRepo_ListAgents(t *testing.T) {
	r, _ := newAgentRepo()
	_ = r.SaveState(context.Background(), domain.AgentState{ID: "a1"})
	_ = r.SaveState(context.Background(), domain.AgentState{ID: "a2"})

	list, err := r.ListAgents(context.Background())
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}
	if len(list) != 2 {
		t.Errorf("expected 2 agents, got %d", len(list))
	}
}
