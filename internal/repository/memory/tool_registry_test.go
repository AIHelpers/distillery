package memory_test

import (
	"context"
	"testing"

	"distillery/internal/domain"
	"distillery/internal/repository/memory"
)

// mockTool is a simple tool for testing the registry.
type mockTool struct {
	name string
}

func (m *mockTool) Name() string { return m.name }

func (m *mockTool) Description() string { return "mock tool" }

func (m *mockTool) InputSchema() map[string]interface{} {
	return map[string]interface{}{"type": "object"}
}

func (m *mockTool) Execute(_ context.Context, _ domain.ToolInput) (domain.ToolOutput, error) {
	return domain.ToolOutput{Result: "ok"}, nil
}

func TestToolRegistry_RegisterAndGet(t *testing.T) {
	r := memory.NewToolRegistry()
	tool := &mockTool{name: "test_tool"}

	if err := r.Register(tool); err != nil {
		t.Fatalf("expected no error, got %v", err)
	}
	got, err := r.Get("test_tool")
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}
	if got.Name() != "test_tool" {
		t.Errorf("expected name 'test_tool', got %q", got.Name())
	}
}

func TestToolRegistry_Get_NotFound(t *testing.T) {
	r := memory.NewToolRegistry()

	_, err := r.Get("missing")
	if err != domain.ErrNotFound {
		t.Errorf("expected ErrNotFound, got %v", err)
	}
}

func TestToolRegistry_List(t *testing.T) {
	r := memory.NewToolRegistry()
	_ = r.Register(&mockTool{name: "a"})
	_ = r.Register(&mockTool{name: "b"})

	list := r.List()
	if len(list) != 2 {
		t.Errorf("expected 2 tools, got %d", len(list))
	}
}
