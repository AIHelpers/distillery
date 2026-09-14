package memory_test

import (
	"context"
	"testing"
	"time"

	"distillery/internal/domain"
	"distillery/internal/repository/memory"
)

func TestAgentMemoryStore_AppendAndGetHistory(t *testing.T) {
	m := memory.NewAgentMemoryStore()
	ctx := context.Background()

	msg1 := domain.Message{Role: "user", Content: "hello", Timestamp: time.Now()}
	msg2 := domain.Message{Role: "assistant", Content: "hi", Timestamp: time.Now()}

	if err := m.Append(ctx, msg1); err != nil {
		t.Fatalf("expected no error, got %v", err)
	}
	if err := m.Append(ctx, msg2); err != nil {
		t.Fatalf("expected no error, got %v", err)
	}

	history, err := m.GetHistory(ctx, 10)
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}
	if len(history) != 2 {
		t.Fatalf("expected 2 messages, got %d", len(history))
	}
	if history[0].Content != "hello" {
		t.Errorf("expected first message 'hello', got %q", history[0].Content)
	}
	if history[1].Content != "hi" {
		t.Errorf("expected second message 'hi', got %q", history[1].Content)
	}
}

func TestAgentMemoryStore_GetHistory_Limit(t *testing.T) {
	m := memory.NewAgentMemoryStore()
	ctx := context.Background()

	for i := 0; i < 5; i++ {
		if err := m.Append(ctx, domain.Message{Role: "user", Content: "msg"}); err != nil {
			t.Fatalf("expected no error, got %v", err)
		}
	}

	history, err := m.GetHistory(ctx, 3)
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}
	if len(history) != 3 {
		t.Errorf("expected 3 messages, got %d", len(history))
	}
}

func TestAgentMemoryStore_GetHistory_Empty(t *testing.T) {
	m := memory.NewAgentMemoryStore()

	history, err := m.GetHistory(context.Background(), 10)
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}
	if len(history) != 0 {
		t.Errorf("expected empty history, got %d messages", len(history))
	}
}

func TestAgentMemoryStore_Clear(t *testing.T) {
	m := memory.NewAgentMemoryStore()
	ctx := context.Background()
	_ = m.Append(ctx, domain.Message{Role: "user", Content: "hello"})

	if err := m.Clear(ctx); err != nil {
		t.Fatalf("expected no error, got %v", err)
	}
	history, err := m.GetHistory(ctx, 10)
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}
	if len(history) != 0 {
		t.Errorf("expected empty history after clear, got %d messages", len(history))
	}
}
