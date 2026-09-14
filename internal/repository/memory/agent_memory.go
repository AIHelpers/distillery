package memory

import (
	"context"
	"sync"

	"distillery/internal/domain"
)

// AgentMemoryStore implements domain.AgentMemory for a single agent.
type AgentMemoryStore struct {
	mu       sync.RWMutex
	messages []domain.Message
}

// NewAgentMemoryStore creates an empty memory store.
func NewAgentMemoryStore() *AgentMemoryStore {
	return &AgentMemoryStore{messages: []domain.Message{}}
}

// Append adds a message to history.
func (m *AgentMemoryStore) Append(_ context.Context, msg domain.Message) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.messages = append(m.messages, msg)
	return nil
}

// GetHistory retrieves last N messages.
func (m *AgentMemoryStore) GetHistory(_ context.Context, limit int) ([]domain.Message, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()
	if len(m.messages) <= limit {
		out := make([]domain.Message, len(m.messages))
		copy(out, m.messages)
		return out, nil
	}
	out := make([]domain.Message, limit)
	copy(out, m.messages[len(m.messages)-limit:])
	return out, nil
}

// Clear resets memory.
func (m *AgentMemoryStore) Clear(_ context.Context) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.messages = []domain.Message{}
	return nil
}
