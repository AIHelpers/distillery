package memory

import (
	"context"
	time "time"

	"distillery/internal/domain"
)

// AgentRepo persists agent states.
type AgentRepo struct{ store *Store }

func NewAgentRepo(store *Store) *AgentRepo { return &AgentRepo{store: store} }

func (r *AgentRepo) SaveState(_ context.Context, state domain.AgentState) error {
	r.store.mu.Lock()
	defer r.store.mu.Unlock()

	r.store.Agents[state.ID] = state
	r.store.persist()

	return nil
}

func (r *AgentRepo) GetState(_ context.Context, agentID string) (domain.AgentState, error) {
	r.store.mu.RLock()
	defer r.store.mu.RUnlock()

	state, ok := r.store.Agents[agentID]
	if !ok {
		return domain.AgentState{
			ID:           "",
			Status:       "",
			CurrentTask:  "",
			Progress:     0,
			ToolCalls:    nil,
			History:      nil,
			Error:        "",
			LastActivity: time.Time{},
			Metadata:     nil,
		}, domain.ErrNotFound
	}

	return state, nil
}

func (r *AgentRepo) ListAgents(_ context.Context) ([]domain.AgentState, error) {
	r.store.mu.RLock()
	defer r.store.mu.RUnlock()

	out := make([]domain.AgentState, 0, len(r.store.Agents))
	for _, s := range r.store.Agents {
		out = append(out, s)
	}

	return out, nil
}
