package memory

import (
	"sync"

	"distillery/internal/domain"
)

// ToolRegistryImpl implements domain.ToolRegistry.
type ToolRegistryImpl struct {
	mu    sync.RWMutex
	tools map[string]domain.Tool
}

// NewToolRegistry creates an empty tool registry.
func NewToolRegistry() *ToolRegistryImpl {
	return &ToolRegistryImpl{tools: make(map[string]domain.Tool)}
}

// Register adds a tool by name.
func (r *ToolRegistryImpl) Register(tool domain.Tool) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.tools[tool.Name()] = tool
	return nil
}

// Get returns a tool by name.
func (r *ToolRegistryImpl) Get(name string) (domain.Tool, error) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	tool, ok := r.tools[name]
	if !ok {
		return nil, domain.ErrNotFound
	}
	return tool, nil
}

// List returns all registered tools.
func (r *ToolRegistryImpl) List() []domain.Tool {
	r.mu.RLock()
	defer r.mu.RUnlock()
	out := make([]domain.Tool, 0, len(r.tools))
	for _, tool := range r.tools {
		out = append(out, tool)
	}
	return out
}
