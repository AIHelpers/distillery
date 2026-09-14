package agent

import (
	"context"
	"fmt"
	"sync"
	"time"

	"distillery/internal/domain"
)

// ReflectAgent implements domain.Agent with reflection-based tool calling.
type ReflectAgent struct {
	id          string
	config      domain.AgentConfig
	state       *domain.AgentState
	memory      domain.AgentMemory
	toolReg     domain.ToolRegistry
	llmProvider domain.LLMProvider
	maxIter     int
	paused      bool
	mu          sync.RWMutex
}

// NewReflectAgent creates a reflect-based agent.
func NewReflectAgent(
	id string,
	memory domain.AgentMemory,
	toolReg domain.ToolRegistry,
	llmProvider domain.LLMProvider,
	maxIter int,
) *ReflectAgent {
	return &ReflectAgent{
		id: id, memory: memory, toolReg: toolReg,
		llmProvider: llmProvider, maxIter: maxIter,
	}
}

// Initialize sets up agent state.
func (a *ReflectAgent) Initialize(_ context.Context, config domain.AgentConfig) error {
	a.mu.Lock()
	defer a.mu.Unlock()

	a.config = config
	a.state = &domain.AgentState{
		ID: a.id, Status: "idle", Progress: 0,
		ToolCalls: []domain.ToolCall{}, History: []domain.Message{},
		LastActivity: time.Now(), Metadata: make(map[string]string), CurrentTask: "", Error: "",
	}

	return nil
}

// Execute runs the agent reasoning loop.
func (a *ReflectAgent) Execute(ctx context.Context, goal string) (domain.AgentState, error) {
	a.mu.Lock()
	a.state.Status = "thinking"
	a.state.CurrentTask = goal
	a.mu.Unlock()

	userMsg := domain.Message{Role: "user", Content: goal, Timestamp: time.Now(), ToolCalls: nil}

	err := a.memory.Append(ctx, userMsg)
	if err != nil {
		return *a.state, err
	}

	ctx, cancel := context.WithTimeout(ctx, a.config.Timeout)
	defer cancel()

	maxIterations := a.config.MaxIterations
	if maxIterations <= 0 {
		maxIterations = a.maxIter
	}

	for i := range maxIterations {
		a.mu.RLock()

		if a.paused {
			a.mu.RUnlock()
			a.state.Status = "paused"

			return *a.state, nil
		}

		a.mu.RUnlock()

		state, err := a.Step(ctx)
		if err != nil {
			state.Error = err.Error()
			return state, err
		}

		if state.Status == "done" || state.Status == "error" {
			return state, nil
		}

		a.mu.Lock()
		a.state.Progress = (i + 1) * 100 / maxIterations
		a.mu.Unlock()
	}

	a.mu.Lock()
	a.state.Status = "done"
	a.state.Progress = 100
	a.mu.Unlock()

	return *a.state, nil
}

// Step advances one reasoning iteration.
func (a *ReflectAgent) Step(ctx context.Context) (domain.AgentState, error) {
	a.mu.Lock()
	defer a.mu.Unlock()

	history, err := a.memory.GetHistory(ctx, 20)
	if err != nil {
		a.state.Status = "error"
		a.state.Error = err.Error()

		return *a.state, err
	}

	tools := a.toolReg.List()
	a.state.Status = "thinking"

	assistantMsg, err := a.llmProvider.GenerateResponse(ctx, history, tools, a.config)
	if err != nil {
		a.state.Status = "error"
		a.state.Error = fmt.Sprintf("LLM error: %v", err)

		return *a.state, err
	}

	assistantMsg.Timestamp = time.Now()

	err = a.memory.Append(ctx, assistantMsg)
	if err != nil {
		a.state.Status = "error"
		return *a.state, err
	}

	a.state.History = append(a.state.History, assistantMsg)

	if len(assistantMsg.ToolCalls) == 0 {
		a.state.Status = "done"
		return *a.state, nil
	}

	a.state.Status = "executing"

	for i := range assistantMsg.ToolCalls {
		tc := &assistantMsg.ToolCalls[i]
		tc.Status = "running"
		a.state.ToolCalls = append(a.state.ToolCalls, *tc)

		tool, err := a.toolReg.Get(tc.ToolName)
		if err != nil {
			tc.Status = "failed"
			tc.Error = err.Error()
		} else {
			output, err := tool.Execute(ctx, domain.ToolInput{Params: tc.Input})
			if err != nil {
				tc.Status = "failed"
				tc.Error = err.Error()
			} else {
				tc.Status = "success"
				tc.Output = output
			}
		}

		resultMsg := domain.Message{
			Role: "tool", Content: fmt.Sprintf("Tool %s returned: %v", tc.ToolName, tc.Output.Result),
			Timestamp: time.Now(), ToolCalls: nil,
		}
		_ = a.memory.Append(ctx, resultMsg)

		// Update the stored copy in state so tool call statuses are reflected.
		a.state.ToolCalls[len(a.state.ToolCalls)-1] = *tc
	}

	a.state.Status = "thinking"
	a.state.LastActivity = time.Now()

	return *a.state, nil
}

// GetState returns current state.
func (a *ReflectAgent) GetState(_ context.Context) (domain.AgentState, error) {
	a.mu.RLock()
	defer a.mu.RUnlock()

	return *a.state, nil
}

// AddTool registers a tool.
func (a *ReflectAgent) AddTool(tool domain.Tool) error { return a.toolReg.Register(tool) }

// Reset clears state.
func (a *ReflectAgent) Reset(ctx context.Context) error {
	a.mu.Lock()
	defer a.mu.Unlock()

	a.state = &domain.AgentState{
		ID: a.id, Status: "idle", Progress: 0,
		ToolCalls: []domain.ToolCall{}, History: []domain.Message{},
		LastActivity: time.Now(), Metadata: make(map[string]string), CurrentTask: "", Error: "",
	}

	return a.memory.Clear(ctx)
}
