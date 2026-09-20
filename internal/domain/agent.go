package domain

import (
	"context"
	"time"
)

// ToolInput represents input to a tool.
type ToolInput struct {
	Params map[string]interface{} `json:"params"`
}

// ToolOutput represents tool execution result.
type ToolOutput struct {
	Result interface{} `json:"result"`
	Error  string      `json:"error,omitempty"`
}

// Tool defines an executable action the agent can invoke.
type Tool interface {
	Name() string
	Description() string
	InputSchema() map[string]interface{}
	Execute(ctx context.Context, input ToolInput) (ToolOutput, error)
}

// ToolRegistry manages available tools.
type ToolRegistry interface {
	Register(tool Tool) error
	Get(name string) (Tool, error)
	List() []Tool
}

// ToolCall represents a single tool invocation.
type ToolCall struct {
	ID       string                 `json:"id"`
	ToolName string                 `json:"tool_name"`
	Input    map[string]interface{} `json:"input"`
	Status   string                 `json:"status"` // pending, running, success, failed.
	Output   ToolOutput             `json:"output,omitempty"`
	Error    string                 `json:"error,omitempty"`
}

// Message represents agent-LLM conversation turn.
type Message struct {
	Role      string     `json:"role"` // user, assistant, tool.
	Content   string     `json:"content"`
	ToolCalls []ToolCall `json:"tool_calls,omitempty"`
	Timestamp time.Time  `json:"timestamp"`
}

// AgentMemory stores and retrieves conversation context.
type AgentMemory interface {
	// Append adds a message to history.
	Append(ctx context.Context, msg Message) error

	// GetHistory retrieves last N messages.
	GetHistory(ctx context.Context, limit int) ([]Message, error)

	// Clear resets memory.
	Clear(ctx context.Context) error
}

// AgentState tracks agent execution state.
type AgentState struct {
	ID           string            `json:"id"`
	Status       string            `json:"status"` // idle, thinking, executing, done, error, paused.
	CurrentTask  string            `json:"current_task"`
	Progress     int               `json:"progress"` // 0-100.
	ToolCalls    []ToolCall        `json:"tool_calls"`
	History      []Message         `json:"history"`
	Error        string            `json:"error,omitempty"`
	LastActivity time.Time         `json:"last_activity"`
	Metadata     map[string]string `json:"metadata"`
}

// AgentConfig describes agent behavior.
type AgentConfig struct {
	Name            string
	SystemPrompt    string
	MaxIterations   int
	Timeout         time.Duration
	TemperatureHint float32
	TopPHint        float32
}

// LLMProvider interface for swapping LLM backends.
type LLMProvider interface {
	// GenerateResponse generates LLM response with tool calling.
	GenerateResponse(
		ctx context.Context,
		messages []Message,
		tools []Tool,
		config AgentConfig,
	) (Message, error)

	// StreamResponse streams LLM tokens.
	StreamResponse(
		ctx context.Context,
		messages []Message,
		tools []Tool,
		config AgentConfig,
		callback func(chunk string) error,
	) error
}

// Agent orchestrates tool use toward domain goals.
type Agent interface {
	// Initialize sets up agent state.
	Initialize(ctx context.Context, config AgentConfig) error

	// Execute runs agent reasoning loop.
	Execute(ctx context.Context, goal string) (AgentState, error)

	// Step advances agent one reasoning iteration.
	Step(ctx context.Context) (AgentState, error)

	// GetState returns current agent state.
	GetState(ctx context.Context) (AgentState, error)

	// AddTool registers a tool dynamically.
	AddTool(tool Tool) error

	// Reset clears state and memory.
	Reset(ctx context.Context) error
}

// AgentRepository persists agent state.
type AgentRepository interface {
	SaveState(ctx context.Context, state AgentState) error
	GetState(ctx context.Context, agentID string) (AgentState, error)
	ListAgents(ctx context.Context) ([]AgentState, error)
}
