package domain

// FineTuningAgent coordinates model fine-tuning workflows.
type FineTuningAgent struct {
	AgentID     string // Implements Agent interface.
	Config      AgentConfig
	TaskID      string
	BaseModel   string
	DatasetID   string
	State       *AgentState
	Memory      AgentMemory
	ToolReg     ToolRegistry
	LLMProvider LLMProvider
}

// InferenceAgent orchestrates model inference optimization.
type InferenceAgent struct {
	AgentID     string
	Config      AgentConfig
	ModelID     string
	State       *AgentState
	Memory      AgentMemory
	ToolReg     ToolRegistry
	LLMProvider LLMProvider
}

// AnalysisAgent examines training results and suggests improvements.
type AnalysisAgent struct {
	AgentID     string
	Config      AgentConfig
	State       *AgentState
	Memory      AgentMemory
	ToolReg     ToolRegistry
	LLMProvider LLMProvider
}

// AgentWorkflow chains multiple agents for complex operations.
type AgentWorkflow struct {
	ID     string
	Name   string
	Goal   string
	Agents []Agent
	State  string // pending, running, success, failed.
	Result interface{}
	Error  string
}
