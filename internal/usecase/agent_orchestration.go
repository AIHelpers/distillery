package usecase

import (
	"context"
	"fmt"
	"sync"
	"time"

	"distillery/internal/domain"
	infraagent "distillery/internal/infra/agent"
	"distillery/internal/repository/memory"
)

// AgentOrchestrationUsecase coordinates AI agents for domain workflows.
type AgentOrchestrationUsecase struct {
	agentRepo    domain.AgentRepository
	toolRegistry domain.ToolRegistry
	llmProvider  domain.LLMProvider
	taskRepo     domain.TaskRepository
	agents       sync.Map
	idGen        IDGenerator
}

// NewAgentOrchestrationUsecase creates the orchestration usecase.
func NewAgentOrchestrationUsecase(
	agentRepo domain.AgentRepository,
	toolRegistry domain.ToolRegistry,
	llmProvider domain.LLMProvider,
	taskRepo domain.TaskRepository,
	idGen IDGenerator,
) *AgentOrchestrationUsecase {
	return &AgentOrchestrationUsecase{
		agentRepo:    agentRepo,
		toolRegistry: toolRegistry,
		llmProvider:  llmProvider,
		taskRepo:     taskRepo,
		idGen:        idGen,
	}
}

// StartFineTuningAgent starts a reflect agent for fine-tuning workflow.
func (u *AgentOrchestrationUsecase) StartFineTuningAgent(ctx context.Context, taskID, baseModel, datasetID string) (domain.AgentState, error) {
	agentID := u.idGen.NewID("fta")
	mem := memory.NewAgentMemoryStore()
	agent := infraagent.NewReflectAgent(agentID, mem, u.toolRegistry, u.llmProvider, 10)
	config := domain.AgentConfig{
		Name:            fmt.Sprintf("FineTuning[%s]", taskID),
		SystemPrompt:    sysPromptFineTune(),
		MaxIterations:   3,
		Timeout:         30 * time.Minute,
		TemperatureHint: 0.3,
	}
	if err := agent.Initialize(ctx, config); err != nil {
		return domain.AgentState{}, err
	}
	u.agents.Store(agentID, agent)
	goal := fmt.Sprintf("Fine-tune model %s on dataset %s for task %s.", baseModel, datasetID, taskID)
	state, err := agent.Execute(ctx, goal)
	if err != nil {
		state.Error = err.Error()
		state.Status = "error"
	}
	_ = u.agentRepo.SaveState(ctx, state)
	return state, err
}

// GetAgentState retrieves an agent state.
func (u *AgentOrchestrationUsecase) GetAgentState(ctx context.Context, agentID string) (domain.AgentState, error) {
	if agent, ok := u.agents.Load(agentID); ok {
		return agent.(domain.Agent).GetState(ctx)
	}
	return u.agentRepo.GetState(ctx, agentID)
}

// ListAgentStates lists all agent states.
func (u *AgentOrchestrationUsecase) ListAgentStates(ctx context.Context) ([]domain.AgentState, error) {
	return u.agentRepo.ListAgents(ctx)
}

// PauseAgent pauses execution.
func (u *AgentOrchestrationUsecase) PauseAgent(ctx context.Context, agentID string) error {
	if agent, ok := u.agents.Load(agentID); ok {
		state, _ := agent.(domain.Agent).GetState(ctx)
		state.Status = "paused"
		return u.agentRepo.SaveState(ctx, state)
	}
	return domain.ErrNotFound
}

// ResumeAgent resumes execution.
func (u *AgentOrchestrationUsecase) ResumeAgent(ctx context.Context, agentID string) error {
	if agent, ok := u.agents.Load(agentID); ok {
		state, _ := agent.(domain.Agent).GetState(ctx)
		state.Status = "thinking"
		return u.agentRepo.SaveState(ctx, state)
	}
	return domain.ErrNotFound
}

func sysPromptFineTune() string {
	return `You are a fine-tuning orchestration agent. Your goal is to:
1. Validate the dataset using the dataset_validator tool
2. Select optimal base model using model_selector tool
3. Configure and start training using training_controller tool
4. Monitor progress and analyze metrics
5. Export the trained model
Use tools intelligently and explain your reasoning at each step.`
}
