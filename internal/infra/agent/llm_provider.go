package agent

import (
	"context"
	"fmt"
	"math/rand"
	"time"

	"distillery/internal/domain"
)

// SimulatedLLMProvider simulates LLM responses with tool calling.
type SimulatedLLMProvider struct{ callCount int }

// NewSimulatedLLMProvider returns a simulated LLM provider.
func NewSimulatedLLMProvider() *SimulatedLLMProvider { return &SimulatedLLMProvider{} }

// GenerateResponse picks a tool sequentially from the registry.
func (s *SimulatedLLMProvider) GenerateResponse(
	_ context.Context,
	messages []domain.Message,
	tools []domain.Tool,
	_ domain.AgentConfig,
) (domain.Message, error) {
	s.callCount++

	var selected []domain.ToolCall

	if len(messages) > 0 && len(tools) > 0 && s.callCount <= len(tools)+1 {
		tool := tools[(s.callCount-1)%len(tools)]
		selected = append(selected, domain.ToolCall{
			ID:       fmt.Sprintf("call_%d", rand.Intn(100000)),
			ToolName: tool.Name(),
			Input:    map[string]interface{}{"datasetID": "default", "baseModel": "gpt2", "taskType": "classification"},
			Status:   "pending", Output: domain.ToolOutput{Result: nil, Error: ""}, Error: "",
		})
	}

	return domain.Message{
		Role:      "assistant",
		Content:   fmt.Sprintf("Iteration %d: executing next tool", s.callCount),
		ToolCalls: selected,
		Timestamp: time.Now(),
	}, nil
}

// StreamResponse simulates streaming.
func (s *SimulatedLLMProvider) StreamResponse(
	_ context.Context,
	_ []domain.Message,
	_ []domain.Tool,
	_ domain.AgentConfig,
	callback func(chunk string) error,
) error {
	for _, chunk := range []string{"Thinking", " about ", "the ", "task", "..."} {
		err := callback(chunk)
		if err != nil {
			return err
		}
	}

	return nil
}
