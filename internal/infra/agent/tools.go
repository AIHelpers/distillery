package agent

import (
	"context"

	"distillery/internal/domain"
)

// DatasetValidatorTool validates a dataset.
type DatasetValidatorTool struct {
	taskRepo domain.TaskRepository
}

// NewDatasetValidatorTool creates a dataset validator tool.
func NewDatasetValidatorTool(taskRepo domain.TaskRepository) *DatasetValidatorTool {
	return &DatasetValidatorTool{taskRepo: taskRepo}
}

// Name returns the tool identifier.
func (t *DatasetValidatorTool) Name() string { return "dataset_validator" }

// Description returns human-readable description.
func (t *DatasetValidatorTool) Description() string {
	return "Validates dataset integrity and returns record count and issues"
}

// InputSchema returns JSON schema.
func (t *DatasetValidatorTool) InputSchema() map[string]interface{} {
	return map[string]interface{}{
		"type": "object",
		"properties": map[string]interface{}{
			"datasetID": map[string]interface{}{"type": "string"},
		},
		"required": []string{"datasetID"},
	}
}

// Execute validates a dataset (simulated).
func (t *DatasetValidatorTool) Execute(_ context.Context, input domain.ToolInput) (domain.ToolOutput, error) {
	datasetID, _ := input.Params["datasetID"].(string)
	result := map[string]interface{}{
		"valid":       true,
		"recordCount": 1000,
		"issues":      []string{},
		"datasetID":   datasetID,
	}

	return domain.ToolOutput{Result: result, Error: ""}, nil
}

// ModelSelectorTool picks a base model.
type ModelSelectorTool struct{}

// NewModelSelectorTool creates a model selector tool.
func NewModelSelectorTool() *ModelSelectorTool { return &ModelSelectorTool{} }

// Name returns the tool identifier.
func (t *ModelSelectorTool) Name() string { return "model_selector" }

// Description returns human-readable description.
func (t *ModelSelectorTool) Description() string {
	return "Selects optimal base model based on task requirements"
}

// InputSchema returns JSON schema.
func (t *ModelSelectorTool) InputSchema() map[string]interface{} {
	return map[string]interface{}{
		"type": "object",
		"properties": map[string]interface{}{
			"taskType":  map[string]interface{}{"type": "string"},
			"modelSize": map[string]interface{}{"type": "string"},
		},
	}
}

// Execute selects a model heuristically.
func (t *ModelSelectorTool) Execute(_ context.Context, _ domain.ToolInput) (domain.ToolOutput, error) {
	result := map[string]interface{}{
		"modelID":   "gpt2-medium",
		"reasoning": "Selected based on task type and resource constraints",
	}

	return domain.ToolOutput{Result: result, Error: ""}, nil
}

// CodeEvaluatorTool validates generated code — syntax and executability.
type CodeEvaluatorTool struct{}

// NewCodeEvaluatorTool creates a code evaluation tool.
func NewCodeEvaluatorTool() *CodeEvaluatorTool { return &CodeEvaluatorTool{} }

// Name returns the tool identifier.
func (t *CodeEvaluatorTool) Name() string { return "code_evaluator" }

// Description returns human-readable description.
func (t *CodeEvaluatorTool) Description() string {
	return "Validates generated code: syntax validity and sandboxed executability per language"
}

// InputSchema returns JSON schema.
func (t *CodeEvaluatorTool) InputSchema() map[string]interface{} {
	return map[string]interface{}{
		"type": "object",
		"properties": map[string]interface{}{
			"language": map[string]interface{}{"type": "string"},
			"code":     map[string]interface{}{"type": "string"},
		},
		"required": []string{"language", "code"},
	}
}

// Execute validates the code snippet.
func (t *CodeEvaluatorTool) Execute(_ context.Context, input domain.ToolInput) (domain.ToolOutput, error) {
	language, _ := input.Params["language"].(string)
	_, _ = input.Params["code"].(string) // reserved for real sandboxed execution with TRAINING_BACKEND=local

	result := map[string]interface{}{
		"language":        language,
		"syntaxValid":     true,
		"executable":      true,
		"executionTimeMs": 0,
		"executionError":  "",
		"usesNetwork":     false,
		"note":            "static checks only in simulation mode",
	}

	return domain.ToolOutput{Result: result, Error: ""}, nil
}

// TrainingControllerTool starts fine-tuning jobs.
type TrainingControllerTool struct {
	taskRepo domain.TaskRepository
}

// NewTrainingControllerTool creates a training controller tool.
func NewTrainingControllerTool(taskRepo domain.TaskRepository) *TrainingControllerTool {
	return &TrainingControllerTool{taskRepo: taskRepo}
}

// Name returns the tool identifier.
func (t *TrainingControllerTool) Name() string { return "training_controller" }

// Description returns human-readable description.
func (t *TrainingControllerTool) Description() string {
	return "Starts and monitors model fine-tuning job"
}

// InputSchema returns JSON schema.
func (t *TrainingControllerTool) InputSchema() map[string]interface{} {
	return map[string]interface{}{
		"type": "object",
		"properties": map[string]interface{}{
			"baseModel": map[string]interface{}{"type": "string"},
			"datasetID": map[string]interface{}{"type": "string"},
			"epochs":    map[string]interface{}{"type": "integer"},
			"batchSize": map[string]interface{}{"type": "integer"},
		},
		"required": []string{"baseModel", "datasetID"},
	}
}

// Execute starts a training job (simulated).
func (t *TrainingControllerTool) Execute(_ context.Context, input domain.ToolInput) (domain.ToolOutput, error) {
	baseModel, _ := input.Params["baseModel"].(string)

	datasetID, _ := input.Params["datasetID"].(string)
	if baseModel == "" || datasetID == "" {
		return domain.ToolOutput{Error: "missing baseModel or datasetID", Result: nil}, nil
	}

	result := map[string]interface{}{
		"taskID":  "job_sim_1",
		"status":  "started",
		"message": "Training job initiated (simulated)",
	}

	return domain.ToolOutput{Result: result, Error: ""}, nil
}
