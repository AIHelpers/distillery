package domain

import "context"

// ToolSpec is a base implementation helper for tools.
type ToolSpec struct {
	toolName        string
	toolDescription string
	schema          map[string]interface{}
}

// Name returns the tool identifier.
func (b *ToolSpec) Name() string { return b.toolName }

// Description returns human-readable description for LLM.
func (b *ToolSpec) Description() string { return b.toolDescription }

// InputSchema returns JSON schema for validation.
func (b *ToolSpec) InputSchema() map[string]interface{} { return b.schema }

// Tool interfaces for the fine-tuning domain.
type (
	// DatasetValidator checks dataset integrity.
	DatasetValidator interface {
		Tool
		ValidateDataset(ctx context.Context, datasetID string) (ValidationResult, error)
	}

	// ModelPickTool picks optimal base model.
	ModelPickTool interface {
		Tool
		SelectModel(ctx context.Context, criteria map[string]interface{}) (string, error)
	}

	// TrainingController starts/monitors training.
	TrainingController interface {
		Tool
		StartTraining(ctx context.Context, req TrainingRequest) (string, error)
		GetTrainingStatus(ctx context.Context, taskID string) (TrainingStatusReport, error)
	}

	// MetricsAnalyzer examines training metrics.
	MetricsAnalyzer interface {
		Tool
		AnalyzeMetrics(ctx context.Context, taskID string) (MetricsReport, error)
	}

	// ModelExporter exports trained models.
	ModelExporter interface {
		Tool
		Export(ctx context.Context, modelID, format string) (ExportResult, error)
	}
)

// ValidationResult from dataset validation.
type ValidationResult struct {
	Valid       bool
	RecordCount int
	Issues      []string
	Warnings    []string
}

// TrainingRequest for starting fine-tuning.
type TrainingRequest struct {
	BaseModel     string
	DatasetID     string
	Epochs        int
	LearningRate  float32
	BatchSize     int
	OutputModelID string
}
