package domain

// TrainingStatusReport reports job progress (from TrainingControllerTool).
type TrainingStatusReport struct {
	TaskID       string
	Status       string
	Progress     int
	Metrics      map[string]float32
	EstRemaining int
}

// MetricsReport from training analysis.
type MetricsReport struct {
	TaskID  string
	Metrics map[string]interface{}
	Charts  []string
}

// ExportResult from model export.
type ExportResult struct {
	ModelID  string
	Format   string
	Location string
	Size     int64
}
