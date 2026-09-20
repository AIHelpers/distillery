package domain

import "time"

type TrainingStatus string

const (
	TrainingQueued    TrainingStatus = "queued"
	TrainingRunning   TrainingStatus = "running"
	TrainingCompleted TrainingStatus = "completed"
	TrainingFailed    TrainingStatus = "failed"
)

// BaseModel is one entry in the curated set of open-weight base models
// the platform can automatically select from.
type BaseModel struct {
	Name           string  `json:"name"`
	ParamsBillions float64 `json:"params_billions"`
	Family         string  `json:"family"`
	// RepoID is the HuggingFace repo identifier (used by the local trainer).
	RepoID string `json:"repo_id,omitempty"`
	// MinVRAMGB is the minimum GPU VRAM (GiB) recommended to run this model with QLoRA.
	MinVRAMGB float64 `json:"min_vram_gb,omitempty"`
	// RecommendedQuant is the default quantization level, e.g. "4bit-nf4".
	RecommendedQuant string `json:"recommended_quant,omitempty"`
}

// TrainingMetrics are the (simulated) results of a LoRA/QLoRA fine-tune run.
type TrainingMetrics struct {
	FinalLoss     float64 `json:"final_loss"`
	EvalAccuracy  float64 `json:"eval_accuracy"`
	Epochs        int     `json:"epochs"`
	TrainExamples int     `json:"train_examples"`
}

// TrainingJob represents one fine-tuning run for a task.
type TrainingJob struct {
	ID          string           `json:"id"`
	TaskID      string           `json:"task_id"`
	Version     int              `json:"version"` // increments each retrain.
	BaseModel   BaseModel        `json:"base_model"`
	Status      TrainingStatus   `json:"status"`
	Progress    int              `json:"progress"` // 0-100
	Metrics     *TrainingMetrics `json:"metrics,omitempty"`
	Error       string           `json:"error,omitempty"`
	CreatedAt   time.Time        `json:"created_at"`
	StartedAt   *time.Time       `json:"started_at,omitempty"`
	CompletedAt *time.Time       `json:"completed_at,omitempty"`
}

// TrainingJobRepository is the port for persisting training jobs.
type TrainingJobRepository interface {
	Create(j *TrainingJob) error
	Get(id string) (*TrainingJob, error)
	ListByTask(taskID string) ([]*TrainingJob, error)
	Update(j *TrainingJob) error
	Delete(id string) error
	LatestCompleted(taskID string) (*TrainingJob, error)
}

// ModelSelector chooses a base model for a task based on dataset complexity.
type ModelSelector interface {
	SelectBaseModel(task *Task, exampleCount int, avgInputLen, avgOutputLen int) BaseModel
	// ListBaseModels returns the curated catalog of available base models
	// the user can choose from.
	ListBaseModels() []BaseModel
}

// FineTuner runs (or simulates) a LoRA/QLoRA fine-tuning job asynchronously,
// invoking onUpdate as progress changes and onDone when it finishes.
type FineTuner interface {
	Start(
		job *TrainingJob,
		examples []*Example,
		onUpdate func(progress int,
		), onDone func(metrics *TrainingMetrics, err error))
}
