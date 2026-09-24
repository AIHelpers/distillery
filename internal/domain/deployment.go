package domain

import "time"

type DeploymentStatus string

const (
	DeploymentActive  DeploymentStatus = "active"
	DeploymentStopped DeploymentStatus = "stopped"
)

// Deployment represents a fine-tuned model exposed as an API endpoint.
// Inference calls against Endpoint must present the raw API key that was
// returned (once) at deploy time; only its hash is ever persisted.
type Deployment struct {
	ID            string `json:"id"`
	TaskID        string `json:"task_id"`
	TrainingJobID string `json:"training_job_id"`
	// Kind is the architectural model kind served by this deployment.
	Kind         ModelKind        `json:"kind"`
	Endpoint     string           `json:"endpoint"`
	Autoscale    bool             `json:"autoscale"`
	Status       DeploymentStatus `json:"status"`
	RequestCount int              `json:"request_count"`
	APIKeyHash   string           `json:"-"` // never serialized; raw key shown once at creation.
	CreatedAt    time.Time        `json:"created_at"`

	// LabelMap maps label -> id for seq_classifier deployments (used by the
	// inference server to decode the model head). Empty for causal_lm.
	LabelMap map[string]int `json:"label_map,omitempty"`
	// ConfidenceThreshold is the default below-threshold cutoff for
	// seq_classifier deployments (predictions under it enter the review queue).
	ConfidenceThreshold float64 `json:"confidence_threshold,omitempty"`
}

// DeploymentRepository is the port for persisting deployments.
type DeploymentRepository interface {
	Create(d *Deployment) error
	Get(id string) (*Deployment, error)
	GetActiveForTask(taskID string) (*Deployment, error)
	ListByTask(taskID string) ([]*Deployment, error)
	Update(d *Deployment) error
}

// InferenceEngine serves predictions from a deployed fine-tuned model.
type InferenceEngine interface {
	Predict(job *TrainingJob, trainingExamples []*Example, input string) (output string, confidence float64)
}

// Exporter builds a portable, self-hostable export package for a trained model.
type Exporter interface {
	// BuildExport returns the bytes of a downloadable archive (e.g. zip)
	// containing a Docker image spec + manifest + weights placeholder.
	BuildExport(task *Task, job *TrainingJob) ([]byte, string, error) // returns bytes, filename, error.
}

// GGUFExportOptions controls how a trained model is converted to the
// GGUF format (HomeBred-LLM / llama.cpp compatible).
type GGUFExportOptions struct {
	// Quantization selects the GGUF quantization scheme (e.g. "q4_k_m").
	Quantization string `json:"quantization"`
}

// GGUFExporter converts a trained model to GGUF and returns the resulting
// GGUF file bytes + suggested filename. It is used to produce downloadable
// model files that can be loaded directly into HomeBred-LLM or llama.cpp.
type GGUFExporter interface {
	// BuildGGUF converts the job's trained model to GGUF.
	// It returns the GGUF bytes, a suggested filename, and any error.
	BuildGGUF(task *Task, job *TrainingJob, opts GGUFExportOptions) ([]byte, string, error)
}

// GGUFProgressInfo is the progress snapshot for an async GGUF conversion.
type GGUFProgressInfo struct {
	SessionID string `json:"session_id"`
	TaskID    string `json:"task_id"`
	JobID     string `json:"job_id,omitempty"`
	Status    string `json:"status"` // "running" | "ready" | "error".
	Step      string `json:"step"`
	Percent   int    `json:"percent"`
	Detail    string `json:"detail,omitempty"`
	Filename  string `json:"filename,omitempty"`
	Size      int64  `json:"size,omitempty"`
	Error     string `json:"error,omitempty"`
	Quant     string `json:"quantization"`
}

// AsyncGGUFExporter supports async GGUF conversion with progress tracking.
type AsyncGGUFExporter interface {
	StartGGUFAsync(sessionID, taskID, jobID string, task *Task, job *TrainingJob, opts GGUFExportOptions)
	GetGGUFProgress(sessionID string) *GGUFProgressInfo
	GetGGUFResult(sessionID string) (string, string, error)
	// CleanupGGUFSession deletes the converted GGUF file from disk and
	// removes the session from the progress store. Called after the file
	// has been streamed to the client so large GGUF files don't accumulate.
	CleanupGGUFSession(sessionID string)
}
