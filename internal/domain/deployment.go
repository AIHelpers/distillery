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
	ID            string           `json:"id"`
	TaskID        string           `json:"task_id"`
	TrainingJobID string           `json:"training_job_id"`
	Endpoint      string           `json:"endpoint"`
	Autoscale     bool             `json:"autoscale"`
	Status        DeploymentStatus `json:"status"`
	RequestCount  int              `json:"request_count"`
	APIKeyHash    string           `json:"-"` // never serialized; raw key shown once at creation.
	CreatedAt     time.Time        `json:"created_at"`
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
