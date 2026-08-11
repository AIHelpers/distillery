package domain

import "time"

// Misprediction is a production example the user flagged as wrong,
// captured for the continuous improvement / retraining loop.
type Misprediction struct {
	ID             string    `json:"id"`
	TaskID         string    `json:"task_id"`
	Input          string    `json:"input"`
	ActualOutput   string    `json:"actual_output"`
	ExpectedOutput string    `json:"expected_output"`
	Resolved       bool      `json:"resolved"` // true once folded into a retrain.
	CreatedAt      time.Time `json:"created_at"`
}

// FeedbackRepository is the port for persisting production feedback.
type FeedbackRepository interface {
	Add(f *Misprediction) error
	ListByTask(taskID string) ([]*Misprediction, error)
	ListUnresolved(taskID string) ([]*Misprediction, error)
	MarkResolved(ids []string) error
}

// SyntheticGenerator bootstraps/augments training examples from a handful
// of user-provided samples (in production this would call a frontier model).
type SyntheticGenerator interface {
	Generate(task *Task, seed []*Example, count int) []*Example
}
