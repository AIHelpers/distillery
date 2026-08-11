package domain

import "time"

// TaskType captures the shape of the narrow AI task being modeled.
type TaskType string

const (
	TaskClassification TaskType = "classification"
	TaskExtraction     TaskType = "extraction"
	TaskGeneration     TaskType = "generation"
)

// Task represents a user-defined narrow task: "define the task, get a model".
type Task struct {
	ID          string    `json:"id"`
	Name        string    `json:"name"`
	Description string    `json:"description"`
	Type        TaskType  `json:"type"`
	CreatedAt   time.Time `json:"created_at"`
	UpdatedAt   time.Time `json:"updated_at"`
}

// TaskRepository is the port for persisting tasks.
type TaskRepository interface {
	Create(t *Task) error
	Get(id string) (*Task, error)
	List() ([]*Task, error)
	Update(t *Task) error
	Delete(id string) error
}
