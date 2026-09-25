package domain

import "time"

// TaskType captures the shape of the narrow AI task being modeled. It remains
// as a user-facing hint; the architecturally significant discriminator is
// ModelKind.
type TaskType string

const (
	TaskClassification TaskType = "classification"
	TaskExtraction     TaskType = "extraction"
	TaskGeneration     TaskType = "generation"
)

// Task represents a user-defined narrow task: "define the task, get a model".
type Task struct {
	ID          string   `json:"id"`
	Name        string   `json:"name"`
	Description string   `json:"description"`
	Type        TaskType `json:"type"`
	// Kind is the architectural model kind (causal_lm, seq_classifier, ...).
	// Defaults to causal_lm for pre-kind records.
	Kind ModelKind `json:"kind"`
	// LabelSet restricts the entity types a token_classifier task may use
	// (e.g. ["ORG","AMOUNT","DATE"]). Empty = any label allowed. Importers
	// validate every span label against it.
	LabelSet []string `json:"label_set,omitempty"`
	// JSONSchema is a JSON Schema (draft-07 subset) that Track B extraction
	// outputs must satisfy: validated at import time and used to generate
	// llama.cpp GBNF for schema-constrained decoding at inference.
	JSONSchema string    `json:"json_schema,omitempty"`
	CreatedAt  time.Time `json:"created_at"`
	UpdatedAt  time.Time `json:"updated_at"`
}

// TaskRepository is the port for persisting tasks.
type TaskRepository interface {
	Create(t *Task) error
	Get(id string) (*Task, error)
	List() ([]*Task, error)
	Update(t *Task) error
	Delete(id string) error
}
