package domain

import "time"

// ExampleSource records whether an example was provided by the user or
// bootstrapped synthetically from a frontier model.
type ExampleSource string

const (
	SourceUser      ExampleSource = "user"
	SourceSynthetic ExampleSource = "synthetic"
	SourceFeedback  ExampleSource = "feedback" // production mispredictions fed back in.
)

// Example is a single input/output training pair for a task.
type Example struct {
	ID        string        `json:"id"`
	TaskID    string        `json:"task_id"`
	Input     string        `json:"input"`
	Output    string        `json:"output"`
	Source    ExampleSource `json:"source"`
	Flagged   bool          `json:"flagged"` // low-quality flag from curation.
	FlagNote  string        `json:"flag_note,omitempty"`
	Duplicate bool          `json:"duplicate"` // deduped against another example.
	CreatedAt time.Time     `json:"created_at"`
}

// ExampleRepository is the port for persisting training examples.
type ExampleRepository interface {
	Add(e *Example) error
	AddBatch(es []*Example) error
	ListByTask(taskID string) ([]*Example, error)
	Get(taskID, id string) (*Example, error)
	Update(e *Example) error
	Delete(taskID, id string) error
	DeleteByTask(taskID string) error
}

// DatasetStats summarizes the state of a task's curated dataset.
type DatasetStats struct {
	TaskID          string         `json:"task_id"`
	Total           int            `json:"total"`
	Duplicates      int            `json:"duplicates"`
	Flagged         int            `json:"flagged"`
	Synthetic       int            `json:"synthetic"`
	UserProvided    int            `json:"user_provided"`
	Feedback        int            `json:"feedback"`
	UsableCount     int            `json:"usable_count"`            // total - duplicates - flagged.
	LabelBalance    map[string]int `json:"label_balance,omitempty"` // classification only.
	ReadyToTrain    bool           `json:"ready_to_train"`
	ReadinessReason string         `json:"readiness_reason,omitempty"`
}
