package domain

import (
	"encoding/json"
	"time"
)

// ImportFormat identifies a supported dataset file format for importer
// registration and dispatch.
type ImportFormat string

const (
	FormatCSV     ImportFormat = "csv"
	FormatJSONL   ImportFormat = "jsonl"
	FormatCoNLL   ImportFormat = "conll"
	FormatParquet ImportFormat = "parquet"
)

// DatasetSchema is the per-kind contract for validating example payloads,
// computing dataset stats, and declaring supported import formats.
type DatasetSchema interface {
	// Kind returns the model kind this schema serves.
	Kind() ModelKind
	// Validate checks a single example payload. It returns an error with a
	// human-readable reason when the payload cannot be used for training.
	Validate(payload json.RawMessage) error
	// Stats computes aggregate stats over examples.
	Stats(examples []*Example) DatasetStats
	// Formats lists the import formats this schema supports.
	Formats() []ImportFormat
}

// ExampleSource records whether an example was provided by the user or
// bootstrapped synthetically from a frontier model.
type ExampleSource string

const (
	SourceUser      ExampleSource = "user"
	SourceSynthetic ExampleSource = "synthetic"
	SourceFeedback  ExampleSource = "feedback" // production mispredictions fed back in.
)

// Example is a single training example for a task. The payload is
// kind-specific (validated by the DatasetSchema for the task's ModelKind).
// For legacy causal_lm records, Payload holds {"text", ...} content and
// Input/Output mirror the underlying text fields for backward compatibility.
type Example struct {
	ID     string    `json:"id"`
	TaskID string    `json:"task_id"`
	Kind   ModelKind `json:"kind"` // mirrored from the parent Task for convenience.
	// Payload is kind-specific JSON — validated by DatasetSchema.Validate.
	Payload json.RawMessage `json:"payload,omitempty"`
	// Input/Output are retained for backward compatibility with the original
	// text-only schema (causal_lm). New kinds use Payload.
	Input     string        `json:"input,omitempty"`
	Output    string        `json:"output,omitempty"`
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
	Kind            ModelKind      `json:"kind"`
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
