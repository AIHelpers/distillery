package dataset

import (
	"encoding/json"
	"strings"

	"distillery/internal/domain"
)

// CausalLMSchema validates the legacy text-to-text payload shape used by the
// original product: every example is an instruction -> output pair. This is
// the schema that keeps existing causal_lm runs byte-for-byte compatible.
type CausalLMSchema struct{}

// Kind implements domain.DatasetSchema.
func (s *CausalLMSchema) Kind() domain.ModelKind { return domain.KindCausalLM }

// causalLMPayload is the JSON object shape for a causal_lm example.
type causalLMPayload struct {
	// Instruction is the primary prompt text.
	Instruction string `json:"instruction"`
	// Input is optional additional context appended after the instruction.
	Input string `json:"input,omitempty"`
	// Output is the expected target text.
	Output string `json:"output"`
}

// Validate implements domain.DatasetSchema.
func (s *CausalLMSchema) Validate(payload json.RawMessage) error {
	var p causalLMPayload

	err := json.Unmarshal(payload, &p)
	if err != nil {
		return &schemaError{kind: domain.KindCausalLM, msg: "payload must be a JSON object with instruction/input/output fields"}
	}

	if strings.TrimSpace(p.Instruction) == "" && strings.TrimSpace(p.Input) == "" {
		return &schemaError{kind: domain.KindCausalLM, msg: "instruction (or input) is required"}
	}

	if strings.TrimSpace(p.Output) == "" {
		return &schemaError{kind: domain.KindCausalLM, msg: "output is required"}
	}

	return nil
}

// Stats implements domain.DatasetSchema.
func (s *CausalLMSchema) Stats(examples []*domain.Example) domain.DatasetStats {
	stats := domain.DatasetStats{
		Kind:         domain.KindCausalLM,
		LabelBalance: map[string]int{}, TaskID: "", Total: 0, Duplicates: 0, Flagged: 0, Synthetic: 0, UserProvided: 0, Feedback: 0, UsableCount: 0, ReadyToTrain: false, ReadinessReason: "",
	}

	for _, e := range examples {
		stats.Total++

		switch e.Source {
		case domain.SourceSynthetic:
			stats.Synthetic++
		case domain.SourceUser:
			stats.UserProvided++
		case domain.SourceFeedback:
			stats.Feedback++
		}

		if e.Duplicate {
			stats.Duplicates++
		}

		if e.Flagged {
			stats.Flagged++
		}
	}

	stats.UsableCount = stats.Total - stats.Duplicates - stats.Flagged
	stats.ReadyToTrain = stats.UsableCount >= 3

	if !stats.ReadyToTrain {
		stats.ReadinessReason = "need at least 3 usable (non-duplicate, non-flagged) examples"
	}

	return stats
}

// Formats implements domain.DatasetSchema.
func (s *CausalLMSchema) Formats() []domain.ImportFormat {
	return []domain.ImportFormat{domain.FormatCSV, domain.FormatJSONL}
}

// schemaError is a descriptive validation error.
type schemaError struct {
	kind domain.ModelKind
	msg  string
}

func (e *schemaError) Error() string {
	return e.msg
}
