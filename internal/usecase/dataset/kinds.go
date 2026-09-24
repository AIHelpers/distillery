package dataset

import (
	"encoding/json"
	"strings"

	"distillery/internal/domain"
)

// SeqClassifierSchema validates payloads for sequence classification:
// {"text": "...", "label": "positive"} optionally with "id"/"source".
type SeqClassifierSchema struct{}

// seqClassifierPayload is the JSON object shape for a seq_classifier example.
type seqClassifierPayload struct {
	Text  string `json:"text"`
	Label string `json:"label"`
}

// Kind implements domain.DatasetSchema.
func (s *SeqClassifierSchema) Kind() domain.ModelKind { return domain.KindSeqClassifier }

// Validate implements domain.DatasetSchema.
func (s *SeqClassifierSchema) Validate(payload json.RawMessage) error {
	var p seqClassifierPayload

	err := json.Unmarshal(payload, &p)
	if err != nil {
		return &schemaError{kind: domain.KindSeqClassifier, msg: "payload must be a JSON object with text/label fields"}
	}

	if strings.TrimSpace(p.Text) == "" {
		return &schemaError{kind: domain.KindSeqClassifier, msg: "text is required"}
	}

	if strings.TrimSpace(p.Label) == "" {
		return &schemaError{kind: domain.KindSeqClassifier, msg: "label is required"}
	}

	return nil
}

// Stats implements domain.DatasetSchema.
func (s *SeqClassifierSchema) Stats(examples []*domain.Example) domain.DatasetStats {
	return classificationStats(domain.KindSeqClassifier, examples)
}

// Formats implements domain.DatasetSchema.
func (s *SeqClassifierSchema) Formats() []domain.ImportFormat {
	return []domain.ImportFormat{domain.FormatCSV, domain.FormatJSONL}
}

// --- Token classifier (NER) ---.

// TokenClassifierSchema validates payloads for token/span classification
// (NER). Each example is {"text": "...", "entities": [{"start","end","label"}]}.
type TokenClassifierSchema struct{}

// TokenClassifierPayload is the JSON shape for a token_classifier example.
type TokenClassifierPayload struct {
	Text     string `json:"text"`
	Entities []struct {
		Start int    `json:"start"`
		End   int    `json:"end"`
		Label string `json:"label"`
	} `json:"entities"`
}

// Kind implements domain.DatasetSchema.
func (s *TokenClassifierSchema) Kind() domain.ModelKind { return domain.KindTokenClassifier }

// Validate implements domain.DatasetSchema.
func (s *TokenClassifierSchema) Validate(payload json.RawMessage) error {
	var p TokenClassifierPayload

	err := json.Unmarshal(payload, &p)
	if err != nil {
		return &schemaError{kind: domain.KindTokenClassifier, msg: "payload must be a JSON object with text/entities fields"}
	}

	if strings.TrimSpace(p.Text) == "" {
		return &schemaError{kind: domain.KindTokenClassifier, msg: "text is required"}
	}

	// Entities are optional — unlabeled text is a valid (neutral) sample.

	return nil
}

// Stats implements domain.DatasetSchema.
func (s *TokenClassifierSchema) Stats(examples []*domain.Example) domain.DatasetStats {
	return classificationStats(domain.KindTokenClassifier, examples)
}

// Formats implements domain.DatasetSchema.
func (s *TokenClassifierSchema) Formats() []domain.ImportFormat {
	return []domain.ImportFormat{domain.FormatJSONL, domain.FormatCoNLL}
}

// --- Embedding ---.

// EmbeddingSchema validates payloads for text embedding training:
// {"text": "...", "label": "..."} (label optional — used for contrastive
// or clustering tasks; may be absent for pure retrieval-style datasets).
type EmbeddingSchema struct{}

// embeddingPayload is the JSON object shape for an embedding example.
type embeddingPayload struct {
	Text  string `json:"text"`
	Label string `json:"label,omitempty"`
}

// Kind implements domain.DatasetSchema.
func (s *EmbeddingSchema) Kind() domain.ModelKind { return domain.KindEmbedding }

// Validate implements domain.DatasetSchema.
func (s *EmbeddingSchema) Validate(payload json.RawMessage) error {
	var p embeddingPayload

	err := json.Unmarshal(payload, &p)
	if err != nil {
		return &schemaError{kind: domain.KindEmbedding, msg: "payload must be a JSON object with a text field"}
	}

	if strings.TrimSpace(p.Text) == "" {
		return &schemaError{kind: domain.KindEmbedding, msg: "text is required"}
	}

	return nil
}

// Stats implements domain.DatasetSchema.
func (s *EmbeddingSchema) Stats(examples []*domain.Example) domain.DatasetStats {
	return genericStats(domain.KindEmbedding, examples)
}

// Formats implements domain.DatasetSchema.
func (s *EmbeddingSchema) Formats() []domain.ImportFormat {
	return []domain.ImportFormat{domain.FormatCSV, domain.FormatJSONL}
}

// --- Reranker ---.

// RerankerSchema validates payloads for query/document relevance scoring:
// {"query": "...", "document": "...", "label": true|false} (label optional).
type RerankerSchema struct{}

// rerankerPayload is the JSON object shape for a reranker example.
type rerankerPayload struct {
	Query    string `json:"query"`
	Document string `json:"document"`
}

// Kind implements domain.DatasetSchema.
func (s *RerankerSchema) Kind() domain.ModelKind { return domain.KindReranker }

// Validate implements domain.DatasetSchema.
func (s *RerankerSchema) Validate(payload json.RawMessage) error {
	var p rerankerPayload

	err := json.Unmarshal(payload, &p)
	if err != nil {
		return &schemaError{kind: domain.KindReranker, msg: "payload must be a JSON object with query/document fields"}
	}

	if strings.TrimSpace(p.Query) == "" {
		return &schemaError{kind: domain.KindReranker, msg: "query is required"}
	}

	if strings.TrimSpace(p.Document) == "" {
		return &schemaError{kind: domain.KindReranker, msg: "document is required"}
	}

	return nil
}

// Stats implements domain.DatasetSchema.
func (s *RerankerSchema) Stats(examples []*domain.Example) domain.DatasetStats {
	return genericStats(domain.KindReranker, examples)
}

// Formats implements domain.DatasetSchema.
func (s *RerankerSchema) Formats() []domain.ImportFormat {
	return []domain.ImportFormat{domain.FormatCSV, domain.FormatJSONL}
}

// --- Preference LM (DPO) ---.

// PreferenceLMSchema validates payloads for preference/reward modeling:
// {"prompt": "...", "chosen": "...", "rejected": "..."}.
type PreferenceLMSchema struct{}

// preferenceLMPayload is the JSON object shape for a preference_lm example.
type preferenceLMPayload struct {
	Prompt   string `json:"prompt"`
	Chosen   string `json:"chosen"`
	Rejected string `json:"rejected"`
}

// Kind implements domain.DatasetSchema.
func (s *PreferenceLMSchema) Kind() domain.ModelKind { return domain.KindPreferenceLM }

// Validate implements domain.DatasetSchema.
func (s *PreferenceLMSchema) Validate(payload json.RawMessage) error {
	var p preferenceLMPayload

	err := json.Unmarshal(payload, &p)
	if err != nil {
		return &schemaError{kind: domain.KindPreferenceLM, msg: "payload must be a JSON object with prompt/chosen/rejected fields"}
	}

	if strings.TrimSpace(p.Prompt) == "" {
		return &schemaError{kind: domain.KindPreferenceLM, msg: "prompt is required"}
	}

	if strings.TrimSpace(p.Chosen) == "" {
		return &schemaError{kind: domain.KindPreferenceLM, msg: "chosen is required"}
	}

	if strings.TrimSpace(p.Rejected) == "" {
		return &schemaError{kind: domain.KindPreferenceLM, msg: "rejected is required"}
	}

	return nil
}

// Stats implements domain.DatasetSchema.
func (s *PreferenceLMSchema) Stats(examples []*domain.Example) domain.DatasetStats {
	return genericStats(domain.KindPreferenceLM, examples)
}

// Formats implements domain.DatasetSchema.
func (s *PreferenceLMSchema) Formats() []domain.ImportFormat {
	return []domain.ImportFormat{domain.FormatJSONL}
}

// --- Shared stats builders ---.

// classificationStats computes stats including label balance for classifier
// kinds (seq_classifier, token_classifier).
func classificationStats(kind domain.ModelKind, examples []*domain.Example) domain.DatasetStats {
	stats := genericStats(kind, examples)

	stats.LabelBalance = map[string]int{}

	for _, e := range examples {
		var p seqClassifierPayload
		if len(e.Payload) > 0 {
			_ = json.Unmarshal(e.Payload, &p)
		}

		label := strings.TrimSpace(p.Label)
		if label == "" {
			label = strings.TrimSpace(e.Output) // fallback to legacy Output.
		}

		if label != "" && !e.Duplicate && !e.Flagged {
			stats.LabelBalance[label]++
		}
	}

	return stats
}

// genericStats computes the shared counters (total, duplicates, sources)
// used by every schema's Stats.
func genericStats(kind domain.ModelKind, examples []*domain.Example) domain.DatasetStats {
	stats := domain.DatasetStats{
		Kind:         kind,
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
