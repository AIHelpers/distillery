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
	Text  string              `json:"text"`
	Spans []domain.EntitySpan `json:"entities"`
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

	// Entities are optional — unlabeled text is a valid (neutral) sample —
	// but when present, every span must satisfy the NER span contract:
	// in-bounds, end > start, non-overlapping, valid label.
	if err := domain.ValidateEntitySpans(p.Text, p.Spans, nil); err != nil {
		return &schemaError{kind: domain.KindTokenClassifier, msg: "invalid entity span: " + err.Error()}
	}

	return nil
}

// Stats implements domain.DatasetSchema.
func (s *TokenClassifierSchema) Stats(examples []*domain.Example) domain.DatasetStats {
	stats := genericStats(domain.KindTokenClassifier, examples)

	stats.LabelBalance = map[string]int{}

	for _, e := range examples {
		var p TokenClassifierPayload

		if len(e.Payload) > 0 && json.Unmarshal(e.Payload, &p) == nil {
			for _, sp := range p.Spans {
				if strings.TrimSpace(sp.Label) != "" && !e.Duplicate && !e.Flagged {
					stats.LabelBalance[strings.TrimSpace(sp.Label)]++
				}
			}
		}
	}

	return stats
}

// Formats implements domain.DatasetSchema.
func (s *TokenClassifierSchema) Formats() []domain.ImportFormat {
	return []domain.ImportFormat{domain.FormatJSONL, domain.FormatCoNLL}
}

// --- Embedding ---.

// EmbeddingDataFormat denotes the shape of an embedding training example.
type EmbeddingDataFormat string

const (
	// EmbeddingPair is {"query","positive"} — minimum viable, in-batch negatives.
	EmbeddingPair EmbeddingDataFormat = "pair"
	// EmbeddingTriplet is {"query","positive","negative"} — better, hard negatives.
	EmbeddingTriplet EmbeddingDataFormat = "triplet"
	// EmbeddingDocsOnly is a set of documents with no queries — synthetic
	// query generation is used to bootstrap pairs.
	EmbeddingDocsOnly EmbeddingDataFormat = "docs_only"
)

// EmbeddingSchema validates payloads for text embedding training.
// Supported shapes:
//
//	pair:     {"query","positive"}
//	triplet:  {"query","positive","negative"}
//	docs only: {"document"} (no query — used for synthetic query generation)
type EmbeddingSchema struct{}

// embeddingPairPayload is the JSON shape for a pair-style embedding example.
type embeddingPairPayload struct {
	Query    string `json:"query"`
	Positive string `json:"positive"`
}

// embeddingTripletPayload is the JSON shape for a triplet-style embedding example.
type embeddingTripletPayload struct {
	Query    string `json:"query"`
	Positive string `json:"positive"`
	Negative string `json:"negative"`
}

// EmbeddingDocPayload is the JSON shape for a docs-only embedding example.
type EmbeddingDocPayload struct {
	Document string `json:"document"`
}

// Kind implements domain.DatasetSchema.
func (s *EmbeddingSchema) Kind() domain.ModelKind { return domain.KindEmbedding }

// Validate implements domain.DatasetSchema.
func (s *EmbeddingSchema) Validate(payload json.RawMessage) error {
	var probe struct {
		Query    string `json:"query"`
		Positive string `json:"positive"`
		Negative string `json:"negative"`
		Document string `json:"document"`
	}

	err := json.Unmarshal(payload, &probe)
	if err != nil {
		return &schemaError{kind: domain.KindEmbedding, msg: "payload must be a JSON object with query/positive (pair), query/positive/negative (triplet), or document (docs-only) fields"}
	}

	// Docs-only.
	if strings.TrimSpace(probe.Document) != "" &&
		strings.TrimSpace(probe.Query) == "" && strings.TrimSpace(probe.Positive) == "" {
		if len(strings.TrimSpace(probe.Document)) < 2 {
			return &schemaError{kind: domain.KindEmbedding, msg: "document is too short"}
		}

		return nil
	}

	// Pair.
	if strings.TrimSpace(probe.Query) == "" {
		return &schemaError{kind: domain.KindEmbedding, msg: "query is required"}
	}

	if strings.TrimSpace(probe.Positive) == "" {
		return &schemaError{kind: domain.KindEmbedding, msg: "positive (relevant document) is required"}
	}

	// Triplet.
	if strings.TrimSpace(probe.Negative) != "" && len(strings.TrimSpace(probe.Negative)) < 2 {
		return &schemaError{kind: domain.KindEmbedding, msg: "negative is too short"}
	}

	return nil
}

// Stats implements domain.DatasetSchema.
func (s *EmbeddingSchema) Stats(examples []*domain.Example) domain.DatasetStats {
	return embeddingStats(examples)
}

// Formats implements domain.DatasetSchema.
func (s *EmbeddingSchema) Formats() []domain.ImportFormat {
	return []domain.ImportFormat{domain.FormatCSV, domain.FormatJSONL}
}

// --- Reranker ---.

// RerankerSchema validates payloads for query/document relevance scoring:
// {"query": "...", "document": "...", "label": 0|1 or graded}.
// Label: bool (true/false), or float/int graded relevance (0.0-1.0).
type RerankerSchema struct{}

// rerankerPayload is the JSON object shape for a reranker example.
type rerankerPayload struct {
	Query    string `json:"query"`
	Document string `json:"document"`
	// Label may be bool (0/1) or a graded relevance float (0.0-1.0).
	Label *float64 `json:"label,omitempty"`
	// Relevance is the float-only synonym for Label (accept both spellings).
	Relevance *float64 `json:"relevance,omitempty"`
}

// Kind implements domain.DatasetSchema.
func (s *RerankerSchema) Kind() domain.ModelKind { return domain.KindReranker }

// Validate implements domain.DatasetSchema.
func (s *RerankerSchema) Validate(payload json.RawMessage) error {
	var p rerankerPayload

	err := json.Unmarshal(payload, &p)
	if err != nil {
		return &schemaError{kind: domain.KindReranker, msg: "payload must be a JSON object with query/document/label fields"}
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
	return rerankerStats(examples)
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

// embeddingStats computes stats for embedding datasets, distinguishing the
// number of pair vs triplet vs docs-only examples.
func embeddingStats(examples []*domain.Example) domain.DatasetStats {
	stats := genericStats(domain.KindEmbedding, examples)
	pairs, triplets, docs := 0, 0, 0

	for _, e := range examples {
		var p struct {
			Query    string `json:"query"`
			Negative string `json:"negative"`
			Document string `json:"document"`
		}
		if len(e.Payload) > 0 {
			_ = json.Unmarshal(e.Payload, &p)
		}

		switch {
		case strings.TrimSpace(p.Document) != "" && strings.TrimSpace(p.Query) == "":
			docs++
		case strings.TrimSpace(p.Negative) != "":
			triplets++
		default:
			pairs++
		}
	}

	stats.LabelBalance = map[string]int{
		"pairs":    pairs,
		"triplets": triplets,
		"docs":     docs,
	}

	return stats
}

// rerankerStats computes stats for reranker datasets (graded pairs).
func rerankerStats(examples []*domain.Example) domain.DatasetStats {
	stats := genericStats(domain.KindReranker, examples)

	var graded, binary int

	for _, e := range examples {
		var p rerankerPayload
		if len(e.Payload) > 0 {
			_ = json.Unmarshal(e.Payload, &p)
		}

		label := 0.0
		if p.Label != nil {
			label = *p.Label
		} else if p.Relevance != nil {
			label = *p.Relevance
		}

		if label == 0 || label == 1 {
			binary++
		} else {
			graded++
		}
	}

	stats.LabelBalance = map[string]int{
		"binary_0_1": binary,
		"graded":     graded,
	}

	return stats
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
