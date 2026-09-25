package domain

// NERConfig carries the token-classification hyperparameters that the Go
// layer can pass to the trainer worker (mirrored by the Python NERConfig in
// trainer/tasks/ner.py).
type NERConfig struct {
	// MaxLength truncates/pads each sliding window to this many tokens.
	MaxLength int `json:"max_length,omitempty"`
	// Stride is the number of overlapping tokens between consecutive sliding
	// windows used to segment long documents (0 = default 64).
	Stride int `json:"stride,omitempty"`
	// LabelScheme selects the tagging scheme: "BIO" (default) or "BILOU".
	LabelScheme string `json:"label_scheme,omitempty"`
	// Epochs / LearningRate / BatchSize override the defaults.
	Epochs       int     `json:"epochs,omitempty"`
	LearningRate float64 `json:"learning_rate,omitempty"`
	BatchSize    int     `json:"batch_size,omitempty"`
}

// EntitySpan is one labeled character span over the input text.
type EntitySpan struct {
	Start int    `json:"start"`
	End   int    `json:"end"`
	Label string `json:"label"`
}

// SpanValidationError describes a single span validation failure.
type SpanValidationError struct {
	Index int    `json:"index"`
	Msg   string `json:"msg"`
}

// Error implements the error interface.
func (e *SpanValidationError) Error() string { return e.Msg }

// ValidateEntitySpans checks spans against the text: bounds must be within
// the text, end > start, labels non-empty, and spans must not overlap
// (allowing shared boundaries). allowedLabels, when non-empty, restricts the
// label set. The offset is added to every span index before validation (used
// for CSV rows that carry absolute offsets into a shared document).
func ValidateEntitySpans(text string, spans []EntitySpan, allowedLabels []string) error {
	if len(text) == 0 {
		return &SpanValidationError{Index: -1, Msg: "text is required"}
	}

	allowed := map[string]bool{}
	for _, l := range allowedLabels {
		allowed[l] = true
	}

	for i, s := range spans {
		if s.Start < 0 || s.End > len(text) {
			return &SpanValidationError{Index: i, Msg: "span out of bounds"}
		}

		if s.End <= s.Start {
			return &SpanValidationError{Index: i, Msg: "span end must be greater than start"}
		}

		label := s.Label
		if label == "" {
			return &SpanValidationError{Index: i, Msg: "span label is required"}
		}

		if len(allowed) > 0 && !allowed[label] {
			return &SpanValidationError{Index: i, Msg: "label not in allowed label set"}
		}
	}

	// Overlap check: O(n^2) is fine for annotation-sized span lists.
	for i := range spans {
		for j := i + 1; j < len(spans); j++ {
			a, b := spans[i], spans[j]
			if a.Start < b.End && b.Start < a.End {
				return &SpanValidationError{Index: j, Msg: "spans overlap"}
			}
		}
	}

	return nil
}

// PerEntityMetrics are the entity-level precision/recall/F1 for one label.
type PerEntityMetrics struct {
	Precision float64 `json:"precision"`
	Recall    float64 `json:"recall"`
	F1        float64 `json:"f1"`
	Support   int     `json:"support"`
}

// EntityMetrics aggregates entity-level P/R/F1 plus partial-match diagnostics
// and the per-entity breakdown (the NER analogue of the classifier's
// per-class metrics).
type EntityMetrics struct {
	// MicroF1 is the headline metric: TP / (TP + FP + FN) over all entities.
	MicroF1 float64 `json:"micro_f1"`
	// StrictMicroF1 equals MicroF1 (exact span + label). Kept explicit so the
	// metrics.json schema mirrors seqeval's "strict" mode name.
	StrictMicroF1 float64 `json:"strict_micro_f1"`
	// PartialMicroF1 credits overlapping spans with a fractional match
	// (character overlap ratio) — a diagnostic for near-miss boundaries.
	PartialMicroF1 float64                     `json:"partial_micro_f1"`
	PerEntity      map[string]PerEntityMetrics `json:"per_entity,omitempty"`
}

// Entity is one predicted entity returned by token-classifier inference.
type Entity struct {
	Text  string  `json:"text"`
	Label string  `json:"label"`
	Start int     `json:"start"`
	End   int     `json:"end"`
	Score float64 `json:"score"`
}

// NERPrediction is the structured result of a token_classifier inference call
// (what /predict returns for token_classifier deployments).
type NERPrediction struct {
	Entities []Entity `json:"entities"`
}

// NERInferenceEngine serves token-classification (span) predictions from a
// deployed fine-tuned model. Implementations must be kind-safe: callers guard
// on Deployment.Kind == KindTokenClassifier.
type NERInferenceEngine interface {
	PredictSpans(
		job *TrainingJob,
		trainingExamples []*Example,
		input string,
		labelMap map[string]int,
	) (NERPrediction, error)
}
