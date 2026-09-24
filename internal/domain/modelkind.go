package domain

// ModelKind identifies the architectural type of a model. It drives dataset
// schema, trainer module selection, export format, and inference result shape.
// Making model kind a first-class concept removes the assumption that every
// task is a causal-LM fine-tune.
type ModelKind string

const (
	// KindCausalLM is the existing behavior: text-to-text generation via a
	// causal language model (e.g. instruction following, code generation).
	KindCausalLM ModelKind = "causal_lm"
	// KindSeqClassifier is a sentence/document sequence classifier
	// (e.g. sentiment, topic, intent) producing label + scores.
	KindSeqClassifier ModelKind = "seq_classifier"
	// KindTokenClassifier is a token-level (NER/span) classifier producing
	// labeled spans over the input.
	KindTokenClassifier ModelKind = "token_classifier"
	// KindEmbedding is an encoder producing a vector per input.
	KindEmbedding ModelKind = "embedding"
	// KindReranker is a cross-encoder producing relevance scores for
	// query/document pairs.
	KindReranker ModelKind = "reranker"
	// KindPreferenceLM is a preference/reward model trained with DPO.
	KindPreferenceLM ModelKind = "preference_lm"
)

// ValidModelKinds returns the set of supported model kinds. Wave 2 will add
// KindVisionLM, KindASR, KindTabular, and KindTimeSeries.
func ValidModelKinds() []ModelKind {
	return []ModelKind{
		KindCausalLM,
		KindSeqClassifier,
		KindTokenClassifier,
		KindEmbedding,
		KindReranker,
		KindPreferenceLM,
	}
}

// IsValidModelKind reports whether k is a supported model kind.
func IsValidModelKind(k ModelKind) bool {
	for _, valid := range ValidModelKinds() {
		if k == valid {
			return true
		}
	}

	return false
}

// DefaultModelKind is used when an older record (from before kind became
// first-class) is migrated. Existing behavior maps to causal_lm.
const DefaultModelKind ModelKind = KindCausalLM

// KindDescription returns a plain-English description of a model kind for
// the UI's "choose your model kind" cards.
func (k ModelKind) Description() string {
	switch k {
	case KindCausalLM:
		return "Generate or transform natural language. Best for assistants, code, summarization, extraction of structured text."
	case KindSeqClassifier:
		return "Assign a label (with confidence scores) to a whole input. Best for sentiment, intent, topic, spam."
	case KindTokenClassifier:
		return "Tag spans of the input — e.g. named entities (people, places, dates), PII, or custom labels."
	case KindEmbedding:
		return "Turn each input into a dense vector for search, clustering, dedup, or as a feature for other models."
	case KindReranker:
		return "Score query-document pairs by relevance. Best for improving retrieval quality on top of an embedder."
	case KindPreferenceLM:
		return "Learn human preferences (DPO) to align a generator or rank candidate outputs."
	default:
		return string(k)
	}
}
