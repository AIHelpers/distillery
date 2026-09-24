package simulation

import (
	"strings"

	"distillery/internal/domain"
)

// InferenceEngine implements domain.InferenceEngine. It simulates a served
// fine-tuned model (vLLM/TGI in production) using nearest-neighbor lookup
// against the training set so the deployed-endpoint demo behaves plausibly
// without requiring real trained weights.
type InferenceEngine struct{}

func NewInferenceEngine() *InferenceEngine { return &InferenceEngine{} }

func (e *InferenceEngine) Predict(
	job *domain.TrainingJob,
	trainingExamples []*domain.Example,
	input string,
) (output string, confidence float64) {
	if len(trainingExamples) == 0 {
		return "(no training data available)", 0
	}

	inputTokens := tokenize(input)

	var best *domain.Example

	bestScore := -1.0

	for _, ex := range trainingExamples {
		score := jaccard(inputTokens, tokenize(ex.Input))
		if score > bestScore {
			bestScore = score
			best = ex
		}
	}

	if best == nil {
		return "(unable to produce a prediction)", 0
	}

	// Confidence blends similarity to the nearest training example with the
	// model's own eval accuracy, so it reads as a genuine model call.
	confidence = bestScore*0.5 + 0.5
	if job.Metrics != nil {
		confidence = confidence*0.5 + job.Metrics.EvalAccuracy*0.5
	}

	if confidence > 0.99 {
		confidence = 0.99
	}

	return best.Output, Round2(confidence)
}

// PredictClassification implements domain.ClassifierInferenceEngine. It reuses
// the same nearest-neighbor approach as Predict but maps the matched training
// example's label into a full per-class score distribution (the top label gets
// the confidence mass, the rest get a small share) so the seq_classifier API
// shape is exercised without real trained weights.
func (e *InferenceEngine) PredictClassification(
	job *domain.TrainingJob,
	trainingExamples []*domain.Example,
	input string,
	labelMap map[string]int,
	threshold float64,
) (domain.ClassifierPrediction, error) {
	if len(labelMap) == 0 {
		return domain.ClassifierPrediction{}, domain.ErrNotFound
	}

	label, conf := e.Predict(job, trainingExamples, input)

	// Distribute the confidence mass: top label gets `conf`, the remainder is
	// split among the other labels (weighted so the ordering is plausible).
	scores := make(map[string]float64, len(labelMap))
	remaining := 1 - conf

	rest := make([]string, 0, len(labelMap)-1)
	for lbl := range labelMap {
		if lbl != label {
			rest = append(rest, lbl)
		}
	}

	if len(rest) == 0 {
		scores[label] = 1
	} else {
		share := remaining / float64(len(rest))
		for _, lbl := range rest {
			scores[lbl] = Round2(share)
		}

		scores[label] = Round2(conf)
	}

	return domain.ClassifierPrediction{
		Label:  label,
		Score:  Round2(conf),
		Scores: scores,
		// Compare the same rounded score that is returned to callers.
		BelowThreshold: Round2(conf) < threshold,
	}, nil
}

func tokenize(s string) map[string]bool {
	words := strings.Fields(strings.ToLower(s))

	set := make(map[string]bool, len(words))
	for _, w := range words {
		set[strings.Trim(w, ".,!?;:\"'()")] = true
	}

	return set
}

func jaccard(a, b map[string]bool) float64 {
	if len(a) == 0 && len(b) == 0 {
		return 1
	}

	inter := 0

	seen := map[string]bool{}
	for w := range a {
		seen[w] = true
	}

	for w := range b {
		seen[w] = true
	}

	union := len(seen)

	for w := range a {
		if b[w] {
			inter++
		}
	}

	if union == 0 {
		return 0
	}

	return float64(inter) / float64(union)
}
