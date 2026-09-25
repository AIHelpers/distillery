package simulation

import (
	"strings"

	"distillery/internal/domain"
)

// InferenceEngine implements domain.InferenceEngine (plus
// domain.EmbeddingEngine and domain.RerankerInferenceEngine). It simulates a
// served fine-tuned model (vLLM/TGI in production) using nearest-neighbor
// lookup against the training set so the deployed-endpoint demo behaves
// plausibly without requiring real trained weights.
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

// Embed implements domain.EmbeddingEngine by returning a deterministic
// vector per input so the /embed endpoint has a working shape without real
// trained weights. The dim matches the training job's recorded
// embedding_dim (default 768 for bge/small encoders).
func (e *InferenceEngine) Embed(
	job *domain.TrainingJob,
	inputs []string,
	embedType string,
) domain.EmbeddingResult {
	dim := 768
	if job.Metrics != nil && job.Metrics.EmbeddingDim > 0 {
		dim = job.Metrics.EmbeddingDim
	}

	// Deterministic pseudo-vector hashing: each token contributes a stable
	// pseudo-random basis vector projected onto the dim. Query/document type
	// shifts the seed so the two prefixes produce distinct vectors.
	seedShift := int64(0)
	if embedType == "document" {
		seedShift = 7
	}

	vecs := make([][]float32, 0, len(inputs))
	for _, in := range inputs {
		vec := make([]float32, dim)
		tokens := tokenize(in)

		if len(tokens) == 0 {
			tokens["__empty__"] = true
		}

		for tok := range tokens {
			h := hashString(tok) + uint64(seedShift)
			for i := range dim {
				// SplitMix64-style PRNG per (token, dimension) for stability.
				x := h + uint64(i)*0x9E3779B97F4A7C15
				x ^= x >> 30
				x *= 0xBF58476D1CE4E5B9
				x ^= x >> 27
				x *= 0x94D049BB133111EB
				x ^= x >> 31
				f := float64(x%100000) / 100000.0
				signed := f*2 - 1
				vec[i] += float32(signed)
			}
		}

		sum := 0.0
		for _, v := range vec {
			sum += float64(v) * float64(v)
		}

		if sum > 0 {
			inv := float32(1.0 / sqrt(sum))
			for i := range vec {
				vec[i] *= inv
			}
		}

		vecs = append(vecs, vec)
	}

	return domain.EmbeddingResult{Dim: dim, Vectors: vecs}
}

// Rerank implements domain.RerankerInferenceEngine using Jaccard overlap
// between the query and each document to produce a deterministic ranking.
func (e *InferenceEngine) Rerank(
	_ *domain.TrainingJob,
	query string,
	documents []string,
) domain.RerankResult {
	queryTokens := tokenize(query)

	type scored struct {
		idx   int
		score float64
	}

	scoredDocs := make([]scored, 0, len(documents))
	for i, doc := range documents {
		s := jaccard(queryTokens, tokenize(doc))
		scoredDocs = append(scoredDocs, scored{idx: i, score: s})
	}

	// Sort desc by score, stable by index.
	for i := 0; i < len(scoredDocs); i++ {
		for j := i + 1; j < len(scoredDocs); j++ {
			if scoredDocs[j].score > scoredDocs[i].score ||
				(scoredDocs[j].score == scoredDocs[i].score && scoredDocs[j].idx < scoredDocs[i].idx) {
				scoredDocs[i], scoredDocs[j] = scoredDocs[j], scoredDocs[i]
			}
		}
	}

	ranking := make([]domain.RerankItem, 0, len(scoredDocs))
	for _, s := range scoredDocs {
		ranking = append(ranking, domain.RerankItem{Index: s.idx, Score: Round2(s.score)})
	}

	return domain.RerankResult{Ranking: ranking}
}

func hashString(s string) uint64 {
	var h uint64 = 14695981039346656037
	for i := range len(s) {
		h ^= uint64(s[i])
		h *= 1099511628211
	}

	return h
}

func sqrt(x float64) float64 {
	// Newton's method; good enough for normalisation.
	if x <= 0 {
		return 0
	}

	z := x
	for range 40 {
		z = (z + x/z) / 2
	}

	return z
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
