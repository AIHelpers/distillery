package simulation

import (
	"testing"

	"distillery/internal/domain"
)

func TestModelSelector_ListIncludesSeqClassifierModels(t *testing.T) {
	t.Parallel()

	ms := NewModelSelector()
	models := ms.ListBaseModels()

	var seqModels []domain.BaseModel

	for _, m := range models {
		if m.Kind == domain.KindSeqClassifier {
			seqModels = append(seqModels, m)
		}
	}

	if len(seqModels) < 5 {
		t.Fatalf("expected >=5 seq_classifier catalog entries, got %d", len(seqModels))
	}

	// The selector's smallest seq_classifier candidate should be DistilBERT
	// and run on CPU (no VRAM requirement).
	got := ms.Select(domain.KindSeqClassifier, domain.DatasetStats{Total: 20}, domain.SelectionConstraints{CPUOnly: true})
	if got.Kind != domain.KindSeqClassifier {
		t.Fatalf("Select(seq_classifier) returned kind %q", got.Kind)
	}

	if !got.Capabilities.RunsOnCPU {
		t.Fatalf("seq_classifier model %q should run on CPU", got.Name)
	}
}

func TestInferenceEngine_PredictClassification(t *testing.T) {
	t.Parallel()

	engine := NewInferenceEngine()
	job := &domain.TrainingJob{
		Kind: domain.KindSeqClassifier,
		Metrics: &domain.TrainingMetrics{
			EvalAccuracy: 0.9,
			LabelMap:     map[string]int{"billing": 0, "technical": 1, "other": 2},
		},
	}
	labelMap := map[string]int{"billing": 0, "technical": 1, "other": 2}

	examples := []*domain.Example{
		{Input: "my invoice is wrong", Output: "billing"},
		{Input: "server is down", Output: "technical"},
		{Input: "thanks", Output: "other"},
	}

	pred, err := engine.PredictClassification(job, examples, "my bill is incorrect", labelMap, 0.5)
	if err != nil {
		t.Fatalf("PredictClassification returned error: %v", err)
	}

	if pred.Label == "" {
		t.Fatal("expected a non-empty predicted label")
	}

	if pred.Score <= 0 || pred.Score > 1 {
		t.Fatalf("expected score in (0,1], got %f", pred.Score)
	}

	if len(pred.Scores) != len(labelMap) {
		t.Fatalf("expected scores for all %d labels, got %d", len(labelMap), len(pred.Scores))
	}

	// The returned label must be the argmax of the scores map.
	best := ""

	bestScore := -1.0
	for lbl, s := range pred.Scores {
		if s > bestScore {
			bestScore = s
			best = lbl
		}
	}

	if best != pred.Label {
		t.Fatalf("argmax label %q != reported label %q", best, pred.Label)
	}

	// A very high threshold should flag below_threshold.
	predHigh, err := engine.PredictClassification(job, examples, "my bill is incorrect", labelMap, 0.99)
	if err != nil {
		t.Fatalf("PredictClassification(high threshold) error: %v", err)
	}

	if !predHigh.BelowThreshold {
		t.Fatal("expected below_threshold=true with threshold 0.99")
	}
}
