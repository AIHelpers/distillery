package training_test

import (
	"context"
	"os"
	"path/filepath"
	"testing"

	"distillery/internal/infra/training"
)

// TestReadMetricsClassifierPropagation guards the Go/Python metrics.json
// contract: the worker writes a fractional epoch and flat classifier keys;
// the adapter must round the epoch and propagate every classifier field.
func TestReadMetricsClassifierPropagation(t *testing.T) {
	dir := t.TempDir()

	metrics := `{` +
		`"status":"completed","kind":"seq_classifier",` +
		`"accuracy":0.9,"macro_f1":0.85,"weighted_f1":0.87,` +
		`"baseline_macro_f1":0.4,"delta_macro_f1":0.45,"eval_loss":0.32,` +
		`"epoch":3.75,"train_examples":120,"default_threshold":0.5,` +
		`"max_length":256,"multi_label":false,` +
		`"per_class":{"a":{"precision":1.0,"recall":0.5,"f1":0.6667,"support":2}},` +
		`"confusion_matrix":[[3,1],[0,2]],` +
		`"label_map":{"a":0,"b":1},` +
		`"threshold_sweep":[{"threshold":0.5,"macro_f1":0.8,"coverage":0.9}]}`

	if err := os.WriteFile(filepath.Join(dir, "metrics.json"), []byte(metrics), 0o600); err != nil {
		t.Fatalf("write metrics: %v", err)
	}

	lt := training.NewLocalTrainer(&training.Config{})

	m, err := lt.ReadMetrics(context.Background(), dir, nil)
	if err != nil {
		t.Fatalf("ReadMetrics: %v", err)
	}

	if m.Epochs != 4 {
		t.Errorf("Epochs = %d, want 4 (3.75 rounded)", m.Epochs)
	}

	if m.EvalAccuracy != 0.9 {
		t.Errorf("EvalAccuracy = %v, want 0.9", m.EvalAccuracy)
	}

	if m.TrainExamples != 120 {
		t.Errorf("TrainExamples = %d, want 120", m.TrainExamples)
	}

	if m.MacroF1 != 0.85 || m.WeightedF1 != 0.87 {
		t.Errorf("macro/weighted F1 not propagated: %+v", m)
	}

	if m.BaselineMacroF1 != 0.4 || m.DeltaMacroF1 != 0.45 {
		t.Errorf("baseline/delta not propagated: %+v", m)
	}

	if m.DefaultThreshold != 0.5 || m.MaxLength != 256 || m.MultiLabel {
		t.Errorf("threshold/max_length/multi_label not propagated: %+v", m)
	}

	if m.PerClass["a"].F1 != 0.6667 {
		t.Errorf("per-class not propagated: %+v", m.PerClass)
	}

	if len(m.ConfusionMatrix) != 2 || m.ConfusionMatrix[0][1] != 1 {
		t.Errorf("confusion matrix not propagated: %+v", m.ConfusionMatrix)
	}

	if m.LabelMap["a"] != 0 || m.LabelMap["b"] != 1 {
		t.Errorf("label map not propagated: %+v", m.LabelMap)
	}

	if len(m.ThresholdSweep) != 1 || m.ThresholdSweep[0].Coverage != 0.9 {
		t.Errorf("threshold sweep not propagated: %+v", m.ThresholdSweep)
	}
}
