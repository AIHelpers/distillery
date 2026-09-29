package training_test

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"distillery/internal/infra/training"
)

// TestReadMetricsTablePropagation guards the Go/Python contract for the
// tabular / time-series kinds: the worker nests the domain-shaped metrics
// under "table" and reports failures with status "failed" + "error".
func TestReadMetricsTablePropagation(t *testing.T) {
	t.Parallel()

	dir := t.TempDir()

	metrics := `{"status":"completed","kind":"tabular","train_examples":240,"epoch":1,"eval_loss":0.7,"table":{` +
		`"primary_metric":"roc_auc","primary_value":0.81,"higher_is_better":true,` +
		`"baselines":{"roc_auc":0.5},"improvement_over_baseline":0.31,"table_backend":"lightgbm","split_used":"stratified",` +
		`"leaderboard":[{"model":"lightgbm","score":0.8,"chosen":true},{"model":"majority_class","score":0.5,"baseline":true}],` +
		`"feature_importance":[{"feature":"tenure","impact":0.4}],` +
		`"confusion_matrix":[[10,2],[3,9]],"label_map":{"no":0,"yes":1},"roc_curve":[[0,0],[0.2,0.7],[1,1]],` +
		`"calibration":{"method":"platt","brier_before":0.2,"brier_after":0.19,"ece_before":0.07,"ece_after":0.05},` +
		`"feature_schema":{"features":[{"name":"tenure","type":"numeric","min":1,"max":60},` +
		`{"name":"plan","type":"categorical","categories":["basic","pro"]}],"target":"churn","task":"classification","classes":["no","yes"]}}}`

	err := os.WriteFile(filepath.Join(dir, "metrics.json"), []byte(metrics), 0o600)
	if err != nil {
		t.Fatal(err)
	}

	m, err := training.NewLocalTrainer(&training.Config{}).ReadMetrics(context.Background(), dir, nil)
	if err != nil {
		t.Fatal(err)
	}

	if m.Primary != "roc_auc" || m.PrimaryVal != 0.81 || !m.HigherIsBetter || m.TableBackend != "lightgbm" {
		t.Errorf("headline metrics lost: %+v", m)
	}

	if m.TrainExamples != 240 || m.Epochs != 1 || len(m.Leaderboard) != 2 || !m.Leaderboard[1].Baseline {
		t.Errorf("leaderboard / counters lost: %+v", m)
	}

	if m.FeatureSchema == nil || len(m.FeatureSchema.Features) != 2 || m.FeatureSchema.Classes[1] != "yes" {
		t.Errorf("feature schema lost: %+v", m.FeatureSchema)
	}

	if len(m.ConfusionMatrix) != 2 || len(m.ROCCurve) != 3 || m.Calibration == nil || m.Calibration.ECEAfter != 0.05 {
		t.Errorf("artifacts lost: %+v", m)
	}
}

func TestReadMetricsFailureCarriesWorkerError(t *testing.T) {
	t.Parallel()

	dir := t.TempDir()
	body := `{"status":"failed","error":"the target has a single class; nothing to learn"}`

	err := os.WriteFile(filepath.Join(dir, "metrics.json"), []byte(body), 0o600)
	if err != nil {
		t.Fatal(err)
	}

	_, err = training.NewLocalTrainer(&training.Config{}).ReadMetrics(context.Background(), dir, nil)
	if err == nil || !strings.Contains(err.Error(), "single class") {
		t.Errorf("worker error message must reach the job, got %v", err)
	}
}
