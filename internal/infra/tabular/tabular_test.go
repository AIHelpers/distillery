package tabular_test

import (
	"archive/zip"
	"bytes"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"distillery/internal/domain"
	"distillery/internal/infra/tabular"
)

// pythonWithNumpy returns a usable interpreter or skips the test.
func pythonWithNumpy(t *testing.T) string {
	t.Helper()

	for _, bin := range []string{os.Getenv("PYTHON_BIN"), "python3", "python"} {
		if bin == "" {
			continue
		}

		if exec.Command(bin, "-c", "import numpy").Run() == nil {
			return bin
		}
	}

	t.Skip("python with numpy is not available")

	return ""
}

func repoRoot(t *testing.T) string {
	t.Helper()

	root, err := filepath.Abs("../../..")
	if err != nil {
		t.Fatal(err)
	}

	_, err = os.Stat(filepath.Join(root, "trainer", "run.py"))
	if err != nil {
		t.Skip("trainer package not found")
	}

	return root
}

// trainJob runs the real Python trainer into jobsDir/<id>.
func trainJob(t *testing.T, py, root, jobsDir, id string, cfg map[string]interface{}, rows []map[string]interface{}) map[string]interface{} {
	t.Helper()

	dir := filepath.Join(jobsDir, id)

	err := os.MkdirAll(dir, 0o755)
	if err != nil {
		t.Fatal(err)
	}

	var ds bytes.Buffer

	for _, r := range rows {
		b, _ := json.Marshal(r) //nolint:errchkjson // test fixture marshalling
		ds.Write(b)
		ds.WriteByte('\n')
	}

	cfgB, _ := json.Marshal(cfg) //nolint:errchkjson // test fixture marshalling

	_ = os.WriteFile(filepath.Join(dir, "dataset.jsonl"), ds.Bytes(), 0o600)
	_ = os.WriteFile(filepath.Join(dir, "config.json"), cfgB, 0o600)

	cmd := exec.Command(py, "-m", "trainer.run", "--job-dir", dir, "--config", filepath.Join(dir, "config.json"),
		"--dataset", filepath.Join(dir, "dataset.jsonl"))
	cmd.Dir = root
	cmd.Env = append(os.Environ(), "PYTHONPATH="+root)

	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("trainer failed: %v\n%s", err, out)
	}

	raw, err := os.ReadFile(filepath.Join(dir, "metrics.json"))
	if err != nil {
		t.Fatal(err)
	}

	var m map[string]interface{}

	err = json.Unmarshal(raw, &m)
	if err != nil || m["status"] != "completed" {
		t.Fatalf("metrics: %v %s", err, raw)
	}

	return m
}

//nolint:cyclop,funlen,gocognit,paralleltest // long sequential scenario that shares one worker process
func TestPythonWorkerServingAndExport(t *testing.T) {
	py := pythonWithNumpy(t)
	root := repoRoot(t)
	jobsDir := t.TempDir()

	// --- tabular ---.
	var rows []map[string]interface{} //nolint:prealloc // test fixture

	for i := range 240 {
		tenure := 1 + (i*7)%60
		plan := []string{"basic", "pro", "ent"}[i%3]
		churn := "no"

		if tenure < 20 && plan == "basic" || i%9 == 0 {
			churn = "yes"
		}

		rows = append(rows, map[string]interface{}{"tenure": tenure, "plan": plan, "churn": churn})
	}

	m := trainJob(t, py, root, jobsDir, "job-tab", map[string]interface{}{
		"kind": "tabular",
		"tabular": map[string]interface{}{
			"target": "churn", "task": "classification", "metric": "roc_auc", "time_budget_sec": 2,
			"columns": map[string]string{"tenure": "numeric", "plan": "categorical", "churn": "categorical"},
		},
	}, rows)

	var tm domain.TrainingMetrics
	{
		b, _ := json.Marshal(m["table"]) //nolint:errchkjson // test fixture marshalling

		err := json.Unmarshal(b, &tm)
		if err != nil {
			t.Fatal(err)
		}
	}

	if tm.FeatureSchema == nil || len(tm.FeatureSchema.Features) != 2 || tm.PrimaryVal <= 0.5 {
		t.Fatalf("metrics do not decode into the domain shape: %+v", tm)
	}

	eng := tabular.NewEngine(tabular.ServingConfig{PythonBin: py, JobsDir: jobsDir, WorkDir: root}, nil, nil)
	defer eng.Close()

	job := &domain.TrainingJob{ID: "job-tab", Kind: domain.KindTabular, Metrics: &tm}

	v, err := tm.FeatureSchema.ValidateRow(map[string]interface{}{"tenure": 5, "plan": "BASIC"})
	if err != nil {
		t.Fatal(err)
	}

	pred, err := eng.PredictTable(job, nil, tm.FeatureSchema, v)
	if err != nil {
		t.Fatalf("predict: %v", err)
	}

	if pred.Probability == nil || pred.Method != "shap" || len(pred.TopFactors) == 0 || pred.Prediction == nil {
		t.Errorf("unexpected prediction: %+v", pred)
	}

	// The worker stays alive across calls.
	for i := range 3 {
		_, err := eng.PredictTable(job, nil, tm.FeatureSchema, v)
		if err != nil {
			t.Fatalf("call %d: %v", i, err)
		}
	}

	// Export: the package must run stand-alone.
	exp := tabular.NewExporter(nil, jobsDir, filepath.Join(root, "trainer"))

	zipB, name, err := exp.BuildExport(&domain.Task{Name: "Churn model"}, job)
	if err != nil {
		t.Fatalf("export: %v", err)
	}

	if !strings.HasSuffix(name, ".zip") {
		t.Errorf("name = %q", name)
	}

	out := t.TempDir()
	unzip(t, zipB, out)

	for _, f := range []string{"schema.json", "tablelib.py", "predict_example.py", "README.md"} {
		_, err := os.Stat(filepath.Join(out, f))
		if err != nil {
			t.Errorf("export is missing %s", f)
		}
	}

	cmd := exec.Command(py, "predict_example.py")
	cmd.Dir = out

	exOut, err := cmd.CombinedOutput()
	if err != nil || !strings.Contains(string(exOut), "prediction") {
		t.Errorf("exported example failed: %v\n%s", err, exOut)
	}

	// --- forecast ---.
	var frows []map[string]interface{} //nolint:prealloc // test fixture

	for i := range 100 {
		day := fmt.Sprintf("2024-%02d-%02d", 1+i/28, 1+i%28)
		frows = append(frows,
			map[string]interface{}{"ds": day, "v": float64(50 + (i%7)*3), "sku": "a"},
			map[string]interface{}{"ds": day, "v": float64(20 + (i%7)*2), "sku": "b"})
	}

	trainJob(t, py, root, jobsDir, "job-fc", map[string]interface{}{
		"kind": "time_series",
		"forecast": map[string]interface{}{
			"timestamp": "ds", "target": "v", "item_id": "sku", "frequency": "D", "horizon": 7, "backtest_windows": 2,
		},
	}, frows)

	fjob := &domain.TrainingJob{ID: "job-fc", Kind: domain.KindTimeSeries}

	res, err := eng.Forecast(fjob, nil, domain.ForecastRequest{Horizon: 4, ItemID: "b"})
	if err != nil {
		t.Fatalf("forecast: %v", err)
	}

	if len(res.Forecast) != 4 || res.ItemID != "b" || res.Forecast[0].Lower > res.Forecast[0].Value {
		t.Errorf("unexpected forecast: %+v", res)
	}

	_, err = eng.Forecast(fjob, nil, domain.ForecastRequest{Horizon: 4})
	if err == nil || !strings.Contains(err.Error(), "item_id") {
		t.Errorf("missing item_id should be a clear error, got %v", err)
	}

	zipF, _, err := exp.BuildExport(&domain.Task{Name: "fc"}, fjob)
	if err != nil {
		t.Fatal(err)
	}

	out2 := t.TempDir()
	unzip(t, zipF, out2)

	cmd = exec.Command(py, "predict_example.py")
	cmd.Dir = out2

	exOut, err2 := cmd.CombinedOutput()
	if err2 != nil || !strings.Contains(string(exOut), "forecast") {
		t.Errorf("exported forecast example failed: %v\n%s", err2, exOut)
	}
}

func TestEngineFallsBackWithoutArtifacts(t *testing.T) {
	t.Parallel()

	eng := tabular.NewEngine(tabular.ServingConfig{JobsDir: t.TempDir()}, nil, nil)
	defer eng.Close()

	_, err := eng.PredictTable(&domain.TrainingJob{ID: "nope"}, nil, nil, &domain.ValidatedRow{})
	if err == nil {
		t.Error("expected ErrNoModel without artifacts or a fallback")
	}

	// Path traversal in a job id must not escape the jobs directory.
	_, err = eng.PredictTable(&domain.TrainingJob{ID: "../../etc"}, nil, nil, &domain.ValidatedRow{})
	if err == nil {
		t.Error("traversal id must be refused")
	}
}

func unzip(t *testing.T, data []byte, dst string) {
	t.Helper()

	zr, err := zip.NewReader(bytes.NewReader(data), int64(len(data)))
	if err != nil {
		t.Fatal(err)
	}

	for _, f := range zr.File {
		if strings.Contains(f.Name, "..") || filepath.IsAbs(f.Name) {
			t.Fatalf("unsafe zip entry %q", f.Name)
		}

		rc, err := f.Open()
		if err != nil {
			t.Fatal(err)
		}

		var b bytes.Buffer

		_, _ = b.ReadFrom(rc)
		_ = rc.Close()

		err = os.WriteFile(filepath.Join(dst, f.Name), b.Bytes(), 0o600) //nolint:gosec // archive is produced by the test itself; names are known
		if err != nil {
			t.Fatal(err)
		}
	}
}
