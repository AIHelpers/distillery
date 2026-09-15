package simulation

import (
	"archive/zip"
	"bytes"
	"strings"
	"sync"
	"testing"
	"time"

	"distillery/internal/domain"
)

// ---------- ModelSelector ----------

func TestModelSelector_ListBaseModels(t *testing.T) {
	t.Parallel()

	ms := NewModelSelector()
	models := ms.ListBaseModels()

	if len(models) == 0 {
		t.Fatal("expected non-empty catalog")
	}

	// All models should have RepoID + MinVRAMGB.
	for _, m := range models {
		if m.RepoID == "" {
			t.Errorf("model %q missing RepoID", m.Name)
		}
		if m.MinVRAMGB <= 0 {
			t.Errorf("model %q missing MinVRAMGB", m.Name)
		}
		if m.RecommendedQuant == "" {
			t.Errorf("model %q missing RecommendedQuant", m.Name)
		}
		if m.ParamsBillions <= 0 {
			t.Errorf("model %q missing ParamsBillions", m.Name)
		}
	}
}

func TestModelSelector_SelectBaseModel_Classification(t *testing.T) {
	t.Parallel()

	ms := NewModelSelector()
	task := &domain.Task{Type: domain.TaskClassification}

	// Small dataset, short inputs/outputs → smallest model (score 0).
	got := ms.SelectBaseModel(task, 100, 50, 20)
	if got.Name != "Qwen2.5-0.5B-Instruct" {
		t.Errorf("expected smallest model for trivial classification, got %q", got.Name)
	}
}

func TestModelSelector_SelectBaseModel_Generation(t *testing.T) {
	t.Parallel()

	ms := NewModelSelector()
	task := &domain.Task{Type: domain.TaskGeneration}

	// Large generation dataset with long outputs → largest reachable model
	// (score 5 = Qwen2.5-Coder-3B; the catalog's 7B entry requires score 6
	// which the current scoring rules cannot produce).
	got := ms.SelectBaseModel(task, 1000, 1000, 500)
	if got.Name != "Qwen2.5-Coder-3B" {
		t.Errorf("expected Qwen2.5-Coder-3B for heavy generation, got %q", got.Name)
	}
}

func TestModelSelector_SelectBaseModel_Extraction(t *testing.T) {
	t.Parallel()

	ms := NewModelSelector()
	task := &domain.Task{Type: domain.TaskExtraction}

	// Medium extraction dataset → score 1, second model.
	got := ms.SelectBaseModel(task, 100, 200, 50)
	if got.Name != "Qwen2.5-1.5B-Instruct" {
		t.Errorf("expected Qwen2.5-1.5B for extraction, got %q", got.Name)
	}
}

func TestModelSelector_SelectBaseModel_TinyDataset(t *testing.T) {
	t.Parallel()

	ms := NewModelSelector()
	task := &domain.Task{Type: domain.TaskGeneration}

	// Generation (+2) + long outputs (+2) - tiny dataset (-1) → score 3.
	got := ms.SelectBaseModel(task, 10, 100, 300)
	if got.Name != "Llama-3.2-3B-Instruct" {
		t.Errorf("expected Llama-3.2-3B for tiny dataset, got %q", got.Name)
	}
}

func TestModelSelector_SelectBaseModel_ScoreCap(t *testing.T) {
	t.Parallel()

	ms := NewModelSelector()
	task := &domain.Task{Type: domain.TaskGeneration}

	// Extreme dataset → score 5 (generation=2 + output>200=+2 + input>800=+1).
	// With 7 catalog entries, index 5 = Qwen2.5-Coder-3B.
	got := ms.SelectBaseModel(task, 10_000, 10_000, 10_000)
	if got.Name != "Qwen2.5-Coder-3B" {
		t.Errorf("expected Qwen2.5-Coder-3B for extreme dataset, got %q", got.Name)
	}
}

// ---------- FineTuner (simulated trainer) ----------

func TestNewFineTuner_DefaultInterval(t *testing.T) {
	t.Parallel()

	f := NewFineTuner()
	if f.TickInterval != 400*time.Millisecond {
		t.Errorf("expected 400ms tick interval, got %v", f.TickInterval)
	}
}

func TestFineTuner_Start_TooFewExamples(t *testing.T) {
	t.Parallel()

	f := NewFineTuner()
	f.TickInterval = time.Millisecond // fast for test

	done := make(chan struct{})
	var (
		mu     sync.Mutex
		gotErr error
	)

	f.Start(&domain.TrainingJob{}, []*domain.Example{
		{ID: "e1"},
		{ID: "e2"},
	}, func(int) {}, func(_ *domain.TrainingMetrics, err error) {
		mu.Lock()
		gotErr = err
		mu.Unlock()
		close(done)
	})

	select {
	case <-done:
		mu.Lock()
		defer mu.Unlock()
		if gotErr == nil {
			t.Fatal("expected error for < 3 examples")
		}
		if !strings.Contains(gotErr.Error(), "at least 3") {
			t.Errorf("expected 'at least 3' error, got %v", gotErr)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("timed out")
	}
}

func TestFineTuner_Start_ProgressAndDone(t *testing.T) {
	t.Parallel()

	f := NewFineTuner()
	f.TickInterval = time.Millisecond // fast for test

	done := make(chan struct{})
	var (
		mu         sync.Mutex
		gotMetrics *domain.TrainingMetrics
		gotErr     error
		updates    []int
	)

	f.Start(&domain.TrainingJob{}, []*domain.Example{
		{ID: "e1", Input: "a", Output: "b"},
		{ID: "e2", Input: "c", Output: "d"},
		{ID: "e3", Input: "e", Output: "f"},
		{ID: "e4", Input: "g", Output: "h"},
	}, func(progress int) {
		mu.Lock()
		updates = append(updates, progress)
		mu.Unlock()
	}, func(metrics *domain.TrainingMetrics, err error) {
		mu.Lock()
		gotMetrics = metrics
		gotErr = err
		mu.Unlock()
		close(done)
	})

	select {
	case <-done:
		mu.Lock()
		defer mu.Unlock()
		if gotErr != nil {
			t.Fatalf("expected no error, got %v", gotErr)
		}
		if gotMetrics == nil {
			t.Fatal("expected metrics, got nil")
		}
		if gotMetrics.Epochs != 3 {
			t.Errorf("expected 3 epochs, got %d", gotMetrics.Epochs)
		}
		if gotMetrics.TrainExamples != 4 {
			t.Errorf("expected 4 train examples, got %d", gotMetrics.TrainExamples)
		}
		if len(updates) != 20 {
			t.Errorf("expected 20 progress updates, got %d", len(updates))
		}
		if len(updates) > 0 && updates[len(updates)-1] != 100 {
			t.Errorf("expected final progress 100, got %d", updates[len(updates)-1])
		}
	case <-time.After(5 * time.Second):
		t.Fatal("timed out waiting for simulation to finish")
	}
}

func TestFineTuner_Start_MetricsBounds(t *testing.T) {
	t.Parallel()

	f := NewFineTuner()
	f.TickInterval = time.Millisecond

	// Large dataset → loss should be low, accuracy high but capped ≤ 0.99.
	examples := make([]*domain.Example, 600)
	for i := range examples {
		examples[i] = &domain.Example{ID: string(rune('a' + i%26)), Input: "x", Output: "y"}
	}

	done := make(chan struct{})
	var (
		mu         sync.Mutex
		gotMetrics *domain.TrainingMetrics
	)

	f.Start(&domain.TrainingJob{}, examples, func(int) {},
		func(metrics *domain.TrainingMetrics, _ error) {
			mu.Lock()
			gotMetrics = metrics
			mu.Unlock()
			close(done)
		})

	<-done
	mu.Lock()
	defer mu.Unlock()

	if gotMetrics == nil {
		t.Fatal("expected metrics")
	}
	if gotMetrics.FinalLoss < 0.0 || gotMetrics.FinalLoss > 1.8 {
		t.Errorf("expected loss in (0, 1.8], got %f", gotMetrics.FinalLoss)
	}
	if gotMetrics.EvalAccuracy < 0.5 || gotMetrics.EvalAccuracy > 0.99 {
		t.Errorf("expected accuracy in [0.5, 0.99], got %f", gotMetrics.EvalAccuracy)
	}
}

func TestRound2(t *testing.T) {
	t.Parallel()

	cases := []struct {
		in   float64
		want float64
	}{
		{1.2356, 1.23},
		{0.999, 0.99},
		{0.05, 0.05},
		{0.049, 0.04},
	}
	for _, c := range cases {
		if got := round2(c.in); got != c.want {
			t.Errorf("round2(%v) = %v, want %v", c.in, got, c.want)
		}
	}
}

// ---------- Exporter ----------

func TestExporter_RejectsNonCompletedJob(t *testing.T) {
	t.Parallel()

	e := NewExporter()
	job := &domain.TrainingJob{ID: "j1", Status: domain.TrainingRunning}

	_, _, err := e.BuildExport(&domain.Task{}, job)
	if err != domain.ErrNoModel {
		t.Errorf("expected ErrNoModel, got %v", err)
	}
}

func TestExporter_BuildsZip(t *testing.T) {
	t.Parallel()

	e := NewExporter()
	job := &domain.TrainingJob{
		ID:      "job_export",
		Version: 2,
		Status:  domain.TrainingCompleted,
		BaseModel: domain.BaseModel{
			Name:   "My Base Model",
			RepoID: "org/my-base-model",
		},
		Metrics: &domain.TrainingMetrics{
			FinalLoss:    0.4,
			EvalAccuracy: 0.88,
		},
	}

	data, filename, err := e.BuildExport(&domain.Task{
		Name: "My Fine-Tune Task",
		Type: domain.TaskClassification,
	}, job)
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}
	if len(data) == 0 {
		t.Fatal("expected non-empty zip data")
	}
	if filename != "My-Fine-Tune-Task-v2-export.zip" {
		t.Errorf("expected export filename, got %q", filename)
	}

	// Verify zip contents.
	zr, err := zip.NewReader(bytes.NewReader(data), int64(len(data)))
	if err != nil {
		t.Fatalf("expected valid zip, got %v", err)
	}

	wantFiles := map[string]bool{
		"manifest.json": false,
		"Dockerfile":    false,
		"README.md":     false,
		"adapter/weights.safetensors.placeholder": false,
	}
	for _, f := range zr.File {
		if _, ok := wantFiles[f.Name]; ok {
			wantFiles[f.Name] = true
		}
	}
	for name, found := range wantFiles {
		if !found {
			t.Errorf("expected zip entry %q not found", name)
		}
	}

	// Check manifest contains expected keys.
	for _, f := range zr.File {
		if f.Name == "manifest.json" {
			rc, err := f.Open()
			if err != nil {
				t.Fatalf("failed to open manifest: %v", err)
			}
			var buf bytes.Buffer
			_, _ = buf.ReadFrom(rc)
			rc.Close()

			content := buf.String()
			for _, want := range []string{"task_name", "task_type", "base_model", "training_job", "version", "metrics", "adapter_format", "exported_at", "note"} {
				if !strings.Contains(content, "\""+want+"\"") {
					t.Errorf("manifest missing key %q", want)
				}
			}
			if !strings.Contains(content, "My Fine-Tune Task") {
				t.Error("manifest missing task name")
			}
			break
		}
	}
}

func TestExporter_NoMetrics_SafeAccZero(t *testing.T) {
	t.Parallel()

	e := NewExporter()
	job := &domain.TrainingJob{
		ID:      "job_no_metrics",
		Version: 1,
		Status:  domain.TrainingCompleted,
		BaseModel: domain.BaseModel{
			Name:   "Model",
			RepoID: "org/model",
		},
		// Metrics nil.
	}

	data, _, err := e.BuildExport(&domain.Task{Name: "Task", Type: domain.TaskGeneration}, job)
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}
	if len(data) == 0 {
		t.Fatal("expected non-empty zip data")
	}
}

// ---------- Helpers ----------

func TestSafeName(t *testing.T) {
	t.Parallel()

	cases := []struct {
		in   string
		want string
	}{
		{"Hello World", "Hello-World"},
		{"My_Task-2", "My-Task-2"},
		{"Symbols!@#", "Symbols"}, // non-alnum dropped, separators replaced
		{"", "task"},
		{"123abc", "123abc"},
	}
	for _, c := range cases {
		got := safeName(c.in)
		if got != c.want {
			t.Errorf("safeName(%q) = %q, want %q", c.in, got, c.want)
		}
	}
}

func TestSafeName_EmptyReturnsTask(t *testing.T) {
	t.Parallel()

	// Even when only separators exist, non-empty should still return "task"
	// only if the output is truly empty.
	if got := safeName("   "); got != "---" {
		t.Errorf("expected '---' for spaces, got %q", got)
	}
	if got := safeName("!!!@@@"); got != "task" {
		t.Errorf("expected 'task' for pure symbols, got %q", got)
	}
}
