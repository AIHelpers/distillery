package simulation_test

import (
	"archive/zip"
	"bytes"
	"errors"
	"strings"
	"sync"
	"testing"
	"time"

	"distillery/internal/domain"
	"distillery/internal/infra/simulation"
)

// ---------- ModelSelector ----------.

func TestModelSelector_ListBaseModels(t *testing.T) {
	t.Parallel()

	ms := simulation.NewModelSelector()
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

	ms := simulation.NewModelSelector()
	task := &domain.Task{Type: domain.TaskClassification}

	// Small dataset, short inputs/outputs → smallest model (score 0).
	got := ms.SelectBaseModel(task, 100, 50, 20)
	if got.Name != "Qwen3-0.6B" {
		t.Errorf("expected smallest model for trivial classification, got %q", got.Name)
	}
}

func TestModelSelector_SelectBaseModel_Generation(t *testing.T) {
	t.Parallel()

	ms := simulation.NewModelSelector()
	task := &domain.Task{Type: domain.TaskGeneration}

	// Large generation dataset with long outputs → largest reachable model
	// (score 5 = Qwen3-4B; the catalog's 7B entry requires score 6
	// which the current scoring rules cannot produce).
	got := ms.SelectBaseModel(task, 1000, 1000, 500)
	if got.Name != "Qwen3-4B" {
		t.Errorf("expected Qwen3-4B for heavy generation, got %q", got.Name)
	}
}

func TestModelSelector_SelectBaseModel_Extraction(t *testing.T) {
	t.Parallel()

	ms := simulation.NewModelSelector()
	task := &domain.Task{Type: domain.TaskExtraction}

	// Medium extraction dataset → score 1, second model.
	got := ms.SelectBaseModel(task, 100, 200, 50)
	if got.Name != "Qwen3-1.7B" {
		t.Errorf("expected Qwen3-1.7B for extraction, got %q", got.Name)
	}
}

func TestModelSelector_SelectBaseModel_TinyDataset(t *testing.T) {
	t.Parallel()

	ms := simulation.NewModelSelector()
	task := &domain.Task{Type: domain.TaskGeneration}

	// Generation (+2) + long outputs (+2) - tiny dataset (-1) → score 3.
	got := ms.SelectBaseModel(task, 10, 100, 300)
	if got.Name != "Llama-3.2-3B-Instruct" {
		t.Errorf("expected Llama-3.2-3B for tiny dataset, got %q", got.Name)
	}
}

func TestModelSelector_SelectBaseModel_ScoreCap(t *testing.T) {
	t.Parallel()

	ms := simulation.NewModelSelector()
	task := &domain.Task{Type: domain.TaskGeneration}

	// Extreme dataset → score 5 (generation=2 + output>200=+2 + input>800=+1).
	// With 7 catalog entries, index 5 = Qwen3-4B.
	got := ms.SelectBaseModel(task, 10_000, 10_000, 10_000)
	if got.Name != "Qwen3-4B" {
		t.Errorf("expected Qwen3-4B for extreme dataset, got %q", got.Name)
	}
}

// ---------- FineTuner (simulated trainer) ----------.

func TestNewFineTuner_DefaultInterval(t *testing.T) {
	t.Parallel()

	f := simulation.NewFineTuner()
	if f.TickInterval != 400*time.Millisecond {
		t.Errorf("expected 400ms tick interval, got %v", f.TickInterval)
	}
}

func TestFineTuner_Start_TooFewExamples(t *testing.T) {
	t.Parallel()

	f := simulation.NewFineTuner()
	f.TickInterval = time.Millisecond // fast for test.

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

	f := simulation.NewFineTuner()
	f.TickInterval = time.Millisecond // fast for test.

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

	f := simulation.NewFineTuner()
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

// ---------- Exporter ----------.

func TestExporter_RejectsNonCompletedJob(t *testing.T) {
	t.Parallel()

	e := simulation.NewExporter()
	job := &domain.TrainingJob{ID: "j1", Status: domain.TrainingRunning}

	_, _, err := e.BuildExport(&domain.Task{}, job)
	if !errors.Is(err, domain.ErrNoModel) {
		t.Errorf("expected ErrNoModel, got %v", err)
	}
}

func TestExporter_BuildsZip(t *testing.T) {
	t.Parallel()

	e := simulation.NewExporter()
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
		if f.Name != "manifest.json" {
			continue
		}

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

func TestExporter_GGUF_RejectsSimulationBackend(t *testing.T) {
	t.Parallel()

	e := simulation.NewExporter()
	job := &domain.TrainingJob{
		ID:      "job_gguf",
		Version: 3,
		Status:  domain.TrainingCompleted,
		BaseModel: domain.BaseModel{
			Name:   "Base Model",
			RepoID: "org/base-model",
		},
	}

	data, filename, err := e.BuildGGUF(&domain.Task{
		Name: "My Fine-Tune Task",
		Type: domain.TaskGeneration,
	}, job, domain.GGUFExportOptions{Quantization: "q4_k_m"})

	if data != nil {
		t.Errorf("expected nil data when no GGUF converter is wired, got %d bytes", len(data))
	}

	if filename != "" {
		t.Errorf("expected empty filename when no GGUF converter is wired, got %q", filename)
	}

	if err == nil {
		t.Fatal("expected an error when no GGUF converter is wired")
	}

	// The error must clearly tell the user how to get a real GGUF — not
	// silently hand them a fake small model.
	if !strings.Contains(err.Error(), "TRAINING_BACKEND=local") {
		t.Errorf("expected error to mention TRAINING_BACKEND=local, got %v", err)
	}
}

// TestExporter_GGUF_DelegatesToRealConverter verifies that when a real
// production converter is wired up, the simulation exporter delegates to it
// instead of returning a stub error. A fake converter that returns a fixed
// payload is used so the test does not depend on the Python toolchain.
func TestExporter_GGUF_DelegatesToRealConverter(t *testing.T) {
	t.Parallel()

	job := &domain.TrainingJob{
		ID:      "job_delegate",
		Version: 1,
		Status:  domain.TrainingCompleted,
		BaseModel: domain.BaseModel{
			Name:   "Base Model",
			RepoID: "org/base-model",
		},
	}

	want := []byte("GGUF\x03\x00\x00\x00") // plausible GGUF magic header.
	wantFilename := "task-v1-q4_k_m.gguf"

	e := simulation.NewExporterWithGGUF(&fakeGGUFExporter{
		data:     want,
		filename: wantFilename,
	})

	data, filename, err := e.BuildGGUF(&domain.Task{
		Name: "Delegate Task",
		Type: domain.TaskGeneration,
	}, job, domain.GGUFExportOptions{Quantization: "q4_k_m"})
	if err != nil {
		t.Fatalf("expected delegation to succeed, got %v", err)
	}

	if !bytes.Equal(data, want) {
		t.Errorf("expected delegated converter payload, got %d bytes", len(data))
	}

	if filename != wantFilename {
		t.Errorf("expected filename %q, got %q", wantFilename, filename)
	}
}

// TestExporter_GGUF_DelegatesError verifies errors from the real converter
// propagate through the simulation exporter unchanged.
func TestExporter_GGUF_DelegatesError(t *testing.T) {
	t.Parallel()

	job := &domain.TrainingJob{
		ID:      "job_delegate_err",
		Version: 1,
		Status:  domain.TrainingCompleted,
		BaseModel: domain.BaseModel{
			Name:   "Base Model",
			RepoID: "org/base-model",
		},
	}

	e := simulation.NewExporterWithGGUF(&fakeGGUFExporter{
		err: errors.New("no trained adapter or merged model to convert"),
	})

	_, _, err := e.BuildGGUF(&domain.Task{
		Name: "Delegate Task",
		Type: domain.TaskGeneration,
	}, job, domain.GGUFExportOptions{Quantization: "q4_k_m"})
	if err == nil {
		t.Fatal("expected propagated error from real converter")
	}

	// Errors from the real converter propagate through the simulation
	// exporter unchanged — no wrapping, so the original message is visible.
	if !strings.Contains(err.Error(), "no trained adapter") {
		t.Errorf("expected converter error to propagate, got %v", err)
	}
}

// fakeGGUFExporter is a test double for domain.GGUFExporter.
type fakeGGUFExporter struct {
	data     []byte
	filename string
	err      error
}

func (f *fakeGGUFExporter) BuildGGUF(_ *domain.Task, _ *domain.TrainingJob, _ domain.GGUFExportOptions) ([]byte, string, error) {
	return f.data, f.filename, f.err
}

func TestExporter_NoMetrics_SafeAccZero(t *testing.T) {
	t.Parallel()

	e := simulation.NewExporter()
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
