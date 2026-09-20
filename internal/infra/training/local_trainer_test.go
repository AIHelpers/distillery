package training_test

import (
	"context"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"distillery/internal/domain"
	"distillery/internal/infra/training"
)

func TestNewLocalTrainer_Defaults(t *testing.T) {
	t.Parallel()

	lt := training.NewLocalTrainer(&training.Config{})

	if lt.Cfg().PythonBin != "python" {
		t.Errorf("expected default python bin, got %q", lt.Cfg().PythonBin)
	}

	if lt.Cfg().TrainerModule != "trainer/run.py" {
		t.Errorf("expected default trainer module, got %q", lt.Cfg().TrainerModule)
	}

	want := filepath.Join(".", "data", "training")
	if lt.Cfg().JobsDir == "" || filepath.Clean(lt.Cfg().JobsDir) != want {
		t.Errorf("expected default jobs dir %q, got %q", want, lt.Cfg().JobsDir)
	}

	if lt.Cfg().MaxConcurrentJobs != 1 {
		t.Errorf("expected default max concurrent jobs 1, got %d", lt.Cfg().MaxConcurrentJobs)
	}
}

func TestNewLocalTrainer_CustomConfig(t *testing.T) {
	t.Parallel()

	lt := training.NewLocalTrainer(&training.Config{
		PythonBin:         "python3",
		TrainerModule:     "custom/run.py",
		JobsDir:           "/tmp/custom-jobs",
		MaxConcurrentJobs: 3,
	})

	if lt.Cfg().PythonBin != "python3" {
		t.Errorf("expected python3, got %q", lt.Cfg().PythonBin)
	}

	if lt.Cfg().TrainerModule != "custom/run.py" {
		t.Errorf("expected custom/run.py, got %q", lt.Cfg().TrainerModule)
	}

	if lt.Cfg().JobsDir != "/tmp/custom-jobs" {
		t.Errorf("expected /tmp/custom-jobs, got %q", lt.Cfg().JobsDir)
	}

	if lt.Cfg().MaxConcurrentJobs != 3 {
		t.Errorf("expected 3 max concurrent jobs, got %d", lt.Cfg().MaxConcurrentJobs)
	}
}

// TestLocalTrainer_Start_WritesJobArtifacts verifies that Start tracks the
// process as active immediately, and that job-dir setup (config.json +
// dataset.jsonl) is eventually produced. The subprocess itself may fail if
// python / trainer module is absent in the test environment, so the
// completion result (metrics or error) is tolerated.
func TestLocalTrainer_Start_WritesJobArtifacts(t *testing.T) {
	t.Parallel()

	dir := t.TempDir()

	lt := training.NewLocalTrainer(&training.Config{
		JobsDir:           dir,
		MaxConcurrentJobs: 1,
		CPUFallback:       true,
	})

	job := &domain.TrainingJob{
		ID:      "job_test_1",
		Version: 1,
		BaseModel: domain.BaseModel{
			Name:             "TestModel",
			RepoID:           "test/test-model",
			MinVRAMGB:        8,
			RecommendedQuant: "4bit-nf4", ParamsBillions: 0, Family: "",
		}, TaskID: "", Status: "", Progress: 0, Metrics: nil, Error: "", CreatedAt: time.Time{}, StartedAt: nil, CompletedAt: nil,
	}

	examples := []*domain.Example{
		{ID: "ex1", Input: "add 1 and 1", Output: "2"},
		{ID: "ex2", Input: "add 2 and 2", Output: "4"},
		{ID: "ex3", Input: "add 3 and 3", Output: "6"},
	}

	done := make(chan struct{})

	var (
		mu         sync.Mutex
		gotMetrics *domain.TrainingMetrics
		gotErr     error
	)

	lt.Start(job, examples,
		func(int) {},
		func(metrics *domain.TrainingMetrics, err error) {
			mu.Lock()
			gotMetrics = metrics
			gotErr = err
			mu.Unlock()
			close(done)
		},
	)

	// The process must be tracked as active synchronously.
	active := lt.ActiveCount()
	if active != 1 {
		t.Errorf("expected 1 active job, got %d", active)
	}

	// Wait (with timeout) for the job artifacts to appear — writing happens
	// in the background goroutine.
	deadline := time.Now().Add(5 * time.Second)

	for {
		cfgPath := filepath.Join(dir, job.ID, "config.json")
		dsPath := filepath.Join(dir, job.ID, "dataset.jsonl")

		_, cfgErr := os.Stat(cfgPath)

		_, dsErr := os.Stat(dsPath)
		if cfgErr == nil && dsErr == nil {
			break
		}

		if time.Now().After(deadline) {
			t.Fatalf("timed out waiting for job artifacts: config=%v dataset=%v", cfgErr, dsErr)
		}

		time.Sleep(10 * time.Millisecond)
	}

	// Verify config.json contents.
	cfgData, err := os.ReadFile(filepath.Join(dir, job.ID, "config.json"))
	if err != nil {
		t.Fatalf("expected config.json readable: %v", err)
	}

	var cfg map[string]interface{}

	err = json.Unmarshal(cfgData, &cfg)
	if err != nil {
		t.Fatalf("expected config.json valid JSON: %v", err)
	}

	if cfg["job_id"] != "job_test_1" {
		t.Errorf("expected job_id in config, got %v", cfg["job_id"])
	}

	if cfg["base_model"] != "test/test-model" {
		t.Errorf("expected base_model in config, got %v", cfg["base_model"])
	}

	// Verify dataset.jsonl contents — should have one line per example.
	dsData, err := os.ReadFile(filepath.Join(dir, job.ID, "dataset.jsonl"))
	if err != nil {
		t.Fatalf("expected dataset.jsonl readable: %v", err)
	}

	if count := strings.Count(string(dsData), "\n"); count != len(examples) {
		t.Errorf("expected %d dataset lines, got %d", len(examples), count)
	}

	// onDone must eventually be invoked (subprocess may fail if python /
	// the trainer module is absent — that's an environment matter; any
	// completion result is acceptable here).
	select {
	case <-done:
		mu.Lock()
		defer mu.Unlock()

		if gotErr != nil {
			t.Logf("subprocess completion error (expected in envs without trainer): %v", gotErr)
		}

		if gotMetrics == nil && gotErr == nil {
			t.Error("expected metrics or an error from onDone")
		}
	case <-time.After(10 * time.Second):
		t.Fatal("timed out waiting for onDone to be invoked")
	}
}

func TestLocalTrainer_Start_TrackedAsActive(t *testing.T) {
	t.Parallel()

	dir := t.TempDir()

	lt := training.NewLocalTrainer(&training.Config{JobsDir: dir, CPUFallback: true})

	job := &domain.TrainingJob{
		ID:      "job_active",
		Version: 1,
		BaseModel: domain.BaseModel{
			Name:      "TestModel",
			RepoID:    "test/test-model",
			MinVRAMGB: 0, ParamsBillions: 0, Family: "",
		}, TaskID: "", Status: "", Progress: 0, Metrics: nil, Error: "", CreatedAt: time.Time{}, StartedAt: nil, CompletedAt: nil,
	}

	done := make(chan struct{})

	lt.Start(job, []*domain.Example{
		{ID: "e1", Input: "i1", Output: "o1"},
		{ID: "e2", Input: "i2", Output: "o2"},
		{ID: "e3", Input: "i3", Output: "o3"},
	}, func(int) {}, func(*domain.TrainingMetrics, error) { close(done) })

	active := lt.ActiveCount()
	if active != 1 {
		t.Errorf("expected 1 active job, got %d", active)
	}

	<-done

	// After completion, the job should be removed from the active map.
	active = lt.ActiveCount()
	if active != 0 {
		t.Errorf("expected 0 active jobs after completion, got %d", active)
	}
}

func TestLocalTrainer_Resume_NoOp(t *testing.T) {
	t.Parallel()

	lt := training.NewLocalTrainer(&training.Config{})

	err := lt.Resume("anything")
	if err != nil {
		t.Errorf("expected no error from Resume, got %v", err)
	}
}

func TestLocalTrainer_Pause_NotFound(t *testing.T) {
	t.Parallel()

	lt := training.NewLocalTrainer(&training.Config{})

	err := lt.Pause("nonexistent-job")
	if !errors.Is(err, domain.ErrNotFound) {
		t.Errorf("expected ErrNotFound, got %v", err)
	}
}

func TestLocalTrainer_Cancel_NotFound(t *testing.T) {
	t.Parallel()

	lt := training.NewLocalTrainer(&training.Config{})

	err := lt.Cancel("nonexistent-job")
	if !errors.Is(err, domain.ErrNotFound) {
		t.Errorf("expected ErrNotFound, got %v", err)
	}
}

func TestLocalTrainer_WriteJobConfig(t *testing.T) {
	t.Parallel()

	dir := t.TempDir()

	lt := training.NewLocalTrainer(&training.Config{JobsDir: dir})

	job := &domain.TrainingJob{
		ID:      "job_cfg",
		Version: 2,
		BaseModel: domain.BaseModel{
			Name:   "Llama-3.2-1B-Instruct",
			RepoID: "meta-llama/Llama-3.2-1B-Instruct", ParamsBillions: 0, Family: "",
		}, TaskID: "", Status: "", Progress: 0, Metrics: nil, Error: "", CreatedAt: time.Time{}, StartedAt: nil, CompletedAt: nil,
	}

	err := lt.WriteJobConfig(job, dir)
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}

	data, err := os.ReadFile(filepath.Join(dir, "config.json"))
	if err != nil {
		t.Fatalf("expected config.json: %v", err)
	}

	var cfg map[string]interface{}

	err = json.Unmarshal(data, &cfg)
	if err != nil {
		t.Fatalf("expected valid JSON: %v", err)
	}

	if cfg["base_model"] != "meta-llama/Llama-3.2-1B-Instruct" {
		t.Errorf("expected base_model repo id, got %v", cfg["base_model"])
	}

	if cfg["job_id"] != "job_cfg" {
		t.Errorf("expected job_id, got %v", cfg["job_id"])
	}

	if cfg["language"] != "python" {
		t.Errorf("expected language, got %v", cfg["language"])
	}
}

func TestLocalTrainer_WriteDataset(t *testing.T) {
	t.Parallel()

	lt := training.NewLocalTrainer(&training.Config{})

	path := filepath.Join(t.TempDir(), "dataset.jsonl")

	examples := []*domain.Example{
		{Input: "hello", Output: "world"},
		{Input: "foo", Output: "bar"},
	}

	err := lt.WriteDataset(examples, path)
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}

	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("expected dataset readable: %v", err)
	}

	lines := strings.Split(strings.TrimSpace(string(data)), "\n")
	if len(lines) != 2 {
		t.Fatalf("expected 2 lines, got %d", len(lines))
	}

	var rec struct {
		Instruction string `json:"instruction"`
		Output      string `json:"output"`
	}

	err = json.Unmarshal([]byte(lines[0]), &rec)
	if err != nil {
		t.Fatalf("expected valid JSON per line: %v", err)
	}

	if rec.Instruction != "hello" || rec.Output != "world" {
		t.Errorf("unexpected record: %+v", rec)
	}
}

func TestLocalTrainer_ReadMetrics(t *testing.T) {
	t.Parallel()

	dir := t.TempDir()

	lt := training.NewLocalTrainer(&training.Config{})

	_, err := lt.ReadMetrics(context.Background(), dir, nil)
	if err == nil {
		t.Fatal("expected error when metrics.json missing")
	}

	// Write valid metrics.json.
	valid := `{"status":"completed","eval_loss":0.42,"epoch":3,"global_step":150,"best_checkpoint":"/ckpt/best","train_runtime":123.4}`

	err = os.WriteFile(filepath.Join(dir, "metrics.json"), []byte(valid), 0o644)
	if err != nil {
		t.Fatalf("failed to write metrics: %v", err)
	}

	metrics, err := lt.ReadMetrics(context.Background(), dir, nil)
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}

	if metrics == nil {
		t.Fatal("expected metrics, got nil")
	}

	if metrics.FinalLoss != 0.42 {
		t.Errorf("expected loss 0.42, got %f", metrics.FinalLoss)
	}

	if metrics.Epochs != 3 {
		t.Errorf("expected 3 epochs, got %d", metrics.Epochs)
	}

	// Failed status.
	failed := `{"status":"failed","eval_loss":1.5,"epoch":1,"global_step":10,"best_checkpoint":"","train_runtime":10.0}`

	err = os.WriteFile(filepath.Join(dir, "metrics.json"), []byte(failed), 0o644)
	if err != nil {
		t.Fatalf("failed to write metrics: %v", err)
	}

	_, err = lt.ReadMetrics(context.Background(), dir, nil)
	if err == nil {
		t.Fatal("expected error when status=failed")
	}

	if !strings.Contains(err.Error(), "failed") {
		t.Errorf("expected failed-status error, got %v", err)
	}
}

func TestLocalTrainer_VRAMPreflight_NoRequirement(t *testing.T) {
	t.Parallel()

	lt := training.NewLocalTrainer(&training.Config{})

	job := &domain.TrainingJob{BaseModel: domain.BaseModel{MinVRAMGB: 0, Name: "", ParamsBillions: 0, Family: ""}, ID: "", TaskID: "", Version: 0, Status: "", Progress: 0, Metrics: nil, Error: "", CreatedAt: time.Time{}, StartedAt: nil, CompletedAt: nil}

	err := lt.VRAMPreflight(context.Background(), job)
	if err != nil {
		t.Errorf("expected no error for 0 VRAM requirement, got %v", err)
	}
}

func TestLocalTrainer_VRAMPreflight_CPUFallback(t *testing.T) {
	t.Parallel()

	lt := training.NewLocalTrainer(&training.Config{CPUFallback: true})

	job := &domain.TrainingJob{
		BaseModel: domain.BaseModel{
			Name:      "BigModel",
			MinVRAMGB: 80, ParamsBillions: 0, Family: "",
		}, ID: "", TaskID: "", Version: 0, Status: "", Progress: 0, Metrics: nil, Error: "", CreatedAt: time.Time{}, StartedAt: nil, CompletedAt: nil,
	}

	err := lt.VRAMPreflight(context.Background(), job)
	if err != nil {
		t.Errorf("expected CPU fallback to allow, got %v", err)
	}
}

func TestLocalTrainer_VRAMPreflight_NoGPU_NoFallback(t *testing.T) {
	t.Parallel()

	// On a machine without nvidia-smi, this will fail with ErrVRAMInsufficient.
	lt := training.NewLocalTrainer(&training.Config{CPUFallback: false})

	job := &domain.TrainingJob{
		BaseModel: domain.BaseModel{
			Name:      "BigModel",
			MinVRAMGB: 80, ParamsBillions: 0, Family: "",
		}, ID: "", TaskID: "", Version: 0, Status: "", Progress: 0, Metrics: nil, Error: "", CreatedAt: time.Time{}, StartedAt: nil, CompletedAt: nil,
	}

	err := lt.VRAMPreflight(context.Background(), job)
	if err == nil {
		t.Skip("nvidia-smi present — VRAM preflight may pass")
	}

	if !strings.Contains(err.Error(), "VRAM") ||
		!strings.Contains(err.Error(), "no NVIDIA GPU") {
		t.Errorf("expected VRAM/no-GPU error, got %v", err)
	}
}

func TestLocalTrainer_GC_SkipsActiveJobs(t *testing.T) {
	t.Parallel()

	dir := t.TempDir()

	// Create old job dirs.
	for i := range 5 {
		jobDir := filepath.Join(dir, "job_"+string(rune('a'+i)))

		err := os.MkdirAll(jobDir, 0o755)
		if err != nil {
			t.Fatalf("failed mkdir: %v", err)
		}

		// Set old timestamp.
		old := time.Now().Add(-time.Duration(i+1) * time.Hour)

		_ = os.Chtimes(jobDir, old, old)
	}

	lt := training.NewLocalTrainer(&training.Config{
		JobsDir:       dir,
		MaxJobHistory: 2,
	})

	// A new "active" job dir — should never be touched.
	activeDir := filepath.Join(dir, "job_active")

	err := os.MkdirAll(activeDir, 0o755)
	if err != nil {
		t.Fatalf("failed mkdir: %v", err)
	}

	now := time.Now()

	_ = os.Chtimes(activeDir, now, now)

	lt.AddActiveJob("job_active")
	lt.GCOldJobs()

	// Active always survives.
	_, err = os.Stat(activeDir)
	if err != nil {
		t.Errorf("active job dir was removed: %v", err)
	}

	// With MaxJobHistory=2, exactly 2 non-active dirs should remain.
	entries, err := os.ReadDir(dir)
	if err != nil {
		t.Fatalf("failed to read dir: %v", err)
	}

	remaining := 0

	for _, e := range entries {
		if !e.IsDir() {
			continue
		}

		remaining++
	}

	if remaining != 3 { // job_active + 2 newest.
		t.Errorf("expected 3 dirs remaining (active + 2 retained), got %d", remaining)
	}
}

func TestLocalTrainer_GC_Disabled(t *testing.T) {
	t.Parallel()

	dir := t.TempDir()

	for i := range 5 {
		jobDir := filepath.Join(dir, "job_"+string(rune('a'+i)))

		err := os.MkdirAll(jobDir, 0o755)
		if err != nil {
			t.Fatalf("failed mkdir: %v", err)
		}
	}

	lt := training.NewLocalTrainer(&training.Config{JobsDir: dir, MaxJobHistory: 0})

	lt.GCOldJobs() // should no-op.

	entries, err := os.ReadDir(dir)
	if err != nil {
		t.Fatalf("failed to read dir: %v", err)
	}

	if len(entries) != 5 {
		t.Errorf("expected 5 dirs (GC disabled), got %d", len(entries))
	}
}

func TestNewLocalExporter_Defaults(t *testing.T) {
	t.Parallel()

	le := training.NewLocalExporter(&training.Config{})
	if le.JobsDir == "" || filepath.Clean(le.JobsDir) != filepath.Join(".", "data", "training") {
		t.Errorf("expected default jobs dir, got %q", le.JobsDir)
	}
}

func TestLocalExporter_RejectsNonCompletedJob(t *testing.T) {
	t.Parallel()

	le := training.NewLocalExporter(&training.Config{JobsDir: t.TempDir()})

	job := &domain.TrainingJob{ID: "job_pending", Status: domain.TrainingQueued, TaskID: "", Version: 0, BaseModel: domain.BaseModel{Name: "", ParamsBillions: 0, Family: ""}, Progress: 0, Metrics: nil, Error: "", CreatedAt: time.Time{}, StartedAt: nil, CompletedAt: nil}

	_, _, err := le.BuildExport(&domain.Task{ID: "", Name: "", Description: "", Type: "", CreatedAt: time.Time{}, UpdatedAt: time.Time{}}, job)
	if err == nil {
		t.Fatal("expected error for non-completed job")
	}

	if !errors.Is(err, domain.ErrNoModel) {
		t.Errorf("expected ErrNoModel, got %v", err)
	}
}

func TestLocalExporter_PlaceholderWeights(t *testing.T) {
	t.Parallel()

	dir := t.TempDir()

	le := training.NewLocalExporter(&training.Config{JobsDir: dir})

	job := &domain.TrainingJob{
		ID:      "job_completed",
		Version: 3,
		Status:  domain.TrainingCompleted,
		BaseModel: domain.BaseModel{
			Name:   "Llama Small",
			RepoID: "meta-llama/Llama-Small", ParamsBillions: 0, Family: "",
		},
		Metrics: &domain.TrainingMetrics{
			FinalLoss:    0.42,
			EvalAccuracy: 0.87, Epochs: 0, TrainExamples: 0,
		}, TaskID: "", Progress: 0, Error: "", CreatedAt: time.Time{}, StartedAt: nil, CompletedAt: nil,
	}

	data, filename, err := le.BuildExport(&domain.Task{Name: "My Task", Type: domain.TaskClassification, ID: "", Description: "", CreatedAt: time.Time{}, UpdatedAt: time.Time{}}, job)
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}

	if len(data) == 0 {
		t.Fatal("expected non-empty zip data")
	}

	if filename != "my-task-v3-export.zip" {
		t.Errorf("expected filename 'my-task-v3-export.zip', got %q", filename)
	}
}

func TestLocalExporter_WithRealWeights(t *testing.T) {
	t.Parallel()

	dir := t.TempDir()

	le := training.NewLocalExporter(&training.Config{JobsDir: dir})

	job := &domain.TrainingJob{
		ID:      "job_weights",
		Version: 1,
		Status:  domain.TrainingCompleted,
		BaseModel: domain.BaseModel{
			Name:   "RealModel",
			RepoID: "org/real-model", ParamsBillions: 0, Family: "",
		},
		Metrics: &domain.TrainingMetrics{
			FinalLoss:    0.1,
			EvalAccuracy: 0.95, Epochs: 0, TrainExamples: 0,
		}, TaskID: "", Progress: 0, Error: "", CreatedAt: time.Time{}, StartedAt: nil, CompletedAt: nil,
	}

	// Simulate real trained weights on disk.
	adapterDir := filepath.Join(dir, job.ID, "adapter")

	err := os.MkdirAll(adapterDir, 0o755)
	if err != nil {
		t.Fatalf("failed mkdir: %v", err)
	}

	err = os.WriteFile(filepath.Join(adapterDir, "adapter_model.safetensors"), []byte("fake-weights"), 0o644)
	if err != nil {
		t.Fatalf("failed to write adapter: %v", err)
	}

	data, _, err := le.BuildExport(&domain.Task{Name: "Real Task", Type: domain.TaskGeneration, ID: "", Description: "", CreatedAt: time.Time{}, UpdatedAt: time.Time{}}, job)
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}

	if len(data) == 0 {
		t.Fatal("expected non-empty zip data")
	}
}

func TestSafeExportName(t *testing.T) {
	t.Parallel()

	cases := []struct {
		in   string
		want string
	}{
		{"Hello World", "hello-world"},
		{"My Task_2", "my-task-2"},
		{"Mixed-Case!Symbols", "mixed-casesymbols"},
		{"", "task"},
		{"Ünïcödé", "ncd"},
		{"123", "123"},
	}

	for _, c := range cases {
		got := training.SafeExportName(c.in)
		if got != c.want {
			t.Errorf("SafeExportName(%q) = %q, want %q", c.in, got, c.want)
		}
	}
}

func TestDirExists(t *testing.T) {
	t.Parallel()

	if training.DirExists(filepath.Join(t.TempDir(), "nonexistent")) {
		t.Error("expected false for nonexistent dir")
	}

	dir := t.TempDir()
	if !training.DirExists(dir) {
		t.Error("expected true for existing dir")
	}
}
