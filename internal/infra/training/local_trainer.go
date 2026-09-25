// Package training provides the local training adapter that launches the
// Python worker as a subprocess and tails its JSONL progress stream.
package training

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"syscall"
	"time"

	"distillery/internal/domain"
)

// ErrVRAMInsufficient is returned when the selected model needs more GPU
// memory than the local machine currently has free.
var ErrVRAMInsufficient = &trainingError{"insufficient GPU VRAM for selected model"}

// ErrTrainingFailed is returned when the Python worker reports a failed
// training run in metrics.json.
var ErrTrainingFailed = &trainingError{"training failed"}

// ErrInvalidJobID is returned when a job ID fails the path-safety validation.
var ErrInvalidJobID = &trainingError{"invalid job ID"}

// trainingError is a descriptive domain-ish error from this adapter.
type trainingError struct{ msg string }

func (e *trainingError) Error() string { return e.msg }

// Config controls the local training worker.
type Config struct {
	// PythonBin is the python interpreter to invoke (default "python").
	PythonBin string
	// TrainerModule is the path to trainer/run.py (default "trainer/run.py").
	TrainerModule string
	// JobsDir is where per-job working dirs are created (default "./data/training").
	JobsDir string
	// ModelCacheDir is the HuggingFace cache dir (default empty → HF default).
	ModelCacheDir string
	// MaxConcurrentJobs limits how many Python workers run at once (default 1).
	MaxConcurrentJobs int
	// CPUFallback allows training on CPU even when no GPU is detected (default false).
	CPUFallback bool
	// MaxJobHistory keeps the N most-recent *completed/failed* job dirs on
	// disk; older checkpoints are removed during Cleanup. 0 = keep all.
	MaxJobHistory int
}

// LocalTrainer implements domain.FineTuner by launching the Python training
// worker as a subprocess and tailing its JSONL progress stream.
//
// It is a drop-in replacement for the simulated trainer: the usecase layer
// only sees the same `Start` callback contract.
type LocalTrainer struct {
	cfg Config

	mu     sync.Mutex
	sem    chan struct{}
	active map[*trainingProcess]bool
}

type trainingProcess struct {
	cmd    *exec.Cmd
	cancel context.CancelFunc
	jobDir string
	job    *domain.TrainingJob
	onDone func(metrics *domain.TrainingMetrics, err error)
}

// NewLocalTrainer creates the adapter.
func NewLocalTrainer(cfg *Config) *LocalTrainer {
	if cfg == nil {
		cfg = &Config{}
	}

	if cfg.PythonBin == "" {
		cfg.PythonBin = "python"
	}

	if cfg.TrainerModule == "" {
		cfg.TrainerModule = "trainer/run.py"
	}

	if cfg.JobsDir == "" {
		cfg.JobsDir = filepath.Join(".", "data", "training")
	}

	if cfg.MaxConcurrentJobs <= 0 {
		cfg.MaxConcurrentJobs = 1
	}

	return &LocalTrainer{
		cfg:    *cfg,
		sem:    make(chan struct{}, cfg.MaxConcurrentJobs),
		active: make(map[*trainingProcess]bool),
	}
}

// Start implements domain.FineTuner.
func (l *LocalTrainer) Start(
	job *domain.TrainingJob,
	examples []*domain.Example,
	onUpdate func(progress int),
	onDone func(metrics *domain.TrainingMetrics, err error),
) {
	ctx, cancel := context.WithCancel(context.Background())

	proc := &trainingProcess{
		cancel: cancel,
		job:    job,
		onDone: onDone,
	}

	l.mu.Lock()
	l.active[proc] = true
	l.mu.Unlock()

	// Run asynchronously like the simulator (which launches a goroutine).
	go l.runJob(ctx, proc, job, examples, onUpdate, onDone)
}

// Pause sends SIGTERM to the running Python worker (it checkpoints and exits).
func (l *LocalTrainer) Pause(jobID string) error {
	l.mu.Lock()
	defer l.mu.Unlock()

	for p := range l.active {
		if p.job.ID == jobID && p.cmd != nil && p.cmd.Process != nil {
			_ = p.cmd.Process.Signal(syscall.SIGTERM)

			return nil
		}
	}

	return domain.ErrNotFound
}

// Resume is a no-op for now (the process auto-restarts from last checkpoint).
func (l *LocalTrainer) Resume(_ string) error {
	return nil
}

// Cancel forcefully kills the subprocess.
func (l *LocalTrainer) Cancel(jobID string) error {
	l.mu.Lock()
	defer l.mu.Unlock()

	for p := range l.active {
		if p.job.ID == jobID && p.cmd != nil && p.cmd.Process != nil {
			_ = p.cmd.Process.Kill()
			p.cancel()

			return nil
		}
	}

	return domain.ErrNotFound
}

// ---------- Test helpers (exported for the external test package) ----------.

// Cfg returns a copy of the trainer configuration.
func (l *LocalTrainer) Cfg() Config {
	return l.cfg
}

// ActiveCount returns how many jobs are currently being trained.
func (l *LocalTrainer) ActiveCount() int {
	l.mu.Lock()
	defer l.mu.Unlock()

	return len(l.active)
}

// WriteJobConfig writes a config.json for a job (test helper).
func (l *LocalTrainer) WriteJobConfig(job *domain.TrainingJob, dir string) error {
	return l.writeJobConfig(job, nil, dir)
}

// WriteDataset writes a dataset.jsonl for a job (test helper).
func (l *LocalTrainer) WriteDataset(examples []*domain.Example, datasetPath string) error {
	return l.writeDataset(examples, datasetPath)
}

// ReadMetrics reads metrics.json (test helper).
func (l *LocalTrainer) ReadMetrics(ctx context.Context, dir string, runErr error) (*domain.TrainingMetrics, error) {
	return l.readMetrics(ctx, dir, runErr)
}

// VRAMPreflight checks VRAM for a job (test helper).
func (l *LocalTrainer) VRAMPreflight(ctx context.Context, job *domain.TrainingJob) error {
	return l.vramPreflight(ctx, job)
}

// AddActiveJob marks a job as active (test helper).
func (l *LocalTrainer) AddActiveJob(jobID string) {
	l.mu.Lock()
	defer l.mu.Unlock()

	l.active[&trainingProcess{job: &domain.TrainingJob{ID: jobID, TaskID: "", Version: 0, BaseModel: domain.BaseModel{Name: "", ParamsBillions: 0, Family: ""}, Status: "", Progress: 0, Metrics: nil, Error: "", CreatedAt: time.Time{}, StartedAt: nil, CompletedAt: nil}}] = true
}

// GCOldJobs runs the retention GC once (test helper).
func (l *LocalTrainer) GCOldJobs() {
	l.gcOldJobsLocked()
}

// runJob executes the Python worker and streams progress.
func (l *LocalTrainer) runJob(
	ctx context.Context,
	proc *trainingProcess,
	job *domain.TrainingJob,
	examples []*domain.Example,
	onUpdate func(progress int),
	onDone func(metrics *domain.TrainingMetrics, err error),
) {
	// Acquire a concurrency slot (max 1 by default).
	select {
	case l.sem <- struct{}{}:
		defer func() { <-l.sem }()
	case <-ctx.Done():
		onDone(nil, ctx.Err())

		return
	}

	// Prepare job dir + config + dataset.
	// Reject job IDs that could escape the jobs root; job IDs are generated
	// internally, but validating neutralizes path-traversal taint before the
	// value reaches the subprocess arguments below.
	if !isSafeJobID(job.ID) {
		l.finish(proc, nil, fmt.Errorf("%w %q", ErrInvalidJobID, job.ID))

		return
	}

	// `filepath.Base` is a gosec-recognised sanitizer. isSafeJobID above
	// already guarantees job.ID is a single safe path component, so Base is
	// a no-op semantically — it just neutralises the G204 taint on the
	// subprocess arguments built from jobDir below.
	jobID := filepath.Base(job.ID)
	jobDir := filepath.Join(l.cfg.JobsDir, jobID)
	proc.jobDir = jobDir

	err := os.MkdirAll(jobDir, 0o755)
	if err != nil {
		l.finish(proc, nil, err)

		return
	}

	err = l.writeJobConfig(job, examples, jobDir)
	if err != nil {
		l.finish(proc, nil, err)

		return
	}

	err = l.writeDataset(examples, filepath.Join(jobDir, "dataset.jsonl"))
	if err != nil {
		l.finish(proc, nil, err)

		return
	}

	// VRAM preflight.
	err = l.vramPreflight(ctx, job)
	if err != nil {
		l.finish(proc, nil, err)

		return
	}

	// Launch subprocess.
	args := []string{
		"-m", "trainer.run",
		"--kind", string(job.Kind),
		"--job-dir", jobDir,
		"--config", filepath.Join(jobDir, "config.json"),
		"--dataset", filepath.Join(jobDir, "dataset.jsonl"),
	}

	_, err = exec.LookPath(l.cfg.PythonBin)
	if err != nil {
		l.finish(proc, nil, fmt.Errorf("python binary %q not found: %w", l.cfg.PythonBin, err))

		return
	}

	cmd := exec.CommandContext(ctx, l.cfg.PythonBin, args...)

	// Stream logs to job-dir/trainer.log too.
	logFile, err := os.Create(filepath.Join(jobDir, "trainer.log"))
	if err != nil {
		l.finish(proc, nil, err)

		return
	}

	defer logFile.Close()

	cmd.Stdout = logFile
	cmd.Stderr = logFile

	err = cmd.Start()
	if err != nil {
		l.finish(proc, nil, err)

		return
	}

	proc.cmd = cmd

	// Tail stdout for JSONL progress (we wrote to log file, but also read the
	// same file asynchronously to avoid a blocking pipe).
	go l.tailProgress(ctx, proc, filepath.Join(jobDir, "progress.jsonl"), onUpdate)

	// Wait for subprocess.
	waitErr := cmd.Wait()

	// After subprocess exits, read final metrics (if any).
	metrics, metricErr := l.readMetrics(ctx, jobDir, waitErr)
	if metricErr != nil && waitErr == nil {
		waitErr = metricErr
	}

	l.finish(proc, metrics, waitErr)

	// Disk-space guardrail: enforce checkpoint retention after the job
	// is done (completed or failed), never while a job is running.
	go l.gcOldJobsLocked()
}

// isSafeJobID reports whether id is a plain identifier usable as a single
// path component (no separators, no "..", no leading dots, alphanumerics and
// - _ only). It neutralizes path-traversal taint before the value is used
// to build subprocess arguments.
func isSafeJobID(id string) bool {
	if id == "" || id == "." || id == ".." {
		return false
	}

	for _, r := range id {
		if r >= 'a' && r <= 'z' || r >= 'A' && r <= 'Z' || r >= '0' && r <= '9' || r == '-' || r == '_' {
			continue
		}

		return false
	}

	return true
}

// finish removes the process from the active map and invokes onDone once.
func (l *LocalTrainer) finish(
	proc *trainingProcess,
	metrics *domain.TrainingMetrics,
	err error,
) {
	l.mu.Lock()
	delete(l.active, proc)
	l.mu.Unlock()

	if err != nil && errors.Is(err, context.Canceled) {
		err = fmt.Errorf("training cancelled: %w", err)
	}

	proc.onDone(metrics, err)
}

// readMetrics reads the metrics.json produced by the Python worker.
func (l *LocalTrainer) readMetrics(ctx context.Context, jobDir string, runErr error) (*domain.TrainingMetrics, error) {
	metricsPath := filepath.Join(jobDir, "metrics.json")

	data, err := os.ReadFile(metricsPath)
	if err != nil {
		// If user cancelled, treat as clean cancel, not a hard failure.
		if runErr != nil || errors.Is(ctx.Err(), context.Canceled) {
			return nil, context.Canceled
		}

		return nil, fmt.Errorf("no metrics.json produced: %w", err)
	}

	var m struct {
		Status string  `json:"status"`
		Loss   float64 `json:"eval_loss"`
		// Python writes a fractional epoch (e.g. 3.75); an int field would
		// fail to unmarshal and mark the whole run "malformed".
		Epochs     float64                           `json:"epoch"`
		Train      int                               `json:"train_examples"`
		Steps      int                               `json:"global_step"`
		Best       string                            `json:"best_checkpoint"`
		TookSec    float64                           `json:"train_runtime"`
		Kind       string                            `json:"kind"`
		Accuracy   float64                           `json:"accuracy"`
		MacroF1    float64                           `json:"macro_f1"`
		WeightedF1 float64                           `json:"weighted_f1"`
		BaselineF1 float64                           `json:"baseline_macro_f1"`
		DeltaF1    float64                           `json:"delta_macro_f1"`
		Threshold  float64                           `json:"default_threshold"`
		MaxLength  int                               `json:"max_length"`
		MultiLabel bool                              `json:"multi_label"`
		PerClass   map[string]domain.PerClassMetrics `json:"per_class"`
		ConfMatrix domain.ConfusionMatrix            `json:"confusion_matrix"`
		LabelMap   map[string]int                    `json:"label_map"`
		Thresholds []domain.ThresholdSweepPoint      `json:"threshold_sweep"`

		// Retrieval metrics (embedding/reranker tasks).
		TunedNDCG10   float64 `json:"tuned_ndcg@10"`
		TunedMRR10    float64 `json:"tuned_mrr@10"`
		TunedRecall1  float64 `json:"tuned_recall@1"`
		TunedRecall5  float64 `json:"tuned_recall@5"`
		TunedRecall10 float64 `json:"tuned_recall@10"`
		BaseNDCG10    float64 `json:"base_ndcg@10"`
		BaseMRR10     float64 `json:"base_mrr@10"`
		BaseRecall1   float64 `json:"base_recall@1"`
		BaseRecall5   float64 `json:"base_recall@5"`
		BaseRecall10  float64 `json:"base_recall@10"`
		EmbeddingDim  int     `json:"embedding_dim"`
		IndexEstimate int64   `json:"index_size_estimate"`
	}

	err = json.Unmarshal(data, &m)
	if err != nil {
		return nil, fmt.Errorf("malformed metrics.json: %w", err)
	}

	if m.Status != "completed" {
		return nil, fmt.Errorf("%w (status=%q)", ErrTrainingFailed, m.Status)
	}

	return &domain.TrainingMetrics{
		FinalLoss:     m.Loss,
		EvalAccuracy:  m.Accuracy,
		Epochs:        int(math.Round(m.Epochs)),
		TrainExamples: m.Train,
		// Classifier fields propagate so the UI can render the confusion
		// matrix, per-class table, and baseline comparison.
		MacroF1:          m.MacroF1,
		WeightedF1:       m.WeightedF1,
		BaselineMacroF1:  m.BaselineF1,
		DeltaMacroF1:     m.DeltaF1,
		PerClass:         m.PerClass,
		ConfusionMatrix:  m.ConfMatrix,
		ThresholdSweep:   m.Thresholds,
		LabelMap:         m.LabelMap,
		DefaultThreshold: m.Threshold,
		MaxLength:        m.MaxLength,
		MultiLabel:       m.MultiLabel,
	}, nil
}

// vramPreflight checks free GPU memory against the model's MinVRAMGB.
// Falls back to CPU mode when no GPU exists (unless disabled).
func (l *LocalTrainer) vramPreflight(ctx context.Context, job *domain.TrainingJob) error {
	if job.BaseModel.MinVRAMGB <= 0 {
		return nil // no requirement — allow.
	}

	freeGB, hasGPU := freeVRAMGB(ctx)
	if !hasGPU {
		if l.cfg.CPUFallback {
			log.Printf("local trainer: no GPU — using CPU fallback for %s", job.BaseModel.Name)

			return nil
		}

		return fmt.Errorf("%w: %s requires %.1f GiB; no NVIDIA GPU detected", ErrVRAMInsufficient, job.BaseModel.Name, job.BaseModel.MinVRAMGB)
	}

	if freeGB < job.BaseModel.MinVRAMGB {
		return fmt.Errorf("%w: %s needs %.1f GiB but only %.1f GiB free", ErrVRAMInsufficient, job.BaseModel.Name, job.BaseModel.MinVRAMGB, freeGB)
	}

	return nil
}

// freeVRAMGB returns the max free VRAM across GPUs, and whether any GPU exists.
func freeVRAMGB(ctx context.Context) (freeGB float64, hasGPU bool) {
	out, err := exec.CommandContext(ctx, "nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits").Output()
	if err != nil {
		return 0, false
	}

	line := strings.TrimSpace(string(out))

	parts := strings.Split(line, "\n") // multiple GPUs — take the max.

	maxMiB := 0.0

	for _, p := range parts {
		var v float64

		_, err := fmt.Sscanf(strings.TrimSpace(p), "%f", &v)
		if err != nil {
			continue
		}

		if v > maxMiB {
			maxMiB = v
		}
	}

	return maxMiB / 1024.0, true
}

// gcOldJobsLocked enforces the MaxJobHistory retention policy by removing
// the oldest completed/failed job dirs beyond the limit. It is safe to call
// after a job finishes; active jobs are never touched.
func (l *LocalTrainer) gcOldJobsLocked() {
	if l.cfg.MaxJobHistory <= 0 {
		return // disabled — keep all checkpoints.
	}

	l.mu.Lock()
	activeJobs := make(map[string]bool, len(l.active))

	for p := range l.active {
		activeJobs[p.job.ID] = true
	}

	l.mu.Unlock()

	entries, err := os.ReadDir(l.cfg.JobsDir)
	if err != nil {
		return
	}

	type jobDirInfo struct {
		name string
		mod  time.Time
	}

	var dirs []jobDirInfo

	for _, e := range entries {
		if !e.IsDir() {
			continue
		}

		if activeJobs[e.Name()] {
			continue // never delete a running job's dir.
		}

		info, err := e.Info()
		if err != nil {
			continue
		}

		dirs = append(dirs, jobDirInfo{name: e.Name(), mod: info.ModTime()})
	}

	// Sort newest-first so we keep the most recent and prune the tail.
	sort.Slice(dirs, func(i, j int) bool { return dirs[i].mod.After(dirs[j].mod) })

	for i := l.cfg.MaxJobHistory; i < len(dirs); i++ {
		dirPath := filepath.Join(l.cfg.JobsDir, dirs[i].name)

		err := os.RemoveAll(dirPath)
		if err != nil {
			log.Printf("local trainer: failed to clean up old job dir %s: %v", dirPath, err)
		} else {
			log.Printf("local trainer: cleaned old job dir %s (retention = %d)", dirPath, l.cfg.MaxJobHistory)
		}
	}
}

// writeJobConfig maps the domain job to the trainer config.json.
func (l *LocalTrainer) writeJobConfig(
	job *domain.TrainingJob,
	examples []*domain.Example,
	jobDir string,
) error {
	_ = examples

	kind := job.Kind
	if kind == "" {
		kind = domain.DefaultModelKind
	}

	cfg := map[string]interface{}{
		"job_id":       job.ID,
		"kind":         string(kind),
		"base_model":   job.BaseModel.RepoID,
		"language":     "python",
		"skill":        "code_generation",
		"dataset_path": filepath.Join(jobDir, "dataset.jsonl"),
		"output_dir":   jobDir,
	}

	// Pass classifier hyperparameters through to the worker when present.
	if job.Classifier != nil {
		cfg["max_length"] = job.Classifier.MaxLength
		cfg["multi_label"] = job.Classifier.MultiLabel
		// Only pass class_weights when enabled: writing the Go zero value
		// (false) would silently disable imbalance handling, whose Python
		// default is on.
		if job.Classifier.ClassWeights {
			cfg["class_weights"] = true
		}

		cfg["epochs"] = job.Classifier.Epochs
		cfg["learning_rate"] = job.Classifier.LearningRate
		cfg["batch_size"] = job.Classifier.BatchSize
		cfg["threshold"] = job.Classifier.Threshold
	}

	// Pass embedding hyperparameters through to the worker when present.
	if job.Embedding != nil {
		cfg["max_seq_len"] = job.Embedding.MaxSeqLen
		cfg["loss"] = job.Embedding.Loss
		// Only pass hard_negatives when explicitly enabled, since the
		// Python default is off.
		if job.Embedding.HardNegatives {
			cfg["hard_negatives"] = true
		}

		if len(job.Embedding.Matryoshka) > 0 {
			cfg["matryoshka"] = job.Embedding.Matryoshka
		}

		cfg["epochs"] = job.Embedding.Epochs
		cfg["learning_rate"] = job.Embedding.LearningRate
		cfg["batch_size"] = job.Embedding.BatchSize
		cfg["grad_cache"] = job.Embedding.GradCache
		cfg["normalize"] = job.Embedding.Normalize
	}

	// Pass reranker hyperparameters through to the worker when present.
	if job.Reranker != nil {
		cfg["max_seq_len"] = job.Reranker.MaxSeqLen
		cfg["loss"] = job.Reranker.Loss
		cfg["epochs"] = job.Reranker.Epochs
		cfg["learning_rate"] = job.Reranker.LearningRate
		cfg["batch_size"] = job.Reranker.BatchSize
		cfg["grad_cache"] = job.Reranker.GradCache
	}

	data, err := json.MarshalIndent(cfg, "", "  ")
	if err != nil {
		return err
	}

	return os.WriteFile(filepath.Join(jobDir, "config.json"), data, 0o600)
}

// writeDataset writes the JSONL dataset file.
//
// For kinds backed by a typed Payload (embedding, reranker, classifier) the
// raw Payload JSON is written verbatim so the Python worker receives the
// query/positive/negative/document/label fields unmodified. Legacy causal_lm
// examples fall back to instruction/output pairs.
func (l *LocalTrainer) writeDataset(examples []*domain.Example, datasetPath string) error {
	var sb strings.Builder

	for _, ex := range examples {
		var line []byte

		if len(ex.Payload) > 0 {
			line = ex.Payload
		} else {
			var err error

			rec := map[string]string{
				"instruction": ex.Input,
				"output":      ex.Output,
			}

			line, err = json.Marshal(rec)
			if err != nil {
				return err
			}
		}

		sb.Write(line)
		sb.WriteByte('\n')
	}

	return os.WriteFile(datasetPath, []byte(sb.String()), 0o600)
}

// tailProgress reads the JSONL progress file and reports percentages.
func (l *LocalTrainer) tailProgress(ctx context.Context, proc *trainingProcess, logPath string, onUpdate func(int)) {
	// Wait a tiny bit for the file to appear.
	for range 20 {
		_, err := os.Stat(logPath)
		if err == nil {
			break
		}

		select {
		case <-ctx.Done():
			return
		case <-time.After(100 * time.Millisecond):
		}
	}

	f, err := os.Open(logPath)
	if err != nil {
		_ = proc

		return
	}

	defer f.Close()

	// Seek to end to avoid re-reading old lines on resume.
	_, _ = f.Seek(0, io.SeekEnd)

	scanner := bufio.NewScanner(f)

	for {
		select {
		case <-ctx.Done():
			return
		default:
		}

		if !scanner.Scan() {
			// EOF — wait briefly for new data.
			time.Sleep(300 * time.Millisecond)

			if ctx.Err() != nil {
				return
			}

			// Re-scan with a new scanner (file continues to grow).
			_, err := f.Seek(0, io.SeekCurrent)
			if err != nil {
				return
			}

			scanner = bufio.NewScanner(f)

			continue
		}

		line := scanner.Bytes()

		var rec map[string]interface{}

		err := json.Unmarshal(line, &rec)
		if err != nil {
			continue
		}

		if rec["type"] != "progress" {
			continue
		}

		step, ok := rec["step"].(float64)
		if !ok {
			continue
		}

		total, ok := rec["total_steps"].(float64)
		if !ok || total <= 0 {
			continue
		}

		pct := int(step / total * 100)

		if pct > 100 {
			pct = 100
		}

		onUpdate(pct)
	}
}
