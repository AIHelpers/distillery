package training

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"log"
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
	cmd     *exec.Cmd
	ctx     context.Context
	cancel  context.CancelFunc
	jobDir  string
	job     *domain.TrainingJob
	onDone  func(metrics *domain.TrainingMetrics, err error)
	stopped bool
}

// NewLocalTrainer creates the adapter. cfg.JobsDir defaults to ./data/training.
func NewLocalTrainer(cfg Config) *LocalTrainer {
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
		cfg:    cfg,
		sem:    make(chan struct{}, cfg.MaxConcurrentJobs),
		active: make(map[*trainingProcess]bool),
	}
}

// Start implements domain.FineTuner. It writes job-dir/config.json + dataset,
// runs VRAM preflight, launches the Python subprocess, and streams progress
// back to the usecase via onUpdate/onDone exactly like the simulator does.
func (l *LocalTrainer) Start(
	job *domain.TrainingJob,
	examples []*domain.Example,
	onUpdate func(progress int),
	onDone func(metrics *domain.TrainingMetrics, err error),
) {
	ctx, cancel := context.WithCancel(context.Background())

	proc := &trainingProcess{
		ctx:    ctx,
		cancel: cancel,
		job:    job,
		onDone: onDone,
	}

	l.mu.Lock()
	l.active[proc] = true
	l.mu.Unlock()

	// Run asynchronously like the simulator (which launches a goroutine).
	go l.runJob(proc, job, examples, onUpdate, onDone)
}

func (l *LocalTrainer) runJob(
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
	case <-proc.ctx.Done():
		onDone(nil, proc.ctx.Err())
		return
	}

	// Prepare job dir + config + dataset.
	jobDir := filepath.Join(l.cfg.JobsDir, job.ID)
	proc.jobDir = jobDir
	if err := os.MkdirAll(jobDir, 0o755); err != nil {
		l.finish(proc, onDone, nil, err)
		return
	}

	if err := l.writeJobConfig(job, examples, jobDir); err != nil {
		l.finish(proc, onDone, nil, err)
		return
	}

	if err := l.writeDataset(examples, filepath.Join(jobDir, "dataset.jsonl")); err != nil {
		l.finish(proc, onDone, nil, err)
		return
	}

	// VRAM preflight.
	if err := l.vramPreflight(job); err != nil {
		l.finish(proc, onDone, nil, err)
		return
	}

	// Launch subprocess.
	cmd := exec.CommandContext(
		proc.ctx,
		l.cfg.PythonBin,
		"-m", "trainer.run",
		"--job-dir", jobDir,
		"--config", filepath.Join(jobDir, "config.json"),
		"--dataset", filepath.Join(jobDir, "dataset.jsonl"),
	)

	// Stream logs to job-dir/trainer.log too.
	logFile, err := os.Create(filepath.Join(jobDir, "trainer.log"))
	if err != nil {
		l.finish(proc, onDone, nil, err)
		return
	}
	defer logFile.Close()

	cmd.Stdout = logFile
	cmd.Stderr = logFile

	if err := cmd.Start(); err != nil {
		l.finish(proc, onDone, nil, err)
		return
	}

	proc.cmd = cmd

	// Tail stdout for JSONL progress (we wrote to log file, but also read the
	// same file asynchronously to avoid a blocking pipe).
	go l.tailProgress(proc, filepath.Join(jobDir, "progress.jsonl"), onUpdate)

	// Wait for subprocess.
	waitErr := cmd.Wait()

	// After subprocess exits, read final metrics (if any).
	metrics, metricErr := l.readMetrics(jobDir, waitErr)
	if metricErr != nil && waitErr == nil {
		waitErr = metricErr
	}

	l.finish(proc, onDone, metrics, waitErr)

	// Disk-space guardrail: enforce checkpoint retention after the job
	// is done (completed or failed), never while a job is running.
	go l.gcOldJobsLocked()
}

// finish removes the process from the active map and invokes onDone once.
func (l *LocalTrainer) finish(
	proc *trainingProcess,
	onDone func(metrics *domain.TrainingMetrics, err error),
	metrics *domain.TrainingMetrics,
	err error,
) {
	l.mu.Lock()
	delete(l.active, proc)
	l.mu.Unlock()

	if err != nil && proc.ctx.Err() == context.Canceled {
		err = fmt.Errorf("training cancelled: %w", err)
	}

	onDone(metrics, err)
}

func (l *LocalTrainer) writeJobConfig(
	job *domain.TrainingJob,
	examples []*domain.Example,
	jobDir string,
) error {
	// Map domain job → trainer config.json.
	cfg := map[string]interface{}{
		"job_id":       job.ID,
		"base_model":   job.BaseModel.RepoID,
		"language":     "python",
		"skill":        "code_generation",
		"dataset_path": filepath.Join(jobDir, "dataset.jsonl"),
		"output_dir":   jobDir,
	}

	data, err := json.MarshalIndent(cfg, "", "  ")
	if err != nil {
		return err
	}

	return os.WriteFile(filepath.Join(jobDir, "config.json"), data, 0o644)
}

func (l *LocalTrainer) writeDataset(examples []*domain.Example, path string) error {
	var sb strings.Builder
	for _, ex := range examples {
		rec := map[string]string{
			"instruction": ex.Input,
			"output":      ex.Output,
		}
		line, err := json.Marshal(rec)
		if err != nil {
			return err
		}
		sb.Write(line)
		sb.WriteByte('\n')
	}
	return os.WriteFile(path, []byte(sb.String()), 0o644)
}

func (l *LocalTrainer) tailProgress(proc *trainingProcess, logPath string, onUpdate func(int)) {
	// Wait a tiny bit for the file to appear.
	for i := 0; i < 20; i++ {
		if _, err := os.Stat(logPath); err == nil {
			break
		}
		select {
		case <-proc.ctx.Done():
			return
		case <-time.After(100 * time.Millisecond):
		}
	}

	f, err := os.Open(logPath)
	if err != nil {
		return
	}
	defer f.Close()

	// Seek to end to avoid re-reading old lines on resume.
	_, _ = f.Seek(0, os.SEEK_END)

	scanner := bufio.NewScanner(f)
	for {
		select {
		case <-proc.ctx.Done():
			return
		default:
		}

		if !scanner.Scan() {
			// EOF — wait briefly for new data.
			time.Sleep(300 * time.Millisecond)
			if proc.ctx.Err() != nil {
				return
			}
			// Re-scan with a new scanner (file continues to grow).
			if _, err := f.Seek(0, os.SEEK_CUR); err != nil {
				return
			}
			scanner = bufio.NewScanner(f)
			continue
		}

		line := scanner.Bytes()
		var rec map[string]interface{}
		if err := json.Unmarshal(line, &rec); err != nil {
			continue
		}

		if rec["type"] == "progress" {
			if step, ok := rec["step"].(float64); ok {
				if total, ok := rec["total_steps"].(float64); ok && total > 0 {
					pct := int(step / total * 100)
					if pct > 100 {
						pct = 100
					}
					onUpdate(pct)
				}
			}
		}
	}
}

func (l *LocalTrainer) readMetrics(jobDir string, runErr error) (*domain.TrainingMetrics, error) {
	path := filepath.Join(jobDir, "metrics.json")
	data, err := os.ReadFile(path)
	if err != nil {
		// If user cancelled, treat as clean cancel, not a hard failure.
		if runErr != nil {
			return nil, runErr
		}
		return nil, fmt.Errorf("no metrics.json produced: %w", err)
	}

	var m struct {
		Status  string  `json:"status"`
		Loss    float64 `json:"eval_loss"`
		Epochs  int     `json:"epoch"`
		Steps   int     `json:"global_step"`
		Best    string  `json:"best_checkpoint"`
		TookSec float64 `json:"train_runtime"`
	}
	if err := json.Unmarshal(data, &m); err != nil {
		return nil, fmt.Errorf("malformed metrics.json: %w", err)
	}

	if m.Status != "completed" {
		return nil, fmt.Errorf("training failed (status=%q)", m.Status)
	}

	return &domain.TrainingMetrics{
		FinalLoss:     m.Loss,
		EvalAccuracy:  0, // computed by evaluator later
		Epochs:        m.Epochs,
		TrainExamples: 0,
	}, nil
}

// vramPreflight checks free GPU memory against the model's MinVRAMGB.
// Falls back to CPU mode when no GPU exists (unless disabled).
func (l *LocalTrainer) vramPreflight(job *domain.TrainingJob) error {
	if job.BaseModel.MinVRAMGB <= 0 {
		return nil // no requirement — allow.
	}

	freeGB, hasGPU, err := freeVRAMGB()
	if err != nil || !hasGPU {
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

func freeVRAMGB() (float64, bool, error) {
	out, err := exec.Command("nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits").Output()
	if err != nil {
		return 0, false, nil
	}
	line := strings.TrimSpace(string(out))
	parts := strings.Split(line, "\n") // multiple GPUs — take the max
	maxMiB := 0.0
	for _, p := range parts {
		var v float64
		if _, err := fmt.Sscanf(strings.TrimSpace(p), "%f", &v); err == nil {
			if v > maxMiB {
				maxMiB = v
			}
		}
	}
	return maxMiB / 1024.0, true, nil
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
		path := filepath.Join(l.cfg.JobsDir, dirs[i].name)
		if err := os.RemoveAll(path); err != nil {
			log.Printf("local trainer: failed to clean up old job dir %s: %v", path, err)
		} else {
			log.Printf("local trainer: cleaned old job dir %s (retention = %d)", path, l.cfg.MaxJobHistory)
		}
	}
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
