package training

import (
	"archive/zip"
	"bufio"
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io/fs"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"time"

	"distillery/internal/domain"
)

// GGUFConversionTimeout is the max time a single `python -m trainer.gguf`
// subprocess may run before it is killed. Models can be large; 2h default.
const GGUFConversionTimeout = 2 * time.Hour

// ggufManifest mirrors the Python converter's gguf-manifest.json so we can
// read back the produced file name + size without globbing.
type ggufManifest struct {
	File      string `json:"file"`
	SizeBytes int64  `json:"size_bytes"`
	Format    string `json:"format"`
	BaseModel string `json:"base_model"`
	Converter string `json:"converter"`
}

// LocalExporter implements domain.Exporter and domain.GGUFExporter for the
// local training backend. It builds the same portable zip as the simulation
// exporter, but bundles the real LoRA adapter weights produced by the Python
// worker (when present) instead of a placeholder, and offers an
// HomeBred-LLM/GGUF-ready manifest. The GGUF exporter additionally materialises the
// trained model as a single GGUF file that HomeBred-LLM / llama.cpp can load
// directly, converting on-demand via the Python worker's `trainer/gguf.py`.
type LocalExporter struct {
	// JobsDir is the same dir the LocalTrainer writes job artifacts to.
	JobsDir string
	// PythonBin is the python interpreter used to run trainer/gguf.py.
	PythonBin string
	// ModelCacheDir is the HuggingFace cache dir passed through to the converter.
	ModelCacheDir string
	// Progress tracks async GGUF conversion sessions (nil = sync mode).
	Progress *GGUFProgressStore
}

// NewLocalExporter creates a local exporter rooted at cfg.JobsDir.
func NewLocalExporter(cfg *Config) *LocalExporter {
	if cfg == nil {
		cfg = &Config{}
	}

	if cfg.JobsDir == "" {
		cfg.JobsDir = filepath.Join(".", "data", "training")
	}

	if cfg.PythonBin == "" {
		cfg.PythonBin = "python"
	}

	return &LocalExporter{
		JobsDir:       cfg.JobsDir,
		PythonBin:     cfg.PythonBin,
		ModelCacheDir: cfg.ModelCacheDir,
	}
}

// BuildExport creates a zip containing:
//   - manifest.json   (training metadata + real metrics)
//   - Dockerfile      (vLLM LoRA serving image)
//   - adapter/*       (real adapter weights + tokenizer if available)
//   - README.md       (deployment instructions incl. HomeBred-LLM/GGUF path)
func (e *LocalExporter) BuildExport(task *domain.Task, job *domain.TrainingJob) (data []byte, filename string, err error) {
	if job.Status != domain.TrainingCompleted {
		return nil, "", domain.ErrNoModel
	}

	buf := new(bytes.Buffer)
	zw := zip.NewWriter(buf)

	manifest, err := buildManifest(task, job)
	if err != nil {
		return nil, "", err
	}

	err = writeZipEntry(zw, "manifest.json", manifest)
	if err != nil {
		return nil, "", err
	}

	dockerfile := buildDockerfile(task, job)

	err = writeZipEntry(zw, "Dockerfile", []byte(dockerfile))
	if err != nil {
		return nil, "", err
	}

	adapterDir := filepath.Join(e.JobsDir, job.ID, "adapter")

	hasRealWeights := DirExists(adapterDir)
	if hasRealWeights {
		err := zipDir(zw, "adapter", adapterDir)
		if err != nil {
			return nil, "", fmt.Errorf("bundling adapter weights: %w", err)
		}
	} else {
		placeholder := []byte("No trained adapter found for this job — it may need to be re-run with TRAINING_BACKEND=local.\n")

		err := writeZipEntry(zw, "adapter/weights.safetensors.placeholder", placeholder)
		if err != nil {
			return nil, "", err
		}
	}

	readme := buildReadme(task, job, hasRealWeights)

	err = writeZipEntry(zw, "README.md", []byte(readme))
	if err != nil {
		return nil, "", err
	}

	err = zw.Close()
	if err != nil {
		return nil, "", err
	}

	return buf.Bytes(), fmt.Sprintf("%s-v%d-export.zip", SafeExportName(task.Name), job.Version), nil
}

// BuildGGUF implements domain.GGUFExporter. When the user requests a GGUF
// download, it invokes the Python converter (`trainer/gguf.py`) on-demand
// with the requested quantization and returns the resulting `.gguf` bytes.
//
// The adapter (LoRA) is merged into the base model, converted to GGUF (f16)
// via llama.cpp, and then quantized to the requested scheme (e.g. q4_k_m)
// — all in a single subprocess call, so nothing is produced during the
// training run itself.
//
// Production-ready behaviours:
//
//   - **Caching** — if the requested quantization has already been produced
//     for this job, the cached file is served directly (no conversion).
//   - **Completeness gate** — every served GGUF (both freshly converted and
//     cached) is validated for tokenizer metadata. A GGUF with architecture
//     and weights keys but zero tokenizer keys is unloadable by llama.cpp /
//     HomeBred-LLM and is rejected with an actionable error.
//   - **Concurrency lock** — a per-job + per-quantization interprocess lock
//     file prevents two goroutines/processes from running redundant
//     conversions simultaneously.
//   - **Timeout** — the subprocess is killed after GGUFConversionTimeout.
func (e *LocalExporter) BuildGGUF(task *domain.Task, job *domain.TrainingJob, opts domain.GGUFExportOptions) ([]byte, string, error) {
	if job.Status != domain.TrainingCompleted {
		return nil, "", domain.ErrNoModel
	}

	jobDir := filepath.Join(e.JobsDir, job.ID)
	adapterDir := filepath.Join(jobDir, "adapter")
	mergedDir := filepath.Join(jobDir, "model")

	outputDir := filepath.Join(jobDir, "gguf")
	if err := os.MkdirAll(outputDir, 0o755); err != nil {
		return nil, "", err
	}

	quantization := strings.ToLower(strings.TrimSpace(opts.Quantization))
	if quantization == "" {
		quantization = "q4_k_m"
	}

	outputName := SafeExportName(task.Name)

	// ---- Cache check: same quant already converted? ----.
	expectedName := fmt.Sprintf("%s-%s.gguf", outputName, quantization)

	cachedPath := filepath.Join(outputDir, expectedName)
	if FileExists(cachedPath) {
		// Completeness gate: refuse to serve an incomplete cached GGUF.
		err := ValidateGGUFCompleteness(cachedPath)
		if err != nil {
			// The cache entry is broken — remove it so a subsequent request
			// triggers a fresh conversion instead of repeatedly failing.
			_ = os.Remove(cachedPath)
			return nil, "", err
		}

		if data, err := os.ReadFile(cachedPath); err == nil && len(data) > 0 {
			filename := fmt.Sprintf("%s-v%d-%s.gguf", outputName, job.Version, strings.TrimSuffix(expectedName, ".gguf"))
			return data, filename, nil
		}
	}

	// ---- Concurrency lock per job + quant ----.
	lockPath := filepath.Join(outputDir, "."+strings.TrimSuffix(expectedName, ".gguf")+".lock")

	unlock, acquired, err := acquireGGUFLock(lockPath, GGUFConversionTimeout)
	if err != nil {
		return nil, "", fmt.Errorf("GGUF conversion lock: %w", err)
	}
	defer unlock()

	if !acquired {
		// Another conversion in-flight — wait and re-check cache.
		if FileExists(cachedPath) {
			// Completeness gate: refuse to serve an incomplete cached GGUF.
			err := ValidateGGUFCompleteness(cachedPath)
			if err != nil {
				_ = os.Remove(cachedPath)
				return nil, "", err
			}

			if data, err := os.ReadFile(cachedPath); err == nil && len(data) > 0 {
				filename := fmt.Sprintf("%s-v%d-%s.gguf", outputName, job.Version, strings.TrimSuffix(expectedName, ".gguf"))
				return data, filename, nil
			}
		}
	}

	args := []string{"-m", "trainer.gguf"}
	args = append(args, "--job-dir", jobDir)
	args = append(args, "--base-model", job.BaseModel.RepoID)
	args = append(args, "--quantization", quantization)
	args = append(args, "--output-name", outputName)
	// Explicitly request GGUF v3 — required for qwen3 and other modern
	// architectures. Passed explicitly so the export is always v3 even if
	// the Python-side default changes.
	args = append(args, "--gguf-version", "3")

	if e.ModelCacheDir != "" {
		args = append(args, "--cache-dir", e.ModelCacheDir)
	}

	if DirExists(adapterDir) {
		args = append(args, "--adapter", adapterDir)
	} else if DirExists(mergedDir) {
		args = append(args, "--model-dir", mergedDir)
	} else {
		// No trained weights on disk (e.g. simulation backend). Fall back
		// to converting the base model itself so the user still gets a
		// real, full-size, loadable GGUF — not a fake pseudo-random file.
		args = append(args, "--base-model-only")
	}

	cmd := exec.Command(e.PythonBin, args...)

	// Timeout the conversion subprocess so a hung converter can't block
	// the HTTP handler indefinitely.
	output, runErr := runWithTimeout(cmd, GGUFConversionTimeout)
	if runErr != nil {
		return nil, "", fmt.Errorf("GGUF conversion failed: %w\n%s", runErr, strings.TrimSpace(string(output)))
	}

	// The Python worker writes to job-dir/gguf/<outputName>-<quant>.gguf and
	// gguf-manifest.json. Prefer the exact expected filename (guaranteed by
	// the converter), falling back to the manifest / glob.
	name := expectedName
	if !FileExists(filepath.Join(outputDir, name)) {
		name, err = PreferredGGUF(outputDir)
		if err != nil {
			return nil, "", err
		}
	}

	// Completeness gate: verify the freshly converted GGUF contains
	// tokenizer metadata before serving it. A GGUF with architecture +
	// weights keys but zero tokenizer keys is an incomplete export that
	// llama.cpp / HomeBred-LLM cannot load at inference time.
	if err := ValidateGGUFCompleteness(filepath.Join(outputDir, name)); err != nil {
		return nil, "", err
	}

	data, err := os.ReadFile(filepath.Join(outputDir, name))
	if err != nil {
		return nil, "", fmt.Errorf("reading GGUF file: %w", err)
	}

	baseName := strings.TrimSuffix(name, filepath.Ext(name))
	filename := fmt.Sprintf("%s-v%d-%s.gguf", SafeExportName(task.Name), job.Version, baseName)

	return data, filename, nil
}

// StartGGUFAsync kicks off an async GGUF conversion in a background goroutine.
// It returns a session ID that can be used to poll progress via GetGGUFProgress
// and retrieve the result via GetGGUFResult.
//
// If the requested quantization is already cached, the session immediately
// transitions to "ready" with the file data pre-loaded.
func (e *LocalExporter) StartGGUFAsync(sessionID, taskID, jobID string, task *domain.Task, job *domain.TrainingJob, opts domain.GGUFExportOptions) {
	if e.Progress == nil {
		return
	}

	quantization := strings.ToLower(strings.TrimSpace(opts.Quantization))
	if quantization == "" {
		quantization = "q4_k_m"
	}

	e.Progress.Create(sessionID, taskID, jobID, quantization)

	go e.runGGUFAsync(sessionID, task, job, opts, quantization)
}

// GetGGUFProgress returns the current progress for an async GGUF session.
// Returns nil if the session does not exist.
func (e *LocalExporter) GetGGUFProgress(sessionID string) *domain.GGUFProgressInfo {
	if e.Progress == nil {
		return nil
	}

	p := e.Progress.Get(sessionID)
	if p == nil {
		return nil
	}

	return &domain.GGUFProgressInfo{
		SessionID: sessionID,
		TaskID:    p.TaskID,
		JobID:     p.JobID,
		Status:    string(p.Status),
		Step:      p.Step,
		Percent:   p.Percent,
		Detail:    p.Detail,
		Filename:  p.Filename,
		Size:      p.Size,
		Error:     p.Error,
		Quant:     p.Quant,
	}
}

// GetGGUFResult returns the file path and filename for a ready session.
// The caller (HTTP handler) streams the file directly from disk — the
// file is never loaded into memory. Returns empty path if the session
// is not ready or does not exist.
func (e *LocalExporter) GetGGUFResult(sessionID string) (string, string, error) {
	if e.Progress == nil {
		return "", "", errors.New("async GGUF not available")
	}

	p := e.Progress.Get(sessionID)
	if p == nil {
		return "", "", errors.New("session not found")
	}

	if p.Status != GGUFStatusReady {
		return "", "", fmt.Errorf("conversion not ready (status: %s)", p.Status)
	}

	return p.filePath, p.Filename, nil
}

// CleanupGGUFSession deletes the converted GGUF file from disk and removes
// the session from the progress store. This is called after the file has
// been successfully streamed to the client so large GGUF files don't
// accumulate on disk.
func (e *LocalExporter) CleanupGGUFSession(sessionID string) {
	if e.Progress == nil {
		return
	}

	p := e.Progress.Get(sessionID)
	if p == nil {
		return
	}

	// Delete the GGUF file from disk.
	if p.filePath != "" {
		_ = os.Remove(p.filePath)
	}

	e.Progress.Delete(sessionID)
}

// runGGUFAsync runs the GGUF conversion in the background, streaming progress
// events from the Python subprocess to the progress store.
func (e *LocalExporter) runGGUFAsync(sessionID string, task *domain.Task, job *domain.TrainingJob, opts domain.GGUFExportOptions, quantization string) {
	if e.Progress == nil {
		return
	}

	defer func() {
		// Clean up the session after 5 minutes to avoid memory growth.
		time.AfterFunc(5*time.Minute, func() {
			e.Progress.Delete(sessionID)
		})
	}()

	jobDir := filepath.Join(e.JobsDir, job.ID)
	adapterDir := filepath.Join(jobDir, "adapter")
	mergedDir := filepath.Join(jobDir, "model")

	outputDir := filepath.Join(jobDir, "gguf")
	if err := os.MkdirAll(outputDir, 0o755); err != nil {
		e.Progress.Update(sessionID, func(p *GGUFProgress) {
			p.Status = GGUFStatusError
			p.Error = fmt.Sprintf("creating output dir: %v", err)
		})

		return
	}

	outputName := SafeExportName(task.Name)
	expectedName := fmt.Sprintf("%s-%s.gguf", outputName, quantization)

	// ---- Cache check ----.
	cachedPath := filepath.Join(outputDir, expectedName)
	if FileExists(cachedPath) {
		// Completeness gate: refuse to serve an incomplete cached GGUF.
		err := ValidateGGUFCompleteness(cachedPath)
		if err != nil {
			// The cache entry is broken — remove it so a subsequent request
			// triggers a fresh conversion instead of repeatedly failing.
			_ = os.Remove(cachedPath)

			e.Progress.Update(sessionID, func(p *GGUFProgress) {
				p.Status = GGUFStatusError
				p.Error = fmt.Sprintf("cached GGUF is incomplete: %v", err)
			})

			return
		}

		if fi, err := os.Stat(cachedPath); err == nil && fi.Size() > 0 {
			absPath, _ := filepath.Abs(cachedPath)
			baseName := strings.TrimSuffix(expectedName, ".gguf")
			filename := fmt.Sprintf("%s-v%d-%s.gguf", outputName, job.Version, baseName)

			e.Progress.Update(sessionID, func(p *GGUFProgress) {
				p.Status = GGUFStatusReady
				p.Percent = 100
				p.Step = "Served from cache"
				p.Filename = filename
				p.Size = fi.Size()
				p.filePath = absPath
			})

			return
		}
	}

	// ---- Build command ----.
	args := []string{"-m", "trainer.gguf"}
	args = append(args, "--job-dir", jobDir)
	args = append(args, "--base-model", job.BaseModel.RepoID)
	args = append(args, "--quantization", quantization)
	args = append(args, "--output-name", outputName)
	// Explicitly request GGUF v3 — required for qwen3 and other modern
	// architectures. Passed explicitly so the export is always v3 even if
	// the Python-side default changes.
	args = append(args, "--gguf-version", "3")

	if e.ModelCacheDir != "" {
		args = append(args, "--cache-dir", e.ModelCacheDir)
	}

	if DirExists(adapterDir) {
		args = append(args, "--adapter", adapterDir)
	} else if DirExists(mergedDir) {
		args = append(args, "--model-dir", mergedDir)
	} else {
		args = append(args, "--base-model-only")
	}

	cmd := exec.Command(e.PythonBin, args...)

	// Capture stdout (JSON event lines) and stderr (error messages).
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		e.Progress.Update(sessionID, func(p *GGUFProgress) {
			p.Status = GGUFStatusError
			p.Error = fmt.Sprintf("pipe error: %v", err)
		})

		return
	}

	var stderrBuf bytes.Buffer

	cmd.Stderr = &stderrBuf

	if err := cmd.Start(); err != nil {
		e.Progress.Update(sessionID, func(p *GGUFProgress) {
			p.Status = GGUFStatusError
			p.Error = fmt.Sprintf("failed to start converter: %v", err)
		})

		return
	}

	// Stream stdout JSON events → progress store.
	scanner := bufio.NewScanner(stdout)
	scanner.Buffer(make([]byte, 64*1024), 1024*1024)

	for scanner.Scan() {
		line := scanner.Bytes()

		var event struct {
			Type      string `json:"type"`
			Message   string `json:"message"`
			File      string `json:"file"`
			SizeBytes int64  `json:"size_bytes"`
		}
		err := json.Unmarshal(line, &event)
		if err != nil {
			continue
		}

		// Handle error events immediately — set error status so the
		// progress endpoint returns the actionable message before
		// cmd.Wait() even finishes.
		if event.Type == "error" {
			e.Progress.Update(sessionID, func(p *GGUFProgress) {
				p.Status = GGUFStatusError
				p.Error = event.Message
			})

			continue
		}

		pct, step := EventPercent(event.Type)

		detail := event.Message
		if detail == "" && event.File != "" {
			detail = event.File
		}

		e.Progress.Update(sessionID, func(p *GGUFProgress) {
			if pct >= 0 {
				p.Percent = pct
			}

			if step != "" {
				p.Step = step
			}

			if detail != "" {
				p.Detail = detail
			}
		})
	}

	// Wait for process to finish.
	waitErr := cmd.Wait()
	if waitErr != nil {
		stderrText := strings.TrimSpace(stderrBuf.String())

		e.Progress.Update(sessionID, func(p *GGUFProgress) {
			p.Status = GGUFStatusError

			if stderrText != "" {
				// Include the last ~500 chars of stderr for actionable diagnostics.
				if len(stderrText) > 500 {
					stderrText = "..." + stderrText[len(stderrText)-500:]
				}

				p.Error = fmt.Sprintf("conversion failed: %v\n%s", waitErr, stderrText)
			} else {
				p.Error = fmt.Sprintf("conversion failed: %v", waitErr)
			}
		})

		return
	}

	// Read the produced GGUF file.
	name := expectedName
	if !FileExists(filepath.Join(outputDir, name)) {
		n, err := PreferredGGUF(outputDir)
		if err != nil {
			e.Progress.Update(sessionID, func(p *GGUFProgress) {
				p.Status = GGUFStatusError
				p.Error = fmt.Sprintf("finding output GGUF: %v", err)
			})

			return
		}

		name = n
	}

	ggufPath := filepath.Join(outputDir, name)

	// Completeness gate: verify the freshly converted GGUF contains
	// tokenizer metadata before marking it ready. A GGUF with architecture +
	// weights keys but zero tokenizer keys is an incomplete export that
	// llama.cpp / HomeBred-LLM cannot load at inference time.
	if err := ValidateGGUFCompleteness(ggufPath); err != nil {
		e.Progress.Update(sessionID, func(p *GGUFProgress) {
			p.Status = GGUFStatusError
			p.Error = fmt.Sprintf("GGUF export is incomplete: %v", err)
		})

		return
	}

	fi, err := os.Stat(ggufPath)
	if err != nil {
		e.Progress.Update(sessionID, func(p *GGUFProgress) {
			p.Status = GGUFStatusError
			p.Error = fmt.Sprintf("reading GGUF file: %v", err)
		})

		return
	}

	absPath, _ := filepath.Abs(ggufPath)
	baseName := strings.TrimSuffix(name, filepath.Ext(name))
	filename := fmt.Sprintf("%s-v%d-%s.gguf", outputName, job.Version, baseName)

	e.Progress.Update(sessionID, func(p *GGUFProgress) {
		p.Status = GGUFStatusReady
		p.Percent = 100
		p.Step = "Ready"
		p.Filename = filename
		p.Size = fi.Size()
		p.filePath = absPath
	})
}

// acquireGGUFLock attempts to create an exclusive lock file at path with the
// given timeout. Returns (unlockFunc, acquired, error). If `acquired` is
// false, some other process holds the lock and the caller should wait for
// that process to finish (the returned unlockFunc is a no-op).
func acquireGGUFLock(path string, timeout time.Duration) (func(), bool, error) {
	// On Windows, file locking is best-effort (no flock). We simulate a
	// simple lock via O_EXCL create + stale-check.
	if runtime.GOOS == "windows" {
		return acquireWindowsLock(path, timeout)
	}

	deadline := time.Now().Add(timeout)

	for {
		lf, err := os.OpenFile(path, os.O_CREATE|os.O_EXCL|os.O_RDWR, 0o600)
		if err == nil {
			// Lock acquired.
			return func() {
				_ = lf.Close()
				_ = os.Remove(path)
			}, true, nil
		}

		if os.IsExist(err) {
			// Check for stale lock: if the file is older than the timeout,
			// it was left by a crashed process — remove and retry.
			st, statErr := os.Stat(path)
			if statErr == nil && time.Since(st.ModTime()) > timeout {
				_ = os.Remove(path)
				continue
			}

			if time.Now().After(deadline) {
				return func() {}, false, errors.New("timed out waiting for GGUF conversion lock (another conversion may be running)")
			}

			time.Sleep(500 * time.Millisecond)

			continue
		}

		return func() {}, false, err
	}
}

// acquireWindowsLock implements a best-effort lock via exclusive-create with
// staleness detection.
func acquireWindowsLock(path string, timeout time.Duration) (func(), bool, error) {
	deadline := time.Now().Add(timeout)

	for {
		lf, err := os.OpenFile(path, os.O_CREATE|os.O_EXCL|os.O_RDWR, 0o600)
		if err == nil {
			return func() {
				_ = lf.Close()
				_ = os.Remove(path)
			}, true, nil
		}

		if os.IsExist(err) {
			st, statErr := os.Stat(path)
			if statErr == nil && time.Since(st.ModTime()) > timeout {
				_ = os.Remove(path)
				continue
			}

			if time.Now().After(deadline) {
				return func() {}, false, errors.New("timed out waiting for GGUF conversion lock")
			}

			time.Sleep(500 * time.Millisecond)

			continue
		}

		return func() {}, false, err
	}
}

// runWithTimeout runs the given command with a timeout, returning combined
// output and an error (with the error including the output truncation if the
// command timed out).
func runWithTimeout(cmd *exec.Cmd, timeout time.Duration) ([]byte, error) {
	var buf bytes.Buffer

	cmd.Stdout = &buf
	cmd.Stderr = &buf

	err := cmd.Start()
	if err != nil {
		return nil, err
	}

	done := make(chan error, 1)
	go func() {
		done <- cmd.Wait()
	}()

	select {
	case err := <-done:
		return buf.Bytes(), err
	case <-time.After(timeout):
		_ = cmd.Process.Kill()

		<-done // drain.

		return buf.Bytes(), fmt.Errorf("GGUF conversion timed out after %v", timeout)
	}
}

// PreferredGGUF returns the GGUF file name to serve. It reads
// gguf-manifest.json when present and prefers its `file` field; otherwise it
// picks the single `.gguf` file in dir (erroring if there are zero or
// multiple candidates).
func PreferredGGUF(dir string) (string, error) {
	manifestPath := filepath.Join(dir, "gguf-manifest.json")

	if data, err := os.ReadFile(manifestPath); err == nil {
		var m struct {
			File string `json:"file"`
		}

		err := json.Unmarshal(data, &m)
		if err == nil && m.File != "" {
			candidate := filepath.Base(m.File) // strip any path, use only the name.
			if strings.HasSuffix(candidate, ".gguf") && FileExists(filepath.Join(dir, candidate)) {
				return candidate, nil
			}
		}
	}

	entries, err := os.ReadDir(dir)
	if err != nil {
		return "", err
	}

	var gguFiles []string

	for _, e := range entries {
		if e.IsDir() {
			continue
		}

		if strings.HasSuffix(strings.ToLower(e.Name()), ".gguf") {
			gguFiles = append(gguFiles, e.Name())
		}
	}

	switch len(gguFiles) {
	case 1:
		return gguFiles[0], nil
	case 0:
		return "", fmt.Errorf("no *.gguf file found in %s", dir)
	default:
		return "", fmt.Errorf("multiple *.gguf files in %s; cannot pick one", dir)
	}
}

// FileExists reports whether path is an existing regular file.
func FileExists(path string) bool {
	st, err := os.Stat(path)

	return err == nil && !st.IsDir()
}

// buildManifest constructs the manifest JSON bytes.
func buildManifest(task *domain.Task, job *domain.TrainingJob) ([]byte, error) {
	manifest := map[string]interface{}{
		"task_name":      task.Name,
		"task_type":      task.Type,
		"base_model":     job.BaseModel.Name,
		"base_model_id":  job.BaseModel.RepoID,
		"training_job":   job.ID,
		"version":        job.Version,
		"metrics":        job.Metrics,
		"adapter_format": "LoRA (safetensors)",
		"exported_at":    time.Now().UTC().Format(time.RFC3339),
		"backend":        "local",
	}

	return json.MarshalIndent(manifest, "", "  ")
}

// buildDockerfile builds the vLLM LoRA serving image.
func buildDockerfile(task *domain.Task, job *domain.TrainingJob) string {
	return fmt.Sprintf(`# Self-hosted serving image for %q
# Portable export — run this anywhere, no platform lock-in.
FROM vllm/vllm-openai:latest

ENV BASE_MODEL=%s
ENV ADAPTER_PATH=/model/adapter

COPY adapter/ /model/adapter/

EXPOSE 8000
ENTRYPOINT ["python3", "-m", "vllm.entrypoints.openai.api_server", \
  "--model", "${BASE_MODEL}", \
  "--enable-lora", \
  "--lora-modules", "task-model=${ADAPTER_PATH}"]
`, task.Name, job.BaseModel.RepoID)
}

// buildReadme builds the README with deployment instructions.
func buildReadme(task *domain.Task, job *domain.TrainingJob, hasRealWeights bool) string {
	var adapterNote string
	if hasRealWeights {
		adapterNote = "adapter/           real trained LoRA adapter + tokenizer"
	} else {
		adapterNote = "adapter/           (placeholder — replace with trained LoRA adapter)"
	}

	return fmt.Sprintf(`# %s — Portable Model Export

Base model: %s (%s)
Task type: %s
Fine-tune version: v%d
Backend: local fine-tuning (QLoRA)

## Run it anywhere (vLLM + LoRA)

    docker build -t task-model .
    docker run -p 8000:8000 --gpus all task-model

Then call it like any OpenAI-compatible chat endpoint at
http://localhost:8000/v1/chat/completions

## Deploy with HomeBred-LLM (GGUF / llama.cpp)

From this export, convert + quantize the merged model:

    # 1. (optional) merge LoRA into base weights:
    python -c "from peft import PeftModel; from transformers import AutoModelForCausalLM; \
    m = AutoModelForCausalLM.from_pretrained('%s', device_map='auto'); \
    m = PeftModel.from_pretrained(m, 'adapter'); m = m.merge_and_unload(); m.save_pretrained('./merged')"

    # 2. convert to GGUF + quantize (via llama.cpp):
    #   python llama.cpp/convert_hf_to_gguf.py merged/ --outfile model.gguf
    #   ./llama.cpp/quantize model.gguf model-Q4_K_M.gguf q4_k_m

    # 3. create in HomeBred-LLM:
    homebred-llm create task-model -f Modelfile  # Modelfile: FROM ./model-Q4_K_M.gguf

## Contents

- Dockerfile        vLLM-based serving image, LoRA adapter mounted at boot
- manifest.json     training metadata + real metrics for this export
- %s

No platform lock-in: this image runs on your own GPU box, any cloud VM,
or Kubernetes — not just on this platform's hosted inference.
`, task.Name, job.BaseModel.Name, job.BaseModel.RepoID, task.Type, job.Version,
		job.BaseModel.RepoID, adapterNote)
}

// zipDir recursively adds a directory into the zip under prefix. It walks the
// tree first to collect relative paths, then reads each file afterwards so
// no filesystem operation happens inside the walk callback (avoids TOCTOU
// symlink races flagged by gosec).
func zipDir(zw *zip.Writer, prefix, dir string) error {
	var rels []string

	err := filepath.WalkDir(dir, func(fullPath string, d fs.DirEntry, walkErr error) error {
		if walkErr != nil {
			return walkErr
		}

		if d.IsDir() {
			return nil
		}

		rel, err := filepath.Rel(dir, fullPath)
		if err != nil {
			return err
		}

		rels = append(rels, rel)

		return nil
	})
	if err != nil {
		return err
	}

	for _, rel := range rels {
		data, err := os.ReadFile(filepath.Join(dir, rel))
		if err != nil {
			return err
		}

		zipName := filepath.ToSlash(filepath.Join(prefix, rel))

		err = writeZipEntry(zw, zipName, data)
		if err != nil {
			return err
		}
	}

	return nil
}

func writeZipEntry(zw *zip.Writer, name string, content []byte) error {
	w, err := zw.Create(name)
	if err != nil {
		return err
	}

	_, err = w.Write(content)

	return err
}

// DirExists reports whether path is an existing directory.
func DirExists(path string) bool {
	st, err := os.Stat(path)

	return err == nil && st.IsDir()
}

// SafeExportName produces a filesystem/URL-safe lowercase name from s.
func SafeExportName(s string) string {
	out := make([]rune, 0, len(s))

	for _, r := range s {
		if (r >= 'a' && r <= 'z') || (r >= 'A' && r <= 'Z') || (r >= '0' && r <= '9') {
			out = append(out, r)
		} else if r == ' ' || r == '-' || r == '_' {
			out = append(out, '-')
		}
	}

	if len(out) == 0 {
		return "task"
	}

	return strings.ToLower(string(out))
}
