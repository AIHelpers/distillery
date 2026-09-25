// Package serving provides a real inference backend that proxies requests
// to a llama.cpp `llama-server` process started per trained model (GGUF).
//
// It implements domain.InferenceEngine so the deployment usecase can serve
// real fine-tuned models end to end, while remaining fully optional: when
// INFERENCE_BACKEND is "simulation" (the default) it falls back to the
// nearest-neighbour demo engine so no GPU is required.
package serving

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"time"

	"distillery/internal/domain"
	"distillery/internal/infra/simulation"
	localtraining "distillery/internal/infra/training"
)

// LlamacppConfig controls the llama.cpp inference backend.
type LlamacppConfig struct {
	// Bin is the path to the `llama-server` binary (default "llama-server").
	Bin string
	// JobsDir is where per-job working dirs live (same as the local trainer).
	JobsDir string
	// Host the llama-server binds to (default "127.0.0.1" — localhost only;
	// API-key verification stays in Go, the backend is never exposed).
	Host string
	// Backend selects the engine: "llamacpp" for real, "simulation" for demo.
	Backend string
	// StartupTimeout is how long to wait for the health check (default 60s).
	StartupTimeout time.Duration
	// HTTPTimeout bounds each completion request (default 120s).
	HTTPTimeout time.Duration
}

func (c *LlamacppConfig) withDefaults() {
	if c.Bin == "" {
		c.Bin = "llama-server"
	}

	if c.JobsDir == "" {
		c.JobsDir = filepath.Join(".", "data", "training")
	}

	if c.Host == "" {
		c.Host = "127.0.0.1"
	}

	if c.Backend == "" {
		if v := os.Getenv("INFERENCE_BACKEND"); v != "" {
			c.Backend = v
		} else {
			c.Backend = "simulation"
		}
	}

	if c.StartupTimeout <= 0 {
		c.StartupTimeout = 60 * time.Second
	}

	if c.HTTPTimeout <= 0 {
		c.HTTPTimeout = 120 * time.Second
	}
}

// LlamacppEngine implements domain.InferenceEngine. It proxies /predict and
// /batch calls to a per-job llama-server process started from the trained
// GGUF, restarting it if it crashes. When the backend is "simulation" (or no
// GGUF exists yet) it delegates to the nearest-neighbour demo engine so the
// no-GPU demo keeps working.
type LlamacppEngine struct {
	cfg LlamacppConfig

	fallback domain.InferenceEngine

	mu      sync.Mutex
	servers map[string]*serverProc // keyed by job ID.
	client  *http.Client
}

type serverProc struct {
	jobID string
	port  int
	cmd   *exec.Cmd
	// done is closed when the process exits (normal or crash), which the
	// wait-goroutine drives. It replaces the non-portable
	// Process.Signal(0) liveness probe.
	done chan struct{}
	// lastErr records the most recent start/health failure for diagnostics.
	lastErr string
}

// NewLlamacppEngine creates the adapter. It wraps the simulation engine so
// the demo still works before real weights are present.
func NewLlamacppEngine(cfg *LlamacppConfig) *LlamacppEngine {
	c := LlamacppConfig{}
	if cfg != nil {
		c = *cfg
	}

	c.withDefaults()

	return &LlamacppEngine{
		cfg:      c,
		fallback: simulation.NewInferenceEngine(),
		servers:  make(map[string]*serverProc),
		client:   &http.Client{Timeout: c.HTTPTimeout},
	}
}

// Predict implements domain.InferenceEngine.
func (e *LlamacppEngine) Predict(
	job *domain.TrainingJob,
	trainingExamples []*domain.Example,
	input string,
) (output string, confidence float64) {
	if e.cfg.Backend != "llamacpp" || job == nil {
		// Demo mode (or missing job): use the nearest-neighbour fallback.
		return e.fallback.Predict(job, trainingExamples, input)
	}

	ggufPath, err := e.resolveGGUF(job.ID)
	if err != nil || ggufPath == "" {
		// No trained GGUF on disk yet — fall back to the demo engine rather
		// than failing the endpoint.
		return e.fallback.Predict(job, trainingExamples, input)
	}

	proc, err := e.ensureServer(job.ID, ggufPath)
	if err != nil {
		e.recordErr(job.ID, err)

		return fmt.Sprintf("(inference backend unavailable: %v)", err), 0
	}

	output, err = e.completion(proc.port, input, "")
	if err != nil {
		// A crashed backend is restarted on the next call (and we retry once
		// right now, which covers a race between health-check and first
		// request).
		e.recordErr(job.ID, err)

		_ = e.killServer(job.ID)

		if p, rErr := e.ensureServer(job.ID, ggufPath); rErr == nil {
			if out, rErr := e.completion(p.port, input, ""); rErr == nil {
				return out, 0.95
			}
		}

		return fmt.Sprintf("(inference error: %v)", err), 0
	}

	return output, 0.95
}

// PredictConstrained implements domain.ConstrainedInferenceEngine: it runs the
// same backend but passes a GBNF grammar to llama-server so decoding is forced
// to produce schema-valid JSON. When the backend is simulation (or no GGUF
// exists) it falls back to the demo engine.
func (e *LlamacppEngine) PredictConstrained(
	job *domain.TrainingJob,
	trainingExamples []*domain.Example,
	input string,
	grammar string,
) (output string, confidence float64) {
	if e.cfg.Backend != "llamacpp" || job == nil || strings.TrimSpace(grammar) == "" {
		return e.fallback.Predict(job, trainingExamples, input)
	}

	ggufPath, err := e.resolveGGUF(job.ID)
	if err != nil || ggufPath == "" {
		return e.fallback.Predict(job, trainingExamples, input)
	}

	proc, err := e.ensureServer(job.ID, ggufPath)
	if err != nil {
		e.recordErr(job.ID, err)

		return fmt.Sprintf("(inference backend unavailable: %v)", err), 0
	}

	output, err = e.completion(proc.port, input, grammar)
	if err != nil {
		e.recordErr(job.ID, err)

		_ = e.killServer(job.ID)

		if p, rErr := e.ensureServer(job.ID, ggufPath); rErr == nil {
			if out, rErr := e.completion(p.port, input, grammar); rErr == nil {
				return out, 0.95
			}
		}

		return fmt.Sprintf("(inference error: %v)", err), 0
	}

	return output, 0.95
}

// Stop shuts down the llama-server process for a deployment (by its job ID)
// and removes it from the registry. Called on POST /deployments/{id}/stop.
func (e *LlamacppEngine) Stop(jobID string) error {
	return e.killServer(jobID)
}

// Close terminates every managed server (cleanup on process shutdown).
func (e *LlamacppEngine) Close() error {
	e.mu.Lock()
	defer e.mu.Unlock()

	var firstErr error

	for id := range e.servers {
		err := e.killServerLocked(id)
		if err != nil && firstErr == nil {
			firstErr = err
		}
	}

	return firstErr
}

// ServerCountForTest returns the number of live managed servers (test helper).
func (e *LlamacppEngine) ServerCountForTest() int {
	e.mu.Lock()
	defer e.mu.Unlock()

	return len(e.servers)
}

// --- internal helpers ---.

// resolveGGUF finds the preferred GGUF file for a completed job in its
// working directory.
func (e *LlamacppEngine) resolveGGUF(jobID string) (string, error) {
	jobDir := filepath.Join(e.cfg.JobsDir, filepath.Base(jobID))

	name, err := localtraining.PreferredGGUF(jobDir)
	if err != nil {
		return "", err
	}

	path := filepath.Join(jobDir, name)
	if !localtraining.FileExists(path) {
		return "", fmt.Errorf("GGUF %q not found for job %s", name, jobID)
	}

	// Completeness gate: refuse to serve an incomplete GGUF (no tokenizer).
	if err := localtraining.ValidateGGUFCompleteness(path); err != nil {
		return "", err
	}

	return path, nil
}

// ensureServer starts (or returns the already-running) llama-server for a job.
func (e *LlamacppEngine) ensureServer(jobID, ggufPath string) (*serverProc, error) {
	e.mu.Lock()
	defer e.mu.Unlock()

	if p, ok := e.servers[jobID]; ok {
		if e.isAlive(p) {
			return p, nil
		}

		// Process died — clean it up so we can restart below.
		_ = e.killServerLocked(jobID)
	}

	port, err := freePort(e.cfg.Host)
	if err != nil {
		return nil, err
	}

	cmd := exec.Command(
		e.cfg.Bin,
		"--model", ggufPath,
		"--host", e.cfg.Host,
		"--port", strconv.Itoa(port),
		"--no-webui",
	)

	// Send server logs to the job dir, mirroring the trainer pattern.
	logDir := filepath.Join(e.cfg.JobsDir, filepath.Base(jobID))
	_ = os.MkdirAll(logDir, 0o755)

	logFile, err := os.Create(filepath.Join(logDir, "llama-server.log"))
	if err == nil {
		cmd.Stdout = logFile
		cmd.Stderr = logFile
	}

	startErr := cmd.Start()
	if startErr != nil {
		if logFile != nil {
			_ = logFile.Close()
		}

		return nil, fmt.Errorf("start llama-server: %w", startErr)
	}

	proc := &serverProc{jobID: jobID, port: port, cmd: cmd, done: make(chan struct{})}
	e.servers[jobID] = proc

	// Reap the process and clear it from the registry on exit (crash or
	// normal shutdown) so the next Predict restarts it.
	go func() {
		_ = cmd.Wait()

		close(proc.done)
		e.mu.Lock()
		if cur, ok := e.servers[jobID]; ok && cur == proc {
			delete(e.servers, jobID)
		}
		e.mu.Unlock()
	}()

	// Health-check in a goroutine so a slow load doesn't block Deploy; the
	// first Predict will wait for readiness.
	go func() {
		err := e.waitHealthy(port, e.cfg.StartupTimeout)
		if err != nil && proc.cmd.Process != nil {
			e.recordErr(jobID, fmt.Errorf("llama-server health: %w", err))
			_ = e.killServer(jobID)
		}
	}()

	return proc, nil
}

// waitHealthy polls the llama-server /health endpoint until it responds.
func (e *LlamacppEngine) waitHealthy(port int, timeout time.Duration) error {
	url := fmt.Sprintf("http://%s:%d/health", e.cfg.Host, port)

	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		resp, err := e.client.Get(url)
		if err == nil {
			_, _ = io.Copy(io.Discard, resp.Body)
			_ = resp.Body.Close()

			if resp.StatusCode == http.StatusOK {
				return nil
			}
		}

		time.Sleep(500 * time.Millisecond)
	}

	return fmt.Errorf("llama-server at %s did not become healthy within %s", url, timeout)
}

// completion calls the llama.cpp /completion endpoint with the input prompt.
// When grammar is non-empty it is passed as llama-server's GBNF `grammar`
// parameter, constraining decoding to schema-valid JSON.
func (e *LlamacppEngine) completion(port int, input, grammar string) (string, error) {
	url := fmt.Sprintf("http://%s:%d/completion", e.cfg.Host, port)

	payload := map[string]any{
		"prompt":       input,
		"n_predict":    128,
		"temperature":  0.7,
		"stop":         []string{"</s>", "<|im_end|>"},
		"stream":       false,
		"cache_prompt": true,
	}

	if strings.TrimSpace(grammar) != "" {
		payload["grammar"] = grammar
	}

	body, err := json.Marshal(payload)
	if err != nil {
		return "", err
	}

	ctx, cancel := context.WithTimeout(context.Background(), e.cfg.HTTPTimeout)
	defer cancel()

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		return "", err
	}

	req.Header.Set("Content-Type", "application/json")

	resp, err := e.client.Do(req)
	if err != nil {
		return "", err
	}

	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		b, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
		return "", fmt.Errorf("llama-server status %d: %s", resp.StatusCode, string(b))
	}

	var out struct {
		Content string `json:"content"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return "", err
	}

	return out.Content, nil
}

// isAlive reports whether the process is still running.
func (e *LlamacppEngine) isAlive(p *serverProc) bool {
	if p == nil || p.cmd == nil || p.cmd.Process == nil {
		return false
	}

	select {
	case <-p.done:
		return false
	default:
		return true
	}
}

// killServer removes the process from the registry and terminates it.
func (e *LlamacppEngine) killServer(jobID string) error {
	e.mu.Lock()
	defer e.mu.Unlock()

	return e.killServerLocked(jobID)
}

func (e *LlamacppEngine) killServerLocked(jobID string) error {
	p, ok := e.servers[jobID]
	if !ok {
		return nil
	}

	delete(e.servers, jobID)

	if p.cmd != nil && p.cmd.Process != nil {
		_ = p.cmd.Process.Kill()
		_ = p.cmd.Wait()
	}

	return nil
}

// recordErr stores the last error for a server (diagnostics).
func (e *LlamacppEngine) recordErr(jobID string, err error) {
	e.mu.Lock()
	defer e.mu.Unlock()

	if p, ok := e.servers[jobID]; ok {
		p.lastErr = err.Error()
	}
}

// freePort finds a free TCP port on the given host.
func freePort(host string) (int, error) {
	ln, err := net.Listen("tcp", host+":0")
	if err != nil {
		return 0, err
	}
	defer ln.Close()

	return ln.Addr().(*net.TCPAddr).Port, nil
}
