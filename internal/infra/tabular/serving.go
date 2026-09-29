package tabular

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"sync"
	"sync/atomic"
	"time"

	"distillery/internal/domain"
)

// ServingConfig locates the Python trainer package and the per-job artifact
// directories written by the local trainer.
type ServingConfig struct {
	PythonBin string
	JobsDir   string
	// WorkDir is the directory containing the `trainer` package (the process
	// working directory of the server by default).
	WorkDir string
	// IdleTimeout stops a worker after this long without requests.
	IdleTimeout time.Duration
	// RequestTimeout bounds one predict/forecast round trip.
	RequestTimeout time.Duration
}

// Engine serves tabular predictions and forecasts from the trained model
// artifacts through a long-lived Python worker per job (so LightGBM and
// SHAP contributions are real and scoring is a pipe round-trip, not a
// process start). Jobs without artifacts (simulation backend) fall back to
// the wrapped engines.
type Engine struct {
	cfg      ServingConfig
	fbTab    domain.TabularInferenceEngine
	fbFc     domain.ForecastInferenceEngine
	mu       sync.Mutex
	workers  map[string]*worker
	closed   bool
	stopOnce sync.Once
}

// NewEngine builds the engine with fallbacks for jobs that have no trained
// artifacts.
func NewEngine(cfg ServingConfig, fbTab domain.TabularInferenceEngine, fbFc domain.ForecastInferenceEngine) *Engine {
	if cfg.PythonBin == "" {
		cfg.PythonBin = "python"
	}

	if cfg.IdleTimeout <= 0 {
		cfg.IdleTimeout = 10 * time.Minute
	}

	if cfg.RequestTimeout <= 0 {
		cfg.RequestTimeout = 30 * time.Second
	}

	return &Engine{cfg: cfg, fbTab: fbTab, fbFc: fbFc, workers: map[string]*worker{}}
}

// PredictTable implements domain.TabularInferenceEngine.
func (e *Engine) PredictTable(
	job *domain.TrainingJob, rows domain.TableRowSource, schema *domain.FeatureSchema, row *domain.ValidatedRow,
) (domain.TablePrediction, error) {
	dir, ok := e.jobDir(job)
	if !ok {
		if e.fbTab == nil {
			return domain.TablePrediction{}, fmt.Errorf("%w: model artifacts not found for job %s", domain.ErrNoModel, job.ID)
		}

		return e.fbTab.PredictTable(job, rows, schema, row)
	}

	var out domain.TablePrediction

	err := e.call(job.ID, dir, map[string]interface{}{"op": "predict", "row": row.Canonical}, &out)
	if err != nil {
		return domain.TablePrediction{}, err
	}

	return out, nil
}

// Forecast implements domain.ForecastInferenceEngine.
func (e *Engine) Forecast(job *domain.TrainingJob, rows domain.TableRowSource, req domain.ForecastRequest) (domain.ForecastResult, error) {
	dir, ok := e.jobDir(job)
	if !ok {
		if e.fbFc == nil {
			return domain.ForecastResult{}, fmt.Errorf("%w: model artifacts not found for job %s", domain.ErrNoModel, job.ID)
		}

		return e.fbFc.Forecast(job, rows, req)
	}

	var out domain.ForecastResult

	err := e.call(job.ID, dir, map[string]interface{}{
		"op": "forecast", "history": req.History, "horizon": req.Horizon, "item_id": req.ItemID,
	}, &out)
	if err != nil {
		return domain.ForecastResult{}, err
	}

	return out, nil
}

// Close stops every worker.
func (e *Engine) Close() {
	e.stopOnce.Do(func() {
		e.mu.Lock()
		defer e.mu.Unlock()

		e.closed = true

		for id, w := range e.workers {
			w.stop()
			delete(e.workers, id)
		}
	})
}

// jobDir returns the artifact directory for a job when it holds a trained
// table model.
func (e *Engine) jobDir(job *domain.TrainingJob) (string, bool) {
	if job == nil || e.cfg.JobsDir == "" || filepath.Base(job.ID) != job.ID {
		return "", false
	}

	dir := filepath.Join(e.cfg.JobsDir, job.ID)

	_, err := os.Stat(filepath.Join(dir, "schema.json"))
	if err != nil {
		return "", false
	}

	return dir, true
}

func (e *Engine) call(jobID, dir string, req map[string]interface{}, out interface{}) error {
	// A dead worker (crash, idle reap) is restarted once per call.
	for range 2 {
		w, err := e.workerFor(jobID, dir)
		if err != nil {
			return err
		}

		raw, err := w.roundTrip(req, e.cfg.RequestTimeout)
		if err == nil {
			return json.Unmarshal(raw, out)
		}

		var wErr *workerError
		if errors.As(err, &wErr) { // The model rejected the input; not a transport fault.
			return fmt.Errorf("%w: %s", domain.ErrInvalidInput, wErr.msg)
		}

		e.drop(jobID, w)
	}

	return fmt.Errorf("%w: model worker for job %s failed", domain.ErrNoModel, jobID)
}

func (e *Engine) drop(jobID string, w *worker) {
	e.mu.Lock()
	defer e.mu.Unlock()

	if e.workers[jobID] == w {
		delete(e.workers, jobID)
	}

	w.stop()
}

var (
	errTimedOut     = errors.New("timed out")
	errWorkerExited = errors.New("worker exited")
	errProtocol     = errors.New("protocol error from model worker")
)

func (e *Engine) workerFor(jobID, dir string) (*worker, error) {
	e.mu.Lock()
	defer e.mu.Unlock()

	if e.closed {
		return nil, fmt.Errorf("%w: serving engine closed", domain.ErrNoModel)
	}

	if w, ok := e.workers[jobID]; ok && w.alive() {
		w.touch()

		return w, nil
	}

	w, err := startWorker(e.cfg, dir, func() {
		e.mu.Lock()
		defer e.mu.Unlock()

		delete(e.workers, jobID)
	})
	if err != nil {
		return nil, err
	}

	e.workers[jobID] = w

	return w, nil
}

// ---------- worker process ----------.

type workerError struct{ msg string }

func (e *workerError) Error() string { return e.msg }

type worker struct {
	cmd    *exec.Cmd
	stdin  io.WriteCloser
	out    *bufio.Reader
	cancel context.CancelFunc

	mu       sync.Mutex // serialises round trips.
	idMu     sync.Mutex
	nextID   int
	lastUsed time.Time
	dead     atomic.Bool
	idle     *time.Timer
}

func startWorker(cfg ServingConfig, dir string, onExit func()) (*worker, error) {
	_, err := exec.LookPath(cfg.PythonBin)
	if err != nil {
		return nil, fmt.Errorf("%w: python binary %q not found: %w", domain.ErrNoModel, cfg.PythonBin, err)
	}

	ctx, cancel := context.WithCancel(context.Background())
	cmd := exec.CommandContext(ctx, cfg.PythonBin, "-m", "trainer.serve_table", "--job-dir", dir) //nolint:gosec,lll // interpreter path comes from server config; arguments are fixed
	cmd.Dir = cfg.WorkDir

	stdin, err := cmd.StdinPipe()
	if err != nil {
		cancel()

		return nil, err
	}

	stdout, err := cmd.StdoutPipe()
	if err != nil {
		cancel()

		return nil, err
	}

	logf, lerr := os.OpenFile(filepath.Join(dir, "serve.log"), os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0o600)
	if lerr == nil {
		cmd.Stderr = logf
	}

	err = cmd.Start()
	if err != nil {
		cancel()

		if lerr == nil {
			_ = logf.Close()
		}

		return nil, err
	}

	w := &worker{cmd: cmd, stdin: stdin, out: bufio.NewReaderSize(stdout, 1<<20), cancel: cancel, lastUsed: time.Now()}

	// The first line is the readiness handshake (or an error).
	line, err := w.readLine(60 * time.Second)
	if err != nil {
		w.stop()

		return nil, fmt.Errorf("%w: model worker did not start: %w", domain.ErrNoModel, err)
	}

	var hello struct {
		Ready bool   `json:"ready"`
		Error string `json:"error"`
	}

	if json.Unmarshal(line, &hello) != nil || !hello.Ready {
		w.stop()

		return nil, fmt.Errorf("%w: model worker failed to load the model: %s", domain.ErrNoModel, hello.Error)
	}

	w.idle = time.AfterFunc(cfg.IdleTimeout, func() {
		w.idleCheck(cfg.IdleTimeout, onExit)
	})

	go func() {
		_ = cmd.Wait()

		if lerr == nil {
			_ = logf.Close()
		}

		w.dead.Store(true)
		onExit()
	}()

	return w, nil
}

func (w *worker) idleCheck(timeout time.Duration, onExit func()) {
	w.idMu.Lock()
	idleFor := time.Since(w.lastUsed)
	w.idMu.Unlock()

	if idleFor >= timeout {
		w.stop()
		onExit()

		return
	}

	w.idle.Reset(timeout - idleFor)
}

func (w *worker) touch() {
	w.idMu.Lock()
	w.lastUsed = time.Now()
	w.idMu.Unlock()
}

func (w *worker) alive() bool { return !w.dead.Load() }

func (w *worker) stop() {
	if w.idle != nil {
		w.idle.Stop()
	}

	_ = w.stdin.Close()
	w.cancel()
}

// readLine reads one line with a timeout (the reader goroutine may outlive a
// timeout; the caller then kills the process).
func (w *worker) readLine(timeout time.Duration) ([]byte, error) {
	type res struct {
		b   []byte
		err error
	}

	ch := make(chan res, 1)

	go func() {
		b, err := w.out.ReadBytes('\n')
		ch <- res{b, err}
	}()

	select {
	case r := <-ch:
		return r.b, r.err
	case <-time.After(timeout):
		return nil, errTimedOut
	}
}

func (w *worker) roundTrip(req map[string]interface{}, timeout time.Duration) (json.RawMessage, error) {
	w.mu.Lock()
	defer w.mu.Unlock()

	if w.dead.Load() {
		return nil, errWorkerExited
	}

	w.touch()

	w.idMu.Lock()
	w.nextID++
	id := w.nextID
	w.idMu.Unlock()

	req["id"] = id

	b, err := json.Marshal(req)
	if err != nil {
		return nil, &workerError{msg: err.Error()}
	}

	_, err = w.stdin.Write(append(b, '\n'))
	if err != nil {
		return nil, err
	}

	line, err := w.readLine(timeout)
	if err != nil {
		w.dead.Store(true)

		return nil, err
	}

	var resp struct {
		ID     int             `json:"id"`
		OK     bool            `json:"ok"`
		Error  string          `json:"error"`
		Result json.RawMessage `json:"result"`
	}

	err = json.Unmarshal(line, &resp)
	if err != nil || resp.ID != id {
		w.dead.Store(true)

		return nil, errProtocol
	}

	if !resp.OK {
		return nil, &workerError{msg: resp.Error}
	}

	return resp.Result, nil
}
