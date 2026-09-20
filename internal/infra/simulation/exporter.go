package simulation

import (
	"archive/zip"
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"time"

	"distillery/internal/domain"
)

// errGGUFNotConfigured is returned when BuildGGUF is called on an Exporter
// with no real production converter wired up. It tells the user how to get a
// real, full-size GGUF instead of fabricating a worthless pseudo-random one.
var errGGUFNotConfigured = errors.New(
	"GGUF export is not configured for this backend. Run the server with " +
		"TRAINING_BACKEND=local (needs an NVIDIA GPU + pip install -r " +
		"trainer/requirements.txt) so a real, full-size GGUF can be produced " +
		"from trained weights, then re-export",
)

// Exporter implements domain.Exporter and domain.GGUFExporter. It builds a
// "run anywhere" export package: a manifest, a self-host Dockerfile
// (vLLM/TGI serving image spec), and a placeholder weights file, as a
// downloadable zip. This is the no-lock-in wedge called out in the product
// spec.
//
// GGUF export is delegated to a real production converter
// (domain.GGUFExporter, e.g. the local training backend's converter) when
// one is configured. When no trained adapter or merged model exists on disk
// (the normal simulation case), the converter falls back to converting the
// base model itself — so the user still gets a real, full-size, loadable
// GGUF rather than a fake pseudo-random file.
type Exporter struct {
	// ggufExporter is the optional real GGUF converter used by BuildGGUF.
	// When nil, BuildGGUF returns a clear error explaining that no converter
	// is wired up.
	ggufExporter domain.GGUFExporter
}

// NewExporter creates a simulation exporter with no GGUF converter wired up.
// Use NewExporterWithGGUF to inject a real converter.
func NewExporter() *Exporter { return &Exporter{} }

// NewExporterWithGGUF creates a simulation exporter that delegates GGUF
// export to the supplied real converter (typically a
// training.LocalExporter). This lets the simulation backend serve real GGUF
// files produced from actual trained weights on disk instead of a fake
// placeholder.
func NewExporterWithGGUF(ggufExporter domain.GGUFExporter) *Exporter {
	return &Exporter{ggufExporter: ggufExporter}
}

// BuildGGUF implements domain.GGUFExporter. It delegates to the configured
// real production converter. When no converter is wired up, it fails fast
// with an actionable message rather than fabricating a worthless
// pseudo-random GGUF. When a converter is wired up but no trained weights
// exist on disk, the converter falls back to base-model conversion.
func (e *Exporter) BuildGGUF(task *domain.Task, job *domain.TrainingJob, opts domain.GGUFExportOptions) ([]byte, string, error) {
	if job.Status != domain.TrainingCompleted {
		return nil, "", domain.ErrNoModel
	}

	if e.ggufExporter == nil {
		return nil, "", errGGUFNotConfigured
	}

	// Delegate to the real production converter. When no trained adapter
	// or merged model exists on disk (the normal simulation case), the
	// converter falls back to converting the base model itself — so the
	// user gets a real, full-size, loadable GGUF rather than a fake file.
	return e.ggufExporter.BuildGGUF(task, job, opts)
}

// StartGGUFAsync forwards to the wrapped converter's async conversion.
func (e *Exporter) StartGGUFAsync(sessionID, taskID, jobID string, task *domain.Task, job *domain.TrainingJob, opts domain.GGUFExportOptions) {
	if asyncExp, ok := e.ggufExporter.(domain.AsyncGGUFExporter); ok {
		asyncExp.StartGGUFAsync(sessionID, taskID, jobID, task, job, opts)
	}
}

// GetGGUFProgress forwards to the wrapped converter's progress lookup.
func (e *Exporter) GetGGUFProgress(sessionID string) *domain.GGUFProgressInfo {
	if asyncExp, ok := e.ggufExporter.(domain.AsyncGGUFExporter); ok {
		return asyncExp.GetGGUFProgress(sessionID)
	}

	return nil
}

// GetGGUFResult forwards to the wrapped converter's result retrieval.
func (e *Exporter) GetGGUFResult(sessionID string) (string, string, error) {
	if asyncExp, ok := e.ggufExporter.(domain.AsyncGGUFExporter); ok {
		return asyncExp.GetGGUFResult(sessionID)
	}

	return "", "", errGGUFNotConfigured
}

// CleanupGGUFSession forwards to the wrapped converter's cleanup.
func (e *Exporter) CleanupGGUFSession(sessionID string) {
	if asyncExp, ok := e.ggufExporter.(domain.AsyncGGUFExporter); ok {
		asyncExp.CleanupGGUFSession(sessionID)
	}
}

func (e *Exporter) BuildExport(task *domain.Task, job *domain.TrainingJob) (data []byte, filename string, err error) {
	if job.Status != domain.TrainingCompleted {
		return nil, "", domain.ErrNoModel
	}

	buf := new(bytes.Buffer)
	zw := zip.NewWriter(buf)

	manifest := map[string]interface{}{
		"task_name":      task.Name,
		"task_type":      task.Type,
		"base_model":     job.BaseModel.Name,
		"training_job":   job.ID,
		"version":        job.Version,
		"metrics":        job.Metrics,
		"adapter_format": "LoRA (safetensors)",
		"exported_at":    time.Now().UTC().Format(time.RFC3339),
		"note": "Demo export from a no-code fine-tuning MVP. " +
			"Replace adapter/weights.safetensors with real trained LoRA weights before serving.",
	}

	manifestBytes, _ := json.MarshalIndent(manifest, "", "  ")

	err = writeZipFile(zw, "manifest.json", manifestBytes)
	if err != nil {
		return nil, "", err
	}

	dockerfile := fmt.Sprintf(`# Self-hosted serving image for %q
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
`, task.Name, job.BaseModel.Name)

	err = writeZipFile(zw, "Dockerfile", []byte(dockerfile))
	if err != nil {
		return nil, "", err
	}

	readme := fmt.Sprintf(`# %s — Portable Model Export

Base model: %s
Task type: %s
Fine-tune version: v%d
Eval accuracy (simulated): %.2f

## Run it anywhere

    docker build -t task-model .
    docker run -p 8000:8000 --gpus all task-model

Then call it like any OpenAI-compatible chat endpoint at
http://localhost:8000/v1/chat/completions

## Contents

- Dockerfile        vLLM-based serving image, LoRA adapter mounted at boot
- manifest.json      training metadata for this export
- adapter/weights.safetensors   placeholder — swap in real trained weights

No platform lock-in: this image runs on your own GPU box, any cloud VM,
or Kubernetes — not just on this platform's hosted inference.
`, task.Name, job.BaseModel.Name, task.Type, job.Version, safeAcc(job))

	err = writeZipFile(zw, "README.md", []byte(readme))
	if err != nil {
		return nil, "", err
	}

	placeholder := []byte("This is a placeholder for the trained LoRA adapter weights (safetensors).\n" +
		"In the full platform this file contains the actual fine-tuned adapter produced by the training job.\n")

	err = writeZipFile(zw, "adapter/weights.safetensors.placeholder", placeholder)
	if err != nil {
		return nil, "", err
	}

	err = zw.Close()
	if err != nil {
		return nil, "", err
	}

	filename = fmt.Sprintf("%s-v%d-export.zip", SafeName(task.Name), job.Version)

	return buf.Bytes(), filename, nil
}

func safeAcc(job *domain.TrainingJob) float64 {
	if job.Metrics == nil {
		return 0
	}

	return job.Metrics.EvalAccuracy
}

// SafeName converts a human label into a filesystem-safe slug.
func SafeName(s string) string {
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

	return string(out)
}

func writeZipFile(zw *zip.Writer, name string, content []byte) error {
	w, err := zw.Create(name)
	if err != nil {
		return err
	}

	_, err = w.Write(content)

	return err
}
