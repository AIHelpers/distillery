package simulation

import (
	"archive/zip"
	"bytes"
	"encoding/json"
	"fmt"
	"time"

	"distillery/internal/domain"
)

// Exporter implements domain.Exporter. It builds a "run anywhere" export
// package: a manifest, a self-host Dockerfile (vLLM/TGI serving image spec),
// and a placeholder weights file, as a downloadable zip. This is the
// no-lock-in wedge called out in the product spec.
type Exporter struct{}

func NewExporter() *Exporter { return &Exporter{} }

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
	if err := writeZipFile(zw, "manifest.json", manifestBytes); err != nil {
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
	if err := writeZipFile(zw, "Dockerfile", []byte(dockerfile)); err != nil {
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
	if err := writeZipFile(zw, "README.md", []byte(readme)); err != nil {
		return nil, "", err
	}

	placeholder := []byte("This is a placeholder for the trained LoRA adapter weights (safetensors).\n" +
		"In the full platform this file contains the actual fine-tuned adapter produced by the training job.\n")
	if err := writeZipFile(zw, "adapter/weights.safetensors.placeholder", placeholder); err != nil {
		return nil, "", err
	}

	if err := zw.Close(); err != nil {
		return nil, "", err
	}

	filename = fmt.Sprintf("%s-v%d-export.zip", safeName(task.Name), job.Version)
	return buf.Bytes(), filename, nil
}

func safeAcc(job *domain.TrainingJob) float64 {
	if job.Metrics == nil {
		return 0
	}
	return job.Metrics.EvalAccuracy
}

func safeName(s string) string {
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
