package training

import (
	"archive/zip"
	"bytes"
	"encoding/json"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"strings"
	"time"

	"distillery/internal/domain"
)

// LocalExporter implements domain.Exporter for the local training backend.
// It builds the same portable zip as the simulation exporter, but bundles
// the real LoRA adapter weights produced by the Python worker (when present)
// instead of a placeholder, and offers an Ollama/GGUF-ready manifest.
type LocalExporter struct {
	// JobsDir is the same dir the LocalTrainer writes job artifacts to.
	JobsDir string
}

// NewLocalExporter creates a local exporter rooted at cfg.JobsDir.
func NewLocalExporter(cfg Config) *LocalExporter {
	if cfg.JobsDir == "" {
		cfg.JobsDir = filepath.Join(".", "data", "training")
	}
	return &LocalExporter{JobsDir: cfg.JobsDir}
}

// BuildExport creates a zip containing:
//   - manifest.json   (training metadata + real metrics)
//   - Dockerfile      (vLLM LoRA serving image)
//   - adapter/*       (real adapter weights + tokenizer if available)
//   - README.md       (deployment instructions incl. Ollama/GGUF path)
func (e *LocalExporter) BuildExport(task *domain.Task, job *domain.TrainingJob) ([]byte, string, error) {
	if job.Status != domain.TrainingCompleted {
		return nil, "", domain.ErrNoModel
	}

	buf := new(bytes.Buffer)
	zw := zip.NewWriter(buf)

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

	manifestBytes, _ := json.MarshalIndent(manifest, "", "  ")
	if err := writeZipEntry(zw, "manifest.json", manifestBytes); err != nil {
		return nil, "", err
	}

	// Dockerfile — vLLM LoRA serving.
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
`, task.Name, job.BaseModel.RepoID)

	if err := writeZipEntry(zw, "Dockerfile", []byte(dockerfile)); err != nil {
		return nil, "", err
	}

	// Real adapter weights (when the local job produced them).
	adapterDir := filepath.Join(e.JobsDir, job.ID, "adapter")
	hasRealWeights := dirExists(adapterDir)
	if hasRealWeights {
		if err := zipDir(zw, "adapter", adapterDir); err != nil {
			return nil, "", fmt.Errorf("bundling adapter weights: %w", err)
		}
	} else {
		placeholder := []byte("No trained adapter found for this job — it may need to be re-run with TRAINING_BACKEND=local.\n")
		if err := writeZipEntry(zw, "adapter/weights.safetensors.placeholder", placeholder); err != nil {
			return nil, "", err
		}
	}

	// README — deployment instructions incl. Ollama/GGUF.
	var adapterNote string
	if hasRealWeights {
		adapterNote = "adapter/           real trained LoRA adapter + tokenizer"
	} else {
		adapterNote = "adapter/           (placeholder — replace with trained LoRA adapter)"
	}

	readme := fmt.Sprintf(`# %s — Portable Model Export

Base model: %s (%s)
Task type: %s
Fine-tune version: v%d
Backend: local fine-tuning (QLoRA)

## Run it anywhere (vLLM + LoRA)

    docker build -t task-model .
    docker run -p 8000:8000 --gpus all task-model

Then call it like any OpenAI-compatible chat endpoint at
http://localhost:8000/v1/chat/completions

## Deploy with Ollama (GGUF / llama.cpp)

From this export, convert + quantize the merged model:

    # 1. (optional) merge LoRA into base weights:
    python -c "from peft import PeftModel; from transformers import AutoModelForCausalLM; \
    m = AutoModelForCausalLM.from_pretrained('%s', device_map='auto'); \
    m = PeftModel.from_pretrained(m, 'adapter'); m = m.merge_and_unload(); m.save_pretrained('./merged')"

    # 2. convert to GGUF + quantize (via llama.cpp):
    #   python llama.cpp/convert_hf_to_gguf.py merged/ --outfile model.gguf
    #   ./llama.cpp/quantize model.gguf model-Q4_K_M.gguf q4_k_m

    # 3. create in Ollama:
    ollama create task-model -f Modelfile  # Modelfile: FROM ./model-Q4_K_M.gguf

## Contents

- Dockerfile        vLLM-based serving image, LoRA adapter mounted at boot
- manifest.json     training metadata + real metrics for this export
- %s

No platform lock-in: this image runs on your own GPU box, any cloud VM,
or Kubernetes — not just on this platform's hosted inference.
`, task.Name, job.BaseModel.Name, job.BaseModel.RepoID, task.Type, job.Version,
		job.BaseModel.RepoID, adapterNote)

	if err := writeZipEntry(zw, "README.md", []byte(readme)); err != nil {
		return nil, "", err
	}

	if err := zw.Close(); err != nil {
		return nil, "", err
	}

	filename := fmt.Sprintf("%s-v%d-export.zip", safeExportName(task.Name), job.Version)

	return buf.Bytes(), filename, nil
}

// zipDir recursively adds a directory into the zip under prefix.
func zipDir(zw *zip.Writer, prefix, dir string) error {
	return filepath.WalkDir(dir, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if d.IsDir() {
			return nil
		}

		rel, err := filepath.Rel(dir, path)
		if err != nil {
			return err
		}

		data, err := os.ReadFile(path)
		if err != nil {
			return err
		}

		zipName := filepath.ToSlash(filepath.Join(prefix, rel))

		return writeZipEntry(zw, zipName, data)
	})
}

func writeZipEntry(zw *zip.Writer, name string, content []byte) error {
	w, err := zw.Create(name)
	if err != nil {
		return err
	}
	_, err = w.Write(content)
	return err
}

func dirExists(path string) bool {
	st, err := os.Stat(path)
	return err == nil && st.IsDir()
}

func safeExportName(s string) string {
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
