# Distillery — No-Code Fine-Tuning (Task-to-Model Platform)

A working MVP of the "define a task, get a model" platform described in the
product spec: task definition → dataset curation → automatic base-model
selection → managed fine-tuning → one-click deployment → portable export →
continuous improvement loop. Single Go binary, clean architecture, one
built-in web UI, runs the same way on your desktop or in a container.

> **Note on scope:** GPU fine-tuning, model serving, and the frontier-model
> synthetic-data bootstrap are **simulated** in this MVP (clearly labeled in
> the code and UI) so the entire product flow can be exercised end-to-end
> without provisioning GPUs or API keys. The `internal/infra/simulation`
> package is the seam where you'd swap in real implementations — see
> "Swapping in the real thing" below.

## Quick start

### Desktop
```
go build -o distillery ./cmd/server
./distillery
```
Opens `http://localhost:8080` in your default browser automatically. Data
persists to `./data/distillery.json` between runs.

> **GGUF downloads** (`?quantization=f16|q4_k_m|…`): the **simulation** backend
> (the default) has no real trained weights, so trying to export a GGUF there
> returns a clear error telling you to switch to the local backend — it never
> hands you a misleading ~1KB stub. To get a real, full-size F16/quantized
> GGUF you must run the local backend and complete an actual training run:
>
> ```
> # 1. install Python deps for the trainer + GGUF converter
> pip install -r trainer/requirements.txt
> # 2. run with the real backend (needs an NVIDIA GPU)
> TRAINING_BACKEND=local go run ./cmd/server
> ```
> (`make run-local` does the same and also sets a shared `MODEL_CACHE_DIR` so
> base-model weights are downloaded once and reused by both training and GGUF
> conversion.) After a real fine-tune completes, the GGUF export invokes
> `trainer/gguf.py` to merge the LoRA adapter and run llama.cpp — producing
> the actual full-size model file.

### Container
```
docker build -t distillery .
docker run --rm -p 8080:8080 -v distillery-data:/data distillery
```
Then open `http://localhost:8080`. `PORT` and `DATA_PATH` are configurable
env vars; `OPEN_BROWSER` is forced off in the container image.

A `Makefile` wraps both (`make run`, `make docker-build`, `make docker-run`).

## Using it

1. **New Task** — describe the task and pick a type
   (classification / extraction / generation).
2. **Dataset tab** — paste `input -> output` example pairs, upload a CSV
   (`input`/`output` columns), upload a fine-tuning **JSONL** file
   (Alpaca-style `{"instruction","input","output"}` or chat-style
   `{"messages":[...]}` — format auto-detected), or generate synthetic
   variations to bootstrap a small seed set. Distillery dedupes, flags
   low-quality rows, checks label balance/readiness live, and lets you edit
   or delete individual examples inline.
3. **Fine-Tuning tab** — once the dataset is ready, start a run. Distillery
   auto-selects a right-sized base model from a curated catalog and runs a
   (simulated) LoRA/QLoRA fine-tune with live progress. Every completed run
   is kept as a numbered version, and any of them can be deployed —
   deploying an older version is an instant rollback.
4. **Deploy & Test tab** — one-click deploy a model as an API endpoint.
   Deployment issues an **API key shown exactly once** — every prediction
   call must present it (`Authorization: Bearer <key>` or `X-API-Key`).
   Test single predictions inline, run **batch inference** by uploading a
   CSV/text file of inputs and downloading a CSV of predictions, or
   download a portable **export package** (Dockerfile + manifest + adapter
   placeholder) to self-host anywhere. **GGUF download** (see below)
   produces a single self-contained `.gguf` ready for HomeBred-LLM /
   llama.cpp / Ollama.
5. **Feedback Loop tab** — log production mispredictions and fold them back
   into the dataset ahead of the next retrain.
6. **API Docs tab** — ready-to-copy `curl` commands for your live endpoint
   (single + batch), the response shape, and a full endpoint reference —
   so integrating from your own app doesn't require reading the source.

### GGUF export (production-ready)

Distillery can export a finished fine-tune as a **GGUF** model file —
the format used by HomeBred-LLM, llama.cpp, and Ollama. The conversion
pipeline is:

1. **Merge LoRA** into the base model weights (via `peft`).
2. **Convert to GGUF F16** using `llama.cpp`'s `convert_hf_to_gguf.py`.
3. **Quantize** to the requested scheme (`f16`, `q4_k_m`, `q8_0`, …).

This happens **on-demand** (when you request a GGUF download), so a
typical training run doesn't pay the upfront conversion cost.

**How it works**

```
GET /api/v1/tasks/{taskID}/training/{jobID}/gguf?quantization=q4_k_m
```

The Go server invokes `python -m trainer.gguf <args>` and streams the
resulting `.gguf` bytes. Supported features:

- **On-demand conversion** — the converter subprocess runs only when a
  download is requested; nothing is produced during training itself.
- **Caching** — once a quantization has been converted for a given job,
  the file is served from disk (`<job>/gguf/<name>-<quant>.gguf`) on
  subsequent requests — no re-conversion.
- **Concurrency lock** — per-job per-quantization interprocess lock file
  prevents two concurrent requests from running redundant conversions.
  Stale locks (crashed process) are detected and reclaimed automatically.
- **Timeout** — each conversion subprocess is killed after
  `GGUFConversionTimeout` (2h default) so a hung converter can never
  block an HTTP handler indefinitely.
- **Zero-preference manifest** — the converter writes `gguf-manifest.json`
  so subsequent requests pick the exact file without globbing.
- **Adapter → merged fallback** — uses `<job>/adapter` LoRA weights if
  present; falls back to `<job>/model` merged weights otherwise.

**Requirements**

To use the GGUF exporter you need:

1. An NVIDIA GPU + the **local** backend:
   ```
   TRAINING_BACKEND=local go run ./cmd/server   (or: make run-local)
   ```
2. Python deps (both installed via `pip install -r trainer/requirements.txt`):
   - `gguf>=0.10.0` (pure-Python GGUF converter + quantizer — no compilation needed)
   - `llama-cpp-python>=0.2.5` (optional fallback, requires C++ compilation)
   - `torchao>=0.1.0` (optional, for advanced quant schemes)
3. A **completed** training job with real adapter weights on disk.

**Quantization formats**

The requested value maps directly to a llama.cpp quantization scheme:

| `quantization` | Description                          |
|----------------|--------------------------------------|
| `f16`          | Half-precision (no quantization)      |
| `q8_0`         | 8-bit quantization (high precision)   |
| `q4_k_m` (default) | K-quants with 4-bit core, medium   |
| `q5_k_m`       | K-quants with 5-bit core, medium      |
| `q6_k`         | 6-bit K, near-Q8 quality              |

**Container**

The Docker image installs the `gguf` extra (`./trainer[gguf]`) and ships
with `PYTHONPATH=/srv` + `TRAINING_BACKEND=local`, so the same GGUF
pipeline works unchanged in production containers.

## Architecture

Clean architecture, dependency direction flows inward, `internal/domain` has
no dependency on anything else:

```
cmd/server/main.go        wiring / composition root
internal/domain/          entities + repository & service interfaces (ports)
internal/usecase/         application/business logic, depends only on domain
internal/repository/memory/  in-memory store + JSON snapshot persistence
internal/infra/simulation/   simulated fine-tuning, inference, export, etc (adapters)
internal/delivery/http/   HTTP handlers + router (Go 1.22 stdlib ServeMux)
web/                       embedded single-page UI (vanilla JS, no build step)
```

- **domain** defines `TaskRepository`, `FineTuner`, `InferenceEngine`, etc.
  as interfaces — the "ports" in ports-and-adapters terms.
- **usecase** orchestrates domain logic (e.g. `TrainingUsecase.StartTraining`
  validates dataset readiness, selects a base model, and kicks off training)
  without knowing whether storage is in-memory or Postgres, or whether
  training is simulated or a real GPU job.
- **infra/simulation** and **repository/memory** are swappable adapters
  implementing those interfaces.
- **delivery/http** is a thin translation layer between HTTP and usecases.

No third-party Go dependencies — the whole backend is the standard library,
so it builds anywhere with no `go.sum`/network access required.

## API

All endpoints are under `/api/v1`. A few examples:

```
POST   /api/v1/tasks
GET    /api/v1/tasks/{taskID}
POST   /api/v1/tasks/{taskID}/examples
POST   /api/v1/tasks/{taskID}/examples/import        (bulk CSV upload)
POST   /api/v1/tasks/{taskID}/examples/import-jsonl  (bulk Alpaca/chat JSONL upload)
PUT    /api/v1/tasks/{taskID}/examples/{exampleID}    (edit one example)
DELETE /api/v1/tasks/{taskID}/examples/{exampleID}
POST   /api/v1/tasks/{taskID}/examples/synthetic
GET    /api/v1/tasks/{taskID}/dataset/stats
POST   /api/v1/tasks/{taskID}/training
GET    /api/v1/training/{jobID}
POST   /api/v1/tasks/{taskID}/deploy                  (deploy latest, returns one-time API key)
POST   /api/v1/tasks/{taskID}/training/{jobID}/deploy (deploy a specific version — rollback)
POST   /api/v1/inference/{deploymentID}/predict       (requires API key)
POST   /api/v1/inference/{deploymentID}/batch         (CSV/text in, CSV predictions out; requires API key)
GET    /api/v1/tasks/{taskID}/export
POST   /api/v1/tasks/{taskID}/feedback
POST   /api/v1/tasks/{taskID}/feedback/fold
GET    /healthz
```

Every `/predict` and `/batch` call must include the deployment's API key,
either as `Authorization: Bearer <key>` or `X-API-Key: <key>`. The raw key
is returned exactly once, in the response to `POST .../deploy` — only its
SHA-256 hash is ever persisted, so it can't be recovered from storage.
Redeploying (including rolling back to an older version) issues a fresh key
and invalidates the previous one.

## Swapping in the real thing

Each simulated adapter implements one small domain interface, so replacing
it doesn't touch usecases, handlers, or the UI:

- `domain.FineTuner` (`internal/infra/simulation/trainer.go`) → orchestrate
  real LoRA/QLoRA runs (HF `peft`/`trl`) on RunPod/Lambda/Modal.
- `domain.SyntheticGenerator` (`synth_generator.go`) → call the Claude API
  to bootstrap/augment examples from seeds.
- `domain.InferenceEngine` (`inference.go`) → call a real vLLM/TGI endpoint.
- `domain.Exporter` (`exporter.go`) → package real trained weights instead
  of the placeholder file.
- `internal/repository/memory` → swap for Postgres/SQLite if you need
  multi-instance deployment instead of a single-node JSON snapshot.

## Known simplifications (MVP)

- Single-node persistence (JSON snapshot file, guarded by an in-process
  mutex) — fine for one desktop user or one container instance, not for
  horizontally-scaled multi-instance deployment.
- Deployed inference endpoints require a per-deployment API key, but the
  task-management API itself (`/api/v1/tasks/...`) has no auth — add a
  middleware in `internal/delivery/http/router.go` before exposing the
  management API beyond localhost/trusted networks.
- Training/inference are simulated, as noted above.
