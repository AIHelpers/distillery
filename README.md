# Distillery — No-Code Fine-Tuning (Task-to-Model Platform)

A working MVP of the "define a task, get a model" platform described in the
product spec: task definition → dataset curation → automatic base-model
selection → managed fine-tuning → one-click deployment → portable export →
continuous improvement loop. Single Go binary, clean architecture, one
built-in web UI, runs the same way on your desktop or in a container.

Distillery ships with a **real, production-grade QLoRA fine-tuning backend**
implemented end-to-end (see "Features" and "Real fine-tuning backend" below).
A simulated backend is bundled as the **default** merely so you can exercise
the entire product flow without provisioning a GPU — flip one env var
(`TRAINING_BACKEND=local`) to run actual gradient-descent training with 4-bit
quantized LoRA on your own machine.

## Features

- **Coding fine-tuning** - target a language + skill, get AI-recommended hyperparameters, dataset-quality analysis, and live training monitoring.
- **Agent orchestration** - autonomous agents that validate datasets, pick base models, launch training, and run a code-evaluator tool.
- **Model stores** - manage base + trained model artifacts across local dirs, HuggingFace Hub, and S3 buckets.
- **Real local QLoRA backend** - Go/Python worker contract drives real GPU fine-tuning (trainer/run.py) — implemented, not a stub.
- **Production GGUF export** - sync or async; merge LoRA, convert, quantize, cached per job+quantization.
- **Embedding & reranker models** - fine-tune sentence-embedding bi-encoders and cross-encoder rerankers on pairs/triplets/graded relevance; serve via `/embed` and `/rerank`, evaluate nDCG@10/MRR/recall base-vs-tuned, and export corpus vectors for any vector DB. See [docs/embeddings-rerankers.md](docs/embeddings-rerankers.md).
- **Simulated demo backend** - `simulation` adapter (default) fakes training/inference so the whole flow works with no GPU. Clearly labeled in the code and UI. Switch to `local` for real training.

## Real fine-tuning backend

This is not a placeholder. The `local` backend is a complete, working
implementation:

- **`trainer/run.py`** — a full HuggingFace `Trainer`-based QLoRA worker:
  - loads a base model from HuggingFace (with `BitsAndBytesConfig` 4-bit
    NF4 quantization when a GPU is present),
  - applies a `peft.LoraConfig` adapter (`q/k/v/o_proj` by default),
  - tokenizes both Alpaca (`{"instruction","input","output"}`) and chat
    (`{"messages":[...]}`) datasets,
  - runs distributed-style training with validation split, checkpoints,
    best-model selection, warmup/scheduler, and gradient checkpointing,
  - streamed **JSONL progress** (loss, eval loss, LR, epoch, step) that the
    Go side tails for live UI updates,
  - writes `metrics.json` (eval loss, epochs, steps) plus the saved LoRA
    adapter + tokenizer on completion.
- **`internal/infra/training/local_trainer.go`** — the Go adapter implementing
  `domain.FineTuner`:
  - launches `python -m trainer.run` as a managed subprocess,
  - **VRAM preflight** — checks `nvidia-smi` free memory against
    `BaseModel.MinVRAMGB`, with an opt-in CPU fallback,
  - **pause** (SIGTERM → graceful checkpoint + exit) and **cancel** (kill),
  - tails `progress.jsonl` → reports percentages to the UI,
  - validates safe job IDs (path-traversal neutralized), streams trainer
    logs to `trainer.log`, and enforces checkpoint retention (`MaxJobHistory`).
- **`internal/infra/training/exporter.go` + `trainer/gguf.py`** — the real
  production GGUF converter (see "GGUF export" below).
- **`scripts/`** — helpers to pre-pull base-model weights and repair Qwen3
  RoPE in GGUF exports.

### Enable it

```
# 1. install Python deps for the trainer + GGUF converter
pip install -r trainer/requirements.txt

# 2. run with the real backend (needs an NVIDIA GPU; set TRAINING_CPU_FALLBACK=true to train on CPU)
TRAINING_BACKEND=local go run ./cmd/server
```

`make run-local` does the same and also sets a shared `MODEL_CACHE_DIR` so
base-model weights are downloaded once and reused by both training and GGUF
conversion. Docker builds already ship with `TRAINING_BACKEND=local` and the
GGUF extra pre-installed.

The simulation backend stays the default because the whole point is that a
no-GPU machine can demo the product flow. The moment you set
`TRAINING_BACKEND=local`, every training run you start is a **real QLoRA
fine-tune** producing actual adapter weights on disk, ready for GGUF export
and deployment.

## Quick start

### Desktop
```
go build -o distillery ./cmd/server
./distillery
```
Opens `http://localhost:8080` in your default browser automatically. Data
persists to `./data/distillery.json` between runs.

Default mode runs the **simulation** backend (no GPU needed) so you can click
through the whole flow. To do **real fine-tuning**, see "Real fine-tuning
backend" above.

### GGUF downloads (`?quantization=f16|q4_k_m|…`)

- **Simulation backend (default):** no trained weights exist on disk unless a
  real `local` run completed, so a GGUF request delegates to the production
  converter which **fails fast** with a clear "switch to the local backend"
  error — it never fabricates a fake file.
- **Local backend (real training):** once real weights exist on disk, GGUF
  export invokes `trainer/gguf.py` to merge the LoRA adapter and run
  llama.cpp — producing the actual full-size model file (see "GGUF export").

### Container
```
docker build -t distillery .
docker run --rm -p 8080:8080 -v distillery-data:/data distillery
```
Then open `http://localhost:8080`. `PORT` and `DATA_PATH` are configurable
env vars; `OPEN_BROWSER` is forced off in the container image.

A `Makefile` wraps both (`make run`, `make docker-build`, `make docker-run`).

## Using it

The web UI has three pages: **Tasks** (`/`), **Agents** (`/agent.html`), and **Models** (`/models.html`). The **Tasks** flow:

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
   LoRA/QLoRA fine-tune with live progress. With `TRAINING_BACKEND=local`
   this is a **real QLoRA run** on your GPU. Every completed run
   is kept as a numbered version, and any of them can be deployed —
   deploying an older version is an instant rollback.
4. **Deploy & Test tab** — one-click deploy a model as an API endpoint.
   Deployment issues an **API key shown exactly once** — every prediction
   call must present it (`Authorization: Bearer <key>` or `X-API-Key`).
   Test single predictions inline, run **batch inference** by uploading a
   CSV/text file of inputs and downloading a CSV of predictions, or
   download a portable **export package** (Dockerfile + manifest + adapter
   weights) to self-host anywhere. **GGUF download** (see below)
   produces a single self-contained `.gguf` ready for HomeBred-LLM /
   llama.cpp / Ollama.
5. **Feedback Loop tab** — log production mispredictions and fold them back
   into the dataset ahead of the next retrain.
6. **API Docs tab** — ready-to-copy `curl` commands for your live endpoint
   (single + batch), the response shape, and a full endpoint reference —
   so integrating from your own app doesn't require reading the source.

### Coding fine-tuning (AI-agent driven)

The fine-tuning subsystem supports coding tasks: pick a language (Python/Go/JS/TS/Java/C#/Rust/C++) and a skill (code generation/completion/bug fixing/optimization/documentation). Agents analyze datasets (quality, syntax validity), recommend hyperparameters, launch QLoRA jobs, and report quality (incl. code-executability). The **Agents** page monitors and controls them; the **Models** page manages base + trained model artifacts in model stores (local dir, HuggingFace Hub, S3).

### GGUF export (production-ready)

Distillery can export a finished fine-tune as a **GGUF** model file —
the format used by HomeBred-LLM, llama.cpp, and Ollama. The conversion
pipeline is:

1. **Merge LoRA** into the base model weights (via `peft`).
2. **Convert to GGUF F16** (pure-Python `gguf` package or llama.cpp's
   `convert_hf_to_gguf.py`).
3. **Quantize** to the requested scheme (`f16`, `q4_k_m`, `q8_0`, …) inline.

This happens **on-demand** (when you request a GGUF download), so a
typical training run doesn't pay the upfront conversion cost.

**How it works**

```
GET /api/v1/tasks/{taskID}/training/{jobID}/export/gguf?quantization=q4_k_m
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
- **Completeness gate** — every served GGUF (fresh or cached) is validated
  for tokenizer metadata so llama.cpp / HomeBred-LLM can actually load it;
  broken cache entries are removed and re-converted.
- **Adapter → merged fallback** — uses `<job>/adapter` LoRA weights if
  present; falls back to `<job>/model` merged weights otherwise; if neither
  exists (simulation backend), it converts the base model itself so you still
  get a real, loadable GGUF — never a fake file.

**Requirements**

To produce a real GGUF from trained weights you need:

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
internal/infra/simulation/   demo backend: simulated training, inference, synthetic gen (adapters)
internal/infra/training/      real local QLoRA trainer + production GGUF exporter
internal/infra/agent/         coding/reflection agents, LLM provider, tools
internal/infra/modelstore/    model artifact transfer (local / HuggingFace / S3)
internal/infra/code/          code parsers (syntax validity, language coverage)
internal/exhaustruct/         struct-literal completeness tool
internal/delivery/http/   HTTP handlers + router (Go 1.22 stdlib ServeMux)
trainer/                   Python package: run.py (worker), gguf.py, eval/, progress.py
scripts/                   helper scripts (model pre-pull, GGUF repair)
web/                       embedded single-page UI (vanilla JS, no build step)
```

- **domain** defines `TaskRepository`, `FineTuner`, `InferenceEngine`, etc.
  as interfaces — the "ports" in ports-and-adapters terms.
- **usecase** orchestrates domain logic (e.g. `TrainingUsecase.StartTraining`
  validates dataset readiness, selects a base model, and kicks off training)
  without knowing whether storage is in-memory or Postgres, or whether
  training is simulated or a real GPU job.
- **infra/simulation** and **infra/training** are swappable adapters
  implementing those interfaces; `repository/memory` is the storage adapter.
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
POST   /api/v1/inference/{deploymentID}/embed         (embedding model; requires API key)
POST   /api/v1/inference/{deploymentID}/rerank        (reranker model; requires API key)
POST   /api/v1/inference/{deploymentID}/embed-corpus  (embed a whole corpus, CSV out; requires API key)
GET    /api/v1/tasks/{taskID}/export
POST   /api/v1/tasks/{taskID}/feedback
POST   /api/v1/tasks/{taskID}/feedback/fold
POST   /api/v1/tasks/{taskID}/deployments
GET    /api/v1/tasks/{taskID}/deployments
GET    /api/v1/tasks/{taskID}/deployments/active
POST   /api/v1/deployments/{deploymentID}/stop
GET    /api/v1/tasks/{taskID}/export/gguf            (sync GGUF - latest model)
GET    /api/v1/tasks/{taskID}/training/{jobID}/export/gguf
POST   /api/v1/tasks/{taskID}/export/gguf/async      (start async GGUF)
POST   /api/v1/tasks/{taskID}/training/{jobID}/export/gguf/async
GET    /api/v1/tasks/{taskID}/export/gguf/progress/{sessionID}
GET    /api/v1/tasks/{taskID}/export/gguf/download/{sessionID}
GET    /api/v1/models                                 (base-model catalog)
POST   /api/v1/agents/fine-tuning                     (start fine-tuning agent)
GET    /api/v1/agents
GET    /api/v1/agents/{agentID}
POST   /api/v1/agents/{agentID}/pause
POST   /api/v1/agents/{agentID}/resume
POST   /api/v1/model-stores                           (configure a store)
GET    /api/v1/model-stores
GET    /api/v1/model-stores/{storeID}
PATCH  /api/v1/model-stores/{storeID}                 (enable/disable)
DELETE /api/v1/model-stores/{storeID}
GET    /api/v1/model-stores/{storeID}/models
POST   /api/v1/model-stores/{storeID}/models/download
POST   /api/v1/model-stores/{storeID}/models/upload
POST   /api/v1/fine-tuning/datasets
GET    /api/v1/fine-tuning/datasets/{datasetID}/analyse
POST   /api/v1/fine-tuning/hyperparameters
POST   /api/v1/fine-tuning/requests
POST   /api/v1/fine-tuning/requests/{requestID}/start
GET    /api/v1/fine-tuning/jobs
GET    /api/v1/fine-tuning/jobs/{jobID}/monitor
GET    /api/v1/fine-tuning/jobs/{jobID}/insights
GET    /api/v1/fine-tuning/jobs/{jobID}/quality
GET    /api/v1/fine-tuning/models
GET    /healthz
```

Every `/predict` and `/batch` call must include the deployment's API key,
either as `Authorization: Bearer <key>` or `X-API-Key: <key>`. The raw key
is returned exactly once, in the response to `POST .../deploy` — only its
SHA-256 hash is ever persisted, so it can't be recovered from storage.
Redeploying (including rolling back to an older version) issues a fresh key
and invalidates the previous one.

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| PORT | 8080 | HTTP listen port |
| DATA_PATH | ./data/distillery.json | JSON snapshot persistence path |
| OPEN_BROWSER | auto | Open browser on start: true/false/auto |
| TRAINING_BACKEND | simulation | simulation (no GPU demo) or local (real QLoRA) |
| TRAINING_CPU_FALLBACK | false | true allows local training on CPU |
| TRAINING_MAX_JOB_HISTORY | 0 | Cap on retained training-job history |
| MODEL_CACHE_DIR | - | Shared HF cache for base-model download reuse |

CLI flag: --exhaustruct runs the internal struct-rewrite tool and exits.

## Swapping in the real thing

Two kinds of "real" things exist here.

**Already implemented — no swap needed:**

- `domain.FineTuner` — implemented for real by `internal/infra/training`
  (QLoRA via `trainer/run.py`). The simulation adapter just shares the same
  interface so the no-GPU demo works.
- `domain.Exporter` / `domain.GGUFExporter` — implemented by
  `internal/infra/training/exporter.go` (production GGUF converter);
  even the simulation backend delegates GGUF to this real converter.

**Still simulated — swap these adapters to go fully live:**

- `domain.SyntheticGenerator` (`infra/simulation/synth_generator.go`) → currently
  template-based variations; to do real frontier-model bootstrap/augmentation,
  replace this adapter with a Claude/OpenAI API call.
- `domain.InferenceEngine` (`infra/simulation/inference.go`) → currently a
  nearest-neighbor lookup against the training set (labeled as a demo
  stand-in for a served fine-tuned model); replace with a real vLLM/TGI
  endpoint call.
- `infra/agent.SimulatedLLMProvider` (`agent/llm_provider.go`) → the
  coding/reflection agents currently use a deterministic simulated LLM that
  walks the tool registry; swap it for a real LLM provider to get genuine
  agent reasoning.
- `internal/repository/memory` → swap for Postgres/SQLite if you need
  multi-instance deployment instead of a single-node JSON snapshot.

Each simulated adapter implements one small domain interface, so replacing
it doesn't touch usecases, handlers, or the UI.

## Known simplifications (MVP)

- The **default** `TRAINING_BACKEND=simulation` demos the full product flow
  with fake training/inference so no GPU or API keys are needed (clearly
  labeled). Set `TRAINING_BACKEND=local` for real QLoRA fine-tuning and
  GGUF export. What remains genuinely simulated even in `local` mode:
  model serving/inference (`simulation.InferenceEngine`), synthetic-data
  bootstrap (template-based, not a real frontier-model call), and the
  agents' LLM reasoning (`SimulatedLLMProvider`).
- Single-node persistence (JSON snapshot file, guarded by an in-process
  mutex) — fine for one desktop user or one container instance, not for
  horizontally-scaled multi-instance deployment.
- Deployed inference endpoints require a per-deployment API key, but the
  task-management API itself (`/api/v1/tasks/...`) has no auth — add a
  middleware in `internal/delivery/http/router.go` before exposing the
  management API beyond localhost/trusted networks.