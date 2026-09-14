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
   (`input`/`output` columns), or generate synthetic variations to bootstrap
   a small seed set. Distillery dedupes, flags low-quality rows, checks
   label balance/readiness live, and lets you edit or delete individual
   examples inline.
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
   placeholder) to self-host anywhere.
5. **Feedback Loop tab** — log production mispredictions and fold them back
   into the dataset ahead of the next retrain.
6. **API Docs tab** — ready-to-copy `curl` commands for your live endpoint
   (single + batch), the response shape, and a full endpoint reference —
   so integrating from your own app doesn't require reading the source.

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
