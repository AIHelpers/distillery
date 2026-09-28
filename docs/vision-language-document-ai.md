# Vision-Language / Document AI

Distillery can fine-tune **vision-language models (VLMs)** on image+prompt→
answer examples — receipts, invoices, forms, IDs, scanned contracts — turning
a handful of labeled pages into a model that extracts structured fields
without an OCR pipeline or a hand-written parser.

## 1. Data format

A `vision_lm` example's data lives in a typed `Payload` (not the legacy
`input`/`output` row layout other kinds use):

```json
{
  "image": "a1b2c3d4e5f6.png",
  "prompt": "Extract vendor, total and date as JSON.",
  "answer": "{\"vendor\":\"Acme\",\"total\":4200.00,\"date\":\"2026-05-03\"}",
  "doc_id": "inv_0042",
  "page": 1
}
```

`image` is never raw bytes inline — it's a key into the task's blob store
(see "Blob storage" below). `answer` is free text, or JSON text when the
task's type is Extraction and it declares a JSON Schema (the same "Track B"
schema card other kinds use, reused here since document extraction is
JSON-schema-shaped by nature). An example can be added with an empty
`answer` — "needs answer" — for later human correction, which the dataset
gallery surfaces as its own status.

Three ways to get examples in:

- **One at a time** — `POST /tasks/{id}/examples/vision` with a base64 (or
  `data:image/png;base64,...`) image, a prompt, and an optional answer.
- **A ZIP + manifest** — `POST /tasks/{id}/examples/import-vision-zip`: a ZIP
  archive of page images plus a `data.jsonl` manifest (one JSON object per
  line, image paths relative to the archive root):
  ```json
  {"image": "images/inv_0042.png", "prompt": "Extract vendor, total and date as JSON.", "answer": "{\"vendor\":\"Acme\"}"}
  ```
  Manifest lines whose image is missing or fails validation are skipped
  (partial success), the same behavior as the other bulk importers.
- **A PDF** — `POST /tasks/{id}/examples/import-vision-pdf?prompt=...&dpi=150`:
  each page is rasterized to a PNG and added as its own example sharing the
  same prompt, so a multi-page document becomes one example per page ready
  for per-page correction.

Every uploaded image is decoded and validated before it reaches the blob
store: it must actually decode (PNG/JPEG/GIF), stay under 15 MiB, and stay
under 40 megapixels — bounds chosen to block decompression-bomb-style images
without rejecting a legitimate high-DPI scan. A dataset ZIP is capped at
200 MiB / 20,000 files; a single `/predict` or PDF upload at 20 MiB / 50 MiB
respectively.

### Blob storage

Images are stored via `domain.BlobStore` — a minimal local-disk
implementation (`internal/infra/blob.LocalStore`) sharded one directory per
task, sized and structured like `internal/infra/modelstore`'s existing
local/HF/S3 layering so a future S3-compatible implementation is a drop-in
behind the same interface, not a rewrite. `GET /tasks/{id}/blobs/{key}`
serves a stored image back (used by the dataset gallery and the test-tab
preview); nothing else exposes blob bytes.

### Retention

Each task has a `retention_days` setting (`PATCH /tasks/{id}`). Unset (`0`)
defaults to 90 days; a negative value disables the sweep for that task
("keep forever" is an explicit opt-in, not the default). A background sweep
(`SweepExpiredBlobs`, interval set by `RETENTION_SWEEP_INTERVAL`, default
24h) deletes images — and the
examples that reference them — past their task's window. This exists because
document images routinely contain PII/financial data (per the plan's privacy
risk callout): the safe default is "don't keep it longer than you have to,"
not "keep everything and let someone remember to clean up."

## 2. Base models

| Model | Family | Params | Min VRAM | Max image | Notes |
|---|---|---|---|---|---|
| `Qwen2-VL-2B-Instruct` | Qwen2-VL | 2B | 6 GB | ~1280×1280 (dynamic) | **Recommended default** |
| `Qwen2-VL-7B-Instruct` | Qwen2-VL | 7B | 16 GB | ~1280×1280 (dynamic) | Higher quality, more VRAM |
| `PaliGemma-3B-mix` | PaliGemma | 3B | 8 GB | 448×448 (fixed) | Catalog entry only — see scope |
| `Florence-2-large` | Florence-2 | 0.77B | 4 GB (CPU-capable) | 768×768 | Catalog entry only — see scope |

The model selector right-sizes from this catalog the same way it does for
every other kind (dataset size / capability match); `MaxImagePixels` on each
model's `Capabilities` is what a future importer-side downscale step would
key off of for a model with a small fixed input size (PaliGemma, Florence-2).

## 3. Training configuration

```jsonc
// VisionConfig — Go domain.VisionConfig / Python trainer.tasks.vision.Config
{
  "max_image_side": 1280,          // downscale cap on the longest edge
  "max_new_tokens": 256,           // generation length cap (eval + inference)
  "freeze_vision_encoder": true,   // LoRA on the language tower only (default, cheaper)
  "epochs": 3,
  "learning_rate": 0.0001,
  "batch_size": 1,                 // per-device; VLM activations are large
  "gradient_accumulation_steps": 8
}
```

`trainer/tasks/vision.py` implements **one model family end-to-end: Qwen2-VL**
(`Qwen2VLForConditionalGeneration` + `AutoProcessor`), per the plan's own
step-by-step scope ("vision.py for one model family (Qwen-VL)"). PaliGemma /
Florence-2 / LayoutLMv3 / Donut are catalog entries so they show up in model
selection and cost estimates, but training one currently fails at the
trainer step — see "Scope & known limitations."

Training details:

- **4-bit QLoRA** (`bitsandbytes` NF4) with LoRA on the language tower;
  the vision encoder is frozen by default (`freeze_vision_encoder: true`),
  which is what keeps a 2B–7B VLM fine-tune in reach of a single consumer
  GPU — un-freezing it is a config flag, not a code change.
- A **chat-template collator** builds Qwen2-VL's multimodal input from each
  record's image + prompt + answer and masks the prompt tokens out of the
  loss (only the answer contributes to the loss, same convention as the
  causal_lm SFT collator).
- **Held-out evaluation**: 15% of records (`validation_split`), capped at
  `max_validation_samples` (30) generations, mirroring how `sft.py` bounds
  its own JSON-validity generation pass so eval doesn't dominate wall clock.
- **Zero-shot baseline**: after evaluating the fine-tuned adapter, the same
  loaded model is re-run on the same held-out sample with the LoRA adapter
  disabled (`peft`'s `disable_adapter()` context manager) — a true zero-shot
  pass at no extra model-load cost, unlike `classifier.py`'s majority-class
  "sanity floor" baseline (which stands in for a zero-shot pass that isn't
  cheaply available there). `baseline_field_f1` / `delta_field_f1` report
  the gain, which is what the plan's definition of done is checked against.

## 4. Serving / inference API

A `vision_lm` deployment's `/predict` is **multipart**, not JSON like every
other kind — an image can't reasonably go in a JSON body:

```http
POST /api/v1/inference/{deploymentID}/predict
Authorization: Bearer <key>
Content-Type: multipart/form-data; boundary=...

--boundary
Content-Disposition: form-data; name="file"; filename="invoice.png"
Content-Type: image/png

<binary image bytes>
--boundary
Content-Disposition: form-data; name="prompt"

Extract vendor, total and date as JSON.
--boundary--
```

```json
{
  "kind": "vision_lm",
  "result": {
    "text": "{\"vendor\":\"Acme\",\"total\":4200.00,\"date\":\"2026-05-03\"}",
    "json": { "vendor": "Acme", "total": 4200.0, "date": "2026-05-03" },
    "json_valid": true
  }
}
```

The image field accepts `file`, `image`, or `file0` (first match wins), so
existing multipart-form conventions from other tools work without
translation. `json` is only populated when `text` parses as JSON; otherwise
`json_valid` is `false` and `json` is omitted. There is currently no
batch (ZIP-in/JSONL-out) vision inference endpoint — `/predict` handles one
image per call; see "Scope & known limitations."

## 5. Evaluation

- **Field-level exact-match / F1**, per field, with normalization (currency
  symbols, thousands separators, percent signs, and whitespace are stripped
  and casing is folded before comparison, so `"$4,200.00"` and `"4200.00"`
  count as a match — note this doesn't parse numbers, so `"4200.00"` and
  `"4200.0"` still differ) — reported both per-field and as a macro average
  (`macro_field_f1`) across fields.
- **JSON validity rate** — the fraction of generations that parse as JSON
  (and, when the task declares a JSON Schema, validate against it).
- **ANLS** (Average Normalized Levenshtein Similarity) for the free-text
  document-VQA case, when gold answers aren't uniformly JSON.
- **Document-level accuracy** — the fraction of examples where every field
  matched exactly (the strict, "would a human accept this extraction
  unedited" number).
- **Baseline comparison** — the same held-out sample run through the base
  model (LoRA disabled) at no extra load cost, so `delta_field_f1` shows the
  fine-tune's actual lift over zero-shot prompting the base VLM directly.

## 6. UI

- **Dataset tab** — an image+prompt+answer add form, a ZIP-of-images bulk
  import, and a PDF-to-page-images import, replacing the generic text-pair /
  CSV / JSONL cards (which don't apply to an image dataset). Examples render
  as an image gallery (thumbnail, prompt, answer, status) rather than the
  generic input/output table, since a vision example's data lives in
  `Payload` rather than the legacy `Input`/`Output` fields — editing an
  example isn't offered here (the generic edit endpoint only updates
  `Input`/`Output`), only delete.
- **Training tab** — the standard training-run list, with a per-run field-
  accuracy table (field / exact-match / F1 / support) and a headline line
  showing macro field F1, the delta vs. the zero-shot baseline, JSON
  validity rate, ANLS, and document accuracy.
- **Test tab** — drag-and-drop (or click-to-choose) an image, type a prompt,
  and see the model's raw text output, whether it parsed as valid JSON, and
  the parsed JSON pretty-printed when it did.

## 7. Scope & known limitations

This is a from-scratch feature area (there was no "C12 blob storage"
prerequisite already in the codebase — `domain.BlobStore` and its local-disk
implementation were built as part of this work, scoped to exactly what the
plan needed: a local directory, not S3). Deliberate scope boundaries, so the
next person extending this knows what's real vs. what's a catalog listing:

- **Trainer**: only Qwen2-VL trains end-to-end. PaliGemma / Florence-2 /
  LayoutLMv3 / Donut are in the model catalog (so they appear in selection
  and cost/VRAM estimates) but `trainer/tasks/vision.py` hard-codes
  `Qwen2VLForConditionalGeneration` — training one of them fails at the
  trainer step today. Each has a different processor/chat-template contract,
  so adding one is a real, separable unit of work, not a config change.
- **Blob storage**: local disk only. `domain.BlobStore` is a plain interface
  (`Put`/`Get`/`Delete`/`DeleteTask`/`Dir`/`Stat`/`ListKeys`) precisely so an
  S3-compatible implementation can be added later without touching any
  caller — same pattern `internal/infra/modelstore` already uses for models.
- **Inference**: `/predict` is single-image, synchronous, multipart. There's
  no batch (ZIP-in/JSONL-out) vision endpoint, unlike the CSV batch endpoint
  every other kind gets — a document-extraction batch job is meaningfully
  different (it needs to shuttle images, not just text rows) and was left
  out rather than built as a thin, likely-wrong version of the real thing.
- **Export**: LoRA adapter export (`safetensors`) works the same way it does
  for every other kind. GGUF export is not offered for `vision_lm` — a
  multimodal projector's llama.cpp support is architecture-specific and was
  out of scope to verify per-model here.
- **UI**: no bounding-box overlay, no side-by-side image/JSON editor, no
  LLM pre-fill of answers, no worst-example viewer — all called out in the
  plan as stretch goals / nice-to-haves beyond the core loop (add data,
  train, see field accuracy, test a prediction), which is what's built.
- **Synthetic bootstrap**: the "generate more examples from seeds" feature
  other kinds get is not offered for `vision_lm` — the existing generator
  works on text input/output pairs and has no notion of synthesizing a
  document image, so it's hidden here rather than silently producing
  malformed examples.

## 8. Definition of done

- A receipt/invoice extraction demo shows the fine-tuned adapter beating the
  base model's zero-shot field F1 on the held-out split (`delta_field_f1` in
  the training-run metrics).
- Images are stored via the blob store and deleted once their task's
  retention window passes (`SweepExpiredBlobs`), not kept indefinitely.
- Upload limits (image bytes/pixels, ZIP size/file count, PDF size) prevent
  a single import from exhausting disk or memory.
