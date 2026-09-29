# Speech-to-Text (Whisper Fine-Tuning) — `asr` kind

Status: implemented (Wave 2 plan `07-speech-asr.md`). This document describes how the
feature works in this codebase and how to use it end to end.

## Overview

The `asr` task kind fine-tunes Whisper on domain audio so it correctly transcribes
jargon, product names, accents and noisy channels. It targets call centers, dictation,
meeting/legal transcription, field-service voice notes and low-resource languages.

Create a task with kind `asr` (via the UI's "New Task" dialog or
`POST /api/v1/tasks` with `"kind": "asr"`).

## Dataset

An `asr` example stores its data in `Payload` as `domain.ASRPayload`:

```json
{ "audio": "audio/call_0193.wav", "text": "Please reset my Acme router to factory settings.", "speaker": "spk_01", "duration": 4.2 }
```

- `audio` is the blob key inside the task's blob store (path within the imported ZIP).
- `text` is the transcript; empty transcripts are counted and must be fixed before training.
- `speaker` is an optional tag — the trainer splits train/eval **by speaker/recording**,
  not by segment, to avoid leakage.
- `duration` (seconds) is an optional hint used for stats; the trainer recomputes true
  durations during its feature pass.

### Adding clips

- **One clip:** `POST /api/v1/tasks/{taskID}/examples/asr` with
  `{"audio_base64": "...", "filename": "call.wav", "text": "...", "speaker": "spk_01"}`.
  Base64 may carry a `data:audio/*;base64,` prefix; it is stripped server-side.
- **Bulk ZIP:** `POST /api/v1/tasks/{taskID}/examples/import-asr-zip` with a ZIP
  containing audio files (wav/mp3/flac/m4a) plus a `manifest.jsonl`, one line per clip:
  `{"audio":"audio/call_0193.wav","text":"...","speaker":"...","duration":4.2}`.

### Privacy and consent

Voice data is personal/biometric in many jurisdictions. The dataset UI shows an explicit
consent note at import; audio is processed locally by default and only kept in the task's
blob store. Delete examples (or whole tasks) to enforce retention limits.

### Stats

`GET /api/v1/tasks/{taskID}/examples/asr-stats` returns clip count, total known hours,
average clip length, empty-transcript count and a duration histogram (bucketed, with a
600s+ overflow bucket). The UI renders this in the Dataset tab's "Audio dataset stats" card.

## Training

Base models: `whisper-tiny`, `whisper-base`, `whisper-small` (default, LoRA on a single
GPU), `whisper-medium`, `whisper-large-v3-turbo`. The model selector picks by language,
latency target and available hardware; Whisper entries appear in `/api/v1/models` and the
recommended model is chosen by the same selector as other kinds.

`trainer/tasks/asr.py` implements the training pipeline:

- HF `Seq2SeqTrainer` with `WhisperForConditionalGeneration` + `WhisperProcessor`;
  LoRA for medium/large, full fine-tune for tiny/base/small when VRAM allows.
- Audio is normalized to 16 kHz mono (ffmpeg) and clipped to Whisper's 30 s window;
  length-bucketed batching; optional SpecAugment.
- Evaluation uses WER (and CER for character-based languages) with a **fixed text
  normalizer** applied to both references and hypotheses.
- Split is grouped by speaker so no speaker appears in both train and eval.

Metrics reported (mapped into `TrainingMetrics`):

| Metric | Meaning |
|---|---|
| `wer` | tuned model's word error rate on the held-out split |
| `cer` | character error rate |
| `base_wer` / `base_cer` | un-fine-tuned base Whisper of the same size on the same split |
| `delta_wer` | absolute WER improvement vs. base (positive = better) |
| `domain_term_recall` | recall of a user-supplied domain glossary (`glossary` in config) |
| `rtf` | real-time factor (processing time / audio duration) on the training hardware |
| `insertions` | insertion rate (hallucination indicator on silence/noise) |

Known issues surfaced to the user: hallucination on silence/noise (VAD + no-speech
thresholds, insertion rate reported) and inconsistent transcription conventions
(numbers/casing/punctuation) which distort WER — the normalizer is fixed and shown in
the trainer logs.

## Inference

`POST /inference/{id}/transcribe` serves a deployed `asr` model:

- Multipart form: `file=@call.wav`, optional `language=en`.
- JSON body: `{"audio_base64": "...", "filename": "call.wav", "language": "en"}`.

Response:

```json
{ "kind": "asr", "result": { "text": "...", "segments": [{"start":0.0,"end":4.2,"text":"..."}], "language": "en" } }
```

Uploads are capped at 50 MB per request. Long files and batch jobs use the async job
pattern; subtitle outputs (SRT/VTT) are produced by the batch exporter when segments
are requested.

The simulation inference engine (used when no local GPU/trainer is attached) returns a
deterministic transcript for the same kind/shape, so the full API surface is testable
without hardware.

## UI

- Dataset tab: audio-player list with transcripts, speaker/duration tags and delete;
  audio stats card; single-clip add form and ZIP importer; consent note.
- Training tab: WER-first metrics row with base-vs-tuned delta, CER, insertion rate,
  glossary recall.
- Deploy & Test tab: file-picker + language field that posts to `/transcribe` and renders
  the transcript plus a per-segment timing table.

## Definition of done (for this implementation)

- [x] `KindASR` end-to-end: dataset, training job, deploy, inference
- [x] WER/CER + base-vs-tuned comparison surfaced in UI and metrics.json
- [x] Speaker-grouped split; duration stats/histogram
- [x] `/transcribe` multipart + base64 variants with upload cap
- [x] Consent note + local-only audio storage; deletion removes blobs