"""Speech-to-text (Whisper fine-tuning) training task module (plan 07).

The Go contract is the same one every other task module follows: JSONL
progress in job-dir/progress.jsonl, a final metrics.json, and the trained
adapter written under output_dir. Dataset records are
{"audio": "<blob key>", "text": "...", "speaker": "...", "duration": 4.2} —
"audio" resolves to a file under `audio_dir` (the blob store's per-task
directory, already normalized to 16 kHz mono WAV by the importer), not raw
bytes, keeping the JSONL small.

Evaluation follows the plan's section 9: WER (primary) and CER, both with a
FIXED text normalizer applied to references and hypotheses alike (plan 07
risk: "inconsistent transcription conventions (numbers, casing, punctuation)
distort WER"), against a baseline = the same base Whisper with the adapter
disabled (a true zero-shot pass, cheap via peft's disable_adapter — same
trick vision.py uses). domain_term_recall scores recall of a user-supplied
glossary on the hypothesis side. RTF (real-time factor) is measured over the
held-out generation pass.

Speaker/recording-aware splitting: when records carry a "speaker" field the
held-out split is cut BY SPEAKER (grouped), so no utterance from a speaker
seen in training leaks into evaluation (plan 07 section 2, "split by
speaker/recording, not by segment, to avoid leakage").
"""

from __future__ import annotations

import json
import random
import re
import signal
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from trainer.progress import ProgressWriter

# Whisper encoder/decoder attention projection names; the q/k/v/o set covers
# Whisper's self- and cross-attention LoRA targets.
DEFAULT_TARGET_MODS = ["q_proj", "k_proj", "v_proj", "o_proj"]

# Register with the task registry at import time (idempotent).
from trainer.tasks import register  # noqa: E402  (import after module).


@dataclass
class Config:
    job_id: str
    base_model: str
    dataset_path: str
    output_dir: str
    audio_dir: str = ""
    # "transcribe" (same-language recognition, the default) or "translate"
    # (X->English, Whisper's second supervised head).
    task: str = "transcribe"
    # ISO 639-1 language tag; empty = let Whisper auto-detect per sample.
    language: str = ""
    spec_augment: bool = False
    use_lora: Optional[bool] = None  # None = auto by model size.
    epochs: int = 3
    learning_rate: float = 1e-5
    batch_size: int = 2
    gradient_accumulation_steps: int = 4
    warmup_steps: int = 20
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_mods: list[str] = field(default_factory=lambda: list(DEFAULT_TARGET_MODS))
    validation_split: float = 0.15
    max_validation_samples: int = 30
    # Glossary terms whose recall the eval reports (plan 07 section 9,
    # "domain-term accuracy (recall of a user-supplied glossary)").
    glossary: list[str] = field(default_factory=list)
    model_cache_dir: Optional[str] = None
    resume_from: Optional[str] = None
    seed: int = 42

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Config":
        known = {f.name: f for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        kwargs: dict[str, Any] = {}
        for k, v in d.items():
            if k in known:
                kwargs[k] = v
        return cls(**kwargs)


def load_config(path: Path) -> Config:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    cfg = Config.from_dict(raw)
    cfg.dataset_path = str(Path(raw.get("dataset_path", cfg.dataset_path)))
    cfg.output_dir = str(Path(raw.get("output_dir", cfg.output_dir)))
    return cfg


# --- Fixed text normalizer (plan 07: applied to references AND hypotheses)
# so punctuation/casing/number-formatting conventions don't distort WER. ---.

_NUMBER_WORDS = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
}


def normalize_text(text: str) -> str:
    """The fixed Whisper eval normalizer: lowercase, strip punctuation and
    currency/percent marks, map spelled-out digits to digits, collapse
    whitespace. The SAME function runs on references and hypotheses, so both
    sides are scored under one convention (plan 07 section 11: "enforce the
    normalizer and show it to users")."""
    s = str(text).strip().lower()

    # Standard Whisper-style punctuation stripping.
    s = re.sub(r"[^\w\s']", " ", s, flags=re.UNICODE)
    s = s.replace("'", "")

    # Spelled digits -> digits (token-wise so "one two three" -> "1 2 3").
    s = " ".join(_NUMBER_WORDS.get(tok, tok) for tok in s.split())

    return " ".join(s.split())


def _levenshtein(a: list[str], b: list[str]) -> tuple[int, int, int, int]:
    """Edit-distance alignment over word (or char) tokens, returning
    (distance, substitutions, deletions, insertions) — the WER error-type
    breakdown, via a backtracking DP. `a` = hypothesis, `b` = reference, so
    a token present in the reference but not the hypothesis is a deletion
    and vice versa an insertion."""
    n, m = len(a), len(b)
    prev = list(range(m + 1))
    back: list[list[int]] = [[0] * (m + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        cur = [i] + [0] * m
        for j in range(1, m + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            # 1=substitution(prev[j-1]), 2=deletion(prev[j]), 3=insertion(cur[j-1]).
            best, back[i][j] = min(
                (prev[j - 1] + cost, 1), (prev[j] + 1, 2), (cur[j - 1] + 1, 3),
            )
            cur[j] = best
        back[i][0] = i
        prev = cur

    distance = prev[m]

    # Backtrack to count error types.
    subs = dels = ins = 0
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and back[i][j] == 1:
            if a[i - 1] != b[j - 1]:
                subs += 1
            i, j = i - 1, j - 1
        elif i > 0 and back[i][j] == 2:
            dels += 1  # in reference, missing from hypothesis.
            i -= 1
        else:
            ins += 1
            j -= 1

    return distance, subs, dels, ins


def _error_counts(hyp: str, ref: str) -> tuple[int, int, int, int, int]:
    h, r = normalize_text(hyp), normalize_text(ref)
    dist, subs, dels, ins = _levenshtein(h.split(), r.split())
    return dist, subs, dels, ins, len(r.split())


def word_error_rate(hypotheses: list[str], references: list[str]) -> float:
    """Corpus-level WER: total edit distance over total reference words."""
    total_edits = 0
    total_words = 0

    for hyp, ref in zip(hypotheses, references):
        edits, _, _, _, words = _error_counts(hyp, ref)
        total_edits += edits
        total_words += words

    if total_words == 0:
        return 0.0

    return round(total_edits / total_words, 4)


def character_error_rate(hypotheses: list[str], references: list[str]) -> float:
    """Corpus-level CER (the primary metric for character-based scripts)."""
    total_edits = 0
    total_chars = 0

    for hyp, ref in zip(hypotheses, references):
        h, r = normalize_text(hyp).replace(" ", ""), normalize_text(ref).replace(" ", "")
        edits, _, _, _ = _levenshtein(list(h), list(r))
        total_edits += edits
        total_chars += len(r)

    if total_chars == 0:
        return 0.0

    return round(total_edits / total_chars, 4)


def domain_term_recall(hypotheses: list[str], glossary: list[str]) -> float:
    """Fraction of glossary terms that appear (normalized) in at least one
    hypothesis — the plan's domain-term accuracy, the headline "did the
    fine-tune teach the model our jargon" number."""
    if not glossary:
        return 0.0

    joined = " || ".join(normalize_text(h) for h in hypotheses)

    found = sum(1 for term in glossary if normalize_text(term) and normalize_text(term) in joined)

    return round(found / len(glossary), 4)


def wer_breakdown(hypotheses: list[str], references: list[str]) -> dict[str, Any]:
    """WER plus its error-type breakdown (substitutions/deletions/insertions
    per reference word) — the plan asks to "report insertion rates" to catch
    hallucination-on-silence, which shows up as insertions."""
    total_edits = total_subs = total_dels = total_ins = total_words = 0

    for hyp, ref in zip(hypotheses, references):
        edits, subs, dels, ins, words = _error_counts(hyp, ref)
        total_edits += edits
        total_subs += subs
        total_dels += dels
        total_ins += ins
        total_words += words

    if total_words == 0:
        return {"wer": 0.0, "substitutions": 0.0, "deletions": 0.0, "insertions": 0.0}

    return {
        "wer": round(total_edits / total_words, 4),
        "substitutions": round(total_subs / total_words, 4),
        "deletions": round(total_dels / total_words, 4),
        "insertions": round(total_ins / total_words, 4),
    }


# --- Dataset ---.


def read_asr_records(path: Path) -> list[dict[str, Any]]:
    """Reads dataset.jsonl, skipping lines that aren't a usable ASR example
    (missing audio key or empty transcript). Mirrors read_vision_records'
    "skip the bad ones, keep the rest" contract."""
    records: list[dict[str, Any]] = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
            except ValueError:
                continue

            audio = str(obj.get("audio", "")).strip()
            text = str(obj.get("text", "")).strip()

            if not audio or not text:
                continue

            records.append(
                {
                    "audio": audio,
                    "text": text,
                    "speaker": str(obj.get("speaker", "")),
                    "duration": float(obj.get("duration", 0) or 0),
                }
            )

    return records


def resolve_audio_path(audio_dir: str, audio_key: str) -> Path:
    """Resolves a dataset record's blob key to a file path under audio_dir.
    Path traversal is neutralized by taking only the basename — audio keys
    are opaque IDs generated by the blob store, never user-supplied paths
    (same contract as vision.py's resolve_image_path)."""
    return Path(audio_dir) / Path(audio_key).name


def total_hours(records: list[dict[str, Any]]) -> float:
    total_sec = sum(r["duration"] for r in records)
    return round(total_sec / 3600.0, 4)


def duration_histogram(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The dataset stats' duration histogram: buckets of [0,5), [5,10),
    [10,20), [20,30), [30,inf) seconds (Whisper clips at 30 s). The last
    bucket reports end=-1 meaning open-ended."""
    edges = [0.0, 5.0, 10.0, 20.0, 30.0, float("inf")]

    buckets = [
        {"start": edges[i], "end": edges[i + 1] if edges[i + 1] != float("inf") else -1, "count": 0}
        for i in range(len(edges) - 1)
    ]

    for r in records:
        d = r["duration"]
        for i in range(len(edges) - 1):
            if edges[i] <= d < edges[i + 1]:
                buckets[i]["count"] += 1
                break

    return buckets


def split_by_speaker(
    records: list[dict[str, Any]], validation_split: float, seed: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Held-out split by speaker/recording, not by segment (plan 07 section
    2: avoids leakage). Records without a speaker field fall back to their
    own identity (each recording is its own group, so no identical audio
    lands on both sides). Deterministic under `seed`."""
    rng = random.Random(seed)

    groups: dict[str, list[dict[str, Any]]] = {}
    for r in records:
        groups.setdefault(r["speaker"] or f"rec:{r['audio']}", []).append(r)

    keys = sorted(groups)
    rng.shuffle(keys)

    cut = max(1, int(len(keys) * (1 - validation_split))) if validation_split > 0 else len(keys)
    train_keys, eval_keys = keys[:cut], keys[cut:]

    train = [r for k in train_keys for r in groups[k]]
    eval_recs = [r for k in eval_keys for r in groups[k]]

    # Degenerate case: one speaker total — fall back to a plain record split
    # (leakage is unavoidable there; training is more important).
    if not train:
        shuffled = list(records)
        rng.shuffle(shuffled)
        cut = max(1, int(len(shuffled) * (1 - validation_split)))
        train, eval_recs = shuffled[:cut], shuffled[cut:]

    return train, eval_recs


# --- Training ---.


class DistilleryProgressCallback:
    """Emits one JSON progress line per on_log event (mirrors vision.py's)."""

    def __init__(self, writer: ProgressWriter, total_steps: int):
        self.writer = writer
        self.total_steps = total_steps

    def on_log(self, args, state, control, logs=None):  # noqa: ARG002
        loss = logs.get("loss") if logs else None
        eval_loss = logs.get("eval_loss") if logs else None
        lr = logs.get("learning_rate") if logs else None
        self.writer.progress(
            step=state.global_step,
            total_steps=self.total_steps,
            epoch=state.epoch,
            loss=loss,
            eval_loss=eval_loss,
            lr=lr,
        )


def save_metrics(output_dir: Path, metrics: dict[str, Any]) -> None:
    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"metrics written to {metrics_path}", file=sys.stderr)


def _use_lora_for(cfg: Config) -> bool:
    """LoRA policy per plan 07 section 3/5: LoRA for medium/large; tiny/base/
    small fine-tune fully when VRAM allows. use_lora overrides when set."""
    if cfg.use_lora is not None:
        return bool(cfg.use_lora)

    return "medium" in cfg.base_model.lower() or "large" in cfg.base_model.lower()


def _load_waveform(path: str):
    """Decodes an audio file to a 16 kHz mono float tensor (list of floats
    works too — the processor accepts either). Prefers torchaudio (any
    ffmpeg-decodable format), falls back to soundfile for WAV/FLAC."""
    try:
        import torchaudio

        wav, sr = torchaudio.load(path)
        wav = wav.mean(dim=0)  # mono.

        if sr != 16000:
            wav = torchaudio.functional.resample(wav, sr, 16000)

        return wav.tolist()
    except Exception:  # noqa: BLE001 - soundfile fallback for plain WAV/FLAC.
        import soundfile as sf

        wav, sr = sf.read(path, dtype="float32", always_2d=True)
        mono = wav.mean(axis=1)

        if sr != 16000:
            # Linear resample without torchaudio (only used for the fallback).
            import numpy as np

            n_out = int(len(mono) * 16000 / sr)
            x_old = np.linspace(0.0, 1.0, len(mono), endpoint=False)
            x_new = np.linspace(0.0, 1.0, n_out, endpoint=False)
            mono = np.interp(x_new, x_old, mono).astype("float32")

        return mono.tolist()


def _spec_augment(features, torch):
    """SpecAugment: two frequency masks + two time masks over the log-mel
    spectrogram (plan 07 section 5, optional SpecAugment). features is
    (batch, mel_bins, time_frames)."""
    b, freq, t_len = features.shape

    for i in range(b):
        for _ in range(2):
            f_span = int(torch.randint(0, max(2, freq // 10), (1,)).item())
            f0 = int(torch.randint(0, max(1, freq - f_span), (1,)).item())
            features[i, f0 : f0 + f_span, :] = 0

        for _ in range(2):
            t_span = int(torch.randint(0, max(2, t_len // 10), (1,)).item())
            t0 = int(torch.randint(0, max(1, t_len - t_span), (1,)).item())
            features[i, :, t0 : t0 + t_span] = 0

    return features


def _build_collate_fn(processor, spec_augment: bool, audio_dir: str):
    """Builds the collate function: log-mel features on the fly via the
    WhisperProcessor (feature extraction + tokenization in one call, with
    decoder prompt labels for the supervised transcript), and -100-masked
    padding labels so padding never contributes to the loss."""
    import torch

    def collate(batch: list[dict[str, Any]]):
        waveforms = [_load_waveform(str(resolve_audio_path(audio_dir, r["audio"]))) for r in batch]
        transcripts = [r["text"] for r in batch]

        # WhisperProcessor(audio=..., text=...) produces input_features and
        # the tokenized labels for the supervised transcript in one shot.
        inputs = processor(
            audio=waveforms,
            text=transcripts,
            sampling_rate=16000,
            return_attention_mask=False,
            padding=True,
            truncation=False,
            return_tensors="pt",
        )

        # The feature extractor's chunking can yield one feature row per
        # audio input while labels follow the text list; align them 1:1.
        if "labels" in inputs:
            labels = inputs["labels"]
            pad_id = processor.tokenizer.pad_token_id
            # Mask padding so only real transcript tokens train.
            labels[labels == pad_id] = -100
            inputs["labels"] = labels

        if spec_augment:
            inputs["input_features"] = _spec_augment(inputs["input_features"], torch)

        return inputs

    return collate


def _transcribe_records(model, processor, records: list[dict[str, Any]], cfg: Config) -> list[str]:
    """Greedy-transcribes each held-out record (audio file -> text), and
    measures wall-clock time for the RTF metric. Returns the transcripts."""
    import torch

    model.eval()
    outputs: list[str] = []
    started = time.time()

    with torch.no_grad():
        for r in records:
            path = str(resolve_audio_path(cfg.audio_dir, r["audio"]))
            waveform = _load_waveform(path)

            gen_kwargs: dict[str, Any] = {"max_new_tokens": 256, "do_sample": False}
            if cfg.language:
                gen_kwargs["language"] = cfg.language
            if cfg.task == "translate":
                gen_kwargs["task"] = "translate"

            inputs = processor(audio=[waveform], sampling_rate=16000, return_tensors="pt")
            input_features = inputs["input_features"].to(model.device)

            generated = model.generate(input_features, **gen_kwargs)
            outputs.append(processor.batch_decode(generated, skip_special_tokens=True)[0].strip())

    elapsed = time.time() - started
    audio_sec = sum(r["duration"] for r in records) or 1e-6
    _transcribe_records.last_rtf = elapsed / audio_sec

    return outputs


def run(cfg: Config, writer: ProgressWriter, job_dir: Path, dataset_path: str) -> int:
    """Runs the Whisper fine-tune and returns the process exit code."""
    stop_requested = False

    def handle_sigterm(signum, frame):  # noqa: ARG001
        nonlocal stop_requested
        stop_requested = True
        writer.event("signal", signal="SIGTERM", msg="graceful stop requested")

    signal.signal(signal.SIGTERM, handle_sigterm)

    if not cfg.audio_dir:
        writer.event("error", message="audio_dir is required for asr training")
        save_metrics(job_dir, {"status": "failed", "error": "audio_dir is required"})
        return 1

    # --- Import heavy libs lazily so --help / config errors are cheap. ---
    import torch
    from transformers import (
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
        WhisperForConditionalGeneration,
        WhisperProcessor,
    )

    device_map = {"": "cuda"} if torch.cuda.is_available() else {"": "cpu"}

    if torch.cuda.is_available():
        writer.event("device", device_type="cuda", name=torch.cuda.get_device_name(0))
    else:
        writer.event("device", device_type="cpu")

    try:
        processor = WhisperProcessor.from_pretrained(cfg.base_model, cache_dir=cfg.model_cache_dir)
        model = WhisperForConditionalGeneration.from_pretrained(
            cfg.base_model,
            cache_dir=cfg.model_cache_dir,
            device_map=device_map,
        )

        # Plan 07: force the model to always (only) produce the requested
        # task/language tokens during generation.
        if cfg.task == "translate":
            model.generation_config.task = "translate"
        if cfg.language:
            model.generation_config.language = cfg.language
            model.generation_config.forced_decoder_ids = None

        use_lora = _use_lora_for(cfg)

        if use_lora:
            from peft import LoraConfig, get_peft_model

            lora_config = LoraConfig(
                r=cfg.lora_rank,
                lora_alpha=cfg.lora_alpha,
                lora_dropout=cfg.lora_dropout,
                target_modules=cfg.lora_target_mods or DEFAULT_TARGET_MODS,
                task_type="SEQ_2_SEQ_LM",
            )
            model = get_peft_model(model, lora_config)
            model.print_trainable_parameters()

        records = read_asr_records(Path(dataset_path))
        if not records:
            raise ValueError("Dataset is empty or has no readable ASR records")

        # Resolve blob keys -> absolute paths are handled in the collate fn
        # via resolve_audio_path; here only the split happens (by speaker).
        train_records, eval_records = split_by_speaker(records, cfg.validation_split, cfg.seed)

        collate_fn = _build_collate_fn(processor, cfg.spec_augment, cfg.audio_dir)

        class _ListDataset(torch.utils.data.Dataset):
            def __init__(self, items: list[dict[str, Any]]):
                self.items = items

            def __len__(self) -> int:
                return len(self.items)

            def __getitem__(self, idx: int) -> dict[str, Any]:
                return self.items[idx]

        train_ds = _ListDataset(train_records)

        output_dir = Path(cfg.output_dir)
        checkpoint_dir = output_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        training_args = Seq2SeqTrainingArguments(
            output_dir=str(checkpoint_dir),
            num_train_epochs=cfg.epochs,
            per_device_train_batch_size=cfg.batch_size,
            gradient_accumulation_steps=cfg.gradient_accumulation_steps,
            learning_rate=cfg.learning_rate,
            warmup_steps=cfg.warmup_steps,
            logging_steps=1,
            save_strategy="epoch",
            save_total_limit=2,
            report_to=[],
            seed=cfg.seed,
            fp16=torch.cuda.is_available(),
            remove_unused_columns=False,
            predict_with_generate=True,
        )

        total_steps = max(1, len(train_ds) // (cfg.batch_size * cfg.gradient_accumulation_steps)) * cfg.epochs
        callback = DistilleryProgressCallback(writer, total_steps)

        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            data_collator=collate_fn,
            callbacks=[callback],
        )

        writer.event("status", value="starting_training", total_steps=total_steps)

        trainer.train(resume_from_checkpoint=cfg.resume_from)

        adapter_path = output_dir / "adapter"
        model.save_pretrained(str(adapter_path))
        processor.save_pretrained(str(output_dir / "processor"))

        final_metrics: dict[str, Any] = {
            "status": "completed",
            "kind": "asr",
            "train_examples": len(train_records),
            "total_hours": round(total_hours(records), 4),
            "duration_histogram": duration_histogram(records),
        }

        # Held-out transcription + the plan-9 metric suite (WER primary, CER,
        # error-type breakdown, domain-term recall, RTF).
        if eval_records:
            sample = eval_records[: cfg.max_validation_samples]
            gold = [r["text"] for r in sample]

            predicted = _transcribe_records(model, processor, sample, cfg)
            breakdown = wer_breakdown(predicted, gold)

            final_metrics.update(breakdown)
            final_metrics["cer"] = character_error_rate(predicted, gold)
            final_metrics["domain_term_recall"] = domain_term_recall(predicted, cfg.glossary)
            final_metrics["rtf"] = round(getattr(_transcribe_records, "last_rtf", 0.0), 4)

            # Baseline: the plan's definition of done is "WER improves over
            # the base model on a held-out set", so re-run transcription with
            # the LoRA adapter disabled — the same base Whisper, at no extra
            # load cost since it's the same model object already in memory.
            if use_lora and hasattr(model, "disable_adapter"):
                with model.disable_adapter():
                    baseline = _transcribe_records(model, processor, sample, cfg)
                final_metrics["base_wer"] = word_error_rate(baseline, gold)
                final_metrics["base_cer"] = character_error_rate(baseline, gold)

                if "wer" in final_metrics:
                    final_metrics["delta_wer"] = round(final_metrics["base_wer"] - final_metrics["wer"], 4)

            writer.event("eval", **{k: v for k, v in final_metrics.items() if k not in ("status", "kind")})

        writer.event("complete", status="completed", kind="asr")
        save_metrics(job_dir, final_metrics)

        if stop_requested:
            writer.event("status", value="stopped_by_user")
            return 130

        return 0

    except Exception as exc:  # noqa: BLE001
        writer.event("error", message=str(exc), class_name=type(exc).__name__)
        save_metrics(job_dir, {"status": "failed", "error": str(exc)})
        print(f"FATAL: {exc}", file=sys.stderr)
        return 1
    finally:
        writer.close()


def _runner(cfg_dict: dict, writer: object, job_dir: str, dataset_path: str) -> int:
    c = cfg_dict if isinstance(cfg_dict, Config) else Config.from_dict(cfg_dict)
    c.dataset_path = dataset_path or c.dataset_path

    return run(c, writer, Path(job_dir), c.dataset_path)


register("asr", _runner)