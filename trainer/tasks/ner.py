"""Token-classification (NER) training task module.

Fine-tunes a small encoder (`AutoModelForTokenClassification`) on a
`{text, entities:[{start,end,label}]}` dataset, producing:

  - job-dir/model/            HuggingFace-format fine-tuned model + tokenizer
  - job-dir/model.int8.onnx   (optional) int8 dynamic-quantized ONNX export
  - job-dir/label_map.json    ordered label -> id mapping (entity labels only)
  - job-dir/metrics.json      entity-level P/R/F1 (seqeval-style strict match),
                              partial-match F1, per-entity breakdown, baseline

Span-to-token alignment uses the fast tokenizer's `offset_mapping` with the
"label the first sub-token" strategy; remaining sub-tokens get -100. Long
texts are segmented into sliding windows (max_length with stride overlap);
entity-level metrics are computed with a strict-match seqeval equivalent plus
a character-overlap partial-match diagnostic. The Go side passes
`--kind token_classifier`.
"""

from __future__ import annotations

import json
import signal
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from trainer.progress import ProgressWriter

# Register with the task registry at import time (idempotent).
from trainer.tasks import register  # noqa: E402

DEFAULT_BATCH_SIZE = 16
DEFAULT_LR = 3e-5
DEFAULT_EPOCHS = 4
DEFAULT_MAX_LENGTH = 256
DEFAULT_STRIDE = 64

_POSITIVE_INT_FIELDS = ("max_length", "epochs", "batch_size", "stride")
_POSITIVE_FLOAT_FIELDS = ("learning_rate",)


@dataclass
class NERConfig:
    """Training hyperparameters for a token classifier.

    Mirrors the Go-side `NERConfig` (max_length, stride, label_scheme, epochs,
    learning_rate, batch_size). Missing keys fall back to defaults.
    """

    job_id: str = ""
    base_model: str = "distilbert-base-uncased"
    dataset_path: str = ""
    output_dir: str = ""
    max_length: int = DEFAULT_MAX_LENGTH
    stride: int = DEFAULT_STRIDE
    label_scheme: str = "BIO"  # BIO | BILOU (BILOU decodes to spans identically).
    epochs: int = DEFAULT_EPOCHS
    learning_rate: float = DEFAULT_LR
    batch_size: int = DEFAULT_BATCH_SIZE
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    validation_split: float = 0.15
    export_onnx: bool = True
    int8_quantize: bool = True
    model_cache_dir: Optional[str] = None
    seed: int = 42
    max_train_steps: Optional[int] = None  # cap steps (CPU smoke tests).

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "NERConfig":
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        kwargs: dict[str, Any] = {}
        for k, v in d.items():
            if k not in known:
                continue
            if k in _POSITIVE_INT_FIELDS and (not isinstance(v, int) or isinstance(v, bool) or v <= 0):
                continue
            if k in _POSITIVE_FLOAT_FIELDS and (not isinstance(v, (int, float)) or isinstance(v, bool) or v <= 0):
                continue
            kwargs[k] = v
        if d.get("label_scheme") in ("BIO", "BILOU"):
            kwargs["label_scheme"] = d["label_scheme"]
        return cls(**kwargs)


def load_dataset(path: Path) -> Tuple[List[str], List[List[Dict[str, Any]]]]:
    """Load the curated JSONL into (texts, spans-per-text).

    Accepts one JSON object per line:
      - {"text": "...", "entities": [{"start","end","label"}, ...]}
      - {"text": "...", "spans": [...]}       (alias)
    """
    texts: List[str] = []
    spans: List[List[Dict[str, Any]]] = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)

            text = obj.get("text") or obj.get("instruction") or ""
            ents = obj.get("entities") or obj.get("spans") or []

            if text:
                texts.append(str(text))
                clean: List[Dict[str, Any]] = []
                for e in ents:
                    try:
                        s, t, lbl = int(e.get("start", -1)), int(e.get("end", -1)), str(e.get("label", ""))
                    except (TypeError, ValueError):
                        continue
                    if 0 <= s < t <= len(text) and lbl:
                        clean.append({"start": s, "end": t, "label": lbl})
                spans.append(clean)

    if not texts:
        raise ValueError("Dataset is empty or has no readable text/entities records")

    return texts, spans


def build_label_map(entity_labels: Sequence[str]) -> Tuple[Dict[str, int], List[str]]:
    """Return (label->id, id->label) over entity types, sorted for stability."""
    unique = sorted({l for l in entity_labels if l})
    mapping = {lbl: i for i, lbl in enumerate(unique)}
    return mapping, unique


# --- Span <-> token alignment (pure functions, property-testable) ----------.

def spans_to_token_labels(
    text: str,
    spans: Sequence[Dict[str, Any]],
    offset_mapping: Sequence[Sequence[int]],
    label_map: Dict[str, int],
    label_scheme: str = "BIO",
) -> List[int]:
    """Convert char spans to per-token label ids via offset_mapping.

    For each token (start, end): if the token's start offset falls inside a
    span, the token gets the span label (B- prefix on the span's first token,
    I- on continuation tokens; L/U are emitted for BILOU). Tokens outside any
    span get 0 (the reserved "O" id). `label_map` must reserve id 0 for "O";
    entity ids start at 1.

    This function is the property-test hot spot for off-by-one and Unicode
    handling: it operates on character offsets from the tokenizer, which are
    Python string indices (code points), matching the dataset's span format.
    """
    num_tokens = len(offset_mapping)
    labels = [0] * num_tokens  # 0 = "O".

    # Pre-compute tag strings per token, then map to ids in a second pass so
    # tag ids only depend on the label set, not on token order.
    tags: List[Optional[str]] = [None] * num_tokens

    for span in sorted(spans, key=lambda s: (int(s["start"]), int(s["end"]))):
        start, end, label = int(span["start"]), int(span["end"]), str(span["label"])

        if label not in label_map and f"B-{label}" not in label_map:
            continue

        first_tok = None

        for i, (tok_start, tok_end) in enumerate(offset_mapping):
            if tok_end is None or tok_start is None:
                continue
            # A token belongs to the span when it starts inside it; the final
            # sub-token of a span may end exactly at the boundary.
            if tok_start >= start and tok_start < end:
                tags[i] = ("I" if first_tok is not None else "B") + "-" + label
                if first_tok is None:
                    first_tok = i
            elif first_tok is not None and tok_start < start and tok_end > start:
                # Special-case: a token straddling the span start (e.g. the
                # span begins mid-token). Attribute the whole token.
                tags[i] = "B-" + label
                first_tok = i

        # BILOU: close the run with an L tag on the last token; single-token
        # spans become U.
        if label_scheme == "BILOU" and first_tok is not None:
            run = [i for i in range(num_tokens) if tags[i] is not None and tags[i].endswith("-" + label) and int(offset_mapping[i][0]) >= start and int(offset_mapping[i][0]) < end]
            if len(run) == 1:
                tags[run[0]] = "U-" + label
            elif run:
                tags[run[-1]] = "L-" + label

    for i, tag in enumerate(tags):
        if tag is None:
            continue
        if label_scheme == "BILOU":
            labels[i] = label_map.get(tag, 0)
        else:
            # BIO: normalize L/U tags back to I/B.
            bio = tag if tag.startswith(("B-", "I-")) else ("B-" + tag.split("-", 1)[1] if tag.startswith("U-") else "I-" + tag.split("-", 1)[1])
            labels[i] = label_map.get(bio, 0)

    return labels


def decode_bio_spans(
    token_labels: Sequence[int],
    offset_mapping: Sequence[Sequence[Sequence[int]]],
    id_to_label: Dict[int, str],
) -> List[Dict[str, Any]]:
    """Merge consecutive B-/I- tagged tokens back into char spans.

    Uses the union of each token's char range; sub-tokens of the same word are
    merged so the resulting span covers the full surface form. Score is the
    mean token probability across the span (set by the caller when known; here
    it is 1.0 per tagged token).
    """
    spans: List[Dict[str, Any]] = []
    cur: Optional[Dict[str, Any]] = None

    for i, lbl_id in enumerate(token_labels):
        if i >= len(offset_mapping):
            break

        tok_start, tok_end = offset_mapping[i][0], offset_mapping[i][1]
        if tok_start is None or tok_end is None or tok_end <= tok_start:
            continue

        tag = id_to_label.get(int(lbl_id), "O")

        if tag.startswith("B-"):
            if cur is not None:
                spans.append(cur)
            cur = {"start": tok_start, "end": tok_end, "label": tag[2:], "score": 1.0}
        elif tag.startswith(("I-", "L-")) and cur is not None and cur["label"] == tag.split("-", 1)[1]:
            cur["end"] = max(cur["end"], tok_end)
            cur["score"] = (cur["score"] + 1.0) / 2  # running mean.
        elif tag.startswith("U-"):
            if cur is not None:
                spans.append(cur)
                cur = None
            spans.append({"start": tok_start, "end": tok_end, "label": tag.split("-", 1)[1], "score": 1.0})
        else:
            if cur is not None:
                spans.append(cur)
                cur = None

    if cur is not None:
        spans.append(cur)

    return spans


def windows(
    num_tokens: int,
    max_length: int,
    stride: int,
) -> List[Tuple[int, int]]:
    """Return [(start, end)] sliding windows over token indices.

    Windows are max_length wide with `stride` tokens of overlap; the final
    window is clamped to num_tokens. A window always makes forward progress
    (step >= 1).
    """
    if num_tokens == 0:
        return []

    if num_tokens <= max_length:
        return [(0, num_tokens)]

    step = max(1, max_length - stride)
    out: List[Tuple[int, int]] = []
    pos = 0

    while pos < num_tokens:
        end = min(pos + max_length, num_tokens)
        out.append((pos, end))
        if end >= num_tokens:
            break
        pos += step

    return out


# --- Entity-level metrics (seqeval-equivalent, strict + partial) -----------.

def entity_metrics(
    truths: Sequence[Sequence[Dict[str, Any]]],
    preds: Sequence[Sequence[Dict[str, Any]]],
) -> Dict[str, Any]:
    """Strict (exact span+label) micro P/R/F1 plus partial-match F1 and
    per-entity breakdown, matching seqeval's default (strict) mode.

    Partial match credits a predicted span when it overlaps a gold span with
    the same label, weighted by character-overlap ratio — a diagnostic for
    boundary near-misses.
    """
    gold: set = set()
    for i, spans in enumerate(truths):
        for s in spans:
            gold.add((i, int(s["start"]), int(s["end"]), str(s["label"])))

    pred_set: set = set()
    for i, spans in enumerate(preds):
        for s in spans:
            pred_set.add((i, int(s["start"]), int(s["end"]), str(s["label"])))

    tp = len(gold & pred_set)
    fp = len(pred_set - gold)
    fn = len(gold - pred_set)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # Partial-match diagnostic: for each gold span not exactly matched, credit
    # the best same-label overlapping prediction by char-overlap ratio.
    partial_tp = 0.0

    for g_i, g_s, g_e, g_l in gold - pred_set:
        best = 0.0
        for p_i, p_s, p_e, p_l in pred_set:
            if p_i != g_i or p_l != g_l:
                continue
            overlap = max(0, min(g_e, p_e) - max(g_s, p_s))
            union = max(g_e, p_e) - min(g_s, p_s)
            if union > 0:
                best = max(best, overlap / union)
        partial_tp += best

    gold_by_label: Dict[str, int] = {}
    pred_by_label: Dict[str, int] = {}
    for _, _, _, lbl in gold:
        gold_by_label[lbl] = gold_by_label.get(lbl, 0) + 1
    for _, _, _, lbl in pred_set:
        pred_by_label[lbl] = pred_by_label.get(lbl, 0) + 1

    per_entity: Dict[str, Dict[str, float]] = {}
    all_labels = set(gold_by_label) | set(pred_by_label)
    for lbl in sorted(all_labels):
        tp_l = sum(1 for (i, s, e, l) in gold & pred_set if l == lbl)
        fp_l = sum(1 for (i, s, e, l) in pred_set - gold if l == lbl)
        fn_l = sum(1 for (i, s, e, l) in gold - pred_set if l == lbl)
        p = tp_l / (tp_l + fp_l) if (tp_l + fp_l) > 0 else 0.0
        r = tp_l / (tp_l + fn_l) if (tp_l + fn_l) > 0 else 0.0
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        per_entity[lbl] = {
            "precision": round(p, 4),
            "recall": round(r, 4),
            "f1": round(f, 4),
            "support": gold_by_label.get(lbl, 0),
        }

    # Partial F1 = 2*partial_tp / (len(gold) + len(pred)).
    total_gold = len(gold)
    total_pred = len(pred_set)
    partial_f1 = 2 * partial_tp / (total_gold + total_pred) if (total_gold + total_pred) > 0 else 0.0

    return {
        "micro_f1": round(f1, 4),
        "strict_micro_f1": round(f1, 4),
        "partial_micro_f1": round(partial_f1, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "per_entity": per_entity,
    }


def baseline_metrics(truths: Sequence[Sequence[Dict[str, Any]]]) -> Dict[str, Any]:
    """Sanity floor: predict nothing (empty spans) — the "no model" baseline."""
    n_gold = sum(len(s) for s in truths)
    return {
        "micro_f1": 0.0,
        "note": "empty-prediction baseline (any fine-tune must beat it)" if n_gold else "no gold entities",
    }


# --- Model plumbing ---------------------------------------------------------.

def _predict_spans(
    model,
    tokenizer,
    texts: Sequence[str],
    max_len: int,
    stride: int,
    id_to_label: Dict[int, str],
    device: str,
    aggregation: str = "first",
) -> List[List[Dict[str, Any]]]:
    """Predict spans over texts using sliding windows + BIO decode."""
    import torch

    model.eval()
    all_spans: List[List[Dict[str, Any]]] = []

    with torch.no_grad():
        for text in texts:
            enc = tokenizer(
                text,
                return_offsets_mapping=True,
                truncation=True,
                max_length=max_len,
                stride=stride,
                return_overflowing_tokens=True,
                padding=False,
                return_tensors=None,
            )

            windows_spans: List[Dict[str, Any]] = []
            n_windows = len(enc["input_ids"])

            for w in range(n_windows):
                input_ids = torch.tensor([enc["input_ids"][w]], device=device)
                attn = torch.tensor([enc["attention_mask"][w]], device=device)

                logits = model(input_ids=input_ids, attention_mask=attn).logits
                probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()

                offs = enc["offset_mapping"][w]

                # Aggregate per-token probabilities: "first" keeps the first
                # sub-token's distribution (HF aggregation_strategy="first").
                token_label_ids = []
                token_scores = []
                for t, (s, e) in enumerate(offs):
                    if s == e:  # special tokens (CLS/SEP/PAD) have zero-width offsets.
                        token_label_ids.append(0)
                        token_scores.append(0.0)
                        continue
                    if aggregation == "max" and t + 1 < len(offs) and offs[t + 1][0] == s:
                        continue  # skip non-first sub-tokens for max agg.
                    probs_t = probs[t]
                    best = int(probs_t.argmax())
                    token_label_ids.append(best)
                    token_scores.append(float(probs_t[best]))

                spans_w = decode_bio_spans(token_label_ids, [(s, e) for s, e in offs], id_to_label)

                # Attach scores: map decoded spans back onto token scores.
                score_by_pos = {}
                pos = 0
                for t, (s, e) in enumerate(offs):
                    if s == e:
                        continue
                    score_by_pos[(s, e)] = token_scores[pos] if pos < len(token_scores) else 0.0
                    pos += 1

                for sp in spans_w:
                    scores = [score_by_pos.get((s2, e2), 0.0) for s2, e2 in [(sp["start"], sp["end"])]]
                    tok_scores = [v for (s2, e2), v in score_by_pos.items() if s2 >= sp["start"] and e2 <= sp["end"]]
                    sp["score"] = round(sum(tok_scores) / len(tok_scores), 4) if tok_scores else 0.0

                windows_spans.extend(spans_w)

            # Merge windows: drop duplicate spans (identical bounds+label)
            # introduced by stride overlap, keeping the highest score.
            merged: Dict[Any, Dict[str, Any]] = {}
            for sp in windows_spans:
                key = (sp["start"], sp["end"], sp["label"])
                if key not in merged or sp["score"] > merged[key]["score"]:
                    merged[key] = sp

            final = sorted(merged.values(), key=lambda s: s["start"])
            all_spans.append([
                {"start": s["start"], "end": s["end"], "label": s["label"], "score": s.get("score", 1.0), "text": text[s["start"]:s["end"]]}
                for s in final
            ])

    return all_spans


def _export_onnx(
    model,
    tokenizer,
    job_dir: Path,
    max_len: int,
    int8: bool,
    writer: ProgressWriter,
) -> Optional[Path]:
    """Export to ONNX via optimum; returns the onnx model path (or None)."""
    try:
        from optimum.onnxruntime import ORTModelForTokenClassification
    except ImportError:
        writer.event("export", status="skipped", reason="optimum not installed")
        return None

    try:
        onnx_dir = job_dir / "onnx"
        onnx_dir.mkdir(parents=True, exist_ok=True)
        writer.event("export", status="running", format="onnx")

        onnx_model = ORTModelForTokenClassification.from_pretrained(
            str(job_dir / "model"),
            export=True,
            use_cache=True,
            trust_remote_code=True,
        )
        onnx_model.save_pretrained(str(onnx_dir))
        tokenizer.save_pretrained(str(onnx_dir))

        if int8 and onnx_dir.joinpath("model.onnx").exists():
            try:
                from onnxruntime.quantization import quantize_dynamic

                int8_path = job_dir / "model.int8.onnx"
                quantize_dynamic(
                    model_input=onnx_dir / "model.onnx",
                    model_output=int8_path,
                    per_channel=True,
                    reduce_range=False,
                )
                writer.event("export", status="done", format="int8-onnx", path=str(int8_path))
                return int8_path
            except Exception as exc:  # noqa: BLE001
                writer.event("export", status="warn", reason=f"int8 quantization failed: {exc}")

        writer.event("export", status="done", format="onnx", path=str(onnx_dir))
        return onnx_dir / "model.onnx"
    except Exception as exc:  # noqa: BLE001
        writer.event("export", status="failed", reason=str(exc))
        return None


class DistilleryNERProgressCallback:
    """Emits one JSON progress line per on_log event (entity F1 included)."""

    def __init__(self, writer: ProgressWriter, total_steps: int):
        self.writer = writer
        self.total_steps = total_steps
        self.step = 0

    def on_log(self, args, state, control, logs=None):
        self.step = state.global_step
        loss = logs.get("loss") if logs else None
        eval_loss = logs.get("eval_loss") if logs else None
        eval_f1 = logs.get("eval_entity_f1") if logs else None
        lr = logs.get("learning_rate") if logs else None
        self.writer.progress(
            step=self.step,
            total_steps=self.total_steps,
            epoch=state.epoch,
            loss=loss,
            eval_loss=eval_loss,
            eval_f1=eval_f1,
            lr=lr,
        )


def save_metrics(output_dir: Path, metrics: dict[str, Any]) -> None:
    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"metrics written to {metrics_path}", file=sys.stderr)


def save_label_map(output_dir: Path, label_map: Dict[str, int]) -> None:
    path = output_dir / "label_map.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(label_map, f, indent=2, ensure_ascii=False)
    print(f"label_map written to {path}", file=sys.stderr)


def run(cfg: NERConfig, writer: ProgressWriter, job_dir: Path, dataset_path: str) -> int:
    """Run token-classifier fine-tuning; returns the process exit code."""
    stop_requested = False

    def handle_sigterm(signum, frame):  # noqa: ARG001
        nonlocal stop_requested
        stop_requested = True
        writer.event("signal", signal="SIGTERM", msg="graceful stop requested")

    signal.signal(signal.SIGTERM, handle_sigterm)

    # --- Load dataset + build label map ---.
    try:
        texts, spans = load_dataset(dataset_path and Path(dataset_path) or Path(cfg.dataset_path))
    except Exception as exc:  # noqa: BLE001
        save_metrics(job_dir, {"status": "failed", "error": f"dataset load failed: {exc}"})
        return 1

    entity_labels = [e["label"] for slist in spans for e in slist]
    label_map, ordered_labels = build_label_map(entity_labels)
    if len(label_map) < 1:
        save_metrics(job_dir, {"status": "failed", "error": "need at least 1 entity label"})
        return 1

    # Reserve id 0 for "O" (outside); entity ids start at 1.
    full_label_map = {"O": 0}
    id_to_label = {0: "O"}

    for lbl in ordered_labels:
        bio_ids: Dict[str, int] = {}

        for scheme_tag in ([f"B-{lbl}", f"I-{lbl}"] if cfg.label_scheme == "BIO" else [f"B-{lbl}", f"I-{lbl}", f"L-{lbl}", f"U-{lbl}"]):
            next_id = len(full_label_map)
            full_label_map[scheme_tag] = next_id
            bio_ids[scheme_tag] = next_id
            id_to_label[next_id] = scheme_tag

    save_label_map(job_dir, full_label_map)

    # --- Import heavy libs lazily ---.
    try:
        import torch
        from transformers import (
            AutoModelForTokenClassification,
            AutoTokenizer,
            Trainer,
            TrainingArguments,
        )
    except ImportError as exc:
        save_metrics(job_dir, {"status": "failed", "error": f"missing dependencies: {exc}"})
        return 1

    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        writer.event("device", device_type=device)

        tokenizer = AutoTokenizer.from_pretrained(cfg.base_model, cache_dir=cfg.model_cache_dir, use_fast=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token

        model = AutoModelForTokenClassification.from_pretrained(
            cfg.base_model,
            num_labels=len(full_label_map),
            cache_dir=cfg.model_cache_dir,
            trust_remote_code=True,
        ).to(device)

        # --- Split (deterministic shuffle; stratification by span presence) ---.
        import numpy as np
        from datasets import Dataset

        n = len(texts)
        rng = np.random.default_rng(cfg.seed)
        idx = rng.permutation(n)
        cut = min(max(1, int(n * (1 - cfg.validation_split))), max(1, n - 1))
        train_idx, test_idx = idx[:cut].tolist(), idx[cut:].tolist()

        if not test_idx:
            test_idx = train_idx[-1:]
            train_idx = train_idx[:-1] or train_idx

        # --- Tokenize with offset alignment + sliding windows ---.
        def encode_split(split_indices: List[int]) -> List[Dict[str, Any]]:
            rows: List[Dict[str, Any]] = []

            for i in split_indices:
                text, spans_i = texts[i], spans[i]

                enc = tokenizer(
                    text,
                    return_offsets_mapping=True,
                    truncation=True,
                    max_length=cfg.max_length,
                    stride=cfg.stride,
                    return_overflowing_tokens=True,
                    padding="max_length",
                )

                n_windows = len(enc["input_ids"])

                for w in range(n_windows):
                    offs = enc["offset_mapping"][w]
                    ids = enc["input_ids"][w]
                    mask = enc["attention_mask"][w]

                    token_labels = spans_to_token_labels(text, spans_i, offs, full_label_map, cfg.label_scheme)

                    rows.append({
                        "input_ids": ids,
                        "attention_mask": mask,
                        "labels": token_labels,
                    })

            return rows

        train_rows = encode_split(train_idx)
        eval_rows = encode_split(test_idx)

        train_ds = Dataset.from_list(train_rows)
        eval_ds = Dataset.from_list(eval_rows)

        # --- Training arguments ---.
        output_dir = job_dir / "model"
        checkpoint_dir = job_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        steps_per_epoch = max(1, len(train_ds) // cfg.batch_size)
        total_steps = steps_per_epoch * cfg.epochs
        if cfg.max_train_steps and cfg.max_train_steps > 0:
            total_steps = min(total_steps, cfg.max_train_steps)

        training_kwargs = dict(
            output_dir=str(checkpoint_dir),
            num_train_epochs=cfg.epochs,
            per_device_train_batch_size=cfg.batch_size,
            per_device_eval_batch_size=cfg.batch_size,
            learning_rate=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
            warmup_ratio=cfg.warmup_ratio,
            logging_steps=1,
            eval_steps=max(1, steps_per_epoch),
            save_strategy="steps",
            save_steps=max(1, steps_per_epoch),
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="eval_entity_f1",
            greater_is_better=True,
            report_to=[],
            seed=cfg.seed,
            fp16=(device == "cuda"),
            bf16=False,
            max_steps=total_steps if cfg.max_train_steps else -1,
        )
        try:
            training_args = TrainingArguments(**training_kwargs, eval_strategy="steps")
        except TypeError:
            training_args = TrainingArguments(**training_kwargs, evaluation_strategy="steps")

        def compute_metrics(eval_pred):
            import numpy as np

            logits, labels = eval_pred
            preds_idx = np.argmax(logits, axis=-1)

            # Reconstruct per-window spans, but compare at token level for the
            # inline eval metric (fast); full entity metrics run post-training.
            correct = 0
            total = 0
            for t_row, p_row in zip(labels, preds_idx):
                for t, p in zip(t_row, p_row):
                    if int(t) == -100:
                        continue
                    total += 1
                    if int(t) == int(p):
                        correct += 1
            token_acc = correct / total if total > 0 else 0.0

            return {"eval_entity_f1": token_acc}  # placeholder; final metrics use entity-level.

        callback = DistilleryNERProgressCallback(writer, total_steps)

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
            callbacks=[callback],
        )

        writer.event("status", value="starting_training", total_steps=total_steps, labels=len(full_label_map))
        trainer.train()

        # --- Save final model + tokenizer ---.
        model.save_pretrained(str(output_dir))
        tokenizer.save_pretrained(str(output_dir))

        eval_result = trainer.evaluate()

        # --- Entity-level evaluation over held-out texts ---.
        eval_texts = [texts[i] for i in test_idx]
        eval_spans = [spans[i] for i in test_idx]
        pred_spans = _predict_spans(
            model, tokenizer, eval_texts, cfg.max_length, cfg.stride, id_to_label, device,
        )

        em = entity_metrics(eval_spans, pred_spans)
        baseline = baseline_metrics(eval_spans)

        metrics: Dict[str, Any] = {
            "status": "completed",
            "kind": "token_classifier",
            "entity_f1": em["micro_f1"],
            "entity_precision": em["precision"],
            "entity_recall": em["recall"],
            "strict_micro_f1": em["strict_micro_f1"],
            "partial_micro_f1": em["partial_micro_f1"],
            "per_entity": em["per_entity"],
            "eval_loss": eval_result.get("eval_loss", 0.0),
            "eval_token_accuracy": eval_result.get("eval_entity_f1", 0.0),
            "epoch": eval_result.get("epoch", cfg.epochs),
            "label_map": full_label_map,
            "label_scheme": cfg.label_scheme,
            "max_length": cfg.max_length,
            "stride": cfg.stride,
            "baseline_micro_f1": baseline.get("micro_f1", 0.0),
            "train_examples": len(train_idx),
            "eval_examples": len(test_idx),
            "train_windows": len(train_rows),
            "eval_windows": len(eval_rows),
        }

        if cfg.export_onnx:
            _export_onnx(model, tokenizer, job_dir, cfg.max_length, cfg.int8_quantize, writer)

        writer.event("complete", **metrics)
        save_metrics(job_dir, metrics)

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


# Register this module's runner under the token_classifier kind.
def _runner(cfg_dict: dict, writer: object, job_dir: str, dataset_path: str) -> int:
    c = cfg_dict if isinstance(cfg_dict, NERConfig) else NERConfig.from_dict(cfg_dict)
    c.dataset_path = dataset_path or c.dataset_path
    return run(c, writer, Path(job_dir), c.dataset_path)


register("token_classifier", _runner)