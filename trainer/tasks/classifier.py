"""Sequence-classification training task module.

Fine-tunes a small encoder model (`AutoModelForSequenceClassification`) on a
`{text, label}` dataset, producing:

  - job-dir/model/            HuggingFace-format fine-tuned model + tokenizer
  - job-dir/model.int8.onnx   (optional) int8 dynamic-quantized ONNX export
  - job-dir/label_map.json    ordered label -> id mapping
  - job-dir/metrics.json      accuracy, macro-F1, weighted-F1, per-class
                              P/R/F1, confusion matrix, threshold sweep,
                              and baseline (zero-shot head / majority class)

Follows the same contract as `sft.py`: JSONL progress in job-dir/progress.jsonl
and a final metrics.json. The Go side passes `--kind seq_classifier`.
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
DEFAULT_MAX_LENGTH = 512

# Hyperparameters that must stay positive. The Go config writer copies
# "ClassifierConfig" zero values verbatim, so from_dict drops non-positive
# values and the defaults apply.
_POSITIVE_INT_FIELDS = ("max_length", "epochs", "batch_size")
_POSITIVE_FLOAT_FIELDS = ("learning_rate", "threshold")


@dataclass
class ClassifierConfig:
    """Training hyperparameters for a sequence classifier.

    Mirrors the Go-side `ClassifierConfig` (max_length, multi_label, epochs,
    learning_rate, batch_size). Missing keys fall back to defaults.
    """

    job_id: str = ""
    base_model: str = "distilbert-base-uncased"
    dataset_path: str = ""
    output_dir: str = ""
    max_length: int = DEFAULT_MAX_LENGTH
    multi_label: bool = False
    epochs: int = DEFAULT_EPOCHS
    learning_rate: float = DEFAULT_LR
    batch_size: int = DEFAULT_BATCH_SIZE
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    class_weights: bool = True  # enable weighted loss when imbalance > 3:1.
    validation_split: float = 0.15
    threshold: float = 0.5  # default confidence threshold.
    export_onnx: bool = True
    int8_quantize: bool = True
    model_cache_dir: Optional[str] = None
    seed: int = 42
    max_train_steps: Optional[int] = None  # cap steps (CPU smoke tests).

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ClassifierConfig":
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
        return cls(**kwargs)


def load_config(path: Path) -> ClassifierConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    cfg = ClassifierConfig.from_dict(raw)
    cfg.dataset_path = str(Path(raw.get("dataset_path", cfg.dataset_path)))
    cfg.output_dir = str(Path(raw.get("output_dir", cfg.output_dir)))
    return cfg


def load_dataset(path: Path) -> Tuple[List[str], List[str]]:
    """Load the curated JSONL into (texts, labels).

    Accepts one JSON object per line:
      - {"text": "...", "label": "billing"}
      - {"instruction": "...", "output": "billing"}  (legacy causal_lm format)
      - {"text": "...", "labels": [...]}              (multi-label phase 2)
    """
    texts: List[str] = []
    labels: List[str] = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)

            text = obj.get("text") or obj.get("instruction") or ""
            label = obj.get("label") or obj.get("output") or ""
            labels_arr = obj.get("labels") or []

            if (label or labels_arr) and text:
                texts.append(str(text).strip())
                if labels_arr:
                    labels.append("\x1f".join(str(l) for l in labels_arr))  # multi-label join marker.
                else:
                    labels.append(str(label).strip())

    if not texts:
        raise ValueError("Dataset is empty or has no readable text/label records")

    return texts, labels


def build_label_map(labels: Sequence[str]) -> Tuple[Dict[str, int], List[str]]:
    """Return (label->id, id->label), sorting for stable ordering."""
    unique = sorted({l for l in labels if l})
    mapping = {lbl: i for i, lbl in enumerate(unique)}
    return mapping, unique


def compute_class_weights(labels: Sequence[str], label_map: Dict[str, int]) -> Optional[List[float]]:
    """Return per-class weights for the loss when the imbalance ratio > 3:1.

    Uses the standard inverse-frequency formula w_i = N / (n_i * C) so
    weights average ~1.0 and the weighted loss keeps its scale.
    Returns None when the ratio is below the threshold (uniform weighting).
    """
    counts = [0] * len(label_map)
    for lbl in labels:
        if lbl in label_map:
            counts[label_map[lbl]] += 1

    total = sum(counts)
    if total == 0:
        return None

    max_c = max(counts)
    min_c = min(c for c in counts if c > 0)
    if max_c == 0 or min_c == 0 or max_c / min_c <= 3.0:
        return None  # balanced enough — no weighting.

    n_classes = len(counts)
    return [total / (c * n_classes) if c > 0 else 0.0 for c in counts]


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


# --- Metrics / evaluation helpers (pure functions, unit-testable) ----------.

def confusion_matrix(
    truths: Sequence[str],
    preds: Sequence[str],
    label_map: Dict[str, int],
) -> List[List[int]]:
    n = len(label_map)
    matrix = [[0] * n for _ in range(n)]
    id_of = {lbl: i for i, lbl in enumerate(sorted(label_map, key=label_map.get))}
    for t, p in zip(truths, preds):
        ti = id_of.get(t)
        pi = id_of.get(p)
        if ti is None or pi is None:
            continue
        matrix[ti][pi] += 1
    return matrix


def per_class_metrics(
    truths: Sequence[str],
    preds: Sequence[str],
    label_map: Dict[str, int],
) -> Dict[str, Dict[str, float]]:
    """Per-class precision/recall/F1 keyed by label."""
    out: Dict[str, Dict[str, float]] = {}
    for lbl in sorted(label_map, key=label_map.get):
        tp = fp = fn = 0
        for t, p in zip(truths, preds):
            if p == lbl and t == lbl:
                tp += 1
            elif p == lbl and t != lbl:
                fp += 1
            elif t == lbl and p != lbl:
                fn += 1
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        out[lbl] = {"precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4), "support": sum(1 for t in truths if t == lbl)}
    return out


def macro_f1(per_class: Dict[str, Dict[str, float]]) -> float:
    if not per_class:
        return 0.0
    return round(sum(p["f1"] for p in per_class.values()) / len(per_class), 4)


def weighted_f1(truths: Sequence[str], preds: Sequence[str]) -> float:
    """Weighted macro-F1 weighted by label support."""
    from collections import Counter

    support = Counter(truths)
    labels = set(truths) | set(preds)
    total = len(truths) or 1
    weighted = 0.0
    for lbl in labels:
        tp = fp = fn = 0
        for t, p in zip(truths, preds):
            if p == lbl and t == lbl:
                tp += 1
            elif p == lbl and t != lbl:
                fp += 1
            elif t == lbl and p != lbl:
                fn += 1
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        weighted += f1 * support.get(lbl, 0)
    return round(weighted / total, 4)


def accuracy(truths: Sequence[str], preds: Sequence[str]) -> float:
    if not truths:
        return 0.0
    return round(sum(1 for t, p in zip(truths, preds) if t == p) / len(truths), 4)


def majority_class_f1(truths: Sequence[str], labels: Sequence[str], label_map: Dict[str, int]) -> float:
    """Baseline sanity floor: predict the majority class."""
    from collections import Counter

    if not truths:
        return 0.0
    majority = Counter(truths).most_common(1)[0][0]
    preds = [majority] * len(truths)
    per = per_class_metrics(truths, preds, label_map)
    return macro_f1(per)


def threshold_sweep(
    truths: Sequence[str],
    scores: Sequence[Dict[str, float]],
    label_map: Dict[str, int],
) -> List[Dict[str, Any]]:
    """For each threshold in 0.05..0.95, report macro-F1 and coverage.

    coverage = fraction of examples whose top score >= threshold (i.e. the
    model is confident enough to answer rather than abstain).
    """
    if not truths or not scores:
        return []
    out: List[Dict[str, Any]] = []
    for t in range(5, 100, 5):
        thr = t / 100.0
        covered_truths: List[str] = []
        covered_preds: List[str] = []
        for truth, sc in zip(truths, scores):
            if not sc:
                continue
            best_lbl = max(sc, key=sc.get)
            if sc[best_lbl] >= thr:
                covered_truths.append(truth)
                covered_preds.append(best_lbl)
        # Abstained examples are excluded from the F1 numerator AND
        # denominator; crediting them with the true label would inflate
        # macro-F1 at high thresholds.
        per = per_class_metrics(covered_truths, covered_preds, label_map)
        out.append({
            "threshold": thr,
            "macro_f1": macro_f1(per),
            "coverage": round(len(covered_preds) / len(truths), 4),
        })
    return out


# --- Training / inference --------------------------------------------------.

def _predict(model, tokenizer, texts, max_len, label_map, device):
    """Run inference over a list of texts; returns (label, score, scores_dict)."""
    import torch

    model.eval()
    results = []
    with torch.no_grad():
        for text in texts:
            enc = tokenizer(
                text,
                truncation=True,
                max_length=max_len,
                padding="max_length",
                return_tensors="pt",
            ).to(device)
            logits = model(**enc).logits
            probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()
            id_to_label = {v: k for k, v in label_map.items()}
            scores = {id_to_label[i]: float(probs[i]) for i in range(len(probs))}
            best_label = max(scores, key=scores.get)
            best_score = scores[best_label]
            results.append((best_label, best_score, scores))
    return results


def _export_onnx(
    model,
    tokenizer,
    output_dir: Path,
    max_len: int,
    int8: bool,
    writer: ProgressWriter,
    device: str,
) -> Optional[Path]:
    """Export to ONNX via optimum; returns the onnx model path (or None)."""
    try:
        from optimum.onnxruntime import ORTModelForSequenceClassification
    except ImportError:
        writer.event("export", status="skipped", reason="optimum not installed")
        return None

    try:
        onnx_dir = output_dir / "onnx"
        onnx_dir.mkdir(parents=True, exist_ok=True)
        writer.event("export", status="running", format="onnx")

        onnx_model = ORTModelForSequenceClassification.from_pretrained(
            output_dir / "model",
            export=True,
            use_cache=True,
            trust_remote_code=True,
        )
        onnx_model.save_pretrained(str(onnx_dir))
        tokenizer.save_pretrained(str(onnx_dir))

        if int8 and onnx_dir.joinpath("model.onnx").exists():
            try:
                # quantize_dynamic ships with onnxruntime, not optimum.
                from onnxruntime.quantization import quantize_dynamic

                int8_path = output_dir / "model.int8.onnx"
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


class DistilleryProgressCallback:
    """Emits one JSON progress line per on_log event (macro-F1 included)."""

    def __init__(self, writer: ProgressWriter, total_steps: int):
        self.writer = writer
        self.total_steps = total_steps
        self.step = 0
        self.epoch = 0.0

    def on_log(self, args, state, control, logs=None):
        self.step = state.global_step
        self.epoch = state.epoch
        loss = logs.get("loss") if logs else None
        eval_loss = logs.get("eval_loss") if logs else None
        eval_f1 = logs.get("eval_macro_f1") if logs else None
        lr = logs.get("learning_rate") if logs else None
        self.writer.progress(
            step=self.step,
            total_steps=self.total_steps,
            epoch=self.epoch,
            loss=loss,
            eval_loss=eval_loss,
            eval_f1=eval_f1,
            lr=lr,
        )

    def on_step_end(self, args, state, control):
        if state.global_step % 10 == 0:
            self.writer.progress(
                step=state.global_step,
                total_steps=self.total_steps,
                epoch=state.epoch,
                loss=state.log_history[-1].get("loss") if state.log_history else None,
            )


def _compute_baseline(labels_text: Sequence[str], label_map: Dict[str, int]) -> Dict[str, Any]:
    """Sanity floor metrics: zero-shot head is unavailable without a GPU, so
    report the majority-class baseline (which any fine-tune must beat)."""
    from collections import Counter

    if not labels_text:
        return {}
    majority = Counter(labels_text).most_common(1)[0][0]
    n = len(labels_text)
    acc = round(sum(1 for l in labels_text if l == majority) / n, 4)
    per = per_class_metrics(labels_text, [majority] * n, label_map)
    return {
        "accuracy": acc,
        "macro_f1": macro_f1(per),
        "weighted_f1": weighted_f1(labels_text, [majority] * n),
        "note": "majority-class baseline (sanity floor)",
    }


def run(cfg: ClassifierConfig, writer: ProgressWriter, job_dir: Path, dataset_path: str) -> int:
    """Run classifier fine-tuning; returns the process exit code."""
    # --- Graceful SIGTERM handling ---.
    stop_requested = False

    def handle_sigterm(signum, frame):  # noqa: ARG001
        nonlocal stop_requested
        stop_requested = True
        writer.event("signal", signal="SIGTERM", msg="graceful stop requested")

    signal.signal(signal.SIGTERM, handle_sigterm)

    # --- Load dataset + build label map ---.
    try:
        texts, labels = load_dataset(dataset_path and Path(dataset_path) or Path(cfg.dataset_path))
    except Exception as exc:  # noqa: BLE001
        save_metrics(job_dir, {"status": "failed", "error": f"dataset load failed: {exc}"})
        return 1

    label_map, ordered_labels = build_label_map(labels)
    if len(label_map) < 2:
        save_metrics(job_dir, {"status": "failed", "error": "need at least 2 distinct labels"})
        return 1

    save_label_map(job_dir, label_map)

    # --- Import heavy libs lazily ---.
    try:
        import torch
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            Trainer,
            TrainingArguments,
        )
    except ImportError as exc:
        save_metrics(job_dir, {"status": "failed", "error": f"missing dependencies: {exc}"})
        return 1

    device_map = "auto"
    if torch.cuda.is_available():
        writer.event("device", device_type="cuda", name=torch.cuda.get_device_name(0))
        use_fp16 = True
    else:
        writer.event("device", device_type="cpu")
        device_map = "cpu"
        use_fp16 = False

    try:
        tokenizer = AutoTokenizer.from_pretrained(cfg.base_model, cache_dir=cfg.model_cache_dir, use_fast=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token

        model = AutoModelForSequenceClassification.from_pretrained(
            cfg.base_model,
            num_labels=len(label_map),
            cache_dir=cfg.model_cache_dir,
            device_map=device_map,
            trust_remote_code=True,
        )

        # --- Stratified split + tokenize ---.
        # Dedup is handled upstream by Go; here we just split stratified.
        from collections import Counter

        # Build a balanced-enough split with stratification on the label.
        from sklearn.model_selection import StratifiedShuffleSplit  # type: ignore[import]
        from datasets import Dataset

        n = len(texts)
        idx = list(range(n))
        labels_flat = [l.split("\x1f")[0] for l in labels]  # first label for stratification (single-label).

        try:
            split = StratifiedShuffleSplit(n_splits=1, test_size=cfg.validation_split, random_state=cfg.seed)
            train_idx, test_idx = next(split.split(idx, labels_flat))
        except ValueError:
            # Some class has < 2 members: stratification is impossible.
            import numpy as np

            rng = np.random.default_rng(cfg.seed)
            shuffled = rng.permutation(n)
            cut = min(max(1, int(n * (1 - cfg.validation_split))), n - 1)
            train_idx, test_idx = shuffled[:cut], shuffled[cut:]

        train_ds = Dataset.from_dict({"text": [texts[i] for i in train_idx], "label": [labels[i] for i in train_idx]})
        eval_ds = Dataset.from_dict({"text": [texts[i] for i in test_idx], "label": [labels[i] for i in test_idx]})

        def tok(examples):
            enc = tokenizer(examples["text"], truncation=True, max_length=cfg.max_length, padding="max_length")
            enc["labels"] = [label_map.get(lbl, -100) for lbl in examples["label"]]
            return enc

        train_ds = train_ds.map(tok, batched=True)
        eval_ds = eval_ds.map(tok, batched=True)

        # --- Class weights (weighted loss) ---.
        class_weights = None
        if cfg.class_weights:
            class_weights = compute_class_weights([l.split("\x1f")[0] for l in labels], label_map)
            if class_weights is not None:
                writer.event("status", value="class_weights_enabled", weights=class_weights)

        if class_weights is not None:
            import torch as _t
            model.config.problem_type = "single_label_classification" if not cfg.multi_label else "multi_label_classification"
            # Attach weights to the loss by wrapping compute_loss via a simple
            # Trainer subclass. (Kept minimal to avoid HF API churn.)
            class WeightedTrainer(Trainer):
                def compute_loss(self, model, inputs, return_outputs=False):
                    labels = inputs.pop("labels")
                    outputs = model(**inputs)
                    logits = outputs.logits
                    loss_fct = _t.nn.CrossEntropyLoss(weight=_t.tensor(class_weights, device=logits.device))
                    loss = loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))
                    return (loss, outputs) if return_outputs else loss

            TrainerCls = WeightedTrainer
        else:
            TrainerCls = Trainer

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
            metric_for_best_model="eval_macro_f1",
            greater_is_better=True,
            report_to=[],
            seed=cfg.seed,
            fp16=use_fp16,
            bf16=False,
            max_steps=total_steps if cfg.max_train_steps else -1,
        )
        try:
            training_args = TrainingArguments(**training_kwargs, eval_strategy="steps")
        except TypeError:
            # transformers < 4.41 names this argument evaluation_strategy.
            training_args = TrainingArguments(**training_kwargs, evaluation_strategy="steps")

        def compute_metrics(eval_pred):
            import numpy as np

            logits, labels = eval_pred
            preds_idx = np.argmax(logits, axis=-1)
            truths: List[str] = []
            pred_labels: List[str] = []
            # Filter truth/pred PAIRS to keep the lists aligned, and index
            # the ordered label LIST directly (list.get does not exist).
            for t, p in zip(labels, preds_idx):
                ti, pi = int(t), int(p)
                if ti < 0 or pi < 0 or ti >= len(ordered_labels) or pi >= len(ordered_labels):
                    continue
                truths.append(ordered_labels[ti])
                pred_labels.append(ordered_labels[pi])
            per = per_class_metrics(truths, pred_labels, label_map)
            return {
                "eval_accuracy": accuracy(truths, pred_labels),
                "eval_macro_f1": macro_f1(per),
            }

        callback = DistilleryProgressCallback(writer, total_steps)

        trainer = TrainerCls(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
            callbacks=[callback],
        )

        writer.event("status", value="starting_training", total_steps=total_steps, labels=len(label_map))
        trainer.train()

        # --- Save final model + tokenizer ---.
        model.save_pretrained(str(output_dir))
        tokenizer.save_pretrained(str(output_dir))

        # --- Evaluate on held-out split for final metrics ---.
        eval_result = trainer.evaluate()
        # Recompute full per-class / confusion matrix / threshold sweep over eval set.
        device = "cuda" if torch.cuda.is_available() else "cpu"
        eval_texts = [texts[i] for i in test_idx]
        eval_labels = [labels[i].split("\x1f")[0] for i in test_idx]  # first label.
        predictions = _predict(model, tokenizer, eval_texts, cfg.max_length, label_map, device)
        pred_labels = [p[0] for p in predictions]
        scores = [p[2] for p in predictions]

        per_class = per_class_metrics(eval_labels, pred_labels, label_map)
        macro = macro_f1(per_class)
        wf1 = weighted_f1(eval_labels, pred_labels)
        acc = accuracy(eval_labels, pred_labels)
        cmat = confusion_matrix(eval_labels, pred_labels, label_map)
        sweep = threshold_sweep(eval_labels, scores, label_map)
        baseline = _compute_baseline(eval_labels, label_map)

        metrics: Dict[str, Any] = {
            "status": "completed",
            "kind": "seq_classifier",
            "accuracy": acc,
            "macro_f1": macro,
            "weighted_f1": wf1,
            "eval_loss": eval_result.get("eval_loss", 0.0),
            "epoch": eval_result.get("epoch", cfg.epochs),
            "per_class": per_class,
            "confusion_matrix": cmat,
            "label_map": label_map,
            "threshold_sweep": sweep,
            "baseline": baseline,
            "baseline_macro_f1": baseline.get("macro_f1", 0.0),
            "delta_macro_f1": round(macro - baseline.get("macro_f1", 0.0), 4),
            "default_threshold": cfg.threshold,
            "max_length": cfg.max_length,
            "multi_label": cfg.multi_label,
            "train_examples": len(train_idx),
            "eval_examples": len(test_idx),
        }

        # --- ONNX export (optional) ---.
        if cfg.export_onnx:
            _export_onnx(model, tokenizer, job_dir, cfg.max_length, cfg.int8_quantize, writer, device)

        # metrics already carries status/kind; re-passing either as an
        # explicit kwarg raises TypeError (multiple values for keyword).
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


# Register this module's runner under the seq_classifier kind.
def _runner(cfg_dict: dict, writer: object, job_dir: str, dataset_path: str) -> int:
    c = cfg_dict if isinstance(cfg_dict, ClassifierConfig) else ClassifierConfig.from_dict(cfg_dict)
    c.dataset_path = dataset_path or c.dataset_path
    return run(c, writer, Path(job_dir), c.dataset_path)


register("seq_classifier", _runner)