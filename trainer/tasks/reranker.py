"""Reranker (cross-encoder) training task module.

Fine-tunes a cross-encoder on graded (query, document, relevance) pairs using
`sentence-transformers.CrossEncoder` with BCE loss. Produces:

  - job-dir/model/                  sentence-transformers CrossEncoder folder
  - job-dir/model.onnx/             (optional) ONNX export
  - job-dir/metrics.json            nDCG@10 / MRR@10 / recall@k (tuned vs base)

Dataset shapes accepted (JSONL):
  {"query","document","label": 0|1}           binary relevance
  {"query","document","label": 0.0-1.0}       graded relevance

Follows the same contract as `classifier.py`: JSONL progress in
job-dir/progress.jsonl and a final metrics.json. The Go side passes
`--kind reranker`.
"""

from __future__ import annotations

import json
import math
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
DEFAULT_EPOCHS = 3
DEFAULT_MAX_SEQ_LEN = 512

_POSITIVE_INT_FIELDS = ("max_seq_len", "epochs", "batch_size")
_POSITIVE_FLOAT_FIELDS = ("learning_rate",)


@dataclass
class RerankerConfig:
    """Training hyperparameters for a reranker (cross-encoder).

    Mirrors the Go-side `RerankerConfig` (max_seq_len, loss, epochs,
    learning_rate, batch_size). Missing keys fall back to defaults.
    """

    job_id: str = ""
    base_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    dataset_path: str = ""
    output_dir: str = ""
    max_seq_len: int = DEFAULT_MAX_SEQ_LEN
    loss: str = "bce"  # "bce" | "margin".
    epochs: int = DEFAULT_EPOCHS
    learning_rate: float = DEFAULT_LR
    batch_size: int = DEFAULT_BATCH_SIZE
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    validation_split: float = 0.15
    export_onnx: bool = True
    model_cache_dir: Optional[str] = None
    seed: int = 42
    max_train_steps: Optional[int] = None  # cap steps (CPU smoke tests).

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "RerankerConfig":
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


def load_config(path: Path) -> RerankerConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    cfg = RerankerConfig.from_dict(raw)
    cfg.dataset_path = str(Path(raw.get("dataset_path", cfg.dataset_path)))
    cfg.output_dir = str(Path(raw.get("output_dir", cfg.output_dir)))
    return cfg


# --- Dataset loading -------------------------------------------------------.

def load_dataset(path: Path) -> Tuple[List[Tuple[str, str, float]], List[Tuple[str, str, float]]]:
    """Load the curated JSONL into (queries, documents, labels).

    Accepts one JSON object per line:
      {"query": "...", "document": "...", "label": 0|1}        binary
      {"query": "...", "document": "...", "label": 0.75}       graded
      {"query": "...", "document": "...", "relevance": 0.75}   graded (alt key)

    Returns (all_pairs, graded_pairs) where graded_pairs is a subset of
    all_pairs when >2 distinct grades are present (used for nDCG).
    """
    all_pairs: List[Tuple[str, str, float]] = []
    graded_pairs: List[Tuple[str, str, float]] = []

    def parse_label(obj: dict) -> Optional[float]:
        if "label" in obj:
            label = obj["label"]
        elif "relevance" in obj:
            label = obj["relevance"]
        else:
            return None
        if isinstance(label, bool):
            return 1.0 if label else 0.0
        return float(label)

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)

            query = str(obj.get("query") or "").strip()
            document = str(obj.get("document") or "").strip()
            instruction = obj.get("instruction") or ""
            output = obj.get("output") or ""

            # Migrate legacy causal_lm rows to pairs when both fields exist.
            if not query and instruction:
                query = str(instruction).strip()
            if not document and output:
                document = str(output).strip()

            if not query or not document:
                continue

            label = parse_label(obj)
            if label is None:
                # Default to positive when no label present.
                label = 1.0

            all_pairs.append((query, document, label))
            if label != 0 and label != 1:
                graded_pairs.append((query, document, label))

    if not all_pairs:
        raise ValueError("Dataset is empty or has no readable query/document records")

    return all_pairs, graded_pairs


def save_metrics(output_dir: Path, metrics: dict[str, Any]) -> None:
    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"metrics written to {metrics_path}", file=sys.stderr)


# --- Retrieval metrics (reuse embedding helpers where sensible) -------------.

def _rank_list(scores: Sequence[float], k: int) -> List[Tuple[int, float]]:
    """Return [(index, score)] sorted by score desc, truncated to k."""
    ranked = sorted(enumerate(scores), key=lambda x: -x[1])
    return ranked[:k]


def recall_at_k(relevant_idx: Sequence[int], ranked: Sequence[Tuple[int, float]], k: int) -> float:
    if not relevant_idx:
        return 0.0
    top_k = {idx for idx, _ in ranked[:k]}
    hit = sum(1 for r in relevant_idx if r in top_k)
    return hit / len(relevant_idx)


def mrr_at_k(relevant_idx: Sequence[int], ranked: Sequence[Tuple[int, float]], k: int = 10) -> float:
    for rank, (idx, _) in enumerate(ranked[:k], start=1):
        if idx in relevant_idx:
            return 1.0 / rank
    return 0.0


def _dcg(gains: Sequence[float], k: int) -> float:
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains[:k]))


def ndcg_at_k(
    relevant_idx: Sequence[int],
    ranked: Sequence[Tuple[int, float]],
    k: int = 10,
    graded: Optional[Dict[int, float]] = None,
) -> float:
    gains: List[float] = []
    for idx, _ in ranked[:k]:
        if graded:
            gains.append(graded.get(idx, 0.0))
        else:
            gains.append(1.0 if idx in relevant_idx else 0.0)

    dcg = _dcg(gains, k)

    ideal_gains: List[float]
    if graded:
        ideal_gains = sorted(graded.values(), reverse=True)[:k]
    else:
        ideal_gains = sorted([1.0 for r in relevant_idx], reverse=True)[:k]

    idcg = _dcg(ideal_gains, k)
    if idcg == 0:
        return 0.0
    return dcg / idcg


def evaluate_rerank(
    qrels: Dict[int, Sequence[int]],  # query index -> relevant doc indices.
    scores_by_query: Sequence[Sequence[float]],  # query idx -> scores over doc pool.
    graded: Optional[Dict[int, Dict[int, float]]] = None,  # query idx -> doc idx -> grade.
    k: int = 10,
) -> Dict[str, float]:
    """Evaluate recall@k / MRR@k / nDCG@k for a reranker given per-query doc scores.

    scores_by_query[i][j] is the reranker score for the j-th document pool
    member w.r.t. query i. qrels maps query index -> relevant doc indices.
    """
    n_q = len(scores_by_query)
    if n_q == 0:
        return {"recall@1": 0.0, "recall@5": 0.0, "recall@10": 0.0, "mrr@10": 0.0, "ndcg@10": 0.0}

    recall1 = recall5 = recall10 = mrr = ndcg = 0.0

    for qi in range(n_q):
        rel = qrels.get(qi, [])
        if not rel:
            continue
        ranked = _rank_list(scores_by_query[qi], max(k, 10))
        graded_for_q = graded.get(qi) if graded else None

        recall1 += recall_at_k(rel, ranked, 1)
        recall5 += recall_at_k(rel, ranked, 5)
        recall10 += recall_at_k(rel, ranked, 10)
        mrr += mrr_at_k(rel, ranked, 10)
        ndcg += ndcg_at_k(rel, ranked, k, graded_for_q)

    n_with_rel = sum(1 for rel in qrels.values() if rel)
    n_with_rel = max(n_with_rel, 1)

    return {
        "recall@1": round(recall1 / n_with_rel, 4),
        "recall@5": round(recall5 / n_with_rel, 4),
        "recall@10": round(recall10 / n_with_rel, 4),
        "mrr@10": round(mrr / n_with_rel, 4),
        "ndcg@10": round(ndcg / n_with_rel, 4),
    }


def run(cfg: RerankerConfig, writer: ProgressWriter, job_dir: Path, dataset_path: str) -> int:
    """Run reranker fine-tuning; returns the process exit code."""
    stop_requested = False

    def handle_sigterm(signum, frame):  # noqa: ARG001
        nonlocal stop_requested
        stop_requested = True
        writer.event("signal", signal="SIGTERM", msg="graceful stop requested")

    signal.signal(signal.SIGTERM, handle_sigterm)

    # --- Load dataset ---.
    try:
        pairs, graded_pairs = load_dataset(dataset_path and Path(dataset_path) or Path(cfg.dataset_path))
    except Exception as exc:  # noqa: BLE001
        save_metrics(job_dir, {"status": "failed", "error": f"dataset load failed: {exc}"})
        return 1

    writer.event("status", value="dataset_loaded", pairs=len(pairs), graded=len(graded_pairs))

    if len(pairs) < 2:
        save_metrics(job_dir, {"status": "failed", "error": "need at least 2 query/document pairs for reranker training"})
        return 1

    # --- Import heavy libs lazily ---.
    try:
        import torch
        from sentence_transformers import CrossEncoder  # noqa: F401
    except ImportError:
        save_metrics(job_dir, {"status": "failed", "error": "missing dependencies: sentence-transformers and torch required for reranker training"})
        return 1

    device = "cuda" if torch.cuda.is_available() else "cpu"
    writer.event("device", device_type=device)

    try:
        # --- Split train/eval (held out by query) ---.
        import numpy as np

        # Group pairs by query for the split (avoids leaking the same query
        # across train/eval).
        by_query: Dict[str, List[Tuple[str, str, float]]] = {}
        for q, d, lbl in pairs:
            by_query.setdefault(q, []).append((q, d, lbl))

        queries = sorted(by_query.keys())
        rng = np.random.default_rng(cfg.seed)
        query_ids = rng.permutation(len(queries)).tolist()
        cut = min(max(1, int(len(queries) * (1 - cfg.validation_split))), len(queries) - 1)

        train_queries = {queries[i] for i in query_ids[:cut]}
        eval_queries = {queries[i] for i in query_ids[cut:]}

        train_pairs = [p for q in train_queries for p in by_query[q]]
        eval_pairs = [p for q in eval_queries for p in by_query[q]]

        if not train_pairs or not eval_pairs:
            save_metrics(job_dir, {"status": "failed", "error": "training or eval split is empty (fewer than 2 queries?)"})
            return 1

        # --- Load base cross-encoder ---.
        model = CrossEncoder(cfg.base_model, num_labels=1, cache_folder=cfg.model_cache_dir, device=device)

        # --- Base-model retrieval metrics (before fine-tuning) ---.
        eval_q_list: List[str] = []
        eval_pool: Dict[str, List[Tuple[str, float]]] = {}  # query -> [(doc, grade)].
        for q, d, lbl in eval_pairs:
            if q not in eval_pool:
                eval_q_list.append(q)
                eval_pool[q] = []
            eval_pool[q].append((d, lbl))

        doc_set: List[str] = []
        for q, pairs_list in eval_pool.items():
            for d, _ in pairs_list:
                if d not in doc_set:
                    doc_set.append(d)
        doc_index = {d: i for i, d in enumerate(doc_set)}

        writer.event("status", value="evaluating_base")

        # Score every (eval query, pool doc) pair with the base model.
        base_scores: List[List[float]] = []
        qrels: Dict[int, Sequence[int]] = {}
        graded_qrels: Dict[int, Dict[int, float]] = {}

        for qi, q in enumerate(eval_q_list):
            pool = list(doc_set)
            scores = model.predict([(q, d) for d in pool], show_progress_bar=False)
            base_scores.append([float(s) for s in scores])

            rel = [doc_index[d] for d, _ in eval_pool[q]]
            qrels[qi] = rel

            grad_map = {doc_index[d]: lbl for d, lbl in eval_pool[q]}
            if grad_map and any(v not in (0.0, 1.0) for v in grad_map.values()):
                graded_qrels[qi] = grad_map

        base_metrics = evaluate_rerank(qrels, base_scores, graded_qrels or None, k=10)
        writer.event("base_metrics", **{k: v for k, v in base_metrics.items()})

        # --- Training data (CrossEncoder InputExamples with label) ---.
        from sentence_transformers import InputExample
        from torch.utils.data import DataLoader

        train_examples = [
            InputExample(texts=[q, d], label=lbl) for q, d, lbl in train_pairs
        ]
        dataloader = DataLoader(train_examples, batch_size=cfg.batch_size, shuffle=True)

        total_steps = max(1, len(dataloader)) * cfg.epochs
        if cfg.max_train_steps and cfg.max_train_steps > 0:
            total_steps = min(total_steps, cfg.max_train_steps)

        # --- Optimizer / scheduler ---.
        import torch.optim as optim
        from transformers import get_linear_schedule_with_warmup

        optimizer = optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
        warmup_steps = int(total_steps * cfg.warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
        )

        # --- Loss ---.
        loss_fn = None
        if cfg.loss == "margin":
            # Margin ranking loss: label = binary (1 = relevant).
            # Construct per-pair (anchor, positive, negative) triples.
            from torch.nn import MarginRankingLoss

            loss_fn = MarginRankingLoss(margin=0.2)
        else:
            from torch.nn import BCEWithLogitsLoss

            loss_fn = BCEWithLogitsLoss()

        # --- Training loop ---.
        writer.event("status", value="starting_training", total_steps=total_steps)
        model.train()
        step = 0
        for epoch in range(cfg.epochs):
            for batch in dataloader:
                if stop_requested:
                    break

                texts = [ex.texts for ex in batch]  # [(q,d), (q,d), ...]
                labels = torch.tensor([ex.label for ex in batch], dtype=torch.float32, device=device)

                if cfg.loss == "margin":
                    # Pairwise margin: for each batch, treat all as positives
                    # and sample negatives from the batch in-batch.
                    scores = model.predict(texts, convert_to_numpy=False)
                    scores = scores.squeeze()
                    # Build positive/negative pairs within the batch.
                    n = scores.size(0)
                    margins = torch.zeros_like(scores)
                    for i in range(n):
                        for j in range(n):
                            if i != j:
                                margins[i] += torch.relu(margins[i] + 0.2 - (scores[i] - scores[j]))
                    loss = margins.mean()
                else:
                    scores = model.predict(texts, convert_to_numpy=False)
                    scores = scores.squeeze()
                    loss = loss_fn(scores, labels)

                loss.backward()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                step += 1

                if step % 5 == 0 or step == total_steps:
                    writer.progress(step=step, total_steps=total_steps, epoch=float(epoch), loss=float(loss.item()))

                if cfg.max_train_steps and step >= cfg.max_train_steps:
                    break
            if stop_requested:
                break

        # --- Save final model (CrossEncoder folder) ---.
        model_output = job_dir / "model"
        model_output.mkdir(parents=True, exist_ok=True)
        model.save(str(model_output))

        # --- Post-tune evaluation ---.
        writer.event("status", value="evaluating_tuned")
        tuned_scores: List[List[float]] = []
        for qi, q in enumerate(eval_q_list):
            scores = model.predict([(q, d) for d in doc_set], show_progress_bar=False)
            tuned_scores.append([float(s) for s in scores])

        tuned_metrics = evaluate_rerank(qrels, tuned_scores, graded_qrels or None, k=10)
        writer.event("tuned_metrics", **{k: v for k, v in tuned_metrics.items()})

        # --- ONNX export (optional) ---.
        if cfg.export_onnx:
            try:
                onnx_dir = job_dir / "model.onnx"
                onnx_dir.mkdir(parents=True, exist_ok=True)
                try:
                    from optimum.onnxruntime import ORTModelForSequenceClassification  # type: ignore[import]

                    onnx_model = ORTModelForSequenceClassification.from_pretrained(str(model_output), export=True)
                    onnx_model.save_pretrained(str(onnx_dir))
                    writer.event("export", status="done", format="onnx", path=str(onnx_dir))
                except ImportError:
                    writer.event("export", status="skipped", reason="optimum not installed")
            except Exception as exc:  # noqa: BLE001
                writer.event("export", status="failed", reason=str(exc))

        # --- Metrics ---.
        has_graded = bool(graded_qrels)
        metrics: Dict[str, Any] = {
            "status": "completed",
            "kind": "reranker",
            "loss_name": cfg.loss,
            "train_examples": len(train_pairs),
            "eval_examples": len(eval_pairs),
            "graded_relevance": has_graded,
        }
        metrics.update({f"tuned_{k}": v for k, v in tuned_metrics.items()})
        metrics.update({f"base_{k}": v for k, v in base_metrics.items()})
        metrics["delta_ndcg@10"] = round(tuned_metrics["ndcg@10"] - base_metrics["ndcg@10"], 4)

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


# Register this module's runner under the reranker kind.
def _runner(cfg_dict: dict, writer: object, job_dir: str, dataset_path: str) -> int:
    c = cfg_dict if isinstance(cfg_dict, RerankerConfig) else RerankerConfig.from_dict(cfg_dict)
    c.dataset_path = dataset_path or c.dataset_path
    return run(c, writer, Path(job_dir), c.dataset_path)


register("reranker", _runner)