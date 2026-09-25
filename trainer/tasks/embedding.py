"""Text-embedding (bi-encoder) training task module.

Fine-tunes a sentence-embedding model on domain pairs/triplets using
`sentence-transformers`. Produces:

  - job-dir/model/                  sentence-transformers folder (safetensors +
                                    pooling/normalize config)
  - job-dir/model.onnx/             (optional) ONNX export for CPU serving
  - job-dir/metrics.json            recall@k / MRR@10 / nDCG@10 (tuned vs base)

Dataset shapes accepted (JSONL):
  {"query","positive"}                       pair (in-batch negatives)
  {"query","positive","negative"}            triplet
  {"document"}                                docs-only (no training — used
                                             for synthetic query generation
                                             upstream; skipped here)

Follows the same contract as `classifier.py`: JSONL progress in
job-dir/progress.jsonl and a final metrics.json. The Go side passes
`--kind embedding`.
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

DEFAULT_BATCH_SIZE = 32
DEFAULT_LR = 2e-5
DEFAULT_EPOCHS = 3
DEFAULT_MAX_SEQ_LEN = 512

_POSITIVE_INT_FIELDS = ("max_seq_len", "epochs", "batch_size")
_POSITIVE_FLOAT_FIELDS = ("learning_rate",)


@dataclass
class EmbeddingConfig:
    """Training hyperparameters for a text embedding model.

    Mirrors the Go-side `EmbeddingConfig` (max_seq_len, loss, hard_negatives,
    matryoshka, epochs, learning_rate, batch_size, grad_cache, normalize).
    Missing keys fall back to defaults.
    """

    job_id: str = ""
    base_model: str = "BAAI/bge-small-en-v1.5"
    dataset_path: str = ""
    output_dir: str = ""
    max_seq_len: int = DEFAULT_MAX_SEQ_LEN
    loss: str = "mnrl"  # "mnrl" | "triplet" | "cosine".
    hard_negatives: bool = False
    matryoshka: List[int] = None  # e.g. [768, 384, 128]; empty = disabled.
    epochs: int = DEFAULT_EPOCHS
    learning_rate: float = DEFAULT_LR
    batch_size: int = DEFAULT_BATCH_SIZE
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    grad_cache: bool = False
    normalize: bool = True  # unit-normalise output vectors.
    validation_split: float = 0.15
    export_onnx: bool = True
    model_cache_dir: Optional[str] = None
    seed: int = 42
    max_train_steps: Optional[int] = None  # cap steps (CPU smoke tests).

    def __post_init__(self) -> None:
        if self.matryoshka is None:
            self.matryoshka = []

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "EmbeddingConfig":
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


def load_config(path: Path) -> EmbeddingConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    cfg = EmbeddingConfig.from_dict(raw)
    cfg.dataset_path = str(Path(raw.get("dataset_path", cfg.dataset_path)))
    cfg.output_dir = str(Path(raw.get("output_dir", cfg.output_dir)))
    return cfg


# --- Dataset loading -------------------------------------------------------.

DatasetRow = Dict[str, str]


def load_dataset(path: Path) -> Tuple[List[DatasetRow], str]:
    """Load the curated JSONL into a list of {query, positive, negative?, document?}.

    Returns (rows, format) where format is one of "pair" | "triplet" | "docs_only".
    Docs-only rows (no query) are returned in the list so the caller can
    report them, but they are skipped by training.
    """
    rows: List[DatasetRow] = []
    has_neg = False
    has_docs = False

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)

            query = obj.get("query") or ""
            positive = obj.get("positive") or ""
            negative = obj.get("negative") or ""
            document = obj.get("document") or ""
            instruction = obj.get("instruction") or ""
            output = obj.get("output") or ""

            # Migrate legacy causal_lm rows to pairs when both fields exist.
            if not query and instruction:
                query = instruction
            if not positive and output:
                positive = output

            row: DatasetRow = {}
            if query:
                row["query"] = str(query).strip()
            if positive:
                row["positive"] = str(positive).strip()
            if negative:
                row["negative"] = str(negative).strip()
                has_neg = True
            if document:
                row["document"] = str(document).strip()
                has_docs = True

            if row:
                rows.append(row)

    if not rows:
        raise ValueError("Dataset is empty or has no readable records")

    if has_neg:
        return rows, "triplet"
    if has_docs and not any("query" in r for r in rows):
        return rows, "docs_only"
    return rows, "pair"


def save_metrics(output_dir: Path, metrics: dict[str, Any]) -> None:
    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"metrics written to {metrics_path}", file=sys.stderr)


# --- Retrieval metrics (pure functions, unit-testable) ---------------------.

def _rank_list(scores: Sequence[float], k: int) -> List[Tuple[int, float]]:
    """Return [(index, score)] sorted by score desc, truncated to k."""
    ranked = sorted(enumerate(scores), key=lambda x: -x[1])
    return ranked[:k]


def recall_at_k(relevant_idx: Sequence[int], ranked: Sequence[Tuple[int, float]], k: int) -> float:
    """Recall@k: fraction of relevant docs surfaced in the top-k."""
    if not relevant_idx:
        return 0.0
    top_k = {idx for idx, _ in ranked[:k]}
    hit = sum(1 for r in relevant_idx if r in top_k)
    return hit / len(relevant_idx)


def mrr_at_k(relevant_idx: Sequence[int], ranked: Sequence[Tuple[int, float]], k: int = 10) -> float:
    """MRR@k: reciprocal rank of the first relevant doc (capped at k)."""
    for rank, (idx, _) in enumerate(ranked[:k], start=1):
        if idx in relevant_idx:
            return 1.0 / rank
    return 0.0


def _dcg(gains: Sequence[float], k: int) -> float:
    """Discounted cumulative gain (binary gains supported)."""
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains[:k]))


def ndcg_at_k(
    relevant_idx: Sequence[int],
    ranked: Sequence[Tuple[int, float]],
    k: int = 10,
    graded: Optional[Dict[int, float]] = None,
) -> float:
    """nDCG@k over the ranking.

    `graded` maps document index -> relevance grade (0.0-1.0). When absent,
    binary gains (relevant=1) are used.
    """
    gains: List[float] = []
    for idx, _ in ranked[:k]:
        if graded:
            gains.append(graded.get(idx, 0.0))
        else:
            gains.append(1.0 if idx in relevant_idx else 0.0)

    dcg = _dcg(gains, k)

    # Ideal ordering: relevant/graded docs first, sorted by grade desc.
    ideal_gains: List[float]
    if graded:
        ideal_gains = sorted(graded.values(), reverse=True)[:k]
    else:
        ideal_gains = sorted([1.0 for r in relevant_idx], reverse=True)[:k]

    idcg = _dcg(ideal_gains, k)
    if idcg == 0:
        return 0.0
    return dcg / idcg


def evaluate_retrieval(
    query_vecs: Sequence[Sequence[float]],
    doc_vecs: Sequence[Sequence[float]],
    qrels: Dict[int, Sequence[int]],  # query index -> relevant doc indices.
    graded: Optional[Dict[int, Dict[int, float]]] = None,  # query idx -> doc idx -> grade.
    k: int = 10,
) -> Dict[str, float]:
    """Evaluate recall@k / MRR@k / nDCG@k over a corpus of query doc pairs.

    Uses numpy cosine similarity. query_vecs[i] is the embedding of the ith
    query; doc_vecs[j] the embedding of the jth doc. qrels maps query idx ->
    list of relevant doc indices.
    """
    import numpy as np

    queries = np.asarray(query_vecs, dtype=np.float32)
    docs = np.asarray(doc_vecs, dtype=np.float32)

    n_q = queries.shape[0]
    if n_q == 0:
        return {"recall@1": 0.0, "recall@5": 0.0, "recall@10": 0.0, "mrr@10": 0.0, "ndcg@10": 0.0}

    recall1 = recall5 = recall10 = mrr = ndcg = 0.0

    for qi in range(n_q):
        rel = qrels.get(qi, [])
        if not rel:
            continue
        sims = (queries[qi] @ docs.T).tolist()
        ranked = _rank_list(sims, max(k, 10))
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


# --- Hard-negative mining ------------------------------------------------.

def _mine_hard_negatives(
    model,
    tokenizer,
    queries: Sequence[str],
    positives: Sequence[str],
    corpus_index,
    top_k: int,
    device: str,
    max_seq_len: int,
) -> List[str]:
    """Mine hard negatives by embedding the corpus with the base model and
    taking the top-k non-relevant hits as negatives for each query. A document
    is relevant if it is the query's own positive; everything else in the
    top-k is treated as a mined (hard) negative."""
    import numpy as np

    corpus_docs = list(corpus_index.keys())
    doc_vecs = np.asarray(_encode(model, tokenizer, corpus_docs, max_seq_len, device), dtype=np.float32)
    query_vecs = np.asarray(_encode(model, tokenizer, list(queries), max_seq_len, device), dtype=np.float32)

    mined: List[str] = []
    for qi, q in enumerate(queries):
        rel_doc = positives[qi]
        sims = (query_vecs[qi] @ doc_vecs.T).tolist()
        ranked = sorted(enumerate(sims), key=lambda x: -x[1])
        negs: List[str] = []
        for idx, _score in ranked:
            doc = corpus_docs[idx]
            if doc == rel_doc:
                continue
            negs.append(doc)
            if len(negs) >= top_k:
                break
        mined.append(negs[0] if negs else "")
    return mined


# --- Training --------------------------------------------------------------.

def _load_model_and_tokenizer(base_model: str, model_cache_dir: Optional[str], device: str):
    """Load a sentence-transformers model or fall back to a HuggingFace model."""
    try:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(base_model, cache_folder=model_cache_dir, device=device)
        tokenizer = model.tokenizer
        return model, tokenizer
    except ImportError:
        from transformers import AutoModel, AutoTokenizer  # type: ignore[import]

        tokenizer = AutoTokenizer.from_pretrained(base_model, cache_dir=model_cache_dir)
        model = AutoModel.from_pretrained(base_model, cache_dir=model_cache_dir).to(device)
        return model, tokenizer


def _encode(model, tokenizer, texts: Sequence[str], max_seq_len: int, device: str) -> List[List[float]]:
    """Encode texts into unit-normalised embeddings (CLS mean pooling)."""
    import torch

    model.eval()
    vecs: List[List[float]] = []

    # Batch to keep VRAM bounded.
    for start in range(0, len(texts), 32):
        batch = texts[start : start + 32]
        enc = tokenizer(
            batch,
            truncation=True,
            max_length=max_seq_len,
            padding=True,
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            out = model(**enc)
            # Mean pooling over token embeddings.
            if hasattr(out, "last_hidden_state"):
                hidden = out.last_hidden_state
            else:
                hidden = out[0]
            mask = enc.get("attention_mask", torch.ones_like(hidden[:, :, 0])).unsqueeze(-1)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        vecs.extend(pooled.cpu().tolist())

    # Unit-normalise rows.
    import numpy as np

    arr = np.asarray(vecs, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    arr = arr / norms
    return arr.tolist()


def run(cfg: EmbeddingConfig, writer: ProgressWriter, job_dir: Path, dataset_path: str) -> int:
    """Run embedding fine-tuning; returns the process exit code."""
    stop_requested = False

    def handle_sigterm(signum, frame):  # noqa: ARG001
        nonlocal stop_requested
        stop_requested = True
        writer.event("signal", signal="SIGTERM", msg="graceful stop requested")

    signal.signal(signal.SIGTERM, handle_sigterm)

    # --- Load dataset ---.
    try:
        rows, fmt = load_dataset(dataset_path and Path(dataset_path) or Path(cfg.dataset_path))
    except Exception as exc:  # noqa: BLE001
        save_metrics(job_dir, {"status": "failed", "error": f"dataset load failed: {exc}"})
        return 1

    writer.event("status", value="dataset_loaded", format=fmt, rows=len(rows))

    # Docs-only datasets have nothing to train on (synthetic generation is the
    # upstream flow) — record that and finish cleanly.
    if fmt == "docs_only":
        save_metrics(job_dir, {"status": "completed", "kind": "embedding", "note": "docs-only dataset; no training performed (synthetic query generation required first)", "train_examples": 0})
        return 0

    # --- Import heavy libs lazily ---.
    try:
        import torch
        from sentence_transformers import SentenceTransformer  # noqa: F401
    except ImportError:
        save_metrics(job_dir, {"status": "failed", "error": "missing dependencies: sentence-transformers and torch required for embedding training"})
        return 1

    device = "cuda" if torch.cuda.is_available() else "cpu"
    writer.event("device", device_type=device)

    try:
        # --- Build training rows ---.
        queries = [r["query"] for r in rows if "query" in r and "positive" in r]
        positives = [r["positive"] for r in rows if "query" in r and "positive" in r]
        negatives = [r.get("negative", "") for r in rows if "query" in r and "positive" in r]

        if len(queries) < 2:
            save_metrics(job_dir, {"status": "failed", "error": "need at least 2 query/positive pairs for embedding training"})
            return 1

        # --- Split train/eval (held out by query) ---.
        import numpy as np

        rng = np.random.default_rng(cfg.seed)
        n = len(queries)
        shuffled = rng.permutation(n)
        cut = min(max(1, int(n * (1 - cfg.validation_split))), n - 1)
        train_idx = shuffled[:cut].tolist()
        eval_idx = shuffled[cut:].tolist()

        # --- Load base embedding model ---.
        model, tokenizer = _load_model_and_tokenizer(cfg.base_model, cfg.model_cache_dir, device)

        # --- Base-model retrieval metrics (before fine-tuning) ---.
        eval_queries = [queries[i] for i in eval_idx]
        # Corpus = all positives (so the eval queries' true positives are in the index).
        corpus = list(dict.fromkeys(positives))  # dedupe preserving order.
        corpus_index = {doc: i for i, doc in enumerate(corpus)}

        writer.event("status", value="evaluating_base")
        q_vecs_base = _encode(model, tokenizer, eval_queries, cfg.max_seq_len, device)
        d_vecs_base = _encode(model, tokenizer, corpus, cfg.max_seq_len, device)
        qrels_base: Dict[int, Sequence[int]] = {}
        for qi, q in enumerate(eval_queries):
            rel = [corpus_index[p] for p in [positives[qi]] if p in corpus_index]
            if rel:
                qrels_base[qi] = rel
        base_metrics = evaluate_retrieval(q_vecs_base, d_vecs_base, qrels_base, k=10)

        writer.event("base_metrics", **{k: v for k, v in base_metrics.items()})

        # --- Training data (transform into sentence-transformers format) ---.
        train_queries = [queries[i] for i in train_idx]
        train_positives = [positives[i] for i in train_idx]
        train_negatives = [negatives[i] for i in train_idx]

        # --- Hard-negative mining (optional, on for pairs/triplets) ---.
        # When enabled and the dataset has no explicit negatives, embed the
        # corpus with the base model and mine the top non-relevant hit as a
        # hard negative for each query.
        if cfg.hard_negatives and not any(n for n in train_negatives):
            writer.event("status", value="mining_hard_negatives", top_k=1)
            mined = _mine_hard_negatives(
                model,
                tokenizer,
                train_queries,
                train_positives,
                corpus_index,
                top_k=1,
                device=device,
                max_seq_len=cfg.max_seq_len,
            )
            train_negatives = [n if n else "" for n in mined]

        # --- Loss selection ---.
        loss_name = cfg.loss or "mnrl"

        loss = None
        try:
            from sentence_transformers import losses, util  # noqa: F401

            if loss_name == "triplet":
                # Triplet loss needs a negative per sample.
                use_neg = [n for n in train_negatives if n]
                if len(use_neg) == len(train_queries):
                    loss = losses.TripletLoss(model=model, distance_metric=losses.cosine_distance)
                else:
                    # No negatives: fall back to MultipleNegativesRankingLoss.
                    writer.event("status", value="no_negatives_falling_back_to_mnrl")
                    loss = losses.MultipleNegativesRankingLoss(model=model)
            else:
                loss = losses.MultipleNegativesRankingLoss(model=model)

            # --- Matryoshka wrapper (optional) ---.
            # When matryoshka dims are configured, wrap the selected loss so
            # every loss term is computed over embeddings truncated to each
            # configured dimension. This produces a single model that can
            # serve vectors at several sizes (and cut storage cost by
            # exporting narrower vectors).
            if cfg.matryoshka and all(dim > 0 for dim in cfg.matryoshka):
                trunc_dims = [int(d) for d in cfg.matryoshka if int(d) > 0]
                trunc_weights = [1.0 / (i + 1) for i in range(len(trunc_dims))]
                base_loss = loss

                class _MatryoshkaLoss(torch.nn.Module):
                    """MultipleNegativesRankingLoss summed over truncated dims."""

                    def __init__(self, inner_loss):
                        super().__init__()
                        self.inner_loss = inner_loss
                        self.dims = trunc_dims
                        self.weights = trunc_weights

                    def forward(self, sentence_features, labels):  # noqa: ARG002
                        from sentence_transformers import util as st_util

                        anchors = self.inner_loss.model.encode(sentence_features[0], convert_to_tensor=True)
                        positives = self.inner_loss.model.encode(sentence_features[1], convert_to_tensor=True)

                        total = torch.zeros((), dtype=anchors.dtype, device=anchors.device)
                        for dim, w in zip(self.dims, self.weights):
                            a = anchors[:, :dim]
                            p = positives[:, :dim]
                            sims = st_util.cos_sim(a, p)
                            logits = sims * 20.0  # temperature.
                            log_probs = torch.log_softmax(logits, dim=-1)
                            total = total + w * -log_probs.diag().mean()
                        return total

                loss = _MatryoshkaLoss(base_loss)
                writer.event("status", value="matryoshka_enabled", dims=trunc_dims)
        except Exception as exc:  # noqa: BLE001
            writer.event("error", message=str(exc), class_name=type(exc).__name__)
            save_metrics(job_dir, {"status": "failed", "error": f"loss construction failed: {exc}"})
            return 1

        # --- TrainDataLoader with in-batch negatives ---.
        from torch.utils.data import DataLoader
        from sentence_transformers import InputExample

        train_examples = []
        for q, p, n in zip(train_queries, train_positives, train_negatives):
            texts = [q, p] + ([n] if n else [])
            train_examples.append(InputExample(texts=texts))

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

        # --- Training loop ---.
        writer.event("status", value="starting_training", total_steps=total_steps, format=fmt)
        model.train()
        step = 0
        for epoch in range(cfg.epochs):
            for batch in dataloader:
                if stop_requested:
                    break
                features = {}
                loss_value = loss(batch, features)
                loss_value.backward()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                step += 1

                if step % 5 == 0 or step == total_steps:
                    writer.progress(step=step, total_steps=total_steps, epoch=float(epoch), loss=float(loss_value.item()))

                if cfg.max_train_steps and step >= cfg.max_train_steps:
                    break
            if stop_requested:
                break

        # --- Save final model (sentence-transformers folder) ---.
        model_output = job_dir / "model"
        model_output.mkdir(parents=True, exist_ok=True)
        model.save(str(model_output))
        tokenizer.save_pretrained(str(model_output))

        # --- Post-tune evaluation ---.
        writer.event("status", value="evaluating_tuned")
        q_vecs_tuned = _encode(model, tokenizer, eval_queries, cfg.max_seq_len, device)
        d_vecs_tuned = _encode(model, tokenizer, corpus, cfg.max_seq_len, device)
        tuned_metrics = evaluate_retrieval(q_vecs_tuned, d_vecs_tuned, qrels_base, k=10)
        writer.event("tuned_metrics", **{k: v for k, v in tuned_metrics.items()})

        # --- Model dimension ---.
        embed_dim = len(q_vecs_tuned[0]) if q_vecs_tuned else 0
        index_estimate = embed_dim * 4 * len(corpus)

        # --- ONNX export (optional) ---.
        if cfg.export_onnx:
            try:
                from sentence_transformers import SentenceTransformer as ST

                onnx_dir = job_dir / "model.onnx"
                onnx_dir.mkdir(parents=True, exist_ok=True)
                st_model = ST(str(model_output), device=device)
                st_model.save(str(onnx_dir))
                try:
                    from optimum.onnxruntime import ORTModelForFeatureExtraction  # type: ignore[import]

                    onnx_model = ORTModelForFeatureExtraction.from_pretrained(str(model_output), export=True)
                    onnx_model.save_pretrained(str(onnx_dir))
                    writer.event("export", status="done", format="onnx", path=str(onnx_dir))
                except ImportError:
                    writer.event("export", status="skipped", reason="optimum not installed")
            except Exception as exc:  # noqa: BLE001
                writer.event("export", status="failed", reason=str(exc))

        # --- Metrics ---.
        metrics: Dict[str, Any] = {
            "status": "completed",
            "kind": "embedding",
            "format": fmt,
            "loss_name": loss_name,
            "train_examples": len(train_queries),
            "eval_examples": len(eval_queries),
            "embedding_dim": embed_dim,
            "index_size_estimate": index_estimate,
            "normalize": cfg.normalize,
            "query_prefix": "",
            "document_prefix": "passage: " if "e5-" in cfg.base_model or "e5-" in cfg.base_model.lower() else "",
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


# Register this module's runner under the embedding kind.
def _runner(cfg_dict: dict, writer: object, job_dir: str, dataset_path: str) -> int:
    c = cfg_dict if isinstance(cfg_dict, EmbeddingConfig) else EmbeddingConfig.from_dict(cfg_dict)
    c.dataset_path = dataset_path or c.dataset_path
    return run(c, writer, Path(job_dir), c.dataset_path)


register("embedding", _runner)