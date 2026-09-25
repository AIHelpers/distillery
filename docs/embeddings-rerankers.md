# Embedding and Reranker Models

Distillery can fine-tune **sentence-embedding models** (bi-encoders) and
**cross-encoder rerankers** on domain data. This is the highest-leverage
upgrade for RAG, enterprise search, deduplication, recommendations and
support-knowledge retrieval — domain-tuned embeddings often beat much larger
general models.

## 1. Data format

Embedding/reranker tasks do **not** use the `input -> output` row layout.
Instead they rely on a **typed `Payload`** with retrieval-specific fields.
Four shapes are supported and auto-detected on import:

| Kind | Shape | Notes |
|------|-------|-------|
| Pair | `{"query","positive"}` | Minimum viable — in-batch negatives |
| Triplet | `{"query","positive","negative"}` | Better — hard negatives |
| Graded | `{"query","document","label"}` | 0/1 or graded relevance (reranker) |
| Docs only | `{"document"}` | **Synthetic query generation**, then review |

Importers handle both **JSONL** (one record per line, shape auto-detected) and
**CSV** (column headers drive mapping):

```csv
query,positive
how do I reset my password,Reset your password from the Account menu

# or with hard negatives
query,positive,negative
how do I reset my password,Reset your password from the Account menu,Passwords must be 8 characters

# or graded relevance (reranker)
query,document,label
how do I reset my password,Reset your password from the Account menu,1
```

### Holdout splitting

The evaluation set is held out **by query group**: a query and all of its
documents go into the same split, so nothing leaks across the train/eval
boundary. On import, ~20% of distinct query groups are deterministically
tagged as eval (marked `holdout:query_group` in the example's flag note).
Curation never drops holdout rows, so the trainer can split on this marker.

### Synthetic query generation

For a "docs only" dataset, Distillery can **generate queries from documents**
via the LLM adapter (C10). Generated `{"query","positive"}` pairs are tagged
synthetic so you can review them in the dataset tab before training. Because
synthetic queries can be too easy or biased, they are always mixed with any
real queries you provide and are human-reviewed.

## 2. Base models

| Kind | Catalog entries |
|------|-----------------|
| Embedding (`KindEmbedding`) | `bge-small-en-v1.5` (384-dim), `bge-base-en-v1.5` (768), `e5-base-v2` (768), `gte-base` (768), `multilingual-e5-base` (768) |
| Reranker (`KindReranker`) | `bge-reranker-base`, `ms-marco-MiniLM-L-6` |

The selector right-sizes from the catalog by language, dimension/latency
budget, and corpus size. **Query/document prefixes** (e5 and bge require
them) are stored with the model record and applied automatically at
encode time — never by hand.

## 3. Training configuration

```jsonc
// EmbeddingConfig
{
  "max_seq_len": 512,
  "loss": "mnrl",            // "mnrl" (pairs, in-batch negatives) | "triplet"
  "hard_negatives": false,   // mine hard negatives on top of pairs
  "matryoshka": [768, 384, 128], // optional truncatable vectors
  "epochs": 10,
  "learning_rate": 0.00002,
  "batch_size": 64,          // large matters for in-batch negatives
  "grad_cache": false,       // trade compute for VRAM
  "normalize": true
}

// RerankerConfig
{
  "max_seq_len": 512,
  "loss": "bce",             // "bce" (graded) | "margin" (pairwise)
  "epochs": 5,
  "learning_rate": 0.00002,
  "batch_size": 32,
  "grad_cache": false
}
```

Trainer workers:

- `trainer/tasks/embedding.py` — `sentence-transformers` with
  `MultipleNegativesRankingLoss` (pairs) or triplet loss. Optional
  **hard-negative mining** (embed the corpus with the base model, take top-k
  non-relevant hits as negatives) and optional **Matryoshka loss** for
  truncatable vectors to cut storage cost.
- `trainer/tasks/reranker.py` — `CrossEncoder` with BCE loss on graded pairs.

## 4. Serving / inference API

Deployed embedding and reranker models expose dedicated endpoints (API key
required, same as `/predict`):

```http
POST /api/v1/inference/{deploymentID}/embed
Authorization: Bearer <key>
Content-Type: application/json

{ "input": ["text a", "text b"], "type": "query" }
```

```json
{
  "kind": "embedding",
  "result": { "dim": 768, "vectors": [[...], [...]] }
}
```

`type` selects the encode-time prefix: `"query"` (default) or `"document"` —

```http
POST /api/v1/inference/{deploymentID}/rerank
Authorization: Bearer <key>
Content-Type: application/json

{ "query": "how do I reset my password?",
  "documents": ["Reset your password...", "Passwords must be 8..."],
  "top_k": 5 }
```

```json
{
  "kind": "reranker",
  "result": { "ranking": [{"index": 1, "score": 0.92}, {"index": 0, "score": 0.31}] }
}
```

`index` refers to the position in the input `documents` array. `top_k`
limits how many ranked items are returned (default `5`).

## 5. Embed-my-corpus batch job

Distillery is **not** a vector database. The `embed-corpus` endpoint lets you
embed a whole corpus of documents and download the vectors for loading into
any vector DB:

```http
POST /api/v1/inference/{deploymentID}/embed-corpus
Authorization: Bearer <key>
```

The corpus is read from the deployment's task dataset (docs-only rows). The
response is a **CSV attachment** (`corpus_embeddings.csv`), one row per
document, with response headers `X-Corpus-Job-ID` and `X-Corpus-Docs` so you
can correlate the run. Output format is columnar (`id,text,embedding[...]`, the
embedding column a JSON array) — reproducible outside Distillery with any
numpy/cosine search consumer.

## 6. Evaluation

The held-out set is evaluated **base vs tuned, side by side**:

- Primary metric: **nDCG@10** (or **MRR@10** when graded relevance is
  unavailable).
- Also reported: **recall@1/5/10**, embedding **dim**, and an
  **index-size estimate** (`dim * 4 bytes * corpus docs`) so you can see the
  storage cost impact.

Each metric is always shown for the base model first, so you can see the gain
from tuning.

## 7. UI

- **Dataset tab** — pair/triplet importer (JSONL + CSV), "generate queries
  from documents" wizard with a review screen, and negative preview.
- **Results** — recall@k / MRR / nDCG **base vs tuned** side-by-side.
- **Test tab** — paste a query and a small document list, see rankings
  before/after tuning.

## 8. Definition of done

- A demo corpus shows a measurable nDCG@10 gain over the base model.
- `/embed` and `/rerank` return the correct shapes.
- Exported vectors are reproducible outside Distillery.
- Warning shown if vectors from different model versions are mixed (they are
  not comparable) — re-embed your index after retraining.