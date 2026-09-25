package domain

import "time"

// EmbeddingConfig carries the hyperparameters that the Go layer can pass to
// the embedding trainer worker (mirrored by the Python EmbeddingConfig in
// trainer/tasks/embedding.py).
type EmbeddingConfig struct {
	// MaxSeqLen truncates/pads inputs to this many tokens.
	MaxSeqLen int `json:"max_seq_len,omitempty"`
	// Loss selects the loss function ("mnrl" for pairs with in-batch
	// negatives, "triplet" for explicit triplets). Default "mnrl".
	Loss string `json:"loss,omitempty"`
	// HardNegatives enables hard-negative mining on top of triplets when the
	// dataset only provides pairs. Off by default.
	HardNegatives bool `json:"hard_negatives,omitempty"`
	// Matryoshka enables Matryoshka (truncatable) loss and lists the output
	// dimensions to truncate to (e.g. [768, 384, 128]). Empty = disabled.
	Matryoshka []int `json:"matryoshka,omitempty"`
	// Epochs / LearningRate / BatchSize override the defaults. Large
	// batch sizes matter for in-batch negatives.
	Epochs       int     `json:"epochs,omitempty"`
	LearningRate float64 `json:"learning_rate,omitempty"`
	BatchSize    int     `json:"batch_size,omitempty"`
	// GradCache enables gradient caching to trade compute for VRAM.
	GradCache bool `json:"grad_cache,omitempty"`
	// Normalize normalises output vectors to unit length. Default true.
	Normalize bool `json:"normalize,omitempty"`
}

// RerankerConfig carries the cross-encoder reranker hyperparameters that the
// Go layer can pass to the reranker trainer worker (mirrored by the Python
// RerankerConfig in trainer/tasks/reranker.py).
type RerankerConfig struct {
	// MaxSeqLen truncates/pads the query+document pair to this many tokens.
	MaxSeqLen int `json:"max_seq_len,omitempty"`
	// Loss selects the loss function ("bce" for graded 0/1 relevance,
	// "margin" for pairwise margin loss). Default "bce".
	Loss string `json:"loss,omitempty"`
	// Epochs / LearningRate / BatchSize override the defaults.
	Epochs       int     `json:"epochs,omitempty"`
	LearningRate float64 `json:"learning_rate,omitempty"`
	BatchSize    int     `json:"batch_size,omitempty"`
	// GradCache enables gradient caching to trade compute for VRAM.
	GradCache bool `json:"grad_cache,omitempty"`
}

// RetrievalMetric identifies a retrieval evaluation metric used by the
// embedding/reranker evaluator.
type RetrievalMetric string

const (
	// MetricNDCG10 is Normalized Discounted Cumulative Gain at k=10.
	MetricNDCG10 RetrievalMetric = "ndcg@10"
	// MetricMRR10 is Mean Reciprocal Rank at k=10.
	MetricMRR10 RetrievalMetric = "mrr@10"
	// MetricRecall1 / Recall5 / Recall10 are recall at k=1/5/10.
	MetricRecall1  RetrievalMetric = "recall@1"
	MetricRecall5  RetrievalMetric = "recall@5"
	MetricRecall10 RetrievalMetric = "recall@10"
)

// RetrievalMetrics holds the evaluation outcome of an embedding/reranker
// model against the held-out query+corpus set (base vs tuned side by side).
type RetrievalMetrics struct {
	// Primary is the headline metric (nDCG@10, or MRR@10 when graded
	// relevance is unavailable).
	Primary RetrievalMetric `json:"primary"`
	// NDG10 / MRR10 / Recall1 / Recall5 / Recall10 hold the evaluated values.
	NDCG10   float64 `json:"ndcg@10"`
	MRR10    float64 `json:"mrr@10"`
	Recall1  float64 `json:"recall@1"`
	Recall5  float64 `json:"recall@5"`
	Recall10 float64 `json:"recall@10"`
	// Dim is the embedding dimension produced by the model.
	Dim int `json:"dim"`
	// IndexSizeEstimate is a rough bytes estimate for the embedded corpus
	// (dim * 4 bytes * corpus docs), used by the UI to surface index cost.
	IndexSizeEstimate int64 `json:"index_size_estimate"`
}

// ModelRecord holds per-model metadata that inference needs to serve the
// right shaped outputs: embedding dimension, normalisation, and the
// query/document prefixes some encoder families (e5, bge) require.
type ModelRecord struct {
	// EmbeddingDim is the output dimension for embedding models.
	EmbeddingDim int `json:"embedding_dim,omitempty"`
	// Normalize indicates the model normalises output vectors to unit length.
	Normalize bool `json:"normalize,omitempty"`
	// QueryPrefix is prepended to user queries at encode time (empty = none).
	QueryPrefix string `json:"query_prefix,omitempty"`
	// DocumentPrefix is prepended to documents at encode time (empty = none).
	DocumentPrefix string `json:"document_prefix,omitempty"`
}

// EmbeddingResult is a single inference result for an embedding model.
type EmbeddingResult struct {
	Dim     int         `json:"dim"`
	Vectors [][]float32 `json:"vectors"`
}

// RerankResult is the ranked output of a reranker model.
type RerankResult struct {
	Ranking []RerankItem `json:"ranking"`
}

// RerankItem is one ranked entry from a reranker (index into the input
// documents list, with a relevance score).
type RerankItem struct {
	Index int     `json:"index"`
	Score float64 `json:"score"`
}

// CorpusEmbedJob is a batch job that embeds a corpus of documents and writes
// the vectors to a CSV/JSONL file so users can load them into any vector DB
// (Distillery does not become a vector database).
type CorpusEmbedJob struct {
	ID            string         `json:"id"`
	DeploymentID  string         `json:"deployment_id,omitempty"`
	TrainingJobID string         `json:"training_job_id"`
	TaskID        string         `json:"task_id"`
	Status        TrainingStatus `json:"status"`
	TotalDocs     int            `json:"total_docs"`
	DoneDocs      int            `json:"done_docs"`
	OutputPath    string         `json:"output_path,omitempty"`
	Error         string         `json:"error,omitempty"`
	CreatedAt     time.Time      `json:"created_at"`
	StartedAt     *time.Time     `json:"started_at,omitempty"`
	CompletedAt   *time.Time     `json:"completed_at,omitempty"`
}
