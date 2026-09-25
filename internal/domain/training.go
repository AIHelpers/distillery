package domain

import "time"

type TrainingStatus string

const (
	TrainingQueued    TrainingStatus = "queued"
	TrainingRunning   TrainingStatus = "running"
	TrainingCompleted TrainingStatus = "completed"
	TrainingFailed    TrainingStatus = "failed"
)

// Capabilities describes what a base model can do, so the model selector can
// right-size a model per kind, dataset size, language, latency target, and
// hardware.
type Capabilities struct {
	// SupportsLoRA indicates the model family supports PEFT LoRA adapters.
	SupportsLoRA bool `json:"supports_lora"`
	// ExportFormats lists the export formats this model can produce
	// (e.g. "gguf", "onnx", "safetensors").
	ExportFormats []string `json:"export_formats,omitempty"`
	// RunsOnCPU is true when the model can be trained/served on CPU.
	RunsOnCPU bool `json:"runs_on_cpu"`
	// MaxSeqLen is the maximum supported sequence length in tokens.
	MaxSeqLen int `json:"max_seq_len"`
	// Languages lists supported natural languages (e.g. "en", "multilingual").
	Languages []string `json:"languages,omitempty"`
}

// BaseModel is one entry in the curated set of open-weight base models
// the platform can automatically select from.
type BaseModel struct {
	Name           string  `json:"name"`
	ParamsBillions float64 `json:"params_billions"`
	Family         string  `json:"family"`
	// RepoID is the HuggingFace repo identifier (used by the local trainer).
	RepoID string `json:"repo_id,omitempty"`
	// MinVRAMGB is the minimum GPU VRAM (GiB) recommended to run this model with QLoRA.
	MinVRAMGB float64 `json:"min_vram_gb,omitempty"`
	// RecommendedQuant is the default quantization level, e.g. "4bit-nf4".
	RecommendedQuant string `json:"recommended_quant,omitempty"`
	// Kind is the model kind this entry is suited for (default causal_lm).
	Kind ModelKind `json:"kind,omitempty"`
	// Capabilities describes what this model can do.
	Capabilities Capabilities `json:"capabilities,omitempty"`
}

// TrainingMetrics are the (simulated) results of a LoRA/QLoRA fine-tune run.
type TrainingMetrics struct {
	FinalLoss     float64 `json:"final_loss"`
	EvalAccuracy  float64 `json:"eval_accuracy"`
	Epochs        int     `json:"epochs"`
	TrainExamples int     `json:"train_examples"`

	// Classifier-specific metrics (populated when Kind == KindSeqClassifier).
	MacroF1          float64                    `json:"macro_f1,omitempty"`
	WeightedF1       float64                    `json:"weighted_f1,omitempty"`
	BaselineMacroF1  float64                    `json:"baseline_macro_f1,omitempty"`
	DeltaMacroF1     float64                    `json:"delta_macro_f1,omitempty"`
	PerClass         map[string]PerClassMetrics `json:"per_class,omitempty"`
	ConfusionMatrix  ConfusionMatrix            `json:"confusion_matrix,omitempty"`
	ThresholdSweep   []ThresholdSweepPoint      `json:"threshold_sweep,omitempty"`
	LabelMap         map[string]int             `json:"label_map,omitempty"`
	DefaultThreshold float64                    `json:"default_threshold,omitempty"`
	MaxLength        int                        `json:"max_length,omitempty"`
	MultiLabel       bool                       `json:"multi_label,omitempty"`

	// Retrieval metrics (populated when Kind == KindEmbedding or KindReranker).
	// NDCG10 / MRR10 / Recall1 / Recall5 / Recall10 are the tuned model's
	// scores; Base* are the base (un-fine-tuned) model on the same split so
	// the UI can show the gain.
	NDCG10       float64 `json:"ndcg10,omitempty"`
	MRR10        float64 `json:"mrr10,omitempty"`
	Recall1      float64 `json:"recall1,omitempty"`
	Recall5      float64 `json:"recall5,omitempty"`
	Recall10     float64 `json:"recall10,omitempty"`
	BaseNDCG10   float64 `json:"base_ndcg10,omitempty"`
	BaseMRR10    float64 `json:"base_mrr10,omitempty"`
	BaseRecall1  float64 `json:"base_recall1,omitempty"`
	BaseRecall5  float64 `json:"base_recall5,omitempty"`
	BaseRecall10 float64 `json:"base_recall10,omitempty"`
	EmbeddingDim int     `json:"embedding_dim,omitempty"`
}

// TrainingJob represents one fine-tuning run for a task.
type TrainingJob struct {
	ID        string           `json:"id"`
	TaskID    string           `json:"task_id"`
	Version   int              `json:"version"` // increments each retrain.
	Kind      ModelKind        `json:"kind"`
	BaseModel BaseModel        `json:"base_model"`
	Status    TrainingStatus   `json:"status"`
	Progress  int              `json:"progress"` // 0-100
	Metrics   *TrainingMetrics `json:"metrics,omitempty"`
	Eval      *EvalReport      `json:"eval,omitempty"`
	Error     string           `json:"error,omitempty"`
	// Classifier holds the hyperparameters passed to a seq_classifier
	// training run (empty for causal_lm).
	Classifier *ClassifierConfig `json:"classifier,omitempty"`
	// Embedding holds the hyperparameters passed to an embedding training
	// run (empty unless Kind == KindEmbedding).
	Embedding *EmbeddingConfig `json:"embedding,omitempty"`
	// Reranker holds the hyperparameters passed to a reranker training
	// run (empty unless Kind == KindReranker).
	Reranker *RerankerConfig `json:"reranker,omitempty"`
	// Model records the per-model inference metadata (dim, prefixes, etc).
	Model ModelRecord `json:"model,omitempty"`
	// ParentJobID links this job to the job it continues from (lineage for
	// DPO and continued training). Empty for the first job in a lineage.
	ParentJobID string     `json:"parent_job_id,omitempty"`
	Seed        int        `json:"seed,omitempty"` // held-out split seed for reproducibility.
	CreatedAt   time.Time  `json:"created_at"`
	StartedAt   *time.Time `json:"started_at,omitempty"`
	CompletedAt *time.Time `json:"completed_at,omitempty"`
}

// TrainingJobRepository is the port for persisting training jobs.
type TrainingJobRepository interface {
	Create(j *TrainingJob) error
	Get(id string) (*TrainingJob, error)
	ListByTask(taskID string) ([]*TrainingJob, error)
	Update(j *TrainingJob) error
	Delete(id string) error
	LatestCompleted(taskID string) (*TrainingJob, error)
}

// SelectionConstraints are the hardware/latency limits a model must satisfy.
type SelectionConstraints struct {
	// MaxVRAMGB caps VRAM usage (0 = no limit).
	MaxVRAMGB float64
	// MaxLatencyMS caps p95 latency for serving (0 = no limit).
	MaxLatencyMS int
	// CPUOnly forces a CPU-capable model (no GPU available).
	CPUOnly bool
}

// ModelSelector chooses a base model for a task based on dataset complexity
// and the requested model kind.
type ModelSelector interface {
	// SelectBaseModel is the legacy task-based selection (kept for
	// backward compatibility); kind-aware callers should use Select.
	SelectBaseModel(task *Task, exampleCount int, avgInputLen, avgOutputLen int) BaseModel
	// Select picks the right-sized base model for a kind + dataset stats.
	Select(kind ModelKind, stats DatasetStats, constraints SelectionConstraints) BaseModel
	// ListBaseModels returns the curated catalog of available base models
	// the user can choose from.
	ListBaseModels() []BaseModel
}

// FineTuner runs (or simulates) a LoRA/QLoRA fine-tuning job asynchronously,
// invoking onUpdate as progress changes and onDone when it finishes.
type FineTuner interface {
	Start(
		job *TrainingJob,
		examples []*Example,
		onUpdate func(progress int,
		), onDone func(metrics *TrainingMetrics, err error))
}
