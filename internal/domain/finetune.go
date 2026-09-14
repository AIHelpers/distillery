package domain

import (
	"context"
	"time"
)

// ProgrammingLanguage is the target coding language for a fine-tune.
type ProgrammingLanguage string

const (
	LangPython     ProgrammingLanguage = "python"
	LangGo         ProgrammingLanguage = "go"
	LangJavaScript ProgrammingLanguage = "javascript"
	LangTypeScript ProgrammingLanguage = "typescript"
	LangJava       ProgrammingLanguage = "java"
	LangCSharp     ProgrammingLanguage = "csharp"
	LangRust       ProgrammingLanguage = "rust"
	LangCpp        ProgrammingLanguage = "cpp"
)

// SkillCategory is the coding skill a fine-tune targets.
type SkillCategory string

const (
	SkillCodeGeneration SkillCategory = "code_generation"
	SkillCodeCompletion SkillCategory = "code_completion"
	SkillBugFixing      SkillCategory = "bug_fixing"
	SkillOptimization   SkillCategory = "optimization"
	SkillDocumentation  SkillCategory = "documentation"
)

// RequestStatus tracks the lifecycle of a fine-tune request.
type RequestStatus string

const (
	RequestDraft     RequestStatus = "draft"
	RequestQueued    RequestStatus = "queued"
	RequestTraining  RequestStatus = "training"
	RequestCompleted RequestStatus = "completed"
	RequestFailed    RequestStatus = "failed"
)

// JobStatus tracks a fine-tuning job's execution state.
type JobStatus string

const (
	JobQueued    JobStatus = "queued"
	JobPreparing JobStatus = "preparing"
	JobRunning   JobStatus = "running"
	JobCompleted JobStatus = "completed"
	JobFailed    JobStatus = "failed"
)

// MetricType identifies a training metric.
type MetricType string

const (
	MetricLoss              MetricType = "loss"
	MetricPerplexity        MetricType = "perplexity"
	MetricBleuScore         MetricType = "bleu_score"
	MetricCodeExecutability MetricType = "code_executability"
	MetricSyntaxValid       MetricType = "syntax_valid"
)

// TrainingParameters are the hyperparameters for a fine-tune run.
type TrainingParameters struct {
	Epochs             int     `json:"epochs"`
	BatchSize          int     `json:"batch_size"`
	LearningRate       float64 `json:"learning_rate"`
	WarmupSteps        int     `json:"warmup_steps"`
	MaxSequenceLength  int     `json:"max_sequence_length"`
	GradientAccumSteps int     `json:"gradient_accumulation_steps"`
	WeightDecay        float64 `json:"weight_decay"`
	SchedulerType      string  `json:"scheduler_type"` // "linear", "cosine".
	OptimizerType      string  `json:"optimizer_type"` // "adam", "adamw".
	PreserveSyntax     bool    `json:"preserve_syntax"`
	ContextWindow      int     `json:"context_window"`
	BalancedSampling   bool    `json:"balanced_sampling"`
}

// ValidationParameters configure how a fine-tune run is evaluated.
type ValidationParameters struct {
	ValidationSplit    float64      `json:"validation_split"`
	ValidateEverySteps int          `json:"validate_every_steps"`
	Metrics            []MetricType `json:"metrics"`
}

// MetricPoint is a single measurement during training.
type MetricPoint struct {
	Step      int       `json:"step"`
	Epoch     int       `json:"epoch"`
	Value     float64   `json:"value"`
	Timestamp time.Time `json:"timestamp"`
}

// CheckpointInfo describes a saved model checkpoint.
type CheckpointInfo struct {
	Path    string             `json:"path"`
	Step    int                `json:"step"`
	Epoch   int                `json:"epoch"`
	Metrics map[string]float64 `json:"metrics,omitempty"`
	IsBest  bool               `json:"is_best"`
}

// FineTuneRequest is a user's request to fine-tune an LLM for a coding skill.
type FineTuneRequest struct {
	ID               string               `json:"id"`
	Name             string               `json:"name"`
	Description      string               `json:"description,omitempty"`
	Language         ProgrammingLanguage  `json:"language"`
	Skill            SkillCategory        `json:"skill"`
	BaseModel        string               `json:"base_model"`
	DatasetID        string               `json:"dataset_id"`
	TrainingParams   TrainingParameters   `json:"training_params"`
	ValidationParams ValidationParameters `json:"validation_params,omitempty"`
	Owner            string               `json:"owner"`
	Status           RequestStatus        `json:"status"`
	CreatedAt        time.Time            `json:"created_at"`
	UpdatedAt        time.Time            `json:"updated_at"`
}

// FineTuneJob is a training job execution.
type FineTuneJob struct {
	ID             string                   `json:"id"`
	RequestID      string                   `json:"request_id"`
	Status         JobStatus                `json:"status"`
	Progress       float64                  `json:"progress"` // 0-100
	CurrentEpoch   int                      `json:"current_epoch"`
	TotalEpochs    int                      `json:"total_epochs"`
	CurrentStep    int                      `json:"current_step"`
	TotalSteps     int                      `json:"total_steps"`
	Loss           []MetricPoint            `json:"loss,omitempty"`
	ValidationLoss []MetricPoint            `json:"validation_loss,omitempty"`
	LearningRate   []MetricPoint            `json:"learning_rate,omitempty"`
	CustomMetrics  map[string][]MetricPoint `json:"custom_metrics,omitempty"`
	FinalMetrics   map[string]float64       `json:"final_metrics,omitempty"`
	Error          string                   `json:"error,omitempty"`
	BestCheckpoint CheckpointInfo           `json:"best_checkpoint,omitempty"`
	OutputModelID  string                   `json:"output_model_id,omitempty"`
	ComputeCost    float64                  `json:"compute_cost"`
	GPUHours       float64                  `json:"gpu_hours"`
	CreatedAt      time.Time                `json:"created_at"`
	StartedAt      *time.Time               `json:"started_at,omitempty"`
	CompletedAt    *time.Time               `json:"completed_at,omitempty"`
}

// TrainedModel is the final output of a fine-tuning job.
type TrainedModel struct {
	ID              string              `json:"id"`
	Name            string              `json:"name"`
	Language        ProgrammingLanguage `json:"language"`
	Skill           SkillCategory       `json:"skill"`
	BaseModel       string              `json:"base_model"`
	FineTuneJobID   string              `json:"finetune_job_id"`
	Version         string              `json:"version"`
	Status          string              `json:"status"` // "ready", "archived".
	Checkpoint      string              `json:"checkpoint"`
	HuggingFaceURL  string              `json:"huggingface_url,omitempty"`
	Quantized       bool                `json:"quantized"`
	QuantBits       int                 `json:"quant_bits,omitempty"`
	TrainingMetrics map[string]float64  `json:"training_metrics,omitempty"`
	BenchmarkScore  float64             `json:"benchmark_score"`
	Owner           string              `json:"owner"`
	IsPublic        bool                `json:"is_public"`
	Downloads       int                 `json:"downloads"`
	Rating          float64             `json:"rating"`
	CreatedAt       time.Time           `json:"created_at"`
}

// DatasetQuality describes the quality assessment of a code dataset.
type DatasetQuality struct {
	OverallScore    float64  `json:"overall_score"` // 0-100
	ValidityRate    float64  `json:"validity_rate"` // 0-1
	ComplexityScore float64  `json:"complexity_score"`
	Issues          []string `json:"issues,omitempty"`
}

// DatasetInfo is metadata for a code dataset used in fine-tuning.
type DatasetInfo struct {
	ID          string              `json:"id"`
	Name        string              `json:"name"`
	Language    ProgrammingLanguage `json:"language"`
	FileCount   int                 `json:"file_count"`
	TotalSize   int64               `json:"total_size"`
	TotalLines  int                 `json:"total_lines"`
	TotalTokens int64               `json:"total_tokens"`
	SampleCount int                 `json:"sample_count"`
	Quality     DatasetQuality      `json:"quality"`
	Status      string              `json:"status"` // "validated", "needs_review", "invalid".
	Owner       string              `json:"owner"`
	CreatedAt   time.Time           `json:"created_at"`
}

// --- Repository interfaces ---.

// FineTuneRequestRepository persists fine-tune requests.
type FineTuneRequestRepository interface {
	Create(r *FineTuneRequest) error
	Get(id string) (*FineTuneRequest, error)
	ListByOwner(owner string) ([]*FineTuneRequest, error)
	List() ([]*FineTuneRequest, error)
	Update(r *FineTuneRequest) error
	Delete(id string) error
}

// FineTuneJobRepository persists fine-tuning jobs and their metrics.
//
//nolint:interfacebloat // the repository API is intentionally granular
type FineTuneJobRepository interface {
	Create(job *FineTuneJob) error
	Get(id string) (*FineTuneJob, error)
	GetByRequestID(requestID string) (*FineTuneJob, error)
	List() ([]*FineTuneJob, error)
	ListActive() ([]*FineTuneJob, error)
	Update(job *FineTuneJob) error
	UpdateStatus(jobID string, status JobStatus) error
	UpdateProgress(jobID string, progress float64, epoch, step int) error
	AddMetric(jobID string, name string, point MetricPoint) error
	SetFinalMetrics(jobID string, metrics map[string]float64) error
	SetError(jobID string, errMsg string) error
}

// TrainedModelRepository persists trained models.
type TrainedModelRepository interface {
	Create(m *TrainedModel) error
	Get(id string) (*TrainedModel, error)
	GetByJobID(jobID string) (*TrainedModel, error)
	List() ([]*TrainedModel, error)
	ListByLanguage(lang ProgrammingLanguage) ([]*TrainedModel, error)
	ListBySkill(skill SkillCategory) ([]*TrainedModel, error)
	Update(m *TrainedModel) error
}

// DatasetRepository persists code dataset metadata.
type DatasetRepository interface {
	Create(d *DatasetInfo) error
	Get(id string) (*DatasetInfo, error)
	ListByOwner(owner string) ([]*DatasetInfo, error)
	List() ([]*DatasetInfo, error)
	UpdateQuality(datasetID string, q DatasetQuality) error
	Delete(id string) error
}

// --- Coding AI Agent orchestration ---.

// DatasetAnalysis is the AI agent's assessment of a dataset.
type DatasetAnalysis struct {
	FileCount          int            `json:"file_count"`
	TotalSize          int64          `json:"total_size"`
	TotalTokens        int64          `json:"total_tokens"`
	LanguageCoverage   map[string]int `json:"language_coverage,omitempty"`
	ComplexityRange    [2]int         `json:"complexity_range"`
	AverageComplexity  float64        `json:"average_complexity"`
	SyntaxValidityRate float64        `json:"syntax_validity_rate"`
	Recommendations    []string       `json:"recommendations,omitempty"`
	ReadyForTraining   bool           `json:"ready_for_training"`
	QualityScore       float64        `json:"quality_score"`
}

// HyperparameterRecommendationReq is input for the AI agent's hyperparameter recommendation.
type HyperparameterRecommendationReq struct {
	Language     ProgrammingLanguage
	Skill        SkillCategory
	DatasetSize  int64
	AvailableGPU int
}

// OptimizationAction is a suggested action from the training monitor.
type OptimizationAction string

const (
	ActionContinue      OptimizationAction = "continue"
	ActionReduceLR      OptimizationAction = "reduce_learning_rate"
	ActionIncreaseBatch OptimizationAction = "increase_batch_size"
	ActionEarlyStop     OptimizationAction = "early_stop"
)

// TrainingOptimization is the AI agent's live training suggestion.
type TrainingOptimization struct {
	CurrentStep   int                `json:"current_step"`
	CurrentLoss   float64            `json:"current_loss"`
	LossDirection string             `json:"loss_direction"` // "decreasing", "stable", "increasing".
	Suggestion    string             `json:"suggestion"`
	Action        OptimizationAction `json:"action"`
	Confidence    float64            `json:"confidence"` // 0-100
}

// QualityReport is the AI agent's post-training quality assessment.
type QualityReport struct {
	TrainingComplete  bool               `json:"training_complete"`
	FinalLoss         float64            `json:"final_loss"`
	BestMetrics       map[string]float64 `json:"best_metrics,omitempty"`
	CodeExecutability float64            `json:"code_executability"`
	SyntaxValidity    float64            `json:"syntax_validity"`
	OverallQuality    string             `json:"overall_quality"` // "excellent", "good", "fair", "poor".
	Issues            []string           `json:"issues,omitempty"`
	Recommendations   []string           `json:"recommendations,omitempty"`
}

// AgentInsights is a summary of the AI agent's analysis of a training run.
type AgentInsights struct {
	Phase            string                 `json:"phase"` // "analysis", "training", "optimization", "completion".
	Summary          string                 `json:"summary"`
	Metrics          map[string]interface{} `json:"metrics,omitempty"`
	Warnings         []string               `json:"warnings,omitempty"`
	NextSteps        []string               `json:"next_steps,omitempty"`
	EstimatedQuality string                 `json:"estimated_quality"`
}

// CodingAIAgent orchestrates the fine-tuning workflow with AI-driven insight.
type CodingAIAgent interface {
	AnalyzeDataset(ctx context.Context, datasetID string) (DatasetAnalysis, error)
	RecommendHyperparameters(ctx context.Context, req HyperparameterRecommendationReq) (TrainingParameters, error)
	StartTraining(ctx context.Context, req *FineTuneRequest) (string, error)
	MonitorTraining(ctx context.Context, jobID string) (TrainingOptimization, error)
	ValidateTrainingQuality(ctx context.Context, jobID string) (QualityReport, error)
	GetInsights(ctx context.Context, jobID string) (AgentInsights, error)
}
