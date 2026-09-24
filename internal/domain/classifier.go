package domain

// ClassifierConfig carries the sequence-classification hyperparameters that
// the Go layer can pass to the trainer worker (mirrored by the Python
// ClassifierConfig in trainer/tasks/classifier.py).
type ClassifierConfig struct {
	// MaxLength truncates/pads inputs to this many tokens.
	MaxLength int `json:"max_length,omitempty"`
	// MultiLabel enables multi-label (sigmoid + per-class threshold) training.
	MultiLabel bool `json:"multi_label,omitempty"`
	// ClassWeights enables inverse-frequency loss weighting when the class
	// imbalance ratio exceeds ~3:1.
	ClassWeights bool `json:"class_weights,omitempty"`
	// Epochs / LearningRate / BatchSize override the defaults.
	Epochs       int     `json:"epochs,omitempty"`
	LearningRate float64 `json:"learning_rate,omitempty"`
	BatchSize    int     `json:"batch_size,omitempty"`
	// Threshold is the default confidence threshold under which predictions
	// are flagged as below_threshold (review queue).
	Threshold float64 `json:"threshold,omitempty"`
}

// ConfusionMatrix is a row-major C x C matrix of true-label -> predicted-label
// counts over the held-out eval split. Row/column order matches LabelMap order.
type ConfusionMatrix [][]int

// ThresholdSweepPoint reports macro-F1 and coverage at a single confidence
// threshold, used by the UI's precision/recall trade-off slider.
type ThresholdSweepPoint struct {
	Threshold float64 `json:"threshold"`
	MacroF1   float64 `json:"macro_f1"`
	Coverage  float64 `json:"coverage"`
}

// PerClassMetrics are the precision/recall/F1/support for one label.
type PerClassMetrics struct {
	Precision float64 `json:"precision"`
	Recall    float64 `json:"recall"`
	F1        float64 `json:"f1"`
	Support   int     `json:"support"`
}

// ClassifierPrediction is the structured result of a seq_classifier inference
// call (what /predict returns for seq_classifier deployments).
type ClassifierPrediction struct {
	Label          string             `json:"label"`
	Score          float64            `json:"score"`
	Scores         map[string]float64 `json:"scores"`
	BelowThreshold bool               `json:"below_threshold"`
	// MultiLabel holds per-label booleans when the model is multi-label.
	MultiLabel map[string]bool `json:"multi_label,omitempty"`
}

// ClassifierInferenceEngine serves sequence-classification predictions from a
// deployed fine-tuned model, returning label + per-class scores. Implementations
// must be kind-safe: callers guard on Deployment.Kind == KindSeqClassifier.
type ClassifierInferenceEngine interface {
	PredictClassification(
		job *TrainingJob,
		trainingExamples []*Example,
		input string,
		labelMap map[string]int,
		threshold float64,
	) (ClassifierPrediction, error)
}
