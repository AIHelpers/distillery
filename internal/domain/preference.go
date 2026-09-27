package domain

// PreferenceMethod selects the preference-tuning algorithm.
type PreferenceMethod string

const (
	// MethodDPO is Direct Preference Optimization: requires a parent SFT
	// job whose adapter is the starting policy (and, with LoRA, the
	// implicit reference model — the same base with the adapter disabled).
	MethodDPO PreferenceMethod = "dpo"
	// MethodORPO is Odds-Ratio Preference Optimization: reference-free, so
	// it is the only method allowed to start from a base model with no
	// prior SFT job.
	MethodORPO PreferenceMethod = "orpo"
)

// IsValidPreferenceMethod reports whether m is a supported preference method.
func IsValidPreferenceMethod(m PreferenceMethod) bool {
	return m == MethodDPO || m == MethodORPO
}

// PreferenceConfig carries the DPO/ORPO hyperparameters that the Go layer
// passes to the trainer worker (mirrored by the Python Config in
// trainer/tasks/dpo.py).
type PreferenceConfig struct {
	// Method selects "dpo" or "orpo".
	Method PreferenceMethod `json:"method,omitempty"`
	// Beta is the DPO/ORPO temperature controlling how sharply the policy
	// is pulled away from the reference (lower = more conservative).
	Beta float64 `json:"beta,omitempty"`
	// LearningRate overrides the (very low, by design) preference-tuning LR.
	LearningRate float64 `json:"learning_rate,omitempty"`
	// Epochs overrides the default epoch count (1-3 is typical).
	Epochs int `json:"epochs,omitempty"`
	// MaxPromptLen / MaxLen cap tokenized prompt and prompt+completion
	// length; preference examples carry two completions per prompt so
	// these matter more for VRAM than plain SFT.
	MaxPromptLen int `json:"max_prompt_len,omitempty"`
	MaxLen       int `json:"max_len,omitempty"`
}

// DefaultPreferenceConfig returns the plan's recommended defaults: a
// conservative beta, a very low learning rate, and few epochs, so an
// over-eager DPO run doesn't drift the model away from its SFT behavior.
func DefaultPreferenceConfig() PreferenceConfig {
	return PreferenceConfig{
		Method:       MethodDPO,
		Beta:         0.1,
		LearningRate: 5e-6,
		Epochs:       2,
		MaxPromptLen: 512,
		MaxLen:       1024,
	}
}

// ApplyDefaults fills zero-valued fields of a partially-specified
// PreferenceConfig with DefaultPreferenceConfig's values, preserving the
// caller's Method.
func (c PreferenceConfig) ApplyDefaults() PreferenceConfig {
	d := DefaultPreferenceConfig()

	if c.Method == "" {
		c.Method = d.Method
	}

	if c.Beta <= 0 {
		c.Beta = d.Beta
	}

	if c.LearningRate <= 0 {
		c.LearningRate = d.LearningRate
	}

	if c.Epochs <= 0 {
		c.Epochs = d.Epochs
	}

	if c.MaxPromptLen <= 0 {
		c.MaxPromptLen = d.MaxPromptLen
	}

	if c.MaxLen <= 0 {
		c.MaxLen = d.MaxLen
	}

	return c
}

// MinPreferencePairs is the minimum number of usable preference pairs the
// UI warns below (risk #3 in the plan: small datasets give unstable results).
const MinPreferencePairs = 200

// PreferencePairStats summarizes a task's preference (prompt/chosen/rejected)
// dataset: counts plus the length-bias diagnostic from the plan's risk list
// ("longer answers win" — report average output length before/after).
type PreferencePairStats struct {
	TaskID          string  `json:"task_id"`
	Total           int     `json:"total"`
	Duplicates      int     `json:"duplicates"`
	Flagged         int     `json:"flagged"`
	UsableCount     int     `json:"usable_count"`
	AvgChosenLen    float64 `json:"avg_chosen_len"`
	AvgRejectedLen  float64 `json:"avg_rejected_len"`
	ChosenLongerPct float64 `json:"chosen_longer_pct"` // fraction of pairs where chosen is the longer completion.
	ReadyToTrain    bool    `json:"ready_to_train"`
	ReadinessReason string  `json:"readiness_reason,omitempty"`
	// LengthBiasWarning is set when chosen is longer than rejected in a
	// large majority of pairs, which can teach the model "longer == better"
	// rather than the intended preference.
	LengthBiasWarning bool `json:"length_bias_warning,omitempty"`
}

// LengthBiasThreshold is the fraction of pairs favoring the longer chosen
// answer above which we surface the length-bias warning (plan risk #4:
// "longer answers win").
const LengthBiasThreshold = 0.85
