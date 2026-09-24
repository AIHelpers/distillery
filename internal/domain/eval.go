package domain

// EvalReport captures a held-out evaluation comparing the fine-tuned model
// against the base model on the same split, so the UI can say "+X over base".
type EvalReport struct {
	// Kind is the model kind the report was produced for.
	Kind ModelKind `json:"kind"`
	// Primary is the single headline metric the UI emphasises.
	Primary MetricValue `json:"primary"`
	// Metrics is the full set of reported metrics by name.
	Metrics map[string]float64 `json:"metrics,omitempty"`
	// Baseline is the same metrics evaluated on the base (un-fine-tuned)
	// model over the same held-out split.
	Baseline map[string]float64 `json:"baseline,omitempty"`
	// Artifacts are optional evaluation outputs (e.g. classification report
	// JSON, charts) keyed by name.
	Artifacts map[string]string `json:"artifacts,omitempty"`
}

// MetricValue is a named metric with its value.
type MetricValue struct {
	Name  string  `json:"name"`
	Value float64 `json:"value"`
}

// DeltaOverBaseline returns primary metric minus its baseline when both are
// present, plus a human-readable sign like "+0.12" or "-0.04".
func (e *EvalReport) DeltaOverBaseline() (float64, bool) {
	if e == nil || e.Baseline == nil {
		return 0, false
	}

	base, ok := e.Baseline[e.Primary.Name]
	if !ok {
		return 0, false
	}

	return e.Primary.Value - base, true
}
