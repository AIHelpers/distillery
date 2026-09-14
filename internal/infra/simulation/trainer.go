package simulation

import (
	"math/rand"
	"time"

	"distillery/internal/domain"
)

// FineTuner implements domain.FineTuner. It simulates a LoRA/QLoRA training
// run (progress ticks + realistic-looking loss/accuracy curves) on a
// background goroutine, standing in for the real GPU orchestration
// (RunPod/Lambda/Modal + HF peft/trl) described in the product spec.
type FineTuner struct {
	// TickInterval controls simulated training speed. Kept short so the
	// UI demo doesn't require a real GPU cluster.
	TickInterval time.Duration
}

func NewFineTuner() *FineTuner {
	return &FineTuner{TickInterval: 400 * time.Millisecond}
}

func (f *FineTuner) Start(
	_ *domain.TrainingJob,
	examples []*domain.Example,
	onUpdate func(progress int,
	), onDone func(metrics *domain.TrainingMetrics, err error),
) {
	go func() {
		r := rand.New(rand.NewSource(time.Now().UnixNano()))

		// Simulate failure on a pathologically small dataset.
		if len(examples) < 3 {
			time.Sleep(f.TickInterval)
			onDone(nil, errTooFewExamples)

			return
		}

		steps := 20
		for step := 1; step <= steps; step++ {
			time.Sleep(f.TickInterval)

			progress := step * 100 / steps
			onUpdate(progress)
		}

		// Loss/accuracy curve loosely improves with more examples, with noise.
		datasetFactor := float64(len(examples))
		if datasetFactor > 500 {
			datasetFactor = 500
		}

		baseLoss := 1.8 - (datasetFactor/500.0)*1.3

		loss := baseLoss + r.Float64()*0.1
		if loss < 0.05 {
			loss = 0.05
		}

		acc := 0.55 + (datasetFactor/500.0)*0.4 + r.Float64()*0.05
		if acc > 0.99 {
			acc = 0.99
		}

		metrics := &domain.TrainingMetrics{
			FinalLoss:     round2(loss),
			EvalAccuracy:  round2(acc),
			Epochs:        3,
			TrainExamples: len(examples),
		}
		onDone(metrics, nil)
	}()
}

func round2(v float64) float64 {
	return float64(int(v*100)) / 100
}

var errTooFewExamples = &fineTuneError{"need at least 3 usable examples to train"}

type fineTuneError struct{ msg string }

func (e *fineTuneError) Error() string { return e.msg }
