// Package tabular wires the plan-08 tabular / time-series kinds into the
// infrastructure layer: a FineTuner router, a serving engine backed by a
// long-lived Python model worker, and the portable model export.
package tabular

import (
	"distillery/internal/domain"
)

// Router is a domain.FineTuner that sends tabular and time_series jobs to the
// table trainer and everything else to the LLM trainer.
type Router struct {
	base  domain.FineTuner
	table domain.FineTuner
}

// NewRouter builds the router.
func NewRouter(base, table domain.FineTuner) *Router {
	return &Router{base: base, table: table}
}

// Start implements domain.FineTuner.
func (r *Router) Start(
	job *domain.TrainingJob,
	examples []*domain.Example,
	onUpdate func(progress int),
	onDone func(metrics *domain.TrainingMetrics, err error),
) {
	if job.Kind == domain.KindTabular || job.Kind == domain.KindTimeSeries {
		r.table.Start(job, examples, onUpdate, onDone)

		return
	}

	r.base.Start(job, examples, onUpdate, onDone)
}

type controller interface {
	Pause(jobID string) error
	Resume(jobID string) error
	Cancel(jobID string) error
}

// Pause forwards to whichever trainer owns the job.
func (r *Router) Pause(jobID string) error {
	return r.each(func(c controller) error { return c.Pause(jobID) })
}

// Resume forwards to whichever trainer owns the job.
func (r *Router) Resume(jobID string) error {
	return r.each(func(c controller) error { return c.Resume(jobID) })
}

// Cancel forwards to whichever trainer owns the job.
func (r *Router) Cancel(jobID string) error {
	return r.each(func(c controller) error { return c.Cancel(jobID) })
}

func (r *Router) each(f func(c controller) error) error {
	var first error

	for _, t := range []domain.FineTuner{r.table, r.base} {
		c, ok := t.(controller)
		if !ok {
			continue
		}

		err := f(c)
		if err == nil {
			return nil
		}

		if first == nil {
			first = err
		}
	}

	return first
}
