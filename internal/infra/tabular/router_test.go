package tabular_test

import (
	"errors"
	"testing"

	"distillery/internal/domain"
	"distillery/internal/infra/tabular"
)

var (
	errNotMine = errors.New("not mine")
	errTableNo = errors.New("table says no")
	errBaseNo  = errors.New("base says no")
)

type fakeTuner struct {
	name    string
	started []string
	calls   []string
	err     error
}

func (f *fakeTuner) Start(job *domain.TrainingJob, _ []*domain.Example, _ func(int), _ func(*domain.TrainingMetrics, error)) {
	f.started = append(f.started, job.ID)
}

func (f *fakeTuner) Pause(id string) error  { f.calls = append(f.calls, "pause:"+id); return f.err }
func (f *fakeTuner) Resume(id string) error { f.calls = append(f.calls, "resume:"+id); return f.err }
func (f *fakeTuner) Cancel(id string) error { f.calls = append(f.calls, "cancel:"+id); return f.err }

// plainTuner has no Pause/Resume/Cancel (like the simulation trainer).
type plainTuner struct{ started int }

func (p *plainTuner) Start(*domain.TrainingJob, []*domain.Example, func(int), func(*domain.TrainingMetrics, error)) {
	p.started++
}

func TestRouterRoutesByKind(t *testing.T) {
	t.Parallel()

	base, table := &fakeTuner{name: "base"}, &fakeTuner{name: "table"}
	r := tabular.NewRouter(base, table)

	for _, k := range []domain.ModelKind{domain.KindTabular, domain.KindTimeSeries} {
		r.Start(&domain.TrainingJob{ID: "tab-" + string(k), Kind: k}, nil, nil, nil)
	}

	for _, k := range []domain.ModelKind{domain.KindCausalLM, domain.KindASR, domain.KindEmbedding} {
		r.Start(&domain.TrainingJob{ID: "llm-" + string(k), Kind: k}, nil, nil, nil)
	}

	if len(table.started) != 2 || len(base.started) != 3 {
		t.Fatalf("table got %v, base got %v", table.started, base.started)
	}
}

func TestRouterControlForwarding(t *testing.T) {
	t.Parallel()

	base, table := &fakeTuner{}, &fakeTuner{err: errNotMine}
	r := tabular.NewRouter(base, table)

	// The table trainer rejects the job, the base trainer accepts it.
	err := r.Pause("j1")
	if err != nil {
		t.Fatalf("pause should succeed via the base trainer: %v", err)
	}

	if len(table.calls) != 1 || len(base.calls) != 1 || base.calls[0] != "pause:j1" {
		t.Errorf("table=%v base=%v", table.calls, base.calls)
	}

	err = r.Resume("j1")
	if err != nil {
		t.Errorf("resume: %v", err)
	}

	err = r.Cancel("j1")
	if err != nil {
		t.Errorf("cancel: %v", err)
	}
}

func TestRouterControlReportsFirstErrorWhenNobodyOwnsTheJob(t *testing.T) {
	t.Parallel()

	first, second := errTableNo, errBaseNo
	r := tabular.NewRouter(&fakeTuner{err: second}, &fakeTuner{err: first})

	err := r.Cancel("j1")
	if !errors.Is(err, first) {
		t.Fatalf("want the table trainer's error first, got %v", err)
	}
}

func TestRouterControlIgnoresTrainersWithoutControls(t *testing.T) {
	t.Parallel()

	plain := &plainTuner{}
	r := tabular.NewRouter(plain, plain)

	err := r.Pause("j1")
	if err != nil {
		t.Errorf("no controllable trainer means nothing to fail: %v", err)
	}

	r.Start(&domain.TrainingJob{ID: "a", Kind: domain.KindTabular}, nil, nil, nil)
	r.Start(&domain.TrainingJob{ID: "b", Kind: domain.KindASR}, nil, nil, nil)

	if plain.started != 2 {
		t.Errorf("started=%d", plain.started)
	}
}
