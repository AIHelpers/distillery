package usecase_test

import (
	"errors"
	"testing"

	"distillery/internal/domain"
	"distillery/internal/usecase"
)

// newPreferenceExamples returns a mixed set of causal_lm (legacy, Kind=="")
// and preference_lm examples for task_1, so tests can verify startTraining
// only feeds the worker examples of the kind it's training.
func newPreferenceExamples(prefPayloads []string) []*domain.Example {
	const taskID = "task_1"

	out := make([]*domain.Example, 0, 3+len(prefPayloads))

	for range 3 {
		out = append(out, &domain.Example{ID: "sft_ex", TaskID: taskID, Input: "in", Output: "out"})
	}

	for _, p := range prefPayloads {
		out = append(out, &domain.Example{
			ID:      "pref_ex",
			TaskID:  taskID,
			Kind:    domain.KindPreferenceLM,
			Payload: []byte(p),
		})
	}

	return out
}

func TestStartTrainingWithOptions_DPO_RequiresParent(t *testing.T) {
	t.Parallel()

	task := &domain.Task{ID: "task_1", Kind: domain.KindCausalLM}
	examples := &mockExampleRepo{byTask: newPreferenceExamples([]string{
		`{"prompt":"p1","chosen":"good","rejected":"bad"}`,
		`{"prompt":"p2","chosen":"good2","rejected":"bad2"}`,
		`{"prompt":"p3","chosen":"good3","rejected":"bad3"}`,
	})}
	jobs := &mockTrainingRepo{}
	selector := &mockModelSelector{model: domain.BaseModel{Name: "Qwen3-1.7B"}}
	tuner := &mockFineTuner{}

	uc := usecase.NewTrainingUsecase(&mockTaskRepo{task: task}, examples, jobs, selector, tuner, &mockIDGen{})

	_, err := uc.StartTrainingWithOptions("task_1", usecase.TrainingStartOptions{Method: "dpo"})
	if !errors.Is(err, domain.ErrInvalidInput) {
		t.Fatalf("expected ErrInvalidInput (dpo needs a parent), got %v", err)
	}
}

func TestStartTrainingWithOptions_DPO_RejectsNonCausalLMParent(t *testing.T) {
	t.Parallel()

	task := &domain.Task{ID: "task_1", Kind: domain.KindCausalLM}
	parent := &domain.TrainingJob{ID: "job_parent", TaskID: "task_1", Kind: domain.KindSeqClassifier, Status: domain.TrainingCompleted}
	examples := &mockExampleRepo{byTask: newPreferenceExamples([]string{
		`{"prompt":"p1","chosen":"good","rejected":"bad"}`,
		`{"prompt":"p2","chosen":"good2","rejected":"bad2"}`,
		`{"prompt":"p3","chosen":"good3","rejected":"bad3"}`,
	})}
	jobs := &mockTrainingRepo{job: parent, jobs: []*domain.TrainingJob{parent}}
	selector := &mockModelSelector{model: domain.BaseModel{Name: "Qwen3-1.7B"}}
	tuner := &mockFineTuner{}

	uc := usecase.NewTrainingUsecase(&mockTaskRepo{task: task}, examples, jobs, selector, tuner, &mockIDGen{})

	_, err := uc.StartTrainingWithOptions("task_1", usecase.TrainingStartOptions{Method: "dpo", ParentJobID: "job_parent"})
	if !errors.Is(err, domain.ErrInvalidInput) {
		t.Fatalf("expected ErrInvalidInput (parent must be causal_lm+completed), got %v", err)
	}
}

func TestStartTrainingWithOptions_DPO_Success(t *testing.T) {
	t.Parallel()

	task := &domain.Task{ID: "task_1", Kind: domain.KindCausalLM}
	parentBase := domain.BaseModel{Name: "Qwen3-1.7B", RepoID: "Qwen/Qwen3-1.7B"}
	parent := &domain.TrainingJob{
		ID: "job_parent", TaskID: "task_1", Version: 1,
		Kind: domain.KindCausalLM, Status: domain.TrainingCompleted, BaseModel: parentBase,
	}
	examples := &mockExampleRepo{byTask: newPreferenceExamples([]string{
		`{"prompt":"p1","chosen":"good","rejected":"bad"}`,
		`{"prompt":"p2","chosen":"good2","rejected":"bad2"}`,
		`{"prompt":"p3","chosen":"good3","rejected":"bad3"}`,
	})}
	jobs := &mockTrainingRepo{job: parent, jobs: []*domain.TrainingJob{parent}}
	// A different catalog model than the parent's, to assert the parent's
	// base model wins over auto-selection/explicit choice for DPO.
	selector := &mockModelSelector{model: domain.BaseModel{Name: "Some-Other-Model"}}
	tuner := &mockFineTuner{}

	uc := usecase.NewTrainingUsecase(&mockTaskRepo{task: task}, examples, jobs, selector, tuner, &mockIDGen{})

	job, err := uc.StartTrainingWithOptions("task_1", usecase.TrainingStartOptions{
		Method: "dpo", ParentJobID: "job_parent", BaseModel: "Some-Other-Model",
	})
	if err != nil {
		t.Fatalf("expected success, got %v", err)
	}

	if job.Kind != domain.KindPreferenceLM {
		t.Fatalf("expected job.Kind == preference_lm, got %v", job.Kind)
	}

	if job.ParentJobID != "job_parent" {
		t.Fatalf("expected ParentJobID == job_parent, got %q", job.ParentJobID)
	}

	if job.BaseModel.Name != "Qwen3-1.7B" {
		t.Fatalf("expected DPO to inherit the parent's base model, got %q", job.BaseModel.Name)
	}

	if job.Preference == nil || job.Preference.Method != domain.MethodDPO {
		t.Fatalf("expected Preference.Method == dpo, got %+v", job.Preference)
	}

	if !tuner.started {
		t.Fatal("expected the tuner to be started")
	}

	if len(tuner.examples) != 3 {
		t.Fatalf("expected only the 3 preference-kind examples to be fed to the trainer, got %d", len(tuner.examples))
	}

	for _, e := range tuner.examples {
		if e.Kind != domain.KindPreferenceLM {
			t.Fatalf("expected only preference_lm examples, got kind %q", e.Kind)
		}
	}
}

func TestStartTrainingWithOptions_ORPO_NoParentRequired(t *testing.T) {
	t.Parallel()

	task := &domain.Task{ID: "task_1", Kind: domain.KindCausalLM}
	examples := &mockExampleRepo{byTask: newPreferenceExamples([]string{
		`{"prompt":"p1","chosen":"good","rejected":"bad"}`,
		`{"prompt":"p2","chosen":"good2","rejected":"bad2"}`,
		`{"prompt":"p3","chosen":"good3","rejected":"bad3"}`,
	})}
	jobs := &mockTrainingRepo{}
	selector := &mockModelSelector{model: domain.BaseModel{Name: "Qwen3-1.7B"}}
	tuner := &mockFineTuner{}

	uc := usecase.NewTrainingUsecase(&mockTaskRepo{task: task}, examples, jobs, selector, tuner, &mockIDGen{})

	job, err := uc.StartTrainingWithOptions("task_1", usecase.TrainingStartOptions{Method: "orpo"})
	if err != nil {
		t.Fatalf("expected orpo to succeed without a parent, got %v", err)
	}

	if job.Kind != domain.KindPreferenceLM || job.Preference.Method != domain.MethodORPO {
		t.Fatalf("expected a preference_lm/orpo job, got kind=%v preference=%+v", job.Kind, job.Preference)
	}

	if job.ParentJobID != "" {
		t.Fatalf("expected no parent job for orpo, got %q", job.ParentJobID)
	}
}

func TestStartTrainingWithOptions_InvalidMethod(t *testing.T) {
	t.Parallel()

	task := &domain.Task{ID: "task_1", Kind: domain.KindCausalLM}
	examples := &mockExampleRepo{byTask: newPreferenceExamples(nil)}
	uc := usecase.NewTrainingUsecase(
		&mockTaskRepo{task: task}, examples, &mockTrainingRepo{},
		&mockModelSelector{model: domain.BaseModel{Name: "Qwen3-1.7B"}}, &mockFineTuner{}, &mockIDGen{},
	)

	_, err := uc.StartTrainingWithOptions("task_1", usecase.TrainingStartOptions{Method: "ppo"})
	if !errors.Is(err, domain.ErrInvalidInput) {
		t.Fatalf("expected ErrInvalidInput for an unsupported method, got %v", err)
	}
}

func TestStartTraining_Backcompat_UnaffectedByPreferenceOptions(t *testing.T) {
	t.Parallel()

	// A plain causal_lm task with only legacy (Kind=="") examples must keep
	// training as an ordinary SFT run through the original StartTraining
	// entry point — no method/parent involved.
	task := &domain.Task{ID: "task_1", Kind: domain.KindCausalLM}
	examples := &mockExampleRepo{byTask: newPreferenceExamples(nil)}
	uc := usecase.NewTrainingUsecase(
		&mockTaskRepo{task: task}, examples, &mockTrainingRepo{},
		&mockModelSelector{model: domain.BaseModel{Name: "Qwen3-1.7B"}}, &mockFineTuner{}, &mockIDGen{},
	)

	job, err := uc.StartTraining("task_1")
	if err != nil {
		t.Fatalf("expected success, got %v", err)
	}

	if job.Kind != domain.KindCausalLM {
		t.Fatalf("expected an ordinary causal_lm job, got %v", job.Kind)
	}

	if job.Preference != nil {
		t.Fatalf("expected no Preference config on an ordinary SFT job, got %+v", job.Preference)
	}
}
