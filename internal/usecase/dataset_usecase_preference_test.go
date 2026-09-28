package usecase_test

import (
	"encoding/json"
	"testing"

	"distillery/internal/domain"
	"distillery/internal/usecase"
)

func newPreferenceDatasetUsecase(task *domain.Task) (*usecase.DatasetUsecase, *mockExampleRepo) {
	examples := &mockExampleRepo{}
	uc := usecase.NewDatasetUsecase(&mockTaskRepo{task: task}, examples, &mockSynthGen{}, &mockIDGen{})

	return uc, examples
}

func TestAddPreferencePair_Success(t *testing.T) {
	t.Parallel()

	task := &domain.Task{ID: "task_1", Kind: domain.KindCausalLM}
	uc, examples := newPreferenceDatasetUsecase(task)

	stats, err := uc.AddPreferencePair("task_1", "What color is the sky?", "Blue.", "I don't know.")
	if err != nil {
		t.Fatalf("expected success, got %v", err)
	}

	if len(examples.addBatch) != 1 {
		t.Fatalf("expected 1 example added, got %d", len(examples.addBatch))
	}

	ex := examples.addBatch[0]
	if ex.Kind != domain.KindPreferenceLM {
		t.Fatalf("expected Kind == preference_lm, got %v", ex.Kind)
	}

	var payload struct {
		Prompt   string `json:"prompt"`
		Chosen   string `json:"chosen"`
		Rejected string `json:"rejected"`
	}

	err = json.Unmarshal(ex.Payload, &payload)
	if err != nil {
		t.Fatalf("malformed payload: %v", err)
	}

	if payload.Prompt != "What color is the sky?" || payload.Chosen != "Blue." || payload.Rejected != "I don't know." {
		t.Fatalf("unexpected payload: %+v", payload)
	}

	if stats.Total != 1 || stats.UsableCount != 1 {
		t.Fatalf("expected stats to reflect the newly added pair, got %+v", stats)
	}
}

func TestAddPreferencePair_RejectsIdenticalChosenRejected(t *testing.T) {
	t.Parallel()

	task := &domain.Task{ID: "task_1", Kind: domain.KindCausalLM}
	uc, examples := newPreferenceDatasetUsecase(task)

	_, err := uc.AddPreferencePair("task_1", "prompt", "same answer", "same answer")
	if err == nil {
		t.Fatal("expected an error when chosen == rejected")
	}

	if len(examples.addBatch) != 0 {
		t.Fatalf("expected nothing to be added, got %d", len(examples.addBatch))
	}
}

func TestAddPreferencePair_RejectsEmptyFields(t *testing.T) {
	t.Parallel()

	task := &domain.Task{ID: "task_1", Kind: domain.KindCausalLM}
	uc, _ := newPreferenceDatasetUsecase(task)

	_, err := uc.AddPreferencePair("task_1", "prompt", "", "rejected")
	if err == nil {
		t.Fatal("expected an error when chosen is empty")
	}
}

func TestImportPreferenceJSONL_SkipsInvalidLines(t *testing.T) {
	t.Parallel()

	task := &domain.Task{ID: "task_1", Kind: domain.KindCausalLM}
	uc, examples := newPreferenceDatasetUsecase(task)

	content := `{"prompt":"p1","chosen":"good","rejected":"bad"}
{"prompt":"p2","chosen":"same","rejected":"same"}
not json
{"prompt":"","chosen":"c","rejected":"r"}
{"prompt":"p3","chosen":"c3","rejected":"r3"}`

	stats, err := uc.ImportPreferenceJSONL("task_1", content)
	if err != nil {
		t.Fatalf("expected success, got %v", err)
	}

	if len(examples.addBatch) != 2 {
		t.Fatalf("expected 2 valid records imported, got %d", len(examples.addBatch))
	}

	if stats.Total != 2 {
		t.Fatalf("expected stats.Total == 2, got %d", stats.Total)
	}
}

func TestImportPreferenceJSONL_AllInvalid(t *testing.T) {
	t.Parallel()

	task := &domain.Task{ID: "task_1", Kind: domain.KindCausalLM}
	uc, _ := newPreferenceDatasetUsecase(task)

	_, err := uc.ImportPreferenceJSONL("task_1", `{"prompt":"p","chosen":"x","rejected":"x"}`)
	if err == nil {
		t.Fatal("expected an error when no valid records are present")
	}
}

func TestPreferenceStats_DedupAndLengthBias(t *testing.T) {
	t.Parallel()

	task := &domain.Task{ID: "task_1", Kind: domain.KindCausalLM}
	examples := &mockExampleRepo{byTask: []*domain.Example{
		// A causal_lm example sharing the task — must be ignored entirely.
		{ID: "ex_sft", TaskID: "task_1", Input: "in", Output: "out"},
		{ID: "ex_1", TaskID: "task_1", Kind: domain.KindPreferenceLM, Payload: mustJSON(t, "p1", "aaaaaaaaaa", "b")},
		// Exact duplicate of ex_1 — should be flagged as a duplicate.
		{ID: "ex_2", TaskID: "task_1", Kind: domain.KindPreferenceLM, Payload: mustJSON(t, "p1", "aaaaaaaaaa", "b")},
		{ID: "ex_3", TaskID: "task_1", Kind: domain.KindPreferenceLM, Payload: mustJSON(t, "p2", "cccccccccc", "d")},
	}}

	uc := usecase.NewDatasetUsecase(&mockTaskRepo{task: task}, examples, &mockSynthGen{}, &mockIDGen{})

	stats, err := uc.PreferenceStats("task_1")
	if err != nil {
		t.Fatalf("expected success, got %v", err)
	}

	if stats.Total != 3 {
		t.Fatalf("expected 3 preference examples counted (causal_lm excluded), got %d", stats.Total)
	}

	if stats.Duplicates != 1 {
		t.Fatalf("expected 1 duplicate, got %d", stats.Duplicates)
	}

	if stats.UsableCount != 2 {
		t.Fatalf("expected 2 usable pairs, got %d", stats.UsableCount)
	}

	if !stats.LengthBiasWarning {
		t.Fatalf("expected a length-bias warning (chosen is longer in both usable pairs), got %+v", stats)
	}
}

func mustJSON(t *testing.T, prompt, chosen, rejected string) []byte {
	t.Helper()

	b, err := json.Marshal(map[string]string{"prompt": prompt, "chosen": chosen, "rejected": rejected})
	if err != nil {
		t.Fatalf("json.Marshal: %v", err)
	}

	return b
}
