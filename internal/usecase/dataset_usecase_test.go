package usecase_test

import (
	"errors"
	"testing"
	time "time"

	"distillery/internal/domain"
	"distillery/internal/usecase"
)

func newDatasetUsecase(tasks *mockTaskRepo, examples *mockExampleRepo, synthGen *mockSynthGen) *usecase.DatasetUsecase {
	return usecase.NewDatasetUsecase(tasks, examples, synthGen, newStubIDGen())
}

// --- AddExamples ---.

func TestDatasetUsecase_AddExamples_Valid(t *testing.T) {
	t.Parallel()

	tasks := &mockTaskRepo{task: &domain.Task{ID: "task_1", Type: domain.TaskClassification, Name: "", Description: "", CreatedAt: time.Time{}, UpdatedAt: time.Time{}}}
	examples := &mockExampleRepo{}
	uc := newDatasetUsecase(tasks, examples, &mockSynthGen{})

	stats, err := uc.AddExamples("task_1", []usecase.ExamplePair{
		{Input: "  hello  ", Output: "  world  "},
		{Input: "foo", Output: "bar"},
	})
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}

	if stats == nil {
		t.Fatal("expected non-nil stats")
	}

	if len(examples.addBatch) != 2 {
		t.Errorf("expected 2 examples in batch, got %d", len(examples.addBatch))
	}

	if examples.addBatch[0].Input != "hello" {
		t.Errorf("expected trimmed input 'hello', got %q", examples.addBatch[0].Input)
	}

	if examples.addBatch[0].Source != domain.SourceUser {
		t.Errorf("expected SourceUser, got %v", examples.addBatch[0].Source)
	}
}

func TestDatasetUsecase_AddExamples_TaskNotFound(t *testing.T) {
	t.Parallel()

	tasks := &mockTaskRepo{} // returns ErrNotFound.
	uc := newDatasetUsecase(tasks, &mockExampleRepo{}, &mockSynthGen{})

	_, err := uc.AddExamples("missing", []usecase.ExamplePair{{Input: "a", Output: "b"}})
	if !errors.Is(err, domain.ErrNotFound) {
		t.Errorf("expected ErrNotFound, got %v", err)
	}
}

func TestDatasetUsecase_AddExamples_SkipsEmpty(t *testing.T) {
	t.Parallel()

	tasks := &mockTaskRepo{task: &domain.Task{ID: "task_1", Type: domain.TaskGeneration, Name: "", Description: "", CreatedAt: time.Time{}, UpdatedAt: time.Time{}}}
	uc := newDatasetUsecase(tasks, &mockExampleRepo{}, &mockSynthGen{})

	_, err := uc.AddExamples("task_1", []usecase.ExamplePair{
		{Input: "", Output: "b"},
		{Input: "a", Output: ""},
		{Input: "   ", Output: "   "},
	})
	if !errors.Is(err, domain.ErrInvalidInput) {
		t.Errorf("expected ErrInvalidInput when all pairs empty, got %v", err)
	}
}

func TestDatasetUsecase_AddExamples_RepoError(t *testing.T) {
	t.Parallel()

	tasks := &mockTaskRepo{task: &domain.Task{ID: "task_1", Name: "", Description: "", Type: "", CreatedAt: time.Time{}, UpdatedAt: time.Time{}}}
	examples := &mockExampleRepo{err: errRepoFailure}
	uc := newDatasetUsecase(tasks, examples, &mockSynthGen{})

	_, err := uc.AddExamples("task_1", []usecase.ExamplePair{{Input: "a", Output: "b"}})
	if !errors.Is(err, errRepoFailure) {
		t.Errorf("expected errRepoFailure, got %v", err)
	}
}

// --- GenerateSynthetic ---.

func TestDatasetUsecase_GenerateSynthetic_Valid(t *testing.T) {
	t.Parallel()

	tasks := &mockTaskRepo{task: &domain.Task{ID: "task_1", Name: "", Description: "", Type: "", CreatedAt: time.Time{}, UpdatedAt: time.Time{}}}
	examples := &mockExampleRepo{
		byTask: []*domain.Example{
			{ID: "ex_1", Source: domain.SourceUser, Input: "seed", Output: "out"},
		},
	}
	sg := &mockSynthGen{generated: []*domain.Example{{Input: "synth", Output: "sout", Source: domain.SourceSynthetic}}}
	uc := newDatasetUsecase(tasks, examples, sg)

	stats, err := uc.GenerateSynthetic("task_1", 5)
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}

	if stats == nil {
		t.Fatal("expected non-nil stats")
	}

	if len(examples.addBatch) != 1 {
		t.Errorf("expected 1 synthetic example added, got %d", len(examples.addBatch))
	}

	if examples.addBatch[0].Source != domain.SourceSynthetic {
		t.Errorf("expected SourceSynthetic, got %v", examples.addBatch[0].Source)
	}
}

func TestDatasetUsecase_GenerateSynthetic_NoSeed(t *testing.T) {
	t.Parallel()

	tasks := &mockTaskRepo{task: &domain.Task{ID: "task_1", Name: "", Description: "", Type: "", CreatedAt: time.Time{}, UpdatedAt: time.Time{}}}
	examples := &mockExampleRepo{byTask: []*domain.Example{
		{Source: domain.SourceSynthetic, Input: "a", Output: "b"},
	}}
	uc := newDatasetUsecase(tasks, examples, &mockSynthGen{})

	_, err := uc.GenerateSynthetic("task_1", 5)
	if !errors.Is(err, domain.ErrInvalidInput) {
		t.Errorf("expected ErrInvalidInput when no user seed, got %v", err)
	}
}

func TestDatasetUsecase_GenerateSynthetic_TaskNotFound(t *testing.T) {
	t.Parallel()

	uc := newDatasetUsecase(&mockTaskRepo{}, &mockExampleRepo{}, &mockSynthGen{})

	_, err := uc.GenerateSynthetic("missing", 5)
	if !errors.Is(err, domain.ErrNotFound) {
		t.Errorf("expected ErrNotFound, got %v", err)
	}
}

// --- ListExamples ---.

func TestDatasetUsecase_ListExamples(t *testing.T) {
	t.Parallel()

	examples := &mockExampleRepo{byTask: []*domain.Example{{ID: "ex_1"}, {ID: "ex_2"}}}
	uc := newDatasetUsecase(&mockTaskRepo{}, examples, &mockSynthGen{})

	list, err := uc.ListExamples("task_1")
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}

	if len(list) != 2 {
		t.Errorf("expected 2 examples, got %d", len(list))
	}
}

// --- ImportCSV ---.

func TestDatasetUsecase_ImportCSV_WithHeader(t *testing.T) {
	t.Parallel()

	tasks := &mockTaskRepo{task: &domain.Task{ID: "task_1", Type: domain.TaskGeneration, Name: "", Description: "", CreatedAt: time.Time{}, UpdatedAt: time.Time{}}}
	uc := newDatasetUsecase(tasks, &mockExampleRepo{}, &mockSynthGen{})

	content := "input,output\nhello,world\nfoo,bar\n"

	_, err := uc.ImportCSV("task_1", content)
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}
}

func TestDatasetUsecase_ImportCSV_NoHeader(t *testing.T) {
	t.Parallel()

	tasks := &mockTaskRepo{task: &domain.Task{ID: "task_1", Type: domain.TaskGeneration, Name: "", Description: "", CreatedAt: time.Time{}, UpdatedAt: time.Time{}}}
	uc := newDatasetUsecase(tasks, &mockExampleRepo{}, &mockSynthGen{})

	content := "hello,world\nfoo,bar\n"

	_, err := uc.ImportCSV("task_1", content)
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}
}

func TestDatasetUsecase_ImportCSV_Empty(t *testing.T) {
	t.Parallel()

	tasks := &mockTaskRepo{task: &domain.Task{ID: "task_1", Name: "", Description: "", Type: "", CreatedAt: time.Time{}, UpdatedAt: time.Time{}}}
	uc := newDatasetUsecase(tasks, &mockExampleRepo{}, &mockSynthGen{})

	_, err := uc.ImportCSV("task_1", "")
	if !errors.Is(err, domain.ErrInvalidInput) {
		t.Errorf("expected ErrInvalidInput for empty CSV, got %v", err)
	}
}

func TestDatasetUsecase_ImportCSV_TaskNotFound(t *testing.T) {
	t.Parallel()

	uc := newDatasetUsecase(&mockTaskRepo{}, &mockExampleRepo{}, &mockSynthGen{})

	_, err := uc.ImportCSV("missing", "a,b")
	if !errors.Is(err, domain.ErrNotFound) {
		t.Errorf("expected ErrNotFound, got %v", err)
	}
}

// --- ImportJSONL ---.

func TestDatasetUsecase_ImportJSONL_Alpaca_AutoDetect(t *testing.T) {
	t.Parallel()

	tasks := &mockTaskRepo{task: &domain.Task{ID: "task_1", Type: domain.TaskGeneration}}
	examples := &mockExampleRepo{}
	uc := newDatasetUsecase(tasks, examples, &mockSynthGen{})

	content := `{"instruction":"Write a function that adds two numbers", "input":"", "output":"func add(a, b int) int { return a + b }", "category":"code-generation"}
{"instruction":"Explain channels", "input":"", "output":"Channels are typed conduits for goroutine communication.", "id":"ex_1"}`

	stats, err := uc.ImportJSONL("task_1", content, "")
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}

	if stats == nil {
		t.Fatal("expected non-nil stats")
	}

	if len(examples.addBatch) != 2 {
		t.Errorf("expected 2 examples, got %d", len(examples.addBatch))
	}

	// Alpaca instruction maps directly to input.
	if examples.addBatch[0].Input != "Write a function that adds two numbers" {
		t.Errorf("unexpected input: %q", examples.addBatch[0].Input)
	}
	if examples.addBatch[0].Output != "func add(a, b int) int { return a + b }" {
		t.Errorf("unexpected output: %q", examples.addBatch[0].Output)
	}
}

func TestDatasetUsecase_ImportJSONL_Alpaca_WithInput(t *testing.T) {
	t.Parallel()

	tasks := &mockTaskRepo{task: &domain.Task{ID: "task_1", Type: domain.TaskGeneration}}
	examples := &mockExampleRepo{}
	uc := newDatasetUsecase(tasks, examples, &mockSynthGen{})

	content := `{"instruction":"Refactor this code", "input":"func old() int { return 1 }", "output":"func new() int { return 1 }"}`

	_, err := uc.ImportJSONL("task_1", content, "alpaca")
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}

	if len(examples.addBatch) != 1 {
		t.Fatalf("expected 1 example, got %d", len(examples.addBatch))
	}

	// Instruction + input are combined into one prompt.
	expected := "Refactor this code\n\nfunc old() int { return 1 }"
	if examples.addBatch[0].Input != expected {
		t.Errorf("expected input %q, got %q", expected, examples.addBatch[0].Input)
	}
}

func TestDatasetUsecase_ImportJSONL_Alpaca_SkipsEmptyOutput(t *testing.T) {
	t.Parallel()

	tasks := &mockTaskRepo{task: &domain.Task{ID: "task_1", Type: domain.TaskGeneration}}
	examples := &mockExampleRepo{}
	uc := newDatasetUsecase(tasks, examples, &mockSynthGen{})

	// Second record has no output — should be skipped.
	content := `{"instruction":"Good", "input":"","output":"response"}
{"instruction":"Bad", "input":"","output":""}`

	_, err := uc.ImportJSONL("task_1", content, "")
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}

	if len(examples.addBatch) != 1 {
		t.Errorf("expected 1 example (bad record skipped), got %d", len(examples.addBatch))
	}
}

func TestDatasetUsecase_ImportJSONL_Chat_AutoDetect(t *testing.T) {
	t.Parallel()

	tasks := &mockTaskRepo{task: &domain.Task{ID: "task_1", Type: domain.TaskGeneration}}
	examples := &mockExampleRepo{}
	uc := newDatasetUsecase(tasks, examples, &mockSynthGen{})

	content := `{"messages":[{"role":"system","content":"You are a Go expert"},{"role":"user","content":"Write a function to reverse a string"},{"role":"assistant","content":"func reverse(s string) string { ... }"}], "id":"rec_1", "category":"code-generation"}`

	stats, err := uc.ImportJSONL("task_1", content, "")
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}

	if stats == nil {
		t.Fatal("expected non-nil stats")
	}

	if len(examples.addBatch) != 1 {
		t.Fatalf("expected 1 example, got %d", len(examples.addBatch))
	}

	// System + user messages become input; assistant becomes output.
	expectedInput := "You are a Go expert\n\nWrite a function to reverse a string"
	if examples.addBatch[0].Input != expectedInput {
		t.Errorf("expected input %q, got %q", expectedInput, examples.addBatch[0].Input)
	}

	if examples.addBatch[0].Output != "func reverse(s string) string { ... }" {
		t.Errorf("unexpected output: %q", examples.addBatch[0].Output)
	}
}

func TestDatasetUsecase_ImportJSONL_Chat_LastAssistantWins(t *testing.T) {
	t.Parallel()

	tasks := &mockTaskRepo{task: &domain.Task{ID: "task_1", Type: domain.TaskGeneration}}
	examples := &mockExampleRepo{}
	uc := newDatasetUsecase(tasks, examples, &mockSynthGen{})

	// Two assistant messages — only the last becomes output.
	content := `{"messages":[{"role":"user","content":"What is a slice?"},{"role":"assistant","content":"A slice is a view into an array."},{"role":"user","content":"And a map?"},{"role":"assistant","content":"A map is an unordered key-value store."}]}`

	_, err := uc.ImportJSONL("task_1", content, "chat")
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}

	if len(examples.addBatch) != 1 {
		t.Fatalf("expected 1 example, got %d", len(examples.addBatch))
	}

	expectedInput := "What is a slice?\n\nAnd a map?"
	if examples.addBatch[0].Input != expectedInput {
		t.Errorf("expected input %q, got %q", expectedInput, examples.addBatch[0].Input)
	}

	if examples.addBatch[0].Output != "A map is an unordered key-value store." {
		t.Errorf("expected last assistant message, got %q", examples.addBatch[0].Output)
	}
}

func TestDatasetUsecase_ImportJSONL_ForcedAlpacaOnChat(t *testing.T) {
	t.Parallel()

	tasks := &mockTaskRepo{task: &domain.Task{ID: "task_1", Type: domain.TaskGeneration}}
	examples := &mockExampleRepo{}
	uc := newDatasetUsecase(tasks, examples, &mockSynthGen{})

	// Forcing alpaca on chat records should yield no valid pairs → ErrInvalidInput.
	content := `{"messages":[{"role":"user","content":"hi"},{"role":"assistant","content":"hello"}]}`

	_, err := uc.ImportJSONL("task_1", content, "alpaca")
	if !errors.Is(err, domain.ErrInvalidInput) {
		t.Errorf("expected ErrInvalidInput when forced format mismatches, got %v", err)
	}
}

func TestDatasetUsecase_ImportJSONL_Empty(t *testing.T) {
	t.Parallel()

	tasks := &mockTaskRepo{task: &domain.Task{ID: "task_1", Type: domain.TaskGeneration}}
	uc := newDatasetUsecase(tasks, &mockExampleRepo{}, &mockSynthGen{})

	_, err := uc.ImportJSONL("task_1", "", "")
	if !errors.Is(err, domain.ErrInvalidInput) {
		t.Errorf("expected ErrInvalidInput for empty content, got %v", err)
	}
}

func TestDatasetUsecase_ImportJSONL_BlankLinesOnly(t *testing.T) {
	t.Parallel()

	tasks := &mockTaskRepo{task: &domain.Task{ID: "task_1", Type: domain.TaskGeneration}}
	uc := newDatasetUsecase(tasks, &mockExampleRepo{}, &mockSynthGen{})

	_, err := uc.ImportJSONL("task_1", "\n\n  \n", "")
	if !errors.Is(err, domain.ErrInvalidInput) {
		t.Errorf("expected ErrInvalidInput for blank-only content, got %v", err)
	}
}

func TestDatasetUsecase_ImportJSONL_MalformedJSON(t *testing.T) {
	t.Parallel()

	tasks := &mockTaskRepo{task: &domain.Task{ID: "task_1", Type: domain.TaskGeneration}}
	uc := newDatasetUsecase(tasks, &mockExampleRepo{}, &mockSynthGen{})

	_, err := uc.ImportJSONL("task_1", "not a json line", "")
	if !errors.Is(err, domain.ErrInvalidInput) {
		t.Errorf("expected ErrInvalidInput for malformed JSON, got %v", err)
	}
}

func TestDatasetUsecase_ImportJSONL_UnknownFieldsIgnored(t *testing.T) {
	t.Parallel()

	tasks := &mockTaskRepo{task: &domain.Task{ID: "task_1", Type: domain.TaskGeneration}}
	examples := &mockExampleRepo{}
	uc := newDatasetUsecase(tasks, examples, &mockSynthGen{})

	// Extra fields (id, category, tags) don't break parsing.
	content := `{"instruction":"Write a test", "input":"", "output":"func TestX(t *testing.T) {}", "id":"ex_1", "category":"testing", "tags":["go","unit"]}`

	_, err := uc.ImportJSONL("task_1", content, "")
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}

	if len(examples.addBatch) != 1 {
		t.Errorf("expected 1 example, got %d", len(examples.addBatch))
	}
}

func TestDatasetUsecase_ImportJSONL_TaskNotFound(t *testing.T) {
	t.Parallel()

	uc := newDatasetUsecase(&mockTaskRepo{}, &mockExampleRepo{}, &mockSynthGen{})

	_, err := uc.ImportJSONL("missing", `{"instruction":"a","input":"","output":"b"}`, "")
	if !errors.Is(err, domain.ErrNotFound) {
		t.Errorf("expected ErrNotFound, got %v", err)
	}
}

// --- UpdateExample ---.

func TestDatasetUsecase_UpdateExample_Valid(t *testing.T) {
	t.Parallel()

	tasks := &mockTaskRepo{task: &domain.Task{ID: "task_1", Name: "", Description: "", Type: "", CreatedAt: time.Time{}, UpdatedAt: time.Time{}}}
	examples := &mockExampleRepo{
		getExample: &domain.Example{ID: "ex_1", TaskID: "task_1", Input: "old", Output: "oldout", Source: "", Flagged: false, FlagNote: "", Duplicate: false, CreatedAt: time.Time{}},
	}
	uc := newDatasetUsecase(tasks, examples, &mockSynthGen{})

	_, err := uc.UpdateExample("task_1", "ex_1", "newin", "newout")
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}

	if examples.updated.Input != "newin" {
		t.Errorf("expected updated input 'newin', got %q", examples.updated.Input)
	}
}

func TestDatasetUsecase_UpdateExample_EmptyInput(t *testing.T) {
	t.Parallel()

	uc := newDatasetUsecase(&mockTaskRepo{}, &mockExampleRepo{}, &mockSynthGen{})

	_, err := uc.UpdateExample("task_1", "ex_1", "  ", "out")
	if !errors.Is(err, domain.ErrInvalidInput) {
		t.Errorf("expected ErrInvalidInput, got %v", err)
	}
}

func TestDatasetUsecase_UpdateExample_NotFound(t *testing.T) {
	t.Parallel()

	tasks := &mockTaskRepo{task: &domain.Task{ID: "task_1", Name: "", Description: "", Type: "", CreatedAt: time.Time{}, UpdatedAt: time.Time{}}}
	uc := newDatasetUsecase(tasks, &mockExampleRepo{}, &mockSynthGen{})

	_, err := uc.UpdateExample("task_1", "missing", "in", "out")
	if !errors.Is(err, domain.ErrNotFound) {
		t.Errorf("expected ErrNotFound, got %v", err)
	}
}

// --- DeleteExample ---.

func TestDatasetUsecase_DeleteExample_Valid(t *testing.T) {
	t.Parallel()

	tasks := &mockTaskRepo{task: &domain.Task{ID: "task_1", Name: "", Description: "", Type: "", CreatedAt: time.Time{}, UpdatedAt: time.Time{}}}
	examples := &mockExampleRepo{}
	uc := newDatasetUsecase(tasks, examples, &mockSynthGen{})

	_, err := uc.DeleteExample("task_1", "ex_1")
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}

	if examples.deletedID != "ex_1" {
		t.Errorf("expected deleted ID 'ex_1', got %q", examples.deletedID)
	}
}

// --- Curate ---.

func TestDatasetUsecase_Curate_DeduplicatesAndFlags(t *testing.T) {
	t.Parallel()

	tasks := &mockTaskRepo{task: &domain.Task{ID: "task_1", Type: domain.TaskClassification, Name: "", Description: "", CreatedAt: time.Time{}, UpdatedAt: time.Time{}}}
	examples := &mockExampleRepo{
		byTask: []*domain.Example{
			{ID: "e1", Input: "hello", Output: "world", Source: domain.SourceUser},
			{ID: "e2", Input: "hello", Output: "world", Source: domain.SourceUser}, // duplicate.
			{ID: "e3", Input: "x", Output: "y", Source: domain.SourceUser},         // flagged: input too short
		},
	}
	uc := newDatasetUsecase(tasks, examples, &mockSynthGen{})

	stats, err := uc.Curate("task_1")
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}

	if stats.Total != 3 {
		t.Errorf("expected total 3, got %d", stats.Total)
	}

	if stats.Duplicates != 1 {
		t.Errorf("expected 1 duplicate, got %d", stats.Duplicates)
	}

	if stats.Flagged != 1 {
		t.Errorf("expected 1 flagged, got %d", stats.Flagged)
	}

	if stats.UserProvided != 3 {
		t.Errorf("expected 3 user-provided, got %d", stats.UserProvided)
	}
}

func TestDatasetUsecase_Curate_NotReady_TooFew(t *testing.T) {
	t.Parallel()

	tasks := &mockTaskRepo{task: &domain.Task{ID: "task_1", Type: domain.TaskGeneration, Name: "", Description: "", CreatedAt: time.Time{}, UpdatedAt: time.Time{}}}
	examples := &mockExampleRepo{byTask: []*domain.Example{
		{Input: "hello", Output: "world", Source: domain.SourceUser},
	}}
	uc := newDatasetUsecase(tasks, examples, &mockSynthGen{})

	stats, _ := uc.Curate("task_1")
	if stats.ReadyToTrain {
		t.Error("expected not ready with <3 usable")
	}
}

func TestDatasetUsecase_Curate_NotReady_SingleLabel(t *testing.T) {
	t.Parallel()

	tasks := &mockTaskRepo{task: &domain.Task{ID: "task_1", Type: domain.TaskClassification, Name: "", Description: "", CreatedAt: time.Time{}, UpdatedAt: time.Time{}}}
	examples := &mockExampleRepo{byTask: []*domain.Example{
		{Input: "hello1", Output: "same", Source: domain.SourceUser},
		{Input: "hello2", Output: "same", Source: domain.SourceUser},
		{Input: "hello3", Output: "same", Source: domain.SourceUser},
	}}
	uc := newDatasetUsecase(tasks, examples, &mockSynthGen{})

	stats, _ := uc.Curate("task_1")
	if stats.ReadyToTrain {
		t.Error("expected not ready with single label")
	}
}

func TestDatasetUsecase_Curate_Ready_Balanced(t *testing.T) {
	t.Parallel()

	tasks := &mockTaskRepo{task: &domain.Task{ID: "task_1", Type: domain.TaskClassification, Name: "", Description: "", CreatedAt: time.Time{}, UpdatedAt: time.Time{}}}
	examples := &mockExampleRepo{byTask: []*domain.Example{
		{Input: "hello1", Output: "yes", Source: domain.SourceUser},
		{Input: "hello2", Output: "no", Source: domain.SourceUser},
		{Input: "hello3", Output: "yes", Source: domain.SourceUser},
	}}
	uc := newDatasetUsecase(tasks, examples, &mockSynthGen{})

	stats, _ := uc.Curate("task_1")
	if !stats.ReadyToTrain {
		t.Error("expected ready with balanced labels")
	}
}

func TestDatasetUsecase_Curate_IdenticalInputOutput(t *testing.T) {
	t.Parallel()

	tasks := &mockTaskRepo{task: &domain.Task{ID: "task_1", Type: domain.TaskGeneration, Name: "", Description: "", CreatedAt: time.Time{}, UpdatedAt: time.Time{}}}
	examples := &mockExampleRepo{byTask: []*domain.Example{
		{ID: "e1", Input: "same", Output: "same", Source: domain.SourceUser},
	}}
	uc := newDatasetUsecase(tasks, examples, &mockSynthGen{})

	stats, _ := uc.Curate("task_1")
	if stats.Flagged != 1 {
		t.Errorf("expected 1 flagged for identical input/output, got %d", stats.Flagged)
	}
}

func TestDatasetUsecase_Curate_TaskNotFound(t *testing.T) {
	t.Parallel()

	uc := newDatasetUsecase(&mockTaskRepo{}, &mockExampleRepo{}, &mockSynthGen{})

	_, err := uc.Curate("missing")
	if !errors.Is(err, domain.ErrNotFound) {
		t.Errorf("expected ErrNotFound, got %v", err)
	}
}
