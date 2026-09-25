package usecase_test

import (
	"encoding/json"
	"testing"

	"distillery/internal/domain"
	"distillery/internal/usecase"
)

type mockIDGen struct{}

func (m *mockIDGen) NewID(prefix string) string {
	return prefix + "_test"
}

func newTestNERUsecase(task *domain.Task) (*usecase.DatasetUsecase, *mockExampleRepo) {
	tasks := &mockTaskRepo{task: task}
	examples := &mockExampleRepo{}
	uc := usecase.NewDatasetUsecase(tasks, examples, &mockSynthGen{}, &mockIDGen{})

	return uc, examples
}

func TestImportNERJSONL_Success(t *testing.T) {
	t.Parallel()

	task := &domain.Task{ID: "task_ner", Kind: domain.KindTokenClassifier}
	uc, examples := newTestNERUsecase(task)

	content := `{"text": "Acme paid $4,200 on 3 May.", "entities": [{"start":0,"end":4,"label":"ORG"},{"start":10,"end":16,"label":"AMOUNT"}]}`

	stats, err := uc.ImportNERJSONL("task_ner", content)
	if err != nil {
		t.Fatalf("ImportNERJSONL failed: %v", err)
	}

	if stats.Total != 1 {
		t.Fatalf("expected 1 total, got %d", stats.Total)
	}

	if len(examples.addBatch) != 1 {
		t.Fatalf("expected 1 example added, got %d", len(examples.addBatch))
	}

	var payload struct {
		Text     string              `json:"text"`
		Entities []domain.EntitySpan `json:"entities"`
	}
	if err := json.Unmarshal(examples.addBatch[0].Payload, &payload); err != nil {
		t.Fatalf("malformed payload: %v", err)
	}

	if len(payload.Entities) != 2 {
		t.Fatalf("expected 2 entities, got %d", len(payload.Entities))
	}
}

func TestImportNERJSONL_LabelSetEnforcement(t *testing.T) {
	t.Parallel()

	// Task allows only ORG.
	task := &domain.Task{ID: "task_ner", Kind: domain.KindTokenClassifier, LabelSet: []string{"ORG"}}
	uc, _ := newTestNERUsecase(task)

	// Line 1: valid ORG
	// Line 2: invalid AMOUNT label (should be skipped).
	content := "{\"text\": \"Acme Corp\", \"entities\": [{\"start\":0,\"end\":4,\"label\":\"ORG\"}]}\n" +
		"{\"text\": \"Paid $100\", \"entities\": [{\"start\":5,\"end\":9,\"label\":\"AMOUNT\"}]}\n"

	stats, err := uc.ImportNERJSONL("task_ner", content)
	if err != nil {
		t.Fatalf("ImportNERJSONL failed: %v", err)
	}

	// Only 1 example should have been successfully imported.
	if stats.Total != 1 {
		t.Fatalf("expected 1 imported example (2nd skipped due to label mismatch), got %d", stats.Total)
	}
}

func TestImportCoNLL_Success(t *testing.T) {
	t.Parallel()

	task := &domain.Task{ID: "task_conll", Kind: domain.KindTokenClassifier}
	uc, examples := newTestNERUsecase(task)

	conll := `Acme B-ORG
paid O
$ B-AMOUNT
4,200 I-AMOUNT
on O
3 B-DATE
May I-DATE
. O
`

	stats, err := uc.ImportCoNLL("task_conll", conll)
	if err != nil {
		t.Fatalf("ImportCoNLL failed: %v", err)
	}

	if stats.Total != 1 {
		t.Fatalf("expected 1 imported sentence, got %d", stats.Total)
	}

	if len(examples.addBatch) != 1 {
		t.Fatalf("expected 1 example in repo, got %d", len(examples.addBatch))
	}

	var payload struct {
		Text     string              `json:"text"`
		Entities []domain.EntitySpan `json:"entities"`
	}

	_ = json.Unmarshal(examples.addBatch[0].Payload, &payload)

	if len(payload.Entities) != 3 {
		t.Fatalf("expected 3 merged entities (ORG, AMOUNT, DATE), got %d: %+v", len(payload.Entities), payload.Entities)
	}

	// Validate surface forms.
	if payload.Text[payload.Entities[0].Start:payload.Entities[0].End] != "Acme" {
		t.Errorf("expected ORG 'Acme', got %q", payload.Text[payload.Entities[0].Start:payload.Entities[0].End])
	}
}

func TestImportNERCSV_JSONInCSV(t *testing.T) {
	t.Parallel()

	task := &domain.Task{ID: "task_csv", Kind: domain.KindTokenClassifier}
	uc, examples := newTestNERUsecase(task)

	csv := `text,entities
"Acme paid $4,200","[{""start"":0,""end"":4,""label"":""ORG""},{""start"":10,""end"":16,""label"":""AMOUNT""}]"
`

	stats, err := uc.ImportNERCSV("task_csv", csv)
	if err != nil {
		t.Fatalf("ImportNERCSV failed: %v", err)
	}

	if stats.Total != 1 {
		t.Fatalf("expected 1 imported example, got %d", stats.Total)
	}

	var payload struct {
		Entities []domain.EntitySpan `json:"entities"`
	}

	_ = json.Unmarshal(examples.addBatch[0].Payload, &payload)
	if len(payload.Entities) != 2 {
		t.Fatalf("expected 2 entities, got %d", len(payload.Entities))
	}
}

func TestImportNERCSV_FlatColumns(t *testing.T) {
	t.Parallel()

	task := &domain.Task{ID: "task_csv_flat", Kind: domain.KindTokenClassifier}
	uc, examples := newTestNERUsecase(task)

	csv := `text,start,end,label
"Acme Corp",0,4,ORG
`

	stats, err := uc.ImportNERCSV("task_csv_flat", csv)
	if err != nil {
		t.Fatalf("ImportNERCSV flat failed: %v", err)
	}

	if stats.Total != 1 {
		t.Fatalf("expected 1 imported example, got %d", stats.Total)
	}

	var payload struct {
		Entities []domain.EntitySpan `json:"entities"`
	}

	_ = json.Unmarshal(examples.addBatch[0].Payload, &payload)
	if len(payload.Entities) != 1 || payload.Entities[0].Label != "ORG" {
		t.Fatalf("expected 1 ORG entity, got %+v", payload.Entities)
	}
}
