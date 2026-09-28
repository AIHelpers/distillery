package dataset_test

import (
	"encoding/json"
	"strings"
	"testing"

	"distillery/internal/domain"
	"distillery/internal/usecase/dataset"
)

func TestPreferenceLMSchema_Validate(t *testing.T) {
	t.Parallel()

	s := &dataset.PreferenceLMSchema{}

	valid, err := json.Marshal(map[string]string{"prompt": "p", "chosen": "good", "rejected": "bad"})
	if err != nil {
		t.Fatal(err)
	}

	err = s.Validate(valid)
	if err != nil {
		t.Errorf("expected a valid payload to pass, got %v", err)
	}

	cases := map[string][]byte{
		"missing prompt":   marshalT(t, map[string]string{"chosen": "good", "rejected": "bad"}),
		"missing chosen":   marshalT(t, map[string]string{"prompt": "p", "rejected": "bad"}),
		"missing rejected": marshalT(t, map[string]string{"prompt": "p", "chosen": "good"}),
		"chosen==rejected": marshalT(t, map[string]string{"prompt": "p", "chosen": "same", "rejected": "same"}),
		"not an object":    []byte(`"just a string"`),
	}

	for name, payload := range cases {
		err := s.Validate(payload)
		if err == nil {
			t.Errorf("%s: expected an error, got nil", name)
		}
	}
}

func TestPreferenceLMSchema_Validate_LengthLimit(t *testing.T) {
	t.Parallel()

	s := &dataset.PreferenceLMSchema{}
	huge := strings.Repeat("a", 20000)

	payload := marshalT(t, map[string]string{"prompt": "p", "chosen": huge, "rejected": "bad"})

	err := s.Validate(payload)
	if err == nil {
		t.Error("expected an oversized field to be rejected")
	}
}

func TestPreferenceLMSchema_KindAndFormats(t *testing.T) {
	t.Parallel()

	s := &dataset.PreferenceLMSchema{}

	if s.Kind() != domain.KindPreferenceLM {
		t.Errorf("expected Kind() == preference_lm, got %v", s.Kind())
	}

	formats := s.Formats()
	if len(formats) != 1 || formats[0] != domain.FormatJSONL {
		t.Errorf("expected only JSONL to be supported, got %v", formats)
	}
}

func TestRegistry_PreferenceLM(t *testing.T) {
	t.Parallel()

	r := dataset.NewRegistry()

	schema := r.Schema(domain.KindPreferenceLM)
	if schema == nil {
		t.Fatal("expected preference_lm to be registered")
	}

	if schema.Kind() != domain.KindPreferenceLM {
		t.Errorf("expected the registered schema's Kind() == preference_lm, got %v", schema.Kind())
	}
}

func TestVisionSchema_Validate(t *testing.T) {
	t.Parallel()

	s := &dataset.VisionSchema{}

	valid := marshalT(t, domain.VisionPayload{Image: "key1", Prompt: "Extract vendor.", Answer: `{"vendor":"Acme"}`})

	err := s.Validate(valid)
	if err != nil {
		t.Errorf("expected valid payload to pass, got %v", err)
	}

	// An empty answer is a valid "awaiting human correction" example.
	unanswered := marshalT(t, domain.VisionPayload{Image: "key1", Prompt: "Extract vendor."})

	err = s.Validate(unanswered)
	if err != nil {
		t.Errorf("expected an unanswered example to be valid, got %v", err)
	}

	missingImage := marshalT(t, domain.VisionPayload{Prompt: "Extract vendor."})

	err = s.Validate(missingImage)
	if err == nil {
		t.Error("expected an error for a missing image key")
	}

	missingPrompt := marshalT(t, domain.VisionPayload{Image: "key1"})

	err = s.Validate(missingPrompt)
	if err == nil {
		t.Error("expected an error for a missing prompt")
	}

	err = s.Validate([]byte("not json"))
	if err == nil {
		t.Error("expected an error for malformed JSON")
	}
}

func TestVisionSchema_Stats(t *testing.T) {
	t.Parallel()

	s := &dataset.VisionSchema{}

	examples := []*domain.Example{
		{Payload: marshalT(t, domain.VisionPayload{Image: "k1", Prompt: "p", Answer: "a"})},
		{Payload: marshalT(t, domain.VisionPayload{Image: "k2", Prompt: "p"})}, // needs_answer.
	}

	stats := s.Stats(examples)

	if stats.LabelBalance["answered"] != 1 || stats.LabelBalance["needs_answer"] != 1 {
		t.Errorf("unexpected label balance: %+v", stats.LabelBalance)
	}
}

func TestVisionSchema_KindAndFormats(t *testing.T) {
	t.Parallel()

	s := &dataset.VisionSchema{}

	if s.Kind() != domain.KindVisionLM {
		t.Errorf("expected Kind() == vision_lm, got %v", s.Kind())
	}

	formats := s.Formats()
	if len(formats) != 1 || formats[0] != domain.FormatJSONL {
		t.Errorf("expected only JSONL to be supported, got %v", formats)
	}
}

func TestRegistry_VisionLM(t *testing.T) {
	t.Parallel()

	r := dataset.NewRegistry()

	schema := r.Schema(domain.KindVisionLM)
	if schema == nil {
		t.Fatal("expected vision_lm to be registered")
	}

	if schema.Kind() != domain.KindVisionLM {
		t.Errorf("expected the registered schema's Kind() == vision_lm, got %v", schema.Kind())
	}
}

func marshalT(t *testing.T, v interface{}) []byte {
	t.Helper()

	b, err := json.Marshal(v)
	if err != nil {
		t.Fatalf("json.Marshal: %v", err)
	}

	return b
}
