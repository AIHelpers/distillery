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

	if err := s.Validate(valid); err != nil {
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
		if err := s.Validate(payload); err == nil {
			t.Errorf("%s: expected an error, got nil", name)
		}
	}
}

func TestPreferenceLMSchema_Validate_LengthLimit(t *testing.T) {
	t.Parallel()

	s := &dataset.PreferenceLMSchema{}
	huge := strings.Repeat("a", 20000)

	payload := marshalT(t, map[string]string{"prompt": "p", "chosen": huge, "rejected": "bad"})

	if err := s.Validate(payload); err == nil {
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

func marshalT(t *testing.T, v interface{}) []byte {
	t.Helper()

	b, err := json.Marshal(v)
	if err != nil {
		t.Fatalf("json.Marshal: %v", err)
	}

	return b
}
