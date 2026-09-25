package domain_test

import (
	"testing"

	"distillery/internal/domain"
)

func TestValidateEntitySpans_Valid(t *testing.T) {
	t.Parallel()

	text := "Apple acquired Beats for $3 billion in 2014."
	spans := []domain.EntitySpan{
		{Start: 0, End: 5, Label: "ORG"},
		{Start: 15, End: 20, Label: "ORG"},
		{Start: 25, End: 35, Label: "MONEY"},
		{Start: 39, End: 43, Label: "DATE"},
	}

	err := domain.ValidateEntitySpans(text, spans, []string{"ORG", "MONEY", "DATE"})
	if err != nil {
		t.Fatalf("expected valid spans, got: %v", err)
	}
}

func TestValidateEntitySpans_EmptyText(t *testing.T) {
	t.Parallel()

	err := domain.ValidateEntitySpans("", []domain.EntitySpan{{Start: 0, End: 1, Label: "A"}}, nil)
	if err == nil {
		t.Fatal("expected error on empty text")
	}
}

func TestValidateEntitySpans_OutOfBounds(t *testing.T) {
	t.Parallel()

	text := "Hello world"
	// Start < 0.
	err := domain.ValidateEntitySpans(text, []domain.EntitySpan{{Start: -1, End: 5, Label: "GREETING"}}, nil)
	if err == nil {
		t.Fatal("expected error for negative start")
	}

	// End > len(text).
	err = domain.ValidateEntitySpans(text, []domain.EntitySpan{{Start: 0, End: 15, Label: "GREETING"}}, nil)
	if err == nil {
		t.Fatal("expected error for end > len(text)")
	}

	// End <= Start.
	err = domain.ValidateEntitySpans(text, []domain.EntitySpan{{Start: 5, End: 5, Label: "EMPTY"}}, nil)
	if err == nil {
		t.Fatal("expected error for start == end")
	}

	err = domain.ValidateEntitySpans(text, []domain.EntitySpan{{Start: 6, End: 5, Label: "INVERTED"}}, nil)
	if err == nil {
		t.Fatal("expected error for start > end")
	}
}

func TestValidateEntitySpans_Overlaps(t *testing.T) {
	t.Parallel()

	text := "New York City is big."
	// Overlapping spans: "New York" (0..8) and "York City" (4..13).
	spans := []domain.EntitySpan{
		{Start: 0, End: 8, Label: "LOC"},
		{Start: 4, End: 13, Label: "LOC"},
	}

	err := domain.ValidateEntitySpans(text, spans, nil)
	if err == nil {
		t.Fatal("expected error for overlapping spans")
	}
}

func TestValidateEntitySpans_AdjacentAllowed(t *testing.T) {
	t.Parallel()

	text := "OneTwoThree"
	// Adjacent non-overlapping spans: [0..3], [3..6], [6..11].
	spans := []domain.EntitySpan{
		{Start: 0, End: 3, Label: "NUM"},
		{Start: 3, End: 6, Label: "NUM"},
		{Start: 6, End: 11, Label: "NUM"},
	}

	err := domain.ValidateEntitySpans(text, spans, nil)
	if err != nil {
		t.Fatalf("adjacent spans should not be considered overlapping: %v", err)
	}
}

func TestValidateEntitySpans_AllowedLabels(t *testing.T) {
	t.Parallel()

	text := "Jane lives in Paris."
	spans := []domain.EntitySpan{
		{Start: 0, End: 4, Label: "PER"},
		{Start: 14, End: 19, Label: "UNKNOWN"},
	}

	err := domain.ValidateEntitySpans(text, spans, []string{"PER", "LOC"})
	if err == nil {
		t.Fatal("expected error for label not in allowed label set")
	}
}

func TestValidateEntitySpans_EmptyLabel(t *testing.T) {
	t.Parallel()

	text := "Hello world"

	err := domain.ValidateEntitySpans(text, []domain.EntitySpan{{Start: 0, End: 5, Label: ""}}, nil)
	if err == nil {
		t.Fatal("expected error for empty span label")
	}
}
