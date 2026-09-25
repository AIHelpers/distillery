package domain_test

import (
	"testing"

	"distillery/internal/domain"
)

func TestValidateJSONSchema_Valid(t *testing.T) {
	t.Parallel()

	validSchemas := []string{
		`{"type": "object", "properties": {"name": {"type": "string"}}}`,
		`{"type": "object", "required": ["id"], "properties": {"id": {"type": "integer"}}}`,
		`{"type": "array", "items": {"type": "string"}}`,
		`{"type": "string", "enum": ["red", "green", "blue"]}`,
		`{"type": "number", "minimum": 0, "maximum": 100}`,
		"", // empty string is valid (no constraint).
	}

	for _, s := range validSchemas {
		err := domain.ValidateJSONSchema(s)
		if err != nil {
			t.Errorf("expected valid schema for %s, got: %v", s, err)
		}
	}
}

func TestValidateJSONSchema_Invalid(t *testing.T) {
	t.Parallel()

	invalidSchemas := []string{
		`not json`,
		`"just a string"`,
		`{"type": "unsupported_type"}`,
		`{"properties": "not an object"}`,
	}

	for _, s := range invalidSchemas {
		err := domain.ValidateJSONSchema(s)
		if err == nil {
			t.Errorf("expected error for invalid schema: %s", s)
		}
	}
}

func TestValidateJSONAgainstSchema_Valid(t *testing.T) {
	t.Parallel()

	schema := `{
		"type": "object",
		"required": ["name", "age"],
		"properties": {
			"name": {"type": "string"},
			"age": {"type": "integer", "minimum": 0},
			"tags": {"type": "array", "items": {"type": "string"}}
		},
		"additionalProperties": false
	}`

	doc := `{"name": "Alice", "age": 30, "tags": ["admin", "user"]}`
	err := domain.ValidateJSONAgainstSchema(schema, doc)
	if err != nil {
		t.Fatalf("expected document to satisfy schema, got: %v", err)
	}
}

func TestValidateJSONAgainstSchema_Invalid(t *testing.T) {
	t.Parallel()

	schema := `{
		"type": "object",
		"required": ["name"],
		"properties": {
			"name": {"type": "string"},
			"status": {"type": "string", "enum": ["active", "inactive"]}
		},
		"additionalProperties": false
	}`

	cases := []struct {
		name string
		doc  string
	}{
		{"missing required", `{}`},
		{"wrong type", `{"name": 123}`},
		{"enum mismatch", `{"name": "Alice", "status": "unknown"}`},
		{"disallowed additional", `{"name": "Alice", "extra": true}`},
		{"invalid json", `{name: "Alice"}`},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			err := domain.ValidateJSONAgainstSchema(schema, tc.doc)
			if err == nil {
				t.Fatalf("expected error for %s: %s", tc.name, tc.doc)
			}
		})
	}
}

func TestGenerateGBNF_Smoke(t *testing.T) {
	t.Parallel()

	schema := `{
		"type": "object",
		"required": ["invoice_id", "total"],
		"properties": {
			"invoice_id": {"type": "string"},
			"total": {"type": "number"},
			"currency": {"type": "string", "enum": ["USD", "EUR", "GBP"]}
		}
	}`

	gbnf := domain.GenerateGBNF(schema)
	if gbnf == "" {
		t.Fatal("expected non-empty GBNF output")
	}

	if !contains(gbnf, "root ::=") {
		t.Errorf("GBNF must define root rule, got:\n%s", gbnf)
	}

	if !contains(gbnf, "invoice_id") || !contains(gbnf, "total") {
		t.Errorf("GBNF must reference schema fields, got:\n%s", gbnf)
	}
}

func contains(s, substr string) bool {
	return len(s) >= len(substr) && (s == substr || len(s) > 0 && len(substr) > 0 && stringContains(s, substr))
}

func stringContains(s, sub string) bool {
	for i := 0; i+len(sub) <= len(s); i++ {
		if s[i:i+len(sub)] == sub {
			return true
		}
	}

	return false
}
