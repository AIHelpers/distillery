// Package dataset implements the per-kind DatasetSchema contract: it maps
// each ModelKind to a schema that validates payloads, computes dataset stats,
// and declares supported import formats. This replaces the assumption that
// every task's dataset is input -> output text.
package dataset

import (
	"encoding/json"
	"errors"
	"fmt"
	"strings"

	"distillery/internal/domain"
)

// Registry maps a ModelKind to its DatasetSchema.
type Registry struct {
	schemas map[domain.ModelKind]domain.DatasetSchema
}

// NewRegistry builds a registry with every supported kind's schema.
func NewRegistry() *Registry {
	r := &Registry{schemas: make(map[domain.ModelKind]domain.DatasetSchema)}

	// Register all kinds.
	register := func(s domain.DatasetSchema) {
		r.schemas[s.Kind()] = s
	}

	register(&CausalLMSchema{})
	register(&SeqClassifierSchema{})
	register(&TokenClassifierSchema{})
	register(&EmbeddingSchema{})
	register(&RerankerSchema{})
	register(&PreferenceLMSchema{})

	return r
}

// Schema returns the schema for a kind, or nil when the kind is unsupported.
func (r *Registry) Schema(kind domain.ModelKind) domain.DatasetSchema {
	return r.schemas[kind]
}

// MustSchema returns the schema for a kind, panicking on an unsupported kind.
// Used in server wiring where the set of registered kinds is a build-time
// invariant.
func (r *Registry) MustSchema(kind domain.ModelKind) domain.DatasetSchema {
	s := r.schemas[kind]
	if s == nil {
		panic(fmt.Sprintf("dataset: no schema registered for kind %q", kind))
	}

	return s
}

// Kinds returns all kinds that have a registered schema.
func (r *Registry) Kinds() []domain.ModelKind {
	out := make([]domain.ModelKind, 0, len(r.schemas))

	for k := range r.schemas {
		out = append(out, k)
	}

	return out
}

// SchemaForTask returns the schema for the task's model kind, defaulting to
// causal_lm when the task has no kind yet (legacy records).
func (r *Registry) SchemaForTask(t *domain.Task) domain.DatasetSchema {
	kind := t.Kind
	if kind == "" {
		kind = domain.DefaultModelKind
	}

	return r.MustSchema(kind)
}

// --- Helpers shared by the per-kind schemas ---.

// normalizePayload decodes a JSON object payload, erroring on malformed JSON.
func normalizePayload(payload []byte) (map[string]interface{}, error) {
	if len(payload) == 0 {
		return nil, errors.New("payload is empty")
	}

	var m map[string]interface{}

	err := jsonUnmarshal(payload, &m)
	if err != nil {
		return nil, fmt.Errorf("payload is not a JSON object: %w", err)
	}

	return m, nil
}

// jsonUnmarshal is a tiny indirection so the package doesn't hardcode json
// everywhere (also lets tests inject a fake).
var jsonUnmarshal = func(b []byte, v interface{}) error {
	return jsonUnmarshalImpl(b, v)
}

func jsonUnmarshalImpl(b []byte, v interface{}) error {
	return json.Unmarshal(b, v)
}

// getString extracts a trimmed non-empty string field from a payload map.
func getString(m map[string]interface{}, keys ...string) (string, bool) {
	for _, k := range keys {
		if v, ok := m[k]; ok && v != nil {
			if s, ok := v.(string); ok {
				if s = strings.TrimSpace(s); s != "" {
					return s, true
				}
			}
		}
	}

	return "", false
}
