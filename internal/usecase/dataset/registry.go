// Package dataset implements the per-kind DatasetSchema contract: it maps
// each ModelKind to a schema that validates payloads, computes dataset stats,
// and declares supported import formats. This replaces the assumption that
// every task's dataset is input -> output text.
package dataset

import (
	"fmt"

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
	register(&VisionSchema{})

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
