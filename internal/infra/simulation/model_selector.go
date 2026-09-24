package simulation

import "distillery/internal/domain"

const quant4BitNF4 = "4bit-nf4"

// ModelSelector implements domain.ModelSelector by right-sizing a base
// model from the curated catalog to the task's dataset complexity, mirroring
// the "automatic base-model selection" differentiator from the product spec.
type ModelSelector struct {
	Catalog []domain.BaseModel
}

func NewModelSelector() *ModelSelector {
	return &ModelSelector{
		Catalog: []domain.BaseModel{
			{
				Name:             "Qwen3-0.6B",
				ParamsBillions:   0.6,
				Family:           "Qwen",
				RepoID:           "Qwen/Qwen3-0.6B",
				MinVRAMGB:        2,
				RecommendedQuant: quant4BitNF4,
			},
			{
				Name:             "Qwen3-1.7B",
				ParamsBillions:   1.7,
				Family:           "Qwen",
				RepoID:           "Qwen/Qwen3-1.7B",
				MinVRAMGB:        4,
				RecommendedQuant: quant4BitNF4,
			},
			{
				Name:             "Llama-3.2-1B-Instruct",
				ParamsBillions:   1.0,
				Family:           "Llama",
				RepoID:           "meta-llama/Llama-3.2-1B-Instruct",
				MinVRAMGB:        2,
				RecommendedQuant: quant4BitNF4,
			},
			{
				Name:             "Llama-3.2-3B-Instruct",
				ParamsBillions:   3,
				Family:           "Llama",
				RepoID:           "meta-llama/Llama-3.2-3B-Instruct",
				MinVRAMGB:        6,
				RecommendedQuant: quant4BitNF4,
			},
			{
				Name:             "Phi-3.5-mini-instruct",
				ParamsBillions:   3.8,
				Family:           "Phi",
				RepoID:           "microsoft/Phi-3.5-mini-instruct",
				MinVRAMGB:        8,
				RecommendedQuant: quant4BitNF4,
			},
			{
				Name:             "Qwen3-4B",
				ParamsBillions:   3,
				Family:           "Qwen",
				RepoID:           "Qwen/Qwen3-4B",
				MinVRAMGB:        6,
				RecommendedQuant: quant4BitNF4,
			},
			{
				Name:             "Qwen3-8B",
				ParamsBillions:   8,
				Family:           "Qwen",
				RepoID:           "Qwen/Qwen3-8B",
				MinVRAMGB:        10,
				RecommendedQuant: quant4BitNF4,
			},

			// --- Sequence classifiers (small encoder models; CPU-friendly) ---.
			{
				Name:           "DistilBERT-base",
				ParamsBillions: 0.067,
				Family:         "DistilBERT",
				RepoID:         "distilbert-base-uncased",
				Kind:           domain.KindSeqClassifier,
				Capabilities: domain.Capabilities{
					SupportsLoRA:  true,
					ExportFormats: []string{"onnx", "safetensors"},
					RunsOnCPU:     true,
					MaxSeqLen:     512,
					Languages:     []string{"en"},
				},
			},
			{
				Name:           "DeBERTa-v3-small",
				ParamsBillions: 0.082,
				Family:         "DeBERTa",
				RepoID:         "microsoft/deberta-v3-small",
				Kind:           domain.KindSeqClassifier,
				Capabilities: domain.Capabilities{
					SupportsLoRA:  true,
					ExportFormats: []string{"onnx", "safetensors"},
					RunsOnCPU:     true,
					MaxSeqLen:     512,
					Languages:     []string{"en"},
				},
			},
			{
				Name:           "DeBERTa-v3-base",
				ParamsBillions: 0.184,
				Family:         "DeBERTa",
				RepoID:         "microsoft/deberta-v3-base",
				Kind:           domain.KindSeqClassifier,
				Capabilities: domain.Capabilities{
					SupportsLoRA:  true,
					ExportFormats: []string{"onnx", "safetensors"},
					RunsOnCPU:     true,
					MaxSeqLen:     512,
					Languages:     []string{"en"},
				},
			},
			{
				Name:           "ModernBERT-base",
				ParamsBillions: 0.149,
				Family:         "ModernBERT",
				RepoID:         "answerdotai/ModernBERT-base",
				Kind:           domain.KindSeqClassifier,
				Capabilities: domain.Capabilities{
					SupportsLoRA:  true,
					ExportFormats: []string{"onnx", "safetensors"},
					RunsOnCPU:     true,
					MaxSeqLen:     8192,
					Languages:     []string{"en"},
				},
			},
			{
				Name:           "XLM-RoBERTa-base",
				ParamsBillions: 0.279,
				Family:         "XLM-RoBERTa",
				RepoID:         "FacebookAI/xlm-roberta-base",
				Kind:           domain.KindSeqClassifier,
				Capabilities: domain.Capabilities{
					SupportsLoRA:  true,
					ExportFormats: []string{"onnx", "safetensors"},
					RunsOnCPU:     true,
					MaxSeqLen:     512,
					Languages:     []string{"multilingual"},
				},
			},
			{
				Name:           "mDeBERTa-v3-base",
				ParamsBillions: 0.278,
				Family:         "DeBERTa",
				RepoID:         "microsoft/mdeberta-v3-base",
				Kind:           domain.KindSeqClassifier,
				Capabilities: domain.Capabilities{
					SupportsLoRA:  true,
					ExportFormats: []string{"onnx", "safetensors"},
					RunsOnCPU:     true,
					MaxSeqLen:     512,
					Languages:     []string{"multilingual"},
				},
			},
		},
	}
}

// ListBaseModels returns the curated catalog of available base models
// the user can choose from for fine-tuning.
func (s *ModelSelector) ListBaseModels() []domain.BaseModel {
	return s.Catalog
}

// candidatesForKind returns the catalog entries suitable for the given kind,
// defaulting to causal_lm entries when a kind has no specific entries yet.
func (s *ModelSelector) candidatesForKind(kind domain.ModelKind) []domain.BaseModel {
	var out []domain.BaseModel

	for _, m := range s.Catalog {
		if m.Kind == "" || m.Kind == kind {
			out = append(out, m)
		}
	}

	if len(out) == 0 {
		return s.Catalog // fall back to all (legacy behavior).
	}

	return out
}

// SelectBaseModel picks the smallest model likely to work for the task,
// scaling up for generation tasks, larger vocabularies, or longer outputs.
func (s *ModelSelector) SelectBaseModel(
	task *domain.Task,
	exampleCount,
	avgInputLen,
	avgOutputLen int,
) domain.BaseModel {
	score := 0

	switch task.Type {
	case domain.TaskClassification:
		score += 0
	case domain.TaskExtraction:
		score++
	case domain.TaskGeneration:
		score += 2
	}

	if avgOutputLen > 200 {
		score += 2
	} else if avgOutputLen > 60 {
		score++
	}

	if avgInputLen > 800 {
		score++
	}

	if exampleCount < 50 {
		// Small dataset: prefer a smaller model to avoid overfitting risk
		// and keep training cost down.
		score--
	}

	if score < 0 {
		score = 0
	}

	if score >= len(s.Catalog) {
		score = len(s.Catalog) - 1
	}

	return s.Catalog[score]
}

// Select picks the right-sized model for a kind + dataset stats, honoring
// hardware constraints (VRAM, CPU-only). Falls back to the legacy task-based
// selection when stats are insufficient.
func (s *ModelSelector) Select(
	kind domain.ModelKind,
	stats domain.DatasetStats,
	constraints domain.SelectionConstraints,
) domain.BaseModel {
	candidates := s.candidatesForKind(kind)
	if len(candidates) == 0 {
		return domain.BaseModel{Name: "", ParamsBillions: 0, Family: ""}
	}

	// Right-sizing: use the smallest candidate that satisfies constraints.
	// Larger datasets justify a larger model; small datasets stay small.
	idx := 0

	switch {
	case stats.Total > 2000:
		idx = len(candidates) / 2
	case stats.Total > 300:
		idx = len(candidates) / 3
	}

	for i := idx; i < len(candidates); i++ {
		m := candidates[i]

		if constraints.CPUOnly && !m.Capabilities.RunsOnCPU {
			continue
		}

		if constraints.MaxVRAMGB > 0 && m.MinVRAMGB > constraints.MaxVRAMGB {
			continue
		}

		if constraints.MaxLatencyMS > 0 && m.ParamsBillions > 3 {
			// Heuristic: models > 3B are unlikely to hit tight latency.
			continue
		}

		return m
	}

	// No candidate satisfies every constraint — return the smallest
	// unconstrained one rather than failing the selection.
	if len(candidates) > 0 {
		return candidates[0]
	}

	// Ultra-safe fallback: the very first catalog entry.
	return s.Catalog[0]
}
