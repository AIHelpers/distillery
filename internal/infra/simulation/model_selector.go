package simulation

import "distillery/internal/domain"

const quant4BitNF4 = "4bit-nf4"

const (
	exportFormatONNX        = "onnx"
	exportFormatSafetensors = "safetensors"
	exportFormatCTranslate2 = "ctranslate2"
	languageMultilingual    = "multilingual"
)

// ModelSelector implements domain.ModelSelector by right-sizing a base
// model from the curated catalog to the task's dataset complexity, mirroring
// the "automatic base-model selection" differentiator from the product spec.
type ModelSelector struct {
	Catalog []domain.BaseModel
}

// defaultCatalog is the curated set of base models NewModelSelector starts
// every ModelSelector with.
var defaultCatalog = []domain.BaseModel{
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
			ExportFormats: []string{exportFormatONNX, exportFormatSafetensors},
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
			ExportFormats: []string{exportFormatONNX, exportFormatSafetensors},
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
			ExportFormats: []string{exportFormatONNX, exportFormatSafetensors},
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
			ExportFormats: []string{exportFormatONNX, exportFormatSafetensors},
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
			ExportFormats: []string{exportFormatONNX, exportFormatSafetensors},
			RunsOnCPU:     true,
			MaxSeqLen:     512,
			Languages:     []string{languageMultilingual},
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
			ExportFormats: []string{exportFormatONNX, exportFormatSafetensors},
			RunsOnCPU:     true,
			MaxSeqLen:     512,
			Languages:     []string{languageMultilingual},
		},
	},

	// --- Token classifiers (NER/span tagging; small encoders) ---.
	{
		Name:           "DistilBERT-base",
		ParamsBillions: 0.067,
		Family:         "DistilBERT",
		RepoID:         "distilbert-base-uncased",
		Kind:           domain.KindTokenClassifier,
		Capabilities: domain.Capabilities{
			SupportsLoRA:  true,
			ExportFormats: []string{exportFormatONNX, exportFormatSafetensors},
			RunsOnCPU:     true,
			MaxSeqLen:     512,
			Languages:     []string{"en"},
		},
	},
	{
		Name:           "RoBERTa-base",
		ParamsBillions: 0.125,
		Family:         "RoBERTa",
		RepoID:         "roberta-base",
		Kind:           domain.KindTokenClassifier,
		Capabilities: domain.Capabilities{
			SupportsLoRA:  true,
			ExportFormats: []string{exportFormatONNX, exportFormatSafetensors},
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
		Kind:           domain.KindTokenClassifier,
		Capabilities: domain.Capabilities{
			SupportsLoRA:  true,
			ExportFormats: []string{exportFormatONNX, exportFormatSafetensors},
			RunsOnCPU:     true,
			MaxSeqLen:     512,
			Languages:     []string{"en"},
		},
	},
	{
		Name:           "XLM-RoBERTa-base",
		ParamsBillions: 0.279,
		Family:         "XLM-RoBERTa",
		RepoID:         "FacebookAI/xlm-roberta-base",
		Kind:           domain.KindTokenClassifier,
		Capabilities: domain.Capabilities{
			SupportsLoRA:  true,
			ExportFormats: []string{exportFormatONNX, exportFormatSafetensors},
			RunsOnCPU:     true,
			MaxSeqLen:     512,
			Languages:     []string{languageMultilingual},
		},
	},

	// --- Embedding models (bi-encoders) ---.
	{
		Name:           "bge-small-en-v1.5",
		ParamsBillions: 0.033,
		Family:         "BGE",
		RepoID:         "BAAI/bge-small-en-v1.5",
		Kind:           domain.KindEmbedding,
		Capabilities: domain.Capabilities{
			SupportsLoRA:  true,
			ExportFormats: []string{exportFormatONNX, exportFormatSafetensors},
			RunsOnCPU:     true,
			MaxSeqLen:     512,
			Languages:     []string{"en"},
		},
	},
	{
		Name:           "bge-base-en-v1.5",
		ParamsBillions: 0.109,
		Family:         "BGE",
		RepoID:         "BAAI/bge-base-en-v1.5",
		Kind:           domain.KindEmbedding,
		Capabilities: domain.Capabilities{
			SupportsLoRA:  true,
			ExportFormats: []string{exportFormatONNX, exportFormatSafetensors},
			RunsOnCPU:     true,
			MaxSeqLen:     512,
			Languages:     []string{"en"},
		},
	},
	{
		Name:           "e5-base-v2",
		ParamsBillions: 0.109,
		Family:         "E5",
		RepoID:         "intfloat/e5-base-v2",
		Kind:           domain.KindEmbedding,
		Capabilities: domain.Capabilities{
			SupportsLoRA:  true,
			ExportFormats: []string{exportFormatONNX, exportFormatSafetensors},
			RunsOnCPU:     true,
			MaxSeqLen:     512,
			Languages:     []string{"en"},
		},
	},
	{
		Name:           "gte-base",
		ParamsBillions: 0.109,
		Family:         "GTE",
		RepoID:         "thenlper/gte-base",
		Kind:           domain.KindEmbedding,
		Capabilities: domain.Capabilities{
			SupportsLoRA:  true,
			ExportFormats: []string{exportFormatONNX, exportFormatSafetensors},
			RunsOnCPU:     true,
			MaxSeqLen:     512,
			Languages:     []string{"en"},
		},
	},
	{
		Name:           "multilingual-e5-base",
		ParamsBillions: 0.278,
		Family:         "E5",
		RepoID:         "intfloat/multilingual-e5-base",
		Kind:           domain.KindEmbedding,
		Capabilities: domain.Capabilities{
			SupportsLoRA:  true,
			ExportFormats: []string{exportFormatONNX, exportFormatSafetensors},
			RunsOnCPU:     true,
			MaxSeqLen:     512,
			Languages:     []string{languageMultilingual},
		},
	},

	// --- Reranker models (cross-encoders) ---.
	{
		Name:           "bge-reranker-base",
		ParamsBillions: 0.279,
		Family:         "BGE",
		RepoID:         "BAAI/bge-reranker-base",
		Kind:           domain.KindReranker,
		Capabilities: domain.Capabilities{
			SupportsLoRA:  true,
			ExportFormats: []string{exportFormatONNX, exportFormatSafetensors},
			RunsOnCPU:     true,
			MaxSeqLen:     512,
			Languages:     []string{"en"},
		},
	},
	{
		Name:           "ms-marco-MiniLM-L-6",
		ParamsBillions: 0.022,
		Family:         "CrossEncoder",
		RepoID:         "cross-encoder/ms-marco-MiniLM-L-6-v2",
		Kind:           domain.KindReranker,
		Capabilities: domain.Capabilities{
			SupportsLoRA:  true,
			ExportFormats: []string{exportFormatONNX, exportFormatSafetensors},
			RunsOnCPU:     true,
			MaxSeqLen:     512,
			Languages:     []string{"en"},
		},
	},

	// --- Vision-language / document AI models ---.
	// Verify current model licenses and GGUF/llama.cpp support before
	// finalizing the catalog (per plan 06 — llama.cpp multimodal
	// projector support is architecture-specific and moves quickly).
	{
		Name:             "Qwen2-VL-2B-Instruct",
		ParamsBillions:   2,
		Family:           "Qwen2-VL",
		RepoID:           "Qwen/Qwen2-VL-2B-Instruct",
		MinVRAMGB:        6,
		RecommendedQuant: quant4BitNF4,
		Kind:             domain.KindVisionLM,
		Capabilities: domain.Capabilities{
			SupportsLoRA:   true,
			ExportFormats:  []string{exportFormatSafetensors},
			RunsOnCPU:      false,
			MaxSeqLen:      4096,
			Languages:      []string{"en", languageMultilingual},
			MaxImagePixels: 1_638_400, // ~1280x1280, dynamic-resolution default.
		},
	},
	{
		Name:             "Qwen2-VL-7B-Instruct",
		ParamsBillions:   7,
		Family:           "Qwen2-VL",
		RepoID:           "Qwen/Qwen2-VL-7B-Instruct",
		MinVRAMGB:        16,
		RecommendedQuant: quant4BitNF4,
		Kind:             domain.KindVisionLM,
		Capabilities: domain.Capabilities{
			SupportsLoRA:   true,
			ExportFormats:  []string{exportFormatSafetensors},
			RunsOnCPU:      false,
			MaxSeqLen:      4096,
			Languages:      []string{"en", languageMultilingual},
			MaxImagePixels: 1_638_400,
		},
	},
	{
		Name:             "PaliGemma-3B-mix",
		ParamsBillions:   3,
		Family:           "PaliGemma",
		RepoID:           "google/paligemma-3b-mix-448",
		MinVRAMGB:        8,
		RecommendedQuant: quant4BitNF4,
		Kind:             domain.KindVisionLM,
		Capabilities: domain.Capabilities{
			SupportsLoRA:   true,
			ExportFormats:  []string{exportFormatSafetensors},
			RunsOnCPU:      false,
			MaxSeqLen:      2048,
			Languages:      []string{"en"},
			MaxImagePixels: 200_704, // fixed 448x448 input.
		},
	},
	{
		Name:             "Florence-2-large",
		ParamsBillions:   0.77,
		Family:           "Florence-2",
		RepoID:           "microsoft/Florence-2-large",
		MinVRAMGB:        4,
		RecommendedQuant: quant4BitNF4,
		Kind:             domain.KindVisionLM,
		Capabilities: domain.Capabilities{
			SupportsLoRA:   true,
			ExportFormats:  []string{exportFormatSafetensors},
			RunsOnCPU:      true,
			MaxSeqLen:      1024,
			Languages:      []string{"en"},
			MaxImagePixels: 589_824, // 768x768.
		},
	},

	// --- Speech-to-text (Whisper fine-tuning) ---.
	// CTranslate2 (faster-whisper) is the primary serving format per plan
	// 07 section 6; safetensors is the generic HF fallback.
	{
		Name:           "whisper-tiny",
		ParamsBillions: 0.039,
		Family:         "Whisper",
		RepoID:         "openai/whisper-tiny",
		Kind:           domain.KindASR,
		Capabilities: domain.Capabilities{
			SupportsLoRA:  false, // tiny trains fine fully on CPU.
			ExportFormats: []string{exportFormatCTranslate2, exportFormatSafetensors},
			RunsOnCPU:     true,
			MaxSeqLen:     448, // 30 s @ 50 mel frames/second.
			Languages:     []string{languageMultilingual},
		},
	},
	{
		Name:           "whisper-base",
		ParamsBillions: 0.074,
		Family:         "Whisper",
		RepoID:         "openai/whisper-base",
		Kind:           domain.KindASR,
		Capabilities: domain.Capabilities{
			SupportsLoRA:  false,
			ExportFormats: []string{exportFormatCTranslate2, exportFormatSafetensors},
			RunsOnCPU:     true,
			MaxSeqLen:     448,
			Languages:     []string{languageMultilingual},
		},
	},
	{
		Name:           "whisper-small",
		ParamsBillions: 0.244,
		Family:         "Whisper",
		RepoID:         "openai/whisper-small",
		Kind:           domain.KindASR,
		Capabilities: domain.Capabilities{
			// The plan's LoRA-on-a-single-GPU default for domain audio.
			SupportsLoRA:  true,
			ExportFormats: []string{exportFormatCTranslate2, exportFormatSafetensors},
			RunsOnCPU:     true,
			MaxSeqLen:     448,
			Languages:     []string{languageMultilingual},
		},
	},
	{
		Name:           "whisper-medium",
		ParamsBillions: 0.769,
		Family:         "Whisper",
		RepoID:         "openai/whisper-medium",
		MinVRAMGB:      6,
		Kind:           domain.KindASR,
		Capabilities: domain.Capabilities{
			SupportsLoRA:  true,
			ExportFormats: []string{exportFormatCTranslate2, exportFormatSafetensors},
			RunsOnCPU:     true,
			MaxSeqLen:     448,
			Languages:     []string{languageMultilingual},
		},
	},
	{
		Name:             "whisper-large-v3-turbo",
		ParamsBillions:   0.809,
		Family:           "Whisper",
		RepoID:           "openai/whisper-large-v3-turbo",
		MinVRAMGB:        8,
		RecommendedQuant: quant4BitNF4,
		Kind:             domain.KindASR,
		Capabilities: domain.Capabilities{
			SupportsLoRA:  true,
			ExportFormats: []string{exportFormatCTranslate2, exportFormatSafetensors},
			RunsOnCPU:     false,
			MaxSeqLen:     448,
			Languages:     []string{languageMultilingual},
		},
	},
}

// NewModelSelector returns a ModelSelector seeded with the default catalog.
func NewModelSelector() *ModelSelector {
	return &ModelSelector{Catalog: defaultCatalog}
}

// ListBaseModels returns the curated catalog of available base models
// the user can choose from for fine-tuning.
func (s *ModelSelector) ListBaseModels() []domain.BaseModel {
	return s.Catalog
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

// candidatesForKind returns the catalog entries suitable for the given kind,
// defaulting to causal_lm entries when a kind has no specific entries yet.
// Kinds that DO have dedicated entries never fall back to causal_lm models:
// a speech task must not be auto-sized to a text LLM (same for vision, NER,
// etc.), so the generic entries only apply when nothing better exists.
func (s *ModelSelector) candidatesForKind(kind domain.ModelKind) []domain.BaseModel {
	var (
		specific []domain.BaseModel
		generic  []domain.BaseModel
	)

	for i := range s.Catalog {
		switch {
		case s.Catalog[i].Kind == kind:
			specific = append(specific, s.Catalog[i])
		case s.Catalog[i].Kind == "":
			generic = append(generic, s.Catalog[i])
		}
	}

	if len(specific) > 0 {
		return specific
	}

	if len(generic) > 0 {
		return generic
	}

	return s.Catalog // fall back to all (legacy behavior).
}
