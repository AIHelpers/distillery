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
		},
	}
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
