package simulation

import "distillery/internal/domain"

// ModelSelector implements domain.ModelSelector by right-sizing a base
// model from the curated catalog to the task's dataset complexity, mirroring
// the "automatic base-model selection" differentiator from the product spec.
type ModelSelector struct {
	Catalog []domain.BaseModel
}

func NewModelSelector() *ModelSelector {
	return &ModelSelector{
		Catalog: []domain.BaseModel{
			{Name: "Qwen2.5-0.5B-Instruct", ParamsBillions: 0.5, Family: "Qwen"},
			{Name: "Llama-3.2-3B-Instruct", ParamsBillions: 3, Family: "Llama"},
			{Name: "Mistral-7B-Instruct-v0.3", ParamsBillions: 7, Family: "Mistral"},
			{Name: "Qwen2.5-14B-Instruct", ParamsBillions: 14, Family: "Qwen"},
		},
	}
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
