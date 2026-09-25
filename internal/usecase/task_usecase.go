package usecase

import (
	"strings"
	"time"

	"distillery/internal/domain"
)

type TaskUsecase struct {
	tasks    domain.TaskRepository
	examples domain.ExampleRepository
	idGen    IDGenerator
}

func NewTaskUsecase(tasks domain.TaskRepository, examples domain.ExampleRepository, idGen IDGenerator) *TaskUsecase {
	return &TaskUsecase{tasks: tasks, examples: examples, idGen: idGen}
}

// CreateTask stores a new task. The optional model kind selects the model
// architecture used for training (seq_classifier, ...); omit it for the
// causal_lm default.
func (u *TaskUsecase) CreateTask(name, description string, taskType domain.TaskType, kind ...domain.ModelKind) (*domain.Task, error) {
	name = strings.TrimSpace(name)
	if name == "" {
		return nil, domain.ErrInvalidInput
	}

	switch taskType {
	case domain.TaskClassification, domain.TaskExtraction, domain.TaskGeneration:
	default:
		return nil, domain.ErrInvalidInput
	}

	modelKind := domain.DefaultModelKind

	if len(kind) > 0 && kind[0] != "" {
		if !domain.IsValidModelKind(kind[0]) {
			return nil, domain.ErrInvalidInput
		}

		modelKind = kind[0]
	}

	now := time.Now().UTC()

	t := &domain.Task{
		ID:          u.idGen.NewID("task"),
		Name:        name,
		Description: strings.TrimSpace(description),
		Type:        taskType,
		Kind:        modelKind,
		CreatedAt:   now,
		UpdatedAt:   now,
	}

	err := u.tasks.Create(t)
	if err != nil {
		return nil, err
	}

	return t, nil
}

func (u *TaskUsecase) GetTask(id string) (*domain.Task, error) {
	return u.tasks.Get(id)
}

// UpdateTask applies a partial update to a task's editable fields (name,
// description, label set, JSON schema). Nil pointers leave a field unchanged.
// The label set is used to validate NER span labels; the JSON schema
// constrains Track B (extraction) outputs.
func (u *TaskUsecase) UpdateTask(id string, name, description *string, labelSet *[]string, jsonSchema *string) (*domain.Task, error) {
	t, err := u.tasks.Get(id)
	if err != nil {
		return nil, err
	}

	if name != nil {
		trimmed := strings.TrimSpace(*name)
		if trimmed == "" {
			return nil, domain.ErrInvalidInput
		}

		t.Name = trimmed
	}

	if description != nil {
		t.Description = strings.TrimSpace(*description)
	}

	if labelSet != nil {
		clean := make([]string, 0, len(*labelSet))

		for _, l := range *labelSet {
			l = strings.TrimSpace(l)
			if l != "" {
				clean = append(clean, l)
			}
		}

		t.LabelSet = clean
	}

	if jsonSchema != nil {
		schema := strings.TrimSpace(*jsonSchema)
		if schema != "" {
			err := domain.ValidateJSONSchema(schema)
			if err != nil {
				return nil, err
			}
		}

		t.JSONSchema = schema
	}

	t.UpdatedAt = time.Now().UTC()

	err = u.tasks.Update(t)
	if err != nil {
		return nil, err
	}

	return t, nil
}

func (u *TaskUsecase) ListTasks() ([]*domain.Task, error) {
	return u.tasks.List()
}

func (u *TaskUsecase) DeleteTask(id string) error {
	return u.tasks.Delete(id)
}
