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

func (u *TaskUsecase) ListTasks() ([]*domain.Task, error) {
	return u.tasks.List()
}

func (u *TaskUsecase) DeleteTask(id string) error {
	return u.tasks.Delete(id)
}
