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

func (u *TaskUsecase) CreateTask(name, description string, taskType domain.TaskType) (*domain.Task, error) {
	name = strings.TrimSpace(name)
	if name == "" {
		return nil, domain.ErrInvalidInput
	}

	switch taskType {
	case domain.TaskClassification, domain.TaskExtraction, domain.TaskGeneration:
	default:
		return nil, domain.ErrInvalidInput
	}

	now := time.Now().UTC()

	t := &domain.Task{
		ID:          u.idGen.NewID("task"),
		Name:        name,
		Description: strings.TrimSpace(description),
		Type:        taskType,
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
