package http

import (
	"errors"
	"net/http"

	"distillery/internal/domain"
	"distillery/internal/usecase"
)

type TaskHandler struct {
	uc *usecase.TaskUsecase
}

func NewTaskHandler(uc *usecase.TaskUsecase) *TaskHandler { return &TaskHandler{uc: uc} }

func (h *TaskHandler) Create(w http.ResponseWriter, r *http.Request) {
	var req createTaskRequest

	err := decodeJSON(r, &req)
	if err != nil {
		writeError(w, http.StatusBadRequest, "invalid JSON body")
		return
	}

	t, err := h.uc.CreateTask(req.Name, req.Description, domain.TaskType(req.Type), domain.ModelKind(req.Kind))
	if err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}

	writeJSON(w, http.StatusCreated, t)
}

func (h *TaskHandler) List(w http.ResponseWriter, _ *http.Request) {
	tasks, err := h.uc.ListTasks()
	if err != nil {
		writeError(w, http.StatusInternalServerError, err.Error())
		return
	}

	writeJSON(w, http.StatusOK, tasks)
}

func (h *TaskHandler) Get(w http.ResponseWriter, _ *http.Request, taskID string) {
	t, err := h.uc.GetTask(taskID)
	if err != nil {
		handleErr(w, err)
		return
	}

	writeJSON(w, http.StatusOK, t)
}

func (h *TaskHandler) Delete(w http.ResponseWriter, _ *http.Request, taskID string) {
	err := h.uc.DeleteTask(taskID)
	if err != nil {
		handleErr(w, err)
		return
	}

	w.WriteHeader(http.StatusNoContent)
}

func handleErr(w http.ResponseWriter, err error) {
	switch {
	case errors.Is(err, domain.ErrNotFound):
		writeError(w, http.StatusNotFound, err.Error())
	case errors.Is(err, domain.ErrInvalidInput):
		writeError(w, http.StatusBadRequest, err.Error())
	case errors.Is(err, domain.ErrNotReady):
		writeError(w, http.StatusUnprocessableEntity, err.Error())
	case errors.Is(err, domain.ErrAlreadyRunning):
		writeError(w, http.StatusConflict, err.Error())
	case errors.Is(err, domain.ErrNoDeployment):
		writeError(w, http.StatusUnprocessableEntity, err.Error())
	case errors.Is(err, domain.ErrNoModel):
		writeError(w, http.StatusUnprocessableEntity, err.Error())
	case errors.Is(err, domain.ErrUnauthorized):
		writeError(w, http.StatusUnauthorized, err.Error())
	default:
		writeError(w, http.StatusInternalServerError, err.Error())
	}
}
