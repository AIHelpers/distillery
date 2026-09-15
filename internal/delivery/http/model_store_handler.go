package http

import (
	"net/http"

	"distillery/internal/domain"
	"distillery/internal/usecase"
)

// ModelStoreHandler exposes the model-store configuration and
// choose/download/upload endpoints for base and trained models.
type ModelStoreHandler struct {
	uc *usecase.ModelStoreUsecase
}

// NewModelStoreHandler wires the model store usecase into HTTP.
func NewModelStoreHandler(uc *usecase.ModelStoreUsecase) *ModelStoreHandler {
	return &ModelStoreHandler{uc: uc}
}

// createStoreRequest is the JSON body for POST /api/v1/model-stores.
type createStoreRequest struct {
	Name   string                  `json:"name"`
	Type   domain.ModelStoreType   `json:"type"`
	Kind   domain.ModelStoreKind   `json:"kind"`
	Config domain.ModelStoreConfig `json:"config"`
}

// updateStoreRequest is the JSON body for PATCH /api/v1/model-stores/{id}.
type updateStoreRequest struct {
	Name    string                  `json:"name,omitempty"`
	Enabled *bool                   `json:"enabled"`
	Config  domain.ModelStoreConfig `json:"config"`
}

// uploadRequest is the JSON body for uploads.
type uploadRequest struct {
	LocalPath    string `json:"local_path"`
	ArtifactName string `json:"artifact_name"`
	BaseModel    string `json:"base_model,omitempty"`
}

// downloadRequest specifies which model entry to download.
type downloadRequest struct {
	File domain.ModelFile `json:"file"`
}

func (h *ModelStoreHandler) CreateStore(w http.ResponseWriter, r *http.Request) {
	var req createStoreRequest

	err := decodeJSON(r, &req)
	if err != nil {
		writeError(w, http.StatusBadRequest, "invalid JSON body")
		return
	}

	s, err := h.uc.CreateStore(req.Name, req.Type, req.Kind, req.Config)
	if err != nil {
		handleErr(w, err)
		return
	}

	writeJSON(w, http.StatusCreated, s)
}

func (h *ModelStoreHandler) ListStores(w http.ResponseWriter, r *http.Request) {
	kind := domain.ModelStoreKind(r.URL.Query().Get("kind"))

	list, err := h.uc.ListStores(kind)
	if err != nil {
		handleErr(w, err)
		return
	}

	writeJSON(w, http.StatusOK, list)
}

func (h *ModelStoreHandler) GetStore(w http.ResponseWriter, _ *http.Request, storeID string) {
	s, err := h.uc.GetStore(storeID)
	if err != nil {
		handleErr(w, err)
		return
	}

	writeJSON(w, http.StatusOK, s)
}

func (h *ModelStoreHandler) UpdateStore(w http.ResponseWriter, r *http.Request, storeID string) {
	var req updateStoreRequest

	err := decodeJSON(r, &req)
	if err != nil {
		writeError(w, http.StatusBadRequest, "invalid JSON body")
		return
	}

	enabled := true
	if req.Enabled != nil {
		enabled = *req.Enabled
	}

	s, err := h.uc.UpdateStore(storeID, req.Name, enabled, req.Config)
	if err != nil {
		handleErr(w, err)
		return
	}

	writeJSON(w, http.StatusOK, s)
}

func (h *ModelStoreHandler) DeleteStore(w http.ResponseWriter, _ *http.Request, storeID string) {
	err := h.uc.DeleteStore(storeID)
	if err != nil {
		handleErr(w, err)
		return
	}

	w.WriteHeader(http.StatusNoContent)
}

// ListModels returns base models available in a base-model store.
func (h *ModelStoreHandler) ListModels(w http.ResponseWriter, _ *http.Request, storeID string) {
	files, err := h.uc.ListModels(storeID)
	if err != nil {
		handleErr(w, err)
		return
	}

	writeJSON(w, http.StatusOK, files)
}

// ListTrainedModels returns trained model artifacts in a trained-model store.
func (h *ModelStoreHandler) ListTrainedModels(w http.ResponseWriter, _ *http.Request, storeID string) {
	files, err := h.uc.ListTrainedModels(storeID)
	if err != nil {
		handleErr(w, err)
		return
	}

	writeJSON(w, http.StatusOK, files)
}

// DownloadModel downloads a base model from a store into the local cache.
func (h *ModelStoreHandler) DownloadModel(w http.ResponseWriter, r *http.Request, storeID string) {
	var req downloadRequest

	err := decodeJSON(r, &req)
	if err != nil {
		writeError(w, http.StatusBadRequest, "invalid JSON body")
		return
	}

	path, err := h.uc.DownloadModel(storeID, &req.File)
	if err != nil {
		handleErr(w, err)
		return
	}

	writeJSON(w, http.StatusOK, map[string]string{"local_path": path})
}

// UploadBaseModel pushes a local model directory into a base-model store.
func (h *ModelStoreHandler) UploadBaseModel(w http.ResponseWriter, r *http.Request, storeID string) {
	var req uploadRequest

	err := decodeJSON(r, &req)
	if err != nil {
		writeError(w, http.StatusBadRequest, "invalid JSON body")
		return
	}

	file, err := h.uc.UploadBaseModel(storeID, req.LocalPath, req.ArtifactName)
	if err != nil {
		handleErr(w, err)
		return
	}

	writeJSON(w, http.StatusCreated, file)
}

// UploadTrainedModel pushes a local trained model directory into a
// trained-model store.
func (h *ModelStoreHandler) UploadTrainedModel(w http.ResponseWriter, r *http.Request, storeID string) {
	var req uploadRequest

	err := decodeJSON(r, &req)
	if err != nil {
		writeError(w, http.StatusBadRequest, "invalid JSON body")
		return
	}

	file, err := h.uc.UploadTrainedModel(storeID, req.LocalPath, req.ArtifactName, req.BaseModel)
	if err != nil {
		handleErr(w, err)
		return
	}

	writeJSON(w, http.StatusCreated, file)
}
