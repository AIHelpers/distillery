package http

import (
	"net/http"

	"distillery/internal/domain"
	"distillery/internal/usecase"
)

// FineTuneHandler exposes the coding AI fine-tuning API.
type FineTuneHandler struct {
	uc *usecase.FineTuneUsecase
}

func NewFineTuneHandler(uc *usecase.FineTuneUsecase) *FineTuneHandler {
	return &FineTuneHandler{uc: uc}
}

func (h *FineTuneHandler) CreateRequest(w http.ResponseWriter, r *http.Request) {
	var req domain.FineTuneRequest
	if err := decodeJSON(r, &req); err != nil {
		writeError(w, http.StatusBadRequest, "invalid JSON body")
		return
	}
	req.Owner = r.Header.Get("X-User-ID")
	if req.Owner == "" {
		req.Owner = "anonymous"
	}
	id, err := h.uc.CreateRequest(&req)
	if err != nil {
		handleErr(w, err)
		return
	}
	writeJSON(w, http.StatusCreated, map[string]interface{}{"request_id": id, "status": "draft"})
}

func (h *FineTuneHandler) StartTraining(w http.ResponseWriter, r *http.Request, requestID string) {
	jobID, err := h.uc.StartTraining(r.Context(), requestID)
	if err != nil {
		handleErr(w, err)
		return
	}
	writeJSON(w, http.StatusAccepted, map[string]interface{}{"job_id": jobID, "status": "queued"})
}

func (h *FineTuneHandler) JobStatus(w http.ResponseWriter, r *http.Request, jobID string) {
	job, err := h.uc.JobStatus(jobID)
	if err != nil {
		handleErr(w, err)
		return
	}
	writeJSON(w, http.StatusOK, job)
}

func (h *FineTuneHandler) JobMonitor(w http.ResponseWriter, r *http.Request, jobID string) {
	opt, err := h.uc.MonitorTraining(r.Context(), jobID)
	if err != nil {
		handleErr(w, err)
		return
	}
	writeJSON(w, http.StatusOK, opt)
}

func (h *FineTuneHandler) JobInsights(w http.ResponseWriter, r *http.Request, jobID string) {
	insights, err := h.uc.Insights(r.Context(), jobID)
	if err != nil {
		handleErr(w, err)
		return
	}
	writeJSON(w, http.StatusOK, insights)
}

func (h *FineTuneHandler) JobQuality(w http.ResponseWriter, r *http.Request, jobID string) {
	report, err := h.uc.QualityReport(r.Context(), jobID)
	if err != nil {
		handleErr(w, err)
		return
	}
	writeJSON(w, http.StatusOK, report)
}

func (h *FineTuneHandler) ListJobs(w http.ResponseWriter, r *http.Request) {
	jobs, err := h.uc.ListJobs()
	if err != nil {
		handleErr(w, err)
		return
	}
	writeJSON(w, http.StatusOK, jobs)
}

func (h *FineTuneHandler) ListRequests(w http.ResponseWriter, r *http.Request) {
	owner := r.Header.Get("X-User-ID")
	if owner == "" {
		owner = "anonymous"
	}
	reqs, err := h.uc.ListRequests(owner)
	if err != nil {
		handleErr(w, err)
		return
	}
	writeJSON(w, http.StatusOK, reqs)
}

func (h *FineTuneHandler) ListModels(w http.ResponseWriter, r *http.Request) {
	lang := r.URL.Query().Get("language")
	skill := r.URL.Query().Get("skill")
	models, err := h.uc.ListModels(lang, skill)
	if err != nil {
		handleErr(w, err)
		return
	}
	writeJSON(w, http.StatusOK, models)
}

func (h *FineTuneHandler) RegisterDataset(w http.ResponseWriter, r *http.Request) {
	var d domain.DatasetInfo
	if err := decodeJSON(r, &d); err != nil {
		writeError(w, http.StatusBadRequest, "invalid JSON body")
		return
	}
	d.Owner = r.Header.Get("X-User-ID")
	if d.Owner == "" {
		d.Owner = "anonymous"
	}
	id, err := h.uc.RegisterDataset(&d)
	if err != nil {
		handleErr(w, err)
		return
	}
	writeJSON(w, http.StatusCreated, map[string]interface{}{"dataset_id": id, "status": d.Status, "quality": d.Quality})
}

func (h *FineTuneHandler) AnalyseDataset(w http.ResponseWriter, r *http.Request, datasetID string) {
	analysis, err := h.uc.AnalyseDataset(r.Context(), datasetID)
	if err != nil {
		handleErr(w, err)
		return
	}
	writeJSON(w, http.StatusOK, analysis)
}

func (h *FineTuneHandler) RecommendHyperparams(w http.ResponseWriter, r *http.Request) {
	var req domain.HyperparameterRecommendationReq
	if err := decodeJSON(r, &req); err != nil {
		writeError(w, http.StatusBadRequest, "invalid JSON body")
		return
	}
	params, err := h.uc.RecommendHyperparameters(r.Context(), req)
	if err != nil {
		handleErr(w, err)
		return
	}
	writeJSON(w, http.StatusOK, params)
}

func (h *FineTuneHandler) ExportModel(w http.ResponseWriter, r *http.Request, modelID string) {
	url, err := h.uc.ExportModel(modelID)
	if err != nil {
		handleErr(w, err)
		return
	}
	writeJSON(w, http.StatusOK, map[string]interface{}{"model_id": modelID, "huggingface_url": url, "status": "ready"})
}

func (h *FineTuneHandler) GetModel(w http.ResponseWriter, r *http.Request, modelID string) {
	m, err := h.uc.GetModel(modelID)
	if err != nil {
		handleErr(w, err)
		return
	}
	writeJSON(w, http.StatusOK, m)
}
