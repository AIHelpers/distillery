package http

import (
	"encoding/csv"
	"errors"
	"fmt"
	"io"
	"net/http"
	"sort"
	"strconv"
	"strings"

	"distillery/internal/domain"
	"distillery/internal/usecase"
)

// maxTabularUploadBytes caps a single table upload (50MB — comfortably above
// any dataset that fits the MVP's 200k-row cap).
const maxTabularUploadBytes = 50 << 20

// maxBatchBytes caps a CSV batch predict body.
const maxBatchBytes = 20 << 20

// TabularHandler serves the plan-08 tabular/time-series surface: table
// upload + column mapping + preview, tabular/forecast training, and the
// JSON predict / forecast / CSV batch inference endpoints.
type TabularHandler struct {
	uc  *usecase.TabularUsecase
	dep *usecase.DeploymentUsecase
}

// NewTabularHandler wires the handler.
func NewTabularHandler(uc *usecase.TabularUsecase, dep *usecase.DeploymentUsecase) *TabularHandler {
	return &TabularHandler{uc: uc, dep: dep}
}

// ---------- Table datasets (upload / mapper / preview) ----------.

// UploadTable handles POST /api/v1/tasks/{taskID}/tables as multipart
// ("file" field) with optional target / timestamp / item_id_column /
// frequency / horizon form fields, returning the inferred TableDataset for
// the column mapper.
func (h *TabularHandler) UploadTable(w http.ResponseWriter, r *http.Request, taskID string) {
	r.Body = http.MaxBytesReader(w, r.Body, maxTabularUploadBytes+(1<<20))

	err := r.ParseMultipartForm(8 << 20)
	if err != nil {
		var tooBig *http.MaxBytesError
		if errors.As(err, &tooBig) {
			writeError(w, http.StatusRequestEntityTooLarge, "table upload is larger than 50MB")

			return
		}

		writeError(w, http.StatusBadRequest, "expected a multipart/form-data body with a \"file\" field")

		return
	}

	file, header, err := r.FormFile("file")
	if err != nil {
		writeError(w, http.StatusBadRequest, "no CSV file found (expected form field \"file\")")

		return
	}

	defer file.Close()

	content, err := io.ReadAll(io.LimitReader(file, maxTabularUploadBytes+1))
	if err != nil {
		writeError(w, http.StatusBadRequest, "failed to read uploaded table")

		return
	}

	if len(content) > maxTabularUploadBytes {
		writeError(w, http.StatusRequestEntityTooLarge, "table upload is larger than 50MB")

		return
	}

	horizon := 0

	if v := strings.TrimSpace(r.FormValue("horizon")); v != "" {
		horizon, err = strconv.Atoi(v)
		if err != nil || horizon < 0 {
			writeError(w, http.StatusBadRequest, "\"horizon\" must be a positive integer")

			return
		}
	}

	td, err := h.uc.UploadTable(taskID, usecase.UploadTableRequest{
		Filename:  header.Filename,
		Content:   content,
		Target:    strings.TrimSpace(r.FormValue("target")),
		Timestamp: strings.TrimSpace(r.FormValue("timestamp")),
		ItemIDCol: strings.TrimSpace(r.FormValue("item_id_column")),
		Frequency: strings.TrimSpace(r.FormValue("frequency")),
		Horizon:   horizon,
	})
	if err != nil {
		handleErr(w, err)

		return
	}

	writeJSON(w, http.StatusCreated, td)
}

// ListTables GET /api/v1/tasks/{taskID}/tables.
func (h *TabularHandler) ListTables(w http.ResponseWriter, _ *http.Request, taskID string) {
	tables, err := h.uc.ListTables(taskID)
	if err != nil {
		handleErr(w, err)

		return
	}

	if tables == nil {
		tables = []*domain.TableDataset{}
	}

	writeJSON(w, http.StatusOK, tables)
}

// GetTable GET /api/v1/tasks/{taskID}/tables/{tableID}.
func (h *TabularHandler) GetTable(w http.ResponseWriter, _ *http.Request, taskID, tableID string) {
	td, err := h.uc.GetTable(taskID, tableID)
	if err != nil {
		handleErr(w, err)

		return
	}

	writeJSON(w, http.StatusOK, td)
}

// DeleteTable DELETE /api/v1/tasks/{taskID}/tables/{tableID}.
func (h *TabularHandler) DeleteTable(w http.ResponseWriter, _ *http.Request, taskID, tableID string) {
	err := h.uc.DeleteTable(taskID, tableID)
	if err != nil {
		handleErr(w, err)

		return
	}

	w.WriteHeader(http.StatusNoContent)
}

// tableMappingRequest is the column-mapper payload.
type tableMappingRequest struct {
	Target      string                            `json:"target"`
	Types       map[string]domain.TableColumnType `json:"types"`
	Excluded    *[]string                         `json:"excluded"`
	Timestamp   string                            `json:"timestamp"`
	ItemID      string                            `json:"item_id"`
	Split       string                            `json:"split"`
	GroupColumn *string                           `json:"group_column"`
	Task        string                            `json:"task"`
	Frequency   string                            `json:"frequency"`
	Horizon     int                               `json:"horizon"`
}

// UpdateTableMapping PUT /api/v1/tasks/{taskID}/tables/{tableID}/mapping.
func (h *TabularHandler) UpdateTableMapping(w http.ResponseWriter, r *http.Request, taskID, tableID string) {
	var req tableMappingRequest

	err := decodeJSON(r, &req)
	if err != nil {
		writeError(w, http.StatusBadRequest, "invalid JSON body: "+err.Error())

		return
	}

	td, err := h.uc.UpdateTableMapping(taskID, tableID, usecase.MappingUpdate{
		Target: req.Target, Types: req.Types, Excluded: req.Excluded, Timestamp: req.Timestamp,
		ItemID: req.ItemID, Split: domain.SplitStrategy(req.Split), GroupColumn: req.GroupColumn,
		Task: req.Task, Frequency: req.Frequency, Horizon: req.Horizon,
	})
	if err != nil {
		handleErr(w, err)

		return
	}

	writeJSON(w, http.StatusOK, td)
}

// TablePreview GET /api/v1/tasks/{taskID}/tables/{tableID}/preview?limit=20.
func (h *TabularHandler) TablePreview(w http.ResponseWriter, r *http.Request, taskID, tableID string) {
	limit, _ := strconv.Atoi(r.URL.Query().Get("limit"))
	if limit <= 0 || limit > 100 {
		limit = 20
	}

	td, err := h.uc.GetTable(taskID, tableID)
	if err != nil {
		handleErr(w, err)

		return
	}

	rows, err := h.uc.TablePreview(taskID, tableID, limit)
	if err != nil {
		handleErr(w, err)

		return
	}

	if rows == nil {
		rows = []map[string]interface{}{}
	}

	cols := make([]string, len(td.Columns))
	for i, c := range td.Columns {
		cols[i] = c.Name
	}

	writeJSON(w, http.StatusOK, map[string]interface{}{"columns": cols, "rows": rows})
}

// ---------- Training ----------.

// startTabularTrainingRequest is the POST /training/tabular body. The target
// and feature set always come from the stored table mapping.
type startTabularTrainingRequest struct {
	TableID       string `json:"table_id"`
	Model         string `json:"model,omitempty"`
	Task          string `json:"task,omitempty"`
	Metric        string `json:"metric,omitempty"`
	TimeBudgetSec int    `json:"time_budget_sec,omitempty"`
	CVFolds       int    `json:"cv_folds,omitempty"`
	Split         string `json:"split_strategy,omitempty"`
	GroupColumn   string `json:"group_column,omitempty"`
}

// StartTabularTraining POST /api/v1/tasks/{taskID}/training/tabular.
func (h *TabularHandler) StartTabularTraining(w http.ResponseWriter, r *http.Request, taskID string) {
	var req startTabularTrainingRequest

	err := decodeJSON(r, &req)
	if err != nil {
		writeError(w, http.StatusBadRequest, "invalid JSON body: "+err.Error())

		return
	}

	if strings.TrimSpace(req.TableID) == "" {
		writeError(w, http.StatusBadRequest, "\"table_id\" is required")

		return
	}

	job, err := h.uc.StartTabularTraining(taskID, req.TableID, domain.TabularConfig{
		Model:         req.Model,
		Task:          req.Task,
		Metric:        req.Metric,
		TimeBudgetSec: req.TimeBudgetSec,
		CVFolds:       req.CVFolds,
		SplitStrategy: domain.SplitStrategy(req.Split),
		GroupColumn:   req.GroupColumn,
	})
	if err != nil {
		handleErr(w, err)

		return
	}

	writeJSON(w, http.StatusAccepted, job)
}

// startForecastTrainingRequest is the POST /training/forecast body.
type startForecastTrainingRequest struct {
	TableID         string `json:"table_id"`
	Horizon         int    `json:"horizon,omitempty"`
	Frequency       string `json:"frequency,omitempty"`
	Metric          string `json:"metric,omitempty"`
	BacktestWindows int    `json:"backtest_windows,omitempty"`
	Model           string `json:"model,omitempty"`
	SeasonLength    int    `json:"season_length,omitempty"`
	TimeBudgetSec   int    `json:"time_budget_sec,omitempty"`
}

// StartForecastTraining POST /api/v1/tasks/{taskID}/training/forecast.
func (h *TabularHandler) StartForecastTraining(w http.ResponseWriter, r *http.Request, taskID string) {
	var req startForecastTrainingRequest

	err := decodeJSON(r, &req)
	if err != nil {
		writeError(w, http.StatusBadRequest, "invalid JSON body: "+err.Error())

		return
	}

	if strings.TrimSpace(req.TableID) == "" {
		writeError(w, http.StatusBadRequest, "\"table_id\" is required")

		return
	}

	job, err := h.uc.StartForecastTraining(taskID, req.TableID, domain.ForecastConfig{
		Horizon:         req.Horizon,
		Frequency:       req.Frequency,
		Metric:          req.Metric,
		BacktestWindows: req.BacktestWindows,
		Model:           req.Model,
		SeasonLength:    req.SeasonLength,
		TimeBudgetSec:   req.TimeBudgetSec,
	})
	if err != nil {
		handleErr(w, err)

		return
	}

	writeJSON(w, http.StatusAccepted, job)
}

// ---------- Inference ----------.

// tabularPredictRequest is the JSON body for tabular /predict:
// {"input": {"age": 41, "plan": "pro", "tenure_months": 18}}.
type tabularPredictRequest struct {
	Input map[string]interface{} `json:"input"`
}

// Predict serves POST /api/v1/inference/{deploymentID}/predict for tabular
// deployments. Response: {"kind":"tabular","result":{"prediction":...,
// "probability":...,"top_factors":[...]}}.
func (h *TabularHandler) Predict(w http.ResponseWriter, r *http.Request, deploymentID string) {
	var req tabularPredictRequest

	err := decodeJSON(r, &req)
	if err != nil {
		writeError(w, http.StatusBadRequest, "invalid JSON body: "+err.Error())

		return
	}

	if req.Input == nil {
		writeError(w, http.StatusBadRequest, "\"input\" must be an object of feature values")

		return
	}

	pred, err := h.dep.InvokeTable(deploymentID, apiKeyFromRequest(r), req.Input)
	if err != nil {
		handleErr(w, err)

		return
	}

	writeJSON(w, http.StatusOK, map[string]interface{}{"kind": "tabular", "result": pred})
}

// forecastRequest is the JSON body for /forecast:
// {"history": [...], "horizon": 14, "item_id": "sku-1"} (history optional —
// defaults to the training series stored with the model).
type forecastRequest struct {
	History []map[string]interface{} `json:"history,omitempty"`
	Horizon int                      `json:"horizon,omitempty"`
	ItemID  string                   `json:"item_id,omitempty"`
}

// Forecast serves POST /api/v1/inference/{deploymentID}/forecast. Response:
// {"kind":"time_series","result":{"forecast":[...],"lower":[...],"upper":[...],...}}.
func (h *TabularHandler) Forecast(w http.ResponseWriter, r *http.Request, deploymentID string) {
	var req forecastRequest

	err := decodeJSON(r, &req)
	if err != nil {
		writeError(w, http.StatusBadRequest, "invalid JSON body: "+err.Error())

		return
	}

	result, err := h.dep.InvokeForecast(deploymentID, apiKeyFromRequest(r), domain.ForecastRequest{
		History: req.History, Horizon: req.Horizon, ItemID: req.ItemID,
	})
	if err != nil {
		handleErr(w, err)

		return
	}

	values := make([]float64, 0, len(result.Forecast))
	lower := make([]float64, 0, len(result.Forecast))
	upper := make([]float64, 0, len(result.Forecast))

	for _, p := range result.Forecast {
		values = append(values, p.Value)
		lower = append(lower, p.Lower)
		upper = append(upper, p.Upper)
	}

	writeJSON(w, http.StatusOK, map[string]interface{}{
		"kind": "time_series",
		"result": map[string]interface{}{
			"forecast":  values,
			"lower":     lower,
			"upper":     upper,
			"points":    result.Forecast,
			"item_id":   result.ItemID,
			"model":     result.Model,
			"metric":    result.Metric,
			"baselines": result.Baselines,
		},
	})
}

// PredictBatch serves the CSV batch endpoint: for tabular deployments a CSV of
// feature columns in, the same CSV plus prediction / probability / top-factor
// / error columns out; for time-series deployments a CSV of history rows in
// (plus ?horizon=N), a forecast CSV out.
func (h *TabularHandler) PredictBatch(w http.ResponseWriter, r *http.Request, deploymentID string) {
	kind, err := h.dep.DeploymentKind(deploymentID)
	if err != nil {
		handleErr(w, err)

		return
	}

	if kind != domain.KindTabular && kind != domain.KindTimeSeries {
		writeError(w, http.StatusBadRequest, "table batch predict is only available for tabular and time_series deployments")

		return
	}

	body, err := io.ReadAll(http.MaxBytesReader(w, r.Body, maxBatchBytes))
	if err != nil {
		writeError(w, http.StatusRequestEntityTooLarge, "batch body is larger than 20MB")

		return
	}

	rows, headers, err := usecase.ParseTableCSV(body)
	if err != nil || len(rows) == 0 {
		writeError(w, http.StatusBadRequest, "expected a CSV with a header row and at least one data row")

		return
	}

	if kind == domain.KindTimeSeries {
		h.forecastBatch(w, r, deploymentID, rows)

		return
	}

	results, err := h.dep.InvokeTableBatch(deploymentID, apiKeyFromRequest(r), rows)
	if err != nil {
		handleErr(w, err)

		return
	}

	writeTableBatchCSV(w, headers, rows, results)
}

func (h *TabularHandler) forecastBatch(w http.ResponseWriter, r *http.Request, deploymentID string, history []map[string]interface{}) {
	horizon, _ := strconv.Atoi(r.URL.Query().Get("horizon"))

	results, err := h.dep.InvokeForecastBatch(deploymentID, apiKeyFromRequest(r), history, horizon)
	if err != nil {
		handleErr(w, err)

		return
	}

	w.Header().Set("Content-Type", "text/csv")
	w.Header().Set("Content-Disposition", `attachment; filename="forecast.csv"`)

	cw := csv.NewWriter(w)
	_ = cw.Write([]string{"item_id", "timestamp", "forecast", "lower", "upper"})

	for _, res := range results {
		for _, p := range res.Forecast {
			_ = cw.Write([]string{
				res.ItemID, p.Timestamp,
				strconv.FormatFloat(p.Value, 'g', -1, 64),
				strconv.FormatFloat(p.Lower, 'g', -1, 64),
				strconv.FormatFloat(p.Upper, 'g', -1, 64),
			})
		}
	}

	cw.Flush()
}

const batchTopFactors = 3

func writeTableBatchCSV(w http.ResponseWriter, headers []string, rows []map[string]interface{}, results []usecase.TableBatchResult) {
	var classes []string

	seen := map[string]bool{}

	for _, res := range results {
		for c := range res.Prediction.Probabilities {
			if !seen[c] {
				seen[c] = true
				classes = append(classes, c)
			}
		}
	}

	sort.Strings(classes)

	w.Header().Set("Content-Type", "text/csv")
	w.Header().Set("Content-Disposition", `attachment; filename="predictions.csv"`)

	cw := csv.NewWriter(w)

	head := append([]string{}, headers...)
	head = append(head, "prediction", "probability")

	for _, c := range classes {
		head = append(head, "probability_"+c)
	}

	for i := 1; i <= batchTopFactors; i++ {
		head = append(head, "factor_"+strconv.Itoa(i), "impact_"+strconv.Itoa(i))
	}

	head = append(head, "error")
	_ = cw.Write(head)

	for i, res := range results {
		out := make([]string, 0, len(head))

		for _, hd := range headers {
			s, _ := rows[i][hd].(string)
			out = append(out, csvSafe(s))
		}

		if res.Error != "" {
			for len(out) < len(head)-1 {
				out = append(out, "")
			}

			out = append(out, res.Error)
			_ = cw.Write(out)

			continue
		}

		p := res.Prediction
		out = append(out, csvSafe(anyString(p.Prediction)))

		if p.Probability != nil {
			out = append(out, strconv.FormatFloat(*p.Probability, 'g', -1, 64))
		} else {
			out = append(out, "")
		}

		for _, c := range classes {
			out = append(out, strconv.FormatFloat(p.Probabilities[c], 'g', -1, 64))
		}

		for k := range batchTopFactors {
			if k < len(p.TopFactors) {
				out = append(out, csvSafe(p.TopFactors[k].Feature), strconv.FormatFloat(p.TopFactors[k].Impact, 'g', -1, 64))
			} else {
				out = append(out, "", "")
			}
		}

		out = append(out, "")
		_ = cw.Write(out)
	}

	cw.Flush()
}

func anyString(v interface{}) string {
	switch x := v.(type) {
	case string:
		return x
	case float64:
		return strconv.FormatFloat(x, 'g', -1, 64)
	case nil:
		return ""
	default:
		return fmt.Sprint(v)
	}
}

// csvSafe neutralises spreadsheet formula injection in echoed cells.
func csvSafe(s string) string {
	if s != "" && strings.ContainsRune("=+-@\t\r", rune(s[0])) {
		_, err := strconv.ParseFloat(s, 64)
		if err == nil {
			return s
		}

		return "'" + s
	}

	return s
}
