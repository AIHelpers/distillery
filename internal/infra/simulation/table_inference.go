package simulation

import (
	"fmt"
	"math"
	"sort"
	"sync"

	"distillery/internal/domain"
)

// TableInferenceEngine serves the simulation backend's tabular and forecast
// models. Tabular: the distance-weighted kNN trained by TableFineTuner, with
// per-job preprocessing cached so a request only pays for the distance scan.
// top_factors are occlusion attributions: how much the predicted-class
// probability (or regression value) changes when that feature is removed
// from the distance. Forecast: the backtest-selected statistical model,
// refit on the stored history.
type TableInferenceEngine struct {
	mu    sync.Mutex
	cache map[string]*knnModel
}

// NewTableInferenceEngine builds the simulated tabular serving engine.
func NewTableInferenceEngine() *TableInferenceEngine {
	return &TableInferenceEngine{cache: map[string]*knnModel{}}
}

const servingK = 15

// PredictTable implements domain.TabularInferenceEngine.
func (e *TableInferenceEngine) PredictTable(
	job *domain.TrainingJob, rows domain.TableRowSource, schema *domain.FeatureSchema, row *domain.ValidatedRow,
) (domain.TablePrediction, error) {
	if schema == nil {
		return domain.TablePrediction{}, fmt.Errorf("%w: deployment has no feature schema", domain.ErrInvalidInput)
	}

	m, err := e.model(job, rows, schema)
	if err != nil {
		return domain.TablePrediction{}, err
	}

	x := m.encodeValidated(row)
	probs, value := m.score(x, servingK, -1)
	pred := domain.TablePrediction{Method: "occlusion"}

	var full float64

	if m.regression {
		pred.Prediction = round4(value)
		full = value
	} else {
		best := argmax(probs)
		pred.Prediction = m.classes[best]
		p := round4(probs[best])
		pred.Probability = &p
		pred.Probabilities = make(map[string]float64, len(probs))

		for i, c := range m.classes {
			pred.Probabilities[c] = round4(probs[i])
		}

		full = probs[best]
	}

	impacts := make([]domain.FeatureImpact, 0, len(m.feats))

	for fi, f := range m.feats {
		if f.typ == domain.FeatureDatetime || m.weights[fi] == 0 {
			continue
		}

		p2, v2 := m.score(x, servingK, fi)

		alt := v2
		if !m.regression {
			alt = p2[argmax(probs)]
		}

		impacts = append(impacts, domain.FeatureImpact{Feature: f.name, Impact: round4(full - alt)})
	}

	sort.Slice(impacts, func(i, j int) bool {
		a, b := math.Abs(impacts[i].Impact), math.Abs(impacts[j].Impact)
		if a != b {
			return a > b
		}

		return impacts[i].Feature < impacts[j].Feature
	})

	if len(impacts) > maxExplain {
		impacts = impacts[:maxExplain]
	}

	pred.TopFactors = impacts

	return pred, nil
}

// Forecast implements domain.ForecastInferenceEngine.
func (e *TableInferenceEngine) Forecast(
	job *domain.TrainingJob, rows domain.TableRowSource, req domain.ForecastRequest,
) (domain.ForecastResult, error) {
	if rows == nil {
		return domain.ForecastResult{}, fmt.Errorf("%w: no history available to forecast from", domain.ErrInvalidInput)
	}

	data, err := rows.JobRows(job)
	if err != nil {
		return domain.ForecastResult{}, err
	}

	return forecastFromRows(job, data, req)
}

func (e *TableInferenceEngine) model(job *domain.TrainingJob, rows domain.TableRowSource, schema *domain.FeatureSchema) (*knnModel, error) {
	e.mu.Lock()
	m, ok := e.cache[job.ID]
	e.mu.Unlock()

	if ok {
		return m, nil
	}

	if rows == nil {
		return nil, fmt.Errorf("%w: no training rows available to score against", domain.ErrInvalidInput)
	}

	data, err := rows.JobRows(job)
	if err != nil {
		return nil, err
	}

	weights := map[string]float64{}

	if job.Metrics != nil {
		for _, imp := range job.Metrics.FeatureImportance {
			weights[imp.Feature] = imp.Impact
		}
	}

	if len(weights) == 0 {
		weights = nil
	}

	m = buildKNN(data, schema, weights)
	if len(m.X) == 0 {
		return nil, fmt.Errorf("%w: no training rows available to score against", domain.ErrInvalidInput)
	}

	e.mu.Lock()
	e.cache[job.ID] = m
	e.mu.Unlock()

	return m, nil
}
