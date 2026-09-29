package simulation_test

import (
	"encoding/json"
	"math"
	"os"
	"strconv"
	"testing"
	"time"

	"distillery/internal/domain"
	"distillery/internal/infra/simulation"
)

func runTable(t *testing.T, job *domain.TrainingJob, rows []map[string]interface{}) (*domain.TrainingMetrics, error) {
	t.Helper()

	var exs []*domain.Example //nolint:prealloc // test fixture

	for i, r := range rows {
		b, _ := json.Marshal(r) //nolint:errchkjson // test fixture marshalling
		exs = append(exs, &domain.Example{ID: strconv.Itoa(i), Payload: b})
	}

	type res struct {
		m   *domain.TrainingMetrics
		err error
	}

	ch := make(chan res, 1)

	simulation.NewTableFineTuner().Start(job, exs, nil, func(m *domain.TrainingMetrics, err error) { ch <- res{m, err} })

	select {
	case r := <-ch:
		return r.m, r.err
	case <-time.After(20 * time.Second):
		t.Fatal("timeout")

		return nil, os.ErrDeadlineExceeded
	}
}

func TestSimTabularBeatsBaseline(t *testing.T) {
	t.Parallel()

	var rows []map[string]interface{} //nolint:prealloc // test fixture

	for i := range 300 {
		x := float64(i%50) / 10
		label := "no"

		if x > 2.5 {
			label = "yes"
		}

		rows = append(rows, map[string]interface{}{"x": x, "noise": float64((i * 7) % 13), "y": label})
	}

	job := &domain.TrainingJob{Kind: domain.KindTabular, Tabular: &domain.TabularConfig{
		Target: "y", Task: domain.TabularClassification, Metric: "roc_auc", SplitStrategy: domain.SplitStratified,
		Columns: map[string]domain.TableColumnType{"x": domain.ColNumeric, "noise": domain.ColNumeric, "y": domain.ColCategorical},
	}}

	m, err := runTable(t, job, rows)
	if err != nil {
		t.Fatal(err)
	}

	if m.PrimaryVal < 0.9 || m.Baselines["roc_auc"] != 0.5 || m.FeatureImportance[0].Feature != "x" {
		t.Errorf("unexpected metrics: %+v", m)
	}

	if len(m.ConfusionMatrix) != 2 || len(m.ROCCurve) < 3 || m.TableBackend != "simulation-knn" {
		t.Errorf("missing artifacts: cm=%v roc=%d backend=%s", m.ConfusionMatrix, len(m.ROCCurve), m.TableBackend)
	}
}

func TestSimTabularRejectsSingleClass(t *testing.T) {
	t.Parallel()

	rows := make([]map[string]interface{}, 40)
	for i := range rows {
		rows[i] = map[string]interface{}{"x": float64(i), "y": "a"}
	}

	job := &domain.TrainingJob{Kind: domain.KindTabular, Tabular: &domain.TabularConfig{
		Target: "y", Task: domain.TabularClassification,
		Columns: map[string]domain.TableColumnType{"x": domain.ColNumeric, "y": domain.ColCategorical},
	}}

	_, err := runTable(t, job, rows)
	if err == nil {
		t.Error("single-class target must fail")
	}
}

func forecastJob(model string, h int) *domain.TrainingJob {
	return &domain.TrainingJob{Kind: domain.KindTimeSeries, Forecast: &domain.ForecastConfig{
		Timestamp: "ds", Target: "v", Frequency: "D", Horizon: h, BacktestWindows: 3, Model: model,
	}}
}

func TestSimForecastSeasonal(t *testing.T) {
	t.Parallel()

	start := time.Date(2024, 1, 1, 0, 0, 0, 0, time.UTC)

	var rows []map[string]interface{} //nolint:prealloc // test fixture

	for i := range 120 {
		rows = append(rows, map[string]interface{}{
			"ds": start.AddDate(0, 0, i).Format("2006-01-02"), "v": 50 + 10*math.Sin(2*math.Pi*float64(i)/7) + float64(i)*0.05,
		})
	}

	m, err := runTable(t, forecastJob("auto", 14), rows)
	if err != nil {
		t.Fatal(err)
	}

	if m.SeasonLength != 7 || m.BacktestPlot == nil || len(m.Backtest) != 3*len(m.Leaderboard) {
		t.Fatalf("unexpected forecast metrics: season=%d plot=%v windows=%d boards=%d", m.SeasonLength, m.BacktestPlot, len(m.Backtest), len(m.Leaderboard))
	}

	if m.PrimaryVal > m.Baselines["seasonal_naive_mase"]+1e-9 {
		t.Errorf("chosen model (%v) is worse than the baseline (%v)", m.PrimaryVal, m.Baselines)
	}
}

func TestSimForecastErrors(t *testing.T) {
	t.Parallel()

	start := time.Date(2024, 1, 1, 0, 0, 0, 0, time.UTC)

	var rows []map[string]interface{} //nolint:prealloc // test fixture

	for i := range 10 {
		rows = append(rows, map[string]interface{}{"ds": start.AddDate(0, 0, i).Format("2006-01-02"), "v": float64(i)})
	}

	_, err := runTable(t, forecastJob("auto", 14), rows)
	if err == nil {
		t.Error("a 10-point series cannot support a 14-step, 3-window backtest")
	}

	_, err = runTable(t, forecastJob("gbm", 2), rows)
	if err == nil {
		t.Error("gbm needs the Python trainer and must be refused by the simulation backend")
	}
}

func tableRowsFn(n int, fn func(i int) map[string]interface{}) []map[string]interface{} {
	rows := make([]map[string]interface{}, n)
	for i := range rows {
		rows[i] = fn(i)
	}

	return rows
}

func TestSimTabularRegression(t *testing.T) {
	t.Parallel()

	rows := tableRowsFn(240, func(i int) map[string]interface{} {
		x := float64(i % 40)

		return map[string]interface{}{"x": x, "junk": float64((i * 11) % 7), "y": 3*x + 5}
	})

	job := &domain.TrainingJob{Kind: domain.KindTabular, Tabular: &domain.TabularConfig{
		Target: "y", Task: domain.TabularRegression, Metric: "rmse",
		Columns: map[string]domain.TableColumnType{"x": domain.ColNumeric, "junk": domain.ColNumeric, "y": domain.ColNumeric},
	}}

	m, err := runTable(t, job, rows)
	if err != nil {
		t.Fatal(err)
	}

	if m.Primary != "rmse" || m.HigherIsBetter || m.PrimaryVal >= m.Baselines["rmse"] {
		t.Errorf("rmse should beat the mean predictor: %+v", m)
	}

	if len(m.Residuals) == 0 || m.FeatureSchema == nil || len(m.FeatureImportance) == 0 || m.FeatureImportance[0].Feature != "x" {
		t.Errorf("missing regression artifacts: %+v", m)
	}
}

func TestSimTabularSplitStrategies(t *testing.T) {
	t.Parallel()

	rows := tableRowsFn(300, func(i int) map[string]interface{} {
		label := "no"
		if i%3 == 0 {
			label = "yes"
		}

		return map[string]interface{}{
			"x": float64(i % 10), "when": "2024-" + strconv.Itoa(1+i/30%12+9*(i/360)) + "-01",
			"shop": "s" + strconv.Itoa(i%15), "y": label,
		}
	})

	cols := map[string]domain.TableColumnType{
		"x": domain.ColNumeric, "when": domain.ColDatetime, "shop": domain.ColCategorical, "y": domain.ColCategorical,
	}

	for _, split := range []domain.SplitStrategy{domain.SplitRandom, domain.SplitStratified, domain.SplitTime, domain.SplitGroup} {
		job := &domain.TrainingJob{Kind: domain.KindTabular, Tabular: &domain.TabularConfig{
			Target: "y", Task: domain.TabularClassification, SplitStrategy: split, GroupColumn: "shop", Columns: cols,
		}}

		m, err := runTable(t, job, rows)
		if err != nil {
			t.Fatalf("%s: %v", split, err)
		}

		if m.SplitUsed != string(split) {
			t.Errorf("%s: split used %q", split, m.SplitUsed)
		}

		if m.PrimaryVal < 0 || m.PrimaryVal > 1 || math.IsNaN(m.PrimaryVal) {
			t.Errorf("%s: implausible primary metric %v", split, m.PrimaryVal)
		}
	}
}

func TestSimTabularMissingConfigFails(t *testing.T) {
	t.Parallel()

	_, err := runTable(t, &domain.TrainingJob{Kind: domain.KindTabular}, []map[string]interface{}{{"x": 1.0}})
	if err == nil {
		t.Error("a job without a tabular config must fail cleanly, not panic")
	}
}

func TestSimForecastAllModelFamilies(t *testing.T) {
	t.Parallel()

	rows := tableRowsFn(120, func(i int) map[string]interface{} {
		return map[string]interface{}{
			"ds": time.Date(2024, 1, 1, 0, 0, 0, 0, time.UTC).AddDate(0, 0, i).Format("2006-01-02"),
			"v":  100 + float64(i%7)*4 + float64(i)*0.2,
		}
	})

	for _, model := range []string{"auto", "seasonal_naive", "stats"} {
		m, err := runTable(t, forecastJob(model, 7), rows)
		if err != nil {
			t.Fatalf("%s: %v", model, err)
		}

		if m.Primary == "" || len(m.Backtest) == 0 || m.FeatureSchema == nil && m.TableBackend == "" {
			t.Errorf("%s: incomplete forecast metrics %+v", model, m)
		}

		if m.Baselines["seasonal_naive"] == 0 && m.PrimaryVal == 0 {
			t.Errorf("%s: baseline and score both zero: %+v", model, m)
		}
	}
}
