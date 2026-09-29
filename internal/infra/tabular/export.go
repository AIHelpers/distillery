package tabular

import (
	"archive/zip"
	"bytes"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"regexp"

	"distillery/internal/domain"
)

// Exporter builds the portable package for tabular / time-series models:
// the native (non-pickle) model files, the feature schema, metrics, the
// standalone scoring library and a runnable Python example. Jobs without
// trained artifacts, and all other kinds, use the wrapped exporter.
type Exporter struct {
	base       domain.Exporter
	jobsDir    string
	trainerDir string
}

// NewExporter wraps base. trainerDir is the directory holding
// tablelib.py (normally ./trainer).
func NewExporter(base domain.Exporter, jobsDir, trainerDir string) *Exporter {
	return &Exporter{base: base, jobsDir: jobsDir, trainerDir: trainerDir}
}

var exportFiles = regexp.MustCompile(`^(model\.(txt|json|ubj|cbm)|forecast_model\.(json|txt)|schema\.json|metrics\.json|importance\.csv|backtest\.json)$`)

var unsafeName = regexp.MustCompile(`[^A-Za-z0-9._-]+`)

// BuildExport implements domain.Exporter.
func (e *Exporter) BuildExport(task *domain.Task, job *domain.TrainingJob) (data []byte, name string, err error) {
	if job == nil || (job.Kind != domain.KindTabular && job.Kind != domain.KindTimeSeries) || filepath.Base(job.ID) != job.ID {
		return e.base.BuildExport(task, job)
	}

	dir := filepath.Join(e.jobsDir, job.ID)

	_, err = os.Stat(filepath.Join(dir, "schema.json"))
	if err != nil {
		return e.base.BuildExport(task, job)
	}

	entries, err := os.ReadDir(dir)
	if err != nil {
		return nil, "", err
	}

	var buf bytes.Buffer

	zw := zip.NewWriter(&buf)

	add := func(name string, data []byte) error {
		w, err := zw.Create(name)
		if err != nil {
			return err
		}

		_, err = w.Write(data)

		return err
	}

	for _, en := range entries {
		if en.IsDir() || !exportFiles.MatchString(en.Name()) {
			continue
		}

		data, rerr := os.ReadFile(filepath.Join(dir, en.Name()))
		if rerr != nil {
			return nil, "", rerr
		}

		err = add(en.Name(), data)
		if err != nil {
			return nil, "", err
		}
	}

	lib, err := os.ReadFile(filepath.Join(e.trainerDir, "tablelib.py"))
	if err != nil {
		return nil, "", fmt.Errorf("scoring library not found: %w", err)
	}

	example, err := e.exampleScript(dir, job)
	if err != nil {
		return nil, "", err
	}

	for name, data := range map[string][]byte{
		"tablelib.py":        lib,
		"predict_example.py": example,
		"README.md":          []byte(readme(job)),
	} {
		err = add(name, data)
		if err != nil {
			return nil, "", err
		}
	}

	err = zw.Close()
	if err != nil {
		return nil, "", err
	}

	name = "model"
	if task != nil {
		name = unsafeName.ReplaceAllString(task.Name, "_")
	}

	return buf.Bytes(), fmt.Sprintf("%s-v%d-%s.zip", name, job.Version, job.Kind), nil
}

func (e *Exporter) exampleScript(dir string, job *domain.TrainingJob) ([]byte, error) {
	if job.Kind == domain.KindTimeSeries {
		return []byte(`"""Forecast with the exported model (needs numpy; lightgbm only for gbm models)."""
import json
from tablelib import load

model = load(".")
# History is optional: without it the series stored with the model is used.
ids = model.series_ids()
print(json.dumps(model.forecast(horizon=14, item_id=ids[0]), indent=2))
`), nil
	}

	raw, err := os.ReadFile(filepath.Join(dir, "schema.json"))
	if err != nil {
		return nil, err
	}

	var sch struct {
		Features []domain.TableFeature `json:"preprocessor"`
	}

	err = json.Unmarshal(raw, &sch)
	if err != nil {
		return nil, err
	}

	row := map[string]interface{}{}

	for _, f := range sch.Features {
		switch {
		case f.Type == domain.FeatureNumeric && f.Min != nil && f.Max != nil:
			row[f.Name] = (*f.Min + *f.Max) / 2
		case f.Type == domain.FeatureNumeric:
			row[f.Name] = 0
		case f.Type == domain.FeatureDatetime:
			row[f.Name] = "2024-01-01T00:00:00"
		case len(f.Categories) > 0:
			row[f.Name] = f.Categories[0]
		default:
			row[f.Name] = ""
		}
	}

	b, err := json.MarshalIndent(row, "", "    ")
	if err != nil {
		return nil, err
	}

	return []byte(fmt.Sprintf(`"""Score one row with the exported model (needs numpy; lightgbm/xgboost/catboost
only if the model was trained with that library — see metrics.json)."""
import json
from tablelib import load

model = load(".")
row = %s
print(json.dumps(model.predict(row), indent=2))
`, string(b))), nil
}

func readme(job *domain.TrainingJob) string {
	return fmt.Sprintf(`# Exported %s model (job %s)

Files:
- model.* / forecast_model.* — native model files (never pickle)
- schema.json — feature schema and preprocessing parameters
- metrics.json — holdout / backtest metrics and the leaderboard
- tablelib.py — dependency-light scoring library (numpy; the training library only when used)
- predict_example.py — runnable example: python predict_example.py

The serving API validates inputs against the deployment's feature schema;
tablelib performs the same checks and raises a ValueError with a clear message
for missing required fields, non-numeric values and unknown categories.
`, job.Kind, job.ID)
}
