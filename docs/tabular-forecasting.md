# Tabular ML & Time-Series Forecasting (Plan 08)

Two model kinds that are *trained*, not fine-tuned (no LoRA, no base model):

| Kind | Task | Models |
|------|------|--------|
| `tabular` | classification / regression on a CSV | LightGBM (default), XGBoost, CatBoost; a dependency-free numpy booster when none is installed |
| `time_series` | per-series forecasting with intervals | seasonal naive (baseline), naive, drift, SES, Holt-Winters, lag-feature GBM; AutoETS if `statsforecast` is installed; Chronos (experimental) |

## Backends

* `TRAINING_BACKEND=simulation` (default, no Python needed): **stand-ins**, not GBMs. Tabular = k-nearest-neighbours with permutation importance and occlusion-based per-prediction factors; forecasting = seasonal naive / naive / drift / SES with normal intervals. Metrics, baselines, ROC, confusion matrix and backtests are computed honestly on real holdouts, but the model family is different from production. The UI says so.
* `TRAINING_BACKEND=local`: the Python trainer (`trainer/tasks/tabular.py`, `forecast.py`, shared `trainer/tablelib.py`). Served by long-lived Python workers (`trainer/serve_table.py`, JSON lines over a pipe, one per deployed job, stopped after 10 idle minutes). Docker installs the `tabular` extra (LightGBM, XGBoost, CatBoost, pandas).

## Data

Upload a CSV (`POST /api/v1/tasks/{task}/tables`, multipart field `file`; optional `target`, `timestamp`, `item_id_column`, `frequency`, `horizon`). Comma, semicolon, tab and pipe delimiters, a UTF-8 BOM, and `NA/null/NaN/#N/A` missing tokens are handled; duplicate/blank headers and ragged rows are rejected. Limit: 50 MB per upload.

Types are inferred per column (`numeric`, `categorical`, `datetime`, `text`, `id`, `ignored`) and editable via `PUT .../tables/{id}/mapping`. Numbers use `.` as decimal separator; `12,5` is **not** numeric. Rows are stored as JSON lines in blob storage; only metadata lives in the repository. Preview: `GET .../tables/{id}/preview`. Delete: `DELETE .../tables/{id}`.

The mapper warns about ID-like columns, and about columns identical to, linearly proxying (|r| ≥ 0.98) or fully determining the target.

Forecasting tables need a timestamp column, a target, an optional series-id column, and a **confirmed frequency** (H, D, W, M, Q, Y). The frequency is inferred from the median gap between distinct timestamps; training is refused until it is confirmed (send `frequency` in the mapping).

## Training

`POST .../training/tabular` `{table_id, model, task?, metric?, time_budget_sec?, cv_folds?, split_strategy?, group_column?}` and `POST .../training/forecast` `{table_id, horizon, frequency?, metric?, backtest_windows?, model?, season_length?}` return a job; poll `GET /api/v1/training/{jobID}` (`running` → `completed` | `failed`).

Tabular pipeline (local backend):

1. Rows without a target are dropped; task inferred if unset (few integer values ⇒ classification). Binary and multiclass are supported.
2. Split: **time-based when a datetime column exists** (default), otherwise stratified (classification) or random; group split on request. 20 % is held out and never used for selection.
3. K-fold CV on the rest (expanding-window for time, group-disjoint for group), random hyper-parameter search across the requested libraries until `time_budget_sec` or 40 trials, early stopping inside each fold.
4. Holdout metrics for the chosen model next to the majority/mean baseline on the same rows; Platt calibration for binary classifiers (fitted on out-of-fold scores, applied only if it improves the holdout Brier score); SHAP importance; ROC curve, confusion matrix or residual sample; leakage warnings (dominant feature, near-perfect score, random split on temporal data).
5. Final refit on all rows. Artifacts: `model.*`, `schema.json`, `metrics.json`, `importance.csv`. **No pickle is ever written or read.**

Metrics: classification ROC-AUC (default), PR-AUC, F1, accuracy, log loss (multiclass = macro one-vs-rest); regression RMSE (default), MAE, R².

Forecasting: series are put on the frequency grid (period duplicates averaged, gaps linearly interpolated), a rolling-origin backtest runs over `backtest_windows` (default 3) windows, and models are ranked by MASE (default), sMAPE or weighted quantile loss. Season length defaults to 24/7/52/12/4/1 for H/D/W/M/Q/Y. Series shorter than `max(2·season, 8) + horizon·windows` points cannot be backtested (the job fails with the required length if none qualifies, and reports how many series were skipped). Intervals are 80 % bands from model residuals, rescaled so the backtest empirical coverage matches.

## Deployment and inference

Deployment is explicit (`POST .../deploy` or `.../training/{job}/deploy`). The response contains the raw API key **once**; send it as `Authorization: Bearer <key>`. The deployment records the feature schema (names, types, allowed categories, ranges, classes) and validates every request against it.

```
POST /api/v1/inference/{id}/predict   {"input": {"age": 41, "plan": "pro"}}
-> {"kind":"tabular","result":{"prediction":"churn","probability":0.73,"probabilities":{…},
    "top_factors":[{"feature":"tenure_months","impact":-0.21}],"explanation_method":"shap"}}

POST /api/v1/inference/{id}/forecast  {"horizon": 14, "item_id": "sku-1", "history": [ … optional … ]}
-> {"kind":"time_series","result":{"forecast":[…],"lower":[…],"upper":[…],"points":[…],"model":"…"}}

POST /api/v1/inference/{id}/predict-batch   (text/csv in, text/csv out)
```

Unknown categories, missing required fields and non-numeric values are 400s with a message. Categories match case-insensitively; a feature whose vocabulary was capped at 64 values accepts unseen values. Optional features (missing in some training rows) may be omitted. Batch tabular: per-row errors go into an `error` column instead of failing the batch; output cells starting with `= + - @` are prefixed to defuse spreadsheet formula injection. Batch time series: history rows in, forecast rows out (`?horizon=N`). `top_factors` are SHAP values in the model's output space (log-odds for classifiers) with the local backend, occlusion effects with the simulation backend (`explanation_method` says which).

Export (`GET .../export`) produces a zip with the native model, `schema.json`, `metrics.json`, `tablelib.py` (numpy-only scoring library) and a runnable `predict_example.py`.

## UI

`/tabular.html`: multipart upload, column mapper (types, target, per-column use, split, task, timestamp/series/frequency confirmation, leakage warnings), training form, live progress, results (metric vs baseline, leaderboard, holdout metrics, importance, confusion matrix, ROC, residuals, backtest chart and per-window table), explicit Deploy button showing the one-time key, a test form generated from the feature schema, and batch CSV. All server text is rendered via `textContent`.

## Scope and known limitations

* **Covariates are not supported** for forecasting (target history only). The earlier design mentioned them; they were removed rather than accepted and ignored.
* Chronos and `statsforecast` are optional, guarded imports and are **not exercised by the automated tests**; treat them as experimental.
* Time-based splitting uses the first (alphabetical) datetime column; the model sees calendar parts (month, weekday, hour, day) of datetime features, never absolute time.
* The category vocabulary and numeric ranges are computed on all uploaded rows (labels are never used for them).
* Row cap: 50 MB upload; rows are held in memory during training (cached, evicted with the job).
* No fairness checks or audit log yet: do not use for credit, hiring or insurance decisions without adding them.
* Multi-series forecast models store the most recent 5 000 points of up to 500 series.
* The API-key gate applies to inference endpoints only; management endpoints follow `ADMIN_TOKEN`.

## Tests

* Go: `go test -race ./...` (domain validation, CSV/type/frequency/leakage inference, simulation trainers, HTTP end-to-end upload → train → deploy → predict/forecast/batch, metrics-contract tests, and — when Python+numpy are available — the real trainer, serving worker and export package).
* Python: `PYTHONPATH=. python -m pytest trainer/tests/test_tabular.py` (boosters round-trip and SHAP additivity, binary/multiclass/regression, time/group splits, leakage flags, forecasts, short series, worker protocol). Boosters that are not installed are skipped.
