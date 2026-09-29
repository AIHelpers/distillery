"""Plan-08 tests: tabular trainer, forecast trainer, tablelib and the serve worker.

Run: PYTHONPATH=. python -m pytest trainer/tests/test_tabular.py -q
Boosters that are not installed are skipped; the numpy booster always runs.
"""

from __future__ import annotations

import datetime as dt
import io
import json
import os
import subprocess
import sys

import numpy as np
import pytest

from trainer import tablelib as T
from trainer.progress import ProgressWriter
from trainer.tasks import get as get_task
from trainer.tasks import tabular, forecast

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def writer():
    return ProgressWriter(stream=io.StringIO())


def write_rows(tmp_path, rows, name="data.jsonl"):
    p = tmp_path / name
    p.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return str(p)


def train(tmp_path, cfg, rows, fn=tabular.run_tabular):
    job = tmp_path / "job"
    job.mkdir(exist_ok=True)
    rc = fn(cfg, writer(), str(job), write_rows(tmp_path, rows))
    return rc, str(job), json.loads((job / "metrics.json").read_text())


def churn_rows(n=400, seed=1):
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n):
        ten, plan, age = int(rng.integers(1, 60)), str(rng.choice(["basic", "pro", "ent"])), int(rng.integers(18, 75))
        lg = -1 + 0.06 * (30 - ten) + (0.8 if plan == "basic" else 0)
        rows.append({"tenure": ten, "plan": plan, "age": age, "churn": "yes" if rng.random() < 1 / (1 + np.exp(-lg)) else "no"})
    return rows


CHURN_COLS = {"tenure": "numeric", "plan": "categorical", "age": "numeric", "churn": "categorical"}


def cfg_tab(**kw):
    base = {"target": "churn", "task": "classification", "metric": "roc_auc", "columns": CHURN_COLS, "time_budget_sec": 3}
    base.update(kw)
    return {"kind": "tabular", "tabular": base}


def test_registry():
    assert get_task("tabular") is not None and get_task("time_series") is not None


@pytest.mark.parametrize("lib", ["lightgbm", "xgboost", "catboost", "numpy"])
def test_boosters_roundtrip(lib, tmp_path):
    if lib != "numpy" and lib not in T.available_boosters():
        pytest.skip(f"{lib} not installed")
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 4))
    X[:, 3] = rng.integers(1, 4, 300)
    for task, y, k in (("classification", (X[:, 0] + X[:, 1] > 0).astype(int), 2), ("regression", X[:, 0] * 2, 1),
                       ("classification", np.digitize(X[:, 0] + X[:, 1], [-0.7, 0.7]), 3)):
        b = T.BOOSTERS[lib](task, k, T.default_params(lib), [3]).fit(X[:220], y[:220], X[220:], y[220:], rounds=80)
        b.save(str(tmp_path))
        b2 = T.BOOSTERS[lib].load(os.path.join(str(tmp_path), "model." + b.ext), task, k, [3])
        assert np.allclose(b.raw(X[220:]), b2.raw(X[220:]), atol=1e-5)
        c = b.contrib(X[:5])
        raw = b.raw(X[:5])
        # SHAP values (+ bias) sum to the raw margin.
        if c.ndim == 2:
            assert np.allclose(c.sum(axis=1), raw, atol=1e-3)
        else:
            assert np.allclose(c.sum(axis=2), raw, atol=1e-3)


def test_binary_classification_beats_baseline(tmp_path):
    rc, job, m = train(tmp_path, cfg_tab(model="auto"), churn_rows() + [{"tenure": 5, "plan": "pro", "age": 30, "churn": None}])
    assert rc == 0 and m["status"] == "completed"
    t = m["table"]
    assert t["primary_metric"] == "roc_auc" and t["primary_value"] > 0.6
    assert t["baselines"]["roc_auc"] == pytest.approx(0.5)
    assert t["improvement_over_baseline"] > 0.1
    assert [c for c in t["leaderboard"] if c.get("baseline")] and sum(1 for c in t["leaderboard"] if c.get("chosen")) == 1
    assert t["feature_importance"][0]["feature"] == "tenure"
    assert t["roc_curve"][0] == [0.0, 0.0] and t["roc_curve"][-1] == [1.0, 1.0]
    assert t["feature_schema"]["classes"] == ["no", "yes"]
    assert {f["name"] for f in t["feature_schema"]["features"]} == {"tenure", "plan", "age"}
    for f in ("schema.json", "importance.csv"):
        assert os.path.exists(os.path.join(job, f))
    assert not [f for f in os.listdir(job) if f.endswith((".pkl", ".pickle", ".joblib"))]


def test_missing_target_rows_are_dropped(tmp_path):
    rows = churn_rows(200) + [{"tenure": 1, "plan": "pro", "age": 30, "churn": None}] * 5
    rc, _, m = train(tmp_path, cfg_tab(), rows)
    assert rc == 0 and m["table"]["train_examples"] == 200


def test_config_and_data_errors(tmp_path):
    rc, _, m = train(tmp_path, {"tabular": {"task": "classification"}}, churn_rows(50))
    assert rc == 1 and m["status"] == "failed" and "target" in m["error"]
    rc, _, m = train(tmp_path, cfg_tab(target="nope"), churn_rows(50))
    assert rc == 1 and "nope" in m["error"]
    rc, _, m = train(tmp_path, cfg_tab(metric="rmse"), churn_rows(50))
    assert rc == 1 and "metric" in m["error"]
    rc, _, m = train(tmp_path, cfg_tab(), [{"tenure": 1, "plan": "a", "age": 2, "churn": "yes"}] * 30)
    assert rc == 1 and "single class" in m["error"]


def test_multiclass_and_serving(tmp_path):
    rng = np.random.default_rng(2)
    rows = [{"a": float(x), "b": float(y), "c": str(rng.choice(["u", "v"])), "t": ["lo", "mid", "hi"][int(x + y > 0) + int(x + y > 1)]}
            for x, y in rng.normal(size=(300, 2))]
    cols = {"a": "numeric", "b": "numeric", "c": "categorical", "t": "categorical"}
    rc, job, m = train(tmp_path, {"tabular": {"target": "t", "task": "classification", "metric": "f1", "columns": cols, "time_budget_sec": 3}}, rows)
    assert rc == 0
    t = m["table"]
    assert t["primary_value"] > 0.7 and len(t["confusion_matrix"]) == 3
    model = T.load(job)
    out = model.predict({"a": 2.0, "b": 2.0, "c": "u"})
    assert out["prediction"] == "hi" and abs(sum(out["probabilities"].values()) - 1) < 1e-6
    with pytest.raises(ValueError, match="missing required"):
        model.predict({"a": 1.0})


def test_regression(tmp_path):
    rng = np.random.default_rng(3)
    rows = [{"a": float(x), "b": float(y), "y": float(3 * x + y + rng.normal() * 0.1)} for x, y in rng.normal(size=(300, 2))]
    cols = {"a": "numeric", "b": "numeric", "y": "numeric"}
    rc, job, m = train(tmp_path, {"tabular": {"target": "y", "task": "regression", "metric": "rmse", "columns": cols,
                                              "time_budget_sec": 3, "split_strategy": "random"}}, rows)
    t = m["table"]
    assert rc == 0 and t["higher_is_better"] is False
    assert t["primary_value"] < 0.5 * t["baselines"]["rmse"] and t["holdout_metrics"]["r2"] > 0.9
    assert len(t["residuals"]) > 10
    assert abs(T.load(job).predict({"a": 1.0, "b": 1.0})["prediction"] - 4.0) < 1.0


def test_time_split_and_leakage_note(tmp_path):
    rng = np.random.default_rng(4)
    rows = [{"ts": (dt.date(2024, 1, 1) + dt.timedelta(days=i)).isoformat(), "x": float(rng.normal()), "y": int(rng.random() < 0.5)}
            for i in range(300)]
    cols = {"ts": "datetime", "x": "numeric", "y": "numeric"}
    base = {"target": "y", "task": "classification", "columns": cols, "time_budget_sec": 2}
    rc, _, m = train(tmp_path, {"tabular": base}, rows)
    assert rc == 0 and m["table"]["split_used"] == "time"  # default for temporal data
    rc, _, m = train(tmp_path, {"tabular": {**base, "split_strategy": "random"}}, rows)
    assert any("datetime" in w for w in m["table"]["leakage_warnings"])


def test_group_split(tmp_path):
    rows = churn_rows(300)
    for i, r in enumerate(rows):
        r["cust"] = f"c{i % 30}"
    rc, _, m = train(tmp_path, cfg_tab(split_strategy="group", group_column="cust"), rows)
    assert rc == 0 and m["table"]["split_used"] == "group"
    rc, _, m = train(tmp_path, cfg_tab(split_strategy="group"), rows)
    assert rc == 1 and "group" in m["error"]


def test_perfect_leak_is_flagged(tmp_path):
    rows = churn_rows(300)
    for r in rows:
        r["leak"] = 1.0 if r["churn"] == "yes" else 0.0
    cols = {**CHURN_COLS, "leak": "numeric"}
    rc, _, m = train(tmp_path, cfg_tab(columns=cols), rows)
    assert rc == 0 and m["table"]["leakage_warnings"]


# --------------------------- forecasting ---------------------------------

def series_rows(n=150, freq="D", items=("a", "b"), season=7, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        d = T.step_time(dt.datetime(2020, 1, 1), freq, i)
        for it in items:
            rows.append({"ds": T.iso(d), "y": 50 + 10 * np.sin(2 * np.pi * i / season) + rng.normal(), "sku": it})
    return rows


def fc_cfg(**kw):
    base = {"timestamp": "ds", "target": "y", "item_id": "sku", "frequency": "D", "horizon": 14, "backtest_windows": 3, "model": "auto"}
    base.update(kw)
    return {"kind": "time_series", "forecast": base}


def test_forecast_daily_beats_or_matches_baseline(tmp_path):
    rc, job, m = train(tmp_path, fc_cfg(), series_rows(), forecast.run_forecast)
    t = m["table"]
    assert rc == 0 and t["season_length"] == 7
    assert t["primary_value"] <= t["baselines"]["seasonal_naive_mase"] + 1e-9
    assert any(c["baseline"] for c in t["leaderboard"]) and t["backtest_plot"]["lower"]
    assert len({w["window"] for w in t["backtest"]}) == 3
    model = T.load(job)
    out = model.forecast(None, 5, "a")
    assert len(out["forecast"]) == 5 and all(p["lower"] <= p["value"] <= p["upper"] for p in out["forecast"])
    with pytest.raises(ValueError, match="item_id"):
        model.forecast(None, 5)
    # Caller history extends the series.
    hist = [{"ds": "2020-06-01", "y": 999.0, "sku": "a"}]
    assert model.forecast(hist, 2, "a")["forecast"][0]["timestamp"] > out["forecast"][0]["timestamp"]


def test_forecast_monthly_uses_season_12(tmp_path):
    rows = [{"ds": f"{2015 + i // 12}-{i % 12 + 1:02d}-01", "y": 100 + i + 20 * np.sin(2 * np.pi * i / 12)} for i in range(72)]
    cfg = fc_cfg(frequency="M", horizon=6, item_id="", backtest_windows=2)
    rc, _, m = train(tmp_path, cfg, rows, forecast.run_forecast)
    assert rc == 0 and m["table"]["season_length"] == 12
    assert m["table"]["primary_value"] < 0.6


def test_forecast_short_series_is_a_clear_error(tmp_path):
    rows = [{"ds": f"2024-01-{i + 1:02d}", "y": float(i), "sku": "a"} for i in range(10)]
    rc, _, m = train(tmp_path, fc_cfg(), rows, forecast.run_forecast)
    assert rc == 1 and "long enough" in m["error"]


def test_forecast_config_errors(tmp_path):
    rc, _, m = train(tmp_path, fc_cfg(frequency="X"), series_rows(60), forecast.run_forecast)
    assert rc == 1 and "frequency" in m["error"]
    rc, _, m = train(tmp_path, fc_cfg(horizon=0), series_rows(60), forecast.run_forecast)
    assert rc == 1 and "horizon" in m["error"]


def test_interpolates_gaps():
    rows = [{"ds": f"2024-01-{d:02d}", "y": float(d)} for d in (1, 2, 3, 6, 7)]
    s, filled = T.build_series(rows, "ds", "y", "", "D")
    assert filled == 2 and list(s[0].y) == [1, 2, 3, 4, 5, 6, 7]


# --------------------------- serve worker --------------------------------

def _talk(job, reqs):
    env = {**os.environ, "PYTHONPATH": REPO}
    p = subprocess.Popen([sys.executable, "-m", "trainer.serve_table", "--job-dir", job], stdin=subprocess.PIPE,
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
    out = [json.loads(p.stdout.readline())]
    for r in reqs:
        p.stdin.write(json.dumps(r) + "\n")
        p.stdin.flush()
        out.append(json.loads(p.stdout.readline()))
    p.stdin.close()
    p.wait(timeout=15)
    return out


def test_serve_worker_roundtrip(tmp_path):
    rc, job, _ = train(tmp_path, cfg_tab(), churn_rows(200))
    assert rc == 0
    hello, ok, bad, unk, ping = _talk(job, [
        {"id": 1, "op": "predict", "row": {"tenure": 3, "plan": "basic", "age": 40}},
        {"id": 2, "op": "predict", "row": {"plan": "basic"}},
        {"id": 3, "op": "explode"}, {"id": 4, "op": "ping"}])
    assert hello == {"ready": True, "kind": "tabular"}
    assert ok["ok"] and ok["result"]["prediction"] in ("yes", "no") and ok["result"]["top_factors"]
    assert not bad["ok"] and "missing required" in bad["error"]
    assert not unk["ok"] and ping["ok"]


def test_serve_worker_forecast_and_bad_dir(tmp_path):
    rc, job, _ = train(tmp_path, fc_cfg(), series_rows(120), forecast.run_forecast)
    assert rc == 0
    hello, ok = _talk(job, [{"id": 1, "op": "forecast", "horizon": 3, "item_id": "b"}])
    assert hello["kind"] == "time_series" and len(ok["result"]["forecast"]) == 3
    empty = tmp_path / "empty"
    empty.mkdir()
    assert _talk(str(empty), [])[0]["ready"] is False
