#!/usr/bin/env python3
"""Distillery time-series forecasting trainer (plan 08), kind `time_series`.

Builds regular per-item series on the confirmed frequency grid, runs a
rolling-origin backtest over several windows for a set of candidate models
(seasonal naive baseline, naive, drift, SES, Holt-Winters, a lag-feature GBM,
and - when installed - statsforecast AutoETS / Chronos), picks the best by
MASE / sMAPE / weighted quantile loss, and stores a small JSON bundle:

    forecast_model.json  config + chosen model + interval scale + history
    schema.json          {"kind": "time_series", ...}
    metrics.json         status "completed" + a "table" block (Go metrics shape)

Nothing is pickled; the chosen model is refit from the stored history at
serving time (milliseconds for the statistical models).
"""

from __future__ import annotations

import json
import os
import time
import traceback
from typing import Any

from trainer import tablelib as T
from trainer.tasks import register

MAX_STORED_POINTS = 5000  # per series, most recent.
MAX_STORED_SERIES = 500


def _save(path: str, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)


def _fail(job_dir: str, writer, msg: str) -> int:
    writer.event("error", message=msg)
    _save(os.path.join(job_dir, "metrics.json"), {"status": "failed", "error": msg})
    return 1


def run_forecast(cfg: dict, writer, job_dir: str, dataset_path: str) -> int:
    t0 = time.time()
    try:
        fc = dict(cfg.get("forecast") or {})
        ts, target = (fc.get("timestamp") or "").strip(), (fc.get("target") or "").strip()
        freq = (fc.get("frequency") or "").strip().upper()
        if not ts or not target:
            return _fail(job_dir, writer, "the forecast config needs timestamp and target columns")
        if freq not in T.FREQ_SEASON:
            return _fail(job_dir, writer, f"unknown frequency {freq!r} (use H, D, W, M, Q or Y)")
        horizon = int(fc.get("horizon") or 0)
        if horizon < 1:
            return _fail(job_dir, writer, "the forecast horizon must be at least 1")
        fc["frequency"] = freq
        item = (fc.get("item_id") or "").strip()

        rows = T.load_jsonl(dataset_path)
        if not rows:
            return _fail(job_dir, writer, "the dataset has no rows")
        series, filled = T.build_series(rows, ts, target, item, freq)
        writer.event("status", value="series_built", series=len(series), rows=len(rows), interpolated=filled)
        if filled:
            writer.event("warning", message=f"{filled} missing periods were filled by linear interpolation")

        total = 100

        def progress(done: int, of: int) -> None:
            writer.progress(step=int(10 + 85 * done / max(of, 1)), total_steps=total, epoch=float(done))

        def log(msg: str) -> None:
            writer.event("warning", message=msg)

        writer.progress(step=5, total_steps=total, epoch=0.0)
        res = T.backtest(series, fc, progress=progress, log=log)
        for name, err in res["failed_models"].items():
            writer.event("warning", message=f"model {name} failed during the backtest and was dropped: {err}")

        # Bundle: config + chosen model + the most recent history per series.
        keep = series[:MAX_STORED_SERIES]
        if len(series) > MAX_STORED_SERIES:
            writer.event("warning", message=f"only the first {MAX_STORED_SERIES} of {len(series)} series are stored for serving")
        bundle = {
            "config": {"timestamp": ts, "target": target, "item_id": item, "frequency": freq,
                       "season_length": res["season_length"], "horizon": horizon},
            "chosen_model": res["chosen_model"], "interval_scale": res["interval_scale"],
            "primary_metric": res["primary_metric"], "primary_value": res["primary_value"],
            "baselines": res["baselines"],
            "series": [{"id": s.id, "t": [T.iso(t) for t in s.times[-MAX_STORED_POINTS:]],
                        "y": [float(v) for v in s.y[-MAX_STORED_POINTS:]]} for s in keep],
        }
        _save(os.path.join(job_dir, "forecast_model.json"), bundle)
        _save(os.path.join(job_dir, "schema.json"), {
            "kind": "time_series", "timestamp": ts, "target": target, "item_id": item, "frequency": freq,
            "season_length": res["season_length"], "horizon": horizon, "model": res["chosen_model"],
            "series_ids": [s.id for s in keep]})

        skip = ("interval_scale", "series_used", "series_total", "failed_models")
        table = {k: v for k, v in res.items() if k not in skip}
        table.update(train_examples=len(rows), epochs=1, table_backend="python-" + res["chosen_model"])
        if res["series_used"] < res["series_total"]:
            table["leakage_warnings"] = [
                f"{res['series_total'] - res['series_used']} of {res['series_total']} series were too short for the "
                "backtest and are not represented in the scores"]
        _save(os.path.join(job_dir, "metrics.json"), {
            "status": "completed", "kind": "time_series", "train_examples": len(rows), "epoch": 1,
            "eval_loss": res["final_loss"], "train_runtime": round(time.time() - t0, 2), "table": table})
        writer.progress(step=total, total_steps=total, epoch=1.0)
        writer.event("complete", chosen=res["chosen_model"], metric=res["primary_metric"], value=res["primary_value"])
        return 0
    except ValueError as e:  # bad data / config
        return _fail(job_dir, writer, str(e))
    except Exception as e:  # noqa: BLE001
        writer.event("error", message=str(e), class_name=type(e).__name__, trace=traceback.format_exc()[-1500:])
        return _fail(job_dir, writer, f"forecast training failed: {e}")


register("time_series", run_forecast)
