#!/usr/bin/env python3
"""Distillery table-model serving worker (plan 08).

Loads a tabular or forecasting model from a job directory and answers
JSON-lines requests on stdin, one JSON response per line on stdout:

    -> {"id": 1, "op": "predict", "row": {...}}
    -> {"id": 2, "op": "forecast", "history": [...], "horizon": 14, "item_id": ""}
    -> {"id": 3, "op": "ping"}
    <- {"id": 1, "ok": true, "result": {...}}   |   {"id": 1, "ok": false, "error": "..."}

The first line printed is the readiness handshake: {"ready": true, "kind": ...}
or {"ready": false, "error": "..."}. Only native model files / JSON are
loaded - never pickle. Stdout is reserved for the protocol; anything a library
prints is redirected to stderr.
"""

from __future__ import annotations

import argparse
import json
import sys


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--job-dir", required=True)
    args = ap.parse_args()

    proto = sys.stdout
    sys.stdout = sys.stderr  # library chatter must not corrupt the protocol.

    def send(obj) -> None:
        proto.write(json.dumps(obj, ensure_ascii=False, allow_nan=False) + "\n")
        proto.flush()

    try:
        from trainer import tablelib as T

        model = T.load(args.job_dir)
    except Exception as e:  # noqa: BLE001
        send({"ready": False, "error": str(e)})
        return 1
    kind = "time_series" if isinstance(model, T.ForecastModel) else "tabular"
    send({"ready": True, "kind": kind})

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        rid = None
        try:
            req = json.loads(line)
            rid = req.get("id")
            op = req.get("op")
            if op == "ping":
                res = {"pong": True}
            elif op == "predict":
                if kind != "tabular":
                    raise ValueError("this model is a forecaster; use the forecast operation")
                res = model.predict(req.get("row") or {})
            elif op == "forecast":
                if kind != "time_series":
                    raise ValueError("this model is not a forecaster")
                res = model.forecast(req.get("history"), int(req.get("horizon") or 0), str(req.get("item_id") or ""))
            else:
                raise ValueError(f"unknown op {op!r}")
            send({"id": rid, "ok": True, "result": res})
        except Exception as e:  # noqa: BLE001 - a bad request must not kill the worker.
            try:
                send({"id": rid, "ok": False, "error": str(e)})
            except Exception:  # noqa: BLE001
                send({"id": rid, "ok": False, "error": "internal error"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
