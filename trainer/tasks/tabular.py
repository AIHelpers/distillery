#!/usr/bin/env python3
"""Distillery tabular trainer (plan 08).

Gradient-boosted classification / regression with:
  * split strategies: random, stratified, time (chronological), group;
  * an untouched 20 % holdout, K-fold CV on the rest (expanding-window for the
    time strategy, group-disjoint for the group strategy);
  * random hyper-parameter search bounded by `time_budget_sec` across the
    available boosters (LightGBM / XGBoost / CatBoost; a dependency-free numpy
    booster is used when none is installed);
  * majority / mean baselines on the same holdout, Platt calibration for
    binary classifiers, SHAP feature importance, leakage warnings, ROC /
    confusion-matrix / residual artifacts;
  * a final refit on all rows.

Artifacts (no pickle, ever): model.<txt|ubj|cbm|json>, schema.json,
metrics.json (status "completed" + a "table" block in the Go
domain.TrainingMetrics shape), importance.csv.
"""

from __future__ import annotations

import csv
import json
import math
import os
import time
import traceback
from typing import Any

import numpy as np

from trainer import tablelib as T
from trainer.tasks import register

HOLDOUT_FRAC = 0.2
MAX_TRIALS = 40
DEFAULT_BUDGET = 60
HIGHER = {"roc_auc", "pr_auc", "f1", "accuracy", "r2"}
CLS_METRICS = ("roc_auc", "pr_auc", "f1", "accuracy", "logloss")
REG_METRICS = ("rmse", "mae", "r2")


class TrainError(Exception):
    """A user-facing training failure (bad data / config)."""


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def roc_auc_binary(y: np.ndarray, s: np.ndarray) -> float:
    pos = y == 1
    n1, n0 = int(pos.sum()), int((~pos).sum())
    if n1 == 0 or n0 == 0:
        return 0.5
    order = np.argsort(s, kind="mergesort")
    ss = s[order]
    r = np.empty(len(s))
    i = 0
    while i < len(ss):  # average ranks over ties.
        j = i
        while j + 1 < len(ss) and ss[j + 1] == ss[i]:
            j += 1
        r[i:j + 1] = (i + j) / 2 + 1
        i = j + 1
    ranks = np.empty(len(s))
    ranks[order] = r
    return float((ranks[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def average_precision(y: np.ndarray, s: np.ndarray) -> float:
    n1 = int((y == 1).sum())
    if n1 == 0:
        return 0.0
    order = np.argsort(-s, kind="mergesort")
    yy = y[order]
    tp = np.cumsum(yy == 1)
    prec = tp / np.arange(1, len(yy) + 1)
    return float(prec[yy == 1].sum() / n1)


def roc_points(y: np.ndarray, s: np.ndarray, max_points: int = 60) -> list[list[float]]:
    n1, n0 = int((y == 1).sum()), int((y == 0).sum())
    if n1 == 0 or n0 == 0:
        return [[0.0, 0.0], [1.0, 1.0]]
    order = np.argsort(-s, kind="mergesort")
    yy, ss = y[order], s[order]
    tp = np.cumsum(yy == 1)
    fp = np.cumsum(yy == 0)
    last = np.r_[ss[1:] != ss[:-1], True]  # one point per distinct threshold.
    tpr, fpr = tp[last] / n1, fp[last] / n0
    pts = np.column_stack([np.r_[0.0, fpr], np.r_[0.0, tpr]])
    if len(pts) > max_points:
        keep = np.unique(np.linspace(0, len(pts) - 1, max_points).astype(int))
        pts = pts[keep]
    return [[round(float(a), 4), round(float(b), 4)] for a, b in pts]


def _f1(tp: float, fp: float, fn: float) -> float:
    d = 2 * tp + fp + fn
    return 2 * tp / d if d > 0 else 0.0


def cls_scores(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    """p: (n, k) probabilities. Binary metrics use class 1 as positive;
    multiclass ROC/PR/F1 are macro one-vs-rest."""
    k = p.shape[1]
    pred = p.argmax(1)
    acc = float((pred == y).mean())
    ll = float(-np.mean(np.log(np.clip(p[np.arange(len(y)), y], 1e-15, 1))))
    if k == 2:
        auc = roc_auc_binary(y, p[:, 1])
        ap = average_precision(y, p[:, 1])
        f1 = _f1(float(((pred == 1) & (y == 1)).sum()), float(((pred == 1) & (y == 0)).sum()),
                 float(((pred == 0) & (y == 1)).sum()))
    else:
        aucs, aps, f1s = [], [], []
        for c in range(k):
            yc = (y == c).astype(int)
            if yc.sum() == 0:
                continue
            aucs.append(roc_auc_binary(yc, p[:, c]))
            aps.append(average_precision(yc, p[:, c]))
            f1s.append(_f1(float(((pred == c) & (y == c)).sum()), float(((pred == c) & (y != c)).sum()),
                           float(((pred != c) & (y == c)).sum())))
        auc, ap, f1 = float(np.mean(aucs)), float(np.mean(aps)), float(np.mean(f1s))
    return {"roc_auc": auc, "pr_auc": ap, "f1": f1, "accuracy": acc, "logloss": ll}


def reg_scores(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    err = y - p
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return {"rmse": float(np.sqrt(np.mean(err ** 2))), "mae": float(np.mean(np.abs(err))),
            "r2": float(1 - (err ** 2).sum() / ss_tot) if ss_tot > 0 else 0.0}


def all_scores(task: str, y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    return reg_scores(y, pred) if task == "regression" else cls_scores(y, pred)


def brier_ece(y: np.ndarray, p1: np.ndarray, bins: int = 10) -> tuple[float, float]:
    brier = float(np.mean((p1 - y) ** 2))
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for i in range(bins):
        m = (p1 >= edges[i]) & ((p1 < edges[i + 1]) if i < bins - 1 else (p1 <= 1))
        if m.any():
            ece += m.mean() * abs(p1[m].mean() - y[m].mean())
    return brier, float(ece)


# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------

def stratified_fold_ids(y: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """Fold id (0..k-1) per row such that every class is spread evenly over
    the folds (each class contributes floor/ceil(n_c / k) rows to each)."""
    fold = np.empty(len(y), dtype=int)
    nxt = 0
    for c in np.unique(y):
        idx = rng.permutation(np.where(y == c)[0])
        fold[idx] = (nxt + np.arange(len(idx))) % k
        nxt = (nxt + len(idx)) % k
    return fold


class Splitter:
    def __init__(self, strategy: str, y: np.ndarray, time_key, groups, classification: bool) -> None:
        self.strategy, self.y, self.time_key, self.groups = strategy, y, time_key, groups
        self.classification = classification

    def holdout(self) -> tuple[np.ndarray, np.ndarray]:
        n = len(self.y)
        rng = np.random.default_rng(42)
        cut = int(n * (1 - HOLDOUT_FRAC))
        if self.strategy == "time":
            order = np.argsort(self.time_key, kind="mergesort")
            return order[:cut], order[cut:]
        if self.strategy == "group":
            uniq = np.unique(self.groups)
            perm = rng.permutation(len(uniq))
            hold_groups = set(uniq[perm[: max(1, int(round(len(uniq) * HOLDOUT_FRAC)))]])
            mask = np.array([g in hold_groups for g in self.groups])
            if mask.any() and (~mask).any():
                return np.where(~mask)[0], np.where(mask)[0]
        if self.strategy == "stratified":
            hold_parts = []
            for c in np.unique(self.y):
                members = rng.permutation(np.where(self.y == c)[0])
                if len(members) >= 2:  # every class with 2+ rows is represented in the holdout.
                    hold_parts.append(members[: max(1, int(round(len(members) * HOLDOUT_FRAC)))])
            hold = np.sort(np.concatenate(hold_parts)) if hold_parts else np.array([], dtype=int)
            if 0 < len(hold) < n:
                return rng.permutation(np.setdiff1d(np.arange(n), hold)), hold
        order = rng.permutation(n)
        return order[:cut], order[cut:]

    def folds(self, idx: np.ndarray, k: int) -> list[tuple[np.ndarray, np.ndarray]]:
        """CV folds over `idx` (indices into the full arrays)."""
        rng = np.random.default_rng(7)
        n = len(idx)
        if self.strategy == "time":
            order = idx[np.argsort(self.time_key[idx], kind="mergesort")]
            chunks = np.array_split(order, k + 1)
            return [(np.concatenate(chunks[:i + 1]), chunks[i + 1]) for i in range(k) if len(chunks[i + 1])]
        if self.strategy == "group":
            g = self.groups[idx]
            uniq = np.unique(g)
            k = min(k, len(uniq))
            if k >= 2:
                assign = {u: i % k for i, u in enumerate(rng.permutation(uniq))}
                fold_of = np.array([assign[x] for x in g])
                return [(idx[fold_of != f], idx[fold_of == f]) for f in range(k)]
        if self.strategy == "stratified" and self.classification:
            fold_of = stratified_fold_ids(self.y[idx], k, rng)
            order = idx
        else:
            order = idx[rng.permutation(n)]
            fold_of = np.arange(n) % k
        return [(order[fold_of != f], order[fold_of == f]) for f in range(k)]


# ---------------------------------------------------------------------------
# Candidates
# ---------------------------------------------------------------------------

def _loss(metric: str, s: float) -> float:
    return -s if metric in HIGHER else s


class Candidate:
    def __init__(self, lib: str, params: dict[str, Any]) -> None:
        self.lib, self.params = lib, params
        self.score = math.nan
        self.std = math.nan
        self.rounds = 0
        self.oof: np.ndarray | None = None
        self.oof_idx: np.ndarray | None = None


def predict_scores(task: str, raw: np.ndarray) -> np.ndarray:
    """raw margins -> probabilities / values (uncalibrated)."""
    if task == "regression":
        return raw.astype(float)
    if raw.ndim == 1:
        p1 = T.sigmoid(raw)
        return np.column_stack([1 - p1, p1])
    return T.softmax(raw)


def evaluate(cand: Candidate, X, y, folds, task: str, k_classes: int, cat_idx, metric: str) -> None:
    scores: list[float] = []
    rounds: list[int] = []
    oof_idx, oof_raw = [], []
    for tr, va in folds:
        # Inner early-stopping split: the last 15 % of the training fold (folds
        # are chronological for the time strategy).
        cut = max(1, int(len(tr) * 0.85))
        itr, iva = tr[:cut], tr[cut:]
        if task != "regression" and len(np.unique(y[itr])) < k_classes:
            itr, iva = tr, tr[:0]  # a rare class is missing: skip early stopping.
        b = T.BOOSTERS[cand.lib](task, k_classes, cand.params, cat_idx)
        b.fit(X[itr], y[itr], X[iva] if len(iva) else None, y[iva] if len(iva) else None,
              rounds=400, early_stopping=30 if len(iva) else 0)
        raw = b.raw(X[va])
        pred = predict_scores(task, raw)
        scores.append(all_scores(task, y[va], pred)[metric])
        rounds.append(b.best_rounds or 100)
        oof_idx.append(va)
        oof_raw.append(raw)
    cand.score, cand.std = float(np.mean(scores)), float(np.std(scores))
    cand.rounds = int(np.median(rounds))
    cand.oof_idx, cand.oof = np.concatenate(oof_idx), np.concatenate(oof_raw)


def cv_baseline(task: str, y, folds, metric: str, k: int) -> float:
    """The constant baseline scored with the same CV folds as the candidates."""
    out = []
    for tr, va in folds:
        if task == "regression":
            pred = np.full(len(va), y[tr].mean())
        else:
            pred = np.tile(np.bincount(y[tr], minlength=k) / len(tr), (len(va), 1))
        out.append(all_scores(task, y[va], pred)[metric])
    return float(np.mean(out))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def class_label(v: Any) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, float) and v == int(v):
        return str(int(v))
    return str(v).strip()


def infer_task(values: list[Any]) -> str:
    uniq = {class_label(v) for v in values}
    nums = [T.to_float(v) for v in values]
    if all(not math.isnan(x) for x in nums) and len(uniq) > 10:
        return "regression"
    return "classification"


def save_json(path: str, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def fail(job_dir: str, writer, msg: str) -> int:
    writer.event("error", message=msg)
    save_json(os.path.join(job_dir, "metrics.json"), {"status": "failed", "error": msg})
    return 1


def signal_warnings(imp: list[dict[str, Any]], primary: float, metric: str) -> list[str]:
    out: list[str] = []
    total = sum(i["impact"] for i in imp)
    if total > 0 and len(imp) >= 3:
        share = imp[0]["impact"] / total
        if share >= 0.7:
            out.append(f"'{imp[0]['feature']}' carries {share:.0%} of the model's signal; check that it is not derived "
                       "from the target or unavailable at prediction time")
    if metric in ("roc_auc", "pr_auc", "accuracy", "r2") and primary >= 0.995:
        out.append("near-perfect holdout score: this is very often target leakage or duplicated rows")
    return out


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_tabular(cfg: dict, writer, job_dir: str, dataset_path: str) -> int:
    t_start = time.time()
    try:
        return _run(cfg, writer, job_dir, dataset_path, t_start)
    except TrainError as e:
        return fail(job_dir, writer, str(e))
    except Exception as e:  # noqa: BLE001
        writer.event("error", message=str(e), class_name=type(e).__name__, trace=traceback.format_exc()[-1500:])
        return fail(job_dir, writer, f"tabular training failed: {e}")


def _run(cfg: dict, writer, job_dir: str, dataset_path: str, t_start: float) -> int:
    tc = cfg.get("tabular") or {}
    target = (tc.get("target") or "").strip()
    if not target:
        raise TrainError("the tabular config has no target column")
    types: dict[str, str] = dict(tc.get("columns") or {})
    excluded = set(tc.get("exclude_columns") or [])
    all_rows = T.load_jsonl(dataset_path)
    if not all_rows:
        raise TrainError("the dataset has no rows")
    if not types:  # no mapping from the server: numbers are numeric, the rest categorical.
        for r in all_rows[:500]:
            for k, v in r.items():
                if k not in types and not T.is_missing(v):
                    types[k] = "numeric" if isinstance(v, (int, float)) and not isinstance(v, bool) else "categorical"
    if not any(target in r for r in all_rows):
        raise TrainError(f"target column {target!r} is not in the data")

    rows = [r for r in all_rows if not T.is_missing(r.get(target))]
    dropped = len(all_rows) - len(rows)
    if dropped:
        writer.event("warning", message=f"{dropped} rows without a target value were ignored")
    if len(rows) < 20:
        raise TrainError(f"only {len(rows)} rows have a target value; at least 20 are needed")

    task = tc.get("task") or infer_task([r[target] for r in rows])
    if task not in ("classification", "regression"):
        raise TrainError(f"unknown task {task!r}")
    metric = tc.get("metric") or ("roc_auc" if task == "classification" else "rmse")
    valid = CLS_METRICS if task == "classification" else REG_METRICS
    if metric not in valid:
        raise TrainError(f"metric {metric!r} is not valid for {task} (use {', '.join(valid)})")

    classes: list[str] = []
    if task == "regression":
        y = np.array([T.to_float(r[target]) for r in rows])
        bad = np.isnan(y)
        if bad.any():
            first = next(r[target] for r, b in zip(rows, bad) if b)
            raise TrainError(f"the regression target has {int(bad.sum())} non-numeric values (first: {first!r})")
        k_classes = 1
    else:
        labels = [class_label(r[target]) for r in rows]

        def order_key(c: str):
            f = T.to_float(c)
            return (0, f, c) if not math.isnan(f) else (1, 0.0, c)

        classes = sorted(set(labels), key=order_key)
        if len(classes) < 2:
            raise TrainError("the target has a single class; nothing to learn")
        if len(classes) > 50:
            raise TrainError(f"the target has {len(classes)} classes; choose a regression task or a different column")
        cmap = {c: i for i, c in enumerate(classes)}
        y = np.array([cmap[c] for c in labels], dtype=int)
        k_classes = len(classes)
        counts = np.bincount(y)
        if counts.min() < 2:
            raise TrainError(f"class {classes[int(counts.argmin())]!r} has a single example; remove it or merge rare classes")

    names = [c for c, t in types.items() if t in ("numeric", "categorical", "datetime") and c != target and c not in excluded]
    if not names:
        raise TrainError("no usable feature columns (all columns are excluded, ignored, text or the target)")
    pre = T.Preprocessor.build(rows, types, names)
    X = pre.transform(rows)
    n = len(rows)

    requested = tc.get("split_strategy") or ""
    dt_cols = sorted(c for c, t in types.items() if t == "datetime" and c != target)
    strategy = requested or ("time" if dt_cols else ("stratified" if task == "classification" else "random"))
    time_key = groups = None
    notes: list[str] = []
    if strategy == "time":
        if not dt_cols:
            raise TrainError("the time split needs a datetime column")
        d = [T.parse_dt(r.get(dt_cols[0])) for r in rows]
        if any(x is None for x in d):
            raise TrainError(f"time split: column {dt_cols[0]!r} has missing or unparseable timestamps")
        time_key = np.array([x.timestamp() for x in d])
    elif strategy == "group":
        gc = tc.get("group_column") or ""
        if not gc:
            raise TrainError("the group split needs a group column")
        groups = np.array([str(r.get(gc)) for r in rows])
        if len(np.unique(groups)) < 5:
            raise TrainError(f"the group split needs at least 5 distinct groups in {gc!r}")
    elif strategy == "stratified" and task != "classification":
        strategy = "random"
    elif strategy not in ("random", "stratified"):
        raise TrainError(f"unknown split strategy {strategy!r}")
    if dt_cols and strategy != "time":
        notes.append(f"the data has a datetime column ({dt_cols[0]}) but the split is {strategy}: a random split on "
                     "temporal data can leak the future into training and inflate the metrics")

    splitter = Splitter(strategy, y, time_key, groups, task == "classification")
    dev, hold = splitter.holdout()
    if len(hold) < 4 or len(dev) < 12:
        raise TrainError("too few rows for a train/holdout split")
    if task == "classification" and len(np.unique(y[dev])) < k_classes:
        raise TrainError("the training split is missing a class; use a random or stratified split, or add data")

    k_folds = max(2, min(int(tc.get("cv_folds") or 5), 10, len(dev) // 6))
    folds = splitter.folds(dev, k_folds)
    if not folds:
        raise TrainError("could not build cross-validation folds")

    if task == "regression":
        base_pred, base_name = np.full(len(hold), y[dev].mean()), "mean_predictor"
    else:
        prior = np.bincount(y[dev], minlength=k_classes) / len(dev)
        base_pred, base_name = np.tile(prior, (len(hold), 1)), "majority_class"
    base_scores = all_scores(task, y[hold], base_pred)

    avail = T.available_boosters() + ["numpy"]
    want = (tc.get("model") or "lightgbm").lower()
    if want == "auto":
        libs = [b for b in avail if b != "numpy"] or ["numpy"]
    elif want in T.BOOSTERS:
        if want in avail:
            libs = [want]
        else:
            libs = [avail[0]]
            writer.event("warning", message=f"{want} is not installed; using {libs[0]} instead")
    else:
        raise TrainError(f"unknown model {want!r} (use lightgbm, xgboost, catboost or auto)")
    budget = float(tc.get("time_budget_sec") or DEFAULT_BUDGET)

    writer.event("status", value="training", task=task, rows=n, features=len(names), split=strategy, libraries=libs)
    total_steps = 100

    def report(frac: float, trials: int) -> None:
        writer.progress(step=int(min(0.95, frac) * total_steps), total_steps=total_steps, epoch=float(trials))

    rng = np.random.default_rng(1234)
    t_search = time.time()
    trials = 0
    queue = [Candidate(lib, T.default_params(lib)) for lib in libs]
    best_by_lib: dict[str, Candidate] = {}
    while queue or (time.time() - t_search < budget and trials < MAX_TRIALS):
        c = queue.pop(0) if queue else Candidate(libs[trials % len(libs)], T.sample_params(libs[trials % len(libs)], rng))
        try:
            evaluate(c, X, y, folds, task, k_classes, pre.cat_idx, metric)
        except T.BoosterUnavailable as e:
            raise TrainError(str(e))
        trials += 1
        cur = best_by_lib.get(c.lib)
        if cur is None or _loss(metric, c.score) < _loss(metric, cur.score):
            best_by_lib[c.lib] = c
        report(max(trials / MAX_TRIALS, (time.time() - t_search) / budget), trials)
        writer.event("trial", trial=trials, library=c.lib, score=round(c.score, 5), metric=metric)
    best = min(best_by_lib.values(), key=lambda c: _loss(metric, c.score))

    # Fit on the dev split (early stopping on its last 15 %), score the holdout.
    cut = max(1, int(len(dev) * 0.85))
    itr, iva = dev[:cut], dev[cut:]
    if task != "regression" and len(np.unique(y[itr])) < k_classes:
        itr, iva = dev, dev[:0]
    b_dev = T.BOOSTERS[best.lib](task, k_classes, best.params, pre.cat_idx)
    b_dev.fit(X[itr], y[itr], X[iva] if len(iva) else None, y[iva] if len(iva) else None,
              rounds=600, early_stopping=40 if len(iva) else 0)
    raw_h = b_dev.raw(X[hold])
    pred_h = predict_scores(task, raw_h)

    # Platt calibration for binary classifiers, fitted on out-of-fold dev scores only.
    calibration: dict[str, Any] = {"method": "none"}
    calib_info = None
    if task == "classification" and k_classes == 2 and best.oof is not None and len(best.oof) >= 30:
        a, b = T.fit_platt(best.oof, y[best.oof_idx])
        p_raw, p_cal = T.sigmoid(raw_h), T.sigmoid(a * raw_h + b)
        yb = y[hold].astype(float)
        br0, ece0 = brier_ece(yb, p_raw)
        br1, ece1 = brier_ece(yb, p_cal)
        calib_info = {"method": "platt", "brier_before": round(br0, 5), "brier_after": round(br1, 5),
                      "ece_before": round(ece0, 5), "ece_after": round(ece1, 5)}
        if br1 <= br0:
            calibration = {"method": "platt", "a": a, "b": b}
            pred_h = np.column_stack([1 - p_cal, p_cal])
        else:
            calib_info["method"] = "platt (not applied: worse on the holdout)"

    hold_scores = all_scores(task, y[hold], pred_h)
    primary, base_val = hold_scores[metric], base_scores[metric]
    improvement = primary - base_val if metric in HIGHER else base_val - primary

    # SHAP importance on (up to 500 of) the holdout rows.
    report(0.96, trials)
    contrib = b_dev.contrib(X[hold[:500]])
    if contrib.ndim == 3:  # multiclass (n, k, d+1)
        mag = np.abs(contrib[:, :, :-1]).mean(axis=(0, 1))
    else:
        mag = np.abs(contrib[:, :-1]).mean(axis=0)
    agg: dict[str, float] = {}
    gain_agg: dict[str, float] = {}
    for owner, v, g in zip(pre.owner, mag, b_dev.gain()):
        agg[owner] = agg.get(owner, 0.0) + float(v)
        gain_agg[owner] = gain_agg.get(owner, 0.0) + float(g)
    imp = [{"feature": f, "impact": round(v, 6)} for f, v in sorted(agg.items(), key=lambda kv: -kv[1])]
    with open(os.path.join(job_dir, "importance.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["feature", "mean_abs_shap", "gain"])
        for it in imp:
            w.writerow([it["feature"], it["impact"], round(gain_agg.get(it["feature"], 0.0), 6)])

    warnings = notes + signal_warnings(imp, primary, metric)
    if isinstance(tc.get("leakage_warnings"), list):
        warnings += [str(x) for x in tc["leakage_warnings"]]

    extra: dict[str, Any] = {}
    if task == "classification":
        cm = np.zeros((k_classes, k_classes), dtype=int)
        for a_, p_ in zip(y[hold], pred_h.argmax(1)):
            cm[a_, p_] += 1
        extra["confusion_matrix"] = cm.tolist()
        extra["label_map"] = {c: i for i, c in enumerate(classes)}
        if k_classes == 2:
            extra["roc_curve"] = roc_points(y[hold], pred_h[:, 1])
    else:
        pick = np.random.default_rng(3).permutation(len(hold))[:300]
        extra["residuals"] = [[round(float(pred_h[i]), 4), round(float(y[hold][i] - pred_h[i]), 4)] for i in pick]

    # Final refit on every row with a fixed number of rounds.
    final = T.BOOSTERS[best.lib](task, k_classes, best.params, pre.cat_idx)
    final_rounds = max(20, int(best.rounds * 1.15))
    final.fit(X, y, None, None, rounds=final_rounds, early_stopping=0)
    model_path = final.save(job_dir)
    save_json(os.path.join(job_dir, "schema.json"), {
        "kind": "tabular", "task": task, "target": target, "classes": classes, "metric": metric,
        "preprocessor": pre.to_json(), "calibration": calibration,
        "model": {"library": best.lib, "file": os.path.basename(model_path), "rounds": final_rounds, "params": best.params},
        "explanation": "shap",
    })

    board = [{"model": lib, "score": round(c.score, 5), "std": round(c.std, 5), "metric": metric, "chosen": c is best,
              "params": {k: (round(v, 5) if isinstance(v, float) else v) for k, v in c.params.items()}}
             for lib, c in best_by_lib.items()]
    board.sort(key=lambda r: _loss(metric, r["score"]))
    board.append({"model": base_name, "score": round(cv_baseline(task, y, folds, metric, k_classes), 5),
                  "metric": metric, "baseline": True})

    table: dict[str, Any] = {
        "train_examples": n, "epochs": 1, "final_loss": round(best.score, 5),
        "eval_accuracy": round(hold_scores.get("accuracy", 0.0), 5),
        "primary_metric": metric, "primary_value": round(primary, 5), "higher_is_better": metric in HIGHER,
        "baselines": {k: round(v, 5) for k, v in base_scores.items()},
        "holdout_metrics": {k: round(v, 5) for k, v in hold_scores.items()},
        "cv_score": round(best.score, 5), "cv_std": round(best.std, 5),
        "improvement_over_baseline": round(improvement, 5),
        "table_backend": best.lib, "split_used": strategy,
        "leaderboard": board, "feature_importance": imp, "leakage_warnings": warnings,
        "feature_schema": {"features": pre.go_schema(), "target": target, "task": task, "classes": classes},
        **extra,
    }
    if calib_info:
        table["calibration"] = calib_info
    save_json(os.path.join(job_dir, "metrics.json"), {
        "status": "completed", "kind": "tabular", "train_examples": n, "epoch": 1, "global_step": trials,
        "eval_loss": round(best.score, 5), "train_runtime": round(time.time() - t_start, 2), "table": table})
    writer.progress(step=total_steps, total_steps=total_steps, epoch=float(trials))
    writer.event("complete", primary_metric=metric, primary_value=round(primary, 5), baseline=round(base_val, 5))
    return 0


register("tabular", run_tabular)
