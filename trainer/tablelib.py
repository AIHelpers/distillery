"""Distillery table-model library (plan 08): shared by the trainers
(`trainer/tasks/tabular.py`, `forecast.py`), the serving worker
(`trainer/serve_table.py`) and the exported package.

Design rules:
  * numpy is the only hard dependency; LightGBM / XGBoost / CatBoost /
    statsforecast / Chronos are imported lazily and only when used;
  * every artifact is a native model file or JSON — never pickle;
  * the tabular half (this section) and the forecast half (below) are
    independent; both expose a `load(job_dir)` returning an object with
    `predict(row)` / `forecast(...)`.
"""

from __future__ import annotations

import json
import math
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable, Sequence

import numpy as np

MAX_CATEGORIES = 64
MISSING_TOKENS = {"", "na", "n/a", "nan", "null", "none", "#n/a"}


# ---------------------------------------------------------------------------
# Value helpers
# ---------------------------------------------------------------------------

def is_missing(v: Any) -> bool:
    if v is None:
        return True
    if isinstance(v, float) and math.isnan(v):
        return True
    return isinstance(v, str) and v.strip().lower() in MISSING_TOKENS


def to_float(v: Any) -> float:
    """Finite float or NaN (never raises)."""
    if is_missing(v):
        return float("nan")
    try:
        f = float(v)
    except (TypeError, ValueError):
        return float("nan")
    return f if math.isfinite(f) else float("nan")


_DT_FORMATS = (
    "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d",
    "%Y/%m/%d", "%d.%m.%Y %H:%M:%S", "%d.%m.%Y %H:%M", "%d.%m.%Y",
    "%m/%d/%Y %H:%M:%S", "%m/%d/%Y %H:%M", "%m/%d/%Y", "%Y-%m",
)


def parse_dt(v: Any) -> datetime | None:
    """Parse a timestamp to a naive-UTC datetime (None when unparseable)."""
    if is_missing(v):
        return None
    if isinstance(v, datetime):
        d = v
    else:
        s = str(v).strip()
        d = None
        for fmt in _DT_FORMATS:
            try:
                d = datetime.strptime(s, fmt)
                break
            except ValueError:
                continue
        if d is None:
            try:
                d = datetime.fromisoformat(s.replace("Z", "+00:00"))
            except ValueError:
                return None
    if d.tzinfo is not None:
        d = d.astimezone(timezone.utc).replace(tzinfo=None)
    return d


def iso(d: datetime) -> str:
    return d.strftime("%Y-%m-%dT%H:%M:%S")


def load_jsonl(path: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -35, 35)))


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

DT_PARTS = ("month", "dow", "hour", "dom")


def _dt_part(d: datetime | None, part: str) -> float:
    if d is None:
        return float("nan")
    if part == "month":
        return float(d.month)
    if part == "dow":
        return float(d.weekday())
    if part == "hour":
        return float(d.hour)
    return float(d.day)


class Preprocessor:
    """Row -> float matrix. Numeric: NaN when missing. Categorical: code 0 =
    missing, 1..K = known category (frequency order), K+1 = unseen. Datetime:
    calendar parts (absolute time is deliberately not a feature: trees cannot
    extrapolate it and it would defeat a time-based holdout)."""

    def __init__(self, features: list[dict[str, Any]]) -> None:
        self.features = features
        self.columns: list[str] = []
        self.owner: list[str] = []  # expanded column -> original feature.
        self.cat_idx: list[int] = []
        self._lookup: dict[str, dict[str, int]] = {}

        for f in features:
            name = f["name"]
            if f["type"] == "datetime":
                for p in DT_PARTS:
                    self.columns.append(f"{name}__{p}")
                    self.owner.append(name)
            else:
                if f["type"] == "categorical":
                    self.cat_idx.append(len(self.columns))
                    self._lookup[name] = {c.lower(): i + 1 for i, c in enumerate(f["categories"])}
                self.columns.append(name)
                self.owner.append(name)

    @staticmethod
    def build(rows: Sequence[dict[str, Any]], columns: dict[str, str], names: Sequence[str]) -> "Preprocessor":
        feats: list[dict[str, Any]] = []
        for name in names:
            ctype = columns.get(name, "categorical")
            vals = [r.get(name) for r in rows]
            present = [v for v in vals if not is_missing(v)]
            spec: dict[str, Any] = {"name": name, "optional": len(present) < len(vals)}
            if ctype == "numeric":
                nums = np.array([to_float(v) for v in present], dtype=float)
                nums = nums[np.isfinite(nums)]
                spec.update(type="numeric", median=float(np.median(nums)) if len(nums) else 0.0,
                            min=float(nums.min()) if len(nums) else None,
                            max=float(nums.max()) if len(nums) else None)
            elif ctype == "datetime":
                spec.update(type="datetime")
            else:
                counts: dict[str, int] = {}
                for v in present:
                    s = str(v).strip()
                    counts[s] = counts.get(s, 0) + 1
                cats = sorted(counts, key=lambda k: (-counts[k], k))
                spec.update(type="categorical", categories=cats[:MAX_CATEGORIES],
                            open_vocabulary=len(cats) > MAX_CATEGORIES)
            feats.append(spec)
        return Preprocessor(feats)

    def transform(self, rows: Sequence[dict[str, Any]]) -> np.ndarray:
        X = np.full((len(rows), len(self.columns)), np.nan, dtype=np.float64)
        for i, r in enumerate(rows):
            j = 0
            for f in self.features:
                v = r.get(f["name"])
                t = f["type"]
                if t == "numeric":
                    X[i, j] = to_float(v)
                    j += 1
                elif t == "datetime":
                    d = parse_dt(v)
                    for p in DT_PARTS:
                        X[i, j] = _dt_part(d, p)
                        j += 1
                else:
                    if is_missing(v):
                        X[i, j] = 0.0
                    else:
                        code = self._lookup[f["name"]].get(str(v).strip().lower())
                        X[i, j] = float(code) if code else float(len(f["categories"]) + 1)
                    j += 1
        return X

    def to_json(self) -> list[dict[str, Any]]:
        return self.features

    def go_schema(self) -> list[dict[str, Any]]:
        """The feature list in the Go domain.FeatureSchema JSON shape."""
        out = []
        for f in self.features:
            g: dict[str, Any] = {"name": f["name"], "type": f["type"], "optional": bool(f.get("optional"))}
            if f["type"] == "numeric":
                if f.get("min") is not None:
                    g["min"], g["max"] = f["min"], f["max"]
            elif f["type"] == "categorical":
                g["categories"] = f["categories"]
                g["open_vocabulary"] = bool(f.get("open_vocabulary"))
            out.append(g)
        return out

    def medians(self) -> np.ndarray:
        m = []
        for f in self.features:
            if f["type"] == "numeric":
                m.append(f["median"])
            elif f["type"] == "datetime":
                m.extend([6.0, 3.0, 12.0, 15.0])
            else:
                m.append(0.0)
        return np.array(m, dtype=float)


# ---------------------------------------------------------------------------
# Boosters
# ---------------------------------------------------------------------------

class BoosterUnavailable(RuntimeError):
    pass


class Booster:
    """Uniform wrapper. `raw` returns margins: regression (n,), binary (n,)
    logit, multiclass (n, k). `contrib` returns SHAP values with the bias in
    the last column: (n, d+1) or, for multiclass, (n, k, d+1)."""

    library = ""
    ext = ""

    def __init__(self, task: str, n_classes: int, params: dict[str, Any], cat_idx: Sequence[int], seed: int = 42) -> None:
        self.task, self.n_classes, self.params = task, n_classes, dict(params)
        self.cat_idx = list(cat_idx)
        self.seed = seed
        self.best_rounds = 0
        self.model: Any = None

    def fit(self, X, y, Xv=None, yv=None, rounds: int = 600, early_stopping: int = 40) -> "Booster":
        raise NotImplementedError

    def raw(self, X) -> np.ndarray:
        raise NotImplementedError

    def contrib(self, X) -> np.ndarray:
        raise NotImplementedError

    def gain(self) -> np.ndarray:
        raise NotImplementedError

    def save(self, job_dir: str) -> str:
        raise NotImplementedError

    @classmethod
    def load(cls, path: str, task: str, n_classes: int, cat_idx: Sequence[int]) -> "Booster":
        raise NotImplementedError


class LightGBMBooster(Booster):
    library, ext = "lightgbm", "txt"

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        try:
            import lightgbm  # noqa: F401
        except Exception as e:  # noqa: BLE001
            raise BoosterUnavailable(f"lightgbm is not installed ({e})")

    def _objective(self) -> dict[str, Any]:
        if self.task == "regression":
            return {"objective": "regression"}
        if self.n_classes <= 2:
            return {"objective": "binary"}
        return {"objective": "multiclass", "num_class": self.n_classes}

    def fit(self, X, y, Xv=None, yv=None, rounds=600, early_stopping=40):
        import lightgbm as lgb

        p = {"verbose": -1, "seed": self.seed, "num_threads": max(1, (os.cpu_count() or 2) - 1),
             "bagging_freq": 1, "min_data_per_group": 5, "cat_smooth": 5, **self.params, **self._objective()}
        cat = self.cat_idx or "auto"
        tr = lgb.Dataset(X, label=y, categorical_feature=cat, free_raw_data=False)
        callbacks = [lgb.log_evaluation(0)]
        sets = None
        if Xv is not None and len(Xv):
            sets = [lgb.Dataset(Xv, label=yv, categorical_feature=cat, reference=tr)]
            if early_stopping:
                callbacks.append(lgb.early_stopping(early_stopping, verbose=False))
        self.model = lgb.train(p, tr, num_boost_round=rounds, valid_sets=sets, callbacks=callbacks)
        self.best_rounds = int(self.model.best_iteration or self.model.current_iteration())
        return self

    def raw(self, X):
        out = self.model.predict(X, raw_score=True, num_iteration=self.best_rounds or None)
        return np.asarray(out)

    def contrib(self, X):
        out = np.asarray(self.model.predict(X, pred_contrib=True, num_iteration=self.best_rounds or None))
        if self.task != "regression" and self.n_classes > 2:
            return out.reshape(len(X), self.n_classes, -1)
        return out

    def gain(self):
        return np.asarray(self.model.feature_importance("gain"), dtype=float)

    def save(self, job_dir):
        path = os.path.join(job_dir, "model.txt")
        self.model.save_model(path, num_iteration=self.best_rounds or None)
        return path

    @classmethod
    def load(cls, path, task, n_classes, cat_idx):
        import lightgbm as lgb

        b = cls(task, n_classes, {}, cat_idx)
        b.model = lgb.Booster(model_file=path)
        return b


class XGBoostBooster(Booster):
    library, ext = "xgboost", "ubj"

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        try:
            import xgboost  # noqa: F401
        except Exception as e:  # noqa: BLE001
            raise BoosterUnavailable(f"xgboost is not installed ({e})")

    def fit(self, X, y, Xv=None, yv=None, rounds=600, early_stopping=40):
        import xgboost as xgb

        p = {"seed": self.seed, "nthread": max(1, (os.cpu_count() or 2) - 1), "verbosity": 0, "tree_method": "hist", **self.params}
        if self.task == "regression":
            p["objective"] = "reg:squarederror"
        elif self.n_classes <= 2:
            p["objective"] = "binary:logistic"
        else:
            p.update(objective="multi:softprob", num_class=self.n_classes)
        dtr = xgb.DMatrix(X, label=y, missing=np.nan)
        evals = []
        if Xv is not None and len(Xv):
            evals = [(xgb.DMatrix(Xv, label=yv, missing=np.nan), "val")]
        self.model = xgb.train(p, dtr, num_boost_round=rounds, evals=evals, verbose_eval=False,
                               early_stopping_rounds=early_stopping if evals and early_stopping else None)
        self.best_rounds = int(getattr(self.model, "best_iteration", None) + 1) if evals and early_stopping else rounds
        return self

    def _dm(self, X):
        import xgboost as xgb

        return xgb.DMatrix(X, missing=np.nan)

    def raw(self, X):
        return np.asarray(self.model.predict(self._dm(X), output_margin=True, iteration_range=(0, self.best_rounds)))

    def contrib(self, X):
        return np.asarray(self.model.predict(self._dm(X), pred_contribs=True, iteration_range=(0, self.best_rounds)))

    def gain(self):
        d = self.model.get_score(importance_type="total_gain")
        n = self.model.num_features()
        out = np.zeros(n)
        for k, v in d.items():
            out[int(k[1:])] = v
        return out

    def save(self, job_dir):
        path = os.path.join(job_dir, "model.ubj")
        self.model.save_model(path)
        return path

    @classmethod
    def load(cls, path, task, n_classes, cat_idx):
        import xgboost as xgb

        b = cls(task, n_classes, {}, cat_idx)
        b.model = xgb.Booster()
        b.model.load_model(path)
        b.best_rounds = b.model.num_boosted_rounds()
        return b


class CatBoostBooster(Booster):
    library, ext = "catboost", "cbm"

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        try:
            import catboost  # noqa: F401
            import pandas  # noqa: F401
        except Exception as e:  # noqa: BLE001
            raise BoosterUnavailable(f"catboost/pandas is not installed ({e})")

    def _pool(self, X, y=None):
        import pandas as pd
        from catboost import Pool

        df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
        for i in self.cat_idx:
            df[f"f{i}"] = df[f"f{i}"].fillna(0).astype(int)
        return Pool(df, label=y, cat_features=[f"f{i}" for i in self.cat_idx])

    def fit(self, X, y, Xv=None, yv=None, rounds=600, early_stopping=40):
        from catboost import CatBoostClassifier, CatBoostRegressor

        p = {"random_seed": self.seed, "verbose": False, "allow_writing_files": False,
             "thread_count": max(1, (os.cpu_count() or 2) - 1), "iterations": rounds, **self.params}
        p["iterations"] = min(p["iterations"], rounds)
        if self.task == "regression":
            m = CatBoostRegressor(**p)
        elif self.n_classes <= 2:
            m = CatBoostClassifier(**p)
        else:
            m = CatBoostClassifier(loss_function="MultiClass", **p)
        kw: dict[str, Any] = {}
        if Xv is not None and len(Xv):
            kw = {"eval_set": self._pool(Xv, yv), "use_best_model": bool(early_stopping),
                  "early_stopping_rounds": early_stopping or None}
        m.fit(self._pool(X, y), **kw)
        self.model = m
        self.best_rounds = int(m.get_best_iteration() + 1) if kw and early_stopping else p["iterations"]
        return self

    def raw(self, X):
        out = np.asarray(self.model.predict(self._pool(X), prediction_type="RawFormulaVal"))
        return out

    def contrib(self, X):
        return np.asarray(self.model.get_feature_importance(self._pool(X), type="ShapValues"))

    def gain(self):
        return np.asarray(self.model.get_feature_importance(), dtype=float)

    def save(self, job_dir):
        path = os.path.join(job_dir, "model.cbm")
        self.model.save_model(path)
        return path

    @classmethod
    def load(cls, path, task, n_classes, cat_idx):
        from catboost import CatBoostClassifier, CatBoostRegressor

        b = cls(task, n_classes, {}, cat_idx)
        m = CatBoostRegressor() if task == "regression" else CatBoostClassifier()
        m.load_model(path)
        b.model = m
        return b


class NumpyBooster(Booster):
    """Dependency-free histogram booster of depth-1 trees (additive model, so
    its SHAP values are exact and cheap). Used when no GBM library is
    installed; JSON-serialisable."""

    library, ext = "numpy", "json"
    BINS = 32

    def fit(self, X, y, Xv=None, yv=None, rounds=300, early_stopping=25):
        X = np.asarray(X, float)
        self.med = np.nanmedian(np.where(np.isfinite(X), X, np.nan), axis=0)
        self.med = np.where(np.isfinite(self.med), self.med, 0.0)
        Xf = np.where(np.isfinite(X), X, self.med)
        lr = float(self.params.get("learning_rate", 0.1))
        min_leaf = int(self.params.get("min_data_in_leaf", 10))
        lam = float(self.params.get("lambda_l2", 1.0))
        n, d = Xf.shape
        self.edges = []
        bins = np.zeros((n, d), dtype=np.int32)
        for j in range(d):
            e = np.unique(np.quantile(Xf[:, j], np.linspace(0, 1, self.BINS + 1)[1:-1]))
            self.edges.append(e)
            bins[:, j] = np.searchsorted(e, Xf[:, j], side="left")
        K = self.n_classes if (self.task != "regression" and self.n_classes > 2) else 1
        y = np.asarray(y)
        self.lr, self.K = lr, K
        if self.task == "regression":
            self.base = np.array([float(np.mean(y))])
        elif K == 1:
            p0 = min(max(float(np.mean(y)), 1e-6), 1 - 1e-6)
            self.base = np.array([math.log(p0 / (1 - p0))])
        else:
            pri = np.clip(np.bincount(y.astype(int), minlength=K) / n, 1e-6, 1)
            self.base = np.log(pri)
        F = np.tile(self.base, (n, 1))
        Fv = None
        if Xv is not None and len(Xv):
            Xvf = np.where(np.isfinite(Xv), Xv, self.med)
            Fv = np.tile(self.base, (len(Xv), 1))
        self.trees: list[list[tuple]] = []  # per round: per class (j, thr_idx, left, right, mean)
        best, best_r, stall = math.inf, 0, 0
        for r in range(rounds):
            if self.task == "regression":
                g, h = F[:, 0] - y, np.ones(n)
                G, H = [g], [h]
            elif K == 1:
                p = sigmoid(F[:, 0])
                G, H = [p - y], [np.maximum(p * (1 - p), 1e-6)]
            else:
                P = softmax(F)
                G = [P[:, k] - (y == k) for k in range(K)]
                H = [np.maximum(P[:, k] * (1 - P[:, k]), 1e-6) for k in range(K)]
            round_trees = []
            for k in range(K):
                t = self._stump(bins, G[k], H[k], lam, min_leaf)
                round_trees.append(t)
                F[:, k] += lr * np.where(bins[:, t[0]] <= t[1], t[2], t[3])
                if Fv is not None:
                    Fv[:, k] += lr * np.where(np.searchsorted(self.edges[t[0]], Xvf[:, t[0]], side="left") <= t[1], t[2], t[3])
            self.trees.append(round_trees)
            if Fv is not None and early_stopping:
                loss = self._loss(Fv, yv)
                if loss < best - 1e-9:
                    best, best_r, stall = loss, r + 1, 0
                else:
                    stall += 1
                    if stall >= early_stopping:
                        break
        self.best_rounds = best_r if (Fv is not None and early_stopping and best_r) else len(self.trees)
        self.trees = self.trees[: self.best_rounds]
        self._finalise(bins)
        return self

    def _loss(self, F, y):
        y = np.asarray(y)
        if self.task == "regression":
            return float(np.mean((F[:, 0] - y) ** 2))
        if self.K == 1:
            p = np.clip(sigmoid(F[:, 0]), 1e-9, 1 - 1e-9)
            return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
        P = np.clip(softmax(F), 1e-9, 1)
        return float(-np.mean(np.log(P[np.arange(len(y)), y.astype(int)])))

    def _stump(self, bins, g, h, lam, min_leaf):
        n, d = bins.shape
        best = (0, 0, 0.0, 0.0)
        best_gain = -math.inf
        Gt, Ht = g.sum(), h.sum()
        for j in range(d):
            nb = len(self.edges[j]) + 1
            if nb < 2:
                continue
            gs = np.bincount(bins[:, j], weights=g, minlength=nb)
            hs = np.bincount(bins[:, j], weights=h, minlength=nb)
            cs = np.bincount(bins[:, j], minlength=nb)
            gl, hl, cl = np.cumsum(gs)[:-1], np.cumsum(hs)[:-1], np.cumsum(cs)[:-1]
            gr, hr, cr = Gt - gl, Ht - hl, n - cl
            ok = (cl >= min_leaf) & (cr >= min_leaf)
            if not ok.any():
                continue
            gain = np.where(ok, gl ** 2 / (hl + lam) + gr ** 2 / (hr + lam), -np.inf)
            b = int(np.argmax(gain))
            if gain[b] > best_gain:
                best_gain = gain[b]
                best = (j, b, float(-gl[b] / (hl[b] + lam)), float(-gr[b] / (hr[b] + lam)))
        return best

    def _finalise(self, bins):
        """Per-stump expected output (for exact additive SHAP)."""
        self.means = []
        for round_trees in self.trees:
            ms = []
            for (j, b, lv, rv) in round_trees:
                frac = float(np.mean(bins[:, j] <= b))
                ms.append(frac * lv + (1 - frac) * rv)
            self.means.append(ms)

    def _bin(self, X):
        Xf = np.where(np.isfinite(X), X, self.med)
        return Xf

    def raw(self, X):
        X = self._bin(np.asarray(X, float))
        F = np.tile(self.base, (len(X), 1))
        for round_trees in self.trees:
            for k, (j, b, lv, rv) in enumerate(round_trees):
                idx = np.searchsorted(self.edges[j], X[:, j], side="left")
                F[:, k] += self.lr * np.where(idx <= b, lv, rv)
        return F[:, 0] if self.K == 1 else F

    def contrib(self, X):
        X = self._bin(np.asarray(X, float))
        n, d = X.shape
        out = np.zeros((n, self.K, d + 1))
        out[:, :, d] = self.base
        for ri, round_trees in enumerate(self.trees):
            for k, (j, b, lv, rv) in enumerate(round_trees):
                idx = np.searchsorted(self.edges[j], X[:, j], side="left")
                val = np.where(idx <= b, lv, rv)
                out[:, k, j] += self.lr * (val - self.means[ri][k])
                out[:, k, d] += self.lr * self.means[ri][k]
        return out[:, 0, :] if self.K == 1 else out

    def gain(self):
        d = len(self.edges)
        g = np.zeros(d)
        for round_trees in self.trees:
            for (j, b, lv, rv) in round_trees:
                g[j] += abs(lv - rv)
        return g

    def save(self, job_dir):
        path = os.path.join(job_dir, "model.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump({
                "format": "distillery-numpy-stumps-v1", "task": self.task, "n_classes": self.n_classes,
                "lr": self.lr, "K": self.K, "base": self.base.tolist(), "median": self.med.tolist(),
                "edges": [e.tolist() for e in self.edges],
                "trees": [[list(t) for t in rt] for rt in self.trees], "means": self.means,
            }, f)
        return path

    @classmethod
    def load(cls, path, task, n_classes, cat_idx):
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        b = cls(task, n_classes, {}, cat_idx)
        b.lr, b.K = d["lr"], d["K"]
        b.base = np.array(d["base"])
        b.med = np.array(d["median"])
        b.edges = [np.array(e) for e in d["edges"]]
        b.trees = [[tuple(t) for t in rt] for rt in d["trees"]]
        b.means = d["means"]
        b.best_rounds = len(b.trees)
        return b


BOOSTERS = {"lightgbm": LightGBMBooster, "xgboost": XGBoostBooster, "catboost": CatBoostBooster, "numpy": NumpyBooster}


def available_boosters() -> list[str]:
    out = []
    for name in ("lightgbm", "xgboost", "catboost"):
        try:
            BOOSTERS[name]("regression", 1, {}, [])
            out.append(name)
        except BoosterUnavailable:
            pass
    return out


# ---------------------------------------------------------------------------
# Platt calibration
# ---------------------------------------------------------------------------

def fit_platt(scores: np.ndarray, y: np.ndarray, iters: int = 50) -> tuple[float, float]:
    """p = sigmoid(a * s + b) fitted by Newton's method with a small ridge."""
    a, b = 1.0, 0.0
    s = np.asarray(scores, float)
    y = np.asarray(y, float)
    for _ in range(iters):
        p = sigmoid(a * s + b)
        w = np.maximum(p * (1 - p), 1e-9)
        g = np.array([np.sum((p - y) * s) + 1e-4 * (a - 1), np.sum(p - y)])
        H = np.array([[np.sum(w * s * s) + 1e-4, np.sum(w * s)], [np.sum(w * s), np.sum(w)]])
        try:
            step = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            break
        a, b = a - step[0], b - step[1]
        if np.max(np.abs(step)) < 1e-8:
            break
    return float(a), float(b)


# ---------------------------------------------------------------------------
# Tabular serving model
# ---------------------------------------------------------------------------

def format_class(c: Any) -> str:
    if isinstance(c, float) and c == int(c):
        return str(int(c))
    return str(c)


class TableModel:
    """A trained tabular model: preprocessing + booster + calibration."""

    def __init__(self, meta: dict[str, Any], pre: Preprocessor, booster: Booster) -> None:
        self.meta, self.pre, self.booster = meta, pre, booster
        self.task = meta["task"]
        self.classes: list[str] = meta.get("classes", [])
        cal = meta.get("calibration") or {}
        self.cal = (cal["a"], cal["b"]) if cal.get("method") == "platt" else None

    @staticmethod
    def load(job_dir: str) -> "TableModel":
        with open(os.path.join(job_dir, "schema.json"), "r", encoding="utf-8") as f:
            meta = json.load(f)
        if meta.get("kind") != "tabular":
            raise ValueError("job directory does not hold a tabular model")
        pre = Preprocessor(meta["preprocessor"])
        m = meta["model"]
        cls = BOOSTERS[m["library"]]
        try:
            booster = cls.load(os.path.join(job_dir, m["file"]), meta["task"], len(meta.get("classes", [])), pre.cat_idx)
        except BoosterUnavailable as e:
            raise ValueError(f"this model needs {m['library']}: {e}")
        return TableModel(meta, pre, booster)

    def probabilities(self, raw: np.ndarray) -> np.ndarray:
        if len(self.classes) <= 2:
            z = raw if self.cal is None else self.cal[0] * raw + self.cal[1]
            p1 = sigmoid(z)
            return np.column_stack([1 - p1, p1])
        return softmax(raw)

    def predict_rows(self, rows: Sequence[dict[str, Any]]) -> tuple[Any, np.ndarray]:
        X = self.pre.transform(rows)
        raw = self.booster.raw(X)
        if self.task == "regression":
            return raw.astype(float), X
        return self.probabilities(raw), X

    def check(self, row: dict[str, Any]) -> None:
        missing = [f["name"] for f in self.pre.features if not f.get("optional") and is_missing(row.get(f["name"]))]
        if missing:
            raise ValueError("missing required fields: " + ", ".join(missing))
        for f in self.pre.features:
            v = row.get(f["name"])
            if is_missing(v):
                continue
            if f["type"] == "numeric" and math.isnan(to_float(v)):
                raise ValueError(f"feature {f['name']!r} must be numeric (got {v!r})")
            if f["type"] == "datetime" and parse_dt(v) is None:
                raise ValueError(f"feature {f['name']!r} must be a date/time (got {v!r})")
            if f["type"] == "categorical" and not f.get("open_vocabulary"):
                if str(v).strip().lower() not in self.pre._lookup[f["name"]]:
                    raise ValueError(f"unknown category {str(v)!r} for feature {f['name']!r} "
                                     f"(allowed: {', '.join(f['categories'])})")

    def predict(self, row: dict[str, Any], top: int = 5) -> dict[str, Any]:
        self.check(row)
        out, X = self.predict_rows([row])
        contrib = self.booster.contrib(X)
        res: dict[str, Any] = {"explanation_method": "shap"}
        if self.task == "regression":
            res["prediction"] = float(out[0])
            c = contrib[0]
            sign = 1.0
        else:
            p = out[0]
            k = int(np.argmax(p))
            res["prediction"] = self.classes[k]
            res["probability"] = float(p[k])
            res["probabilities"] = {self.classes[i]: float(p[i]) for i in range(len(self.classes))}
            if len(self.classes) <= 2:
                c, sign = contrib[0], (1.0 if k == 1 else -1.0)
            else:
                c, sign = contrib[0][k], 1.0
        agg: dict[str, float] = {}
        for col_owner, v in zip(self.pre.owner, c[:-1]):
            agg[col_owner] = agg.get(col_owner, 0.0) + float(v) * sign
        ranked = sorted(agg.items(), key=lambda kv: -abs(kv[1]))[:top]
        res["top_factors"] = [{"feature": k, "impact": round(v, 6)} for k, v in ranked]
        return res



# ---------------------------------------------------------------------------
# Hyper-parameter search spaces (library-native names)
# ---------------------------------------------------------------------------

def sample_params(lib: str, rng: np.random.Generator) -> dict[str, Any]:
    lr = float(10 ** rng.uniform(-2.0, -0.7))
    l2 = float(10 ** rng.uniform(-1, 1.5))
    if lib == "lightgbm":
        return {"learning_rate": lr, "num_leaves": int(rng.integers(4, 64)),
                "min_data_in_leaf": int(rng.integers(3, 40)), "lambda_l2": l2,
                "feature_fraction": float(rng.uniform(0.6, 1.0)), "bagging_fraction": float(rng.uniform(0.6, 1.0))}
    if lib == "xgboost":
        return {"eta": lr, "max_depth": int(rng.integers(2, 9)), "min_child_weight": float(rng.integers(1, 12)),
                "lambda": l2, "colsample_bytree": float(rng.uniform(0.6, 1.0)), "subsample": float(rng.uniform(0.6, 1.0))}
    if lib == "catboost":
        return {"learning_rate": lr, "depth": int(rng.integers(3, 9)), "l2_leaf_reg": l2}
    return {"learning_rate": min(lr * 2, 0.3), "min_data_in_leaf": int(rng.integers(3, 40)), "lambda_l2": l2}


def default_params(lib: str) -> dict[str, Any]:
    return {"lightgbm": {"learning_rate": 0.05, "num_leaves": 31, "min_data_in_leaf": 10, "lambda_l2": 1.0,
                         "feature_fraction": 0.9, "bagging_fraction": 0.9},
            "xgboost": {"eta": 0.05, "max_depth": 6, "min_child_weight": 1.0, "lambda": 1.0,
                        "colsample_bytree": 0.9, "subsample": 0.9},
            "catboost": {"learning_rate": 0.06, "depth": 6, "l2_leaf_reg": 3.0},
            "numpy": {"learning_rate": 0.1, "min_data_in_leaf": 10, "lambda_l2": 1.0}}[lib]


# ---------------------------------------------------------------------------
# Forecasting
# ---------------------------------------------------------------------------

FREQ_SEASON = {"H": 24, "D": 7, "W": 52, "M": 12, "Q": 4, "Y": 1}
Z90 = 1.2815515655446004
MAX_GRID_POINTS = 200_000
TS_FMT = "%Y-%m-%dT%H:%M:%S"


def season_for(freq: str, override: int = 0) -> int:
    return override if override and override > 0 else FREQ_SEASON.get(freq, 7)


def _add_months(d: datetime, n: int) -> datetime:
    m = d.month - 1 + n
    return d.replace(year=d.year + m // 12, month=m % 12 + 1, day=1)


def align_time(d: datetime, freq: str) -> datetime:
    if freq == "H":
        return d.replace(minute=0, second=0, microsecond=0)
    d0 = d.replace(hour=0, minute=0, second=0, microsecond=0)
    if freq == "W":
        return d0 - timedelta(days=d0.weekday())
    if freq == "M":
        return d0.replace(day=1)
    if freq == "Q":
        return d0.replace(month=(d0.month - 1) // 3 * 3 + 1, day=1)
    if freq == "Y":
        return d0.replace(month=1, day=1)
    return d0


def step_time(d: datetime, freq: str, n: int = 1) -> datetime:
    if freq == "H":
        return d + timedelta(hours=n)
    if freq == "W":
        return d + timedelta(days=7 * n)
    if freq == "M":
        return _add_months(d, n)
    if freq == "Q":
        return _add_months(d, 3 * n)
    if freq == "Y":
        return d.replace(year=d.year + n)
    return d + timedelta(days=n)


class Series:
    __slots__ = ("id", "times", "y")

    def __init__(self, sid: str, times: list[datetime], y: np.ndarray) -> None:
        self.id, self.times, self.y = sid, times, y


def build_series(rows: Iterable[dict[str, Any]], ts_col: str, y_col: str, id_col: str, freq: str) -> tuple[list[Series], int]:
    """Group rows into regular per-item series (period duplicates averaged,
    gaps linearly interpolated). Returns (series, interpolated_points)."""
    buckets: dict[str, dict[datetime, list[float]]] = {}
    for r in rows:
        d = parse_dt(r.get(ts_col))
        v = to_float(r.get(y_col))
        if d is None or math.isnan(v):
            continue
        sid = "" if not id_col or is_missing(r.get(id_col)) else str(r.get(id_col)).strip()
        buckets.setdefault(sid, {}).setdefault(align_time(d, freq), []).append(v)
    out: list[Series] = []
    filled = 0
    for sid in sorted(buckets):
        b = buckets[sid]
        keys = sorted(b)
        t, last = keys[0], keys[-1]
        times: list[datetime] = []
        vals: list[float] = []
        while t <= last:
            if len(times) >= MAX_GRID_POINTS:
                raise ValueError(f"series {sid!r} spans too many {freq} periods; check the frequency")
            times.append(t)
            vals.append(float(np.mean(b[t])) if t in b else float("nan"))
            t = step_time(t, freq)
        y = np.array(vals, dtype=float)
        nan = np.isnan(y)
        if nan.any():
            idx = np.arange(len(y))
            y[nan] = np.interp(idx[nan], idx[~nan], y[~nan])
            filled += int(nan.sum())
        out.append(Series(sid, times, y))
    if not out:
        raise ValueError("no usable (timestamp, target) rows")
    return out, filled


def _sd(x: np.ndarray) -> float:
    return float(np.std(x, ddof=1)) if len(x) > 1 else 0.0


def _steps(h: int) -> np.ndarray:
    return np.arange(1, h + 1, dtype=float)


def m_naive(y, m, h):
    sd = _sd(np.diff(y)) if len(y) > 1 else 0.0
    return np.full(h, y[-1]), sd * np.sqrt(_steps(h))


def m_seasonal_naive(y, m, h):
    n = len(y)
    if m < 2 or n < m + 1:
        return m_naive(y, m, h)
    sd = _sd(y[m:] - y[:-m])
    i = np.arange(h)
    return y[n - m + (i % m)], sd * np.sqrt(i // m + 1.0)


def m_drift(y, m, h):
    n = len(y)
    if n < 3:
        return m_naive(y, m, h)
    slope = (y[-1] - y[0]) / (n - 1)
    sd = _sd(np.diff(y) - slope)
    s = _steps(h)
    return y[-1] + slope * s, sd * np.sqrt(s * (1 + s / n))


def m_ses(y, m, h):
    n = len(y)
    if n < 3:
        return m_naive(y, m, h)
    best_a, best = 0.5, math.inf
    for a in np.arange(0.1, 0.95, 0.1):
        level, sse = y[0], 0.0
        for t in range(1, n):
            e = y[t] - level
            sse += e * e
            level += a * e
        if sse < best:
            best_a, best = float(a), sse
    level, errs = y[0], []
    for t in range(1, n):
        e = y[t] - level
        errs.append(e)
        level += best_a * e
    sd = _sd(np.array(errs))
    return np.full(h, level), sd * np.sqrt(1 + np.arange(h) * best_a ** 2)


def m_holt_winters(y, m, h):
    """Additive Holt-Winters (damped trend). Falls back to damped Holt when no
    usable season, to SES when the series is tiny."""
    n = len(y)
    seasonal = m >= 2 and n >= 2 * m
    if n < 6:
        return m_ses(y, m, h)
    grid_a, grid_b, grid_g = (0.1, 0.3, 0.6), (0.01, 0.1), ((0.05, 0.2, 0.5) if seasonal else (0.0,))
    phi = 0.98

    def run(a, b, g):
        if seasonal:
            first = y[:m].mean()
            season = list(y[:m] - first)
            slope = (y[m:2 * m].mean() - first) / m
            level = first
            t0 = m
        else:
            level, slope = y[0], y[1] - y[0]
            season = []
            t0 = 1
        sse, errs = 0.0, []
        for t in range(t0, n):
            s = season[t % m] if seasonal else 0.0
            pred = level + phi * slope + s
            e = y[t] - pred
            errs.append(e)
            sse += e * e
            new_level = level + phi * slope + a * e
            slope = phi * slope + b * a * e
            if seasonal:
                season[t % m] = s + g * (1 - a) * e
            level = new_level
        return sse, level, slope, list(season), errs, t0

    best = None
    for a in grid_a:
        for b in grid_b:
            for g in grid_g:
                r = run(a, b, g)
                if best is None or r[0] < best[0][0]:
                    best = (r, a)
    (sse, level, slope, season, errs, t0), a = best
    sd = _sd(np.array(errs))
    i = np.arange(1, h + 1)
    damp = np.cumsum(phi ** i)
    mean = level + damp * slope
    if seasonal:
        mean = mean + np.array([season[(n + k) % m] for k in range(h)])
    return mean, sd * np.sqrt(1 + (i - 1) * a * a)


def _gbm_lags(y: np.ndarray, m: int) -> list[int]:
    base = list(range(1, 8))
    if m >= 2:
        base += [m - 1, m, m + 1, 2 * m]
    return sorted({lag for lag in base if 1 <= lag < len(y) // 2 or lag <= 3})


def _lag_features(hist: np.ndarray, lags: list[int], m: int, pos: int) -> list[float]:
    f = [hist[-lag] if lag <= len(hist) else float("nan") for lag in lags]
    win = hist[-max(m, 7):]
    f += [float(win.mean()), float(win.std()), float(pos % max(m, 1))]
    return f


def gbm_min_train(m: int) -> int:
    return max(30, 3 * max(m, 7))


def m_gbm_lag(y, m, h):
    """Global-lag GBM with recursive multi-step prediction."""
    n = len(y)
    lags = _gbm_lags(y, m)
    start = max(lags)
    if n - start < 20:
        raise ValueError("series too short for the lag model")
    X = np.array([_lag_features(y[:t], lags, m, t) for t in range(start, n)], dtype=float)
    tgt = y[start:]
    # Trend-robust: model the delta from the last value.
    ref = np.array([y[t - 1] for t in range(start, n)])
    yt = tgt - ref
    cut = max(10, int(len(X) * 0.85))
    lib = next((b for b in ("lightgbm", "xgboost") if b in available_boosters()), "numpy")
    def mk():
        return BOOSTERS[lib]("regression", 1, default_params(lib), [])
    bst = mk().fit(X[:cut], yt[:cut], X[cut:], yt[cut:], rounds=300, early_stopping=25)
    res = yt[cut:] - bst.raw(X[cut:]) if cut < len(X) else yt - bst.raw(X)
    sd = float(np.std(res)) if len(res) > 1 else _sd(yt)
    bst = mk().fit(X, yt, None, None, rounds=max(bst.best_rounds, 20), early_stopping=0)
    hist = list(y)
    out = []
    for k in range(h):
        arr = np.array(hist)
        f = np.array([_lag_features(arr, lags, m, n + k)])
        v = float(arr[-1] + bst.raw(f)[0])
        out.append(v)
        hist.append(v)
    return np.array(out), sd * np.sqrt(_steps(h))


def m_auto_ets(y, m, h):
    """statsforecast AutoETS (optional dependency)."""
    from statsforecast.models import AutoETS  # noqa: WPS433

    mod = AutoETS(season_length=max(m, 1)).fit(y.astype(np.float64))
    p = mod.predict(h=h, level=[80])
    mean = np.asarray(p["mean"], float)
    hi = np.asarray(p["hi-80"], float)
    return mean, np.maximum((hi - mean) / Z90, 0.0)


def m_chronos(y, m, h):
    """Chronos zero-shot forecaster (optional dependency, downloads weights
    on first use). Not exercised by the automated tests."""
    import torch  # noqa: WPS433
    from chronos import BaseChronosPipeline  # noqa: WPS433

    global _CHRONOS
    if "_CHRONOS" not in globals() or _CHRONOS is None:
        _CHRONOS = BaseChronosPipeline.from_pretrained(os.environ.get("DISTILLERY_CHRONOS", "amazon/chronos-bolt-small"),
                                                       device_map="cpu", torch_dtype=torch.float32)
    q, mean = _CHRONOS.predict_quantiles(context=torch.tensor(y[-2048:], dtype=torch.float32),
                                         prediction_length=h, quantile_levels=[0.1, 0.5, 0.9])
    q = q[0].numpy()
    return q[:, 1].astype(float), np.maximum((q[:, 2] - q[:, 0]) / (2 * Z90), 0.0)


_CHRONOS = None

FORECAST_MODELS = {
    "seasonal_naive": m_seasonal_naive, "naive": m_naive, "drift": m_drift, "ses": m_ses,
    "holt_winters": m_holt_winters, "gbm_lag": m_gbm_lag, "auto_ets": m_auto_ets, "chronos": m_chronos,
}


def _module_available(name: str) -> bool:
    import importlib.util

    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def candidate_models(family: str) -> list[str]:
    """Model names for a family selector, always led by the baseline."""
    base = ["seasonal_naive"]
    stats = ["naive", "drift", "ses", "holt_winters"] + (["auto_ets"] if _module_available("statsforecast") else [])
    if family in ("", "auto"):
        return base + stats + ["gbm_lag"]
    if family == "seasonal_naive":
        return base
    if family == "stats":
        return base + stats
    if family == "gbm":
        return base + ["gbm_lag"]
    if family == "chronos":
        return base + ["chronos"]
    raise ValueError(f"unknown forecast model family {family!r}")


def model_min_train(name: str, m: int) -> int:
    if name == "gbm_lag":
        return gbm_min_train(m)
    if name == "holt_winters":
        return max(6, 2 * m if m >= 2 else 6)
    return max(2 * m, 8) if name == "seasonal_naive" else 8


def _safe_forecast(name: str, y: np.ndarray, m: int, h: int) -> tuple[np.ndarray, np.ndarray]:
    mean, sig = FORECAST_MODELS[name](np.asarray(y, float), m, h)
    mean = np.asarray(mean, float)
    sig = np.asarray(sig, float)
    if not (np.all(np.isfinite(mean)) and np.all(np.isfinite(sig))):
        raise ValueError(f"{name} produced non-finite output")
    return mean, sig


def mase_scale(train: np.ndarray, m: int) -> float:
    lag = m if m >= 1 and len(train) > m else 1
    d = np.abs(train[lag:] - train[:-lag]) if len(train) > lag else np.array([])
    s = float(d.mean()) if len(d) else 0.0
    return s if s > 0 else 1.0


def score_forecast(train, actual, mean, sigma, m) -> dict[str, float]:
    actual, mean, sigma = map(np.asarray, (actual, mean, sigma))
    err = actual - mean
    mase = float(np.abs(err).mean() / mase_scale(np.asarray(train), m))
    denom = np.abs(actual) + np.abs(mean)
    smape = float(np.mean(np.where(denom > 0, 2 * np.abs(err) / np.where(denom > 0, denom, 1), 0.0)))
    pin = 0.0
    for z, tau in ((-Z90, 0.1), (0.0, 0.5), (Z90, 0.9)):
        qv = mean + z * sigma
        pin += float(np.sum(np.where(actual >= qv, tau * (actual - qv), (1 - tau) * (qv - actual))))
    den = float(np.abs(actual).sum())
    return {"mase": mase, "smape": smape, "wql": (2 * pin / 3 / den) if den > 0 else 0.0}


def _mean_scores(ss: list[dict[str, float]]) -> dict[str, float]:
    return {k: float(np.mean([s[k] for s in ss])) for k in ("mase", "smape", "wql")}


def backtest(series: list[Series], cfg: dict[str, Any], progress=None, log=None) -> dict[str, Any]:
    """Rolling-origin backtest. `cfg`: frequency, season_length, horizon,
    backtest_windows, model, metric, time_budget_sec. Returns the metrics
    block in the Go TrainingMetrics shape plus `interval_scale`, `chosen`."""
    import time as _t

    freq, h = cfg["frequency"], int(cfg["horizon"])
    m = season_for(freq, int(cfg.get("season_length") or 0))
    windows = max(1, int(cfg.get("backtest_windows") or 3))
    metric = cfg.get("metric") or "mase"
    if metric not in ("mase", "smape", "wql"):
        raise ValueError(f"unknown forecast metric {metric!r}")
    names = candidate_models(cfg.get("model") or "auto")
    budget = float(cfg.get("time_budget_sec") or 0) or 300.0
    t0 = _t.time()

    need = max(2 * m, 8) + h * windows
    used = [s for s in series if len(s.y) >= need]
    if not used:
        longest = max(len(s.y) for s in series)
        raise ValueError(f"no series is long enough for horizon {h} x {windows} backtest windows "
                         f"(needs >= {need} points, longest has {longest}); shorten the horizon or the number of windows")

    # Drop models that cannot be trained on the shortest window.
    shortest_train = min(len(s.y) for s in used) - h * windows
    kept: list[str] = []
    for n in names:
        if shortest_train < model_min_train(n, m):
            if log:
                log(f"model {n} skipped: shortest training window has {shortest_train} points")
            continue
        if n == "auto_ets" or n == "chronos":
            pass
        kept.append(n)
    names = kept

    # records[model] = list of (window, series_idx, train, actual, mean, sigma)
    records: dict[str, list[tuple]] = {}
    failed: dict[str, str] = {}
    for mi, name in enumerate(names):
        if name != "seasonal_naive" and _t.time() - t0 > budget:
            if log:
                log(f"time budget reached: model {name} skipped")
            continue
        recs = []
        try:
            for si, s in enumerate(used):
                for w in range(1, windows + 1):
                    cut = len(s.y) - h * (windows - w + 1)
                    train, actual = s.y[:cut], s.y[cut:cut + h]
                    mean, sig = _safe_forecast(name, train, m, h)
                    recs.append((w, si, train, actual, mean, sig))
        except Exception as e:  # noqa: BLE001
            failed[name] = str(e)
            if name == "seasonal_naive":
                raise
            if log:
                log(f"model {name} failed and was dropped: {e}")
            continue
        records[name] = recs
        if progress:
            progress(mi + 1, len(names))

    def interval_scale(recs: list[tuple]) -> float:
        z = []
        for _, _, _, actual, mean, sig in recs:
            ok = sig > 1e-12
            z.extend(np.abs(actual - mean)[ok] / sig[ok])
        if len(z) < 8:
            return 1.0
        return float(np.clip(np.quantile(z, 0.8) / Z90, 0.5, 4.0))

    scale = {n: interval_scale(r) for n, r in records.items()}

    def scored(name: str, recs: list[tuple], window: int | None = None):
        return [score_forecast(t, a, mn, sg * scale[name], m) for (w, _, t, a, mn, sg) in recs
                if window is None or w == window]

    overall = {n: _mean_scores(scored(n, r)) for n, r in records.items()}
    best = min(overall, key=lambda n: overall[n][metric])
    base_val = overall["seasonal_naive"][metric]
    board = [{"model": n, "score": round(overall[n][metric], 4), "metric": metric,
              "baseline": n == "seasonal_naive", "chosen": n == best} for n in overall]
    board.sort(key=lambda c: c["score"])

    origins: dict[int, str] = {}
    for (w, si, train, *_r) in records["seasonal_naive"]:
        origins.setdefault(w, iso(used[si].times[len(train) - 1]))
    bt = []
    for w in range(1, windows + 1):
        bs = _mean_scores(scored("seasonal_naive", records["seasonal_naive"], w))
        for n in overall:
            s = _mean_scores(scored(n, records[n], w))
            bt.append({"window": w, "origin": origins.get(w, ""), "model": n, "mase": round(s["mase"], 4),
                       "smape": round(s["smape"], 4), "wql": round(s["wql"], 4), "seasonal_naive_mase": round(bs["mase"], 4)})

    # Plot: last window of the first series for the chosen model.
    plot = None
    s0 = used[0]
    cut = len(s0.y) - h
    for (w, si, train, actual, mean, sig) in records[best]:
        if si == 0 and w == windows:
            sn = next(r for r in records["seasonal_naive"] if r[0] == w and r[1] == 0)[4]
            k = scale[best]
            plot = {"timestamps": [iso(t) for t in s0.times[cut:cut + h]], "actual": [float(v) for v in actual],
                    "predicted": [float(v) for v in mean], "seasonal_naive": [float(v) for v in sn],
                    "lower": [float(v) for v in mean - Z90 * sig * k], "upper": [float(v) for v in mean + Z90 * sig * k]}
            break

    return {
        "primary_metric": metric, "primary_value": round(overall[best][metric], 4), "higher_is_better": False,
        "final_loss": round(overall[best][metric], 4),
        "baselines": {f"seasonal_naive_{metric}": round(base_val, 4)},
        "improvement_over_baseline": round(base_val - overall[best][metric], 4),
        "leaderboard": board, "backtest": bt, "backtest_plot": plot, "chosen_model": best, "season_length": m,
        "interval_scale": scale[best], "series_used": len(used), "series_total": len(series),
        "failed_models": failed,
    }


class ForecastModel:
    """A trained forecaster: the chosen model family refit on the stored
    history (statistical models are fitted per call, which takes milliseconds;
    nothing is pickled)."""

    def __init__(self, meta: dict[str, Any]) -> None:
        self.meta = meta
        c = meta["config"]
        self.ts_col, self.y_col, self.id_col = c["timestamp"], c["target"], c.get("item_id") or ""
        self.freq, self.m = c["frequency"], c["season_length"]
        self.chosen = meta["chosen_model"]
        self.scale = float(meta.get("interval_scale", 1.0))
        self.stored = [(d["id"], [parse_dt(t) for t in d["t"]], np.array(d["y"], float)) for d in meta["series"]]

    @staticmethod
    def load(job_dir: str) -> "ForecastModel":
        with open(os.path.join(job_dir, "forecast_model.json"), "r", encoding="utf-8") as f:
            return ForecastModel(json.load(f))

    def _rows(self) -> list[dict[str, Any]]:
        return [{self.ts_col: iso(t), self.y_col: float(v), self.id_col: sid}
                for sid, ts, ys in self.stored for t, v in zip(ts, ys)] if self.id_col else \
               [{self.ts_col: iso(t), self.y_col: float(v)} for _sid, ts, ys in self.stored for t, v in zip(ts, ys)]

    def forecast(self, history: Sequence[dict[str, Any]] | None = None, horizon: int = 14, item_id: str = "") -> dict[str, Any]:
        if horizon < 1 or horizon > 1000:
            raise ValueError("horizon must be between 1 and 1000")
        rows = self._rows()
        if history:
            # Caller observations replace stored ones of the same period.
            def key(r):
                d = parse_dt(r.get(self.ts_col))
                sid = str(r.get(self.id_col, "")).strip() if self.id_col else ""
                return (sid, align_time(d, self.freq)) if d else None
            over = {k for k in map(key, history) if k}
            rows = [r for r in rows if key(r) not in over] + list(history)
        series, _ = build_series(rows, self.ts_col, self.y_col, self.id_col, self.freq)
        by_id = {s.id: s for s in series}
        if not item_id:
            if len(by_id) != 1:
                raise ValueError(f"this model has {len(by_id)} series; pass \"item_id\" (e.g. {series[0].id!r})")
            item_id = series[0].id
        if item_id not in by_id:
            raise ValueError(f"unknown item_id {item_id!r}")
        s = by_id[item_id]
        try:
            mean, sig = _safe_forecast(self.chosen, s.y, self.m, horizon)
            used = self.chosen
        except Exception:  # noqa: BLE001 - e.g. history too short for the lag model
            mean, sig = _safe_forecast("seasonal_naive", s.y, self.m, horizon)
            used = "seasonal_naive"
        sig = sig * self.scale
        pts = []
        for i in range(horizon):
            pts.append({"timestamp": iso(step_time(s.times[-1], self.freq, i + 1)), "value": round(float(mean[i]), 4),
                        "lower": round(float(mean[i] - Z90 * sig[i]), 4), "upper": round(float(mean[i] + Z90 * sig[i]), 4)})
        return {"forecast": pts, "item_id": item_id, "model": used, "metric": self.meta.get("primary_metric", ""),
                "metric_value": self.meta.get("primary_value", 0.0), "baselines": self.meta.get("baselines", {})}

    def series_ids(self) -> list[str]:
        return [sid for sid, _, _ in self.stored]


def load(job_dir: str):
    """Load whatever model lives in `job_dir` (tabular or forecast)."""
    with open(os.path.join(job_dir, "schema.json"), "r", encoding="utf-8") as f:
        kind = json.load(f).get("kind")
    if kind == "tabular":
        return TableModel.load(job_dir)
    if kind == "time_series":
        return ForecastModel.load(job_dir)
    raise ValueError(f"unknown model kind {kind!r}")
