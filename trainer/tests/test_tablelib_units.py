"""Fast, dependency-light unit tests for the tabular / forecasting helpers.

Nothing here trains a real booster: these cover the pure functions (value
parsing, preprocessing, metrics, splitters, calibration, the time grid and the
statistical forecasters) so regressions surface in milliseconds.
"""

from __future__ import annotations

import math
from datetime import datetime

import numpy as np
import pytest

from trainer import tablelib as T
from trainer.tasks import tabular as TB


# --- value parsing ---------------------------------------------------------

@pytest.mark.parametrize("v", [None, "", "  ", "NaN", "null", "NA", float("nan")])
def test_is_missing_true(v):
    assert T.is_missing(v)


@pytest.mark.parametrize("v", [0, "0", "abc", False, 3.5])
def test_is_missing_false(v):
    assert not T.is_missing(v)


def test_to_float_never_raises():
    assert T.to_float("3.5") == 3.5
    assert T.to_float(7) == 7.0
    for bad in ("x", None, "", "inf", float("inf"), [1]):
        assert math.isnan(T.to_float(bad))


@pytest.mark.parametrize("raw,want", [
    ("2024-03-05", datetime(2024, 3, 5)),
    ("2024-03-05 14:30:00", datetime(2024, 3, 5, 14, 30)),
    ("05.03.2024", datetime(2024, 3, 5)),
    ("2024-03", datetime(2024, 3, 1)),
    ("2024-03-05T10:00:00+02:00", datetime(2024, 3, 5, 8, 0)),  # converted to naive UTC.
    ("2024-03-05T10:00:00Z", datetime(2024, 3, 5, 10, 0)),
])
def test_parse_dt_formats(raw, want):
    assert T.parse_dt(raw) == want


@pytest.mark.parametrize("bad", [None, "", "not a date", "2024-13-45"])
def test_parse_dt_rejects(bad):
    assert T.parse_dt(bad) is None


def test_softmax_and_sigmoid_are_stable():
    z = np.array([[1000.0, 1001.0, 999.0]])
    p = T.softmax(z)
    assert p.shape == z.shape
    assert p.sum() == pytest.approx(1.0)
    assert np.all(np.isfinite(p))
    s = T.sigmoid(np.array([-1e9, 0.0, 1e9]))
    assert s[0] == pytest.approx(0.0, abs=1e-12)
    assert s[1] == 0.5
    assert s[2] == pytest.approx(1.0)


def test_load_jsonl_skips_blank_and_broken_lines(tmp_path):
    p = tmp_path / "d.jsonl"
    p.write_text('{"a": 1}\n\nnot json\n[1, 2]\n{"a": 2}\n', encoding="utf-8")
    assert T.load_jsonl(str(p)) == [{"a": 1}, {"a": 2}]


# --- preprocessor ----------------------------------------------------------

ROWS = [
    {"age": 30, "plan": "basic", "seen": "2024-01-02 10:00:00"},
    {"age": 40, "plan": "pro", "seen": "2024-02-03 11:00:00"},
    {"age": None, "plan": "basic", "seen": None},
    {"age": 50, "plan": "basic", "seen": "2024-03-04 12:00:00"},
]
COLS = {"age": "numeric", "plan": "categorical", "seen": "datetime"}


def _pre():
    return T.Preprocessor.build(ROWS, COLS, ["age", "plan", "seen"])


def test_preprocessor_build_records_schema():
    pre = _pre()
    by = {f["name"]: f for f in pre.features}
    assert by["age"]["median"] == 40.0 and by["age"]["min"] == 30.0 and by["age"]["max"] == 50.0
    assert by["age"]["optional"] is True
    assert by["plan"]["categories"] == ["basic", "pro"]  # frequency order.
    assert by["plan"]["open_vocabulary"] is False
    assert pre.columns == ["age", "plan", "seen__month", "seen__dow", "seen__hour", "seen__dom"]
    assert pre.cat_idx == [1]
    assert pre.owner[2:] == ["seen"] * 4


def test_preprocessor_transform_codes():
    pre = _pre()
    X = pre.transform([
        {"age": 25, "plan": "PRO", "seen": "2024-05-06 07:00:00"},  # case-insensitive category.
        {"age": "bad", "plan": None, "seen": "garbage"},
        {"age": 1, "plan": "enterprise", "seen": None},  # unseen category.
    ])
    assert X.shape == (3, 6)
    assert X[0, 0] == 25 and X[0, 1] == 2  # pro = code 2.
    assert list(X[0, 2:]) == [5.0, 0.0, 7.0, 6.0]  # month, Monday, hour, day.
    assert math.isnan(X[1, 0]) and X[1, 1] == 0  # missing numeric = NaN; missing category = 0.
    assert np.all(np.isnan(X[1, 2:]))
    assert X[2, 1] == 3  # K+1 = unseen.


def test_preprocessor_go_schema_and_medians():
    pre = _pre()
    g = {f["name"]: f for f in pre.go_schema()}
    assert g["age"]["min"] == 30.0 and g["age"]["max"] == 50.0
    assert g["plan"]["categories"] == ["basic", "pro"]
    assert g["seen"]["type"] == "datetime"
    med = pre.medians()
    assert len(med) == len(pre.columns)
    assert med[0] == 40.0 and med[1] == 0.0


def test_preprocessor_caps_categories():
    rows = [{"c": f"v{i}"} for i in range(T.MAX_CATEGORIES + 25)]
    pre = T.Preprocessor.build(rows, {"c": "categorical"}, ["c"])
    f = pre.features[0]
    assert len(f["categories"]) == T.MAX_CATEGORIES
    assert f["open_vocabulary"] is True


def test_preprocessor_all_missing_numeric():
    pre = T.Preprocessor.build([{"x": None}, {"x": ""}], {"x": "numeric"}, ["x"])
    f = pre.features[0]
    assert f["median"] == 0.0 and f["min"] is None
    assert "min" not in pre.go_schema()[0]


# --- tabular metrics -------------------------------------------------------

def test_roc_auc_known_values():
    y = np.array([0, 0, 1, 1])
    assert TB.roc_auc_binary(y, np.array([0.1, 0.2, 0.8, 0.9])) == 1.0
    assert TB.roc_auc_binary(y, np.array([0.9, 0.8, 0.2, 0.1])) == 0.0
    assert TB.roc_auc_binary(y, np.array([0.5, 0.5, 0.5, 0.5])) == 0.5  # ties.
    assert TB.roc_auc_binary(np.ones(4, int), np.arange(4.0)) == 0.5  # one class only.


def test_average_precision_perfect_and_worst():
    y = np.array([0, 0, 1, 1])
    assert TB.average_precision(y, np.array([0.1, 0.2, 0.8, 0.9])) == pytest.approx(1.0)
    assert TB.average_precision(y, np.array([0.9, 0.8, 0.2, 0.1])) < 0.6


def test_roc_points_are_bounded_and_monotone():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500)
    s = rng.random(500) + y * 0.3
    pts = TB.roc_points(y, s, max_points=60)
    assert 2 <= len(pts) <= 61
    assert pts[0] == [0.0, 0.0] and pts[-1] == [1.0, 1.0]
    fpr = [p[0] for p in pts]
    tpr = [p[1] for p in pts]
    assert fpr == sorted(fpr) and tpr == sorted(tpr)


def test_cls_scores_binary_and_multiclass():
    y = np.array([0, 1, 1, 0])
    p = np.array([[0.9, 0.1], [0.2, 0.8], [0.3, 0.7], [0.6, 0.4]])
    s = TB.cls_scores(y, p)
    assert s["accuracy"] == 1.0 and s["roc_auc"] == 1.0 and s["f1"] == 1.0
    assert s["logloss"] == pytest.approx(-np.mean(np.log([0.9, 0.8, 0.7, 0.6])))

    y3 = np.array([0, 1, 2, 2])
    p3 = np.eye(3)[[0, 1, 2, 1]] * 0.9 + 0.05
    s3 = TB.cls_scores(y3, p3)
    assert s3["accuracy"] == 0.75
    assert 0 < s3["f1"] < 1


def test_reg_scores():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert TB.reg_scores(y, y)["rmse"] == 0.0
    assert TB.reg_scores(y, y)["r2"] == 1.0
    s = TB.reg_scores(y, y + 1)
    assert s["rmse"] == pytest.approx(1.0) and s["mae"] == pytest.approx(1.0)
    assert TB.reg_scores(np.ones(3), np.ones(3))["r2"] == 0.0  # constant target.
    assert TB.all_scores("regression", y, y)["rmse"] == 0.0
    assert "roc_auc" in TB.all_scores("classification", np.array([0, 1]), np.array([[0.7, 0.3], [0.2, 0.8]]))


def test_brier_ece_perfect_and_miscalibrated():
    y = np.array([0, 0, 1, 1], float)
    b, e = TB.brier_ece(y, y)
    assert b == 0.0 and e == 0.0
    b2, e2 = TB.brier_ece(y, np.full(4, 0.5))
    assert b2 == pytest.approx(0.25) and e2 == pytest.approx(0.0)
    _, e3 = TB.brier_ece(y, np.array([0.95, 0.95, 0.05, 0.05]))
    assert e3 > 0.9


def test_class_label_and_infer_task():
    assert TB.class_label(True) == "true"
    assert TB.class_label(3.0) == "3"
    assert TB.class_label(" a ") == "a"
    assert TB.infer_task([0, 1, 0, 1]) == "classification"
    assert TB.infer_task(list(range(30))) == "regression"
    assert TB.infer_task([f"c{i}" for i in range(30)]) == "classification"  # non-numeric.


# --- splitters -------------------------------------------------------------

def _disjoint_cover(a, b, n):
    assert set(a).isdisjoint(b)
    assert sorted(set(a) | set(b)) == list(range(n))


def test_stratified_fold_ids_spread_every_class_evenly():
    for pos in (5, 20, 37):
        y = np.array([0] * (100 - pos) + [1] * pos)
        fold = TB.stratified_fold_ids(y, 5, np.random.default_rng(pos))
        assert set(fold) == {0, 1, 2, 3, 4}
        counts = [int(y[fold == f].sum()) for f in range(5)]
        assert max(counts) - min(counts) <= 1, counts  # no fold is starved of the minority class.
        sizes = [int((fold == f).sum()) for f in range(5)]
        assert max(sizes) - min(sizes) <= 2


def test_splitter_random_and_stratified_holdout():
    y = np.array([0] * 90 + [1] * 10)
    for strat in ("random", "stratified"):
        tr, ho = TB.Splitter(strat, y, None, None, True).holdout()
        _disjoint_cover(tr, ho, 100)
        assert len(ho) == 20
    for seed_pos in (3, 10, 25):  # regression: the holdout must always contain the minority class.
        y2 = np.array([0] * (100 - seed_pos) + [1] * seed_pos)
        _, ho = TB.Splitter("stratified", y2, None, None, True).holdout()
        assert y2[ho].sum() >= 1
        folds = TB.Splitter("stratified", y2, None, None, True).folds(np.arange(100), 5)
        if seed_pos >= 5:  # fewer minority rows than folds cannot fill every fold.
            assert all(y2[va].sum() >= 1 for _, va in folds)


def test_splitter_time_holdout_is_chronological():
    t = np.array([5, 1, 4, 2, 3, 9, 8, 7, 6, 0], float)
    tr, ho = TB.Splitter("time", np.zeros(10), t, None, False).holdout()
    assert t[tr].max() < t[ho].min()
    assert len(ho) == 2


def test_splitter_group_holdout_never_splits_a_group():
    groups = np.array(["a", "a", "b", "b", "c", "c", "d", "d", "e", "e"])
    tr, ho = TB.Splitter("group", np.zeros(10), None, groups, False).holdout()
    assert set(groups[tr]).isdisjoint(groups[ho])
    _disjoint_cover(tr, ho, 10)


def test_splitter_folds_cover_and_respect_strategy():
    n = 60
    idx = np.arange(n)
    y = np.array([0, 1] * 30)
    for strat in ("random", "stratified"):
        folds = TB.Splitter(strat, y, None, None, True).folds(idx, 5)
        assert len(folds) == 5
        seen = np.concatenate([v for _, v in folds])
        assert sorted(seen) == list(range(n))
        for tr, va in folds:
            assert set(tr).isdisjoint(va)

    t = np.arange(n, dtype=float)
    folds = TB.Splitter("time", y, t, None, True).folds(idx, 4)
    assert len(folds) == 4
    for tr, va in folds:
        assert t[tr].max() < t[va].min()  # never trains on the future.

    groups = np.array([f"g{i % 6}" for i in range(n)])
    folds = TB.Splitter("group", y, None, groups, True).folds(idx, 3)
    for tr, va in folds:
        assert set(groups[tr]).isdisjoint(groups[va])


def test_group_folds_shrink_to_group_count():
    groups = np.array(["a", "a", "b", "b"])
    folds = TB.Splitter("group", np.zeros(4), None, groups, False).folds(np.arange(4), 5)
    assert len(folds) == 2


# --- Platt calibration -----------------------------------------------------

def test_fit_platt_recovers_shape():
    rng = np.random.default_rng(0)
    s = rng.normal(0, 2, 4000)
    true_a, true_b = 0.7, -0.3
    y = (rng.random(4000) < T.sigmoid(true_a * s + true_b)).astype(float)
    a, b = T.fit_platt(s, y)
    assert a == pytest.approx(true_a, abs=0.12)
    assert b == pytest.approx(true_b, abs=0.12)


def test_fit_platt_survives_separable_data():
    s = np.array([-3.0, -2.0, 2.0, 3.0])
    y = np.array([0.0, 0.0, 1.0, 1.0])
    a, b = T.fit_platt(s, y)
    assert math.isfinite(a) and math.isfinite(b) and a > 0


# --- parameter sampling ----------------------------------------------------

@pytest.mark.parametrize("lib", ["lightgbm", "xgboost", "catboost", "numpy"])
def test_sample_and_default_params(lib):
    rng = np.random.default_rng(3)
    for _ in range(20):
        p = T.sample_params(lib, rng)
        assert all(math.isfinite(float(v)) for v in p.values())
    a = T.sample_params(lib, np.random.default_rng(5))
    b = T.sample_params(lib, np.random.default_rng(5))
    assert a == b  # seeded.
    assert T.default_params(lib)


# --- the time grid ---------------------------------------------------------

def test_align_and_step_time():
    d = datetime(2024, 5, 15, 13, 45, 10)  # a Wednesday.
    assert T.align_time(d, "H") == datetime(2024, 5, 15, 13)
    assert T.align_time(d, "D") == datetime(2024, 5, 15)
    assert T.align_time(d, "W") == datetime(2024, 5, 13)  # Monday.
    assert T.align_time(d, "M") == datetime(2024, 5, 1)
    assert T.align_time(d, "Q") == datetime(2024, 4, 1)
    assert T.align_time(d, "Y") == datetime(2024, 1, 1)

    assert T.step_time(datetime(2024, 1, 31), "D", 2) == datetime(2024, 2, 2)
    assert T.step_time(datetime(2024, 12, 1), "M") == datetime(2025, 1, 1)
    assert T.step_time(datetime(2024, 11, 1), "Q") == datetime(2025, 2, 1)
    assert T.step_time(datetime(2024, 1, 1), "Y", 3) == datetime(2027, 1, 1)
    assert T.step_time(datetime(2024, 1, 1, 23), "H") == datetime(2024, 1, 2)
    assert T.step_time(datetime(2024, 1, 1), "W") == datetime(2024, 1, 8)


def test_season_for():
    assert T.season_for("D") == 7 and T.season_for("M") == 12 and T.season_for("Q") == 4
    assert T.season_for("D", 30) == 30
    assert T.season_for("D", -1) == 7
    assert T.season_for("??") == 7


def test_build_series_multi_item_duplicates_and_gaps():
    rows = [
        {"d": "2024-01-01", "v": 10, "id": "a"},
        {"d": "2024-01-01", "v": 20, "id": "a"},  # duplicate period -> mean.
        {"d": "2024-01-04", "v": 40, "id": "a"},  # two-day gap.
        {"d": "2024-01-01", "v": 1, "id": "b"},
        {"d": "2024-01-02", "v": 2, "id": "b"},
        {"d": "bad", "v": 3, "id": "b"},  # dropped.
        {"d": "2024-01-03", "v": None, "id": "b"},  # dropped.
    ]
    series, filled = T.build_series(rows, "d", "v", "id", "D")
    assert [s.id for s in series] == ["a", "b"]
    a = series[0]
    assert list(a.y) == pytest.approx([15.0, 15 + 25 / 3, 15 + 50 / 3, 40.0])
    assert filled == 2
    assert len(series[1].y) == 2


def test_build_series_without_item_column_and_errors():
    series, _ = T.build_series([{"d": "2024-01-01", "v": 1}, {"d": "2024-01-02", "v": 2}], "d", "v", "", "D")
    assert len(series) == 1 and series[0].id == ""
    with pytest.raises(ValueError, match="no usable"):
        T.build_series([{"d": "x", "v": 1}], "d", "v", "", "D")
    with pytest.raises(ValueError, match="too many"):
        T.build_series([{"d": "1900-01-01", "v": 1}, {"d": "2100-01-01", "v": 1}], "d", "v", "", "H")


# --- statistical forecasters ----------------------------------------------

def _seasonal(n=120, m=7, slope=0.0):
    t = np.arange(n)
    return 100 + slope * t + 10 * np.sin(2 * np.pi * t / m)


@pytest.mark.parametrize("name", ["naive", "seasonal_naive", "drift", "ses", "holt_winters"])
def test_forecasters_shapes_and_finiteness(name):
    y = _seasonal(60)
    mean, sig = T._safe_forecast(name, y, 7, 10)
    assert mean.shape == (10,) and sig.shape == (10,)
    assert np.all(sig >= 0)


@pytest.mark.parametrize("name", ["naive", "seasonal_naive", "drift", "ses", "holt_winters"])
def test_forecasters_survive_tiny_and_constant_series(name):
    for y in (np.array([5.0]), np.array([5.0, 5.0]), np.full(20, 3.0)):
        mean, sig = T._safe_forecast(name, y, 7, 5)
        assert np.all(np.isfinite(mean)) and np.all(np.isfinite(sig))
    assert T._safe_forecast(name, np.full(20, 3.0), 7, 3)[0] == pytest.approx([3.0] * 3)


def test_naive_family_exact_values():
    y = np.array([1.0, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    assert list(T.m_naive(y, 7, 3)[0]) == [10, 10, 10]
    assert list(T.m_drift(y, 7, 3)[0]) == pytest.approx([11, 12, 13])
    sn = T.m_seasonal_naive(y, 3, 4)[0]
    assert list(sn) == [8, 9, 10, 8]  # last full season repeated.
    assert list(T.m_seasonal_naive(y, 1, 2)[0]) == [10, 10]  # no season -> naive.


def test_holt_winters_tracks_a_seasonal_trend():
    y = _seasonal(140, 7, slope=0.5)
    mean, _ = T.m_holt_winters(y, 7, 14)
    truth = _seasonal(154, 7, slope=0.5)[140:]
    assert np.mean(np.abs(mean - truth)) < 3.0
    naive_err = np.mean(np.abs(T.m_naive(y, 7, 14)[0] - truth))
    assert np.mean(np.abs(mean - truth)) < naive_err


def test_lag_helpers():
    lags = T._gbm_lags(np.arange(100.0), 7)
    assert lags == sorted(set(lags)) and 7 in lags and 1 in lags and all(lag >= 1 for lag in lags)
    hist = np.arange(1.0, 31.0)
    f = T._lag_features(hist, [1, 7, 100], 7, 3)
    assert f[0] == 30.0 and f[1] == 24.0 and math.isnan(f[2])
    assert f[-1] == 3.0  # position in the season.
    assert T.gbm_min_train(7) > 7


def test_candidate_models_and_min_train():
    for fam in ("auto", "stats"):
        names = T.candidate_models(fam)
        assert "seasonal_naive" in names and "naive" in names
    assert T.candidate_models("seasonal_naive") == ["seasonal_naive"]
    with pytest.raises(ValueError):
        T.candidate_models("nope")
    assert "gbm_lag" not in T.candidate_models("stats")
    assert T.model_min_train("naive", 7) >= 1
    assert T.model_min_train("holt_winters", 7) > T.model_min_train("naive", 7)


def test_mase_scale_and_score_forecast():
    train = np.arange(10.0)
    assert T.mase_scale(train, 1) == 1.0
    assert T.mase_scale(np.ones(10), 1) == 1.0  # never zero.
    actual = np.array([10.0, 11.0])
    perfect = T.score_forecast(train, actual, actual, np.ones(2), 1)
    assert perfect["mase"] == 0.0 and perfect["smape"] == 0.0
    off = T.score_forecast(train, actual, actual + 2, np.ones(2), 1)
    assert off["mase"] == pytest.approx(2.0)
    assert 0 < off["smape"] < 1 and off["wql"] > 0
    assert T.score_forecast(train, np.zeros(2), np.zeros(2), np.ones(2), 1)["smape"] == 0.0


# --- ForecastModel ---------------------------------------------------------

def _meta(item=""):
    y = _seasonal(60)
    ts = [T.iso(T.step_time(datetime(2024, 1, 1), "D", i)) for i in range(60)]
    series = [{"id": "a", "t": ts, "y": list(y)}]
    if item:
        series.append({"id": "b", "t": ts, "y": list(y + 50)})
    return {
        "config": {"timestamp": "ds", "target": "v", "item_id": item, "frequency": "D", "season_length": 7, "horizon": 7},
        "chosen_model": "seasonal_naive", "interval_scale": 1.5, "primary_metric": "mase", "primary_value": 0.8,
        "baselines": {"naive": 1.2}, "series": series,
    }


def test_forecast_model_single_series():
    fm = T.ForecastModel(_meta())
    out = fm.forecast(horizon=5)
    assert out["model"] == "seasonal_naive" and out["item_id"] == "a" or out["item_id"] == ""
    pts = out["forecast"]
    assert len(pts) == 5
    assert pts[0]["timestamp"] == "2024-03-01T00:00:00"
    assert all(p["lower"] <= p["value"] <= p["upper"] for p in pts)
    assert out["metric"] == "mase" and out["baselines"] == {"naive": 1.2}


def test_forecast_model_interval_scale_widens_bands():
    narrow_meta, wide_meta = _meta(), _meta()
    noise = np.random.default_rng(0).normal(0, 2, 60)
    for m in (narrow_meta, wide_meta):
        m["series"][0]["y"] = list(np.array(m["series"][0]["y"]) + noise)
    narrow_meta["interval_scale"], wide_meta["interval_scale"] = 1.0, 3.0
    n = T.ForecastModel(narrow_meta).forecast(horizon=3)["forecast"]
    w = T.ForecastModel(wide_meta).forecast(horizon=3)["forecast"]
    assert (w[0]["upper"] - w[0]["lower"]) > (n[0]["upper"] - n[0]["lower"])


def test_forecast_model_multi_series_needs_item_id():
    fm = T.ForecastModel(_meta(item="sku"))
    assert fm.series_ids() == ["a", "b"]
    with pytest.raises(ValueError, match="item_id"):
        fm.forecast(horizon=3)
    with pytest.raises(ValueError, match="unknown item_id"):
        fm.forecast(horizon=3, item_id="zzz")
    a = fm.forecast(horizon=3, item_id="a")["forecast"][0]["value"]
    b = fm.forecast(horizon=3, item_id="b")["forecast"][0]["value"]
    assert b - a == pytest.approx(50, abs=1e-3)


def test_forecast_model_horizon_bounds():
    fm = T.ForecastModel(_meta())
    for h in (0, 1001):
        with pytest.raises(ValueError, match="horizon"):
            fm.forecast(horizon=h)


def test_forecast_model_caller_history_overrides_and_extends():
    fm = T.ForecastModel(_meta())
    base = fm.forecast(horizon=1)["forecast"][0]
    extra = [{"ds": "2024-03-01", "v": 1000.0}, {"ds": "2024-03-02", "v": 1000.0}]
    out = fm.forecast(history=extra, horizon=1)["forecast"][0]
    assert out["timestamp"] == "2024-03-03T00:00:00"  # continues after the caller's newest point.
    assert base["timestamp"] == "2024-03-01T00:00:00"


def test_forecast_model_falls_back_when_chosen_model_fails():
    meta = _meta()
    meta["chosen_model"] = "gbm"
    meta["series"][0]["t"] = meta["series"][0]["t"][:6]  # far too short for the lag model.
    meta["series"][0]["y"] = meta["series"][0]["y"][:6]
    out = T.ForecastModel(meta).forecast(horizon=2)
    assert out["model"] == "seasonal_naive"
