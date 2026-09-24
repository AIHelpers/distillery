"""Unit tests for the pure (framework-free) helpers in trainer/tasks/classifier.py."""

from pathlib import Path

import pytest

from trainer.tasks.classifier import (
    accuracy,
    build_label_map,
    compute_class_weights,
    confusion_matrix,
    load_dataset,
    macro_f1,
    per_class_metrics,
    threshold_sweep,
    weighted_f1,
)


def test_build_label_map_sorted():
    mapping, ordered = build_label_map(["billing", "other", "billing", "technical"])
    assert ordered == ["billing", "other", "technical"]
    assert mapping == {"billing": 0, "other": 1, "technical": 2}


def test_per_class_metrics_and_macro_f1():
    label_map = {"a": 0, "b": 1}
    truths = ["a", "a", "a", "b"]
    preds = ["a", "a", "b", "b"]
    per = per_class_metrics(truths, preds, label_map)
    assert per["a"]["precision"] == 1.0  # two a-preds, both a.
    assert per["a"]["support"] == 3
    assert per["b"]["recall"] == 1.0  # one b-truth, one b-pred.
    assert macro_f1(per) > 0.0


def test_accuracy_and_weighted_f1():
    truths = ["a", "a", "b"]
    preds = ["a", "b", "b"]
    # accuracy() rounds to 4 decimals by design; compare with tolerance.
    assert accuracy(truths, preds) == pytest.approx(2 / 3, abs=1e-4)
    w = weighted_f1(truths, preds)
    assert 0.0 < w <= 1.0


def test_confusion_matrix_shape():
    label_map = {"a": 0, "b": 1}
    cm = confusion_matrix(["a", "b"], ["a", "a"], label_map)
    assert len(cm) == 2
    assert len(cm[0]) == 2
    assert cm[0][0] == 1  # a->a.
    assert cm[1][0] == 1  # b misclassified as a.


def test_compute_class_weights_imbalanced():
    # 90 / 10 -> ratio 9:1 > 3 -> weights returned.
    label_map = {"a": 0, "b": 1}
    labels = ["a"] * 90 + ["b"] * 10
    weights = compute_class_weights(labels, label_map)
    assert weights is not None
    assert weights[0] < weights[1]  # minority class gets higher weight.


def test_compute_class_weights_balanced_returns_none():
    label_map = {"a": 0, "b": 1}
    labels = ["a"] * 10 + ["b"] * 10
    assert compute_class_weights(labels, label_map) is None


def test_threshold_sweep_coverage():
    label_map = {"a": 0, "b": 1}
    truths = ["a", "b"]
    scores = [{"a": 0.98, "b": 0.02}, {"a": 0.4, "b": 0.6}]
    sweep = threshold_sweep(truths, scores, label_map)
    assert sweep, "expected non-empty sweep"
    assert sweep[0]["threshold"] == 0.05
    # At very low threshold, both examples are covered.
    assert sweep[0]["coverage"] == 1.0


def test_load_dataset_jsonl(tmp_path: Path):
    ds = tmp_path / "dataset.jsonl"
    ds.write_text(
        '{"text": "my invoice", "label": "billing"}\n'
        '{"text": "server down", "label": "technical"}\n',
        encoding="utf-8",
    )
    texts, labels = load_dataset(ds)
    assert texts == ["my invoice", "server down"]
    assert labels == ["billing", "technical"]


def test_load_dataset_legacy_format(tmp_path: Path):
    # Legacy causal_lm {instruction, output} records are accepted as a fallback.
    ds = tmp_path / "dataset.jsonl"
    ds.write_text('{"instruction": "x", "output": "a"}\n', encoding="utf-8")
    texts, labels = load_dataset(ds)
    assert texts == ["x"]
    assert labels == ["a"]


def test_from_dict_drops_zero_hyperparams():
    from trainer.tasks.classifier import ClassifierConfig

    cfg = ClassifierConfig.from_dict(
        {"max_length": 0, "epochs": 0, "learning_rate": 0, "batch_size": 32, "threshold": 0.0}
    )
    assert cfg.max_length == 512
    assert cfg.epochs == 4
    assert cfg.learning_rate == 3e-5
    assert cfg.batch_size == 32
    assert cfg.threshold == 0.5


def test_class_weights_inverse_frequency_values():
    # w_i = N / (n_i * C): 10/(9*2) and 10/(1*2).
    label_map = {"a": 0, "b": 1}
    weights = compute_class_weights(["a"] * 9 + ["b"] * 1, label_map)
    assert weights is not None
    assert abs(weights[0] - 10 / 18) < 1e-9
    assert abs(weights[1] - 5.0) < 1e-9


def test_threshold_sweep_never_credits_abstentions():
    label_map = {"a": 0, "b": 1}
    truths = ["a", "a", "b"]
    scores = [{"a": 0.9, "b": 0.1}, {"a": 0.3, "b": 0.7}, {"b": 0.8, "a": 0.2}]
    sweep = threshold_sweep(truths, scores, label_map)
    at_80 = [p for p in sweep if p["threshold"] == 0.8][0]
    assert at_80["coverage"] == pytest.approx(2 / 3, abs=1e-4)
    assert at_80["macro_f1"] == 1.0  # only the two confident, correct preds count
    at_90 = [p for p in sweep if p["threshold"] == 0.9][0]
    assert at_90["coverage"] == pytest.approx(1 / 3, abs=1e-4)


def test_registry_registers_all_kinds():
    from trainer.tasks import kinds

    assert "causal_lm" in kinds()
    assert "seq_classifier" in kinds()