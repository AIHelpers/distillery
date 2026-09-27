"""Unit tests for the DPO/ORPO preference-tuning trainer's pure functions.

These cover dataset loading, the length-bias diagnostic, the regression-gate
arithmetic, and config parsing — everything that doesn't need torch/trl —
mirroring how test_ner.py / test_progress.py test their modules' pure logic.
"""

from __future__ import annotations

import json

from trainer.tasks.dpo import (
    Config,
    compute_length_stats,
    evaluate_regression,
    read_preference_records,
)


def test_config_from_dict_defaults() -> None:
    cfg = Config.from_dict(
        {
            "job_id": "job-1",
            "base_model": "Qwen/Qwen3-0.6B",
            "dataset_path": "/tmp/ds.jsonl",
            "output_dir": "/tmp/out",
        }
    )
    assert cfg.method == "dpo"
    assert cfg.beta == 0.1
    assert cfg.learning_rate == 5e-6
    assert cfg.epochs == 2
    assert cfg.parent_adapter_dir is None


def test_config_from_dict_overrides() -> None:
    cfg = Config.from_dict(
        {
            "job_id": "job-1",
            "base_model": "Qwen/Qwen3-0.6B",
            "dataset_path": "/tmp/ds.jsonl",
            "output_dir": "/tmp/out",
            "method": "orpo",
            "beta": 0.2,
            "parent_job_id": "job_sft_1",
            "parent_adapter_dir": "/data/jobs/job_sft_1/adapter",
        }
    )
    assert cfg.method == "orpo"
    assert cfg.beta == 0.2
    assert cfg.parent_job_id == "job_sft_1"
    assert cfg.parent_adapter_dir == "/data/jobs/job_sft_1/adapter"


def test_read_preference_records_skips_invalid_lines(tmp_path) -> None:
    path = tmp_path / "dataset.jsonl"
    lines = [
        json.dumps({"prompt": "p1", "chosen": "good", "rejected": "bad"}),
        json.dumps({"prompt": "p2", "chosen": "same", "rejected": "same"}),  # chosen == rejected.
        json.dumps({"prompt": "p3", "chosen": "", "rejected": "bad"}),  # empty chosen.
        "not json",
        json.dumps({"prompt": "p4", "chosen": "c4", "rejected": "r4"}),
    ]
    path.write_text("\n".join(lines), encoding="utf-8")

    records = read_preference_records(path)

    assert len(records) == 2
    assert records[0]["prompt"] == "p1"
    assert records[1]["prompt"] == "p4"


def test_compute_length_stats_empty() -> None:
    stats = compute_length_stats([])
    assert stats == {"avg_chosen_len": 0.0, "avg_rejected_len": 0.0, "chosen_longer_pct": 0.0}


def test_compute_length_stats_length_bias() -> None:
    records = [
        {"prompt": "p", "chosen": "a" * 100, "rejected": "b" * 10},
        {"prompt": "p", "chosen": "a" * 100, "rejected": "b" * 10},
        {"prompt": "p", "chosen": "a" * 5, "rejected": "b" * 50},
    ]

    stats = compute_length_stats(records)

    assert stats["avg_chosen_len"] == round((100 + 100 + 5) / 3, 2)
    assert stats["avg_rejected_len"] == round((10 + 10 + 50) / 3, 2)
    # chosen is longer in 2 of 3 pairs.
    assert stats["chosen_longer_pct"] == round(2 / 3, 4)


def test_evaluate_regression_passes_within_tolerance() -> None:
    result = evaluate_regression(candidate_loss=1.05, base_loss=1.0, tolerance=0.10)

    assert result["regression_checked"] is True
    assert result["regression_metric"] == "eval_loss"
    assert result["regression_base"] == 1.0
    assert result["regression_value"] == 1.05
    assert result["regression_passed"] is True  # +5% is within the 10% tolerance.


def test_evaluate_regression_fails_beyond_tolerance() -> None:
    result = evaluate_regression(candidate_loss=1.5, base_loss=1.0, tolerance=0.10)

    assert result["regression_passed"] is False
    assert result["regression_delta"] == 0.5


def test_evaluate_regression_improvement_always_passes() -> None:
    # A lower eval_loss than the parent's is never a regression, whatever
    # the tolerance.
    result = evaluate_regression(candidate_loss=0.8, base_loss=1.0, tolerance=0.0)

    assert result["regression_passed"] is True
