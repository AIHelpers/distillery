"""Unit tests for the vision-language trainer's pure functions.

These cover dataset loading, image-key resolution, field-level metrics
(exact match / F1 / normalization), ANLS, JSON validity, and document
accuracy -- everything that doesn't need torch/transformers/PIL -- mirroring
how test_ner.py / test_dpo.py test their modules' pure logic.
"""

from __future__ import annotations

import json

from trainer.tasks.vision import (
    Config,
    anls_score,
    average_anls,
    document_accuracy,
    evaluate_predictions,
    field_level_metrics,
    json_validity_rate,
    macro_field_f1,
    normalize_field_value,
    read_vision_records,
    resolve_image_path,
    try_parse_json,
)


def test_config_from_dict_defaults() -> None:
    cfg = Config.from_dict(
        {
            "job_id": "job-1",
            "base_model": "Qwen/Qwen2-VL-2B-Instruct",
            "dataset_path": "/tmp/ds.jsonl",
            "output_dir": "/tmp/out",
        }
    )
    assert cfg.freeze_vision_encoder is True
    assert cfg.max_image_side == 1280
    assert cfg.epochs == 3
    assert cfg.images_dir == ""


def test_config_from_dict_overrides() -> None:
    cfg = Config.from_dict(
        {
            "job_id": "job-1",
            "base_model": "Qwen/Qwen2-VL-7B-Instruct",
            "dataset_path": "/tmp/ds.jsonl",
            "output_dir": "/tmp/out",
            "images_dir": "/tmp/blobs/task_1",
            "freeze_vision_encoder": False,
            "max_image_side": 896,
            "json_schema": '{"type":"object"}',
        }
    )
    assert cfg.images_dir == "/tmp/blobs/task_1"
    assert cfg.freeze_vision_encoder is False
    assert cfg.max_image_side == 896
    assert cfg.json_schema == '{"type":"object"}'


def test_read_vision_records_skips_invalid_lines(tmp_path) -> None:
    path = tmp_path / "dataset.jsonl"
    lines = [
        json.dumps({"image": "img1.png", "prompt": "Extract vendor", "answer": "Acme"}),
        json.dumps({"image": "", "prompt": "no image"}),  # missing image.
        json.dumps({"image": "img2.png", "prompt": ""}),  # missing prompt.
        "not json",
        json.dumps({"image": "img3.png", "prompt": "Extract total", "answer": "42", "doc_id": "d1", "page": 2}),
    ]
    path.write_text("\n".join(lines), encoding="utf-8")

    records = read_vision_records(path)

    assert len(records) == 2
    assert records[0]["image"] == "img1.png"
    assert records[0]["answer"] == "Acme"
    assert records[1]["doc_id"] == "d1"
    assert records[1]["page"] == 2


def test_resolve_image_path_neutralizes_traversal() -> None:
    # A malicious-looking key still resolves to a single path component under
    # images_dir -- traversal segments can't escape it.
    p = resolve_image_path("/data/blobs/task_1", "../../etc/passwd")
    assert p.parent == p.parent  # sanity: still a Path.
    assert p.name == "passwd"
    assert str(p).startswith("/data/blobs/task_1")


def test_normalize_field_value_strips_currency_and_commas() -> None:
    assert normalize_field_value("$4,200.00") == normalize_field_value("4200.00")
    assert normalize_field_value(" Acme Corp ") == "acmecorp"
    assert normalize_field_value(4200) == "4200"


def test_try_parse_json_strips_code_fence() -> None:
    assert try_parse_json('{"a": 1}') == {"a": 1}
    assert try_parse_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert try_parse_json("not json") is None
    assert try_parse_json("[1, 2, 3]") is None  # top-level array is not a field object.


def test_field_level_metrics_exact_and_partial() -> None:
    golds = [
        {"vendor": "Acme", "total": "4200.00"},
        {"vendor": "Acme", "total": "4200.00"},
    ]
    predictions = [
        {"vendor": "Acme", "total": "$4,200.00"},  # exact after normalization.
        {"vendor": "Acmme", "total": "4200.00"},  # near-miss vendor.
    ]

    metrics = field_level_metrics(predictions, golds)

    assert metrics["vendor"]["support"] == 2
    assert metrics["vendor"]["exact_match"] == 0.5
    assert 0.5 < metrics["vendor"]["f1"] < 1.0  # partial credit for the typo.
    assert metrics["total"]["exact_match"] == 1.0
    assert metrics["total"]["f1"] == 1.0


def test_field_level_metrics_missing_prediction_counts_as_miss() -> None:
    golds = [{"vendor": "Acme"}]
    predictions = [None]

    metrics = field_level_metrics(predictions, golds)

    assert metrics["vendor"]["exact_match"] == 0.0
    assert metrics["vendor"]["f1"] == 0.0


def test_macro_field_f1() -> None:
    fm = {"a": {"exact_match": 1.0, "f1": 1.0, "support": 2}, "b": {"exact_match": 0.0, "f1": 0.4, "support": 2}}
    assert macro_field_f1(fm) == 0.7
    assert macro_field_f1({}) == 0.0


def test_document_accuracy() -> None:
    golds = [{"vendor": "Acme", "total": "42"}, {"vendor": "Acme", "total": "42"}]
    predictions = [{"vendor": "Acme", "total": "42"}, {"vendor": "Acme", "total": "43"}]

    assert document_accuracy(predictions, golds) == 0.5
    assert document_accuracy([], []) == 0.0


def test_anls_score_identical_and_clamped() -> None:
    assert anls_score("hello world", "hello world") == 1.0
    # Very dissimilar strings clamp to 0 rather than a small positive score.
    assert anls_score("completely different text here", "x") == 0.0


def test_anls_score_empty_gold() -> None:
    assert anls_score("", "") == 1.0
    assert anls_score("something", "") == 0.0


def test_average_anls() -> None:
    preds = ["hello world", "goodbye"]
    golds = ["hello world", "goodbye"]
    assert average_anls(preds, golds) == 1.0
    assert average_anls([], []) == 0.0


def test_json_validity_rate_parse_only() -> None:
    preds = ['{"a": 1}', "not json", '{"b": 2}']
    assert json_validity_rate(preds, None) == round(2 / 3, 4)
    assert json_validity_rate([], None) == 0.0


def test_json_validity_rate_with_schema() -> None:
    schema = json.dumps({"type": "object", "required": ["vendor"], "properties": {"vendor": {"type": "string"}}})
    preds = [
        json.dumps({"vendor": "Acme"}),  # valid.
        json.dumps({"total": 42}),  # missing required "vendor".
        "not json",
    ]

    assert json_validity_rate(preds, schema) == round(1 / 3, 4)


def test_evaluate_predictions_structured_path() -> None:
    golds = [json.dumps({"vendor": "Acme", "total": "42"}), json.dumps({"vendor": "Acme", "total": "42"})]
    preds = [json.dumps({"vendor": "Acme", "total": "42"}), json.dumps({"vendor": "Acme", "total": "43"})]

    result = evaluate_predictions(preds, golds, None)

    assert "field_metrics" in result
    assert result["document_accuracy"] == 0.5
    assert result["json_valid_rate"] == 1.0
    assert "anls" not in result


def test_evaluate_predictions_freetext_path() -> None:
    golds = ["The invoice total is $42.", "Paid in full."]
    preds = ["The invoice total is $42.", "Not paid."]

    result = evaluate_predictions(preds, golds, None)

    assert "anls" in result
    assert "field_metrics" not in result
    assert result["json_valid_rate"] == 0.0
