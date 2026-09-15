"""Unit tests for the Distillery trainer progress writer and config loading."""

from __future__ import annotations

import io
import json

import pytest

from trainer.progress import ProgressWriter
from trainer.run import Config


def test_progress_writer_emits_json_lines() -> None:
    buf = io.StringIO()
    writer = ProgressWriter(stream=buf)

    writer.progress(step=1, total_steps=100, epoch=0.1, loss=1.5, lr=2e-5)

    lines = buf.getvalue().strip().splitlines()
    assert len(lines) == 1

    rec = json.loads(lines[0])
    assert rec["type"] == "progress"
    assert rec["step"] == 1
    assert rec["total_steps"] == 100
    assert rec["loss"] == 1.5
    assert rec["lr"] == 2e-5


def test_progress_writer_logs_to_file(tmp_path) -> None:
    log = tmp_path / "progress.jsonl"
    buf = io.StringIO()
    writer = ProgressWriter(stream=buf, log_path=str(log))

    writer.event("device", type="cuda")
    writer.close()

    assert log.exists()
    lines = log.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert rec["type"] == "device"


def test_config_from_dict_defaults() -> None:
    cfg = Config.from_dict(
        {
            "job_id": "job-1",
            "base_model": "Qwen2.5-0.5B-Instruct",
            "language": "python",
            "skill": "code_generation",
            "dataset_path": "/tmp/ds.jsonl",
            "output_dir": "/tmp/out",
        }
    )
    assert cfg.epochs == 3
    assert cfg.use_lora is True
    assert cfg.load_4bit is True
    assert cfg.lora_rank == 16
    assert cfg.lora_target_mods == ["q_proj", "k_proj", "v_proj", "o_proj"]


def test_config_from_dict_overrides() -> None:
    cfg = Config.from_dict(
        {
            "job_id": "job-1",
            "base_model": "Qwen2.5-0.5B-Instruct",
            "language": "python",
            "skill": "code_generation",
            "dataset_path": "/tmp/ds.jsonl",
            "output_dir": "/tmp/out",
            "epochs": 5,
            "use_lora": False,
            "lora_rank": 4,
        }
    )
    assert cfg.epochs == 5
    assert cfg.use_lora is False
    assert cfg.lora_rank == 4