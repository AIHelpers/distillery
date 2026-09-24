"""Per-kind training task modules.

Each model kind maps to one module here (`sft.py`, `classifier.py`, ...).
`trainer/run.py` is a thin dispatcher that reads `--kind` (or the `kind`
field in config.json) and delegates to the matching module.
"""

from __future__ import annotations

from typing import Callable, Optional

# Each task module exposes `run(cfg, writer, job_dir, dataset_path) -> int`.
TaskRunner = Callable[[dict, object, str, str], int]  # config dict, writer, dir, dataset.

_REGISTRY: dict[str, TaskRunner] = {}


def register(kind: str, runner: TaskRunner) -> None:
    _REGISTRY[kind] = runner


def get(kind: str) -> Optional[TaskRunner]:
    return _REGISTRY.get(kind)


def kinds() -> list[str]:
    return sorted(_REGISTRY)