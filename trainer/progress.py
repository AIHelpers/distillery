"""Progress streaming utilities for Distillery trainer.

Writes one JSON line per event to stdout (and appends to job-dir/progress.jsonl)
so Go's `local_trainer` adapter can tail it and update the domain TrainingJob.
"""

from __future__ import annotations

import json
import sys
import time
from typing import Any, Optional, TextIO


class ProgressWriter:
    """Writes JSON events as one line per event."""

    def __init__(self, stream: TextIO | None = None, log_path: Optional[str] = None) -> None:
        self.stream = stream or sys.stdout
        self.log_path = log_path
        self._fh: Optional[TextIO] = None

    def _ensure_file(self) -> None:
        if self.log_path and self._fh is None:
            self._fh = open(self.log_path, "a", encoding="utf-8")

    def write(self, event_type: str, **fields: Any) -> None:
        rec = {"type": event_type, "timestamp": time.time()}
        rec.update(fields)

        line = json.dumps(rec, ensure_ascii=False)
        print(line, file=self.stream, flush=True)

        if self.log_path:
            self._ensure_file()
            if self._fh is not None:
                self._fh.write(line + "\n")
                self._fh.flush()

    def progress(
        self,
        step: int,
        total_steps: int,
        epoch: float,
        loss: Optional[float] = None,
        eval_loss: Optional[float] = None,
        eval_f1: Optional[float] = None,
        lr: Optional[float] = None,
    ) -> None:
        self.write(
            "progress",
            step=step,
            total_steps=total_steps,
            epoch=epoch,
            loss=loss,
            eval_loss=eval_loss,
            eval_f1=eval_f1,
            lr=lr,
        )

    def metric(self, name: str, value: float, step: int, epoch: int) -> None:
        self.write("metric", name=name, value=value, step=step, epoch=epoch)

    def event(self, event_type: str, **fields: Any) -> None:
        # A caller-supplied `type` field must not collide with the event kind.
        fields.pop("type", None)
        self.write(event_type, **fields)

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None