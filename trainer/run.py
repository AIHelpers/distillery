#!/usr/bin/env python3
"""Distillery local training worker — dispatcher.

Go orchestrates this script once per training job. It delegates to the
per-kind task module registered in `trainer.tasks` based on `--kind` (or the
`kind` field in config.json). Each module owns its own metrics.json /
progress.jsonl contract under the job dir.

    python -m trainer.run \
        --kind causal_lm \
        --job-dir /data/jobs/{jobID} \
        --config  /data/jobs/{jobID}/config.json \
        --dataset /data/jobs/{jobID}/dataset.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from trainer.progress import ProgressWriter
from trainer.tasks import get, kinds
from trainer.tasks.sft import Config  # noqa: F401  (back-compat re-export)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", default="", help="Model kind (causal_lm, seq_classifier, ...). Default: from config.json.")
    parser.add_argument("--job-dir", required=True, help="Job working directory (created if needed).")
    parser.add_argument("--config", required=True, help="Path to config.json written by Go.")
    parser.add_argument("--dataset", required=True, help="Path to the curated JSONL dataset.")
    args = parser.parse_args()

    job_dir = Path(args.job_dir)
    job_dir.mkdir(parents=True, exist_ok=True)

    cfg_path = Path(args.config)
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg_dict = json.load(f)

    # Resolve kind: --kind wins; fall back to config's kind; default causal_lm.
    kind = (args.kind or "").strip() or str(cfg_dict.get("kind", "")).strip() or "causal_lm"

    runner = get(kind)
    if runner is None:
        print(f"FATAL: unknown model kind {kind!r}. Known kinds: {', '.join(kinds())}", file=sys.stderr)
        print(f"metrics written to {job_dir / 'metrics.json'}", file=sys.stderr)

        with open(job_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump({"status": "failed", "error": f"unknown model kind: {kind}"}, f, indent=2)

        return 2

    log_path = str(job_dir / "progress.jsonl")
    writer = ProgressWriter(log_path=log_path)

    # The runner receives the raw config dict, job dir, and dataset path.
    return runner(cfg_dict, writer, str(job_dir), str(Path(args.dataset)))


if __name__ == "__main__":
    raise SystemExit(main())