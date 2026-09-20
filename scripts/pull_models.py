#!/usr/bin/env python3
"""Pre-download Distillery's base-model catalog into a local HF cache.

Run before offline training so `trainer/run.py` doesn't need the internet
mid-job:

    python scripts/pull_models.py [--model Qwen3-0.6B] [--all]

Requires network access (pip install transformers huggingface_hub).
"""

import argparse
import sys

MODEL_REPOS = {
    "Qwen3-0.6B": "Qwen/Qwen3-0.6B",
    "Qwen3-1.7B": "Qwen/Qwen3-1.7B",
    "Llama-3.2-1B-Instruct": "meta-llama/Llama-3.2-1B-Instruct",
    "Llama-3.2-3B-Instruct": "meta-llama/Llama-3.2-3B-Instruct",
    "Phi-3.5-mini-instruct": "microsoft/Phi-3.5-mini-instruct",
    "Qwen3-4B": "Qwen/Qwen3-4B",
    "Qwen3-8B": "Qwen/Qwen3-8B",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        choices=sorted(MODEL_REPOS),
        help="Which model repo to pre-pull (repeatable).",
        action="append",
    )
    parser.add_argument("--all", action="store_true", help="Pre-pull the whole catalog.")
    parser.add_argument("--cache-dir", default=None, help="HF cache dir (default ~/.cache/huggingface).")
    args = parser.parse_args()

    if args.all:
        selected = list(MODEL_REPOS.values())
    elif args.model:
        selected = [MODEL_REPOS[m] for m in args.model]
    else:
        parser.print_help()
        return 2

    from huggingface_hub import snapshot_download

    failures = 0

    for repo in selected:
        print(f"Pulling {repo} ...", file=sys.stderr)
        try:
            snapshot_download(repo_id=repo, cache_dir=args.cache_dir)
        except Exception as exc:  # noqa: BLE001 - surface everything to the user
            print(f"  FAILED: {exc}", file=sys.stderr)
            failures += 1

    print(f"Done. {len(selected) - failures}/{len(selected)} pulled.", file=sys.stderr)

    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())