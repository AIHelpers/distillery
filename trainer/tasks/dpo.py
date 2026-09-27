"""Preference-tuning (DPO / ORPO) training task module.

Teaches an existing causal-LM which of two answers is better from
`{"prompt", "chosen", "rejected"}` triples (the Feedback Loop and/or
side-by-side review UI). The Go contract is identical to every other kind:
JSONL progress in job-dir/progress.jsonl, final metrics.json, adapter output.

DPO requires a parent SFT job: its adapter is loaded as the starting policy,
and — because we train with LoRA — the *same* base model with the adapter
disabled serves as the (otherwise unmodified) reference model, so VRAM stays
close to a plain SFT run. ORPO is reference-free and is the only method that
can start straight from a base model with no parent job.
"""

from __future__ import annotations

import json
import signal
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from trainer.progress import ProgressWriter

DEFAULT_TARGET_MODS = ["q_proj", "k_proj", "v_proj", "o_proj"]

# Register with the task registry at import time (idempotent).
from trainer.tasks import register  # noqa: E402  (import after module).


@dataclass
class Config:
    job_id: str
    base_model: str
    dataset_path: str
    output_dir: str
    language: str = "python"
    skill: str = "code_generation"
    # "dpo" (needs parent_adapter_dir) or "orpo" (reference-free, parent optional).
    method: str = "dpo"
    beta: float = 0.1
    learning_rate: float = 5e-6
    epochs: int = 2
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    warmup_steps: int = 10
    max_prompt_len: int = 512
    max_len: int = 1024
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_mods: list[str] = field(default_factory=lambda: list(DEFAULT_TARGET_MODS))
    load_4bit: bool = True
    grad_checkpoint: bool = False
    validation_split: float = 0.1
    validate_every_steps: int = 20
    model_cache_dir: Optional[str] = None
    seed: int = 42
    # parent_job_id / parent_adapter_dir: the SFT job this run continues
    # from. Required for method="dpo"; unused for "orpo".
    parent_job_id: Optional[str] = None
    parent_adapter_dir: Optional[str] = None
    # Regression gate: how much the re-evaluated SFT eval_loss is allowed to
    # rise (relative) before the run is flagged as a regression.
    regression_tolerance: float = 0.10

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Config":
        known = {f.name: f for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        kwargs: dict[str, Any] = {}
        for k, v in d.items():
            if k in known:
                kwargs[k] = v
        return cls(**kwargs)


def load_config(path: Path) -> Config:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    cfg = Config.from_dict(raw)
    cfg.dataset_path = str(Path(raw.get("dataset_path", cfg.dataset_path)))
    cfg.output_dir = str(Path(raw.get("output_dir", cfg.output_dir)))
    return cfg


def read_preference_records(path: Path) -> list[dict[str, str]]:
    """Reads the curated JSONL into a list of {"prompt","chosen","rejected"}
    dicts. Malformed or incomplete lines are skipped (the Go layer already
    validates on ingest; this is a defensive second pass)."""
    records: list[dict[str, str]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except ValueError:
                continue
            prompt, chosen, rejected = obj.get("prompt"), obj.get("chosen"), obj.get("rejected")
            if prompt and chosen and rejected and chosen != rejected:
                records.append({"prompt": prompt, "chosen": chosen, "rejected": rejected})
    return records


def load_preference_dataset(path: Path):
    """Loads the curated JSONL into an HF Dataset with prompt/chosen/rejected
    columns — the shape TRL's DPOTrainer/ORPOTrainer expect natively."""
    from datasets import Dataset

    records = read_preference_records(path)
    if not records:
        raise ValueError("Preference dataset is empty or has no valid prompt/chosen/rejected records")

    return Dataset.from_list(records)


def compute_length_stats(records: list[dict[str, str]]) -> dict[str, float]:
    """Plan risk #4 ("longer answers win"): report average chosen/rejected
    completion length and the fraction of pairs where chosen is the longer
    one, so a length-bias warning can be raised in the UI. Pure function —
    no model/tokenizer needed, so it's exercised directly by unit tests."""
    if not records:
        return {"avg_chosen_len": 0.0, "avg_rejected_len": 0.0, "chosen_longer_pct": 0.0}

    chosen_lens = [len(r["chosen"]) for r in records]
    rejected_lens = [len(r["rejected"]) for r in records]
    longer = sum(1 for c, r in zip(chosen_lens, rejected_lens) if c > r)

    n = len(records)
    return {
        "avg_chosen_len": round(sum(chosen_lens) / n, 2),
        "avg_rejected_len": round(sum(rejected_lens) / n, 2),
        "chosen_longer_pct": round(longer / n, 4),
    }


def evaluate_regression(candidate_loss: float, base_loss: float, tolerance: float) -> dict[str, Any]:
    """Compares the DPO/ORPO-tuned model's re-evaluated SFT eval_loss
    (`candidate_loss`) against the parent job's own recorded eval_loss
    (`base_loss`). Pure arithmetic — the (heavy, model-dependent) job of
    actually computing `candidate_loss` lives in `recompute_parent_eval_loss`
    below, kept separate so this comparison is unit-testable on its own."""
    delta = candidate_loss - base_loss
    # eval_loss: lower is better, so a regression is an *increase* beyond
    # the allowed relative tolerance (a tiny absolute floor avoids a
    # near-zero base_loss making the tolerance meaningless).
    allowed = max(abs(base_loss) * tolerance, 1e-6)
    passed = delta <= allowed

    return {
        "regression_checked": True,
        "regression_metric": "eval_loss",
        "regression_base": round(base_loss, 6),
        "regression_value": round(candidate_loss, 6),
        "regression_delta": round(delta, 6),
        "regression_passed": passed,
    }


def recompute_parent_eval_loss(model, tokenizer, parent_job_dir: Path, device: str) -> Optional[float]:
    """Re-runs the parent SFT job's own held-out eval split through the
    (now DPO/ORPO-tuned) candidate model and returns its average
    cross-entropy loss, so evaluate_regression can compare apples to apples.

    Rebuilds the exact split the parent used (same seed/validation_split,
    same Alpaca/chat formatting via trainer.tasks.sft) from the parent
    job's own dataset.jsonl + config.json. Returns None when the parent
    artifacts aren't available (e.g. the parent job dir was GC'd), in which
    case the regression gate is simply skipped rather than failing the run.
    """
    import torch

    from trainer.tasks.sft import build_tokenize_fn, load_dataset

    parent_dataset = parent_job_dir / "dataset.jsonl"
    parent_config = parent_job_dir / "config.json"
    if not parent_dataset.exists() or not parent_config.exists():
        return None

    with open(parent_config, "r", encoding="utf-8") as f:
        parent_cfg = json.load(f)

    validation_split = float(parent_cfg.get("validation_split", 0.1) or 0.1)
    seed = int(parent_cfg.get("seed", 42) or 42)
    max_seq_len = int(parent_cfg.get("max_sequence_length", 1024) or 1024)

    if validation_split <= 0:
        return None

    dataset = load_dataset(parent_dataset)
    tokenized = dataset.map(build_tokenize_fn(tokenizer, max_seq_len), batched=True)
    split = tokenized.train_test_split(test_size=validation_split, seed=seed)
    eval_ds = split["test"]

    if len(eval_ds) == 0:
        return None

    model.eval()
    total_loss, batches = 0.0, 0

    with torch.no_grad():
        for i in range(0, len(eval_ds), 8):
            batch = eval_ds[i : i + 8]
            input_ids = torch.tensor(batch["input_ids"]).to(device)
            attention_mask = torch.tensor(batch["attention_mask"]).to(device)
            out = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
            total_loss += float(out.loss.item())
            batches += 1

    if batches == 0:
        return None

    return total_loss / batches


class DistilleryProgressCallback:
    """Emits one JSON progress line per on_log event (mirrors sft.py's)."""

    def __init__(self, writer: ProgressWriter, total_steps: int):
        self.writer = writer
        self.total_steps = total_steps

    def on_log(self, args, state, control, logs=None):
        logs = logs or {}
        self.writer.progress(
            step=state.global_step,
            total_steps=self.total_steps,
            epoch=state.epoch,
            loss=logs.get("loss"),
            eval_loss=logs.get("eval_loss"),
            lr=logs.get("learning_rate"),
        )


def save_metrics(output_dir: Path, metrics: dict[str, Any]):
    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"metrics written to {metrics_path}", file=sys.stderr)


def _build_trainer(trainer_cls, **kwargs):
    """TRL renamed DPOTrainer/ORPOTrainer's `tokenizer=` kwarg to
    `processing_class=` in newer releases; try the modern name first and
    fall back so this module works across the trl>=0.8 range we depend on."""
    tokenizer = kwargs.pop("tokenizer", None)
    try:
        return trainer_cls(processing_class=tokenizer, **kwargs)
    except TypeError:
        return trainer_cls(tokenizer=tokenizer, **kwargs)


def run(cfg: Config, writer: ProgressWriter, job_dir: Path, dataset_path: str) -> int:
    """Runs the DPO/ORPO preference-tuning job and returns the exit code."""
    stop_requested = False

    def handle_sigterm(signum, frame):  # noqa: ARG001
        nonlocal stop_requested
        stop_requested = True
        writer.event("signal", signal="SIGTERM", msg="graceful stop requested")

    signal.signal(signal.SIGTERM, handle_sigterm)

    method = (cfg.method or "dpo").strip().lower()
    if method not in ("dpo", "orpo"):
        writer.event("error", message=f"unknown preference method: {cfg.method}")
        save_metrics(job_dir, {"status": "failed", "error": f"unknown preference method: {cfg.method}"})
        return 1

    if method == "dpo" and not cfg.parent_adapter_dir:
        msg = "dpo requires parent_adapter_dir (a completed SFT job); use method=orpo to tune from a base model directly"
        writer.event("error", message=msg)
        save_metrics(job_dir, {"status": "failed", "error": msg})
        return 1

    # --- Import heavy libs lazily so --help / config errors are cheap. ---
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    try:
        from trl import DPOConfig, DPOTrainer, ORPOConfig, ORPOTrainer
    except ImportError as exc:  # pragma: no cover - depends on installed trl.
        msg = f"trl is required for preference tuning: {exc}"
        writer.event("error", message=msg)
        save_metrics(job_dir, {"status": "failed", "error": msg})
        return 1

    device_map = "auto"
    if torch.cuda.is_available():
        writer.event("device", device_type="cuda", name=torch.cuda.get_device_name(0))
    else:
        writer.event("device", device_type="cpu")
        device_map = "cpu"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        tokenizer = AutoTokenizer.from_pretrained(cfg.base_model, cache_dir=cfg.model_cache_dir, use_fast=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model_kwargs: dict[str, Any] = {
            "cache_dir": cfg.model_cache_dir,
            "device_map": device_map,
            "trust_remote_code": True,
        }

        if cfg.load_4bit and torch.cuda.is_available():
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )

        base = AutoModelForCausalLM.from_pretrained(cfg.base_model, **model_kwargs)

        if cfg.load_4bit and torch.cuda.is_available():
            base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=cfg.grad_checkpoint)

        if cfg.parent_adapter_dir:
            # DPO: continue from the parent SFT adapter. With ref_model=None
            # below, TRL uses this same PeftModel with the adapter disabled
            # as the reference — no second copy of the base model needed.
            writer.event("status", value="loading_parent_adapter", path=cfg.parent_adapter_dir)
            model = PeftModel.from_pretrained(base, cfg.parent_adapter_dir, is_trainable=True)
        else:
            # ORPO with no parent: fresh LoRA on the base model, same as SFT.
            lora_config = LoraConfig(
                r=cfg.lora_rank,
                lora_alpha=cfg.lora_alpha,
                lora_dropout=cfg.lora_dropout,
                target_modules=cfg.lora_target_mods or DEFAULT_TARGET_MODS,
                task_type="CAUSAL_LM",
            )
            model = get_peft_model(base, lora_config)

        model.print_trainable_parameters()

        dataset = load_preference_dataset(Path(dataset_path))
        raw_records = read_preference_records(Path(dataset_path))
        length_stats = compute_length_stats(raw_records)

        if cfg.validation_split > 0 and len(dataset) > 1:
            split = dataset.train_test_split(test_size=cfg.validation_split, seed=cfg.seed)
            train_ds, eval_ds = split["train"], split["test"]
        else:
            train_ds, eval_ds = dataset, None

        output_dir = Path(cfg.output_dir)
        checkpoint_dir = output_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        common_args = dict(
            output_dir=str(checkpoint_dir),
            num_train_epochs=cfg.epochs,
            per_device_train_batch_size=cfg.batch_size,
            per_device_eval_batch_size=cfg.batch_size,
            gradient_accumulation_steps=cfg.gradient_accumulation_steps,
            learning_rate=cfg.learning_rate,
            warmup_steps=cfg.warmup_steps,
            beta=cfg.beta,
            max_prompt_length=cfg.max_prompt_len,
            max_length=cfg.max_len,
            logging_steps=1,
            eval_strategy="steps" if eval_ds is not None else "no",
            eval_steps=cfg.validate_every_steps,
            save_strategy="steps",
            save_steps=cfg.validate_every_steps,
            save_total_limit=2,
            report_to=[],
            seed=cfg.seed,
            fp16=torch.cuda.is_available(),
            gradient_checkpointing=cfg.grad_checkpoint,
        )

        total_steps = max(1, (len(train_ds) // (cfg.batch_size * cfg.gradient_accumulation_steps))) * cfg.epochs
        callback = DistilleryProgressCallback(writer, total_steps)

        writer.event("status", value="starting_training", total_steps=total_steps, method=method)

        if method == "dpo":
            training_args = DPOConfig(**common_args)
            trainer = _build_trainer(
                DPOTrainer,
                model=model,
                ref_model=None,  # LoRA: same base with the adapter disabled.
                args=training_args,
                train_dataset=train_ds,
                eval_dataset=eval_ds,
                tokenizer=tokenizer,
                callbacks=[callback],
            )
        else:
            training_args = ORPOConfig(**common_args)
            trainer = _build_trainer(
                ORPOTrainer,
                model=model,
                args=training_args,
                train_dataset=train_ds,
                eval_dataset=eval_ds,
                tokenizer=tokenizer,
                callbacks=[callback],
            )

        trainer.train()

        adapter_path = output_dir / "adapter"
        model.save_pretrained(str(adapter_path))
        tokenizer.save_pretrained(str(output_dir / "tokenizer"))

        final_metrics: dict[str, Any] = {
            "status": "completed",
            "kind": "preference_lm",
            "method": method,
            **length_stats,
        }

        if eval_ds is not None:
            eval_result = trainer.evaluate()
            final_metrics["eval_loss"] = eval_result.get("eval_loss", 0.0)
            final_metrics["reward_accuracy"] = eval_result.get(
                "eval_rewards/accuracies", eval_result.get("eval_rewards/accuracy", 0.0)
            )
            final_metrics["reward_margin"] = eval_result.get("eval_rewards/margins", 0.0)

        final_metrics["epoch"] = float(cfg.epochs)
        final_metrics["train_examples"] = len(train_ds)

        # Regression gate (plan section 9 + "Definition of done"): re-run the
        # parent SFT job's own held-out eval split through this tuned model
        # and compare eval_loss. Only meaningful for DPO (which has a
        # parent); skipped (not failed) when unavailable.
        if cfg.parent_adapter_dir:
            try:
                parent_job_dir = Path(cfg.parent_adapter_dir).parent
                base_loss = None
                parent_metrics_path = parent_job_dir / "metrics.json"
                if parent_metrics_path.exists():
                    with open(parent_metrics_path, "r", encoding="utf-8") as f:
                        base_loss = json.load(f).get("eval_loss")

                if base_loss is not None:
                    candidate_loss = recompute_parent_eval_loss(model, tokenizer, parent_job_dir, device)
                    if candidate_loss is not None:
                        regression = evaluate_regression(candidate_loss, float(base_loss), cfg.regression_tolerance)
                        final_metrics.update(regression)
                        writer.event(
                            "regression_check",
                            passed=regression["regression_passed"],
                            base=regression["regression_base"],
                            value=regression["regression_value"],
                        )
            except Exception as exc:  # noqa: BLE001 - regression check must never fail the run.
                writer.event("regression_check", status="error", reason=str(exc))

        writer.event("complete", **final_metrics)
        save_metrics(job_dir, final_metrics)

        if stop_requested:
            writer.event("status", value="stopped_by_user")
            return 130

        return 0

    except Exception as exc:  # noqa: BLE001
        writer.event("error", message=str(exc), class_name=type(exc).__name__)
        save_metrics(job_dir, {"status": "failed", "error": str(exc)})
        print(f"FATAL: {exc}", file=sys.stderr)
        return 1
    finally:
        writer.close()


def _runner(cfg_dict: dict, writer: object, job_dir: str, dataset_path: str) -> int:
    c = cfg_dict if isinstance(cfg_dict, Config) else Config.from_dict(cfg_dict)
    c.dataset_path = dataset_path or c.dataset_path

    return run(c, writer, Path(job_dir), c.dataset_path)


register("preference_lm", _runner)
