"""Causal-LM (SFT) training task module.

This is the original `trainer/run.py` implementation, moved here unchanged so
the dispatcher (run.py) can route per kind. The Go contract is identical:
JSONL progress in job-dir/progress.jsonl, final metrics.json, adapter output.
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
    language: str
    skill: str
    dataset_path: str
    output_dir: str
    epochs: int = 3
    batch_size: int = 8
    learning_rate: float = 2e-5
    warmup_steps: int = 100
    max_sequence_length: int = 1024
    gradient_accumulation_steps: int = 1
    weight_decay: float = 0.0
    scheduler_type: str = "linear"
    optimizer_type: str = "adamw"
    use_lora: bool = True
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_mods: list[str] = field(default_factory=lambda: list(DEFAULT_TARGET_MODS))
    load_4bit: bool = True
    grad_checkpoint: bool = False
    validation_split: float = 0.1
    validate_every_steps: int = 50
    metrics: list[str] = field(default_factory=lambda: ["loss", "perplexity"])
    model_cache_dir: Optional[str] = None
    resume_from: Optional[str] = None
    seed: int = 42
    # Track B: JSON schema the task's outputs must satisfy. When set, the SFT
    # eval reports a "json validity rate" (fraction of held-out outputs that
    # parse as JSON and satisfy the schema).
    json_schema: Optional[str] = None
    max_validation_samples: int = 50
    max_new_tokens: int = 256

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


def load_dataset(path: Path):
    """Load the curated JSONL into a HF Dataset. Expects one JSON object per line.

    Each record may be:
      - {"instruction": ..., "input": ..., "output": ...}   (Alpaca format)
      - {"messages": [{"role": ..., "content": ...}, ...]}  (chat format)
    """
    from datasets import Dataset

    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)

            if "messages" in obj:
                # Pre-tokenize chat format into a single "text" field.
                text = "\n".join(
                    f"{msg.get('role', 'user')}: {msg.get('content', '')}"
                    for msg in obj["messages"]
                )
                records.append({"text": text})
            else:
                instruction = obj.get("instruction", "")
                inp = obj.get("input", "")
                out = obj.get("output", "")
                if instruction or inp or out:
                    text = f"### Instruction\n{instruction}\n### Input\n{inp}\n### Output\n{out}"
                    records.append({"text": text})

    if not records:
        raise ValueError("Dataset is empty or has no readable records")

    return Dataset.from_list(records)


def build_tokenize_fn(tokenizer, max_len: int):
    def tokenize(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_len,
            padding="max_length",
        )

    return tokenize


class DistilleryProgressCallback:
    """Emits one JSON progress line per on_log / on_step_end event."""

    def __init__(self, writer: ProgressWriter, total_steps: int):
        self.writer = writer
        self.total_steps = total_steps
        self.step = 0
        self.epoch = 0.0

    def on_log(self, args, state, control, logs=None):
        self.step = state.global_step
        self.epoch = state.epoch
        loss = logs.get("loss") if logs else None
        eval_loss = logs.get("eval_loss") if logs else None
        lr = logs.get("learning_rate") if logs else None
        self.writer.progress(
            step=self.step,
            total_steps=self.total_steps,
            epoch=self.epoch,
            loss=loss,
            eval_loss=eval_loss,
            lr=lr,
        )

    def on_step_end(self, args, state, control):
        if state.global_step % 10 == 0:
            self.writer.progress(
                step=state.global_step,
                total_steps=self.total_steps,
                epoch=state.epoch,
                loss=state.log_history[-1].get("loss") if state.log_history else None,
            )


def _read_raw_records(path: Path) -> list[dict[str, Any]]:
    """Read the raw (un-tokenized) JSONL records for output validation."""
    records: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except ValueError:
                continue
    return records


def _load_json_schema_validator(schema_str: str):
    """Build a validator for the schema string, preferring the `jsonschema`
    package (full draft-07) and falling back to a JSON-parse check.

    Returns a callable (output_text) -> bool.
    """
    try:
        import jsonschema  # type: ignore

        schema = json.loads(schema_str)
        validator = jsonschema.Draft7Validator(schema)

        def _validate(output: str) -> bool:
            try:
                doc = json.loads(output)
            except (TypeError, ValueError):
                return False
            return validator.is_valid(doc)

        return _validate
    except Exception:
        def _parse_only(output: str) -> bool:
            try:
                json.loads(output)
                return True
            except (TypeError, ValueError):
                return False

        return _parse_only


def compute_json_validity_rate(
    model,
    tokenizer,
    eval_records: list[dict[str, Any]],
    schema_str: str,
    max_samples: int,
    max_new_tokens: int,
    device: str,
) -> float:
    """Generate on a sample of held-out records and return the fraction whose
    output is schema-valid JSON (Track B's "JSON validity rate" metric)."""
    import torch

    validator = _load_json_schema_validator(schema_str)
    sample = eval_records[:max_samples]
    if not sample:
        return 0.0

    valid = 0
    model.eval()
    with torch.no_grad():
        for rec in sample:
            instruction = rec.get("instruction", "")
            inp = rec.get("input", "")
            prompt = f"### Instruction\n{instruction}\n### Input\n{inp}\n### Output\n"
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
            text = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            if validator(text.strip()):
                valid += 1

    return round(valid / len(sample), 4)


def save_metrics(output_dir: Path, metrics: dict[str, Any]):
    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"metrics written to {metrics_path}", file=sys.stderr)


def run(cfg: Config, writer: ProgressWriter, job_dir: Path, dataset_path: str) -> int:
    """Runs the causal-LM SFT training and returns the process exit code."""
    # --- Graceful SIGTERM handling (Go pause/cancel) ---.
    stop_requested = False

    def handle_sigterm(signum, frame):  # noqa: ARG001
        nonlocal stop_requested
        stop_requested = True
        writer.event("signal", signal="SIGTERM", msg="graceful stop requested")

    signal.signal(signal.SIGTERM, handle_sigterm)

    # --- Import heavy libs lazily so --help / config errors are cheap. ---
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    device_map = "auto"
    if torch.cuda.is_available():
        writer.event("device", device_type="cuda", name=torch.cuda.get_device_name(0))
    else:
        writer.event("device", device_type="cpu")
        device_map = "cpu"

    try:
        # --- Load tokenizer + model. ---
        tokenizer = AutoTokenizer.from_pretrained(
            cfg.base_model,
            cache_dir=cfg.model_cache_dir,
            use_fast=True,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model_kwargs: dict[str, Any] = {
            "cache_dir": cfg.model_cache_dir,
            "device_map": device_map,
            "trust_remote_code": True,
        }

        if cfg.load_4bit and torch.cuda.is_available():
            bnb = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            model_kwargs["quantization_config"] = bnb

        model = AutoModelForCausalLM.from_pretrained(cfg.base_model, **model_kwargs)

        if cfg.use_lora:
            if cfg.load_4bit and torch.cuda.is_available():
                model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=cfg.grad_checkpoint)
            lora_config = LoraConfig(
                r=cfg.lora_rank,
                lora_alpha=cfg.lora_alpha,
                lora_dropout=cfg.lora_dropout,
                target_modules=cfg.lora_target_mods or DEFAULT_TARGET_MODS,
                task_type="CAUSAL_LM",
            )
            model = get_peft_model(model, lora_config)
            model.print_trainable_parameters()

        # --- Load dataset and tokenize. ---
        dataset = load_dataset(Path(dataset_path))
        tokenized = dataset.map(build_tokenize_fn(tokenizer, cfg.max_sequence_length), batched=True)

        if cfg.validation_split > 0:
            split = tokenized.train_test_split(test_size=cfg.validation_split, seed=cfg.seed)
            train_ds, eval_ds = split["train"], split["test"]
        else:
            train_ds, eval_ds = tokenized, None

        # --- TrainingArguments. ---
        output_dir = Path(cfg.output_dir)
        checkpoint_dir = output_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        training_args = TrainingArguments(
            output_dir=str(checkpoint_dir),
            num_train_epochs=cfg.epochs,
            per_device_train_batch_size=cfg.batch_size,
            per_device_eval_batch_size=cfg.batch_size,
            gradient_accumulation_steps=cfg.gradient_accumulation_steps,
            learning_rate=cfg.learning_rate,
            warmup_steps=cfg.warmup_steps,
            weight_decay=cfg.weight_decay,
            lr_scheduler_type=cfg.scheduler_type,
            optim=cfg.optimizer_type,
            logging_steps=1,
            eval_strategy="steps" if eval_ds is not None else "no",
            eval_steps=cfg.validate_every_steps,
            save_strategy="steps",
            save_steps=cfg.validate_every_steps,
            save_total_limit=2,
            load_best_model_at_end=eval_ds is not None,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            report_to=[],
            seed=cfg.seed,
            fp16=torch.cuda.is_available(),
            bf16=False,
            gradient_checkpointing=cfg.grad_checkpoint,
        )

        total_steps = (len(train_ds) // (cfg.batch_size * cfg.gradient_accumulation_steps)) * cfg.epochs
        callback = DistilleryProgressCallback(writer, total_steps)

        # --- Trainer. ---
        # Always use the plain HF Trainer with tokenized data.
        from transformers import Trainer

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            tokenizer=tokenizer,
            callbacks=[callback],
        )

        writer.event("status", value="starting_training", total_steps=total_steps)

        trainer.train(resume_from_checkpoint=cfg.resume_from)

        # Save final adapter + tokenizer.
        if cfg.use_lora:
            adapter_path = output_dir / "adapter"
            model.save_pretrained(str(adapter_path))
        else:
            model.save_pretrained(str(output_dir / "model"))
        tokenizer.save_pretrained(str(output_dir / "tokenizer"))

        # Final evaluation.
        final_metrics: dict[str, Any] = {"status": "completed", "kind": "causal_lm"}
        if eval_ds is not None:
            eval_result = trainer.evaluate()
            final_metrics.update(eval_result)

        # Track B: JSON validity rate over held-out outputs when a schema is
        # present, so the UI can display a validity badge.
        if cfg.json_schema:
            try:
                raw_records = _read_raw_records(Path(dataset_path))
                if cfg.validation_split > 0 and raw_records:
                    cut = max(1, int(len(raw_records) * (1 - cfg.validation_split)))
                    held_out = raw_records[cut:]
                else:
                    held_out = raw_records

                rate = compute_json_validity_rate(
                    model,
                    tokenizer,
                    held_out,
                    cfg.json_schema,
                    cfg.max_validation_samples,
                    cfg.max_new_tokens,
                    "cuda" if torch.cuda.is_available() else "cpu",
                )
                final_metrics["json_validity_rate"] = rate
                writer.event("json_validity", rate=rate)
            except Exception as exc:  # noqa: BLE001
                final_metrics["json_validity_rate"] = None
                writer.event("json_validity", status="error", reason=str(exc))

        writer.event("complete", **final_metrics)  # final_metrics already carries status and kind
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


# Register this module's runner under the causal_lm kind.
def _runner(cfg_dict: dict, writer: object, job_dir: str, dataset_path: str) -> int:
    # Accept either a Config object or a raw dict for loose coupling from run.py.
    c = cfg_dict if isinstance(cfg_dict, Config) else Config.from_dict(cfg_dict)
    # Ensure paths are resolved like the legacy loader did.
    c.dataset_path = dataset_path or c.dataset_path

    return run(c, writer, Path(job_dir), c.dataset_path)


register("causal_lm", _runner)