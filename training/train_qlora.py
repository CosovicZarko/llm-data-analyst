"""QLoRA supervised fine-tuning on a single NVIDIA CUDA GPU.

Run only on Linux with a CUDA GPU, not inside the Mac Conda base shell.
Public base weights: Qwen/Qwen3-4B-Instruct-2507.
The LoRA adapter is saved separately; base weights are not modified.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import random

MODEL = "Qwen/Qwen3-4B-Instruct-2507"
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "training" / "data"


def _read(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing training data: {path}. Run python -m training.build_dataset")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows or any(not row.get("prompt") or not row.get("completion") for row in rows):
        raise ValueError(f"Malformed prompt/completion data: {path}")
    return rows


def validate_splits(train: list[dict], val: list[dict]) -> None:
    a = {row["schema_id"].rsplit("_v", 1)[0] for row in train}
    b = {row["schema_id"].rsplit("_v", 1)[0] for row in val}
    if a & b:
        raise ValueError(f"Train and validation share domains: {sorted(a & b)}")
    if len({r["id"] for r in train}) != len(train):
        raise ValueError("Duplicate training example IDs")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "qwen3_sql_qlora")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--max-steps", type=int, default=-1, help="Use 3 for GPU smoke test; -1 for full training")
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--eval-samples", type=int, default=144, help="Validation loss subset, sampled deterministically")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.max_steps == 0 or args.max_steps < -1:
        parser.error("--max-steps must be -1 or positive")
    if args.batch_size < 1 or args.grad_accum < 1 or args.eval_samples < 1:
        parser.error("batch size, grad accum and eval samples must be positive")
    # Fail early and clearly before any model download or CUDA dependency use.
    import torch
    if not torch.cuda.is_available():
        raise SystemExit("QLoRA requires a CUDA GPU on Linux. Prepare data on Mac; train on Google Cloud GPU.")
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed
    from trl import SFTConfig, SFTTrainer

    train = _read(args.data / "train.jsonl")
    validation = _read(args.data / "validation.jsonl")
    validate_splits(train, validation)
    if args.output.resolve().is_relative_to(args.data.resolve()):
        raise ValueError("Training output must not overwrite the input data")
    args.output.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    # Keep a balanced validation subset across completely unseen schemas.
    by_schema: dict[str, list[dict]] = {}
    for row in validation:
        by_schema.setdefault(row["schema_id"], []).append(row)
    rng = random.Random(args.seed)
    for rows in by_schema.values():
        rng.shuffle(rows)
    selected: list[dict] = []
    while len(selected) < min(args.eval_samples, len(validation)):
        advanced = False
        for schema in sorted(by_schema):
            if by_schema[schema] and len(selected) < args.eval_samples:
                selected.append(by_schema[schema].pop())
                advanced = True
        if not advanced:
            break
    train_data = Dataset.from_list([{"prompt": r["prompt"], "completion": r["completion"]} for r in train])
    eval_data = Dataset.from_list([{"prompt": r["prompt"], "completion": r["completion"]} for r in selected])

    bf16 = torch.cuda.is_bf16_supported()
    compute_dtype = torch.bfloat16 if bf16 else torch.float16
    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )
    print(f"Loading {args.model} with 4-bit NF4 quantization on {torch.cuda.get_device_name(0)}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, quantization_config=quant, dtype=compute_dtype, device_map={"": 0},
    )
    model.config.use_cache = False
    # The SFTTrainer prepares the k-bit model for PEFT when peft_config is supplied.
    peft = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules="all-linear", bias="none", task_type="CAUSAL_LM",
    )
    config = SFTConfig(
        output_dir=str(args.output / "checkpoints"),
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=1e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        max_length=args.max_length,
        completion_only_loss=True,
        packing=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        bf16=bf16,
        fp16=not bf16,
        eval_strategy="steps",
        eval_steps=100 if args.max_steps == -1 else max(1, args.max_steps),
        save_strategy="steps",
        save_steps=100 if args.max_steps == -1 else max(1, args.max_steps),
        save_total_limit=2,
        logging_steps=10,
        report_to="none",
        seed=args.seed,
        data_seed=args.seed,
        optim="adamw_torch",
        dataloader_num_workers=0,
    )
    trainer = SFTTrainer(
        model=model, args=config,
        train_dataset=train_data, eval_dataset=eval_data,
        processing_class=tokenizer,
        peft_config=peft,
    )
    result = trainer.train()
    adapter_dir = args.output / "adapter"
    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    metadata = {
        "base_model": args.model,
        "adapter": str(adapter_dir),
        "device": torch.cuda.get_device_name(0),
        "precision": "bf16" if bf16 else "fp16",
        "quantization": "NF4 4bit + double quant",
        "train_examples": len(train),
        "validation_examples_used": len(selected),
        "seed": args.seed,
        "hyperparameters": {"epochs": args.epochs, "max_steps": args.max_steps,
                            "batch_size": args.batch_size, "grad_accum": args.grad_accum,
                            "max_length": args.max_length, "learning_rate": 1e-4, "lora_rank": 16},
        "dataset_sha256": {
            name: hashlib.sha256((args.data / f"{name}.jsonl").read_bytes()).hexdigest()
            for name in ("train", "validation")
        },
        "training_metrics": result.metrics,
    }
    (args.output / "training_run.json").write_text(json.dumps(metadata, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"Adapter saved: {adapter_dir}", flush=True)
    print(f"Run metadata: {args.output / 'training_run.json'}", flush=True)


if __name__ == "__main__":
    main()
