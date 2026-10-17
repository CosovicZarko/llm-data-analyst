"""Execution accuracy on the unchanged 80-case benchmark using HF base/LoRA.

Optional adapter path: --adapter outputs/qwen3_sql_qlora/adapter

Use --limit 10 before --limit 0; records are appended and can resume.
For a fair comparison, evaluate HF base and HF adapter in the same environment,
quantization format, prompts, SQL guard and generation parameters.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import time

import torch

from llm_client import SYSTEM_PROMPT, extract_sql
from evaluate_baseline import (
    case_fingerprint, evaluate_case, load_cases, load_datasets, summarize,
)
from training.train_qlora import MODEL

ROOT = Path(__file__).resolve().parents[1]


class SqlGenerator:
    def __init__(self, base_model: str, adapter: str | None):
        if not torch.cuda.is_available():
            raise SystemExit("Hugging Face 4bit evaluation requires an NVIDIA CUDA GPU.")
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        from peft import AutoPeftModelForCausalLM

        self.adapter = adapter
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        quant = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype,
        )
        kw = {"quantization_config": quant, "dtype": dtype, "device_map": {"": 0}}
        print(f"Loading {'fine-tuned adapter' if adapter else 'base model'} on GPU", flush=True)
        if adapter:
            self.model = AutoPeftModelForCausalLM.from_pretrained(adapter, **kw)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(base_model, **kw)
        self.model.eval()

    def run(self, messages: list[dict]) -> str:
        input_ids = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, return_tensors="pt",
        ).to("cuda:0")
        with torch.inference_mode():
            result = self.model.generate(
                input_ids, do_sample=False, max_new_tokens=320,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        text = self.tokenizer.decode(result[0, input_ids.shape[-1]:], skip_special_tokens=True)
        return extract_sql(text)

    def generate(self, question: str, schema: str, *, model: str, url: str) -> str:
        return self.run([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"{schema}\n\nUser question: {question.strip()}\n\nSQL:"},
        ])

    def repair(self, question: str, schema: str, previous_sql: str, database_error: str,
               *, model: str, url: str) -> str:
        return self.run([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"{schema}\n\nUser question: {question.strip()}\n\nSQL:"},
            {"role": "assistant", "content": previous_sql},
            {"role": "user", "content": (
                f"DuckDB rejected your previous SQL with this error:\n{database_error[:1500]}\n\n"
                "Fix the error using only the provided schema. "
                "Return exactly one corrected SELECT query, and no other text."
            )},
        ])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", default=MODEL)
    parser.add_argument("--adapter", type=str, default=None)
    parser.add_argument("--limit", type=int, default=10, help="0 = all 80")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if args.limit < 0:
        parser.error("--limit must be >=0")
    name = "hf_adapter" if args.adapter else "hf_base"
    model_id = ("adapter:" + str(Path(args.adapter).resolve())) if args.adapter else ("base:" + args.base_model)
    output = args.output or (ROOT / "reports" / f"{name}.jsonl")
    output.parent.mkdir(parents=True, exist_ok=True)
    cases = load_cases()
    datasets = load_datasets(cases)
    targets = cases if args.limit == 0 else cases[:args.limit]
    fingerprints = {
        c["id"]: case_fingerprint(
            c, (ROOT / "benchmark" / "datasets" / (c["dataset"] + ".csv")).read_bytes()
        ) for c in cases
    }
    existing: dict[str, dict] = {}
    if output.exists():
        for line in output.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row["id"] in existing or row.get("model") != model_id or row.get("case_fingerprint") != fingerprints.get(row["id"]):
                raise ValueError("Result mismatch or duplicate. Choose a new --output path.")
            existing[row["id"]] = row
    pending = [c for c in targets if c["id"] not in existing]
    print(f"Benchmark {len(targets)}/80, {len(existing)} saved, {len(pending)} pending; model: {model_id}", flush=True)
    if pending:
        generator = SqlGenerator(args.base_model, args.adapter)
        with output.open("a", encoding="utf-8") as fp:
            for ix, case in enumerate(pending, 1):
                record = evaluate_case(case, datasets[case["dataset"]],
                                       model=model_id, url="local://hf",
                                       generate=generator.generate, repair=generator.repair)
                record["case_fingerprint"] = fingerprints[case["id"]]
                fp.write(json.dumps(record, ensure_ascii=False) + "\n")
                fp.flush()
                existing[case["id"]] = record
                print(f"[{ix}/{len(pending)}] {case['id']} "
                      f"{'PASS' if record['final_correct'] else 'FAIL'} ({record['seconds']}s)", flush=True)
    result = summarize([existing[c["id"]] for c in targets], total=80)
    result["backend"] = "Hugging Face Transformers, 4-bit NF4"
    result["model_id"] = model_id
    summ_path = output.with_name(output.stem + "_summary.json")
    summ_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Accuracy: {result['first_pass_execution_accuracy_pct']}% first / "
          f"{result['after_one_repair_execution_accuracy_pct']}% after repair")
    print(f"Saved: {output}; {summ_path}")


if __name__ == "__main__":
    main()
