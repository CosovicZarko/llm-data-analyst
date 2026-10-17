# Fine-tuning Qwen3-4B-Instruct-2507 for text-to-SQL

**Status:** reproducible dataset and scripts, NOT a trained adapter. The actual CUDA training/evaluation must be run on a GPU machine.

## Dataset

`python -m training.build_dataset` generates the files shipped here again, deterministically (seed 20261015). All examples are *synthetic* and template-derived, rather than human-labeled production questions. Generated records use the exact schema formatting and SQL-only system prompt from `dataset_utils.py` and `llm_client.py`. Reference SQL was validated with SQLite during construction. For DuckDB/SQLGlot verification, use `python -m training.validate_data`.

| Split | Examples | Schemas | Domains |
|---|---:|---:|---:|
| Train | 2,484 | 54 | 9 |
| Validation | 828 | 18 | 3 new domains |
| Final synthetic test | 828 | 18 | 3 other domains |

The existing 80-question benchmark uses **five different real-looking toy schemas**, and **none of its question/SQL pairs or input tables are used for training**. Those 80 were studied to choose which *skills* to emphasize, so the benchmark is no longer a fully untouched blind test. Use `test.jsonl` (synthetic new domains) plus future independently sourced datasets for additional testing. Synthetic test cases use the same question templates and cannot prove robust real-world generalization.

Priority skills: `GROUP BY`+`ORDER BY`+`LIMIT`, `HAVING`, status-independent aggregates, weighted arithmetic, conditional counts, date filters. Every row is conversational prompt/completion, so TRL computes loss only on the SQL assistant completion. Models see only the schema and three sample rows, never the full uploaded dataset or benchmark gold SQL.

## Prepare and inspect on Mac (existing Python)

```bash
cd ~/Desktop/llm-data-analyst
python -m unittest discover -s tests -v
python -m training.validate_data
python -m training.build_dataset
```

`validate_data` must run on the Mac with the app's existing DuckDB/SQLGlot packages; no GPU, Hugging Face, PyTorch or cloud SDK is needed. The generator regenerates *exactly* the shipped files and hashes. Train/val/test fixtures are tiny CSV files under `training/data/fixtures/`.

## Google Cloud GPU setup

Choose a Linux Compute Engine instance with an NVIDIA **L4 24 GB GPU** (a `g2-standard-8` or `g2-standard-12` has one L4). Availability, quota and credit eligibility depend on your account/region; check those before creating an instance. Use a suitable recent Ubuntu image and install the NVIDIA driver/CUDA-compatible PyTorch according to the official documentation. Note that **G2 VMs do not support Deep Learning VM Images** as boot disks; use Ubuntu and install drivers or a supported container instead. Check `nvidia-smi` and `python3 -c 'import torch; print(torch.cuda.is_available())'` first. For Linux, we use a **separate GPU-side environment**; no changes to your Mac's Conda base are required.

Copy the unpacked project to the VM using your usual SCP/cloud workflow, then on the GPU VM:

```bash
cd ~/llm-data-analyst
python3 -m pip install -r requirements.txt -r requirements-train.txt
python3 -c 'import torch; print(torch.__version__, torch.cuda.is_available())'
python3 -m training.validate_data

# Tiny GPU integration test (still downloads ~8 GB base weights):
python3 -m training.train_qlora --max-steps 3 --output outputs/smoke_qlora

# Full experiment, a new output dir:
python3 -m training.train_qlora --output outputs/qwen3_sql_qlora
```

Do NOT run `train_qlora` on Mac CPU/Apple Metal. It uses CUDA 4-bit NF4, PEFT/LoRA rank 16, cosine LR 1e-4, 2 epochs, effective training batch of 8, seq length 1024, completion-only loss, and gradient checkpointing. `--max-steps` and `--max-length` allow reducing resource use. The default two epochs can take hours depending on the available GPU and CUDA libraries; **no run duration or accuracy increase is guaranteed**.

Adapter: `outputs/qwen3_sql_qlora/adapter/` (plus run metadata and training logs). Keep Cloud GPU shut down when not using it to avoid consuming credits.

## Fair evaluation on the same GPU

The previous **73.75%** is a **local Ollama Q4_K_M baseline**, not a directly controlled comparison with GPU 4-bit NF4 inference. To isolate fine-tuning impact, evaluate **both** the Hugging Face base and fine-tuned adapter on the same 80 questions, using the same environment, prompts and decoding:

```bash
# First 10; then all 80 (saved cases resume)
python3 -m training.evaluate_hf --limit 10
python3 -m training.evaluate_hf --limit 0

python3 -m training.evaluate_hf --adapter outputs/qwen3_sql_qlora/adapter --limit 10
python3 -m training.evaluate_hf --adapter outputs/qwen3_sql_qlora/adapter --limit 0
```

Results: `reports/hf_base_summary.json` and `reports/hf_adapter_summary.json`, with detailed `.jsonl` per-case results. They use the same DuckDB SQL guard, scorer, datasets, and one execution-error repair rule. Valid-but-wrong SQL is not repaired. The test set `training/data/test.jsonl` is held back from choosing checkpoints/hyperparameters. **After finalizing the adapter** test on these completely separate domains:

```bash
python3 -m training.evaluate_synthetic --split test --limit 0
python3 -m training.evaluate_synthetic --split test --adapter outputs/qwen3_sql_qlora/adapter --limit 0
```

These 828-question runs may be lengthy; use `--limit 20` for a smoke run before choosing a different output filename for the full test. They cover unseen *domains* but repeated *template families*, so they remain synthetic evidence, not real-world reliability benchmarks.

**The adapter is not automatically installed into Ollama or the Streamlit UI.** Once accuracy is measured, we can integrate the adapter for inference or convert a merged model into Ollama-compatible weights. Do not represent this package as a trained deployed model yet.

## Reproducibility and limitations

- `training/data/manifest.json` records splits, skill counts, SHA256 checksums and seed.
- All train/validation/final-test splits hold out *entire domains*; the final test still shares our manually authored SQL question templates.
- No benchmark data was copied into training. This is an initial experiment, not a large curated Spider/BIRD corpus.
- Reference checks use SQLite for easy local reproducibility and DuckDB/SQLGlot via `validate_data` for exact application compatibility.
- `model_id` and file fingerprints are recorded in baseline HF evaluation files; change `--output` if you compare a different adapter.
- DuckDB SQL validation is a defense-in-depth guard, not a process-level public sandbox.
