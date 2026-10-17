"""Model selection and inference routing for the Streamlit app.

Both models deliberately use the `raw` inference mode from the audited
benchmark. Keep the SQL validator and the bounded DuckDB repair loop in
query_runner.execute_with_one_repair; do not duplicate or bypass them here.
"""

from __future__ import annotations

from functools import partial
from typing import Any, Callable

INFERENCE_MODE = "raw"

# Ollama model tags must match `ollama list` on the user's machine.
MODEL_OPTIONS = {
    "Fine-tuned Qwen3-4B (Google Cloud)": "qwen-sql-analyst:v1",
    "Pretrained Qwen3-4B (Q4_K_M)": "qwen3:4b-q4_K_M",
}


def _validate_model(model: str) -> None:
    if model not in MODEL_OPTIONS.values():
        raise ValueError("Select a supported model from the model selector.")


def generate_for_model(
    question: str,
    schema: str,
    model: str,
    *,
    generator: Callable[..., str] | None = None,
) -> str:
    """Use the same raw Qwen prompt path for either installed model."""
    _validate_model(model)
    if generator is None:
        from llm_client import generate_sql

        generator = generate_sql
    return generator(question, schema, model=model, mode=INFERENCE_MODE)


def execute_for_model(
    frame: Any,
    question: str,
    schema: str,
    generated_sql: str,
    model: str,
    *,
    runner: Callable[..., Any] | None = None,
    repairer: Callable[..., str] | None = None,
) -> Any:
    """Execute with existing guard and at most one repair using SAME model/mode."""
    _validate_model(model)
    if runner is None:
        from query_runner import execute_with_one_repair

        runner = execute_with_one_repair
    if repairer is None:
        from llm_client import repair_sql

        repairer = repair_sql
    configured_repairer = partial(repairer, model=model, mode=INFERENCE_MODE)
    return runner(frame, question, schema, generated_sql, repairer=configured_repairer)
