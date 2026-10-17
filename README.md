# LLM Data Analyst

Analyze CSV and Excel datasets using natural-language questions and a fine-tuned Qwen3-4B language model.

## Features

- Upload CSV and Excel datasets.
- Ask multiple questions about the same dataset.
- Generate and execute DuckDB SQL queries.
- View generated SQL and query results.
- Automatically correct SQL execution errors.

## Tech Stack

Python, Streamlit, DuckDB, Ollama, Qwen3-4B, SQLGlot, Google Cloud Vertex AI.

## Run Locally

Start Ollama with the fine-tuned model registered as `qwen-sql-analyst:v1`.

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```
