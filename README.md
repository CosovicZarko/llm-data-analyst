# LLM Data Analyst

A data analysis application powered by a fine-tuned Qwen3-4B model that converts natural-language questions into SQL queries and executes them to return results.

**[Live Demo]([https://occupations-years-bus-systematic.trycloudflare.com](https://waves-friends-representations-they.trycloudflare.com?utm_source=chatgpt.com)**

## Features

- Upload CSV and Excel files.
- Ask multiple questions in natural language.
- Generate and execute SQL using DuckDB.
- Display query results and generated SQL.
- Automatically repair SQL execution errors.

## Model

Qwen3-4B fine-tuned on 2,484 text-to-SQL examples using Google Cloud Vertex AI.

**Benchmark accuracy (80 questions):**

| Model | Accuracy |
|---|---:|
| Pretrained | 78.75% |
| Fine-tuned | 83.75% |

## Tech Stack

Python · Streamlit · Qwen3-4B · Ollama · DuckDB · SQLGlot · Google Cloud Vertex AI

## Run Locally

Requires the fine-tuned model registered in Ollama as `qwen-sql-analyst:v1`.

```bash
pip install -r requirements.txt
streamlit run app.py
```
