"""Schema-aware natural-language examples for the original Streamlit interface.

Only uses columns present in the normalized DataFrame. No model inference
or SQL execution is performed when constructing the list.
"""

from __future__ import annotations

import pandas as pd

MAX_BATCH_QUESTIONS = 5


def suggested_questions(frame: pd.DataFrame) -> list[str]:
    """Return up to 8 distinct sample questions grounded in actual columns."""
    columns = list(frame.columns)
    numeric = [
        c for c in columns
        if pd.api.types.is_numeric_dtype(frame[c])
        and not pd.api.types.is_bool_dtype(frame[c])
    ]
    categorical = [
        c for c in columns
        if not pd.api.types.is_numeric_dtype(frame[c])
        or pd.api.types.is_bool_dtype(frame[c])
    ]

    def numeric_priority(column: str) -> tuple[int, int]:
        label = column.lower()
        if any(token in label for token in ('revenue', 'net_sales', 'sales', 'amount', 'total', 'cost', 'price', 'profit')):
            return (0, columns.index(column))
        if any(token in label for token in ('quantity', 'units', 'hours', 'salary', 'rate', 'score')):
            return (1, columns.index(column))
        if label == 'id' or label.endswith('_id') or label.startswith('id_') or 'number' in label:
            return (3, columns.index(column))
        return (2, columns.index(column))

    usable_numeric = [c for c in sorted(numeric, key=numeric_priority) if numeric_priority(c)[0] != 3]
    # Avoid treating free-form customer IDs or unique descriptions as useful grouping dimensions.
    def dimension_priority(column: str) -> tuple[int, int, int]:
        label = column.lower()
        preferred = ("region", "territory", "market", "category", "product", "department", "team", "status")
        return (
            int(frame[column].nunique(dropna=True) > 25),
            0 if any(token in label for token in preferred) else 1,
            columns.index(column),
        )

    categorical.sort(key=dimension_priority)
    dimensions = [c for c in categorical if not (c.lower() == 'id' or c.lower().endswith('_id'))]

    suggestions: list[str] = []

    def add(q: str) -> None:
        if q not in suggestions:
            suggestions.append(q)

    if usable_numeric and dimensions:
        measure, group = usable_numeric[0], dimensions[0]
        add(f'Which {group} has the highest total {measure}?')
        add(f'Show total {measure} by {group}, highest first.')

    add('How many records are in this dataset?')

    if usable_numeric:
        measure = usable_numeric[0]
        add(f'What is the total {measure}?')
        add(f'What is the average {measure}?')
        if len(usable_numeric) > 1:
            add(f'What is the total {usable_numeric[1]}?')

    if dimensions:
        group = dimensions[0]
        add(f'How many distinct {group} values are there?')
        add(f'How many records are there for each {group}?')
        if len(dimensions) > 1:
            add(f'How many distinct {dimensions[1]} values are there?')

    if not usable_numeric and not dimensions and columns:
        add(f'How many distinct {columns[0]} values are there?')

    return suggestions[:8]


def prepare_questions(selected: list[str], custom: str, *, max_questions: int = MAX_BATCH_QUESTIONS) -> list[str]:
    """Validate and deduplicate selected examples plus one optional free-form question."""
    questions: list[str] = []
    seen: set[str] = set()
    for item in [*selected, custom]:
        question = item.strip()
        if not question:
            continue
        if len(question) > 1200:
            raise ValueError('Each question must be 1,200 characters or fewer.')
        key = question.casefold()
        if key not in seen:
            seen.add(key)
            questions.append(question)
    if not questions:
        raise ValueError('Choose an example question or type your own.')
    if len(questions) > max_questions:
        raise ValueError(f'Run at most {max_questions} questions at a time.')
    return questions
