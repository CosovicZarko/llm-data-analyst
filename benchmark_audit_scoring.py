"""Pure comparison policy for the offline benchmark audit.

This module does NOT modify the evaluator or its historical results.
"""
from __future__ import annotations

from collections import Counter
from datetime import date, datetime
from decimal import Decimal
from itertools import permutations
from numbers import Number
import math

# These benchmark questions request a SET of answers, not a presentation order.
# All other explicitly ordered cases retain their original order requirement.
UNORDERED_CASE_IDS = frozenset({
    "ret-14", "sup-13", "sta-14", "saa-12", "war-12", "war-14",
})


def cell(value):
    """Match the existing benchmark's six-decimal numeric precision."""
    if value is None:
        return ("null", "")
    if isinstance(value, (datetime, date)):
        return ("time", value.isoformat())
    if isinstance(value, bool):
        return ("bool", value)
    if isinstance(value, (Number, Decimal)):
        number = float(value)
        if math.isnan(number):
            return ("null", "")
        if not math.isfinite(number):
            return ("nonfinite", str(number))
        return ("num", round(number, 6))
    if isinstance(value, str):
        return ("text", value)
    # Handle pandas NA and Timestamp without making this pure module depend on pandas.
    try:
        import pandas as pd
        if pd.isna(value):
            return ("null", "")
    except (ImportError, TypeError, ValueError):
        pass
    return ("text", str(value))


def tied_group_top(actual, all_gold, k: int, *, ascending: bool) -> bool:
    """Accept any valid top-k selection when the benchmark data contain ties.

    Requires matching two-column (group key, group metric) output, exactly k
    distinct reference groups, correct aggregate metric for every returned
    group, every group strictly above the cutoff, and ranking in score order.
    Tied groups at the cutoff may be chosen in any order. Does not accept
    additional/missing fields, unrelated groups, or incorrect metrics.
    """
    if k < 1 or actual.shape != (k, 2) or all_gold.shape[1] != 2 or len(all_gold) < k:
        return False
    reference = [tuple(cell(v) for v in row) for row in all_gold.itertuples(index=False, name=None)]
    if any(value[1][0] != "num" for value in reference):
        return False
    # Values from the reference full ranking, not the model, define the cutoff.
    cutoff = reference[k - 1][1][1]
    compulsory = {row for row in reference if (row[1][1] < cutoff if ascending else row[1][1] > cutoff)}
    allowed = Counter(reference)
    for cols in permutations((0, 1)):
        proposed = [tuple(cell(r[i]) for i in cols) for r in actual.itertuples(index=False, name=None)]
        if any(v[1][0] != "num" for v in proposed):
            continue
        if not all((v[1][1] <= cutoff if ascending else v[1][1] >= cutoff) for v in proposed):
            continue
        if not compulsory.issubset(set(proposed)):
            continue
        if Counter(proposed) - allowed:
            continue
        scores = [v[1][1] for v in proposed]
        if all((a <= b if ascending else a >= b) for a, b in zip(scores, scores[1:])):
            return True
    return False


def has_rank_ties(all_gold, k: int) -> bool:
    """True if ties affect order inside the displayed top-k or its cutoff."""
    if all_gold.shape[1] != 2 or len(all_gold) < k or k < 1:
        return False
    values = [cell(row[1]) for row in all_gold.itertuples(index=False, name=None)]
    return any(values[i] == values[i - 1] for i in range(1, min(len(values), k + 1)))
