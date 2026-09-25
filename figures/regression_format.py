"""Shared display conventions for the active regression-table renderers.

These helpers format already-computed estimates; they do not fit models or
change stored values.  Regression artifacts use hyphenated provider names and
``persona`` prompt keys, unlike the statistics hub's separate key convention.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any


PROVIDERS = (
    ("GPT-5.6 Sol", "ChatGPT"),
    ("Claude Opus 4.8", "Claude"),
    ("Gemini 3.5 Flash", "Gemini"),
)

def source_key(provider: str, effort: str, prompt: str) -> str:
    """Join the exact regression-artifact keys without alias normalization."""
    return f"{provider}|{effort}|{prompt}"


def _display_number(value: float) -> float:
    """Suppress negative zero at two decimals without changing stored values."""
    return 0.0 if abs(value) < 0.005 else value


def interval_cell(
    estimate: float,
    interval: Sequence[float],
    *,
    bold_if_excludes_zero: bool = False,
    stacked: bool = False,
) -> str:
    """Format a point and CI, testing zero exclusion before display rounding."""
    low, high = (float(value) for value in interval)
    point = f"{_display_number(float(estimate)):.2f}"
    if bold_if_excludes_zero and (low > 0.0 or high < 0.0):
        point = rf"\boldsymbol{{{point}}}"
    bounds = f"[{_display_number(low):.2f},{_display_number(high):.2f}]"
    if stacked:
        return rf"\shortstack{{${point}$\\${bounds}$}}"
    return rf"${point}{bounds}$"


def validate_feature_inventory(
    data: dict[str, Any], expected: tuple[str, ...], family: str
) -> None:
    """Reject stale or reordered feature artifacts before rendering any table."""
    actual = tuple(data["feature_names"])
    if actual != expected:
        raise ValueError(
            f"Unexpected {family} feature inventory: expected {expected}, got {actual}"
        )
