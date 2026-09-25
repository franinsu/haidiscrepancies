#!/usr/bin/env python3
"""Distribution metrics shared by the statistics scripts.

Every function accepts either a ``Counter`` of class counts or a mapping of
class probabilities and works on the union of the two supports.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Mapping

import numpy as np
from scipy.special import entr, rel_entr

Distribution = Mapping[str, float]


def distribution(counter: Counter[str] | Distribution) -> dict[str, float]:
    if isinstance(counter, Counter):
        total = sum(counter.values())
        return {key: value / total for key, value in counter.items()} if total else {}
    return dict(counter)


def tv(left: Counter[str] | Distribution, right: Counter[str] | Distribution) -> float:
    """Total-variation distance, 0.5 * sum |p - q| over the union support."""
    a, b = distribution(left), distribution(right)
    return 0.5 * sum(abs(a.get(key, 0.0) - b.get(key, 0.0)) for key in set(a) | set(b))


def js(left: Counter[str] | Distribution, right: Counter[str] | Distribution) -> float:
    """Base-2 Jensen--Shannon divergence, in bits."""
    a, b = distribution(left), distribution(right)
    keys = sorted(set(a) | set(b))
    p, q = (np.asarray([d.get(key, 0.0) for key in keys]) for d in (a, b))
    midpoint = (p + q) / 2
    divergence = float((rel_entr(p, midpoint).sum() + rel_entr(q, midpoint).sum()) / (2 * math.log(2)))
    # Nearly identical distributions can round below zero. Preserve nonfinite
    # values so that invalid inputs cannot turn into a zero distance.
    return max(divergence, 0.0) if math.isfinite(divergence) else divergence


def js_distance(left: Counter[str] | Distribution, right: Counter[str] | Distribution) -> float:
    """Square-root Jensen--Shannon distance, bounded between zero and one."""
    return math.sqrt(js(left, right))


def hellinger(left: Counter[str] | Distribution, right: Counter[str] | Distribution) -> float:
    """Hellinger distance, bounded between zero and one."""
    a, b = distribution(left), distribution(right)
    return math.sqrt(
        0.5 * sum((math.sqrt(a.get(key, 0.0)) - math.sqrt(b.get(key, 0.0))) ** 2 for key in set(a) | set(b))
    )


def entropy(counter: Counter[str] | Distribution) -> float:
    """Plug-in Shannon entropy in bits."""
    p = distribution(counter)
    # entr preserves supplied probability masses, including an empty sample;
    # stats.entropy would renormalize mappings a second time.
    return float(entr(list(p.values())).sum() / math.log(2))


def mass(counter: Counter[str] | Distribution, targets: set[str] | frozenset[str]) -> float:
    """Probability mass on a set of classes; zero for an empty sample."""
    p = distribution(counter)
    return sum(p.get(key, 0.0) for key in targets)


def mutual_information(pairs: list[tuple[str, str]]) -> float:
    """Plug-in mutual information in bits between paired class labels."""
    if not pairs:
        raise ValueError("Cannot compute mutual information without paired responses")
    total = len(pairs)
    joint = Counter(pairs)
    left = Counter(a for a, _ in pairs)
    right = Counter(b for _, b in pairs)
    return sum((count / total) * math.log2((count / total) / ((left[a] / total) * (right[b] / total))) for (a, b), count in joint.items())
