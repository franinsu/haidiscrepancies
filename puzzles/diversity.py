from __future__ import annotations

import random
from typing import Dict, List, Sequence

from .common import Puzzle, equal_bucket_counts, select_diverse, select_diverse_by_key


PROBE_PRESSURE_ORDER = ["low", "medium", "high"]


def choose_final_dataset(
    candidates: Sequence[Puzzle],
    num: int,
    seed: int,
    desired: Dict[str, int] | None = None,
) -> List[Puzzle]:
    rng = random.Random(seed)
    if candidates and "search_pressure_bucket" in candidates[0].get("features", {}):
        return select_diverse_by_key(
            list(candidates),
            num,
            rng,
            key_fn=lambda p: p["features"]["search_pressure_bucket"],
            desired=desired or equal_bucket_counts(num, PROBE_PRESSURE_ORDER),
            bucket_order=PROBE_PRESSURE_ORDER,
        )
    return select_diverse(list(candidates), num, rng, desired)
