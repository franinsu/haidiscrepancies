from __future__ import annotations

import hashlib
import json
import math
import os
import random
from collections import Counter, defaultdict
from typing import Any, Callable, Dict, Iterable, List, Sequence, Tuple


Puzzle = Dict[str, Any]

MIN_MULTI_SOLUTIONS = 3
MAX_MULTI_SOLUTIONS = 8
CUE_MIN_SOLUTIONS = 4


def solution_count_in_range(
    count: int,
    min_solutions: int = MIN_MULTI_SOLUTIONS,
    max_solutions: int = MAX_MULTI_SOLUTIONS,
) -> bool:
    return min_solutions <= count <= max_solutions


REQUIRED_FIELDS = {
    "id",
    "puzzle_type",
    "prompt_text",
    "rendered_puzzle",
    "machine_readable_instance",
    "num_solutions",
    "solutions",
    "features",
    "probe_metadata",
    "difficulty_bucket",
    "diversity_signature",
}


def stable_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def stable_hash(obj: Any) -> str:
    return hashlib.sha256(stable_json(obj).encode("utf-8")).hexdigest()


def write_jsonl(path: str, rows: Iterable[Puzzle]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")


def read_jsonl(path: str) -> List[Puzzle]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def bucket_counts(
    total: int, desired: Dict[str, int] | None = None
) -> Dict[str, int]:
    if desired is None:
        desired = {"easy": 20, "medium": 60, "hard": 20}
    if total == 100:
        return dict(desired)
    scale = total / 100.0
    counts = {k: int(round(v * scale)) for k, v in desired.items()}
    diff = total - sum(counts.values())
    counts["medium"] = counts.get("medium", 0) + diff
    return counts


def equal_bucket_counts(total: int, buckets: Sequence[str]) -> Dict[str, int]:
    base = total // len(buckets)
    remainder = total % len(buckets)
    return {bucket: base + (1 if i < remainder else 0) for i, bucket in enumerate(buckets)}


def numeric_feature_vector(features: Dict[str, Any]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for k, v in features.items():
        if isinstance(v, bool):
            out[k] = 1.0 if v else 0.0
        elif isinstance(v, (int, float)) and not isinstance(v, bool):
            if math.isfinite(float(v)):
                out[k] = float(v)
        elif isinstance(v, list):
            for i, item in enumerate(v):
                if isinstance(item, (int, float)) and not isinstance(item, bool):
                    out[f"{k}_{i}"] = float(item)
    return out


def _normalized_vectors(items: Sequence[Puzzle]) -> List[Dict[str, float]]:
    raw = [numeric_feature_vector(p["features"]) for p in items]
    keys = sorted({k for row in raw for k in row})
    mins = {k: min(row.get(k, 0.0) for row in raw) for k in keys}
    maxs = {k: max(row.get(k, 0.0) for row in raw) for k in keys}
    vectors: List[Dict[str, float]] = []
    for row in raw:
        vec: Dict[str, float] = {}
        for k in keys:
            span = maxs[k] - mins[k]
            vec[k] = 0.0 if span == 0 else (row.get(k, 0.0) - mins[k]) / span
        vectors.append(vec)
    return vectors


def diversity_distance(a: Puzzle, b: Puzzle, va: Dict[str, float], vb: Dict[str, float]) -> float:
    keys = set(va) | set(vb)
    num = math.sqrt(sum((va.get(k, 0.0) - vb.get(k, 0.0)) ** 2 for k in keys))
    sig_a = a.get("diversity_signature", {})
    sig_b = b.get("diversity_signature", {})
    cat_penalty = 0.0
    for k in sorted(set(sig_a) | set(sig_b)):
        av, bv = sig_a.get(k), sig_b.get(k)
        if isinstance(av, (str, bool, int)) and isinstance(bv, (str, bool, int)) and av != bv:
            cat_penalty += 0.25
        elif isinstance(av, list) and isinstance(bv, list) and av != bv:
            cat_penalty += 0.15
    return num + cat_penalty


def select_diverse(
    candidates: Sequence[Puzzle],
    num: int,
    rng: random.Random,
    desired: Dict[str, int] | None = None,
) -> List[Puzzle]:
    """Stratified greedy max-min diversity selection."""
    return select_diverse_by_key(
        candidates,
        num,
        rng,
        key_fn=lambda p: p["difficulty_bucket"],
        desired=desired or bucket_counts(num),
        bucket_order=("easy", "medium", "hard"),
    )


def select_diverse_by_key(
    candidates: Sequence[Puzzle],
    num: int,
    rng: random.Random,
    key_fn: Callable[[Puzzle], str],
    desired: Dict[str, int],
    bucket_order: Sequence[str] | None = None,
) -> List[Puzzle]:
    """Greedy max-min diversity selection within requested study-design strata."""
    if len(candidates) < num:
        raise ValueError(f"Need {num} candidates, only have {len(candidates)}")
    by_bucket: Dict[str, List[Puzzle]] = defaultdict(list)
    for p in candidates:
        by_bucket[key_fn(p)].append(p)

    final_counts = _rebalance_targets(
        desired,
        {k: len(v) for k, v in by_bucket.items()},
        num,
        bucket_order=bucket_order,
    )
    selected: List[Puzzle] = []
    used_hashes: set[str] = set()
    order = list(bucket_order or sorted(final_counts))
    for bucket in order:
        pool = by_bucket.get(bucket, [])
        chosen = _select_bucket(pool, final_counts.get(bucket, 0), rng, used_hashes)
        selected.extend(chosen)
        used_hashes.update(p["features"].get("canonical_hash", "") for p in chosen)

    if len(selected) < num:
        leftovers = [p for p in candidates if p not in selected]
        selected.extend(_select_bucket(leftovers, num - len(selected), rng, used_hashes))
    return selected[:num]


def _rebalance_targets(
    targets: Dict[str, int],
    available: Dict[str, int],
    total: int,
    bucket_order: Sequence[str] | None = None,
) -> Dict[str, int]:
    order = list(bucket_order or sorted(set(targets) | set(available)))
    counts = {k: min(targets.get(k, 0), available.get(k, 0)) for k in order}
    missing = total - sum(counts.values())
    refill_order = sorted(order, key=lambda k: (targets.get(k, 0) - counts.get(k, 0), available.get(k, 0)), reverse=True)
    for bucket in refill_order:
        if missing <= 0:
            break
        add = min(missing, available.get(bucket, 0) - counts.get(bucket, 0))
        if add > 0:
            counts[bucket] += add
            missing -= add
    if missing > 0:
        for bucket in sorted(available):
            if missing <= 0:
                break
            add = min(missing, available.get(bucket, 0) - counts.get(bucket, 0))
            if add > 0:
                counts[bucket] = counts.get(bucket, 0) + add
                missing -= add
    if missing > 0:
        raise ValueError(f"Could not fill target: missing {missing}")
    return counts


def _select_bucket(
    pool: Sequence[Puzzle], count: int, rng: random.Random, used_hashes: set[str]
) -> List[Puzzle]:
    if count <= 0:
        return []
    pool = [p for p in pool if p["features"].get("canonical_hash", "") not in used_hashes]
    if len(pool) < count:
        raise ValueError(f"Bucket has {len(pool)} candidates, need {count}")
    vectors = _normalized_vectors(pool)
    selected_idx = [rng.randrange(len(pool))]
    remaining = set(range(len(pool))) - set(selected_idx)
    while len(selected_idx) < count:
        best_idx = None
        best_score = -1.0
        for idx in remaining:
            score = min(
                diversity_distance(pool[idx], pool[j], vectors[idx], vectors[j])
                for j in selected_idx
            )
            if score > best_score:
                best_score = score
                best_idx = idx
        assert best_idx is not None
        selected_idx.append(best_idx)
        remaining.remove(best_idx)
    return [pool[i] for i in selected_idx]




def counter_to_regular_dict(counter: Counter) -> Dict[str, int]:
    return {str(k): int(v) for k, v in sorted(counter.items(), key=lambda x: str(x[0]))}


def prune_overgenerated(
    rows: Sequence[Puzzle],
    final_num: int,
    quality_fn: Callable[[Puzzle], float],
    rng: random.Random,
    desired: Dict[str, int] | None = None,
    stratify_key: str = "difficulty_bucket",
    bucket_order: Sequence[str] | None = None,
) -> List[Puzzle]:
    """Prune an overgenerated selected set down to the final size.

    The generation pipeline intentionally selects more finished puzzles than
    needed, then drops exact duplicate hashes and lower-quality examples while
    preserving the target difficulty mix as much as the selected set allows.
    """
    if desired is None:
        desired = bucket_counts(final_num) if stratify_key == "difficulty_bucket" else None
    if desired is None:
        observed = sorted({str(row["features"].get(stratify_key, row.get(stratify_key, ""))) for row in rows})
        desired = equal_bucket_counts(final_num, observed)
    deduped: List[Puzzle] = []
    seen: set[str] = set()
    for row in sorted(rows, key=lambda p: (-quality_fn(p), stable_json(p["machine_readable_instance"]))):
        h = row["features"].get("canonical_hash") or stable_hash(
            {"type": row["puzzle_type"], "instance": row["machine_readable_instance"]}
        )
        if h in seen:
            continue
        seen.add(h)
        deduped.append(row)
    by_bucket: Dict[str, List[Puzzle]] = defaultdict(list)
    for row in deduped:
        by_bucket[_stratify_value(row, stratify_key)].append(row)
    for bucket_rows in by_bucket.values():
        bucket_rows.sort(key=lambda p: (-quality_fn(p), stable_json(p["machine_readable_instance"])))
    final_counts = _rebalance_targets(
        desired,
        {k: len(v) for k, v in by_bucket.items()},
        final_num,
        bucket_order=bucket_order,
    )
    chosen: List[Puzzle] = []
    for bucket in list(bucket_order or sorted(final_counts)):
        chosen.extend(by_bucket.get(bucket, [])[: final_counts.get(bucket, 0)])
    if len(chosen) < final_num:
        chosen_ids = {id(p) for p in chosen}
        leftovers = [p for p in deduped if id(p) not in chosen_ids]
        leftovers.sort(key=lambda p: (-quality_fn(p), rng.random()))
        chosen.extend(leftovers[: final_num - len(chosen)])
    return chosen[:final_num]


def _stratify_value(row: Puzzle, stratify_key: str) -> str:
    if stratify_key in row:
        return str(row[stratify_key])
    return str(row.get("features", {}).get(stratify_key, "unknown"))
