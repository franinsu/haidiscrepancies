"""Numerical effort for the study.

Extracted from the reviewed study implementation; presentation is separate.
"""

from __future__ import annotations

import math


from collections import defaultdict


from itertools import combinations


from pathlib import Path


from statistics import mean


from typing import Any


import numpy as np


from . import main_statistics as core  # noqa: E402  (conventions, bootstrap, metrics)


from . import stimulus_modules  # noqa: E402  (design families for the stimulus bootstrap)


from . import entropy


SOURCES = ("Human", "GPT 5.6 Sol", "Claude Opus 4.8", "Gemini 3.5 Flash")


PUZZLE_TYPES = entropy.PUZZLE_TYPES


CONDITIONS = entropy.CONDITIONS


PAIR_ORDER = tuple(combinations(SOURCES, 2))


BOOTSTRAP_REPEATS = 5000


BOOTSTRAP_SEED = 0


EFFORT_UNITS = {"Human": "seconds", **{provider: "reasoning tokens" for provider in core.PROVIDERS}}


MODEL_TOKEN_FLOOR = 1.0  # applied to the per-puzzle aggregate before taking its logarithm


def load_effort(
    repro: Path,
    main_ids: set[str],
) -> dict[str, dict[str, list[float]]]:
    """Return per-source, per-puzzle lists of response-level effort."""
    efforts: dict[str, dict[str, list[float]]] = {"Human": defaultdict(list)}
    for row in core.read_jsonl(repro / "human/main_retained.jsonl"):
        puzzle_id = str(row["puzzle_id"])
        if puzzle_id not in main_ids or not row.get("is_correct"):
            continue
        if row.get("response_time_ms") in (None, ""):
            continue
        efforts["Human"][puzzle_id].append(float(row["response_time_ms"]) / 1000.0)

    requests: dict[str, dict[str, dict[str, float]]] = defaultdict(lambda: defaultdict(dict))
    for provider, provider_meta in core.PROVIDERS.items():
        inverse_conditions = {value: key for key, value in provider_meta["conditions"].items()}
        for row in core.read_jsonl(repro / "ai" / f"{provider_meta['slug']}_main.jsonl"):
            effort = inverse_conditions.get(str(row.get("api_model_condition")))
            prompt = str(row.get("prompt_condition") or "")
            if effort != core.PRIMARY_EFFORT or prompt != core.PRIMARY_PROMPT:
                continue
            puzzle_id = str(row["puzzle_id"])
            if puzzle_id not in main_ids or not row.get("is_valid"):
                continue
            token_value = row.get("hidden_reasoning_tokens")
            if token_value in (None, ""):
                token_value = row.get("reasoning_tokens")
            if token_value in (None, ""):
                token_value = row.get("thinking_tokens")
            if token_value in (None, ""):
                continue
            source = core.condition_key(provider, effort, prompt)
            request_key = str(
                row.get("request_id")
                or row.get("provider_response_id")
                or f"{row.get('sample_index')}|{row.get('response_index')}"
            )
            value = float(token_value)
            previous = requests[source][puzzle_id].get(request_key)
            if previous is not None and previous != value:
                raise ValueError(f"Conflicting effort for request {request_key}: {previous} vs {value}")
            requests[source][puzzle_id][request_key] = value
    for source, by_puzzle in requests.items():
        efforts[source] = {puzzle_id: list(values.values()) for puzzle_id, values in by_puzzle.items()}
    efforts["Human"] = dict(efforts["Human"])
    return efforts


def puzzle_summary(values: list[float], context: str, floor: float | None) -> dict[str, Any]:
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        raise ValueError(f"No effort observations for {context}")
    if np.any(array < 0):
        raise ValueError(f"Negative effort for {context}")
    median_value = float(np.median(array))
    if floor is None and np.any(array <= 0):
        raise ValueError(f"Nonpositive effort cannot be logged without a floor: {context}")
    median_for_log = median_value if floor is None else max(median_value, floor)
    return {
        "n": int(array.size),
        "n_zero": int(np.sum(array == 0)),
        "n_floored": 0 if floor is None else int(np.sum(array < floor)),
        "median": median_value,
        "log_median": math.log(median_for_log),
        "median_was_floored": floor is not None and bool(median_value < floor),
    }


def demeaned_difficulty(
    summaries: dict[str, dict[str, Any]],
    puzzle_order: list[str],
    field: str,
) -> tuple[dict[str, float], float]:
    """Demean one per-puzzle log-effort field within a source."""
    raw = {}
    for puzzle_id in puzzle_order:
        value = summaries[puzzle_id][field]
        if value is None or not math.isfinite(float(value)):
            raise ValueError(f"Undefined {field} for {puzzle_id}")
        raw[puzzle_id] = float(value)
    constant = mean(raw.values())
    return {puzzle_id: value - constant for puzzle_id, value in raw.items()}, constant


def family_summaries(
    values_by_puzzle: dict[str, float],
    puzzle_types: dict[str, str],
    draws: list[list[tuple[str, ...]]],
) -> dict[str, Any]:
    """Summarize centered puzzle values and re-center every bootstrap draw."""
    bootstrap_by_type: dict[str, list[float]] = {
        puzzle_type: [] for puzzle_type, _ in PUZZLE_TYPES
    }
    for sampled_clusters in draws:
        sampled_ids = [
            puzzle_id
            for cluster in sampled_clusters
            for puzzle_id in cluster
            if puzzle_id in values_by_puzzle
        ]
        if not sampled_ids:
            raise ValueError("A bootstrap replicate contains no core-battery puzzles")
        sampled_center = float(
            np.mean([values_by_puzzle[puzzle_id] for puzzle_id in sampled_ids])
        )
        for puzzle_type, _ in PUZZLE_TYPES:
            family_ids = [
                puzzle_id
                for puzzle_id in sampled_ids
                if puzzle_types[puzzle_id] == puzzle_type
            ]
            if not family_ids:
                raise ValueError(
                    f"A bootstrap replicate contains no {puzzle_type} puzzles"
                )
            bootstrap_by_type[puzzle_type].append(
                float(
                    np.mean([values_by_puzzle[puzzle_id] for puzzle_id in family_ids])
                    - sampled_center
                )
            )

    by_type: dict[str, Any] = {}
    for puzzle_type, _ in PUZZLE_TYPES:
        type_values = {
            puzzle_id: value
            for puzzle_id, value in values_by_puzzle.items()
            if puzzle_types[puzzle_id] == puzzle_type
        }
        values = list(type_values.values())
        q1, med, q3 = np.quantile(values, (0.25, 0.5, 0.75))
        by_type[puzzle_type] = {
            "n_puzzles": len(values),
            "values": values,
            "q1": float(q1),
            "median": float(med),
            "q3": float(q3),
            "mean": float(np.mean(values)),
            "ci95": [
                float(value)
                for value in np.quantile(
                    bootstrap_by_type[puzzle_type], (0.025, 0.975)
                )
            ],
        }
    all_values = np.asarray(list(values_by_puzzle.values()), dtype=float)
    return {
        "by_type": by_type,
        "overall": {
            "n_puzzles": int(all_values.size),
            "mean": float(np.mean([item["mean"] for item in by_type.values()])),
            "sd_across_puzzles": float(np.std(all_values, ddof=1)),
            "ci95": [0.0, 0.0],
        },
    }


def _paired_correlations(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x_centered = x - np.mean(x, axis=1, keepdims=True)
    y_centered = y - np.mean(y, axis=1, keepdims=True)
    numerator = np.sum(x_centered * y_centered, axis=1)
    denominator = np.sqrt(np.sum(x_centered**2, axis=1) * np.sum(y_centered**2, axis=1))
    result = np.full(numerator.shape, np.nan, dtype=float)
    np.divide(numerator, denominator, out=result, where=denominator > 0)
    return result


def _summary(x: np.ndarray, y: np.ndarray, bx: np.ndarray, by: np.ndarray, context: str) -> dict[str, Any]:
    estimate = float(_paired_correlations(x[np.newaxis, :], y[np.newaxis, :])[0])
    if not math.isfinite(estimate):
        raise ValueError(f"Pearson correlation is undefined for {context}")
    boots = _paired_correlations(bx, by)
    finite = boots[np.isfinite(boots)]
    if not len(finite):
        raise ValueError(f"No finite bootstrap correlations for {context}")
    low, high = np.quantile(finite, (0.025, 0.975))
    return {
        "estimate": estimate,
        "ci95": [float(low), float(high)],
        "n_puzzles": int(len(x)),
        "n_bootstrap_valid": int(len(finite)),
    }


def compute_correlations(
    values_by_source: dict[str, dict[str, float]],
    puzzle_order: list[str],
    puzzle_types: dict[str, str],
    *,
    repeats: int = None,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Paired per-puzzle Pearson correlations, mirroring the entropy analysis."""
    if repeats is None:
        repeats = BOOTSTRAP_REPEATS
    ids_by_type = {
        puzzle_type: [puzzle_id for puzzle_id in puzzle_order if puzzle_types[puzzle_id] == puzzle_type]
        for puzzle_type, _ in PUZZLE_TYPES
    }
    for puzzle_type, ids in ids_by_type.items():
        if len(ids) != 20:
            raise ValueError(f"Expected 20 {puzzle_type} puzzles, found {len(ids)}")
    vectors = {
        source: {
            puzzle_type: np.asarray([values_by_source[source][puzzle_id] for puzzle_id in ids], dtype=float)
            for puzzle_type, ids in ids_by_type.items()
        }
        for source in SOURCES
    }
    rng = np.random.default_rng(seed)
    draw_indices = rng.integers(0, 20, size=(repeats, len(PUZZLE_TYPES), 20))
    results: dict[str, Any] = {}
    for source_x, source_y in PAIR_ORDER:
        key = f"{source_x}|{source_y}"
        family: dict[str, Any] = {}
        family_boots: list[np.ndarray] = []
        pooled_x: list[np.ndarray] = []
        pooled_y: list[np.ndarray] = []
        pooled_bx: list[np.ndarray] = []
        pooled_by: list[np.ndarray] = []
        for family_index, (puzzle_type, _) in enumerate(PUZZLE_TYPES):
            x = vectors[source_x][puzzle_type]
            y = vectors[source_y][puzzle_type]
            indices = draw_indices[:, family_index, :]
            bx, by = x[indices], y[indices]
            family[puzzle_type] = _summary(x, y, bx, by, f"{key}, {puzzle_type}")
            family_boots.append(_paired_correlations(bx, by))
            pooled_x.append(x)
            pooled_y.append(y)
            pooled_bx.append(bx)
            pooled_by.append(by)
        boot_matrix = np.column_stack(family_boots)
        valid = boot_matrix[np.all(np.isfinite(boot_matrix), axis=1)]
        mean_boot = np.mean(valid, axis=1)
        low, high = np.quantile(mean_boot, (0.025, 0.975))
        results[key] = {
            "sources": [source_x, source_y],
            "mean": {
                "estimate": float(np.mean([family[puzzle_type]["estimate"] for puzzle_type, _ in PUZZLE_TYPES])),
                "ci95": [float(low), float(high)],
                "n_puzzles": len(puzzle_order),
                "n_families": len(PUZZLE_TYPES),
                "n_bootstrap_valid": int(len(mean_boot)),
            },
            "pooled": _summary(
                np.concatenate(pooled_x),
                np.concatenate(pooled_y),
                np.concatenate(pooled_bx, axis=1),
                np.concatenate(pooled_by, axis=1),
                f"{key}, pooled",
            ),
            "by_type": family,
        }
    return {
        "metric": "Pearson correlation across puzzles of paired source-demeaned log effort",
        "bootstrap": {
            "method": "pointwise paired-puzzle percentile bootstrap",
            "confidence": 0.95,
            "repeats": repeats,
            "seed": seed,
            "shared_draws_across_source_pairs": True,
            "mean": "resample 20 paired puzzles within each family, compute five correlations, average equally",
            "pooled": "same within-family draws concatenated over the five families",
            "held_fixed": "per-puzzle log-effort values; centering is irrelevant to correlation and is therefore held fixed",
        },
        "pair_order": [f"{a}|{b}" for a, b in PAIR_ORDER],
        "results": results,
    }


def compute(project: Path, repro: Path) -> dict[str, Any]:
    puzzles, blocks = core.puzzle_metadata(project)
    main_ids = [str(row["id"]) for row in core.read_jsonl(project / "stimuli/all_puzzles.jsonl")]
    puzzle_order = sorted(main_ids)
    puzzle_types = {puzzle_id: str(puzzles[puzzle_id]["puzzle_type"]) for puzzle_id in puzzle_order}
    design, _ = stimulus_modules.load_design(project)
    draws = core.cluster_bootstrap_draws(core.stimulus_clusters(main_ids, design, puzzles, set(puzzle_order)))

    efforts = load_effort(repro, set(puzzle_order))
    expected_sources = 1 + len(core.PROVIDERS)
    if len(efforts) != expected_sources:
        raise ValueError(f"Expected {expected_sources} effort sources, found {len(efforts)}")
    summaries: dict[str, dict[str, dict[str, Any]]] = {}
    for source, by_puzzle in efforts.items():
        missing = set(puzzle_order) - set(by_puzzle)
        if missing:
            raise ValueError(f"{source} lacks effort for {sorted(missing)}")
        floor = None if source == "Human" else MODEL_TOKEN_FLOOR
        summaries[source] = {
            puzzle_id: puzzle_summary(by_puzzle[puzzle_id], f"{source}, {puzzle_id}", floor)
            for puzzle_id in puzzle_order
        }

    primary_keys = {"Human": "Human"}
    primary_keys.update({provider: core.condition_key(provider, core.PRIMARY_EFFORT, core.PRIMARY_PROMPT) for provider in core.PROVIDERS})

    output: dict[str, Any] = {
        "definition": {
            "response_effort": {
                "Human": "elapsed response time in seconds, correct responses only",
                "models": "provider-reported reasoning tokens (hidden_reasoning_tokens, else reasoning_tokens, else thinking_tokens), valid token-bearing responses only",
            },
            "puzzle_difficulty": "Human: log median elapsed time over correct timed responses to a puzzle; models: log median reported reasoning-token count over valid token-bearing responses to a puzzle",
            "floor": {
                "human_seconds": None,
                "model_tokens": MODEL_TOKEN_FLOOR,
                "note": "Human times are not floored. Model per-puzzle medians are floored at one reasoning token before the log; the floor does not bind in the low-effort/plain primary analysis, and binds for two GPT low-effort/persona puzzles, one medium-effort/persona puzzle and one medium-effort/plain puzzle in the condition appendix",
            },
            "zero_token_note": "GPT reports zero reasoning tokens on some responses; on a few Arithmetic puzzles it does so for most or all responses in a cell, so a per-puzzle median or mean can be exactly zero and neither could be logged without a floor",
            "source_demeaning": "Human: subtract the mean across the 100 core puzzles of log per-puzzle median time; models: subtract the mean across the same puzzles of log per-puzzle median tokens; each provider effort/prompt cell is centered separately",
            "unit_invariance": "multiplying every effort by a constant adds a constant to every log and cancels after demeaning",
            "scale_note": "demeaning removes the level only; dispersion across puzzles remains source-specific and is reported as an across-puzzle standard deviation",
        },
        "primary_condition": {"effort": "low", "prompt": "plain", "aggregation": "none"},
        "bootstrap": {
            "method": "design-stratified percentile cluster bootstrap",
            "confidence": 0.95,
            "repeats": BOOTSTRAP_REPEATS,
            "seed": BOOTSTRAP_SEED,
            "cluster_unit": "core-battery puzzle",
            "centering": "the source mean is recomputed inside every bootstrap draw before family contrasts are formed",
            "held_fixed": "the empirical per-puzzle effort summaries",
        },
        "puzzle_order": puzzle_order,
        "puzzle_types": puzzle_types,
        "source_keys": primary_keys,
        "response_counts": {
            source: {
                "responses": int(sum(item["n"] for item in by_puzzle.values())),
                "zero_effort_responses": int(sum(item["n_zero"] for item in by_puzzle.values())),
                "min_per_puzzle": int(min(item["n"] for item in by_puzzle.values())),
                "max_per_puzzle": int(max(item["n"] for item in by_puzzle.values())),
                "zero_median_puzzles": int(sum(item["median"] == 0 for item in by_puzzle.values())),
                "floored_median_puzzles": int(sum(item["median_was_floored"] for item in by_puzzle.values())),
            }
            for source, by_puzzle in summaries.items()
        },
        "puzzle_summaries": summaries,
        "primary": {},
    }

    # Primary cell: Human plus low/plain for every provider.
    values_by_source: dict[str, dict[str, float]] = {}
    for display, source_key in primary_keys.items():
        field = "log_median"
        values, constant = demeaned_difficulty(summaries[source_key], puzzle_order, field)
        values_by_source[display] = values
        summary = family_summaries(values, puzzle_types, draws)
        summary["removed_constant"] = {
            "summary_field": field,
            "mean_log_effort": constant,
            "geometric_mean_effort": math.exp(constant),
            "units": EFFORT_UNITS[display],
        }
        summary["values_by_puzzle"] = values
        output["primary"][display] = summary

    by_type_for_svd = {
        puzzle_type: {"sources": {source: {"mean": output["primary"][source]["by_type"][puzzle_type]["mean"]} for source in SOURCES}}
        for puzzle_type, _ in PUZZLE_TYPES
    }
    projection = entropy.compute_source_svd_projection(by_type_for_svd)
    projection.update(
        {
            "input_matrix": (
                "five family means by four sources of source-centered log "
                "puzzle effort"
            ),
            "score_definition": (
                "u_k^T L_centered[:,s] = sigma_k V[s,k]"
            ),
            "centering": (
                "subtract each source column mean across the five equally "
                "weighted family means (zero by construction up to rounding)"
            ),
        }
    )
    output["source_svd_projection"] = projection
    output["correlations"] = compute_correlations(values_by_source, puzzle_order, puzzle_types)

    return output
