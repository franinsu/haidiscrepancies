"""Numerical main statistics for the study.

Extracted from the reviewed study implementation; presentation is separate.
"""

from __future__ import annotations

import json


import math


from collections import Counter, defaultdict


from pathlib import Path


from statistics import mean


from typing import Any, Callable, Iterable


import numpy as np
from scipy.stats import pearsonr


from . import resampling, stimulus_modules


from .metrics import entropy, hellinger, js_distance, tv


PROVIDERS = {
    "GPT 5.6 Sol": {
        "slug": "openai",
        "conditions": {"low": "OAI_GPT56_SOL_LOW", "medium": "OAI_GPT56_SOL_MEDIUM"},
    },
    "Claude Opus 4.8": {
        "slug": "anthropic",
        "conditions": {"low": "ANT_OPUS48_LOW", "medium": "ANT_OPUS48_MEDIUM"},
    },
    "Gemini 3.5 Flash": {
        "slug": "gemini",
        "conditions": {"low": "GEMINI_35_FLASH_LOW", "medium": "GEMINI_35_FLASH_MEDIUM"},
    },
}


COMMON_DISTRIBUTIONAL_SOURCE_ORDER = ("Human", *PROVIDERS)


COMMON_DISTRIBUTIONAL_LABEL_ORDER = ("Human", "GPT", "Claude", "Gemini")


PROMPTS = {"direct_solve": "plain", "human_participant": "as-human"}


PRIMARY_EFFORT = "low"


PRIMARY_PROMPT = "direct_solve"


PRIMARY_PROMPT_LABEL = PROMPTS[PRIMARY_PROMPT]


TYPE_ORDER = ("arithmetic24", "maze", "grid_placement", "minesweeper_lite", "mini_sudoku")


TYPE_LABELS = {
    "arithmetic24": "Arithmetic",
    "maze": "Maze",
    "grid_placement": "Rooks",
    "minesweeper_lite": "Minesweeper",
    "mini_sudoku": "Sudoku",
}


TV_BOOTSTRAP_REPEATS = 5000


TV_BOOTSTRAP_SEED = 0


LOG_DENSITY_PSEUDOCOUNT = 0.5


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def tie_aware_modal_agreement(
    left: Counter[str] | dict[str, float],
    right: Counter[str] | dict[str, float],
) -> float:
    """Return one when two nonempty distributions share an empirical mode.

    The full argmax set is retained on each side.  Thus a tied mode agrees
    whenever at least one of its maximizers is also a maximizer on the other
    side; ties never create a mismatch merely because one representative was
    chosen arbitrarily.
    """
    if not left or sum(left.values()) <= 0:
        raise ValueError("Modal agreement requires a nonempty left sample")
    if not right or sum(right.values()) <= 0:
        raise ValueError("Modal agreement requires a nonempty right sample")
    left_maximum = max(left.values())
    right_maximum = max(right.values())
    left_modes = {key for key, value in left.items() if value == left_maximum}
    right_modes = {key for key, value in right.items() if value == right_maximum}
    return float(bool(left_modes & right_modes))


def q(values: list[float], probability: float) -> float:
    return float(np.quantile(values, probability, method="linear"))


def pearson(left: list[float], right: list[float]) -> float:
    if len(set(left)) < 2 or len(set(right)) < 2:
        return float("nan")
    return float(pearsonr(left, right).statistic)


def condition_key(provider: str, effort: str, prompt: str) -> str:
    return f"{provider}|{effort}|{PROMPTS[prompt]}"


def puzzle_metadata(project: Path) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    rows = list(read_jsonl(project / "stimuli/all_puzzles.jsonl"))
    rows += list(read_jsonl(project / "stimuli/modules/all_module_trials.jsonl"))
    blocks = list(read_jsonl(project / "stimuli/modules/module_blocks.jsonl"))
    return {row["id"]: row for row in rows}, blocks


def stimulus_clusters(
    main_puzzle_ids: list[str],
    design: stimulus_modules.Design,
    puzzles: dict[str, dict[str, Any]],
    included_puzzle_ids: set[str],
) -> dict[tuple[str, str, str], list[tuple[str, ...]]]:
    """Group related trials for the design-stratified stimulus bootstrap."""
    clusters: dict[tuple[str, str], set[str]] = defaultdict(set)
    cluster_strata: dict[tuple[str, str], tuple[str, str, str]] = {}

    for puzzle_id in main_puzzle_ids:
        if puzzle_id not in included_puzzle_ids:
            continue
        key = ("main", puzzle_id)
        clusters[key].add(puzzle_id)
        cluster_strata[key] = ("main", "main", str(puzzles[puzzle_id]["puzzle_type"]))

    # Module trials cluster by stimulus family (baseline plus its perturbed
    # versions, or the prime/alone sequences of one target), as laid out in
    # the design files rather than inferred from trial-id spelling.
    for module_families in design.families.values():
        for family in module_families:
            key = (family.module, family.family_id)
            cluster_strata[key] = ("module", family.module, family.puzzle_type)
            clusters[key].update(trial.id for trial in family.trials if trial.id in included_puzzle_ids)

    assigned: dict[str, tuple[str, str]] = {}
    by_stratum: dict[tuple[str, str, str], list[tuple[str, ...]]] = defaultdict(list)
    for key, puzzle_ids in sorted(clusters.items()):
        if not puzzle_ids:
            continue
        for puzzle_id in puzzle_ids:
            if puzzle_id in assigned:
                raise ValueError(f"Puzzle {puzzle_id} belongs to both {assigned[puzzle_id]} and {key}")
            assigned[puzzle_id] = key
        by_stratum[cluster_strata[key]].append(tuple(sorted(puzzle_ids)))

    missing = included_puzzle_ids - set(assigned)
    extra = set(assigned) - included_puzzle_ids
    if missing or extra:
        raise ValueError(f"Bootstrap cluster coverage mismatch; missing={sorted(missing)}, extra={sorted(extra)}")
    return {stratum: sorted(values) for stratum, values in sorted(by_stratum.items())}


def cluster_bootstrap_draws(
    clusters_by_stratum: dict[tuple[str, str, str], list[tuple[str, ...]]],
    repeats: int = None,
    seed: int = TV_BOOTSTRAP_SEED,
) -> list[list[tuple[str, ...]]]:
    """Sample stimulus families within fixed design strata."""
    if repeats is None:
        repeats = TV_BOOTSTRAP_REPEATS
    return resampling.cluster_draws(clusters_by_stratum, repeats=repeats, seed=seed)


def clustered_mean_ci(
    values_by_puzzle: dict[str, float],
    draws: list[list[tuple[str, ...]]],
) -> list[float]:
    """Percentile interval for a trial-weighted mean under cluster resampling."""
    bootstrap_means: list[float] = []
    for sampled_clusters in draws:
        total = 0.0
        count = 0
        for cluster in sampled_clusters:
            for puzzle_id in cluster:
                if puzzle_id in values_by_puzzle:
                    total += values_by_puzzle[puzzle_id]
                    count += 1
        if not count:
            raise ValueError("A bootstrap replicate contains no observations for the requested estimand")
        bootstrap_means.append(total / count)
    return [q(bootstrap_means, 0.025), q(bootstrap_means, 0.975)]


def family_balanced_metric_summary(
    values_by_puzzle: dict[str, float],
    puzzles: dict[str, dict[str, Any]],
    draws: list[list[tuple[str, ...]]],
) -> dict[str, Any]:
    """Summarize puzzle values by family and as an equal-family mean."""
    by_type: dict[str, dict[str, Any]] = {}
    family_means: list[float] = []
    for puzzle_type in TYPE_ORDER:
        type_values = {
            puzzle_id: value
            for puzzle_id, value in values_by_puzzle.items()
            if puzzles[puzzle_id]["puzzle_type"] == puzzle_type
        }
        if not type_values:
            raise ValueError(f"No values for puzzle family {puzzle_type}")
        type_mean = mean(type_values.values())
        family_means.append(type_mean)
        by_type[puzzle_type] = {
            "estimate": type_mean,
            "ci95": clustered_mean_ci(type_values, draws),
            "n_puzzles": len(type_values),
        }
    return {
        "estimate": mean(family_means),
        "ci95": clustered_mean_ci(values_by_puzzle, draws),
        "n_puzzles": len(values_by_puzzle),
        "puzzle_values": values_by_puzzle,
        "by_type": by_type,
    }


def primary_pairwise_metric(
    metric: Callable[
        [Counter[str] | dict[str, float], Counter[str] | dict[str, float]],
        float,
    ],
    puzzles: dict[str, dict[str, Any]],
    analysis_puzzles: list[str],
    counters: dict[str, dict[str, Counter[str]]],
    primary_source_lookup: dict[str, str],
    bootstrap_draws: list[list[tuple[str, ...]]],
    bootstrap_draws_by_type: dict[str, list[list[tuple[str, ...]]]],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Pairwise primary-condition metric overall and by puzzle family."""
    sources = ["Uniform", "Human", *PROVIDERS]

    def puzzle_distribution(
        source: str,
        puzzle_id: str,
    ) -> Counter[str] | dict[str, float]:
        if source != "Uniform":
            return counters[primary_source_lookup.get(source, source)][puzzle_id]
        solution_ids = {
            str(solution.get("abstract_solution_id") or solution.get("solution_id"))
            for solution in puzzles[puzzle_id]["solutions"]
        }
        return {solution_id: 1 / len(solution_ids) for solution_id in solution_ids}

    def summarize(
        puzzle_ids: list[str],
        draws: list[list[tuple[str, ...]]],
    ) -> dict[str, Any]:
        matrix: list[list[float]] = []
        ci95_matrix: list[list[list[float] | None]] = []
        for left_index, left in enumerate(sources):
            matrix_row: list[float] = []
            ci95_row: list[list[float] | None] = []
            for right_index, right in enumerate(sources):
                if left_index == right_index:
                    matrix_row.append(0.0)
                    ci95_row.append(None)
                    continue
                if right_index < left_index:
                    matrix_row.append(matrix[right_index][left_index])
                    ci95_row.append(ci95_matrix[right_index][left_index])
                    continue
                values_by_puzzle = {
                    puzzle_id: metric(
                        puzzle_distribution(left, puzzle_id),
                        puzzle_distribution(right, puzzle_id),
                    )
                    for puzzle_id in puzzle_ids
                }
                matrix_row.append(mean(values_by_puzzle.values()))
                ci95_row.append(clustered_mean_ci(values_by_puzzle, draws))
            matrix.append(matrix_row)
            ci95_matrix.append(ci95_row)
        return {
            "order": sources,
            "matrix": matrix,
            "ci95_matrix": ci95_matrix,
            "n_puzzles": len(puzzle_ids),
        }

    global_result = summarize(analysis_puzzles, bootstrap_draws)
    by_type_result: dict[str, dict[str, Any]] = {}
    for puzzle_type in TYPE_ORDER:
        puzzle_ids = [
            puzzle_id
            for puzzle_id in analysis_puzzles
            if puzzles[puzzle_id]["puzzle_type"] == puzzle_type
        ]
        by_type_result[puzzle_type] = summarize(
            puzzle_ids,
            bootstrap_draws_by_type[puzzle_type],
        )
    return global_result, by_type_result


def smoothed_log_density_correlation(
    left: Counter[str],
    right: Counter[str],
    solution_ids: list[str],
    *,
    pseudocount: float = LOG_DENSITY_PSEUDOCOUNT,
) -> float:
    """Pearson correlation across a puzzle's smoothed log densities."""
    if pseudocount <= 0.0:
        raise ValueError("Log-density correlation requires a positive pseudocount")
    if len(solution_ids) < 2:
        raise ValueError("Log-density correlation requires at least two solutions")
    left_total = sum(left.values())
    right_total = sum(right.values())
    if left_total <= 0 or right_total <= 0:
        raise ValueError("Log-density correlation requires two nonempty samples")
    left_denominator = left_total + pseudocount * len(solution_ids)
    right_denominator = right_total + pseudocount * len(solution_ids)
    left_log_density = [
        math.log((left[solution_id] + pseudocount) / left_denominator)
        for solution_id in solution_ids
    ]
    right_log_density = [
        math.log((right[solution_id] + pseudocount) / right_denominator)
        for solution_id in solution_ids
    ]
    estimate = pearson(left_log_density, right_log_density)
    if not math.isfinite(estimate):
        raise ValueError("Log-density correlation is undefined for a constant vector")
    return estimate


def primary_log_density_correlations(
    puzzles: dict[str, dict[str, Any]],
    analysis_puzzles: list[str],
    counters: dict[str, dict[str, Counter[str]]],
    primary_source_lookup: dict[str, str],
    bootstrap_draws: list[list[tuple[str, ...]]],
    bootstrap_draws_by_type: dict[str, list[list[tuple[str, ...]]]],
) -> dict[str, Any]:
    """Mean within-puzzle log-density correlations, overall and by family."""
    sources = ["Human", *PROVIDERS]
    pair_results: list[dict[str, Any]] = []
    for left_index, left in enumerate(sources):
        for right in sources[left_index + 1 :]:
            left_source = primary_source_lookup.get(left, left)
            right_source = primary_source_lookup.get(right, right)
            values_by_puzzle: dict[str, float] = {}
            for puzzle_id in analysis_puzzles:
                solution_ids = sorted({
                    str(
                        solution.get("abstract_solution_id")
                        or solution.get("solution_id")
                    )
                    for solution in puzzles[puzzle_id]["solutions"]
                })
                values_by_puzzle[puzzle_id] = smoothed_log_density_correlation(
                    counters[left_source][puzzle_id],
                    counters[right_source][puzzle_id],
                    solution_ids,
                )

            by_type: dict[str, dict[str, Any]] = {}
            family_means: list[float] = []
            for puzzle_type in TYPE_ORDER:
                type_values = {
                    puzzle_id: value
                    for puzzle_id, value in values_by_puzzle.items()
                    if puzzles[puzzle_id]["puzzle_type"] == puzzle_type
                }
                type_mean = mean(type_values.values())
                family_means.append(type_mean)
                by_type[puzzle_type] = {
                    "n_puzzles": len(type_values),
                    "mean": type_mean,
                    "ci95": clustered_mean_ci(
                        type_values,
                        bootstrap_draws_by_type[puzzle_type],
                    ),
                }

            pair_results.append({
                "left": left,
                "right": right,
                "puzzle_values": values_by_puzzle,
                "overall": {
                    "n_puzzles": len(values_by_puzzle),
                    "mean": mean(family_means),
                    "ci95": clustered_mean_ci(values_by_puzzle, bootstrap_draws),
                },
                "by_type": by_type,
            })

    return {
        "condition": {
            "effort": PRIMARY_EFFORT,
            "prompt": PRIMARY_PROMPT_LABEL,
            "aggregation": "none",
        },
        "source_order": sources,
        "uniform_included": False,
        "uniform_exclusion": (
            "the uniform log-density vector is constant within every puzzle, "
            "so its Pearson correlation is undefined"
        ),
        "definition": (
            "Pearson correlation across valid solution classes of two "
            "half-count-smoothed empirical log densities, computed within puzzle"
        ),
        "smoothing": {
            "method": "symmetric Dirichlet half-count",
            "alpha_per_class": LOG_DENSITY_PSEUDOCOUNT,
            "formula": "(N_tsj + 0.5) / (N_ts + 0.5 k_t)",
        },
        "log_base": "natural; immaterial to Pearson correlation",
        "aggregation": {
            "within_family": "arithmetic mean of 20 puzzle-level correlations",
            "overall": "equal-weight arithmetic mean of the five family means",
        },
        "bootstrap": {
            "method": "design-stratified percentile cluster bootstrap",
            "confidence": 0.95,
            "repeats": TV_BOOTSTRAP_REPEATS,
            "seed": TV_BOOTSTRAP_SEED,
            "resampling_unit": "core-battery puzzle",
            "strata": ["puzzle_type"],
        },
        "pairs": pair_results,
    }


def global_tv_response_bootstrap_cis(
    puzzles: dict[str, dict[str, Any]],
    puzzle_ids: list[str],
    display_sources: list[str],
    source_lookup: dict[str, str],
    puzzle_draws: list[list[tuple[str, ...]]],
    human_trials_by_participant: dict[str, dict[str, str | None]],
    model_trials_by_source: dict[str, dict[str, list[str | None]]],
    *,
    metric_name: str = "total_variation",
    repeats: int = None,
    seed: int = TV_BOOTSTRAP_SEED,
) -> dict[str, list[list[list[float] | None]]]:
    """Bootstrap a mean same-puzzle distance over trials and over both levels.

    The trial-only bootstrap holds the balanced puzzle battery fixed.  It
    resamples retained Human participants as complete response profiles and
    original model requests within source--puzzle cells, then reapplies the
    correct/valid filters.  The joint bootstrap crosses those trial draws with
    the existing family-stratified puzzle multiplicities.  Uniform is fixed,
    and one resampled source--puzzle distribution is shared by every pairwise
    table cell within a replicate.
    """
    if repeats is None:
        repeats = TV_BOOTSTRAP_REPEATS
    if metric_name not in {
        "total_variation",
        "jensen_shannon_distance",
        "hellinger_distance",
    }:
        raise ValueError(f"Unknown response-bootstrap metric: {metric_name}")
    if len(puzzle_draws) != repeats:
        raise ValueError(
            f"Expected {repeats} puzzle-bootstrap draws, found {len(puzzle_draws)}"
        )
    if "Uniform" not in display_sources or "Human" not in display_sources:
        raise ValueError("Global-TV bootstrap requires Uniform and Human")
    model_display_sources = [
        source for source in display_sources if source not in {"Uniform", "Human"}
    ]
    pairs = [
        (left, right)
        for left in range(len(display_sources))
        for right in range(left + 1, len(display_sources))
    ]
    rng = np.random.default_rng(seed)

    solution_ids_by_puzzle: dict[str, list[str]] = {}
    for puzzle_id in puzzle_ids:
        solution_ids = sorted(
            {
                str(solution.get("abstract_solution_id") or solution.get("solution_id"))
                for solution in puzzles[puzzle_id]["solutions"]
            }
        )
        if not solution_ids:
            raise ValueError(f"Puzzle {puzzle_id} has no valid solution classes")
        solution_ids_by_puzzle[puzzle_id] = solution_ids

    participant_ids = sorted(human_trials_by_participant)
    if not participant_ids:
        raise ValueError("No retained Human participants for trial bootstrap")
    expected_puzzles = set(puzzle_ids)
    for participant_id in participant_ids:
        observed = set(human_trials_by_participant[participant_id])
        if observed != expected_puzzles:
            raise ValueError(
                f"Human participant {participant_id} has {len(observed)} rather than "
                f"{len(expected_puzzles)} core-battery trials"
            )
    for display_source in model_display_sources:
        source = source_lookup[display_source]
        if set(model_trials_by_source[source]) != expected_puzzles:
            raise ValueError(
                f"Model source {source} does not cover the full core battery"
            )
        request_counts = {
            len(model_trials_by_source[source][puzzle_id])
            for puzzle_id in puzzle_ids
        }
        if request_counts != {100}:
            raise ValueError(
                f"Model source {source} has unexpected per-puzzle request counts: "
                f"{sorted(request_counts)}"
            )

    # Redrawing cannot recover a cell that has no valid original responses.
    for puzzle_id in puzzle_ids:
        if not any(
            human_trials_by_participant[participant_id][puzzle_id] is not None
            for participant_id in participant_ids
        ):
            raise ValueError(
                f"No valid responses for required primary cell: Human, {puzzle_id}"
            )
        for display_source in model_display_sources:
            source = source_lookup[display_source]
            if not any(
                outcome is not None
                for outcome in model_trials_by_source[source][puzzle_id]
            ):
                raise ValueError(
                    f"No valid responses for required primary cell: {source}, {puzzle_id}"
                )

    # Convert the family-stratified puzzle draws to product-bootstrap weights.
    # A repeated puzzle gets greater multiplicity, but reuses the same trial
    # resample rather than being assigned a fictitious independent dataset.
    puzzle_index = {puzzle_id: index for index, puzzle_id in enumerate(puzzle_ids)}
    puzzle_multiplicities = np.zeros((repeats, len(puzzle_ids)), dtype=np.int16)
    for repeat, sampled_clusters in enumerate(puzzle_draws):
        for cluster in sampled_clusters:
            for puzzle_id in cluster:
                if puzzle_id not in puzzle_index:
                    raise ValueError(f"Joint bootstrap selected unknown puzzle {puzzle_id}")
                puzzle_multiplicities[repeat, puzzle_index[puzzle_id]] += 1
    if not np.all(puzzle_multiplicities.sum(axis=1) == len(puzzle_ids)):
        raise ValueError(
            "Joint bootstrap does not preserve the 100-puzzle balanced battery size"
        )

    def rowwise_distance(left: np.ndarray, right: np.ndarray) -> np.ndarray:
        if metric_name == "total_variation":
            return 0.5 * np.abs(left - right).sum(axis=1)
        if metric_name == "hellinger_distance":
            return np.sqrt(
                0.5 * ((np.sqrt(left) - np.sqrt(right)) ** 2).sum(axis=1)
            )
        midpoint = 0.5 * (left + right)
        left_term = np.zeros_like(left)
        right_term = np.zeros_like(right)
        left_mask = left > 0
        right_mask = right > 0
        left_term[left_mask] = left[left_mask] * np.log2(
            left[left_mask] / midpoint[left_mask]
        )
        right_term[right_mask] = right[right_mask] * np.log2(
            right[right_mask] / midpoint[right_mask]
        )
        return np.sqrt(0.5 * (left_term.sum(axis=1) + right_term.sum(axis=1)))

    def draw_trial_worlds(size: int) -> tuple[
        dict[tuple[int, int], np.ndarray], np.ndarray
    ]:
        """Draw complete response worlds; flag worlds with an empty denominator."""
        participant_weights = rng.multinomial(
            len(participant_ids),
            np.full(len(participant_ids), 1.0 / len(participant_ids)),
            size=size,
        )
        pair_values = {
            pair: np.zeros((size, len(puzzle_ids)), dtype=float) for pair in pairs
        }
        invalid_world = np.zeros(size, dtype=bool)

        for puzzle_position, puzzle_id in enumerate(puzzle_ids):
            solution_ids = solution_ids_by_puzzle[puzzle_id]
            solution_index = {
                solution_id: index for index, solution_id in enumerate(solution_ids)
            }
            distributions: dict[str, np.ndarray] = {
                "Uniform": np.full(
                    (size, len(solution_ids)), 1.0 / len(solution_ids)
                )
            }

            # One participant draw is used for every puzzle, preserving the
            # crossed 104-by-100 Human response design.
            human_indicators = np.zeros(
                (len(participant_ids), len(solution_ids)), dtype=np.int16
            )
            for participant_position, participant_id in enumerate(participant_ids):
                outcome = human_trials_by_participant[participant_id][puzzle_id]
                if outcome is None:
                    continue
                if outcome not in solution_index:
                    raise ValueError(
                        f"Unknown Human solution class for {puzzle_id}: {outcome}"
                    )
                human_indicators[
                    participant_position, solution_index[outcome]
                ] = 1
            human_counts = participant_weights @ human_indicators
            human_denominator = human_counts.sum(axis=1)
            invalid_world |= human_denominator == 0
            distributions["Human"] = np.divide(
                human_counts,
                human_denominator[:, None],
                out=np.zeros_like(human_counts, dtype=float),
                where=human_denominator[:, None] > 0,
            )

            # For models, sampling raw requests is equivalent to a multinomial
            # draw over valid solution classes plus one invalid-response class.
            for display_source in model_display_sources:
                source = source_lookup[display_source]
                outcomes = model_trials_by_source[source][puzzle_id]
                if not outcomes:
                    raise ValueError(f"No model requests for {source}, {puzzle_id}")
                raw_counts = np.zeros(len(solution_ids) + 1, dtype=int)
                for outcome in outcomes:
                    if outcome is None:
                        raw_counts[-1] += 1
                    elif outcome in solution_index:
                        raw_counts[solution_index[outcome]] += 1
                    else:
                        raise ValueError(
                            f"Unknown model solution class for {source}, "
                            f"{puzzle_id}: {outcome}"
                        )
                sampled_counts = rng.multinomial(
                    len(outcomes), raw_counts / len(outcomes), size=size
                )[:, :-1]
                model_denominator = sampled_counts.sum(axis=1)
                invalid_world |= model_denominator == 0
                distributions[display_source] = np.divide(
                    sampled_counts,
                    model_denominator[:, None],
                    out=np.zeros_like(sampled_counts, dtype=float),
                    where=model_denominator[:, None] > 0,
                )

            for pair in pairs:
                left, right = pair
                pair_values[pair][:, puzzle_position] = rowwise_distance(
                    distributions[display_sources[left]],
                    distributions[display_sources[right]],
                )
        return pair_values, invalid_world

    # A zero retained denominator is extraordinarily unlikely here, but it is
    # not a valid conditional distribution.  Discard that complete bootstrap
    # world and redraw every participant/request cell for the same replicate.
    accepted = {
        pair: np.zeros((repeats, len(puzzle_ids)), dtype=float) for pair in pairs
    }
    pending = np.arange(repeats)
    while len(pending):
        candidate_values, invalid_world = draw_trial_worlds(len(pending))
        valid_positions = np.flatnonzero(~invalid_world)
        for pair in pairs:
            accepted[pair][pending[valid_positions], :] = candidate_values[pair][
                valid_positions, :
            ]
        pending = pending[np.flatnonzero(invalid_world)]

    def interval_matrix(
        values_by_pair: dict[tuple[int, int], np.ndarray],
        *,
        puzzle_weights: np.ndarray | None,
    ) -> list[list[list[float] | None]]:
        matrix: list[list[list[float] | None]] = [
            [None for _ in display_sources] for _ in display_sources
        ]
        for (left, right), puzzle_values in values_by_pair.items():
            if puzzle_weights is None:
                values = puzzle_values.mean(axis=1)
            else:
                values = (
                    puzzle_values * puzzle_weights
                ).sum(axis=1) / puzzle_weights.sum(axis=1)
            interval = [
                q(values.tolist(), 0.025),
                q(values.tolist(), 0.975),
            ]
            matrix[left][right] = interval
            matrix[right][left] = interval
        return matrix

    return {
        "ci95_trial_matrix": interval_matrix(accepted, puzzle_weights=None),
        "ci95_joint_matrix": interval_matrix(
            accepted, puzzle_weights=puzzle_multiplicities.astype(float)
        ),
    }


def primary_modal_agreement_matrices(
    puzzles: dict[str, dict[str, Any]],
    puzzle_ids: list[str],
    counters: dict[str, dict[str, Counter[str]]],
    source_lookup: dict[str, str],
    puzzle_draws: list[list[tuple[str, ...]]],
    puzzle_draws_by_type: dict[str, list[list[tuple[str, ...]]]],
) -> dict[str, Any]:
    """Tie-aware modal agreement for the common four-source comparison."""
    sources = list(COMMON_DISTRIBUTIONAL_SOURCE_ORDER)

    def summarize(
        selected_puzzles: list[str],
        draws: list[list[tuple[str, ...]]],
    ) -> dict[str, Any]:
        matrix: list[list[float]] = []
        ci95_matrix: list[list[list[float] | None]] = []
        for left_index, left in enumerate(sources):
            matrix_row: list[float] = []
            ci95_row: list[list[float] | None] = []
            for right_index, right in enumerate(sources):
                if left_index == right_index:
                    matrix_row.append(1.0)
                    ci95_row.append(None)
                    continue
                if right_index < left_index:
                    matrix_row.append(matrix[right_index][left_index])
                    ci95_row.append(ci95_matrix[right_index][left_index])
                    continue
                left_source = source_lookup.get(left, left)
                right_source = source_lookup.get(right, right)
                values_by_puzzle = {
                    puzzle_id: tie_aware_modal_agreement(
                        counters[left_source][puzzle_id],
                        counters[right_source][puzzle_id],
                    )
                    for puzzle_id in selected_puzzles
                }
                matrix_row.append(mean(values_by_puzzle.values()))
                ci95_row.append(clustered_mean_ci(values_by_puzzle, draws))
            matrix.append(matrix_row)
            ci95_matrix.append(ci95_row)
        return {
            "order": sources,
            "matrix": matrix,
            "ci95_matrix": ci95_matrix,
            "ci95_puzzle_matrix": ci95_matrix,
            "n_puzzles": len(selected_puzzles),
        }

    global_result = summarize(puzzle_ids, puzzle_draws)
    by_type = {}
    for puzzle_type in TYPE_ORDER:
        family_puzzles = [
            puzzle_id
            for puzzle_id in puzzle_ids
            if puzzles[puzzle_id]["puzzle_type"] == puzzle_type
        ]
        by_type[puzzle_type] = summarize(
            family_puzzles,
            puzzle_draws_by_type[puzzle_type],
        )
    return {"global": global_result, "by_type": by_type}


def build_distribution_metric_comparison_primary(
    global_tv: dict[str, Any],
    tv_by_type: dict[str, dict[str, Any]],
    alternative_divergences: dict[str, Any],
    log_density: dict[str, Any],
    modal_agreement: dict[str, Any],
) -> dict[str, Any]:
    """Assemble a parallel four-source schema without changing legacy keys."""
    source_keys = list(COMMON_DISTRIBUTIONAL_SOURCE_ORDER)
    source_labels = list(COMMON_DISTRIBUTIONAL_LABEL_ORDER)

    def subset_matrix(
        matrix: list[list[Any]],
        order: list[str],
    ) -> list[list[Any]]:
        positions = [order.index(source) for source in source_keys]
        return [[matrix[left][right] for right in positions] for left in positions]

    def common_existing_metric(
        global_item: dict[str, Any],
        family_items: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        global_order = list(global_item["order"])
        result_global = {
            "order": source_labels,
            "matrix": subset_matrix(global_item["matrix"], global_order),
            "ci95_puzzle_matrix": subset_matrix(
                global_item.get("ci95_puzzle_matrix", global_item["ci95_matrix"]),
                global_order,
            ),
            "n_puzzles": int(global_item["n_puzzles"]),
        }
        result_by_type: dict[str, dict[str, Any]] = {}
        for puzzle_type in TYPE_ORDER:
            family_item = family_items[puzzle_type]
            family_order = list(family_item["order"])
            result_by_type[puzzle_type] = {
                "order": source_labels,
                "matrix": subset_matrix(family_item["matrix"], family_order),
                "ci95_puzzle_matrix": subset_matrix(
                    family_item.get("ci95_puzzle_matrix", family_item["ci95_matrix"]),
                    family_order,
                ),
                "n_puzzles": int(family_item["n_puzzles"]),
            }
        return {"global": result_global, "by_type": result_by_type}

    def log_density_metric() -> dict[str, Any]:
        if list(log_density["source_order"]) != source_keys:
            raise ValueError("Unexpected log-density source order")
        size = len(source_keys)
        global_matrix = [[1.0 if i == j else 0.0 for j in range(size)] for i in range(size)]
        global_ci: list[list[list[float] | None]] = [
            [None for _ in range(size)] for _ in range(size)
        ]
        family_matrices = {
            puzzle_type: [
                [1.0 if i == j else 0.0 for j in range(size)] for i in range(size)
            ]
            for puzzle_type in TYPE_ORDER
        }
        family_cis: dict[str, list[list[list[float] | None]]] = {
            puzzle_type: [[None for _ in range(size)] for _ in range(size)]
            for puzzle_type in TYPE_ORDER
        }
        observed_pairs: set[tuple[int, int]] = set()
        for pair in log_density["pairs"]:
            left = source_keys.index(pair["left"])
            right = source_keys.index(pair["right"])
            key = (min(left, right), max(left, right))
            if key in observed_pairs:
                raise ValueError(f"Duplicate log-density pair: {pair['left']}, {pair['right']}")
            observed_pairs.add(key)
            estimate = float(pair["overall"]["mean"])
            interval = [float(value) for value in pair["overall"]["ci95"]]
            global_matrix[left][right] = global_matrix[right][left] = estimate
            global_ci[left][right] = global_ci[right][left] = interval
            for puzzle_type in TYPE_ORDER:
                family = pair["by_type"][puzzle_type]
                family_estimate = float(family["mean"])
                family_interval = [float(value) for value in family["ci95"]]
                family_matrices[puzzle_type][left][right] = (
                    family_matrices[puzzle_type][right][left]
                ) = family_estimate
                family_cis[puzzle_type][left][right] = (
                    family_cis[puzzle_type][right][left]
                ) = family_interval
        expected_pairs = {
            (left, right)
            for left in range(size)
            for right in range(left + 1, size)
        }
        if observed_pairs != expected_pairs:
            raise ValueError("Incomplete log-density source-pair inventory")
        return {
            "global": {
                "order": source_labels,
                "matrix": global_matrix,
                "ci95_puzzle_matrix": global_ci,
                "n_puzzles": 100,
            },
            "by_type": {
                puzzle_type: {
                    "order": source_labels,
                    "matrix": family_matrices[puzzle_type],
                    "ci95_puzzle_matrix": family_cis[puzzle_type],
                    "n_puzzles": int(
                        next(iter(log_density["pairs"]))["by_type"][puzzle_type][
                            "n_puzzles"
                        ]
                    ),
                }
                for puzzle_type in TYPE_ORDER
            },
        }

    def modal_metric() -> dict[str, Any]:
        result = {
            "global": {
                "order": source_labels,
                "matrix": modal_agreement["global"]["matrix"],
                "ci95_puzzle_matrix": modal_agreement["global"][
                    "ci95_puzzle_matrix"
                ],
                "n_puzzles": int(modal_agreement["global"]["n_puzzles"]),
            },
            "by_type": {},
        }
        for puzzle_type in TYPE_ORDER:
            family = modal_agreement["by_type"][puzzle_type]
            result["by_type"][puzzle_type] = {
                "order": source_labels,
                "matrix": family["matrix"],
                "ci95_puzzle_matrix": family["ci95_puzzle_matrix"],
                "n_puzzles": int(family["n_puzzles"]),
            }
        return result

    alternative_metrics = alternative_divergences["metrics"]
    metrics = {
        "total_variation_distance": {
            "metadata": {
                "kind": "distance",
                "range": [0.0, 1.0],
                "preferred_direction": "lower means more similar",
                "definition": "one half of the L1 distance between empirical distributions",
                "smoothing": "none",
            },
            **common_existing_metric(global_tv, tv_by_type),
        },
        "jensen_shannon_distance": {
            "metadata": {
                "kind": "distance",
                "range": [0.0, 1.0],
                "preferred_direction": "lower means more similar",
                "definition": "square root of base-2 Jensen--Shannon divergence",
                "smoothing": "none; the empirical distance is finite at support mismatches",
            },
            **common_existing_metric(
                alternative_metrics["jensen_shannon_distance"]["global"],
                alternative_metrics["jensen_shannon_distance"]["by_type"],
            ),
        },
        "hellinger_distance": {
            "metadata": {
                "kind": "distance",
                "range": [0.0, 1.0],
                "preferred_direction": "lower means more similar",
                "definition": "root-mean-square difference between square-root probabilities",
                "smoothing": "none; the empirical distance is finite at support mismatches",
            },
            **common_existing_metric(
                alternative_metrics["hellinger_distance"]["global"],
                alternative_metrics["hellinger_distance"]["by_type"],
            ),
        },
        "log_density_correlation": {
            "metadata": {
                "kind": "similarity",
                "range": [-1.0, 1.0],
                "preferred_direction": "higher means more similar",
                "definition": (
                    "within-puzzle Pearson correlation across valid solution "
                    "classes of half-count-smoothed empirical log densities"
                ),
                "smoothing": {
                    "method": "symmetric Dirichlet half-count",
                    "alpha_per_class": LOG_DENSITY_PSEUDOCOUNT,
                    "formula": "(N_tsj + 0.5) / (N_ts + 0.5 k_t)",
                },
            },
            **log_density_metric(),
        },
        "tie_aware_modal_agreement": {
            "metadata": {
                "kind": "similarity",
                "range": [0.0, 1.0],
                "preferred_direction": "higher means more similar",
                "definition": (
                    "indicator that the two full empirical argmax sets have "
                    "nonempty intersection, averaged within family"
                ),
                "smoothing": "none",
                "ties": "retain every maximizer on each side",
            },
            **modal_metric(),
        },
    }

    for metric_name, metric in metrics.items():
        if int(metric["global"]["n_puzzles"]) != 100:
            raise ValueError(f"{metric_name} does not cover 100 core puzzles")
        for puzzle_type in TYPE_ORDER:
            if int(metric["by_type"][puzzle_type]["n_puzzles"]) != 20:
                raise ValueError(f"{metric_name}, {puzzle_type} does not cover 20 puzzles")

    return {
        "schema_version": 1,
        "condition": {
            "effort": PRIMARY_EFFORT,
            "prompt": PRIMARY_PROMPT_LABEL,
            "aggregation": "none",
        },
        "source_order": source_labels,
        "source_keys": source_keys,
        "uniform_included": False,
        "metric_order": [
            "total_variation_distance",
            "jensen_shannon_distance",
            "hellinger_distance",
            "log_density_correlation",
            "tie_aware_modal_agreement",
        ],
        "type_order": list(TYPE_ORDER),
        "type_labels": TYPE_LABELS,
        "aggregation": (
            "compute the metric within puzzle, average within each 20-puzzle "
            "family, then give the five family means equal weight"
        ),
        "bootstrap": {
            "method": "nonparametric percentile bootstrap",
            "confidence": 0.95,
            "repeats": TV_BOOTSTRAP_REPEATS,
            "seed": TV_BOOTSTRAP_SEED,
            "puzzles": "20 whole puzzles with replacement within each family",
            "coverage": "pointwise (marginal), not simultaneous",
            "shared_draws": "draws are shared across source pairs and metrics",
        },
        "metrics": metrics,
    }


def compute(data_dir, processed_dir):
    puzzles, blocks = puzzle_metadata(data_dir)
    design, _ = stimulus_modules.load_design(data_dir)
    main_puzzle_ids = [str(row['id']) for row in read_jsonl(data_dir / 'stimuli/all_puzzles.jsonl')]
    analysis_puzzles = sorted(main_puzzle_ids)
    main_set = set(main_puzzle_ids)
    human_dir = processed_dir / 'human'
    scored_dir = processed_dir / 'ai'
    counters = defaultdict(lambda: defaultdict(Counter))
    human_main_trials_by_participant = defaultdict(dict)
    model_main_trials_by_source = defaultdict(lambda: defaultdict(list))
    human_rows = 0
    for row in read_jsonl(human_dir / 'main_retained.jsonl'):
        human_rows += 1
        puzzle_id = str(row['puzzle_id'])
        if puzzle_id not in main_set:
            raise ValueError(f'Unexpected main puzzle {puzzle_id}')
        participant = str(row.get('participant_id') or '')
        if not participant or puzzle_id in human_main_trials_by_participant[participant]:
            raise ValueError(f'Missing participant or duplicate main trial: {puzzle_id}')
        solution = row.get('abstract_solution_id') or row.get('solution_id')
        valid = str(solution) if row.get('is_correct') and solution else None
        human_main_trials_by_participant[participant][puzzle_id] = valid
        if valid is not None:
            counters['Human'][puzzle_id][valid] += 1
    for provider, meta in PROVIDERS.items():
        inverse_conditions = {value: key for key, value in meta['conditions'].items()}
        for row in read_jsonl(scored_dir / (meta['slug'] + '_main.jsonl')):
            effort = inverse_conditions.get(str(row.get('api_model_condition')))
            prompt = str(row.get('prompt_condition') or '')
            if effort is None or prompt not in PROMPTS:
                continue
            puzzle_id = str(row['puzzle_id'])
            if puzzle_id not in main_set:
                raise ValueError(f'Unexpected main puzzle {puzzle_id}')
            source = condition_key(provider, effort, prompt)
            solution = row.get('abstract_solution_id') or row.get('solution_id')
            valid = str(solution) if row.get('is_valid') and solution else None
            model_main_trials_by_source[source][puzzle_id].append(valid)
            if valid is not None:
                counters[source][puzzle_id][valid] += 1
    primary_sources = {provider: condition_key(provider, PRIMARY_EFFORT, PRIMARY_PROMPT)
                       for provider in PROVIDERS}
    condition_sources = {
        provider: [condition_key(provider, effort, prompt)
                   for effort in ('low', 'medium')
                   for prompt in ('direct_solve', 'human_participant')]
        for provider in PROVIDERS
    }
    # Every displayed condition needs a defined distribution for every puzzle.
    # Check before metrics or random draws, including secondary conditions.
    for source in ['Human', *[s for values in condition_sources.values() for s in values]]:
        for puzzle_id in analysis_puzzles:
            if not counters[source][puzzle_id]:
                kind = 'primary' if source == 'Human' or source in primary_sources.values() else 'secondary'
                raise ValueError(
                    f'No valid responses for required {kind} cell: {source}, {puzzle_id}'
                )
    clusters_by_stratum = stimulus_clusters(main_puzzle_ids, design, puzzles, main_set)
    bootstrap_draws = cluster_bootstrap_draws(clusters_by_stratum)
    entropy_per_puzzle_condition = {
        source: {puzzle_id: entropy(counters[source][puzzle_id]) for puzzle_id in analysis_puzzles}
        for source in ['Human', *[s for values in condition_sources.values() for s in values]]
    }
    tv_conditions: dict[str, dict[str, Any]] = {}
    for provider, sources in condition_sources.items():
        order = ['Human', *sources]
        matrix: list[list[float]] = []
        ci95_matrix: list[list[list[float] | None]] = []
        for left_index, left in enumerate(order):
            matrix_row: list[float] = []
            ci95_row: list[list[float] | None] = []
            for right_index, right in enumerate(order):
                if left == right:
                    matrix_row.append(0.0)
                    ci95_row.append(None)
                    continue
                if right_index < left_index:
                    matrix_row.append(matrix[right_index][left_index])
                    ci95_row.append(ci95_matrix[right_index][left_index])
                    continue
                values_by_puzzle = {puzzle_id: tv(counters[left][puzzle_id], counters[right][puzzle_id]) for puzzle_id in analysis_puzzles}
                matrix_row.append(mean(values_by_puzzle.values()))
                ci95_row.append(clustered_mean_ci(values_by_puzzle, bootstrap_draws))
            matrix.append(matrix_row)
            ci95_matrix.append(ci95_row)
        tv_conditions[provider] = {'order': order, 'matrix': matrix, 'ci95_matrix': ci95_matrix, 'n_puzzles': len(analysis_puzzles)}
    tv_by_type_condition: dict[str, dict[str, dict[str, Any]]] = {}
    for sources in condition_sources.values():
        for source in sources:
            tv_by_type_condition[source] = {}
            for puzzle_type in TYPE_ORDER:
                puzzle_ids = [puzzle_id for puzzle_id in analysis_puzzles if puzzles[puzzle_id]['puzzle_type'] == puzzle_type]
                values_by_puzzle = {puzzle_id: tv(counters[source][puzzle_id], counters['Human'][puzzle_id]) for puzzle_id in puzzle_ids}
                tv_by_type_condition[source][puzzle_type] = {'estimate': mean(values_by_puzzle.values()), 'ci95': clustered_mean_ci(values_by_puzzle, bootstrap_draws), 'n_puzzles': len(puzzle_ids)}
    tv_by_type_primary = {provider: tv_by_type_condition[source] for provider, source in primary_sources.items()}
    contrast_specs = {'persona_minus_plain_low': (('low', 'human_participant'), ('low', 'direct_solve')), 'persona_minus_plain_medium': (('medium', 'human_participant'), ('medium', 'direct_solve')), 'medium_minus_low_plain': (('medium', 'direct_solve'), ('low', 'direct_solve')), 'medium_minus_low_persona': (('medium', 'human_participant'), ('low', 'human_participant'))}
    tv_condition_contrasts: dict[str, dict[str, dict[str, Any]]] = {}
    for provider in PROVIDERS:
        tv_condition_contrasts[provider] = {}
        for contrast, (positive_spec, negative_spec) in contrast_specs.items():
            positive_source = condition_key(provider, *positive_spec)
            negative_source = condition_key(provider, *negative_spec)
            deltas_by_puzzle = {puzzle_id: tv(counters[positive_source][puzzle_id], counters['Human'][puzzle_id]) - tv(counters[negative_source][puzzle_id], counters['Human'][puzzle_id]) for puzzle_id in analysis_puzzles}
            tv_condition_contrasts[provider][contrast] = family_balanced_metric_summary(deltas_by_puzzle, puzzles, bootstrap_draws)
    global_sources = ['Uniform', 'Human', *PROVIDERS]
    global_tv_by_condition: dict[str, dict[str, Any]] = {}
    for effort in ('low', 'medium'):
        for prompt, prompt_label in PROMPTS.items():
            condition_label = f'{effort}|{prompt_label}'
            source_lookup = {provider: condition_key(provider, effort, prompt) for provider in PROVIDERS}
            condition_matrix: list[list[float]] = []
            condition_ci95_matrix: list[list[list[float] | None]] = []
            for left_index, left in enumerate(global_sources):
                matrix_row: list[float] = []
                ci95_row: list[list[float] | None] = []
                for right_index, right in enumerate(global_sources):
                    if left_index == right_index:
                        matrix_row.append(0.0)
                        ci95_row.append(None)
                        continue
                    if right_index < left_index:
                        matrix_row.append(condition_matrix[right_index][left_index])
                        ci95_row.append(condition_ci95_matrix[right_index][left_index])
                        continue
                    values_by_puzzle: dict[str, float] = {}
                    for puzzle_id in analysis_puzzles:
                        solution_ids = {str(solution.get('abstract_solution_id') or solution.get('solution_id')) for solution in puzzles[puzzle_id]['solutions']}
                        uniform = {solution_id: 1 / len(solution_ids) for solution_id in solution_ids}
                        left_source = source_lookup.get(left, left)
                        right_source = source_lookup.get(right, right)
                        if left == 'Uniform':
                            value = tv(uniform, counters[right_source][puzzle_id])
                        elif right == 'Uniform':
                            value = tv(counters[left_source][puzzle_id], uniform)
                        else:
                            value = tv(counters[left_source][puzzle_id], counters[right_source][puzzle_id])
                        values_by_puzzle[puzzle_id] = value
                    matrix_row.append(mean(values_by_puzzle.values()))
                    ci95_row.append(clustered_mean_ci(values_by_puzzle, bootstrap_draws))
                condition_matrix.append(matrix_row)
                condition_ci95_matrix.append(ci95_row)
            global_tv_by_condition[condition_label] = {'order': global_sources, 'matrix': condition_matrix, 'ci95_matrix': condition_ci95_matrix, 'n_puzzles': len(analysis_puzzles), 'effort': effort, 'prompt': prompt_label}
    global_tv_primary = global_tv_by_condition[f'{PRIMARY_EFFORT}|{PRIMARY_PROMPT_LABEL}']
    primary_source_lookup = {provider: primary_sources[provider] for provider in PROVIDERS}
    primary_response_intervals = global_tv_response_bootstrap_cis(puzzles, analysis_puzzles, global_sources, primary_source_lookup, bootstrap_draws, human_main_trials_by_participant, model_main_trials_by_source)
    global_tv_primary['ci95_puzzle_matrix'] = global_tv_primary['ci95_matrix']
    global_tv_primary.update(primary_response_intervals)
    bootstrap_draws_by_type = {puzzle_type: [[cluster for cluster in sampled_clusters if puzzles[cluster[0]]['puzzle_type'] == puzzle_type] for sampled_clusters in bootstrap_draws] for puzzle_type in TYPE_ORDER}
    pairwise_tv_by_type_condition: dict[str, dict[str, dict[str, Any]]] = {}
    for effort in ('low', 'medium'):
        for prompt, prompt_label in PROMPTS.items():
            condition_label = f'{effort}|{prompt_label}'
            source_lookup = {provider: condition_key(provider, effort, prompt) for provider in PROVIDERS}
            pairwise_tv_by_type_condition[condition_label] = {}
            for puzzle_type in TYPE_ORDER:
                puzzle_ids = [puzzle_id for puzzle_id in analysis_puzzles if puzzles[puzzle_id]['puzzle_type'] == puzzle_type]
                type_matrix: list[list[float]] = []
                type_ci95_matrix: list[list[list[float] | None]] = []
                for left_index, left in enumerate(global_sources):
                    matrix_row: list[float] = []
                    ci95_row: list[list[float] | None] = []
                    for right_index, right in enumerate(global_sources):
                        if left_index == right_index:
                            matrix_row.append(0.0)
                            ci95_row.append(None)
                            continue
                        if right_index < left_index:
                            matrix_row.append(type_matrix[right_index][left_index])
                            ci95_row.append(type_ci95_matrix[right_index][left_index])
                            continue
                        values_by_puzzle: dict[str, float] = {}
                        for puzzle_id in puzzle_ids:
                            solution_ids = {str(solution.get('abstract_solution_id') or solution.get('solution_id')) for solution in puzzles[puzzle_id]['solutions']}
                            uniform = {solution_id: 1 / len(solution_ids) for solution_id in solution_ids}
                            left_source = source_lookup.get(left, left)
                            right_source = source_lookup.get(right, right)
                            if left == 'Uniform':
                                value = tv(uniform, counters[right_source][puzzle_id])
                            elif right == 'Uniform':
                                value = tv(counters[left_source][puzzle_id], uniform)
                            else:
                                value = tv(counters[left_source][puzzle_id], counters[right_source][puzzle_id])
                            values_by_puzzle[puzzle_id] = value
                        matrix_row.append(mean(values_by_puzzle.values()))
                        ci95_row.append(clustered_mean_ci(values_by_puzzle, bootstrap_draws_by_type[puzzle_type]))
                    type_matrix.append(matrix_row)
                    type_ci95_matrix.append(ci95_row)
                pairwise_tv_by_type_condition[condition_label][puzzle_type] = {'order': global_sources, 'matrix': type_matrix, 'ci95_matrix': type_ci95_matrix, 'n_puzzles': len(puzzle_ids), 'effort': effort, 'prompt': prompt_label}
    pairwise_tv_by_type_primary = pairwise_tv_by_type_condition[f'{PRIMARY_EFFORT}|{PRIMARY_PROMPT_LABEL}']
    log_density_correlation_primary = primary_log_density_correlations(puzzles, analysis_puzzles, counters, primary_source_lookup, bootstrap_draws, bootstrap_draws_by_type)
    condition_geometry_order = ['Human', *[source for provider in PROVIDERS for source in condition_sources[provider]]]
    condition_geometry_metadata: list[dict[str, str]] = [{'source': 'Human', 'provider': 'Human', 'effort': '-', 'prompt': '-'}]
    for provider in PROVIDERS:
        for effort_value in ('low', 'medium'):
            for prompt_key in ('direct_solve', 'human_participant'):
                condition_geometry_metadata.append({'source': condition_key(provider, effort_value, prompt_key), 'provider': provider, 'effort': effort_value, 'prompt': 'plain' if prompt_key == 'direct_solve' else 'persona'})
    condition_tv_matrix: list[list[float]] = []
    for left_index, left in enumerate(condition_geometry_order):
        tv_row: list[float] = []
        for right_index, right in enumerate(condition_geometry_order):
            if left_index == right_index:
                tv_row.append(0.0)
                continue
            if right_index < left_index:
                tv_row.append(condition_tv_matrix[right_index][left_index])
                continue
            tv_values: dict[str, float] = {}
            for puzzle_id in analysis_puzzles:
                tv_values[puzzle_id] = tv(counters[left][puzzle_id], counters[right][puzzle_id])
            tv_family_means = [mean((value for puzzle_id, value in tv_values.items() if puzzles[puzzle_id]['puzzle_type'] == puzzle_type)) for puzzle_type in TYPE_ORDER]
            tv_row.append(mean(tv_family_means))
        condition_tv_matrix.append(tv_row)
    condition_geometry = {'source_order': condition_geometry_order, 'source_metadata': condition_geometry_metadata, 'n_puzzles': len(analysis_puzzles), 'aggregation': 'within-puzzle metric, arithmetic mean within each of five 20-puzzle families, then equal-weight mean across families', 'tv_matrix': condition_tv_matrix, }
    alternative_divergences: dict[str, Any] = {'metadata': {'condition': 'low effort, plain prompt', 'aggregation': 'equal-weight mean of same-puzzle distances', 'bootstrap': 'same puzzle draws as the primary TV tables', 'logarithm_base': 2}, 'metrics': {}}
    for metric_name, metric_function in (('jensen_shannon_distance', js_distance), ('hellinger_distance', hellinger)):
        global_metric, by_type_metric = primary_pairwise_metric(metric_function, puzzles, analysis_puzzles, counters, primary_source_lookup, bootstrap_draws, bootstrap_draws_by_type)
        global_metric['ci95_puzzle_matrix'] = global_metric['ci95_matrix']
        alternative_divergences['metrics'][metric_name] = {'global': global_metric, 'by_type': by_type_metric}
    modal_agreement_primary = primary_modal_agreement_matrices(puzzles, analysis_puzzles, counters, primary_source_lookup, bootstrap_draws, bootstrap_draws_by_type)
    distribution_metric_comparison_primary = build_distribution_metric_comparison_primary(global_tv_primary, pairwise_tv_by_type_primary, alternative_divergences, log_density_correlation_primary, modal_agreement_primary)
    output = {
        "counts": {
            "multi_solution_puzzles": len(puzzles) - len(stimulus_modules.primer_trial_ids(design)),
            "all_trials": len(puzzles),
            "effort_single_answer_units": len(main_puzzle_ids),
            "primary_analysis_puzzles": len(analysis_puzzles),
            "human_main_rows": sum(1 for _ in read_jsonl(human_dir / "main_retained.jsonl")),
            "human_module_rows": sum(1 for _ in read_jsonl(human_dir / "module_retained.jsonl")),
        },
        "type_order": list(TYPE_ORDER),
        "type_labels": TYPE_LABELS,
        "puzzle_types": {
            puzzle_id: puzzles[puzzle_id]["puzzle_type"]
            for puzzle_id in analysis_puzzles
        },
        "main_puzzle_ids": main_puzzle_ids,
        "trial_order": analysis_puzzles,
        "primary_condition": {
            "effort": PRIMARY_EFFORT,
            "prompt": PRIMARY_PROMPT_LABEL,
            "source_keys": primary_sources,
            "aggregation": "none",
        },
        "entropy_per_puzzle_condition": entropy_per_puzzle_condition,
        "tv_conditions": tv_conditions,
        "tv_by_type_condition": tv_by_type_condition,
        "tv_by_type_primary": tv_by_type_primary,
        "tv_condition_contrasts": tv_condition_contrasts,
        "condition_source_geometry": condition_geometry,
        "tv_bootstrap": {
            "method": "design-stratified percentile cluster bootstrap",
            "confidence": 0.95,
            "repeats": TV_BOOTSTRAP_REPEATS,
            "seed": TV_BOOTSTRAP_SEED,
            "resampling_unit": "core-battery puzzle",
            "strata": ["puzzle_type"],
            "n_clusters": sum(len(clusters) for clusters in clusters_by_stratum.values()),
            "n_puzzles": len(analysis_puzzles),
        },
        "global_tv_response_bootstrap": {
            "method": "nonparametric percentile bootstrap",
            "confidence": 0.95,
            "repeats": TV_BOOTSTRAP_REPEATS,
            "seed": TV_BOOTSTRAP_SEED,
            "scope": "primary low-effort plain-prompt Table 1",
            "coverage": "pointwise (marginal), not simultaneous",
            "estimand": (
                "equal-weight mean of same-puzzle total-variation distances "
                "over the balanced 100-puzzle core battery"
            ),
            "trial_only": {
                "puzzles": "fixed observed 100-puzzle core battery",
                "human_resampling_unit": (
                    "retained participant response profile across all 100 puzzles"
                ),
                "human_sample_size": len(human_main_trials_by_participant),
                "human_filter": "correct responses retained after resampling",
                "model_resampling_unit": (
                    "original API request within model-source--puzzle cell"
                ),
                "model_sample_size": "100 requests per model-source--puzzle cell",
                "model_filter": "valid responses retained after resampling",
                "sharing": (
                    "one participant draw and one source--puzzle distribution "
                    "shared across all pairwise comparisons"
                ),
                "uniform_reference": "fixed",
            },
            "joint": {
                "puzzle_resampling": (
                    "20 whole puzzles with replacement within each puzzle family"
                ),
                "response_resampling": "same trial resampling as trial-only",
                "repeated_puzzle_occurrences": (
                    "weighted by puzzle multiplicity; trial draw reused"
                ),
                "uniform_reference": "fixed",
            },
            "zero_denominator": "redraw entire bootstrap replicate",
        },
        "global_tv_by_condition": global_tv_by_condition,
        "global_tv_primary": global_tv_primary,
        "pairwise_tv_by_type_condition": pairwise_tv_by_type_condition,
        "pairwise_tv_by_type_primary": pairwise_tv_by_type_primary,
        "log_density_correlation_primary": log_density_correlation_primary,
        "alternative_divergences_primary": alternative_divergences,
        "distribution_metric_comparison_primary": distribution_metric_comparison_primary,
    }
    return output
