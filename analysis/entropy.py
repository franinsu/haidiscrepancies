"""Numerical entropy for the study.

Extracted from the reviewed study implementation; presentation is separate.
"""

from __future__ import annotations

import math


import random


import re


from collections import defaultdict


from typing import Any


import numpy as np


SOURCES = ("Human", "GPT 5.6 Sol", "Claude Opus 4.8", "Gemini 3.5 Flash")


CONDITIONS = (
    ("low", "plain", "Low", "plain"),
    ("low", "as-human", "Low", "persona"),
    ("medium", "plain", "Medium", "plain"),
    ("medium", "as-human", "Medium", "persona"),
)


CONTRASTS = (
    (
        "persona_minus_plain_low",
        ("low", "as-human"),
        ("low", "plain"),
        r"\shortstack[l]{Persona $-$ plain\\(low effort)}",
    ),
    (
        "persona_minus_plain_medium",
        ("medium", "as-human"),
        ("medium", "plain"),
        r"\shortstack[l]{Persona $-$ plain\\(medium effort)}",
    ),
    (
        "medium_minus_low_plain",
        ("medium", "plain"),
        ("low", "plain"),
        r"\shortstack[l]{Medium $-$ low\\(plain prompt)}",
    ),
    (
        "medium_minus_low_persona",
        ("medium", "as-human"),
        ("low", "as-human"),
        r"\shortstack[l]{Medium $-$ low\\(persona prompt)}",
    ),
)


SOURCE_HEADERS = {
    "Human": "Human",
    "GPT 5.6 Sol": "ChatGPT",
    "Claude Opus 4.8": "Claude",
    "Gemini 3.5 Flash": "Gemini",
}


PUZZLE_TYPES = (
    ("arithmetic24", "Arithmetic"),
    ("maze", "Maze"),
    ("grid_placement", "Rooks"),
    ("minesweeper_lite", "Minesweeper"),
    ("mini_sudoku", "Sudoku"),
)


BOOTSTRAP_REPEATS = 5000


BOOTSTRAP_SEED = 0


def abstract_solution_count(puzzle: dict[str, Any]) -> int:
    solution_ids = {
        str(solution.get("abstract_solution_id") or solution.get("solution_id"))
        for solution in puzzle["solutions"]
    }
    if len(solution_ids) < 2:
        raise ValueError(f"Normalized entropy requires a multi-solution trial: {puzzle['id']}")
    return len(solution_ids)


def normalized_entropy(observed_entropy: float, n_solutions: int, context: str) -> float:
    """Map Shannon entropy to [0, 1] using delta and uniform extrema."""
    minimum_entropy = 0.0
    maximum_entropy = math.log2(n_solutions)
    value = (observed_entropy - minimum_entropy) / (
        maximum_entropy - minimum_entropy
    )
    if value < -1e-10 or value > 1.0 + 1e-10:
        raise ValueError(
            f"Normalized entropy outside [0, 1] for {context}: {value} "
            f"(H={observed_entropy}, k={n_solutions})"
        )
    return min(1.0, max(0.0, value))


def stimulus_clusters(
    main_puzzle_ids: list[str],
    blocks: list[dict[str, Any]],
    puzzles: dict[str, dict[str, Any]],
    included_puzzle_ids: set[str],
) -> dict[tuple[str, str, str], list[tuple[str, ...]]]:
    """Match the manuscript's independent stimulus-family bootstrap units."""
    clusters: dict[tuple[str, str], set[str]] = defaultdict(set)
    cluster_strata: dict[tuple[str, str], tuple[str, str, str]] = {}

    for puzzle_id in main_puzzle_ids:
        if puzzle_id not in included_puzzle_ids:
            continue
        key = ("main", puzzle_id)
        clusters[key].add(puzzle_id)
        cluster_strata[key] = ("main", "main", str(puzzles[puzzle_id]["puzzle_type"]))

    for block in blocks:
        module = str(block.get("module") or "")
        block_id = str(block.get("block_id") or "")
        if module == "pair":
            family_id = re.sub(r"_(?:NO_PRIME|REL|CTRL)$", "", block_id)
        elif module == "transfer":
            family_id = re.sub(r"_(?:PRIME|ALONE)$", "", block_id)
        else:
            family_id = block_id
        key = (module, family_id)
        puzzle_type = str(block.get("puzzle_type") or "")
        stratum = ("module", module, puzzle_type)
        previous = cluster_strata.setdefault(key, stratum)
        if previous != stratum:
            raise ValueError(f"Stimulus family {key} spans bootstrap strata: {previous} vs {stratum}")
        clusters[key].update(
            str(puzzle_id)
            for puzzle_id in block.get("trial_ids", [])
            if str(puzzle_id) in included_puzzle_ids
        )

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
) -> list[list[tuple[str, ...]]]:
    """Resample stimulus families within each fixed design stratum."""
    rng = random.Random(BOOTSTRAP_SEED)
    ordered_strata = sorted(clusters_by_stratum)
    draws: list[list[tuple[str, ...]]] = []
    for _ in range(BOOTSTRAP_REPEATS):
        sampled: list[tuple[str, ...]] = []
        for stratum in ordered_strata:
            clusters = clusters_by_stratum[stratum]
            sampled.extend(clusters[rng.randrange(len(clusters))] for _ in clusters)
        draws.append(sampled)
    return draws


def clustered_mean_ci(
    values_by_puzzle: dict[str, float],
    draws: list[list[tuple[str, ...]]],
) -> list[float]:
    """Percentile interval for a trial-weighted mean under cluster resampling."""
    bootstrap_means: list[float] = []
    for sampled_clusters in draws:
        sampled_values = [
            values_by_puzzle[puzzle_id]
            for cluster in sampled_clusters
            for puzzle_id in cluster
            if puzzle_id in values_by_puzzle
        ]
        if not sampled_values:
            raise ValueError("A bootstrap replicate contains no observations for the requested estimand")
        bootstrap_means.append(float(np.mean(sampled_values)))
    return [float(value) for value in np.quantile(bootstrap_means, (0.025, 0.975))]


def compute_source_svd_projection(by_type: dict[str, Any]) -> dict[str, Any]:
    """Project demeaned source profiles onto entropy-matrix directions."""
    family_order = [puzzle_type for puzzle_type, _ in PUZZLE_TYPES]
    entropy_matrix = np.asarray(
        [
            [by_type[puzzle_type]["sources"][source]["mean"] for source in SOURCES]
            for puzzle_type in family_order
        ],
        dtype=float,
    )
    source_means = np.mean(entropy_matrix, axis=0)
    centered_matrix = entropy_matrix - source_means[np.newaxis, :]
    left_vectors, singular_values, right_vectors_t = np.linalg.svd(
        centered_matrix,
        full_matrices=False,
    )

    # Singular-vector signs are arbitrary.  Orient each direction by its most
    # prominent family loading so generated tables are stable across systems.
    for component in range(len(singular_values)):
        anchor = int(np.argmax(np.abs(left_vectors[:, component])))
        if left_vectors[anchor, component] < 0:
            left_vectors[:, component] *= -1
            right_vectors_t[component, :] *= -1

    source_scores = singular_values[:, np.newaxis] * right_vectors_t
    energy_share = singular_values**2 / float(np.sum(singular_values**2))
    return {
        "matrix_shape": list(entropy_matrix.shape),
        "family_order": family_order,
        "source_order": list(SOURCES),
        "source_means": {
            source: float(source_means[index])
            for index, source in enumerate(SOURCES)
        },
        "centering": "subtract each source column mean across puzzle families",
        "centered_column_means": {
            source: float(np.mean(centered_matrix[:, index]))
            for index, source in enumerate(SOURCES)
        },
        "score_definition": "u_k^T E_centered[:,s] = sigma_k V[s,k]",
        "sign_convention": (
            "orient each left singular vector so its largest-magnitude "
            "family loading is positive"
        ),
        "singular_values": [float(value) for value in singular_values],
        "frobenius_energy_share": [float(value) for value in energy_share],
        "left_singular_vectors_top_two": {
            puzzle_type: [float(value) for value in left_vectors[index, :2]]
            for index, puzzle_type in enumerate(family_order)
        },
        "source_scores_top_two": {
            source: [float(value) for value in source_scores[:2, index]]
            for index, source in enumerate(SOURCES)
        },
    }


def compute_normalized_entropy(
    stats: dict[str, Any],
    puzzles: dict[str, dict[str, Any]],
    main_puzzle_ids: list[str],
    blocks: list[dict[str, Any]],
) -> dict[str, Any]:
    output: dict[str, Any] = {
        "definition": "(H_source - H_min) / (H_max - H_min)",
        "minimum_entropy": "0 bits (delta distribution)",
        "maximum_entropy": "log2(number of valid abstract solution classes) bits (uniform distribution)",
        "simplified_definition": "H_source / log2(number of valid abstract solution classes)",
        "range": [0.0, 1.0],
        "units": "unitless",
        "orientation": "0 = delta distribution; 1 = uniform distribution",
        "aggregation": "normalize within puzzle, then summarize across puzzles",
        "primary_condition": {
            "effort": "low",
            "prompt": "plain",
            "aggregation": "none",
        },
        "bootstrap": {
            "method": "design-stratified percentile cluster bootstrap",
            "confidence": 0.95,
            "repeats": BOOTSTRAP_REPEATS,
            "seed": BOOTSTRAP_SEED,
            "cluster_unit": "core-battery puzzle",
        },
        "by_type": {},
    }
    included_ids = sorted(main_puzzle_ids)
    if len(included_ids) != 100:
        raise ValueError(f"Expected 100 core-battery puzzles, found {len(included_ids)}")
    draws = cluster_bootstrap_draws(
        stimulus_clusters(main_puzzle_ids, blocks, puzzles, set(included_ids))
    )
    primary_values_by_source: dict[str, dict[str, float]] = {
        source: {} for source in SOURCES
    }

    for puzzle_type, _ in PUZZLE_TYPES:
        puzzle_ids = [
            puzzle_id
            for puzzle_id in included_ids
            if stats["puzzle_types"][puzzle_id] == puzzle_type
        ]
        type_result: dict[str, Any] = {"n_puzzles": len(puzzle_ids), "sources": {}}
        for source in SOURCES:
            values_by_puzzle: dict[str, float] = {}
            for puzzle_id in puzzle_ids:
                n_solutions = abstract_solution_count(puzzles[puzzle_id])
                if source == "Human":
                    observed_entropy = float(
                        stats["entropy_per_puzzle_condition"]["Human"][puzzle_id]
                    )
                else:
                    observed_entropy = float(
                        stats["entropy_per_puzzle_condition"][
                            f"{source}|low|plain"
                        ][puzzle_id]
                    )
                values_by_puzzle[puzzle_id] = normalized_entropy(
                    observed_entropy,
                    n_solutions,
                    f"{source}, {puzzle_id}",
                )
            primary_values_by_source[source].update(values_by_puzzle)
            values = list(values_by_puzzle.values())
            q1, median, q3 = np.quantile(values, (0.25, 0.5, 0.75))
            type_result["sources"][source] = {
                "values": values,
                "q1": float(q1),
                "median": float(median),
                "q3": float(q3),
                "mean": float(np.mean(values)),
                "ci95": clustered_mean_ci(values_by_puzzle, draws),
            }
        output["by_type"][puzzle_type] = type_result

    output["overall"] = {
        "n_puzzles": len(included_ids),
        "aggregation": "equal-weight arithmetic mean of the five family means",
        "sources": {},
    }
    for source in SOURCES:
        values_by_puzzle = primary_values_by_source[source]
        if set(values_by_puzzle) != set(included_ids):
            raise ValueError(f"Incomplete overall normalized entropy for {source}")
        values = [values_by_puzzle[puzzle_id] for puzzle_id in included_ids]
        q1, median, q3 = np.quantile(values, (0.25, 0.5, 0.75))
        family_means = [
            output["by_type"][puzzle_type]["sources"][source]["mean"]
            for puzzle_type, _ in PUZZLE_TYPES
        ]
        output["overall"]["sources"][source] = {
            "values": values,
            "q1": float(q1),
            "median": float(median),
            "q3": float(q3),
            "mean": float(np.mean(family_means)),
            "ci95": clustered_mean_ci(values_by_puzzle, draws),
        }

    # Preserve the puzzle-level association explicitly.  The older ``values``
    # arrays above remain for backwards compatibility, while these keyed maps
    # let downstream paired analyses verify that two sources are aligned on
    # the same puzzle rather than relying on an implicit list order.
    output["puzzle_order"] = included_ids
    output["puzzle_types"] = {
        puzzle_id: str(stats["puzzle_types"][puzzle_id])
        for puzzle_id in included_ids
    }
    output["values_by_puzzle"] = {
        source: {
            puzzle_id: float(primary_values_by_source[source][puzzle_id])
            for puzzle_id in included_ids
        }
        for source in SOURCES
    }

    output["source_svd_projection"] = compute_source_svd_projection(
        output["by_type"]
    )

    output["by_condition"] = {}
    condition_values: dict[str, dict[str, dict[str, float]]] = {}
    for provider in SOURCES[1:]:
        output["by_condition"][provider] = {}
        condition_values[provider] = {}
        for effort, prompt, effort_label, prompt_label in CONDITIONS:
            condition_label = f"{effort}|{prompt}"
            source_key = f"{provider}|{effort}|{prompt}"
            values_for_condition = {
                puzzle_id: normalized_entropy(
                    float(
                        stats["entropy_per_puzzle_condition"][source_key][puzzle_id]
                    ),
                    abstract_solution_count(puzzles[puzzle_id]),
                    f"{source_key}, {puzzle_id}",
                )
                for puzzle_id in included_ids
            }
            condition_values[provider][condition_label] = values_for_condition
            condition_result: dict[str, Any] = {
                "effort": effort_label,
                "prompt": prompt_label,
                "by_type": {},
            }
            for puzzle_type, _ in PUZZLE_TYPES:
                puzzle_ids = [
                    puzzle_id
                    for puzzle_id in included_ids
                    if stats["puzzle_types"][puzzle_id] == puzzle_type
                ]
                values_by_puzzle = {
                    puzzle_id: values_for_condition[puzzle_id]
                    for puzzle_id in puzzle_ids
                }
                values = list(values_by_puzzle.values())
                q1, median, q3 = np.quantile(values, (0.25, 0.5, 0.75))
                condition_result["by_type"][puzzle_type] = {
                    "n_puzzles": len(values),
                    "values": values,
                    "q1": float(q1),
                    "median": float(median),
                    "q3": float(q3),
                    "mean": float(np.mean(values)),
                    "ci95": clustered_mean_ci(values_by_puzzle, draws),
                }
            output["by_condition"][provider][condition_label] = condition_result

    output["condition_contrasts"] = {}
    for provider in SOURCES[1:]:
        output["condition_contrasts"][provider] = {}
        for contrast, positive, negative, _ in CONTRASTS:
            positive_values = condition_values[provider]["|".join(positive)]
            negative_values = condition_values[provider]["|".join(negative)]
            deltas = {
                puzzle_id: positive_values[puzzle_id] - negative_values[puzzle_id]
                for puzzle_id in included_ids
            }
            by_type: dict[str, dict[str, Any]] = {}
            for puzzle_type, _ in PUZZLE_TYPES:
                family_values = {
                    puzzle_id: value
                    for puzzle_id, value in deltas.items()
                    if stats["puzzle_types"][puzzle_id] == puzzle_type
                }
                by_type[puzzle_type] = {
                    "estimate": float(np.mean(list(family_values.values()))),
                    "ci95": clustered_mean_ci(family_values, draws),
                    "n_puzzles": len(family_values),
                }
            output["condition_contrasts"][provider][contrast] = {
                "estimate": float(
                    np.mean([item["estimate"] for item in by_type.values()])
                ),
                "ci95": clustered_mean_ci(deltas, draws),
                "n_puzzles": len(deltas),
                "by_type": by_type,
                "puzzle_values": deltas,
            }
    return output
