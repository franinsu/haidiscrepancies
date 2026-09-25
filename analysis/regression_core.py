#!/usr/bin/env python3
"""Shared choice/source regression, observation loading and whole-puzzle validation.

Feature construction and command-line workflows live in their own modules.
Configure FEATURE_NAMES in each fit worker before constructing designs/results;
unconfigured workers have no implicit family feature specification. Numerical
objectives and bootstrap batching remain in regression_optimizer/bootstrap.
"""
from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.special import expit
from . import DEFAULT_L2_PENALTY
from .constant_feature_means import ConstantFeatureMeans
from .regression_optimizer import LogitObjective, damped_newton, method_metadata
from .regression_bootstrap import bootstrap_validation as run_bootstrap_validation


_OPTIMIZER_LOG = []

MODEL_FILES = {
    "GPT-5.6 Sol": "openai_main.jsonl",
    "Claude Opus 4.8": "anthropic_main.jsonl",
    "Gemini 3.5 Flash": "gemini_main.jsonl",
}

FEATURE_NAMES = ()

CONDITION_EFFORT = {
    "OAI_GPT56_SOL_LOW": "low",
    "OAI_GPT56_SOL_MEDIUM": "medium",
    "ANT_OPUS48_LOW": "low",
    "ANT_OPUS48_MEDIUM": "medium",
    "GEMINI_35_FLASH_LOW": "low",
    "GEMINI_35_FLASH_MEDIUM": "medium",
}

PROMPT_LABEL = {
    "direct_solve": "plain",
    "human_participant": "persona",
}

_FIT_PUZZLE_IDS: list[str] | None = None

_FIT_FEATURES: dict[str, dict[str, tuple[float, ...]]] | None = None

_FIT_OBSERVATIONS: dict[str, dict[str, list[str]]] | None = None


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def percentile_interval(values: list[float]) -> list[float]:
    return [float(value) for value in np.percentile(np.asarray(values), [2.5, 97.5])]


def normal_p_value(estimate: float, standard_error: float) -> float:
    if not math.isfinite(standard_error) or standard_error <= 0:
        return float("nan")
    return math.erfc(abs(estimate / standard_error) / math.sqrt(2.0))


def load_observations(
    repro: Path,
    puzzle_ids: list[str],
    features: dict[str, dict[str, tuple[float, ...]]],
) -> tuple[dict[str, dict[str, list[str]]], dict[str, dict[str, str]]]:
    valid_puzzles = set(puzzle_ids)
    observations: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    source_metadata: dict[str, dict[str, str]] = {
        "Human": {"provider": "Human", "effort": "", "prompt": ""}
    }

    for row in read_jsonl(repro / "human/main_retained.jsonl"):
        puzzle_id = str(row.get("puzzle_id"))
        solution_id = row.get("abstract_solution_id") or row.get("solution_id")
        if puzzle_id in valid_puzzles and row.get("is_correct") and solution_id:
            observations["Human"][puzzle_id].append(str(solution_id))

    for provider, filename in MODEL_FILES.items():
        for row in read_jsonl(repro / "ai" / filename):
            puzzle_id = str(row.get("puzzle_id"))
            solution_id = row.get("abstract_solution_id") or row.get("solution_id")
            condition = str(row.get("api_model_condition") or "")
            prompt = str(row.get("prompt_condition") or "")
            if CONDITION_EFFORT.get(condition) != 'low' or prompt != 'direct_solve':
                continue
            source = f"{provider}|{CONDITION_EFFORT[condition]}|{PROMPT_LABEL[prompt]}"
            source_metadata[source] = {
                "provider": provider,
                "effort": CONDITION_EFFORT[condition],
                "prompt": PROMPT_LABEL[prompt],
            }
            if puzzle_id in valid_puzzles and row.get("is_valid") and solution_id:
                observations[source][puzzle_id].append(str(solution_id))

    expected_sources = 1 + len(MODEL_FILES)
    if len(observations) != expected_sources:
        raise ValueError(f"Expected {expected_sources} sources, found {len(observations)}")
    for source, by_puzzle in observations.items():
        missing = valid_puzzles - set(by_puzzle)
        if missing:
            raise ValueError(f"{source} has no retained responses for {sorted(missing)}")
        for puzzle_id, solution_ids in by_puzzle.items():
            unknown = set(solution_ids) - set(features[puzzle_id])
            if unknown:
                raise ValueError(f"Unknown solutions for {source}, {puzzle_id}: {sorted(unknown)}")
    # A plain nested dictionary keeps the analysis state serializable for the
    # optional fit-level process pool.
    plain_observations = {
        source: {
            puzzle_id: list(solution_ids)
            for puzzle_id, solution_ids in by_puzzle.items()
        }
        for source, by_puzzle in observations.items()
    }
    return plain_observations, source_metadata


def empirical_solution_distribution(
    sources: tuple[str, ...],
    puzzle_id: str,
    solution_ids: list[str],
    observations: dict[str, dict[str, list[str]]],
) -> np.ndarray:
    """Return the equal-source mixture of within-puzzle response distributions."""
    if not sources:
        raise ValueError("At least one source is required")
    distribution = np.zeros(len(solution_ids), dtype=float)
    for source in sources:
        counts = Counter(observations[source][puzzle_id])
        source_distribution = np.asarray(
            [counts[solution_id] for solution_id in solution_ids], dtype=float
        )
        total = float(source_distribution.sum())
        if total <= 0.0:
            raise ValueError(f"{source} has no responses for {puzzle_id}")
        distribution += source_distribution / total / len(sources)
    if not np.isclose(distribution.sum(), 1.0):
        raise ValueError(f"Response distribution does not sum to one for {puzzle_id}")
    return distribution


def choice_tv_accuracy_bounds(
    puzzle_ids: list[str],
    mask: np.ndarray,
    observed_probabilities: np.ndarray,
) -> list[dict[str, Any]]:
    """Compute the TV-implied upper bound on top-choice accuracy."""
    records: list[dict[str, Any]] = []
    for group, puzzle_id in enumerate(puzzle_ids):
        valid = mask[group]
        candidate_count = int(valid.sum())
        distribution = observed_probabilities[group, valid]
        uniform_probability = 1.0 / candidate_count
        distance = float(
            0.5 * np.abs(distribution - uniform_probability).sum()
        )
        records.append({
            "puzzle_id": puzzle_id,
            "candidate_count": candidate_count,
            "normalized_accuracy_upper_bound": float(distribution.max()),
            "tv_distance_for_accuracy_bound": distance,
            "uniform_accuracy": uniform_probability,
            "tv_accuracy_upper_bound": float(uniform_probability + distance),
            "tv_accuracy_upper_bound_relative_to_uniform": float(
                1.0 + candidate_count * distance
            ),
        })
    return records


def source_tv_accuracy_bounds(
    model_sources: tuple[str, ...],
    puzzle_ids: list[str],
    features: dict[str, dict[str, tuple[float, ...]]],
    observations: dict[str, dict[str, list[str]]],
) -> list[dict[str, Any]]:
    """Compute the TV-implied Bayes upper bound on balanced accuracy."""
    records: list[dict[str, Any]] = []
    for puzzle_id in puzzle_ids:
        solution_ids = list(features[puzzle_id])
        human_distribution = empirical_solution_distribution(
            ("Human",), puzzle_id, solution_ids, observations
        )
        model_distribution = empirical_solution_distribution(
            model_sources, puzzle_id, solution_ids, observations
        )
        distance = float(
            0.5 * np.abs(model_distribution - human_distribution).sum()
        )
        records.append({
            "puzzle_id": puzzle_id,
            "tv_distance_for_accuracy_bound": distance,
            "uniform_accuracy": 0.5,
            "tv_accuracy_upper_bound": float(0.5 * (1.0 + distance)),
            "tv_accuracy_upper_bound_relative_to_uniform": float(1.0 + distance),
        })
    return records


def chance_to_ceiling_normalized_accuracy(
    accuracy: float,
    chance_accuracy: float,
    accuracy_increment: float,
) -> float:
    """Map chance to zero and the attainable accuracy ceiling to one."""
    if accuracy_increment <= 1e-12:
        return 0.0
    return float((accuracy - chance_accuracy) / accuracy_increment)


def source_arrays(
    model_sources: tuple[str, ...],
    puzzle_ids: list[str],
    features: dict[str, dict[str, tuple[float, ...]]],
    observations: dict[str, dict[str, list[str]]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    feature_rows: list[tuple[float, ...]] = []
    outcomes: list[float] = []
    base_weights: list[float] = []
    groups: list[int] = []
    for group, puzzle_id in enumerate(puzzle_ids):
        for class_sources, outcome in (
            (("Human",), 0.0),
            (model_sources, 1.0),
        ):
            source_weight = 1.0 / len(class_sources)
            for source in class_sources:
                selected = observations[source][puzzle_id]
                response_weight = source_weight / len(selected)
                for solution_id in selected:
                    feature_rows.append(features[puzzle_id][solution_id])
                    outcomes.append(outcome)
                    base_weights.append(response_weight)
                    groups.append(group)
    arrays = (
        np.asarray(feature_rows, dtype=float),
        np.asarray(outcomes),
        np.asarray(base_weights),
        np.asarray(groups),
    )
    _, outcome_array, weight_array, group_array = arrays
    for group in range(len(puzzle_ids)):
        for outcome in (0.0, 1.0):
            class_weight = weight_array[
                (group_array == group) & (outcome_array == outcome)
            ].sum()
            if not np.isclose(class_weight, 1.0):
                raise ValueError(
                    "Binary source weights do not sum to one for "
                    f"{puzzle_ids[group]}, class {int(outcome)}"
                )
    return arrays


def summarize_optimizer_diagnostics(records):
    from collections import Counter
    return {"fits": len(records),
            "gradient_tolerance_met": sum(r["gradient_tolerance_met"] for r in records),
            "max_gradient_inf": max((r["gradient_inf"] for r in records), default=0.0),
            "status_counts": dict(Counter(r["status"] for r in records)),
            "all_estimates_retained": True}


def fit_binary_logit(
    feature_matrix: np.ndarray,
    outcomes: np.ndarray,
    weights: np.ndarray,
    *,
    l2_penalty: float = DEFAULT_L2_PENALTY,
) -> tuple[np.ndarray, np.ndarray]:
    design = np.column_stack((np.ones(len(feature_matrix)), feature_matrix))
    coefficients, information, diagnostic = damped_newton(
        LogitObjective('source', design, outcomes, weights, l2_penalty=l2_penalty))
    _OPTIMIZER_LOG.append(diagnostic)
    return coefficients, information * max(float(np.sum(weights)), 1.0)


def source_lomo_validation(
    feature_matrix: np.ndarray,
    outcomes: np.ndarray,
    base_weights: np.ndarray,
    groups: np.ndarray,
    puzzle_ids: list[str],
    tv_distances: np.ndarray,
    multiplicity: np.ndarray,
    *,
    include_details: bool = False,
    constant_means: ConstantFeatureMeans | None = None,
    l2_penalty: float = DEFAULT_L2_PENALTY,
) -> dict[str, Any]:
    """Evaluate an identity-safe, multiplicity-weighted LOMO procedure."""
    multiplicity = np.asarray(multiplicity, dtype=float)
    total_weight = float(np.sum(multiplicity))
    if total_weight <= 0.0:
        raise ValueError("LOMO validation requires positive maze multiplicity")

    log_loss = 0.0
    balanced_accuracy = 0.0
    normalized_accuracy = 0.0
    details: list[dict[str, Any]] = []
    for held_out in np.flatnonzero(multiplicity > 0.0):
        training_multiplicity = multiplicity.copy()
        # Remove every duplicate copy of the held-out maze.  Leaving only one
        # copy out would leak that maze's candidate set into the fit.
        training_multiplicity[held_out] = 0.0
        training_weights = base_weights * training_multiplicity[groups]
        fold_matrix = (constant_means.transform(feature_matrix, groups, training_multiplicity)
                       if constant_means is not None else feature_matrix)
        fold_coefficients, _ = fit_binary_logit(
            fold_matrix,
            outcomes,
            training_weights,
            l2_penalty=l2_penalty,
        )
        test = groups == held_out
        test_design = np.column_stack((np.ones(test.sum()), fold_matrix[test]))
        fold_probabilities = np.clip(
            expit(test_design @ fold_coefficients),
            1e-12,
            1.0 - 1e-12,
        )
        # Each class has total base weight one within a maze; division by two
        # turns this into balanced test mass summing to one.
        test_weights = base_weights[test] / 2.0
        fold_loss = float(-np.sum(test_weights * (
            outcomes[test] * np.log(fold_probabilities)
            + (1.0 - outcomes[test]) * np.log(1.0 - fold_probabilities)
        )))
        fold_accuracy = float(np.sum(
            test_weights * ((fold_probabilities >= 0.5) == outcomes[test])
        ))
        fold_relative_accuracy = fold_accuracy / 0.5
        fold_normalized_accuracy = chance_to_ceiling_normalized_accuracy(
            fold_accuracy,
            0.5,
            0.5 * float(tv_distances[held_out]),
        )
        fold_weight = float(multiplicity[held_out] / total_weight)
        log_loss += fold_weight * fold_loss
        balanced_accuracy += fold_weight * fold_accuracy
        normalized_accuracy += fold_weight * fold_normalized_accuracy
        if include_details:
            details.append({
                "puzzle_id": puzzle_ids[held_out],
                "balanced_accuracy": fold_accuracy,
                "uniform_accuracy": 0.5,
                "balanced_accuracy_relative_to_uniform": fold_relative_accuracy,
                "normalized_accuracy": fold_normalized_accuracy,
            })

    return {
        "log_loss": float(log_loss),
        "log_loss_gain": float(math.log(2.0) - log_loss),
        "balanced_accuracy": float(balanced_accuracy),
        "normalized_accuracy": float(normalized_accuracy),
        "balanced_accuracy_relative_to_uniform": float(
            balanced_accuracy / 0.5
        ),
        "details": details,
    }


def source_result(
    model_sources: tuple[str, ...],
    puzzle_ids: list[str],
    features: dict[str, dict[str, tuple[float, ...]]],
    observations: dict[str, dict[str, list[str]]],
    rng: np.random.Generator,
    repeats: int,
    *,
    l2_penalty: float = DEFAULT_L2_PENALTY,
) -> dict[str, Any]:
    _OPTIMIZER_LOG.clear()
    diagnostic_start = 0
    feature_matrix, outcomes, base_weights, groups = source_arrays(
        model_sources, puzzle_ids, features, observations
    )
    constant_means = ConstantFeatureMeans(puzzle_ids, features)
    raw_matrix = feature_matrix
    feature_matrix = constant_means.transform(raw_matrix, groups, np.ones(len(puzzle_ids)))
    tv_bound_records = source_tv_accuracy_bounds(
        model_sources, puzzle_ids, features, observations
    )
    tv_distances = np.asarray([
        record["tv_distance_for_accuracy_bound"]
        for record in tv_bound_records
    ], dtype=float)
    coefficients, information = fit_binary_logit(
        feature_matrix, outcomes, base_weights, l2_penalty=l2_penalty
    )
    design = np.column_stack((np.ones(len(feature_matrix)), feature_matrix))
    probabilities = expit(design @ coefficients)
    cluster_scores = np.vstack([
        design[groups == group].T
        @ (base_weights[groups == group] * (outcomes[groups == group] - probabilities[groups == group]))
        for group in range(len(puzzle_ids))
    ])
    inverse_information = np.linalg.pinv(information, hermitian=True)
    covariance = (
        inverse_information
        @ (cluster_scores.T @ cluster_scores)
        @ inverse_information
        * len(puzzle_ids)
        / (len(puzzle_ids) - 1)
    )
    robust_se = np.sqrt(np.diag(covariance)[1:])

    point_validation = source_lomo_validation(
        raw_matrix,
        outcomes,
        base_weights,
        groups,
        puzzle_ids,
        tv_distances,
        np.ones(len(puzzle_ids)),
        include_details=True,
        constant_means=constant_means,
        l2_penalty=l2_penalty,
    )
    for validation_record, bound_record in zip(
        point_validation["details"], tv_bound_records, strict=True
    ):
        if validation_record["puzzle_id"] != bound_record["puzzle_id"]:
            raise ValueError("Source accuracy and TV-bound puzzle orders differ")
        validation_record.update({
            "tv_distance_for_accuracy_bound": bound_record[
                "tv_distance_for_accuracy_bound"
            ],
            "tv_accuracy_upper_bound_relative_to_uniform": bound_record[
                "tv_accuracy_upper_bound_relative_to_uniform"
            ],
        })
        if (
            validation_record["balanced_accuracy_relative_to_uniform"]
            > bound_record["tv_accuracy_upper_bound_relative_to_uniform"] + 1e-12
        ):
            raise ValueError("Source accuracy exceeds its TV upper bound")
    tv_accuracy_upper_bound = float(np.mean([
        record["tv_accuracy_upper_bound_relative_to_uniform"]
        for record in tv_bound_records
    ]))

    candidate_tensor, candidate_mask, model_probabilities = choice_design(
        model_sources, puzzle_ids, features, observations)
    _, _, human_probabilities = choice_design(('Human',), puzzle_ids, features, observations)
    bootstrap_coefficients, bootstrap_validation, diagnostics = run_bootstrap_validation(
        candidate_tensor, candidate_mask, model_probabilities, rng, repeats, human_probabilities,
        l2_penalty=l2_penalty)
    _OPTIMIZER_LOG.extend(diagnostics)
    bootstrap_array = np.vstack(bootstrap_coefficients)
    coefficient_intervals = np.percentile(
        bootstrap_array, [2.5, 97.5], axis=0
    )

    human_means = [
        np.mean(np.asarray([
            features[puzzle_id][solution]
            for solution in observations["Human"][puzzle_id]
        ]), axis=0)
        for puzzle_id in puzzle_ids
    ]
    model_means = [
        np.mean(np.asarray([
            np.mean(np.asarray([
                features[puzzle_id][solution]
                for solution in observations[source][puzzle_id]
            ]), axis=0)
            for source in model_sources
        ]), axis=0)
        for puzzle_id in puzzle_ids
    ]
    all_puzzles = np.arange(len(puzzle_ids))
    human_means = constant_means.transform(np.vstack(human_means), all_puzzles, np.ones(len(puzzle_ids)))
    model_means = constant_means.transform(np.vstack(model_means), all_puzzles, np.ones(len(puzzle_ids)))
    human_mean_array = np.mean(human_means, axis=0)
    model_mean_array = np.mean(model_means, axis=0)
    difference_array = np.mean(
        np.vstack(model_means) - np.vstack(human_means), axis=0
    )
    def validation_interval(field: str) -> list[float]:
        return percentile_interval([
            float(replicate[field]) for replicate in bootstrap_validation
        ])

    result = {
        "constant_feature_replacement": "training mean of within-puzzle scaled values; uniform candidates, equal nonconstant puzzles with bootstrap multiplicity; re-estimated in each validation fold",
        "pooled_sources": list(model_sources),
        "pooling": "equal weight for each supplied source member within maze and class",
        "intercept": float(coefficients[0]),
        "feature_coefficients": {
            name: float(coefficients[index + 1])
            for index, name in enumerate(FEATURE_NAMES)
        },
        "feature_coefficient_ci_95": {
            name: [
                float(coefficient_intervals[0, index]),
                float(coefficient_intervals[1, index]),
            ]
            for index, name in enumerate(FEATURE_NAMES)
        },
        "bootstrap_rejected_fits": 0,
        "optimizer": method_metadata(l2_penalty),
        "optimizer_diagnostics": summarize_optimizer_diagnostics(_OPTIMIZER_LOG[diagnostic_start:]),
        "bootstrap_coefficients": np.asarray(bootstrap_coefficients).tolist(),
        "bootstrap_validation": {key: [row[key] for row in bootstrap_validation]
                                 for key in bootstrap_validation[0] if key != "details"},
        "bootstrap_validation_repeats": repeats,
        "bootstrap_shared_draws_for_coefficients_and_validation": True,
        "feature_cluster_robust_se": {
            name: float(robust_se[index])
            for index, name in enumerate(FEATURE_NAMES)
        },
        "feature_p_value_normal_cluster_robust": {
            name: normal_p_value(
                float(coefficients[index + 1]), float(robust_se[index])
            )
            for index, name in enumerate(FEATURE_NAMES)
        },
        "human_mean_features": {
            name: float(human_mean_array[index])
            for index, name in enumerate(FEATURE_NAMES)
        },
        "model_mean_features": {
            name: float(model_mean_array[index])
            for index, name in enumerate(FEATURE_NAMES)
        },
        "mean_within_maze_difference": {
            name: float(difference_array[index])
            for index, name in enumerate(FEATURE_NAMES)
        },
        "cv_log_loss": point_validation["log_loss"],
        "cv_log_loss_ci_95": validation_interval("log_loss"),
        "baseline_log_loss": math.log(2.0),
        "cv_log_loss_gain": point_validation["log_loss_gain"],
        "cv_log_loss_gain_ci_95": validation_interval("log_loss_gain"),
        "cv_balanced_accuracy": point_validation["balanced_accuracy"],
        "cv_balanced_accuracy_ci_95": validation_interval("balanced_accuracy"),
        "baseline_balanced_accuracy": 0.5,
        "cv_normalized_accuracy": point_validation["normalized_accuracy"],
        "cv_normalized_accuracy_ci_95": validation_interval(
            "normalized_accuracy"
        ),
        "cv_balanced_accuracy_relative_to_uniform": point_validation[
            "balanced_accuracy_relative_to_uniform"
        ],
        "cv_balanced_accuracy_relative_to_uniform_ci_95": validation_interval(
            "balanced_accuracy_relative_to_uniform"
        ),
        "baseline_balanced_accuracy_relative_to_uniform": 1.0,
        "tv_accuracy_upper_bound_relative_to_uniform": tv_accuracy_upper_bound,
        "tv_accuracy_upper_bound_relative_to_uniform_by_puzzle": tv_bound_records,
        "cv_balanced_accuracy_by_puzzle": point_validation["details"],
        "human_responses": int(sum(len(observations["Human"][puzzle]) for puzzle in puzzle_ids)),
        "model_responses": int(sum(
            len(observations[source][puzzle])
            for source in model_sources
            for puzzle in puzzle_ids
        )),
    }
    return regression_inference(result, l2_penalty)


def regression_inference(result, l2_penalty):
    if l2_penalty:
        # The legacy sandwich/Wald diagnostics are for the unpenalized estimator.
        # Published inference already uses the complete bootstrap refits.
        result.pop('feature_cluster_robust_se')
        result.pop('feature_p_value_normal_cluster_robust')
        result['inference'] = ('Penalized point estimates and whole-puzzle percentile bootstrap intervals; '
                               'unpenalized sandwich/Wald diagnostics omitted. '
                               'Validation metrics exclude the fitting penalty.')
    return result


def choice_design(
    sources: tuple[str, ...],
    puzzle_ids: list[str],
    features: dict[str, dict[str, tuple[float, ...]]],
    observations: dict[str, dict[str, list[str]]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    width = max(len(features[puzzle_id]) for puzzle_id in puzzle_ids)
    feature_tensor = np.zeros((len(puzzle_ids), width, len(FEATURE_NAMES)))
    mask = np.zeros((len(puzzle_ids), width), dtype=bool)
    observed_probabilities = np.zeros((len(puzzle_ids), width))
    for group, puzzle_id in enumerate(puzzle_ids):
        solution_ids = list(features[puzzle_id])
        candidate_features = np.asarray([
            features[puzzle_id][solution_id] for solution_id in solution_ids
        ], dtype=float)
        feature_tensor[group, : len(candidate_features)] = candidate_features
        mask[group, : len(candidate_features)] = True
        observed_probabilities[group, : len(candidate_features)] = (
            empirical_solution_distribution(
                sources, puzzle_id, solution_ids, observations
            )
        )
    if not np.allclose(observed_probabilities.sum(axis=1), 1.0):
        raise ValueError("Selected-solution probabilities do not sum to one")
    return feature_tensor, mask, observed_probabilities


def choice_probabilities(
    coefficients: np.ndarray,
    feature_tensor: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    linear_scores = np.einsum("gkd,d->gk", feature_tensor, coefficients)
    scores = np.where(
        mask,
        linear_scores,
        -np.inf,
    )
    scores -= np.max(scores, axis=1, keepdims=True)
    probabilities = np.where(mask, np.exp(scores), 0.0)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    return probabilities


def choice_score_information(
    coefficients: np.ndarray,
    feature_tensor: np.ndarray,
    mask: np.ndarray,
    observed_probabilities: np.ndarray,
    multiplicity: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    probabilities = choice_probabilities(coefficients, feature_tensor, mask)
    dimension = feature_tensor.shape[2]
    score = np.zeros(dimension)
    information = np.zeros((dimension, dimension))
    cluster_scores = np.zeros((len(feature_tensor), dimension))
    for group in range(len(feature_tensor)):
        if multiplicity[group] == 0.0:
            continue
        valid = mask[group]
        design = feature_tensor[group, valid]
        residual = (
            observed_probabilities[group, valid]
            - probabilities[group, valid]
        )
        group_score = design.T @ residual
        cluster_scores[group] = group_score
        score += multiplicity[group] * group_score
        mean_design = probabilities[group, valid] @ design
        covariance = (
            design.T
            @ (probabilities[group, valid, None] * design)
            - np.outer(mean_design, mean_design)
        )
        information += multiplicity[group] * covariance
    return score, information, cluster_scores


def fit_choice_logit(
    feature_tensor: np.ndarray,
    mask: np.ndarray,
    observed_probabilities: np.ndarray,
    multiplicity: np.ndarray | None = None,
    *,
    l2_penalty: float = DEFAULT_L2_PENALTY,
) -> np.ndarray:
    if multiplicity is None:
        multiplicity = np.ones(len(observed_probabilities))
    coefficients, _, diagnostic = damped_newton(LogitObjective(
        'choice', feature_tensor, observed_probabilities, multiplicity, mask, l2_penalty=l2_penalty))
    _OPTIMIZER_LOG.append(diagnostic)
    return coefficients


def choice_lomo_validation(
    feature_tensor: np.ndarray,
    mask: np.ndarray,
    observed_probabilities: np.ndarray,
    puzzle_ids: list[str],
    features: dict[str, dict[str, tuple[float, ...]]],
    multiplicity: np.ndarray,
    *,
    include_details: bool = False,
    l2_penalty: float = DEFAULT_L2_PENALTY,
) -> dict[str, Any]:
    """Evaluate an identity-safe, multiplicity-weighted choice LOMO fit."""
    multiplicity = np.asarray(multiplicity, dtype=float)
    total_weight = float(np.sum(multiplicity))
    if total_weight <= 0.0:
        raise ValueError("LOMO validation requires positive maze multiplicity")

    log_loss = 0.0
    baseline_log_loss = 0.0
    top_choice_accuracy = 0.0
    baseline_accuracy = 0.0
    relative_accuracy = 0.0
    normalized_accuracy = 0.0
    details: list[dict[str, Any]] = []
    for held_out in np.flatnonzero(multiplicity > 0.0):
        training_multiplicity = multiplicity.copy()
        # Exclude all resampled copies of this maze before predicting it.
        training_multiplicity[held_out] = 0.0
        fold_coefficients = fit_choice_logit(
            feature_tensor,
            mask,
            observed_probabilities,
            training_multiplicity,
            l2_penalty=l2_penalty,
        )
        puzzle_id = puzzle_ids[held_out]
        solution_ids = list(features[puzzle_id])
        candidate_count = len(solution_ids)
        held_out_distribution = observed_probabilities[
            held_out, :candidate_count
        ]
        probabilities = choice_probabilities(
            fold_coefficients,
            feature_tensor[held_out : held_out + 1],
            mask[held_out : held_out + 1],
        )[0, :candidate_count]
        fold_loss = float(
            -held_out_distribution
            @ np.log(np.clip(probabilities, 1e-12, 1.0))
        )
        fold_baseline_loss = math.log(candidate_count)
        maximum = probabilities.max()
        top = np.isclose(probabilities, maximum, rtol=1e-12, atol=1e-12)
        fold_accuracy = float(held_out_distribution[top].sum() / top.sum())
        fold_baseline_accuracy = 1.0 / candidate_count
        fold_relative_accuracy = fold_accuracy / fold_baseline_accuracy
        fold_tv_distance = float(
            0.5
            * np.abs(held_out_distribution - fold_baseline_accuracy).sum()
        )
        fold_normalized_accuracy = chance_to_ceiling_normalized_accuracy(
            fold_accuracy,
            fold_baseline_accuracy,
            float(held_out_distribution.max()) - fold_baseline_accuracy,
        )
        fold_weight = float(multiplicity[held_out] / total_weight)
        log_loss += fold_weight * fold_loss
        baseline_log_loss += fold_weight * fold_baseline_loss
        top_choice_accuracy += fold_weight * fold_accuracy
        baseline_accuracy += fold_weight * fold_baseline_accuracy
        relative_accuracy += fold_weight * fold_relative_accuracy
        normalized_accuracy += fold_weight * fold_normalized_accuracy
        if include_details:
            details.append({
                "puzzle_id": puzzle_id,
                "candidate_count": candidate_count,
                "predicted_top_set_size": int(top.sum()),
                "predicted_top_solution_ids": [
                    solution_id
                    for solution_id, is_top in zip(
                        solution_ids, top, strict=True
                    )
                    if is_top
                ],
                "top_choice_accuracy": fold_accuracy,
                "uniform_accuracy": fold_baseline_accuracy,
                "top_choice_accuracy_relative_to_uniform": (
                    fold_relative_accuracy
                ),
                "normalized_accuracy_upper_bound": float(held_out_distribution.max()),
                "tv_distance_for_accuracy_bound": fold_tv_distance,
                "normalized_accuracy": fold_normalized_accuracy,
            })

    return {
        "log_loss": float(log_loss),
        "baseline_log_loss": float(baseline_log_loss),
        "log_loss_gain": float(baseline_log_loss - log_loss),
        "top_choice_accuracy": float(top_choice_accuracy),
        "baseline_top_choice_accuracy": float(baseline_accuracy),
        "top_choice_accuracy_relative_to_uniform": float(relative_accuracy),
        "normalized_accuracy": float(normalized_accuracy),
        "details": details,
    }


def choice_result(
    sources: tuple[str, ...],
    puzzle_ids: list[str],
    features: dict[str, dict[str, tuple[float, ...]]],
    observations: dict[str, dict[str, list[str]]],
    rng: np.random.Generator,
    repeats: int,
    *,
    l2_penalty: float = DEFAULT_L2_PENALTY,
) -> dict[str, Any]:
    _OPTIMIZER_LOG.clear()
    diagnostic_start = 0
    feature_tensor, mask, observed_probabilities = choice_design(
        sources, puzzle_ids, features, observations
    )
    coefficients = fit_choice_logit(
        feature_tensor, mask, observed_probabilities, l2_penalty=l2_penalty
    )
    point_validation = choice_lomo_validation(
        feature_tensor,
        mask,
        observed_probabilities,
        puzzle_ids,
        features,
        np.ones(len(puzzle_ids)),
        include_details=True,
        l2_penalty=l2_penalty,
    )
    tv_bound_records = choice_tv_accuracy_bounds(
        puzzle_ids, mask, observed_probabilities
    )
    for validation_record, bound_record in zip(
        point_validation["details"], tv_bound_records, strict=True
    ):
        if validation_record["puzzle_id"] != bound_record["puzzle_id"]:
            raise ValueError("Choice accuracy and TV-bound puzzle orders differ")
        validation_record.update({
            "tv_distance_for_accuracy_bound": bound_record[
                "tv_distance_for_accuracy_bound"
            ],
            "tv_accuracy_upper_bound_relative_to_uniform": bound_record[
                "tv_accuracy_upper_bound_relative_to_uniform"
            ],
        })
        if (
            validation_record["top_choice_accuracy_relative_to_uniform"]
            > bound_record["tv_accuracy_upper_bound_relative_to_uniform"] + 1e-12
        ):
            raise ValueError("Choice accuracy exceeds its TV upper bound")
    tv_accuracy_upper_bound = float(np.mean([
        record["tv_accuracy_upper_bound_relative_to_uniform"]
        for record in tv_bound_records
    ]))

    bootstrap_coefficients, bootstrap_validation, diagnostics = run_bootstrap_validation(
        feature_tensor, mask, observed_probabilities, rng, repeats, l2_penalty=l2_penalty)
    _OPTIMIZER_LOG.extend(diagnostics)
    bootstrap_array = np.vstack(bootstrap_coefficients)
    coefficient_intervals = np.percentile(
        bootstrap_array, [2.5, 97.5], axis=0
    )

    _, information, cluster_scores = choice_score_information(
        coefficients,
        feature_tensor,
        mask,
        observed_probabilities,
        np.ones(len(puzzle_ids)),
    )
    inverse_information = np.linalg.pinv(information, hermitian=True)
    covariance = (
        inverse_information
        @ (cluster_scores.T @ cluster_scores)
        @ inverse_information
        * len(puzzle_ids)
        / (len(puzzle_ids) - 1)
    )
    robust_se = np.sqrt(np.diag(covariance))

    selected_means: list[np.ndarray] = []
    uniform_means: list[np.ndarray] = []
    for group, puzzle_id in enumerate(puzzle_ids):
        candidate_features = np.asarray(
            list(features[puzzle_id].values()), dtype=float
        )
        observed_mean = np.mean(np.asarray([
            np.mean(np.asarray([
                features[puzzle_id][solution]
                for solution in observations[source][puzzle_id]
            ]), axis=0)
            for source in sources
        ]), axis=0)
        selected_means.append(observed_mean)
        uniform_means.append(candidate_features.mean(axis=0))

    selected_mean_array = np.mean(np.vstack(selected_means), axis=0)
    uniform_mean_array = np.mean(np.vstack(uniform_means), axis=0)
    difference_array = np.mean(
        np.vstack(selected_means) - np.vstack(uniform_means), axis=0
    )
    def validation_interval(field: str) -> list[float]:
        return percentile_interval([
            float(replicate[field]) for replicate in bootstrap_validation
        ])

    result = {
        "pooled_sources": list(sources),
        "pooling": "equal weight for each supplied source member within maze",
        "score": "sum_q theta_q * x_q; main effects only",
        "feature_coefficients": {
            name: float(coefficients[index])
            for index, name in enumerate(FEATURE_NAMES)
        },
        "bootstrap_rejected_fits": 0,
        "optimizer": method_metadata(l2_penalty),
        "optimizer_diagnostics": summarize_optimizer_diagnostics(_OPTIMIZER_LOG[diagnostic_start:]),
        "bootstrap_coefficients": np.asarray(bootstrap_coefficients).tolist(),
        "bootstrap_validation": {key: [row[key] for row in bootstrap_validation]
                                 for key in bootstrap_validation[0] if key != "details"},
        "bootstrap_validation_repeats": repeats,
        "bootstrap_shared_draws_for_coefficients_and_validation": True,
        "feature_coefficient_ci_95": {
            name: [
                float(coefficient_intervals[0, index]),
                float(coefficient_intervals[1, index]),
            ]
            for index, name in enumerate(FEATURE_NAMES)
        },
        "feature_cluster_robust_se": {
            name: float(robust_se[index])
            for index, name in enumerate(FEATURE_NAMES)
        },
        "feature_p_value_normal_cluster_robust": {
            name: normal_p_value(
                float(coefficients[index]), float(robust_se[index])
            )
            for index, name in enumerate(FEATURE_NAMES)
        },
        "selected_mean_features": {
            name: float(selected_mean_array[index])
            for index, name in enumerate(FEATURE_NAMES)
        },
        "uniform_candidate_mean_features": {
            name: float(uniform_mean_array[index])
            for index, name in enumerate(FEATURE_NAMES)
        },
        "mean_within_maze_difference_from_uniform": {
            name: float(difference_array[index])
            for index, name in enumerate(FEATURE_NAMES)
        },
        "cv_log_loss": point_validation["log_loss"],
        "cv_log_loss_ci_95": validation_interval("log_loss"),
        "baseline_log_loss": point_validation["baseline_log_loss"],
        "cv_log_loss_gain": point_validation["log_loss_gain"],
        "cv_log_loss_gain_ci_95": validation_interval("log_loss_gain"),
        "cv_top_choice_accuracy": point_validation["top_choice_accuracy"],
        "cv_top_choice_accuracy_ci_95": validation_interval(
            "top_choice_accuracy"
        ),
        "baseline_top_choice_accuracy": point_validation[
            "baseline_top_choice_accuracy"
        ],
        "cv_normalized_accuracy": point_validation["normalized_accuracy"],
        "cv_normalized_accuracy_ci_95": validation_interval(
            "normalized_accuracy"
        ),
        "cv_top_choice_accuracy_relative_to_uniform": point_validation[
            "top_choice_accuracy_relative_to_uniform"
        ],
        "cv_top_choice_accuracy_relative_to_uniform_ci_95": (
            validation_interval("top_choice_accuracy_relative_to_uniform")
        ),
        "baseline_top_choice_accuracy_relative_to_uniform": 1.0,
        "tv_accuracy_upper_bound_relative_to_uniform": tv_accuracy_upper_bound,
        "tv_accuracy_upper_bound_relative_to_uniform_by_puzzle": tv_bound_records,
        "cv_top_choice_accuracy_by_puzzle": point_validation["details"],
        "responses": int(sum(
            len(observations[source][puzzle])
            for source in sources
            for puzzle in puzzle_ids
        )),
    }
    return regression_inference(result, l2_penalty)


def initialize_fit_worker(
    puzzle_ids: list[str],
    features: dict[str, dict[str, tuple[float, ...]]],
    observations: dict[str, dict[str, list[str]]],
) -> None:
    """Install read-only analysis inputs once per worker process."""
    global _FIT_PUZZLE_IDS, _FIT_FEATURES, _FIT_OBSERVATIONS
    _FIT_PUZZLE_IDS = puzzle_ids
    _FIT_FEATURES = features
    _FIT_OBSERVATIONS = observations


def execute_fit_task(
    task: tuple[str, str, tuple[str, ...], int, int],
    *,
    l2_penalty: float = DEFAULT_L2_PENALTY,
) -> tuple[str, dict[str, Any]]:
    """Run one independently seeded fit for serial or parallel execution."""
    kind, key, sources, seed, repeats = task
    if (
        _FIT_PUZZLE_IDS is None
        or _FIT_FEATURES is None
        or _FIT_OBSERVATIONS is None
    ):
        raise RuntimeError("Fit worker was not initialized")
    rng = np.random.default_rng(seed)
    if kind == "choice":
        result = choice_result(
            sources,
            _FIT_PUZZLE_IDS,
            _FIT_FEATURES,
            _FIT_OBSERVATIONS,
            rng,
            repeats,
            l2_penalty=l2_penalty,
        )
    elif kind == "source":
        result = source_result(
            sources,
            _FIT_PUZZLE_IDS,
            _FIT_FEATURES,
            _FIT_OBSERVATIONS,
            rng,
            repeats,
            l2_penalty=l2_penalty,
        )
    else:
        raise ValueError(f"Unknown fit task kind: {kind}")
    return key, result
