"""Numerical grid features for the study.

Extracted from the reviewed study implementation; presentation is separate.
"""

from __future__ import annotations

import json


import math


from collections import Counter


from pathlib import Path


from typing import Any, Iterable


import numpy as np


FEATURE_NAMES = (
    "normalized_mean_marginal_cell_rarity",
    "normalized_symmetric_monotonicity_penalty",
    "normalized_column_sequence_direction_changes",
)


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _minmax_normalize(
    raw: dict[str, tuple[float, ...]],
) -> tuple[dict[str, tuple[float, ...]], np.ndarray, np.ndarray]:
    solution_ids = list(raw)
    values = np.asarray([raw[solution_id] for solution_id in solution_ids])
    low = values.min(axis=0)
    high = values.max(axis=0)
    span = high - low
    normalized = np.divide(
        values - low,
        span,
        out=np.zeros_like(values, dtype=float),
        where=span != 0.0,
    )
    return (
        {
            solution_id: tuple(float(value) for value in normalized[index])
            for index, solution_id in enumerate(solution_ids)
        },
        low,
        high,
    )


def _raw_grid_features(puzzle: dict[str, Any]) -> dict[str, tuple[float, ...]]:
    puzzle_id = str(puzzle["id"])
    instance = puzzle["machine_readable_instance"]
    n = int(instance["n"])
    m = int(instance["m"])
    board = [str(row) for row in instance["board"]]
    solutions = puzzle["solutions"]
    if m != n - 1:
        raise ValueError(f"Expected m=n-1 for {puzzle_id}, found n={n}, m={m}")
    if m < 3:
        raise ValueError(f"Direction changes are undefined for m<3 in {puzzle_id}")
    if len(solutions) != int(puzzle["num_solutions"]):
        raise ValueError(f"Solution-count mismatch for {puzzle_id}")

    coordinate_sets: dict[str, tuple[tuple[int, int], ...]] = {}
    cell_counts: Counter[tuple[int, int]] = Counter()
    for solution in solutions:
        solution_id = str(solution["abstract_solution_id"])
        coordinates = tuple(
            sorted((int(row), int(column)) for row, column in solution["coordinates"])
        )
        if solution_id in coordinate_sets:
            raise ValueError(f"Duplicate solution ID {solution_id} in {puzzle_id}")
        if len(coordinates) != m:
            raise ValueError(f"Expected {m} tokens in {puzzle_id}, {solution_id}")
        if len({row for row, _ in coordinates}) != m:
            raise ValueError(f"Repeated row in {puzzle_id}, {solution_id}")
        if len({column for _, column in coordinates}) != m:
            raise ValueError(f"Repeated column in {puzzle_id}, {solution_id}")
        for row, column in coordinates:
            if not (1 <= row <= n and 1 <= column <= n):
                raise ValueError(f"Out-of-bounds cell in {puzzle_id}, {solution_id}")
            if board[row - 1][column - 1] == "X":
                raise ValueError(f"Forbidden occupied cell in {puzzle_id}, {solution_id}")
        coordinate_sets[solution_id] = coordinates
        cell_counts.update(coordinates)

    k = len(coordinate_sets)
    inversion_maximum = math.comb(m, 2)
    raw: dict[str, tuple[float, ...]] = {}
    stored_by_id = {
        str(solution["abstract_solution_id"]): solution for solution in solutions
    }
    for solution_id, coordinates in coordinate_sets.items():
        rarity = -sum(
            math.log(cell_counts[cell] / k) for cell in coordinates
        ) / m

        columns = [column for _, column in coordinates]
        inversions = sum(
            columns[left] > columns[right]
            for left in range(m)
            for right in range(left + 1, m)
        )
        monotonicity = min(
            inversions, inversion_maximum - inversions
        ) / inversion_maximum

        directions = [
            1 if columns[index + 1] > columns[index] else -1
            for index in range(m - 1)
        ]
        direction_changes = sum(
            directions[index] != directions[index - 1]
            for index in range(1, len(directions))
        ) / (m - 2)

        recorded = (
            stored_by_id[solution_id]
            .get("probe_features", {})
            .get("simplicity_components", {})
            .get("monotonicity_penalty")
        )
        if recorded is not None and not math.isclose(
            float(recorded), monotonicity, rel_tol=0.0, abs_tol=5e-4
        ):
            raise ValueError(
                f"Monotonicity mismatch for {puzzle_id}, {solution_id}: "
                f"stored={recorded}, derived={monotonicity}"
            )
        raw[solution_id] = (rarity, monotonicity, direction_changes)
    return raw


def load_grid_features(
    project: Path,
) -> tuple[
    list[str],
    dict[str, dict[str, tuple[float, ...]]],
    dict[str, Any],
]:
    puzzle_ids: list[str] = []
    features: dict[str, dict[str, tuple[float, ...]]] = {}
    raw_ranges: dict[str, dict[str, list[float]]] = {
        name: {} for name in FEATURE_NAMES
    }
    varying: dict[str, list[str]] = {name: [] for name in FEATURE_NAMES}
    constant: dict[str, list[str]] = {name: [] for name in FEATURE_NAMES}
    board_size_ids: dict[int, list[str]] = {}
    n4_relation_checked: list[str] = []
    n4_normalized_equal: list[str] = []

    for puzzle in _read_jsonl(project / "stimuli/all_puzzles.jsonl"):
        if puzzle.get("puzzle_type") != "grid_placement":
            continue
        puzzle_id = str(puzzle["id"])
        if puzzle.get("puzzle_type") != "grid_placement":
            raise ValueError(f"Unexpected puzzle type for {puzzle_id}")
        if puzzle_id in features:
            raise ValueError(f"Duplicate Rooks puzzle {puzzle_id}")
        n = int(puzzle["machine_readable_instance"]["n"])
        board_size_ids.setdefault(n, []).append(puzzle_id)
        raw = _raw_grid_features(puzzle)
        normalized, low, high = _minmax_normalize(raw)
        features[puzzle_id] = normalized
        puzzle_ids.append(puzzle_id)

        for index, feature_name in enumerate(FEATURE_NAMES):
            raw_ranges[feature_name][puzzle_id] = [
                float(low[index]), float(high[index])
            ]
            target = varying if high[index] > low[index] else constant
            target[feature_name].append(puzzle_id)

        if n == 4:
            n4_relation_checked.append(puzzle_id)
            for solution_id, raw_vector in raw.items():
                if not math.isclose(
                    raw_vector[1], raw_vector[2] / 3.0,
                    rel_tol=0.0, abs_tol=1e-12,
                ):
                    raise ValueError(
                        f"Expected M=Z/3 for {puzzle_id}, {solution_id}"
                    )
            if high[1] > low[1] and high[2] > low[2]:
                if not all(
                    math.isclose(vector[1], vector[2], abs_tol=1e-12)
                    for vector in normalized.values()
                ):
                    raise ValueError(
                        f"Expected normalized M=Z for n=4 puzzle {puzzle_id}"
                    )
                n4_normalized_equal.append(puzzle_id)

    if len(puzzle_ids) != 20:
        raise ValueError(
            f"Expected 20 main Rooks puzzles, found {len(puzzle_ids)}"
        )
    if set(board_size_ids) != {4, 5, 6}:
        raise ValueError(f"Unexpected Rooks board sizes: {board_size_ids}")

    null_information = np.zeros((len(FEATURE_NAMES), len(FEATURE_NAMES)))
    for puzzle_id in puzzle_ids:
        matrix = np.asarray(list(features[puzzle_id].values()), dtype=float)
        mean = matrix.mean(axis=0)
        null_information += matrix.T @ matrix / len(matrix) - np.outer(mean, mean)
    eigenvalues = np.linalg.eigvalsh(null_information)
    if eigenvalues[0] <= 0.0:
        raise ValueError(
            "The pooled three-feature Rooks design is rank deficient"
        )

    fold_eigenvalues: dict[str, list[float]] = {}
    for held_out in puzzle_ids:
        fold_information = np.zeros_like(null_information)
        for puzzle_id in puzzle_ids:
            if puzzle_id == held_out:
                continue
            matrix = np.asarray(list(features[puzzle_id].values()), dtype=float)
            mean = matrix.mean(axis=0)
            fold_information += (
                matrix.T @ matrix / len(matrix) - np.outer(mean, mean)
            )
        values = np.linalg.eigvalsh(fold_information)
        if values[0] <= 1e-12:
            raise ValueError(f"Rooks design is rank deficient without {held_out}")
        fold_eigenvalues[held_out] = [float(value) for value in values]

    diagnostics = {
        "features": {
            feature_name: {
                "varying_puzzle_count": len(varying[feature_name]),
                "constant_puzzle_count": len(constant[feature_name]),
                "varying_puzzle_ids": sorted(varying[feature_name]),
                "constant_puzzle_ids": sorted(constant[feature_name]),
                "raw_range_by_puzzle": raw_ranges[feature_name],
            }
            for feature_name in FEATURE_NAMES
        },
        "board_sizes": {
            str(n): {
                "puzzle_count": len(ids),
                "puzzle_ids": sorted(ids),
            }
            for n, ids in sorted(board_size_ids.items())
        },
        "n4_monotonicity_direction_change_relation": {
            "raw_relation": "M=Z/3 for every candidate when n=4 and m=3",
            "verified_puzzle_ids": sorted(n4_relation_checked),
            "normalized_coordinates_equal_when_varying": sorted(
                n4_normalized_equal
            ),
            "identification_note": (
                "n=4 puzzles cannot distinguish the M and Z coefficients; "
                "their separation in the joint fit comes from n=5 and n=6 puzzles"
            ),
        },
        "pooled_uniform_choice_information": {
            "eigenvalues": [float(value) for value in eigenvalues],
            "condition_number": float(eigenvalues[-1] / eigenvalues[0]),
            "rank": int(np.linalg.matrix_rank(null_information)),
        },
        "leave_one_puzzle_out_uniform_choice_information": {
            "minimum_eigenvalue": float(
                min(values[0] for values in fold_eigenvalues.values())
            ),
            "maximum_condition_number": float(
                max(values[-1] / values[0] for values in fold_eigenvalues.values())
            ),
            "eigenvalues_by_held_out_puzzle": fold_eigenvalues,
        },
    }
    return sorted(puzzle_ids), features, diagnostics
