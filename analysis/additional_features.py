"""Numerical additional features for the study.

Extracted from the reviewed study implementation; presentation is separate.
"""

from __future__ import annotations

import json


from dataclasses import dataclass


from fractions import Fraction


from functools import lru_cache


from pathlib import Path


from typing import Any, Iterable


import numpy as np


FAMILY_SPECS: dict[str, dict[str, Any]] = {
    "arithmetic24": {
        "display": "Arithmetic",
        "short": "Arithmetic",
        "raw_features": (
            "normalized_minimum_division_count",
            "minimum_awkward_intermediate_count",
            "balanced_plan_affordance",
            "normalized_minimum_display_order_inversions",
        ),
        "features": (
            "normalized_minimum_division_count",
            "balanced_plan_affordance",
            "normalized_minimum_display_order_inversions",
        ),
        "regression_indices": (0, 2, 3),
        "expected_raw_varying": (15, 13, 20, 20),
        "expected_candidate_count": 141,
        "feature_specification": {
            "normalized_minimum_division_count": {
                "raw_definition": (
                    "minimum division-node count over every admissible parse "
                    "tree in the canonical expression class"
                ),
                "normalization": "within-puzzle min-max; zero when constant",
            },
            "balanced_plan_affordance": {
                "raw_definition": (
                    "indicator that some parse tree in the canonical class "
                    "has a two-token versus two-token root split"
                ),
                "normalization": "binary natural scale",
            },
            "normalized_minimum_display_order_inversions": {
                "raw_definition": (
                    "minimum inversion count of parse-tree leaves relative "
                    "to displayed token positions, including repeated-token matchings"
                ),
                "normalization": "within-puzzle min-max; zero when constant",
            },
        },
        "excluded_feature_specification": {
            "minimum_awkward_intermediate_count": {
                "raw_definition": (
                    "minimum count of negative or noninteger internal values "
                    "over every parse tree in the canonical expression class"
                ),
                "structural_variation": "varies within 13 of 20 puzzles",
                "exclusion_reason": (
                    "near-zero empirical support in model responses causes "
                    "separation and a nonexistent finite unpenalized estimate"
                ),
            }
        },
    },
    "minesweeper_lite": {
        "display": "Minesweeper",
        "short": "Minesweeper",
        "features": (
            "normalized_minimum_certificate_size",
            "normalized_minimum_certificate_scope",
        ),
        "expected_raw_varying": (15, 20),
        "feature_specification": {
            "normalized_minimum_certificate_size": {
                "raw_definition": (
                    "minimum number of displayed clue equations sufficient "
                    "to force the candidate cell's requested status"
                ),
                "normalization": "within-puzzle min-max; zero when constant",
            },
            "normalized_minimum_certificate_scope": {
                "raw_definition": (
                    "minimum number of hidden variables touched among the "
                    "minimum-size certificates"
                ),
                "normalization": "within-puzzle min-max; zero when constant",
            },
        },
    },
    "mini_sudoku": {
        "display": "Sudoku",
        "short": "Sudoku",
        "features": (
            "normalized_cell_domain_size",
            "normalized_target_location_count",
        ),
        "expected_raw_varying": (18, 15),
        "feature_specification": {
            "normalized_cell_domain_size": {
                "raw_definition": (
                    "number of digits locally legal at the candidate cell"
                ),
                "normalization": "within-puzzle min-max; zero when constant",
            },
            "normalized_target_location_count": {
                "raw_definition": (
                    "minimum, over the candidate's row, column, and box, of "
                    "the number of legal locations for the named digit"
                ),
                "normalization": "within-puzzle min-max; zero when constant",
            },
        },
    },
}


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def minmax_normalize(
    raw: dict[str, tuple[float, ...]],
) -> tuple[dict[str, tuple[float, ...]], np.ndarray, np.ndarray]:
    solution_ids = list(raw)
    matrix = np.asarray([raw[solution_id] for solution_id in solution_ids], dtype=float)
    low = matrix.min(axis=0)
    high = matrix.max(axis=0)
    span = high - low
    normalized = np.divide(
        matrix - low,
        span,
        out=np.zeros_like(matrix),
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


@dataclass(frozen=True)
class ArithmeticTree:
    value: Fraction
    canonical_key: tuple[Any, ...]
    raw_key: tuple[Any, ...]
    division_count: int
    awkward_count: int
    leaves: tuple[int, ...]
    root_left_size: int


def arithmetic_class_features(
    puzzle: dict[str, Any], project: Path
) -> dict[str, tuple[float, ...]]:
    """Enumerate all four-token trees and aggregate invariant class features."""
    from puzzles import arithmetic24 as arithmetic  # type: ignore

    puzzle_id = str(puzzle["id"])
    numbers = tuple(int(value) for value in puzzle["machine_readable_instance"]["numbers"])
    if len(numbers) != 4:
        raise ValueError(f"Expected four Arithmetic tokens in {puzzle_id}")

    @lru_cache(maxsize=None)
    def build(mask: int) -> tuple[ArithmeticTree, ...]:
        if mask and mask & (mask - 1) == 0:
            index = mask.bit_length() - 1
            value = Fraction(numbers[index], 1)
            return (
                ArithmeticTree(
                    value=value,
                    canonical_key=("n", numbers[index]),
                    raw_key=("n", index),
                    division_count=0,
                    awkward_count=0,
                    leaves=(index,),
                    root_left_size=0,
                ),
            )

        trees: dict[tuple[Any, ...], ArithmeticTree] = {}
        subset = (mask - 1) & mask
        while subset:
            other = mask ^ subset
            if subset < other:
                for first in build(subset):
                    for second in build(other):
                        for left, right in ((first, second), (second, first)):
                            operations = (
                                ("+", left.value + right.value),
                                ("*", left.value * right.value),
                                ("-", left.value - right.value),
                            )
                            if right.value != 0:
                                operations += (("/", left.value / right.value),)
                            for operation, value in operations:
                                raw_key = (operation, left.raw_key, right.raw_key)
                                tree = ArithmeticTree(
                                    value=value,
                                    canonical_key=arithmetic._canonical_key(
                                        operation,
                                        left.canonical_key,
                                        right.canonical_key,
                                    ),
                                    raw_key=raw_key,
                                    division_count=(
                                        left.division_count
                                        + right.division_count
                                        + int(operation == "/")
                                    ),
                                    awkward_count=(
                                        left.awkward_count
                                        + right.awkward_count
                                        + int(value < 0 or value.denominator != 1)
                                    ),
                                    leaves=left.leaves + right.leaves,
                                    root_left_size=len(left.leaves),
                                )
                                trees[raw_key] = tree
            subset = (subset - 1) & mask
        return tuple(trees.values())

    aggregate: dict[str, list[float]] = {}
    full_mask = (1 << len(numbers)) - 1
    for tree in build(full_mask):
        if tree.value != arithmetic.TARGET:
            continue
        canonical = arithmetic._key_to_string(tree.canonical_key)
        inversions = sum(
            tree.leaves[left] > tree.leaves[right]
            for left in range(len(tree.leaves))
            for right in range(left + 1, len(tree.leaves))
        )
        candidate = [
            float(tree.division_count),
            float(tree.awkward_count),
            float(tree.root_left_size == 2),
            float(inversions),
        ]
        if canonical not in aggregate:
            aggregate[canonical] = candidate
        else:
            current = aggregate[canonical]
            current[0] = min(current[0], candidate[0])
            current[1] = min(current[1], candidate[1])
            current[2] = max(current[2], candidate[2])
            current[3] = min(current[3], candidate[3])

    stored = {
        str(solution["canonical_expression"]): str(solution["abstract_solution_id"])
        for solution in puzzle["solutions"]
    }
    if set(aggregate) != set(stored):
        missing = sorted(set(stored) - set(aggregate))
        extra = sorted(set(aggregate) - set(stored))
        raise ValueError(
            f"Arithmetic class mismatch in {puzzle_id}: missing={missing}, extra={extra}"
        )
    # Return all four audited attributes.  The loader omits the awkward-
    # intermediate minimum from the regression only after recording its
    # structural variation; empirical near-separation prevents a finite
    # unpenalized estimate for that attribute.
    return {
        stored[canonical]: tuple(values)
        for canonical, values in aggregate.items()
    }


def minesweeper_certificate_features(
    puzzle: dict[str, Any]
) -> dict[str, tuple[float, ...]]:
    """Compute exact minimum clue certificates by a Boolean subset transform."""
    puzzle_id = str(puzzle["id"])
    board = [str(row) for row in puzzle["machine_readable_instance"]["board"]]
    rows = len(board)
    columns = len(board[0])
    if any(len(row) != columns for row in board):
        raise ValueError(f"Malformed Minesweeper board {puzzle_id}")

    variables = [
        (row, column)
        for row in range(rows)
        for column in range(columns)
        if board[row][column] == "?"
    ]
    variable_index = {coordinate: index for index, coordinate in enumerate(variables)}
    flags = {
        (row, column)
        for row in range(rows)
        for column in range(columns)
        if board[row][column] == "F"
    }
    clues: list[tuple[tuple[int, ...], int]] = []
    clue_variable_masks: list[int] = []
    for row in range(rows):
        for column in range(columns):
            if not board[row][column].isdigit():
                continue
            neighbors = [
                (other_row, other_column)
                for other_row in range(max(0, row - 1), min(rows, row + 2))
                for other_column in range(
                    max(0, column - 1), min(columns, column + 2)
                )
                if (other_row, other_column) != (row, column)
            ]
            indices = tuple(
                variable_index[coordinate]
                for coordinate in neighbors
                if coordinate in variable_index
            )
            residual = int(board[row][column]) - sum(
                coordinate in flags for coordinate in neighbors
            )
            if residual < 0 or residual > len(indices):
                raise ValueError(f"Invalid residual clue in {puzzle_id}")
            clues.append((indices, residual))
            clue_variable_masks.append(sum(1 << index for index in indices))

    assignment_ids = np.arange(1 << len(variables), dtype=np.uint32)
    assignments = (
        (assignment_ids[:, None] >> np.arange(len(variables), dtype=np.uint32))
        & 1
    ).astype(np.int8)
    satisfied_masks = np.zeros(len(assignments), dtype=np.uint32)
    for clue_index, (indices, residual) in enumerate(clues):
        if indices:
            satisfied = assignments[:, indices].sum(axis=1) == residual
        else:
            satisfied = np.full(len(assignments), residual == 0, dtype=bool)
        satisfied_masks |= satisfied.astype(np.uint32) << clue_index

    full_clue_mask = (1 << len(clues)) - 1
    fully_consistent = satisfied_masks == full_clue_mask
    if not fully_consistent.any():
        raise ValueError(f"No clue-consistent assignment for {puzzle_id}")

    desired = 0 if puzzle["machine_readable_instance"]["target_kind"] == "safe" else 1
    popcounts = np.fromiter(
        (mask.bit_count() for mask in range(1 << len(clues))),
        dtype=np.uint8,
        count=1 << len(clues),
    )
    raw: dict[str, tuple[float, ...]] = {}
    for solution in puzzle["solutions"]:
        solution_id = str(solution["abstract_solution_id"])
        row, column = (int(value) - 1 for value in solution["coordinate"])
        coordinate = (row, column)
        if coordinate not in variable_index:
            raise ValueError(f"Non-hidden Minesweeper candidate in {puzzle_id}")
        target_index = variable_index[coordinate]
        if not np.all(assignments[fully_consistent, target_index] == desired):
            raise ValueError(f"Candidate is not forced in {puzzle_id}, {solution_id}")

        wrong = assignments[:, target_index] != desired
        has_wrong_superset = np.zeros(1 << len(clues), dtype=bool)
        has_wrong_superset[np.unique(satisfied_masks[wrong])] = True
        for bit in range(len(clues)):
            step = 1 << bit
            blocks = has_wrong_superset.reshape(-1, 2 * step)
            blocks[:, :step] |= blocks[:, step:]

        certificate_masks = np.flatnonzero(~has_wrong_superset)
        if len(certificate_masks) == 0:
            raise ValueError(f"No certificate found for {puzzle_id}, {solution_id}")
        minimum_size = int(popcounts[certificate_masks].min())
        minimum_masks = certificate_masks[
            popcounts[certificate_masks] == minimum_size
        ]
        scopes: list[int] = []
        for certificate_mask in minimum_masks:
            variable_mask = 0
            for clue_index, clue_mask in enumerate(clue_variable_masks):
                if int(certificate_mask) & (1 << clue_index):
                    variable_mask |= clue_mask
            scopes.append(variable_mask.bit_count())
        minimum_scope = min(scopes)
        raw[solution_id] = (float(minimum_size), float(minimum_scope))
    return raw


def sudoku_features(puzzle: dict[str, Any]) -> dict[str, tuple[float, ...]]:
    puzzle_id = str(puzzle["id"])
    instance = puzzle["machine_readable_instance"]
    n = int(instance["n"])
    box_rows = int(instance["box_rows"])
    box_columns = int(instance["box_cols"])
    target = int(instance["digit"])
    board = [str(row) for row in instance["board"]]
    digits = set(range(1, n + 1))

    def box_cells(row: int, column: int) -> list[tuple[int, int]]:
        row_start = (row // box_rows) * box_rows
        column_start = (column // box_columns) * box_columns
        return [
            (other_row, other_column)
            for other_row in range(row_start, row_start + box_rows)
            for other_column in range(column_start, column_start + box_columns)
        ]

    empty_cells = [
        (row, column)
        for row in range(n)
        for column in range(n)
        if board[row][column] == "."
    ]
    domains: dict[tuple[int, int], set[int]] = {}
    for row, column in empty_cells:
        used = {
            int(board[row][other_column])
            for other_column in range(n)
            if board[row][other_column].isdigit()
        }
        used.update(
            int(board[other_row][column])
            for other_row in range(n)
            if board[other_row][column].isdigit()
        )
        used.update(
            int(board[other_row][other_column])
            for other_row, other_column in box_cells(row, column)
            if board[other_row][other_column].isdigit()
        )
        domains[(row, column)] = digits - used

    legal_target_cells = {
        coordinate for coordinate, domain in domains.items() if target in domain
    }
    stored_cells = {
        (int(solution["coordinate"][0]) - 1, int(solution["coordinate"][1]) - 1):
        str(solution["abstract_solution_id"])
        for solution in puzzle["solutions"]
    }
    if legal_target_cells != set(stored_cells):
        raise ValueError(f"Sudoku legal-cell mismatch in {puzzle_id}")

    raw: dict[str, tuple[float, ...]] = {}
    for coordinate, solution_id in stored_cells.items():
        row, column = coordinate
        units = [
            {(row, other_column) for other_column in range(n)},
            {(other_row, column) for other_row in range(n)},
            set(box_cells(row, column)),
        ]
        target_location_count = min(
            len(legal_target_cells & unit) for unit in units
        )
        raw[solution_id] = (
            float(len(domains[coordinate])),
            float(target_location_count),
        )
    return raw


def joint_design_diagnostics(
    puzzle_ids: list[str],
    features: dict[str, dict[str, tuple[float, ...]]],
) -> dict[str, Any]:
    dimension = len(next(iter(next(iter(features.values())).values())))

    def information_for(ids: Iterable[str]) -> np.ndarray:
        information = np.zeros((dimension, dimension))
        for puzzle_id in ids:
            matrix = np.asarray(list(features[puzzle_id].values()), dtype=float)
            mean = matrix.mean(axis=0)
            information += matrix.T @ matrix / len(matrix) - np.outer(mean, mean)
        return information

    full = information_for(puzzle_ids)
    eigenvalues = np.linalg.eigvalsh(full)
    if eigenvalues[0] <= 1e-12:
        raise ValueError("Pooled feature design is rank deficient")
    fold_eigenvalues: dict[str, list[float]] = {}
    for held_out in puzzle_ids:
        values = np.linalg.eigvalsh(
            information_for(puzzle_id for puzzle_id in puzzle_ids if puzzle_id != held_out)
        )
        if values[0] <= 1e-12:
            raise ValueError(f"LOPO feature design is rank deficient without {held_out}")
        fold_eigenvalues[held_out] = [float(value) for value in values]
    return {
        "pooled_uniform_choice_information": {
            "eigenvalues": [float(value) for value in eigenvalues],
            "condition_number": float(eigenvalues[-1] / eigenvalues[0]),
            "rank": int(np.linalg.matrix_rank(full)),
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


def load_family_features(
    project: Path, family: str
) -> tuple[list[str], dict[str, dict[str, tuple[float, ...]]], dict[str, Any]]:
    spec = FAMILY_SPECS[family]
    raw_feature_names = tuple(
        spec["raw_features"] if "raw_features" in spec else spec["features"]
    )
    regression_indices = tuple(
        spec.get("regression_indices", range(len(raw_feature_names)))
    )
    feature_names = tuple(raw_feature_names[index] for index in regression_indices)
    if feature_names != tuple(spec["features"]):
        raise ValueError(f"Regression-feature index mismatch for {spec['display']}")
    puzzle_ids: list[str] = []
    features: dict[str, dict[str, tuple[float, ...]]] = {}
    raw_ranges: dict[str, dict[str, list[float]]] = {
        feature: {} for feature in raw_feature_names
    }
    varying: dict[str, list[str]] = {feature: [] for feature in raw_feature_names}
    constant: dict[str, list[str]] = {feature: [] for feature in raw_feature_names}
    candidate_count = 0

    for puzzle in read_jsonl(project / "stimuli/all_puzzles.jsonl"):
        if puzzle.get("puzzle_type") != family:
            continue
        puzzle_id = str(puzzle["id"])
        if puzzle.get("puzzle_type") != family:
            raise ValueError(f"Unexpected family for {puzzle_id}")
        if family == "arithmetic24":
            raw = arithmetic_class_features(puzzle, project)
        elif family == "minesweeper_lite":
            raw = minesweeper_certificate_features(puzzle)
        elif family == "mini_sudoku":
            raw = sudoku_features(puzzle)
        else:
            raise ValueError(f"Unsupported family {family}")
        normalized_raw, low, high = minmax_normalize(raw)
        features[puzzle_id] = {
            solution_id: tuple(values[index] for index in regression_indices)
            for solution_id, values in normalized_raw.items()
        }
        puzzle_ids.append(puzzle_id)
        candidate_count += len(raw)
        for index, feature_name in enumerate(raw_feature_names):
            raw_ranges[feature_name][puzzle_id] = [
                float(low[index]),
                float(high[index]),
            ]
            target = varying if high[index] > low[index] else constant
            target[feature_name].append(puzzle_id)

    puzzle_ids.sort()
    if len(puzzle_ids) != 20:
        raise ValueError(f"Expected 20 {spec['display']} puzzles, found {len(puzzle_ids)}")
    observed_varying = tuple(len(varying[name]) for name in raw_feature_names)
    if observed_varying != tuple(spec["expected_raw_varying"]):
        raise ValueError(
            f"{spec['display']} variation audit changed: expected "
            f"{spec['expected_raw_varying']}, observed {observed_varying}"
        )
    expected_candidate_count = spec.get("expected_candidate_count")
    if expected_candidate_count is not None and candidate_count != expected_candidate_count:
        raise ValueError(
            f"{spec['display']} candidate audit changed: expected "
            f"{expected_candidate_count}, observed {candidate_count}"
        )
    diagnostics = {
        "features": {
            feature_name: {
                "varying_puzzle_count": len(varying[feature_name]),
                "constant_puzzle_count": len(constant[feature_name]),
                "varying_puzzle_ids": sorted(varying[feature_name]),
                "constant_puzzle_ids": sorted(constant[feature_name]),
                "raw_range_by_puzzle": raw_ranges[feature_name],
            }
            for feature_name in raw_feature_names
        },
        "candidate_count": candidate_count,
        "regression_feature_names": list(feature_names),
        "excluded_from_regression": [
            feature_name
            for index, feature_name in enumerate(raw_feature_names)
            if index not in regression_indices
        ],
        **joint_design_diagnostics(puzzle_ids, features),
    }
    return puzzle_ids, features, diagnostics
