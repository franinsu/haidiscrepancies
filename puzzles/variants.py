from __future__ import annotations

import copy
import random
from typing import Any, Dict, List, Sequence, Tuple

from . import minesweeper_lite, mini_sudoku
from .common import Puzzle, stable_hash


Coord = Tuple[int, int]

VARIANT_SUFFIXES = {
    "canonical": "CAN",
    "formulation": "FORM",
}


def build_family_variants(
    base: Puzzle,
    family_id: str,
    rng: random.Random,
    include_formulation: bool = True,
) -> List[Puzzle]:
    abstract_ids = _abstract_ids(base, family_id)
    variants = [
        _finalize_variant(
            copy.deepcopy(base),
            family_id,
            "canonical",
            "canonical",
            "none",
            abstract_ids,
            {},
        )
    ]
    if include_formulation:
        variants.append(_formulation_variant(base, family_id, rng, abstract_ids))
    return variants


def _abstract_ids(base: Puzzle, family_id: str) -> Dict[str, str]:
    return {
        _solution_key(base["puzzle_type"], solution): f"{family_id}_abs_{idx:02d}"
        for idx, solution in enumerate(base["solutions"], 1)
    }


def _formulation_variant(
    base: Puzzle, family_id: str, rng: random.Random, abstract_ids: Dict[str, str]
) -> Puzzle:
    ptype = base["puzzle_type"]
    if ptype == "arithmetic24":
        row = copy.deepcopy(base)
        numbers = list(row["machine_readable_instance"]["numbers"])
        display = list(numbers)
        for _ in range(20):
            rng.shuffle(display)
            if display != numbers:
                break
        row["machine_readable_instance"]["display_numbers"] = display
        row["rendered_puzzle"] = "Numbers: " + ", ".join(map(str, display))
        row["prompt_text"] = (
            "Question {{ID}}:\n"
            "Use each of the numbers below exactly once, together with +, -, *, / and parentheses, to make 24.\n\n"
            f"{row['rendered_puzzle']}\n\nGive one valid expression."
        )
        return _finalize_variant(row, family_id, "formulation", "number_order_randomized", "number_order_randomized", abstract_ids, {})



    if ptype == "mini_sudoku":
        board = base["machine_readable_instance"]["board"]
        transform = "sudoku_row_col_permutation"
        new_board, coord_map = _permute_sudoku_rows_cols(board, rng)
        new_digit = base["machine_readable_instance"]["digit"]
        row = mini_sudoku.make_puzzle(new_board, new_digit)
        transformed_ids = _transformed_abstract_ids(base, abstract_ids, coord_map, {}, ptype)
        return _finalize_variant(row, family_id, "formulation", transform, transform, transformed_ids, {})

    if ptype == "minesweeper_lite":
        board = base["machine_readable_instance"]["board"]
        transform = rng.choice(["mirror_h", "mirror_v", "rotate90", "rotate180", "rotate270"])
        new_board, coord_map = _transform_char_grid(board, transform)
        row = minesweeper_lite.make_puzzle(new_board, base["machine_readable_instance"]["target_kind"])
        transformed_ids = _transformed_abstract_ids(base, abstract_ids, coord_map, {}, ptype)
        return _finalize_variant(row, family_id, "formulation", transform, transform, transformed_ids, {})

    raise ValueError(f"Unsupported formulation variant for {ptype}")


def _finalize_variant(
    row: Puzzle | None,
    family_id: str,
    variant_type: str,
    variant_name: str,
    formulation_transform: str,
    abstract_ids_by_solution_key: Dict[str, str],
    metadata_update: Dict[str, Any],
) -> Puzzle:
    if row is None:
        raise ValueError(f"Could not build {variant_type} variant for {family_id}")
    variant_id = f"{family_id}_{VARIANT_SUFFIXES[variant_type]}"
    row["id"] = variant_id
    row["prompt_text"] = row["prompt_text"].replace("{{ID}}", variant_id)
    row["family_id"] = family_id
    row["base_puzzle_id"] = family_id
    row["variant_id"] = variant_id
    row["variant_type"] = variant_type
    row["variant_name"] = variant_name
    row["presentation_metadata"] = {
        "formulation_transform": formulation_transform,
        "irrelevant_cue": None,
        "redundant_rule": None,
        "context_condition": "single",
    }
    row["presentation_metadata"].update(metadata_update)
    row["abstract_solution_map"] = {}
    for idx, solution in enumerate(row["solutions"], 1):
        solution_id = f"{variant_id}_sol_{idx:02d}"
        abstract_id = abstract_ids_by_solution_key.get(_solution_key(row["puzzle_type"], solution))
        if abstract_id is None:
            abstract_id = f"{family_id}_abs_unmapped_{idx:02d}"
        solution["solution_id"] = solution_id
        solution["abstract_solution_id"] = abstract_id
        row["abstract_solution_map"][solution_id] = abstract_id
    row["features"]["family_id"] = family_id
    row["features"]["variant_type"] = variant_type
    row["features"]["variant_name"] = variant_name
    row["features"]["variant_hash"] = stable_hash(
        {
            "family_id": family_id,
            "variant_type": variant_type,
            "instance": row["machine_readable_instance"],
            "presentation": row["presentation_metadata"],
        }
    )
    row["diversity_signature"]["variant_type"] = variant_type
    return row


def _solution_key(ptype: str, solution: Dict[str, Any]) -> str:
    if ptype == "arithmetic24":
        return solution["canonical_expression"]
    if ptype == "maze":
        return solution["moves"]
    if ptype == "grid_placement":
        return str(sorted(tuple(coord) for coord in solution["coordinates"]))
    if ptype in {"mini_sudoku", "minesweeper_lite"}:
        return str(tuple(solution["coordinate"]))
    raise ValueError(f"Unknown puzzle type: {ptype}")


def _transformed_abstract_ids(
    base: Puzzle,
    abstract_ids: Dict[str, str],
    coord_map: Dict[Coord, Coord],
    move_map: Dict[str, str],
    ptype: str,
) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for solution in base["solutions"]:
        old_key = _solution_key(ptype, solution)
        abstract_id = abstract_ids[old_key]
        if ptype == "maze":
            moves = "".join(move_map[ch] for ch in solution["moves"])
            out[moves] = abstract_id
        elif ptype == "grid_placement":
            coords = sorted(coord_map[tuple(coord)] for coord in solution["coordinates"])
            out[str(coords)] = abstract_id
        elif ptype in {"mini_sudoku", "minesweeper_lite"}:
            out[str(tuple(coord_map[tuple(solution["coordinate"])]))] = abstract_id
    return out




def _transform_char_grid(
    grid: Sequence[str], transform: str
) -> Tuple[List[str], Dict[Coord, Coord]]:
    rows, cols = len(grid), len(grid[0])
    if transform == "mirror_h":
        new = [row[::-1] for row in grid]
        coord_map = {(r + 1, c + 1): (r + 1, cols - c) for r in range(rows) for c in range(cols)}
        return new, coord_map
    if transform == "mirror_v":
        new = list(reversed(grid))
        coord_map = {(r + 1, c + 1): (rows - r, c + 1) for r in range(rows) for c in range(cols)}
        return new, coord_map
    if transform == "rotate90":
        new = ["".join(grid[rows - 1 - r][c] for r in range(rows)) for c in range(cols)]
        coord_map = {(r + 1, c + 1): (c + 1, rows - r) for r in range(rows) for c in range(cols)}
        return new, coord_map
    if transform == "rotate180":
        new = [row[::-1] for row in reversed(grid)]
        coord_map = {(r + 1, c + 1): (rows - r, cols - c) for r in range(rows) for c in range(cols)}
        return new, coord_map
    if transform == "rotate270":
        new = ["".join(grid[r][cols - 1 - c] for r in range(rows)) for c in range(cols)]
        coord_map = {(r + 1, c + 1): (cols - c, r + 1) for r in range(rows) for c in range(cols)}
        return new, coord_map
    raise ValueError(f"Unsupported char-grid transform: {transform}")




def _permute_sudoku_rows_cols(
    board: Sequence[str], rng: random.Random
) -> Tuple[List[str], Dict[Coord, Coord]]:
    n = len(board)
    br, bc = mini_sudoku.box_shape(n)
    row_order: List[int] = []
    band_order = list(range(n // br))
    rng.shuffle(band_order)
    for band in band_order:
        rows = list(range(band * br, band * br + br))
        rng.shuffle(rows)
        row_order.extend(rows)
    col_order: List[int] = []
    stack_order = list(range(n // bc))
    rng.shuffle(stack_order)
    for stack in stack_order:
        cols = list(range(stack * bc, stack * bc + bc))
        rng.shuffle(cols)
        col_order.extend(cols)
    old_to_new_row = {old + 1: new + 1 for new, old in enumerate(row_order)}
    old_to_new_col = {old + 1: new + 1 for new, old in enumerate(col_order)}
    new = ["".join(board[old_r][old_c] for old_c in col_order) for old_r in row_order]
    coord_map = {
        (r + 1, c + 1): (old_to_new_row[r + 1], old_to_new_col[c + 1])
        for r in range(n)
        for c in range(n)
    }
    return new, coord_map
