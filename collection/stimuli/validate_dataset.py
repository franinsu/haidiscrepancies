from __future__ import annotations

import argparse
import os
import sys
from collections import Counter
from typing import Any, Dict, List

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from puzzles import arithmetic24, grid_placement, maze, minesweeper_lite, mini_sudoku
from puzzles.common import REQUIRED_FIELDS, read_jsonl, solution_count_in_range, stable_hash


EXPECTED_TYPES = {"arithmetic24", "maze", "grid_placement", "minesweeper_lite", "mini_sudoku"}
FAMILY_FIELDS = {
    "family_id",
    "base_puzzle_id",
    "variant_id",
    "variant_type",
    "variant_name",
    "abstract_solution_map",
    "presentation_metadata",
}


def validate_dataset(out_dir: str, expected_per_type: int = 20, expected_base_per_type: int | None = None) -> List[str]:
    errors: List[str] = []
    path = os.path.join(out_dir, "all_puzzles.jsonl")
    if not os.path.exists(path):
        errors.append(f"Missing {path}")
        return errors
    rows = read_jsonl(path)
    counts = Counter(row.get("puzzle_type") for row in rows)
    if expected_base_per_type is None:
        for puzzle_type in EXPECTED_TYPES:
            if counts[puzzle_type] != expected_per_type:
                errors.append(f"Expected {expected_per_type} {puzzle_type} puzzles, found {counts[puzzle_type]}")
    else:
        for puzzle_type in EXPECTED_TYPES:
            families = {row.get("family_id") for row in rows if row.get("puzzle_type") == puzzle_type}
            if len(families) != expected_base_per_type:
                errors.append(f"Expected {expected_base_per_type} {puzzle_type} families, found {len(families)}")
    ids = set()
    instance_hashes = set()
    for idx, row in enumerate(rows, 1):
        missing = REQUIRED_FIELDS - set(row)
        family_missing = FAMILY_FIELDS - set(row)
        if missing:
            errors.append(f"Row {idx} missing fields: {sorted(missing)}")
            continue
        if family_missing:
            errors.append(f"Row {idx} missing family fields: {sorted(family_missing)}")
            continue
        pid = row["id"]
        if pid in ids:
            errors.append(f"Duplicate puzzle id: {pid}")
        ids.add(pid)
        if row["puzzle_type"] not in EXPECTED_TYPES:
            errors.append(f"{pid}: unknown puzzle type {row['puzzle_type']}")
        if not solution_count_in_range(int(row["num_solutions"])):
            errors.append(f"{pid}: num_solutions out of range: {row['num_solutions']}")
        if row["num_solutions"] != len(row["solutions"]):
            errors.append(f"{pid}: num_solutions does not match solution list length")
        ih = stable_hash({"type": row["puzzle_type"], "family": row.get("family_id"), "variant": row.get("variant_type"), "instance": row["machine_readable_instance"]})
        if ih in instance_hashes:
            errors.append(f"{pid}: duplicate machine-readable instance")
        instance_hashes.add(ih)
        for sol in row["solutions"]:
            if "abstract_solution_id" not in sol:
                errors.append(f"{pid}: solution missing abstract_solution_id")
            elif "unmapped" in str(sol["abstract_solution_id"]):
                errors.append(f"{pid}: solution has unmapped abstract_solution_id {sol['abstract_solution_id']}")
        errors.extend(_validate_row(row))
    errors.extend(_validate_family_abstract_solution_sets(rows))
    return errors


def _validate_family_abstract_solution_sets(rows: List[Dict[str, Any]]) -> List[str]:
    errors: List[str] = []
    by_family: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        by_family.setdefault(str(row.get("family_id")), []).append(row)
    for family_id, family_rows in by_family.items():
        by_variant = {row.get("variant_type"): row for row in family_rows}
        if "canonical" not in by_variant:
            errors.append(f"{family_id}: missing canonical variant")
            continue
        canonical_ids = {sol["abstract_solution_id"] for sol in by_variant["canonical"]["solutions"]}
        for variant_type, row in by_variant.items():
            ids = {sol["abstract_solution_id"] for sol in row["solutions"]}
            if ids != canonical_ids:
                errors.append(f"{row['id']}: abstract solution IDs differ from canonical family mapping")
    return errors


def _validate_row(row: Dict[str, Any]) -> List[str]:
    pid = row["id"]
    ptype = row["puzzle_type"]
    inst = row["machine_readable_instance"]
    errors: List[str] = []
    if ptype == "arithmetic24":
        numbers = inst["numbers"]
        recomputed = arithmetic24.solve_numbers(numbers)
        generated = {s["canonical_expression"] for s in row["solutions"]}
        exact = {s["canonical_expression"] for s in recomputed}
        if generated != exact:
            errors.append(f"{pid}: arithmetic solution list does not match exact solver")
        for sol in row["solutions"]:
            if not arithmetic24.validate_expression(sol["expression"], numbers, inst.get("target", 24)):
                errors.append(f"{pid}: invalid expression {sol['expression']}")
    elif ptype == "maze":
        grid = inst["grid"]
        length, paths, exceeded = maze.enumerate_shortest_paths(grid, max_paths=None)
        exact = {p["moves"] for p in paths}
        listed = {p["moves"] for p in row["solutions"]}
        if exceeded or listed != exact:
            errors.append(f"{pid}: maze solution list does not match exact shortest-path solver")
        for sol in row["solutions"]:
            if not maze.validate_path(grid, sol["moves"]):
                errors.append(f"{pid}: invalid shortest path {sol['moves']}")
    elif ptype == "grid_placement":
        board = inst["board"]
        m = inst["m"]
        exact = {tuple(tuple(c) for c in s["coordinates"]) for s in grid_placement.enumerate_placements(board, m)}
        listed = {tuple(tuple(c) for c in s["coordinates"]) for s in row["solutions"]}
        if listed != exact:
            errors.append(f"{pid}: grid-placement solution list does not match exact solver")
        for sol in row["solutions"]:
            if not grid_placement.validate_placement(board, m, sol["coordinates"]):
                errors.append(f"{pid}: invalid placement {sol['coordinates']}")
    elif ptype == "minesweeper_lite":
        board = inst["board"]
        target_kind = inst["target_kind"]
        exact = {tuple(s["coordinate"]) for s in minesweeper_lite.forced_cells(board, target_kind)[0]}
        listed = {tuple(s["coordinate"]) for s in row["solutions"]}
        if listed != exact:
            errors.append(f"{pid}: minesweeper solution list does not match exact solver")
        for sol in row["solutions"]:
            if not minesweeper_lite.validate_cell(board, target_kind, sol["coordinate"]):
                errors.append(f"{pid}: invalid minesweeper cell {sol['coordinate']}")
    elif ptype == "mini_sudoku":
        board = inst["board"]
        digit = inst["digit"]
        exact = {tuple(s["coordinate"]) for s in mini_sudoku.enumerate_placements(board, digit)}
        listed = {tuple(s["coordinate"]) for s in row["solutions"]}
        if listed != exact:
            errors.append(f"{pid}: mini-sudoku solution list does not match exact solver")
        for sol in row["solutions"]:
            if not mini_sudoku.validate_placement(board, digit, sol["coordinate"]):
                errors.append(f"{pid}: invalid mini-sudoku placement {sol['coordinate']}")
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate generated puzzle dataset.")
    parser.add_argument("--out_dir", default="data/stimuli")
    parser.add_argument("--expected_base_per_type", type=int, default=20)
    args = parser.parse_args(argv)
    errors = validate_dataset(args.out_dir, expected_base_per_type=args.expected_base_per_type)
    if errors:
        print("Validation failed:")
        for err in errors:
            print(f"- {err}")
        return 1
    print("Validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
