from __future__ import annotations

import argparse
import os
import sys
from collections import Counter
from typing import Any, Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from puzzles import arithmetic24, minesweeper_lite, mini_sudoku
from puzzles.common import REQUIRED_FIELDS, read_jsonl, solution_count_in_range
from collection.stimuli import generate_study_modules as study_generation


EXPECTED_MODULES = {"cue", "spatial", "pair", "transfer", "formulation"}
EXPECTED_TYPES = {"arithmetic24", "minesweeper_lite", "mini_sudoku"}


def validate_modules(module_dir: str = "data/stimuli/modules") -> List[str]:
    errors: List[str] = []
    trial_path = os.path.join(module_dir, "all_module_trials.jsonl")
    block_path = os.path.join(module_dir, "module_blocks.jsonl")
    if not os.path.exists(trial_path):
        return [f"Missing {trial_path}"]
    if not os.path.exists(block_path):
        return [f"Missing {block_path}"]
    rows = read_jsonl(trial_path)
    blocks = read_jsonl(block_path)
    ids = set()
    for idx, row in enumerate(rows, 1):
        missing = REQUIRED_FIELDS - set(row)
        if missing:
            errors.append(f"Row {idx} missing required fields: {sorted(missing)}")
            continue
        if "module_metadata" not in row:
            errors.append(f"{row.get('id', idx)} missing module_metadata")
            continue
        pid = row["id"]
        if pid in ids:
            errors.append(f"Duplicate trial id: {pid}")
        ids.add(pid)
        ptype = row["puzzle_type"]
        module = row["module_metadata"].get("module")
        if ptype not in EXPECTED_TYPES:
            errors.append(f"{pid}: unsupported module puzzle type {ptype}")
        if module not in EXPECTED_MODULES:
            errors.append(f"{pid}: unsupported module {module}")
        if _is_single_solution_transfer_prime(row):
            if row["num_solutions"] != 1:
                errors.append(f"{pid}: transfer A_prime should have exactly one solution")
        elif not solution_count_in_range(int(row["num_solutions"])):
            errors.append(f"{pid}: solution count out of range; expected 3-8")
        if row["num_solutions"] != len(row["solutions"]):
            errors.append(f"{pid}: num_solutions mismatch")
        for sol in row["solutions"]:
            if not sol.get("solution_id") or not sol.get("abstract_solution_id"):
                errors.append(f"{pid}: solution missing solution_id or abstract_solution_id")
        errors.extend(_validate_exact_solutions(row))
        errors.extend(_validate_target(row))

    for block in blocks:
        for trial_id in block.get("trial_ids", []):
            if trial_id not in ids:
                errors.append(f"Block {block.get('block_id')} references unknown trial {trial_id}")
    counts = Counter(row["module_metadata"]["module"] for row in rows if "module_metadata" in row)
    for module in EXPECTED_MODULES:
        if counts[module] == 0:
            errors.append(f"No rows for module {module}")
    errors.extend(_validate_pair_design(rows))
    errors.extend(_validate_cue_design(rows))
    errors.extend(_validate_spatial_design(rows))
    errors.extend(_validate_formulation_design(rows))
    errors.extend(_validate_transfer_design(rows))
    return errors


def _is_single_solution_transfer_prime(row: Dict[str, Any]) -> bool:
    meta = row.get("module_metadata", {})
    return meta.get("module") == "transfer" and meta.get("role") == "A_prime"


def _validate_exact_solutions(row: Dict[str, Any]) -> List[str]:
    ptype = row["puzzle_type"]
    inst = row["machine_readable_instance"]
    if ptype == "minesweeper_lite":
        listed = {tuple(sol["coordinate"]) for sol in row["solutions"]}
        exact = {tuple(sol["coordinate"]) for sol in minesweeper_lite.forced_cells(inst["board"], inst["target_kind"])[0]}
    elif ptype == "mini_sudoku":
        listed = {tuple(sol["coordinate"]) for sol in row["solutions"]}
        exact = {tuple(sol["coordinate"]) for sol in mini_sudoku.enumerate_placements(inst["board"], inst["digit"])}
    elif ptype == "arithmetic24":
        exact = {sol["canonical_expression"] for sol in arithmetic24.solve_numbers(inst["numbers"])}
        listed = {sol["canonical_expression"] for sol in row["solutions"]}
        return [] if listed == exact else [f"{row['id']}: listed solutions do not match exact solver"]
    else:
        return []
    return [] if listed == exact else [f"{row['id']}: listed solutions do not match exact solver"]


def _validate_target(row: Dict[str, Any]) -> List[str]:
    meta = row.get("module_metadata", {})
    coords = meta.get("target_coordinates")
    if coords:
        valid_coords = {tuple(sol["coordinate"]) for sol in row["solutions"]}
        bad = [coord for coord in coords if tuple(coord) not in valid_coords]
        if bad:
            return [f"{row['id']}: target_coordinates contain invalid solutions: {bad}"]
        cue = row.get("machine_readable_instance", {}).get("visual_cue") or row.get("presentation_metadata", {}).get("visual_cue")
        cue_coords = cue.get("coordinates") if isinstance(cue, dict) else None
        if cue_coords and meta.get("module") == "cue":
            errors: List[str] = []
            cue_coord_set = {tuple(c) for c in cue_coords}
            target_set = {tuple(c) for c in coords}
            highlighted_valid = {tuple(c) for c in cue.get("highlighted_valid_coordinates", [])}
            highlighted_invalid = {tuple(c) for c in cue.get("highlighted_invalid_coordinates", [])}
            if highlighted_valid != target_set:
                errors.append(f"{row['id']}: highlighted_valid_coordinates do not match target_coordinates")
            if not highlighted_invalid:
                errors.append(f"{row['id']}: cue is missing invalid highlighted decoys")
            if highlighted_invalid & valid_coords:
                errors.append(f"{row['id']}: highlighted invalid decoys include valid solutions")
            if cue_coord_set != highlighted_valid | highlighted_invalid:
                errors.append(f"{row['id']}: cue coordinates are not the union of valid highlights and invalid decoys")
            if len(highlighted_valid) != len(highlighted_invalid):
                errors.append(f"{row['id']}: cue should highlight equal numbers of valid and invalid cells")
            return errors
        if cue_coords and {tuple(c) for c in cue_coords} != {tuple(c) for c in coords}:
            return [f"{row['id']}: cue coordinates do not match target_coordinates"]
        return []
    coord = meta.get("target_coordinate")
    if not coord:
        return []
    coords = {tuple(sol["coordinate"]) for sol in row["solutions"]}
    if tuple(coord) not in coords:
        return [f"{row['id']}: target_coordinate is not a valid solution"]
    return []


def _validate_transfer_design(rows: List[Dict[str, Any]]) -> List[str]:
    errors: List[str] = []
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        meta = row.get("module_metadata", {})
        if meta.get("module") == "transfer":
            groups.setdefault(meta.get("transfer_id", ""), []).append(row)
    for transfer_id, group in groups.items():
        by_role = {(row["module_metadata"]["condition"], row["module_metadata"]["role"]): row for row in group}
        prime_a = by_role.get(("strategy_prime", "A_prime"))
        prime_b = by_role.get(("strategy_prime", "B_target"))
        alone_b = by_role.get(("no_prime", "B_alone"))
        if not all([prime_a, prime_b, alone_b]):
            errors.append(f"{transfer_id}: transfer block must contain A_prime, B_target, and B_alone")
            continue
        if any(row["module_metadata"].get("condition") == "control_prime" for row in group):
            errors.append(f"{transfer_id}: old control_prime condition should not be present")
        if prime_a["num_solutions"] != 1:
            errors.append(f"{transfer_id}: A_prime is not single-solution")
        prime_witness = prime_a["module_metadata"].get("prime_strategy_witness")
        if not prime_witness:
            errors.append(f"{transfer_id}: A_prime missing prime_strategy_witness")
        elif prime_witness.get("target_coordinate") != prime_a["solutions"][0].get("coordinate"):
            errors.append(f"{transfer_id}: A_prime witness target does not match its single solution")
        if prime_b.get("source_puzzle_id") != alone_b.get("source_puzzle_id"):
            errors.append(f"{transfer_id}: prime B and alone B should be the same source puzzle")
        target_coords = prime_b["module_metadata"].get("target_coordinates") or []
        b_coords = {tuple(sol["coordinate"]) for sol in prime_b["solutions"]}
        if not target_coords:
            errors.append(f"{transfer_id}: B target strategy subset is empty")
        elif len({tuple(coord) for coord in target_coords}) != 1:
            errors.append(f"{transfer_id}: B target strategy subset should contain exactly one solution")
        elif len({tuple(coord) for coord in target_coords}) >= len(b_coords):
            errors.append(f"{transfer_id}: B target strategy subset covers all B solutions")
        if {tuple(coord) for coord in target_coords} != {
            tuple(coord) for coord in (alone_b["module_metadata"].get("target_coordinates") or [])
        }:
            errors.append(f"{transfer_id}: B target subset differs between prime and no-prime")
        target_witness = prime_b["module_metadata"].get("target_strategy_witness")
        if not target_witness:
            errors.append(f"{transfer_id}: B missing target_strategy_witness")
        elif target_coords and target_witness.get("target_coordinate") != target_coords[0]:
            errors.append(f"{transfer_id}: B witness target does not match target_coordinates")
        if target_witness:
            strategy = prime_b["module_metadata"].get("target_strategy_signature", "")
            match_coords = study_generation._transfer_strategy_match_coordinates(prime_b, strategy, target_witness)
            declared_match_coords = prime_b["module_metadata"].get("target_strategy_match_coordinates") or []
            if match_coords != target_coords:
                errors.append(
                    f"{transfer_id}: B has strategy-matched solutions {match_coords}, expected only {target_coords}"
                )
            if declared_match_coords and declared_match_coords != match_coords:
                errors.append(f"{transfer_id}: declared B strategy-match coordinates are stale")
            if alone_b["module_metadata"].get("target_strategy_match_coordinates") not in (None, [], match_coords):
                errors.append(f"{transfer_id}: no-prime B strategy-match coordinates differ from prime B")
        if target_witness != alone_b["module_metadata"].get("target_strategy_witness"):
            errors.append(f"{transfer_id}: B witness differs between prime and no-prime")
        match = prime_b["module_metadata"].get("transfer_witness_match", {})
        if not match.get("human_strategy_match"):
            errors.append(f"{transfer_id}: A and B target witnesses are not human-strategy matched")
        if prime_b["puzzle_type"] == "minesweeper_lite":
            relation = match.get("motif_relation", {})
            if not relation.get("dihedral_equivalent"):
                errors.append(f"{transfer_id}: Minesweeper A/B motifs are not rotation/reflection equivalent")
            if relation.get("exact_same"):
                errors.append(f"{transfer_id}: Minesweeper A/B motifs should not have the same orientation")
        quality = prime_b["module_metadata"].get("transfer_target_quality", {})
        if not quality.get("quality_ok"):
            errors.append(f"{transfer_id}: B target quality constraints failed")
        if quality != alone_b["module_metadata"].get("transfer_target_quality"):
            errors.append(f"{transfer_id}: B target quality metadata differs between prime and no-prime")
        if prime_a["puzzle_type"] in {"minesweeper_lite", "mini_sudoku"}:
            a_candidates = _candidate_count(prime_a)
            if a_candidates <= prime_a["num_solutions"]:
                errors.append(f"{transfer_id}: A_prime has no invalid answer candidates")
        if prime_b["puzzle_type"] in {"minesweeper_lite", "mini_sudoku"}:
            b_candidates = _candidate_count(prime_b)
            if b_candidates <= prime_b["num_solutions"]:
                errors.append(f"{transfer_id}: B_target has no invalid answer candidates")
        if prime_b["puzzle_type"] == "minesweeper_lite":
            invalid_b = _candidate_count(prime_b) - prime_b["num_solutions"]
            invalid_a = _candidate_count(prime_a) - prime_a["num_solutions"]
            if invalid_a < 3:
                errors.append(f"{transfer_id}: Minesweeper A_prime should have at least 3 invalid candidates")
            if invalid_b < 1:
                errors.append(f"{transfer_id}: Minesweeper B_target should have at least 1 invalid candidate")
    return errors


def _validate_cue_design(rows: List[Dict[str, Any]]) -> List[str]:
    errors: List[str] = []
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        meta = row.get("module_metadata", {})
        if meta.get("module") == "cue":
            groups.setdefault(row.get("family_id", ""), []).append(row)
    for family_id, group in groups.items():
        by_condition = {row["module_metadata"].get("condition"): row for row in group}
        required = {"baseline_no_cue", "cue_half_a", "cue_half_b"}
        if set(by_condition) != required:
            errors.append(f"{family_id}: cue family conditions are {sorted(by_condition)}")
            continue
        baseline = by_condition["baseline_no_cue"]
        cue_a = by_condition["cue_half_a"]
        cue_b = by_condition["cue_half_b"]
        baseline_valid = {tuple(sol["coordinate"]) for sol in baseline["solutions"]}
        a_valid, a_invalid, a_all = _cue_sets(cue_a)
        b_valid, b_invalid, b_all = _cue_sets(cue_b)
        if a_valid & b_valid:
            errors.append(f"{family_id}: cue A/B valid highlights overlap")
        if a_invalid & b_invalid:
            errors.append(f"{family_id}: cue A/B invalid highlights overlap")
        if a_all & b_all:
            errors.append(f"{family_id}: cue A/B total highlighted cells overlap")
        if a_valid | b_valid != baseline_valid:
            errors.append(f"{family_id}: cue A/B valid highlights do not cover all valid answers")
        if len(a_valid) != len(a_invalid) or len(b_valid) != len(b_invalid):
            errors.append(f"{family_id}: cue A/B valid and invalid highlight counts are not balanced")
    return errors


def _cue_sets(row: Dict[str, Any]) -> tuple[set[tuple[int, int]], set[tuple[int, int]], set[tuple[int, int]]]:
    cue = row.get("machine_readable_instance", {}).get("visual_cue") or row.get("presentation_metadata", {}).get("visual_cue") or {}
    valid = {tuple(coord) for coord in cue.get("highlighted_valid_coordinates", [])}
    invalid = {tuple(coord) for coord in cue.get("highlighted_invalid_coordinates", [])}
    all_highlighted = {tuple(coord) for coord in cue.get("coordinates", [])}
    return valid, invalid, all_highlighted


def _validate_pair_design(rows: List[Dict[str, Any]]) -> List[str]:
    errors: List[str] = []
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        meta = row.get("module_metadata", {})
        if meta.get("module") == "pair":
            groups.setdefault(meta.get("pair_id", ""), []).append(row)
    allowed_sudoku_primary_transforms = {"mirror_h", "mirror_v", "rotate180"}
    allowed_sudoku_light_tweaks = {None, "swap_rows_within_band", "swap_cols_within_stack"}
    for pair_id, group in groups.items():
        by_role = {(row["module_metadata"]["condition"], row["module_metadata"]["role"]): row for row in group}
        related_a = by_role.get(("related", "A"))
        related_b = by_role.get(("related", "B"))
        ctrl_a = by_role.get(("unrelated_control", "A"))
        ctrl_b = by_role.get(("unrelated_control", "B"))
        alone_b = by_role.get(("no_prime", "B_alone"))
        if not all([related_a, related_b, ctrl_a, ctrl_b, alone_b]):
            errors.append(f"{pair_id}: pair block must contain no-prime B, unrelated A->B, and related A->B")
            continue
        b_sources = {related_b.get("source_puzzle_id"), ctrl_b.get("source_puzzle_id"), alone_b.get("source_puzzle_id")}
        if len(b_sources) != 1:
            errors.append(f"{pair_id}: related/control/no-prime B should be the same B puzzle")
        if ctrl_a.get("source_family_id") == related_b.get("source_family_id"):
            errors.append(f"{pair_id}: unrelated-control A should come from a different family than B")
        if related_b["puzzle_type"] == "mini_sudoku":
            metadata = related_b.get("presentation_metadata", {})
            transform = metadata.get("formulation_transform")
            primary = metadata.get("formulation_transform_primary")
            light_tweak = metadata.get("formulation_transform_light_tweak")
            if primary not in allowed_sudoku_primary_transforms:
                errors.append(f"{pair_id}: Mini Sudoku pair B should use mirror/180 as primary transform, got {transform}")
            if light_tweak not in allowed_sudoku_light_tweaks:
                errors.append(f"{pair_id}: Mini Sudoku pair B has too much row/column permutation, got {light_tweak}")
            if light_tweak and "swap" not in str(light_tweak):
                errors.append(f"{pair_id}: Mini Sudoku pair B light tweak is not a local row/column swap")
            relation_map = related_b["module_metadata"].get("related_solution_map") or {}
            if len(relation_map) != related_b["num_solutions"]:
                errors.append(f"{pair_id}: Mini Sudoku pair B missing full abstract solution map")
    return errors


def _validate_spatial_design(rows: List[Dict[str, Any]]) -> List[str]:
    errors: List[str] = []
    groups: Dict[str, List[Dict[str, Any]]] = {}
    expected_conditions = {"original", "mirror_lr", "mirror_tb", "mirror_lr_tb"}
    for row in rows:
        meta = row.get("module_metadata", {})
        if meta.get("module") == "spatial":
            groups.setdefault(meta.get("spatial_id", ""), []).append(row)
    for spatial_id, group in groups.items():
        by_condition = {row["module_metadata"]["condition"]: row for row in group}
        if set(by_condition) != expected_conditions:
            errors.append(f"{spatial_id}: spatial family conditions are {sorted(by_condition)}")
            continue
        abstract_sets = {
            condition: {solution.get("abstract_solution_id") for solution in row["solutions"]}
            for condition, row in by_condition.items()
        }
        original_set = abstract_sets["original"]
        for condition, ids in abstract_sets.items():
            if ids != original_set:
                errors.append(f"{spatial_id}: {condition} abstract solution set differs from original")
        for condition, row in by_condition.items():
            transform = row["module_metadata"].get("spatial_transform")
            if condition == "original" and transform != "none":
                errors.append(f"{spatial_id}: original condition has transform {transform}")
            if condition != "original" and transform not in {"mirror_h", "mirror_v", "mirror_hv"}:
                errors.append(f"{spatial_id}: unsupported spatial transform {transform}")
            coord_map = row["machine_readable_instance"].get("coordinate_map_from_original")
            if not coord_map:
                errors.append(f"{row['id']}: missing coordinate_map_from_original")
    return errors


def _validate_formulation_design(rows: List[Dict[str, Any]]) -> List[str]:
    errors: List[str] = []
    groups: Dict[str, List[Dict[str, Any]]] = {}
    expected_conditions = {"original_order", "shuffled_order"}
    for row in rows:
        meta = row.get("module_metadata", {})
        if meta.get("module") == "formulation":
            groups.setdefault(meta.get("formulation_id", ""), []).append(row)
    for formulation_id, group in groups.items():
        by_condition = {row["module_metadata"]["condition"]: row for row in group}
        if set(by_condition) != expected_conditions:
            errors.append(f"{formulation_id}: formulation conditions are {sorted(by_condition)}")
            continue
        original = by_condition["original_order"]
        shuffled = by_condition["shuffled_order"]
        if original["puzzle_type"] != "arithmetic24" or shuffled["puzzle_type"] != "arithmetic24":
            errors.append(f"{formulation_id}: formulation module should use arithmetic24 only")
        if sorted(original["machine_readable_instance"]["numbers"]) != sorted(shuffled["machine_readable_instance"]["numbers"]):
            errors.append(f"{formulation_id}: original/shuffled number multisets differ")
        display = shuffled["machine_readable_instance"].get("display_numbers")
        if not display:
            errors.append(f"{formulation_id}: shuffled condition missing display_numbers")
        elif display == original["machine_readable_instance"]["numbers"]:
            errors.append(f"{formulation_id}: shuffled display order did not change")
        original_ids = {solution.get("abstract_solution_id") for solution in original["solutions"]}
        shuffled_ids = {solution.get("abstract_solution_id") for solution in shuffled["solutions"]}
        if original_ids != shuffled_ids:
            errors.append(f"{formulation_id}: abstract solution IDs differ between number-order conditions")
    return errors


def _candidate_count(row: Dict[str, Any]) -> int:
    board = row["machine_readable_instance"].get("board", [])
    if row["puzzle_type"] == "minesweeper_lite":
        return sum(ch == "?" for line in board for ch in line)
    if row["puzzle_type"] == "mini_sudoku":
        return sum(ch == "." for line in board for ch in line)
    return row["num_solutions"]


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate focused cue/spatial/pair/transfer study modules.")
    parser.add_argument("--module_dir", default="data/stimuli/modules")
    args = parser.parse_args()
    errors = validate_modules(args.module_dir)
    if errors:
        print("Module validation failed:")
        for error in errors:
            print(f"- {error}")
        return 1
    print("Module validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
