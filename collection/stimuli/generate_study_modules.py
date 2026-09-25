from __future__ import annotations

import argparse
import copy
import os
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

if __name__ == "__main__":
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from puzzles import arithmetic24, minesweeper_lite, mini_sudoku
from puzzles.common import (
    CUE_MIN_SOLUTIONS,
    MAX_MULTI_SOLUTIONS,
    MIN_MULTI_SOLUTIONS,
    Puzzle,
    read_jsonl,
    solution_count_in_range,
    stable_hash,
    write_jsonl,
)
from puzzles.probes import annotate_puzzle
from puzzles.variants import build_family_variants


MODULE_TYPES = ("minesweeper_lite", "mini_sudoku")
TRANSFER_STRATEGIES = {
    "minesweeper_lite": ("safe_by_satisfied_clue", "mine_by_exact_remaining"),
    "mini_sudoku": ("sudoku_row_elimination", "sudoku_box_elimination"),
}
TRANSFER_MIN_B_SOLUTIONS = 3
TRANSFER_MIN_TARGET_DISTANCE = 2
TRANSFER_MAX_B_SIMPLICITY_RANGE = 3.0
TRANSFER_MAX_TARGET_MEDIAN_GAP = 1.8
TRANSFER_MIN_PRIME_UNKNOWN_CELLS = 4
TRANSFER_MIN_B_INVALID_CANDIDATES = 1
TRANSFER_MS_B_VALID_MOTIFS = 3
SUDOKU_PAIR_TRANSFORMS = (
    ("mirror_h", None),
    ("mirror_v", None),
    ("rotate180", None),
    ("mirror_h", "swap_rows_within_band"),
    ("mirror_v", "swap_cols_within_stack"),
    ("rotate180", "swap_rows_within_band"),
)


def build_study_modules(
    data_path: str = "data/stimuli/all_puzzles.jsonl",
    out_dir: str = "data/stimuli/modules",
    per_type: int = 5,
    seed: int = 0,
) -> Dict[str, int]:
    _prepare_output_dir(out_dir)
    rng = random.Random(seed)
    main_rows = read_jsonl(data_path)
    main_hashes = _main_canonical_hashes(main_rows)
    by_type = _fresh_visual_module_sources(main_hashes, per_type, seed)
    cue_rows, cue_blocks = build_cue_module(by_type, per_type, rng)
    spatial_rows, spatial_blocks = build_spatial_module(by_type, per_type, rng)
    pair_rows, pair_blocks = build_pair_module(by_type, per_type, rng)
    transfer_rows, transfer_blocks = build_transfer_module(by_type, per_type, rng)
    formulation_rows, formulation_blocks = build_formulation_module(main_hashes, per_type * 2, rng, seed)

    all_rows = cue_rows + spatial_rows + pair_rows + transfer_rows + formulation_rows
    all_blocks = cue_blocks + spatial_blocks + pair_blocks + transfer_blocks + formulation_blocks
    write_jsonl(os.path.join(out_dir, "all_module_trials.jsonl"), all_rows)
    write_jsonl(os.path.join(out_dir, "module_blocks.jsonl"), all_blocks)
    return {
        "cue_trials": len(cue_rows),
        "spatial_trials": len(spatial_rows),
        "pair_trials": len(pair_rows),
        "transfer_trials": len(transfer_rows),
        "formulation_trials": len(formulation_rows),
        "all_trials": len(all_rows),
        "blocks": len(all_blocks),
    }


def _main_canonical_hashes(rows: Sequence[Puzzle]) -> Dict[str, set[str]]:
    hashes: Dict[str, set[str]] = defaultdict(set)
    for row in rows:
        h = row.get("features", {}).get("canonical_hash")
        if h:
            hashes[row["puzzle_type"]].add(h)
    return hashes


def _fresh_visual_module_sources(
    main_hashes: Dict[str, set[str]],
    per_type: int,
    seed: int,
) -> Dict[str, List[Puzzle]]:
    out: Dict[str, List[Puzzle]] = {}
    specs = {
        "minesweeper_lite": (minesweeper_lite.generate_pool, "MS", seed + 4400),
        "mini_sudoku": (mini_sudoku.generate_pool, "SUD", seed + 5500),
    }
    source_families = max(160, per_type * 24)
    for puzzle_type, (pool_fn, prefix, type_seed) in specs.items():
        pool = pool_fn(type_seed, valid_target=max(520, source_families * 6), min_attempts=3500)
        fresh = [
            row
            for row in pool
            if row.get("features", {}).get("canonical_hash") not in main_hashes.get(puzzle_type, set())
            and solution_count_in_range(int(row.get("num_solutions", 0)))
        ]
        fresh.sort(key=_visual_module_source_score, reverse=True)
        if len(fresh) < source_families:
            raise ValueError(f"Need {source_families} fresh {puzzle_type} module sources, found {len(fresh)}")
        rows: List[Puzzle] = []
        for idx, base in enumerate(fresh[:source_families], 1):
            family_id = f"MODULE_{prefix}_{idx:04d}"
            rows.extend(
                build_family_variants(
                    base,
                    family_id,
                    random.Random(type_seed * 1000 + idx),
                    include_formulation=True,
                )
            )
        out[puzzle_type] = rows
    return out


def _visual_module_source_score(row: Puzzle) -> Tuple[float, str]:
    f = row.get("features", {})
    score = 0.0
    score += 1.4 * float(f.get("solution_diversity_score", 0.0))
    score += 1.2 * float(f.get("spatial_contrast_score", 0.0))
    score += 0.9 * float(f.get("simplicity_contrast_score", 0.0))
    score += 0.7 * (1.0 - abs(float(row.get("num_solutions", 0)) - 6.0) / 3.0)
    score += 0.4 * float(f.get("human_difficulty_score", 40.0)) / 100.0
    return score, str(row.get("features", {}).get("canonical_hash", row.get("id", "")))


def build_formulation_module(
    main_hashes: Dict[str, set[str]],
    count: int,
    rng: random.Random,
    seed: int,
) -> Tuple[List[Puzzle], List[Dict[str, Any]]]:
    pool = arithmetic24.generate_pool(seed + 6600, valid_target=max(260, count * 8), min_attempts=800)
    fresh = [
        row
        for row in pool
        if row.get("features", {}).get("canonical_hash") not in main_hashes.get("arithmetic24", set())
        and solution_count_in_range(int(row.get("num_solutions", 0)))
    ]
    fresh.sort(key=_arithmetic_formulation_source_score, reverse=True)
    if len(fresh) < count:
        raise ValueError(f"Need {count} fresh arithmetic formulation sources, found {len(fresh)}")
    rows: List[Puzzle] = []
    blocks: List[Dict[str, Any]] = []
    for idx, base in enumerate(fresh[:count], 1):
        formulation_id = f"FORM_A24_{idx:04d}"
        variants = build_family_variants(
            base,
            formulation_id,
            random.Random(seed * 7000 + idx),
            include_formulation=True,
        )
        by_variant = {row["variant_type"]: row for row in variants}
        original = _mark_formulation_trial(
            by_variant["canonical"],
            formulation_id,
            "original_order",
            "original displayed number order",
        )
        shuffled = _mark_formulation_trial(
            by_variant["formulation"],
            formulation_id,
            "shuffled_order",
            "shuffled displayed number order",
        )
        rows.extend([original, shuffled])
        blocks.append(
            {
                "module": "formulation",
                "block_id": formulation_id,
                "puzzle_type": "arithmetic24",
                "trial_ids": [original["id"], shuffled["id"]],
                "design": "between-participant 24-point formulation sensitivity: original sorted number order vs shuffled displayed number order",
                "analysis": "compare distribution over abstract expression IDs after only changing the order in which the four numbers are displayed",
            }
        )
    return rows, blocks


def _arithmetic_formulation_source_score(row: Puzzle) -> Tuple[float, str]:
    f = row.get("features", {})
    score = 0.0
    score += 1.6 * (1.0 - abs(float(row.get("num_solutions", 0)) - 5.0) / 5.0)
    score += 1.2 * (1.0 - abs(float(f.get("min_expression_cost", 4.2)) - 4.2) / 3.0)
    score += 0.5 if not f.get("classic_overused", False) else -2.0
    score += 0.4 if len(f.get("operation_pattern_counts", {})) >= 2 else 0.0
    score += 0.3 * float(f.get("simplicity_contrast_score", 0.0))
    return score, str(f.get("canonical_hash", row.get("id", "")))


def _mark_formulation_trial(
    row: Puzzle,
    formulation_id: str,
    condition: str,
    readable: str,
) -> Puzzle:
    row = copy.deepcopy(row)
    row["module_metadata"] = {
        "module": "formulation",
        "formulation_id": formulation_id,
        "condition": condition,
        "role": "formulation_variant",
        "formulation_manipulation": "displayed_number_order",
        "formulation_readable": readable,
        "analysis": "same 24-point multiset and same abstract expression classes; only the displayed order of the four numbers changes",
    }
    row["source_puzzle_id"] = row["id"]
    row["source_family_id"] = formulation_id
    row["family_id"] = formulation_id
    row["base_puzzle_id"] = formulation_id
    row["variant_type"] = condition
    row["variant_name"] = condition
    row["presentation_metadata"] = dict(row.get("presentation_metadata", {}))
    row["presentation_metadata"].update(
        {
            "module": "formulation",
            "module_condition": condition,
            "module_role": "formulation_variant",
            "context_condition": "formulation_sensitivity",
        }
    )
    row["machine_readable_instance"]["formulation_module_condition"] = condition
    row["features"]["module"] = "formulation"
    row["features"]["module_condition"] = condition
    row["features"]["module_role"] = "formulation_variant"
    row["diversity_signature"]["module"] = "formulation"
    row["diversity_signature"]["module_condition"] = condition
    return row


def build_cue_module(
    by_type: Dict[str, List[Puzzle]], per_type: int, rng: random.Random
) -> Tuple[List[Puzzle], List[Dict[str, Any]]]:
    rows: List[Puzzle] = []
    blocks: List[Dict[str, Any]] = []
    for puzzle_type in MODULE_TYPES:
        candidates = _select_cue_candidates(_canonical_rows(by_type[puzzle_type]), per_type)
        for idx, base in enumerate(candidates, 1):
            module_family = f"CUE_{_prefix(puzzle_type)}_{idx:04d}"
            abstract_prefix = f"{module_family}_abs"
            selected = _cue_splits(base, rng)
            trial_ids = []
            baseline = _clone_trial(
                base,
                f"{module_family}_BASE",
                {
                    "module": "cue",
                    "module_family_id": module_family,
                    "condition": "baseline_no_cue",
                    "role": "baseline",
                    "analysis": "baseline target-solution mass for later cue contrasts",
                },
                abstract_prefix=abstract_prefix,
            )
            rows.append(baseline)
            trial_ids.append(baseline["id"])
            for cue_role, targets, invalid_decoys in selected:
                trial = _clone_trial(
                    base,
                    f"{module_family}_{cue_role.upper()}",
                    {
                        "module": "cue",
                        "module_family_id": module_family,
                        "condition": cue_role,
                        "role": "cue_trial",
                        "target_coordinates": [target["coordinate"] for target in targets],
                        "highlighted_valid_coordinates": [target["coordinate"] for target in targets],
                        "highlighted_invalid_coordinates": invalid_decoys,
                        "target_count": len(targets),
                        "invalid_decoy_count": len(invalid_decoys),
                        "analysis": "compare highlighted-valid solution mass against baseline while invalid highlighted cells test visual-cue salience",
                    },
                    abstract_prefix=abstract_prefix,
                    visual_cue=_visual_cue_set(targets, invalid_decoys, cue_role),
                )
                rows.append(trial)
                trial_ids.append(trial["id"])
            blocks.append(
                {
                    "module": "cue",
                    "block_id": module_family,
                    "puzzle_type": puzzle_type,
                    "trial_ids": trial_ids,
                    "design": "within-family baseline plus two cue sets; each cue set highlights one half of the valid answers plus matched invalid candidate cells",
                    "analysis": "P(choose highlighted-valid solution | cue condition) - P(choose same abstract solution set | baseline), with invalid highlights controlling for generic visual salience",
                }
            )
    return rows, blocks


def build_spatial_module(
    by_type: Dict[str, List[Puzzle]], per_type: int, rng: random.Random
) -> Tuple[List[Puzzle], List[Dict[str, Any]]]:
    rows: List[Puzzle] = []
    blocks: List[Dict[str, Any]] = []
    transforms = [
        ("original", "none", "original"),
        ("mirror_lr", "mirror_h", "left-right mirror"),
        ("mirror_tb", "mirror_v", "top-bottom mirror"),
        ("mirror_lr_tb", "mirror_hv", "left-right plus top-bottom mirror"),
    ]
    for puzzle_type in MODULE_TYPES:
        candidates = _select_spatial_candidates(_canonical_rows(by_type[puzzle_type]), per_type)
        for idx, base in enumerate(candidates, 1):
            spatial_id = f"SPATIAL_{_prefix(puzzle_type)}_{idx:04d}"
            trial_ids = []
            for condition, transform, readable in transforms:
                new_id = f"{spatial_id}_{_spatial_suffix(condition)}"
                meta = {
                    "module": "spatial",
                    "spatial_id": spatial_id,
                    "condition": condition,
                    "role": "spatial_variant",
                    "spatial_transform": transform,
                    "spatial_transform_readable": readable,
                    "analysis": "compare transformed solution distributions after mapping answers back to the same abstract solution IDs",
                }
                if transform == "none":
                    trial = _clone_trial(base, new_id, meta)
                    trial["presentation_metadata"]["spatial_transform"] = "none"
                    trial["machine_readable_instance"]["spatial_transform"] = "none"
                    trial["machine_readable_instance"]["coordinate_map_from_original"] = {
                        f"{solution['coordinate'][0]},{solution['coordinate'][1]}": solution["coordinate"]
                        for solution in trial["solutions"]
                    }
                else:
                    trial = _spatial_transform_trial(base, new_id, meta, transform)
                rows.append(trial)
                trial_ids.append(trial["id"])
            blocks.append(
                {
                    "module": "spatial",
                    "block_id": spatial_id,
                    "puzzle_type": puzzle_type,
                    "trial_ids": trial_ids,
                    "design": "between-participant spatial reformulation family: original, left-right mirror, top-bottom mirror, and combined mirror",
                    "analysis": "test whether solution distributions are equivariant under spatial transforms, or shifted by left/right/top/bottom display bias",
                }
            )
    return rows, blocks


def build_pair_module(
    by_type: Dict[str, List[Puzzle]], per_type: int, rng: random.Random
) -> Tuple[List[Puzzle], List[Dict[str, Any]]]:
    rows: List[Puzzle] = []
    blocks: List[Dict[str, Any]] = []
    for puzzle_type in MODULE_TYPES:
        groups = _families_with_variants(by_type[puzzle_type])
        families = _select_pair_families(groups, per_type)
        family_ids = [fid for fid, _ in families]
        for idx, (family_id, variants) in enumerate(families, 1):
            pair_id = f"PAIR_{_prefix(puzzle_type)}_{idx:04d}"
            related_a = variants["canonical"]
            related_b = (
                _simple_mini_sudoku_pair_variant(related_a, idx)
                if puzzle_type == "mini_sudoku"
                else variants["formulation"]
            )
            control_family = _control_family(groups, family_ids, family_id, idx)
            control_a = groups[control_family]["canonical"]
            related_map = _abstract_map(related_a, related_b)

            rel_a = _clone_trial(
                related_a,
                f"{pair_id}_REL_A",
                {
                    "module": "pair",
                    "pair_id": pair_id,
                    "condition": "related",
                    "role": "A",
                    "sequence_id": f"{pair_id}_REL",
                    "sequence_position": 1,
                    "relation_to_B": "same_family_formulation_pair",
                    "analysis": "solution_A should predict solution_B if answers are context-dependent",
                },
            )
            rel_b = _clone_trial(
                related_b,
                f"{pair_id}_REL_B",
                {
                    "module": "pair",
                    "pair_id": pair_id,
                    "condition": "related",
                    "role": "B",
                    "sequence_id": f"{pair_id}_REL",
                    "sequence_position": 2,
                    "relation_to_A": "same_family_formulation_pair",
                    "related_solution_map": related_map,
                    "analysis": "compare P(solution_B | solution_A, related) against unrelated control",
                },
            )
            alone_b = _clone_trial(
                related_b,
                f"{pair_id}_NO_PRIME_B",
                {
                    "module": "pair",
                    "pair_id": pair_id,
                    "condition": "no_prime",
                    "role": "B_alone",
                    "sequence_id": f"{pair_id}_NO_PRIME",
                    "sequence_position": 1,
                    "relation_to_A": "none",
                    "analysis": "baseline distribution for B without a preceding same-session puzzle",
                },
            )
            ctrl_a = _clone_trial(
                control_a,
                f"{pair_id}_CTRL_A",
                {
                    "module": "pair",
                    "pair_id": pair_id,
                    "condition": "unrelated_control",
                    "role": "A",
                    "sequence_id": f"{pair_id}_CTRL",
                    "sequence_position": 1,
                    "relation_to_B": "different_family_same_type_control",
                    "analysis": "control prior puzzle for independence baseline",
                },
            )
            ctrl_b = _clone_trial(
                related_b,
                f"{pair_id}_CTRL_B",
                {
                    "module": "pair",
                    "pair_id": pair_id,
                    "condition": "unrelated_control",
                    "role": "B",
                    "sequence_id": f"{pair_id}_CTRL",
                    "sequence_position": 2,
                    "relation_to_A": "different_family_same_type_control",
                    "analysis": "same B puzzle as related condition, preceded by unrelated A",
                },
            )
            rows.extend([alone_b, rel_a, rel_b, ctrl_a, ctrl_b])
            blocks.extend(
                [
                    _sequence_block("pair", f"{pair_id}_NO_PRIME", puzzle_type, [alone_b["id"]], "no_prime"),
                    _sequence_block("pair", f"{pair_id}_REL", puzzle_type, [rel_a["id"], rel_b["id"]], "related"),
                    _sequence_block("pair", f"{pair_id}_CTRL", puzzle_type, [ctrl_a["id"], ctrl_b["id"]], "unrelated_control"),
                ]
            )
    return rows, blocks


def build_transfer_module(
    by_type: Dict[str, List[Puzzle]], per_type: int, rng: random.Random
) -> Tuple[List[Puzzle], List[Dict[str, Any]]]:
    rows: List[Puzzle] = []
    blocks: List[Dict[str, Any]] = []
    for puzzle_type in MODULE_TYPES:
        canonical = _canonical_rows(by_type[puzzle_type])
        plans = _select_transfer_plans(puzzle_type, canonical, per_type, rng)
        for idx, plan in enumerate(plans, 1):
            transfer_id = f"TRANSFER_{_prefix(puzzle_type)}_{idx:04d}"
            target_b = plan["target_b"]
            prime_a = plan["prime_a"]
            strategy = plan["strategy_signature"]
            target_coordinates = plan["target_coordinates"]
            prime_witness = plan["prime_strategy_witness"]
            target_witness = plan["target_strategy_witness"]
            witness_match = plan["witness_match"]
            target_quality = plan["transfer_target_quality"]

            prime_a_row = _clone_trial(
                prime_a,
                f"{transfer_id}_PRIME_A",
                {
                    "module": "transfer",
                    "transfer_id": transfer_id,
                    "condition": "strategy_prime",
                    "role": "A_prime",
                    "sequence_id": f"{transfer_id}_PRIME",
                    "sequence_position": 1,
                    "prime_strategy_signature": strategy,
                    "prime_strategy_witness": prime_witness,
                    "transfer_witness_match": witness_match,
                    "prime_is_single_solution": True,
                    "analysis": "single-solution A exposes one clear strategy before the measured B puzzle",
                },
                abstract_prefix=f"{transfer_id}_prime_abs",
            )
            prime_b_row = _clone_trial(
                target_b,
                f"{transfer_id}_PRIME_B",
                {
                    "module": "transfer",
                    "transfer_id": transfer_id,
                    "condition": "strategy_prime",
                    "role": "B_target",
                    "sequence_id": f"{transfer_id}_PRIME",
                    "sequence_position": 2,
                    "target_coordinates": target_coordinates,
                    "target_strategy_signature": strategy,
                    "target_strategy_witness": target_witness,
                    "target_strategy_match_coordinates": plan["target_strategy_match_coordinates"],
                    "non_target_strategy_fingerprints": plan["non_target_strategy_fingerprints"],
                    "transfer_witness_match": witness_match,
                    "transfer_target_quality": target_quality,
                    "target_strategy_solution_count": len(target_coordinates),
                    "analysis": "measure whether A increases the probability of choosing B answers supported by the primed strategy",
                },
                abstract_prefix=f"{transfer_id}_b_abs",
            )
            alone_b_row = _clone_trial(
                target_b,
                f"{transfer_id}_ALONE_B",
                {
                    "module": "transfer",
                    "transfer_id": transfer_id,
                    "condition": "no_prime",
                    "role": "B_alone",
                    "sequence_id": f"{transfer_id}_ALONE",
                    "sequence_position": 1,
                    "target_coordinates": target_coordinates,
                    "target_strategy_signature": strategy,
                    "target_strategy_witness": target_witness,
                    "target_strategy_match_coordinates": plan["target_strategy_match_coordinates"],
                    "non_target_strategy_fingerprints": plan["non_target_strategy_fingerprints"],
                    "transfer_witness_match": witness_match,
                    "transfer_target_quality": target_quality,
                    "target_strategy_solution_count": len(target_coordinates),
                    "analysis": "baseline B distribution without an immediately preceding strategy prime",
                },
                abstract_prefix=f"{transfer_id}_b_abs",
            )
            rows.extend([prime_a_row, prime_b_row, alone_b_row])
            blocks.extend(
                [
                    _sequence_block("transfer", f"{transfer_id}_PRIME", puzzle_type, [prime_a_row["id"], prime_b_row["id"]], "strategy_prime"),
                    _sequence_block("transfer", f"{transfer_id}_ALONE", puzzle_type, [alone_b_row["id"]], "no_prime"),
                ]
            )
    return rows, blocks


def _clone_trial(
    source: Puzzle,
    new_id: str,
    module_metadata: Dict[str, Any],
    visual_cue: Dict[str, Any] | None = None,
    abstract_prefix: str | None = None,
) -> Puzzle:
    row = copy.deepcopy(source)
    original_id = row["id"]
    row["id"] = new_id
    row["prompt_text"] = _replace_prompt_id(row["prompt_text"], new_id)

    source_family_id = row.get("family_id")
    row["source_puzzle_id"] = original_id
    row["source_family_id"] = source_family_id
    row["family_id"] = module_metadata.get("module_family_id") or module_metadata.get("pair_id") or module_metadata.get("transfer_id") or source_family_id
    row["base_puzzle_id"] = row["family_id"]
    row["variant_id"] = new_id
    row["variant_type"] = module_metadata.get("condition", row.get("variant_type", "module"))
    row["variant_name"] = module_metadata.get("role", row.get("variant_name", "module"))
    row["module_metadata"] = dict(module_metadata)
    row["presentation_metadata"] = dict(row.get("presentation_metadata", {}))
    row["presentation_metadata"].update(
        {
            "module": module_metadata.get("module"),
            "module_condition": module_metadata.get("condition"),
            "module_role": module_metadata.get("role"),
            "context_condition": module_metadata.get("condition", "module"),
        }
    )
    if visual_cue:
        row["presentation_metadata"]["visual_cue"] = visual_cue
        row["machine_readable_instance"]["visual_cue"] = visual_cue
        row["machine_readable_instance"]["irrelevant_cue"] = visual_cue

    row["abstract_solution_map"] = {}
    for idx, solution in enumerate(row["solutions"], 1):
        solution["solution_id"] = f"{new_id}_sol_{idx:02d}"
        if abstract_prefix:
            solution["abstract_solution_id"] = f"{abstract_prefix}_{idx:02d}"
        row["abstract_solution_map"][solution["solution_id"]] = solution.get("abstract_solution_id")

    _attach_target_solution_metadata(row)
    row["features"]["module"] = module_metadata.get("module")
    row["features"]["module_condition"] = module_metadata.get("condition")
    row["features"]["module_role"] = module_metadata.get("role")
    row["features"]["variant_hash"] = stable_hash(
        {
            "id": new_id,
            "source": original_id,
            "module_metadata": row["module_metadata"],
            "presentation": row["presentation_metadata"],
        }
    )
    row["diversity_signature"]["module"] = module_metadata.get("module")
    row["diversity_signature"]["module_condition"] = module_metadata.get("condition")
    return row


def _attach_target_solution_metadata(row: Puzzle) -> None:
    coords = row["module_metadata"].get("target_coordinates")
    if coords:
        targets = []
        for coord in coords:
            solution = _solution_by_coordinate(row, coord)
            if solution is not None:
                targets.append(solution)
        if targets:
            target = {
                "target_solution_ids": [solution["solution_id"] for solution in targets],
                "target_abstract_solution_ids": [solution.get("abstract_solution_id") for solution in targets],
                "target_coordinates": [solution["coordinate"] for solution in targets],
                "target_simplicity_ranks": [
                    solution["probe_features"].get("simplicity_rank_within_puzzle") for solution in targets
                ],
                "target_strategy_classes": [
                    solution["probe_features"].get("solution_strategy_class") for solution in targets
                ],
                "target_strategy_signatures": [
                    _strategy_signature(row["puzzle_type"], solution) for solution in targets
                ],
            }
            row["module_metadata"].update(target)
            if "visual_cue" in row["presentation_metadata"]:
                row["presentation_metadata"]["visual_cue"].update(target)
                row["machine_readable_instance"]["visual_cue"] = row["presentation_metadata"]["visual_cue"]
                row["machine_readable_instance"]["irrelevant_cue"] = row["presentation_metadata"]["visual_cue"]
        return

    coord = row["module_metadata"].get("target_coordinate")
    if not coord:
        cue = row.get("presentation_metadata", {}).get("visual_cue")
        coord = cue.get("coordinate") if isinstance(cue, dict) else None
    if not coord:
        return
    solution = _solution_by_coordinate(row, coord)
    if solution is None:
        return
    target = {
        "target_solution_id": solution["solution_id"],
        "target_abstract_solution_id": solution.get("abstract_solution_id"),
        "target_coordinate": solution["coordinate"],
        "target_simplicity_rank": solution["probe_features"].get("simplicity_rank_within_puzzle"),
        "target_strategy_class": solution["probe_features"].get("solution_strategy_class"),
        "target_strategy_signature": _strategy_signature(row["puzzle_type"], solution),
    }
    row["module_metadata"].update({k: v for k, v in target.items() if row["module_metadata"].get(k) is None or k not in row["module_metadata"]})
    if "visual_cue" in row["presentation_metadata"]:
        row["presentation_metadata"]["visual_cue"].update(target)
        row["machine_readable_instance"]["visual_cue"] = row["presentation_metadata"]["visual_cue"]
        row["machine_readable_instance"]["irrelevant_cue"] = row["presentation_metadata"]["visual_cue"]


def _replace_prompt_id(prompt: str, new_id: str) -> str:
    lines = prompt.splitlines()
    if lines and lines[0].startswith("Question "):
        lines[0] = f"Question {new_id}:"
        return "\n".join(lines)
    return f"Question {new_id}:\n" + prompt


def _visual_cue_set(
    solutions: Sequence[Dict[str, Any]],
    invalid_decoys: Sequence[Sequence[int]],
    cue_role: str,
) -> Dict[str, Any]:
    valid_coords = [solution["coordinate"] for solution in solutions]
    invalid_coords = [list(coord) for coord in invalid_decoys]
    all_coords = sorted(valid_coords + invalid_coords)
    return {
        "kind": "cell_highlight_set",
        "rule_relevance": "irrelevant",
        "coordinates": all_coords,
        "highlighted_valid_coordinates": valid_coords,
        "highlighted_invalid_coordinates": invalid_coords,
        "cue_role": cue_role,
        "cue_strength": "soft",
        "description": "border outlines around a mix of legal answer cells and non-answer candidate cells",
    }


def _cue_splits(base: Puzzle, rng: random.Random) -> List[Tuple[str, List[Dict[str, Any]], List[List[int]]]]:
    sols = list(base["solutions"])
    rng.shuffle(sols)
    split = (len(sols) + 1) // 2
    first = sorted(sols[:split], key=lambda sol: sol["coordinate"])
    second = sorted(sols[split:], key=lambda sol: sol["coordinate"])
    invalid = _invalid_candidate_coordinates(base)
    if len(invalid) < len(first) + len(second):
        raise ValueError(f"{base['id']}: not enough disjoint invalid cue decoys")
    rng.shuffle(invalid)
    first_invalid = sorted(invalid[: len(first)])
    second_invalid = sorted(invalid[len(first) : len(first) + len(second)])
    return [("cue_half_a", first, first_invalid), ("cue_half_b", second, second_invalid)]


def _invalid_candidate_coordinates(base: Puzzle) -> List[List[int]]:
    valid = {tuple(solution["coordinate"]) for solution in base["solutions"]}
    inst = base["machine_readable_instance"]
    board = inst["board"]
    invalid: List[List[int]] = []
    if base["puzzle_type"] == "minesweeper_lite":
        for r, row in enumerate(board, 1):
            for c, ch in enumerate(row, 1):
                if ch == "?" and (r, c) not in valid:
                    invalid.append([r, c])
    elif base["puzzle_type"] == "mini_sudoku":
        for r, row in enumerate(board, 1):
            for c, ch in enumerate(row, 1):
                if ch == "." and (r, c) not in valid:
                    invalid.append([r, c])
    else:
        raise ValueError(f"Unsupported cue puzzle type: {base['puzzle_type']}")
    return invalid


def _select_cue_candidates(rows: List[Puzzle], count: int) -> List[Puzzle]:
    candidates = [
        row
        for row in rows
        if solution_count_in_range(int(row["num_solutions"]), CUE_MIN_SOLUTIONS, MAX_MULTI_SOLUTIONS)
        and len(_invalid_candidate_coordinates(row)) >= row["num_solutions"]
        and row["features"].get("solution_diversity_score", 0.0) >= 0.5
    ]
    candidates.sort(
        key=lambda row: (
            -row["features"].get("solution_diversity_score", 0.0),
            -row["features"].get("spatial_contrast_score", 0.0),
            abs(float(row["num_solutions"]) - 6.0),
            row["id"],
        )
    )
    if len(candidates) < count:
        raise ValueError(f"Need {count} cue candidates, found {len(candidates)}")
    return candidates[:count]


def _select_spatial_candidates(rows: List[Puzzle], count: int) -> List[Puzzle]:
    candidates = []
    for row in rows:
        if not solution_count_in_range(int(row.get("num_solutions", 0))):
            continue
        solution_coords = [tuple(solution["coordinate"]) for solution in row["solutions"]]
        if len(solution_coords) < 3:
            continue
        board = row["machine_readable_instance"]["board"]
        rows_n, cols_n = len(board), len(board[0])
        has_left = any(c <= cols_n / 2 for _, c in solution_coords)
        has_right = any(c > cols_n / 2 for _, c in solution_coords)
        has_top = any(r <= rows_n / 2 for r, _ in solution_coords)
        has_bottom = any(r > rows_n / 2 for r, _ in solution_coords)
        if not (has_left and has_right and has_top and has_bottom):
            continue
        if row["features"].get("spatial_contrast_score", 0.0) < 1.0:
            continue
        candidates.append(row)
    candidates.sort(
        key=lambda row: (
            -row["features"].get("spatial_contrast_score", 0.0),
            -row["features"].get("solution_diversity_score", 0.0),
            abs(float(row["num_solutions"]) - 6.0),
            row["id"],
        )
    )
    if len(candidates) < count:
        raise ValueError(f"Need {count} spatial candidates, found {len(candidates)}")
    return candidates[:count]


def _spatial_transform_trial(
    source: Puzzle,
    new_id: str,
    module_metadata: Dict[str, Any],
    transform: str,
) -> Puzzle:
    inst = source["machine_readable_instance"]
    board, coord_map = _transform_rect_board_with_coord_map(inst["board"], transform)
    if source["puzzle_type"] == "minesweeper_lite":
        puzzle = minesweeper_lite.make_puzzle(board, inst["target_kind"])
    elif source["puzzle_type"] == "mini_sudoku":
        puzzle = mini_sudoku.make_puzzle(board, int(inst["digit"]))
    else:
        raise ValueError(f"Unsupported spatial puzzle type: {source['puzzle_type']}")
    if puzzle is None:
        raise ValueError(f"{source['id']}: spatial transform {transform} no longer has 3-8 solutions")
    if puzzle["num_solutions"] != source["num_solutions"]:
        raise ValueError(f"{source['id']}: spatial transform {transform} changed solution count")

    old_to_abs = {
        tuple(solution["coordinate"]): solution.get("abstract_solution_id")
        for solution in source["solutions"]
    }
    new_to_abs = {
        coord_map[old_coord]: abstract_id
        for old_coord, abstract_id in old_to_abs.items()
        if old_coord in coord_map and abstract_id
    }
    solution_coords = {tuple(solution["coordinate"]) for solution in puzzle["solutions"]}
    if solution_coords != set(new_to_abs):
        raise ValueError(f"{source['id']}: spatial transform {transform} changed abstract solution coordinates")

    row = copy.deepcopy(puzzle)
    original_id = source["id"]
    row["id"] = new_id
    row["prompt_text"] = _replace_prompt_id(row["prompt_text"], new_id)
    row["source_puzzle_id"] = original_id
    row["source_family_id"] = source.get("family_id")
    row["family_id"] = module_metadata["spatial_id"]
    row["base_puzzle_id"] = module_metadata["spatial_id"]
    row["variant_id"] = new_id
    row["variant_type"] = module_metadata["condition"]
    row["variant_name"] = "spatial_variant"
    row["module_metadata"] = dict(module_metadata)
    row["presentation_metadata"] = dict(source.get("presentation_metadata", {}))
    row["presentation_metadata"].update(
        {
            "module": "spatial",
            "module_condition": module_metadata["condition"],
            "module_role": module_metadata["role"],
            "context_condition": "spatial_reformulation",
            "spatial_transform": transform,
            "spatial_transform_readable": module_metadata["spatial_transform_readable"],
            "irrelevant_cue": None,
            "redundant_rule": None,
        }
    )
    row["machine_readable_instance"]["spatial_transform"] = transform
    row["machine_readable_instance"]["spatial_transform_readable"] = module_metadata["spatial_transform_readable"]
    row["machine_readable_instance"]["coordinate_map_from_original"] = {
        f"{r},{c}": [nr, nc]
        for (r, c), (nr, nc) in sorted(coord_map.items())
    }

    row["abstract_solution_map"] = {}
    for idx, solution in enumerate(row["solutions"], 1):
        solution["solution_id"] = f"{new_id}_sol_{idx:02d}"
        solution["abstract_solution_id"] = new_to_abs[tuple(solution["coordinate"])]
        row["abstract_solution_map"][solution["solution_id"]] = solution["abstract_solution_id"]
    row["features"]["module"] = "spatial"
    row["features"]["module_condition"] = module_metadata["condition"]
    row["features"]["module_role"] = module_metadata["role"]
    row["features"]["spatial_transform"] = transform
    row["features"]["variant_hash"] = stable_hash(
        {
            "id": new_id,
            "source": original_id,
            "module_metadata": row["module_metadata"],
            "machine_readable_instance": row["machine_readable_instance"],
        }
    )
    row["diversity_signature"]["module"] = "spatial"
    row["diversity_signature"]["module_condition"] = module_metadata["condition"]
    row["diversity_signature"]["spatial_transform"] = transform
    return row


def _transform_rect_board_with_coord_map(
    board: Sequence[str],
    transform: str,
) -> Tuple[List[str], Dict[Tuple[int, int], Tuple[int, int]]]:
    rows, cols = len(board), len(board[0])
    chars = [list(row) for row in board]
    if transform == "mirror_h":
        transformed = ["".join(reversed(row)) for row in chars]
        coord_map = {(r + 1, c + 1): (r + 1, cols - c) for r in range(rows) for c in range(cols)}
    elif transform == "mirror_v":
        transformed = ["".join(row) for row in reversed(chars)]
        coord_map = {(r + 1, c + 1): (rows - r, c + 1) for r in range(rows) for c in range(cols)}
    elif transform == "mirror_hv":
        transformed = ["".join(reversed(row)) for row in reversed(chars)]
        coord_map = {(r + 1, c + 1): (rows - r, cols - c) for r in range(rows) for c in range(cols)}
    else:
        raise ValueError(f"Unsupported spatial transform: {transform}")
    return transformed, coord_map


def _spatial_suffix(condition: str) -> str:
    return {
        "original": "ORIG",
        "mirror_lr": "LR",
        "mirror_tb": "TB",
        "mirror_lr_tb": "LR_TB",
    }[condition]


def _select_pair_families(groups: Dict[str, Dict[str, Puzzle]], count: int) -> List[Tuple[str, Dict[str, Puzzle]]]:
    candidates = [
        (family_id, variants)
        for family_id, variants in sorted(groups.items())
        if "canonical" in variants
        and "formulation" in variants
        and solution_count_in_range(int(variants["canonical"]["num_solutions"]))
        and solution_count_in_range(int(variants["formulation"]["num_solutions"]))
    ]
    candidates.sort(
        key=lambda item: (
            -item[1]["canonical"]["features"].get("spatial_contrast_score", 0.0),
            -item[1]["canonical"]["features"].get("simplicity_contrast_score", 0.0),
            item[0],
        )
    )
    if len(candidates) < count:
        raise ValueError(f"Need {count} pair candidates, found {len(candidates)}")
    return candidates[:count]


def _simple_mini_sudoku_pair_variant(source: Puzzle, idx: int) -> Puzzle:
    for offset in range(len(SUDOKU_PAIR_TRANSFORMS)):
        transform, light_tweak = SUDOKU_PAIR_TRANSFORMS[(idx - 1 + offset) % len(SUDOKU_PAIR_TRANSFORMS)]
        variant = _make_simple_mini_sudoku_pair_variant(source, transform, light_tweak, idx)
        if variant is not None:
            return variant
    raise ValueError(f"Could not build simple pair formulation for {source['id']}")


def _make_simple_mini_sudoku_pair_variant(
    source: Puzzle,
    transform: str,
    light_tweak: str | None,
    idx: int,
) -> Puzzle | None:
    inst = source["machine_readable_instance"]
    board = inst["board"]
    digit = int(inst["digit"])
    transformed_board, coord_map = _transform_square_board_with_coord_map(board, transform)
    tweak_detail = None
    if light_tweak:
        transformed_board, tweak_map, tweak_detail = _light_sudoku_pair_tweak(
            transformed_board,
            light_tweak,
            idx,
        )
        coord_map = {
            old_coord: tweak_map[after_primary]
            for old_coord, after_primary in coord_map.items()
        }
    puzzle = mini_sudoku.make_puzzle(transformed_board, digit)
    if puzzle is None or puzzle["num_solutions"] != source["num_solutions"]:
        return None

    old_to_abs = {
        tuple(solution["coordinate"]): solution.get("abstract_solution_id")
        for solution in source["solutions"]
    }
    new_to_abs = {
        coord_map[old_coord]: abstract_id
        for old_coord, abstract_id in old_to_abs.items()
        if old_coord in coord_map and abstract_id
    }
    solution_coords = {tuple(solution["coordinate"]) for solution in puzzle["solutions"]}
    if solution_coords != set(new_to_abs):
        return None

    for solution in puzzle["solutions"]:
        solution["abstract_solution_id"] = new_to_abs[tuple(solution["coordinate"])]
    puzzle["abstract_solution_map"] = {
        f"{source['family_id']}_PAIR_{_pair_transform_id(transform, light_tweak)}_sol_{sol_idx:02d}": solution["abstract_solution_id"]
        for sol_idx, solution in enumerate(puzzle["solutions"], 1)
    }
    transform_id = _pair_transform_id(transform, light_tweak)
    puzzle["id"] = f"{source['id']}_PAIR_{transform_id.upper()}"
    puzzle["family_id"] = source["family_id"]
    puzzle["base_puzzle_id"] = source.get("base_puzzle_id", source["family_id"])
    puzzle["variant_id"] = puzzle["id"]
    puzzle["variant_type"] = "formulation"
    puzzle["variant_name"] = f"pair_{transform_id}"
    puzzle["presentation_metadata"] = dict(source.get("presentation_metadata", {}))
    puzzle["presentation_metadata"].update(
        {
            "context_condition": "pair_related_reformulation",
            "formulation_transform": f"pair_{transform_id}",
            "formulation_transform_primary": transform,
            "formulation_transform_light_tweak": light_tweak,
            "formulation_transform_tweak_detail": tweak_detail,
            "formulation_transform_readable": _readable_pair_transform(transform, light_tweak),
            "irrelevant_cue": None,
            "redundant_rule": None,
        }
    )
    puzzle["machine_readable_instance"]["formulation_transform"] = f"pair_{transform_id}"
    puzzle["machine_readable_instance"]["formulation_transform_primary"] = transform
    puzzle["machine_readable_instance"]["formulation_transform_light_tweak"] = light_tweak
    puzzle["machine_readable_instance"]["formulation_transform_tweak_detail"] = tweak_detail
    puzzle["machine_readable_instance"]["coordinate_map_from_canonical"] = {
        f"{r},{c}": [nr, nc]
        for (r, c), (nr, nc) in sorted(coord_map.items())
    }
    puzzle["features"]["family_id"] = source["family_id"]
    puzzle["features"]["variant_type"] = "formulation"
    puzzle["features"]["variant_name"] = f"pair_{transform_id}"
    puzzle["features"]["pair_relation_transform"] = transform
    puzzle["features"]["pair_relation_light_tweak"] = light_tweak
    puzzle["diversity_signature"]["variant_type"] = "formulation"
    puzzle["diversity_signature"]["pair_relation_transform"] = transform
    puzzle["diversity_signature"]["pair_relation_light_tweak"] = light_tweak
    return puzzle


def _pair_transform_id(transform: str, light_tweak: str | None) -> str:
    if not light_tweak:
        return transform
    return f"{transform}_{light_tweak}"


def _light_sudoku_pair_tweak(
    board: Sequence[str],
    light_tweak: str,
    idx: int,
) -> Tuple[List[str], Dict[Tuple[int, int], Tuple[int, int]], Dict[str, Any]]:
    n = len(board)
    br, bc = mini_sudoku.box_shape(n)
    row_order = list(range(n))
    col_order = list(range(n))
    if light_tweak == "swap_rows_within_band":
        band_count = n // br
        band = (idx - 1) % band_count
        r1 = band * br
        r2 = r1 + 1
        row_order[r1], row_order[r2] = row_order[r2], row_order[r1]
        detail = {"kind": light_tweak, "rows_1_indexed_after_primary": [r1 + 1, r2 + 1]}
    elif light_tweak == "swap_cols_within_stack":
        stack_count = n // bc
        stack = (idx - 1) % stack_count
        offset = (idx - 1) % (bc - 1)
        c1 = stack * bc + offset
        c2 = c1 + 1
        col_order[c1], col_order[c2] = col_order[c2], col_order[c1]
        detail = {"kind": light_tweak, "cols_1_indexed_after_primary": [c1 + 1, c2 + 1]}
    else:
        raise ValueError(f"Unsupported light Sudoku pair tweak: {light_tweak}")

    old_to_new_row = {old + 1: new + 1 for new, old in enumerate(row_order)}
    old_to_new_col = {old + 1: new + 1 for new, old in enumerate(col_order)}
    new = ["".join(board[old_r][old_c] for old_c in col_order) for old_r in row_order]
    coord_map = {
        (r + 1, c + 1): (old_to_new_row[r + 1], old_to_new_col[c + 1])
        for r in range(n)
        for c in range(n)
    }
    return new, coord_map, detail


def _transform_square_board_with_coord_map(
    board: Sequence[str],
    transform: str,
) -> Tuple[List[str], Dict[Tuple[int, int], Tuple[int, int]]]:
    n = len(board)
    chars = [list(row) for row in board]
    if transform == "mirror_h":
        transformed = ["".join(reversed(row)) for row in chars]
        coord_map = {(r + 1, c + 1): (r + 1, n - c) for r in range(n) for c in range(n)}
    elif transform == "mirror_v":
        transformed = ["".join(row) for row in reversed(chars)]
        coord_map = {(r + 1, c + 1): (n - r, c + 1) for r in range(n) for c in range(n)}
    elif transform == "rotate180":
        transformed = ["".join(reversed(row)) for row in reversed(chars)]
        coord_map = {(r + 1, c + 1): (n - r, n - c) for r in range(n) for c in range(n)}
    else:
        raise ValueError(f"Unsupported pair transform: {transform}")
    return transformed, coord_map


def _readable_pair_transform(transform: str, light_tweak: str | None = None) -> str:
    base = {
        "mirror_h": "left-right mirror",
        "mirror_v": "top-bottom mirror",
        "rotate180": "180-degree rotation",
    }.get(transform, transform)
    if light_tweak == "swap_rows_within_band":
        return f"{base} + one within-band row swap"
    if light_tweak == "swap_cols_within_stack":
        return f"{base} + one within-stack column swap"
    return base


def _select_transfer_plans(
    puzzle_type: str,
    rows: List[Puzzle],
    count: int,
    rng: random.Random,
) -> List[Dict[str, Any]]:
    strategies = TRANSFER_STRATEGIES[puzzle_type]
    desired = [strategies[i % len(strategies)] for i in range(count)]
    target_pools = {strategy: _transfer_b_candidates(puzzle_type, rows, strategy) for strategy in strategies}
    plans: List[Dict[str, Any]] = []
    used_b: set[str] = set()
    used_b_hashes: set[str] = {row["features"].get("canonical_hash", row["id"]) for row in rows}
    used_primes: set[str] = set()
    used_transfer_motifs: set[str] = set()
    for strategy in desired:
        target = _next_transfer_target(target_pools[strategy], used_b)
        if target is None:
            target = _build_fresh_transfer_target_candidate(
                puzzle_type, strategy, rng, used_b_hashes, used_transfer_motifs
            )
        target_b = target["row"]
        target_witness = target.get("target_strategy_witness") or _strategy_witness_metadata(
            target_b, strategy, target["target_coordinates"][0]
        )
        match_coordinates = _transfer_strategy_match_coordinates(target_b, strategy, target_witness)
        if match_coordinates != target["target_coordinates"]:
            raise ValueError(
                f"Transfer B for {puzzle_type}/{strategy} has non-unique strategy-match coordinates: {match_coordinates}"
            )
        prime_a = _build_single_prime_for_target(puzzle_type, strategy, target_b, target_witness, rng, used_primes)
        prime_witness = _strategy_witness_metadata(prime_a, strategy, prime_a["solutions"][0]["coordinate"])
        witness_match = _witness_match_metadata(prime_a, prime_witness, target_b, target_witness)
        if not witness_match["human_strategy_match"]:
            raise ValueError(f"Could not build human-similar prime for {puzzle_type}/{strategy}")
        plans.append(
            {
                "target_b": target_b,
                "prime_a": prime_a,
                "strategy_signature": strategy,
                "target_coordinates": target["target_coordinates"],
                "prime_strategy_witness": prime_witness,
                "target_strategy_witness": target_witness,
                "target_strategy_match_coordinates": match_coordinates,
                "non_target_strategy_fingerprints": _non_target_strategy_fingerprints(
                    target_b, strategy, target_witness
                ),
                "witness_match": witness_match,
                "transfer_target_quality": target["transfer_target_quality"],
            }
        )
        used_b.add(target_b["id"])
        used_b_hashes.add(target_b["features"].get("canonical_hash", target_b["id"]))
        used_primes.add(_prime_seen_hash(prime_a))
        motif_key = target_b["machine_readable_instance"].get("transfer_target_motif_key")
        if motif_key:
            used_transfer_motifs.add(str(motif_key))
    return plans


def _transfer_b_candidates(puzzle_type: str, rows: List[Puzzle], strategy: str) -> List[Dict[str, Any]]:
    if puzzle_type == "minesweeper_lite":
        return []
    candidates: List[Dict[str, Any]] = []
    for row in rows:
        if row["puzzle_type"] != puzzle_type or not solution_count_in_range(int(row["num_solutions"])):
            continue
        target_coordinates = _strategy_target_coordinates(row, strategy)
        if len(target_coordinates) != 1 or len(target_coordinates) >= row["num_solutions"]:
            continue
        witness = _strategy_witness_metadata(row, strategy, target_coordinates[0])
        if puzzle_type == "minesweeper_lite" and not _minesweeper_witness_has_clear_motif(witness):
            continue
        if _transfer_strategy_match_coordinates(row, strategy, witness) != target_coordinates:
            continue
        quality = _transfer_target_quality(row, target_coordinates[0]) if witness else {"quality_ok": False}
        if not quality["quality_ok"]:
            continue
        candidates.append(
            {
                "row": row,
                "target_coordinates": target_coordinates,
                "target_strategy_witness": witness,
                "transfer_target_quality": quality,
            }
        )
    candidates.sort(
        key=lambda item: (
            item["transfer_target_quality"]["simplicity_range"],
            item["transfer_target_quality"]["target_median_gap"],
            -item["transfer_target_quality"]["min_non_target_distance"],
            -item["row"]["features"].get("solution_diversity_score", 0.0),
            -item["row"]["num_solutions"],
            item["row"]["id"],
        )
    )
    return candidates


def _next_transfer_target(candidates: Sequence[Dict[str, Any]], used_b: set[str]) -> Dict[str, Any] | None:
    for candidate in candidates:
        if candidate["row"]["id"] not in used_b and len(candidate["target_coordinates"]) == 1:
            return candidate
    return None


def _transfer_target_quality(row: Puzzle, target_coord: Sequence[int]) -> Dict[str, Any]:
    target = (int(target_coord[0]), int(target_coord[1]))
    scores = [float(sol["probe_features"]["simplicity_score"]) for sol in row["solutions"]]
    coords = [tuple(sol["coordinate"]) for sol in row["solutions"]]
    target_score = next(
        float(sol["probe_features"]["simplicity_score"])
        for sol in row["solutions"]
        if tuple(sol["coordinate"]) == target
    )
    non_target_distances = [abs(target[0] - r) + abs(target[1] - c) for r, c in coords if (r, c) != target]
    ordered_scores = sorted(scores)
    mid = len(ordered_scores) // 2
    median_score = ordered_scores[mid] if len(ordered_scores) % 2 else (ordered_scores[mid - 1] + ordered_scores[mid]) / 2.0
    simplicity_range = max(scores) - min(scores)
    target_median_gap = abs(target_score - median_score)
    min_non_target_distance = min(non_target_distances) if non_target_distances else 0
    quality_ok = (
        row["num_solutions"] >= TRANSFER_MIN_B_SOLUTIONS
        and row["num_solutions"] <= MAX_MULTI_SOLUTIONS
        and simplicity_range <= TRANSFER_MAX_B_SIMPLICITY_RANGE
        and target_median_gap <= TRANSFER_MAX_TARGET_MEDIAN_GAP
        and min_non_target_distance >= TRANSFER_MIN_TARGET_DISTANCE
    )
    return {
        "quality_ok": quality_ok,
        "min_required_solutions": TRANSFER_MIN_B_SOLUTIONS,
        "num_solutions": row["num_solutions"],
        "simplicity_range": round(simplicity_range, 3),
        "max_allowed_simplicity_range": TRANSFER_MAX_B_SIMPLICITY_RANGE,
        "target_median_gap": round(target_median_gap, 3),
        "max_allowed_target_median_gap": TRANSFER_MAX_TARGET_MEDIAN_GAP,
        "min_non_target_distance": min_non_target_distance,
        "min_required_non_target_distance": TRANSFER_MIN_TARGET_DISTANCE,
    }


def _transfer_strategy_match_coordinates(
    row: Puzzle,
    strategy: str,
    target_witness: Dict[str, Any],
) -> List[List[int]]:
    """Solutions whose human-visible local strategy matches the transfer target."""
    target_fingerprint = _human_witness_fingerprint(row, target_witness)
    matches: List[List[int]] = []
    if row["puzzle_type"] == "minesweeper_lite":
        inst = row["machine_readable_instance"]
        board = inst["board"]
        target_kind = inst["target_kind"]
        for sol in row["solutions"]:
            coord = tuple(sol["coordinate"])
            witnesses = _minesweeper_strategy_witnesses_for_target(board, target_kind, strategy, coord)
            if any(_human_witness_fingerprint(row, witness) == target_fingerprint for witness in witnesses):
                matches.append([int(coord[0]), int(coord[1])])
        return sorted(matches)
    if row["puzzle_type"] == "mini_sudoku":
        for sol in row["solutions"]:
            coord = tuple(sol["coordinate"])
            witness = _strategy_witness_metadata(row, strategy, coord)
            if _human_witness_fingerprint(row, witness) == target_fingerprint:
                matches.append([int(coord[0]), int(coord[1])])
        return sorted(matches)
    return []


def _non_target_strategy_fingerprints(
    row: Puzzle,
    strategy: str,
    target_witness: Dict[str, Any],
) -> List[Dict[str, Any]]:
    target = tuple(target_witness.get("target_coordinate") or ())
    out: List[Dict[str, Any]] = []
    for sol in row["solutions"]:
        coord = tuple(sol["coordinate"])
        if coord == target:
            continue
        fingerprints: List[Dict[str, Any]] = []
        if row["puzzle_type"] == "minesweeper_lite":
            inst = row["machine_readable_instance"]
            witnesses = _minesweeper_strategy_witnesses_for_target(
                inst["board"], inst["target_kind"], strategy, coord
            )
            fingerprints = [_human_witness_fingerprint(row, witness) for witness in witnesses]
        elif row["puzzle_type"] == "mini_sudoku":
            fingerprints = [_human_witness_fingerprint(row, _strategy_witness_metadata(row, strategy, coord))]
        out.append(
            {
                "coordinate": [int(coord[0]), int(coord[1])],
                "fingerprints": fingerprints,
            }
        )
    return out


def _witness_match_metadata(
    prime_row: Puzzle,
    prime_witness: Dict[str, Any],
    target_row: Puzzle,
    target_witness: Dict[str, Any],
) -> Dict[str, Any]:
    prime_fingerprint = _human_witness_fingerprint(prime_row, prime_witness)
    target_fingerprint = _human_witness_fingerprint(target_row, target_witness)
    metadata = {
        "human_strategy_match": prime_fingerprint == target_fingerprint,
        "prime_fingerprint": prime_fingerprint,
        "target_fingerprint": target_fingerprint,
    }
    if (
        prime_witness.get("witness_kind") == "minesweeper_clue"
        and target_witness.get("witness_kind") == "minesweeper_clue"
    ):
        relation = _motif_relation_metadata(
            prime_witness.get("local_motif") or [],
            target_witness.get("local_motif") or [],
        )
        metadata["motif_relation"] = relation
        metadata["human_strategy_match"] = bool(metadata["human_strategy_match"] and relation["dihedral_equivalent"])
    return metadata


def _human_witness_fingerprint(row: Puzzle, witness: Dict[str, Any]) -> Dict[str, Any]:
    kind = witness.get("witness_kind")
    strategy = witness.get("strategy")
    if kind == "minesweeper_clue":
        local_motif = witness.get("local_motif") or []
        return {
            "kind": kind,
            "strategy": strategy,
            "motif_canonical_key": _motif_canonical_key(local_motif) if local_motif else None,
        }
    if kind == "sudoku_row":
        return {
            "kind": kind,
            "strategy": strategy,
            "n": witness.get("n"),
            "target_column": witness.get("target_column"),
            "empty_count": witness.get("empty_count"),
        }
    if kind == "sudoku_box":
        return {
            "kind": kind,
            "strategy": strategy,
            "n": witness.get("n"),
            "box_rows": witness.get("box_rows"),
            "box_cols": witness.get("box_cols"),
            "target_local_position": witness.get("target_local_position"),
            "empty_count": witness.get("empty_count"),
        }
    return {"kind": kind, "strategy": strategy}


def _build_fresh_transfer_target_candidate(
    puzzle_type: str,
    strategy: str,
    rng: random.Random,
    seen_hashes: set[str],
    used_transfer_motifs: set[str] | None = None,
) -> Dict[str, Any]:
    for _ in range(60000):
        if puzzle_type == "minesweeper_lite":
            row = _random_minesweeper_transfer_target(strategy, rng, used_transfer_motifs)
        else:
            row = _random_mini_sudoku_transfer_target(strategy, rng)
        if row is None:
            continue
        if not solution_count_in_range(int(row.get("num_solutions", 0))):
            continue
        h = row["features"].get("canonical_hash", row["id"])
        if h in seen_hashes:
            continue
        target_coordinates = _strategy_target_coordinates(row, strategy)
        witness = _strategy_witness_metadata(row, strategy, target_coordinates[0]) if len(target_coordinates) == 1 else None
        if puzzle_type == "minesweeper_lite" and not _minesweeper_witness_has_clear_motif(witness):
            continue
        quality = _transfer_target_quality(row, target_coordinates[0]) if witness else {"quality_ok": False}
        if (
            quality["quality_ok"]
            and len(target_coordinates) == 1
            and len(target_coordinates) < row["num_solutions"]
            and _transfer_strategy_match_coordinates(row, strategy, witness) == target_coordinates
        ):
            return {
                "row": row,
                "target_coordinates": target_coordinates,
                "target_strategy_witness": witness,
                "transfer_target_quality": quality,
            }
    raise ValueError(f"Could not generate fresh transfer B candidate for {puzzle_type}/{strategy}")


def _random_minesweeper_transfer_target(
    strategy: str,
    rng: random.Random,
    used_transfer_motifs: set[str] | None = None,
) -> Puzzle | None:
    puzzle = _build_minesweeper_motif_transfer_target(strategy, rng, used_transfer_motifs)
    if puzzle is None:
        return None
    return puzzle


def _build_minesweeper_motif_transfer_target(
    strategy: str,
    rng: random.Random,
    used_transfer_motifs: set[str] | None = None,
) -> Puzzle | None:
    target_kind = "safe" if strategy == "safe_by_satisfied_clue" else "mine"
    motifs = _minesweeper_transfer_motifs(strategy)
    available_targets = [
        motif for motif in motifs if _motif_key(motif) not in (used_transfer_motifs or set())
    ]
    if not available_targets:
        available_targets = motifs
    target_motif = rng.choice(available_targets)
    target_key = _motif_key(target_motif)
    target_canonical = _motif_canonical_key(target_motif)
    distractors = [
        motif
        for motif in _unique_motifs_by_canonical(motifs)
        if _motif_canonical_key(motif) != target_canonical
    ]
    rng.shuffle(distractors)
    if len(distractors) < TRANSFER_MS_B_VALID_MOTIFS - 1:
        return None
    selected = [target_motif] + distractors[: TRANSFER_MS_B_VALID_MOTIFS - 1]
    decoy_strategy = _opposite_minesweeper_strategy(strategy)
    decoy_kind = "mine" if target_kind == "safe" else "safe"
    decoy_motifs = _minesweeper_transfer_motifs(decoy_strategy)
    rng.shuffle(decoy_motifs)
    decoys = decoy_motifs[:TRANSFER_MIN_B_INVALID_CANDIDATES]
    rows, cols, centers = _random_minesweeper_motif_layout(rng, len(selected) + len(decoys))
    rng.shuffle(centers)
    specs = [
        {"role": "valid", "motif": motif, "target_kind": target_kind, "center": center}
        for motif, center in zip(selected, centers[: len(selected)])
    ]
    specs.extend(
        {"role": "invalid_decoy", "motif": motif, "target_kind": decoy_kind, "center": center}
        for motif, center in zip(decoys, centers[len(selected) :])
    )
    board, placements = _compose_minesweeper_motif_board(rows, cols, specs)
    target = placements[0]["target"]
    target_coord = [target[0] + 1, target[1] + 1]
    decoy_targets = [placement["target"] for placement in placements if placement["role"] == "invalid_decoy"]
    puzzle = minesweeper_lite.make_puzzle(board, target_kind)
    if puzzle is None:
        return None
    puzzle["machine_readable_instance"]["transfer_target_local_motif"] = target_motif
    if _strategy_target_coordinates(puzzle, strategy) != [target_coord]:
        return None
    if _minesweeper_unknown_count(board) - puzzle["num_solutions"] < TRANSFER_MIN_B_INVALID_CANDIDATES:
        return None
    actual_board = puzzle["machine_readable_instance"]["board"]
    puzzle["id"] = f"TRANSFER_SRC_MS_{stable_hash({'board': actual_board, 'target_kind': target_kind, 'strategy': strategy})[:10]}"
    puzzle["machine_readable_instance"]["transfer_target_local_motif"] = target_motif
    puzzle["machine_readable_instance"]["transfer_target_motif_key"] = target_key
    puzzle["machine_readable_instance"]["transfer_target_motif_canonical_key"] = target_canonical
    puzzle["machine_readable_instance"]["transfer_non_target_valid_motif_canonical_keys"] = [
        _motif_canonical_key(motif) for motif in selected[1:]
    ]
    puzzle["machine_readable_instance"]["transfer_layout"] = {
        "rows": rows,
        "cols": cols,
        "centers_0_indexed": [[r, c] for r, c in centers],
    }
    puzzle["machine_readable_instance"]["transfer_generation"] = "valid_motifs_plus_opposite_decoys"
    puzzle["machine_readable_instance"]["transfer_decoy_unknown_coordinates"] = [[r + 1, c + 1] for r, c in decoy_targets]
    puzzle["features"]["candidate_unknown_count"] = _minesweeper_unknown_count(puzzle["machine_readable_instance"]["board"])
    puzzle["features"]["invalid_candidate_count"] = puzzle["features"]["candidate_unknown_count"] - puzzle["num_solutions"]
    return puzzle


def _opposite_minesweeper_strategy(strategy: str) -> str:
    if strategy == "safe_by_satisfied_clue":
        return "mine_by_exact_remaining"
    if strategy == "mine_by_exact_remaining":
        return "safe_by_satisfied_clue"
    raise ValueError(f"Unknown Minesweeper strategy: {strategy}")


def _compose_minesweeper_motif_board(
    rows: int,
    cols: int,
    specs: Sequence[Dict[str, Any]],
) -> Tuple[List[str], List[Dict[str, Any]]]:
    hidden_mines: set[Tuple[int, int]] = set()
    flags: set[Tuple[int, int]] = set()
    placements: List[Dict[str, Any]] = []
    for spec in specs:
        target = _place_minesweeper_motif(
            spec["motif"],
            spec["center"],
            spec["target_kind"],
            hidden_mines,
            flags,
            rows,
            cols,
        )
        placements.append(
            {
                "role": spec["role"],
                "target": target,
                "center": spec["center"],
                "motif": spec["motif"],
                "target_kind": spec["target_kind"],
            }
        )

    digits: List[List[str]] = [["0" for _ in range(cols)] for _ in range(rows)]
    for r in range(rows):
        for c in range(cols):
            digits[r][c] = str(sum(n in hidden_mines for n in minesweeper_lite.neighbors(rows, cols, (r, c))))
    board_chars = [row[:] for row in digits]
    for r, c in flags:
        board_chars[r][c] = "F"
    for placement in placements:
        r, c = placement["target"]
        board_chars[r][c] = "?"
    return ["".join(row) for row in board_chars], placements


def _minesweeper_unknown_count(board: Sequence[str]) -> int:
    return sum(ch == "?" for row in board for ch in row)


def _minesweeper_transfer_motifs(strategy: str) -> List[List[str]]:
    if strategy == "safe_by_satisfied_clue":
        seeds = [
            ["F . .", "T C1 .", ". . ."],
            [". F .", "T C1 .", ". . ."],
            [". . .", "F C1 T", ". . ."],
            ["F F .", "T C2 .", ". . ."],
            ["F . .", "T C2 .", ". F ."],
            [". F .", "T C2 F", ". . ."],
            ["F F .", "T C3 F", ". . ."],
            ["F . F", ". C3 T", "F . ."],
        ]
    else:
        seeds = [
            [". . .", "T C1 .", ". . ."],
            ["T . .", ". C1 .", ". . ."],
            ["F . .", "T C2 .", ". . ."],
            [". F .", "T C2 .", ". . ."],
            ["F . .", ". C2 T", ". . ."],
            ["F F .", "T C3 .", ". . ."],
            ["F . .", "T C3 .", ". F ."],
            [". F .", "T C3 F", ". . ."],
        ]
    motifs: List[List[str]] = []
    seen: set[str] = set()
    for seed in seeds:
        for motif in _motif_dihedral_variants(seed):
            key = _motif_key(motif)
            if key not in seen:
                seen.add(key)
                motifs.append(motif)
    return motifs


def _unique_motifs_by_canonical(motifs: Sequence[Sequence[str]]) -> List[List[str]]:
    out: List[List[str]] = []
    seen: set[str] = set()
    for motif in motifs:
        key = _motif_canonical_key(motif)
        if key in seen:
            continue
        seen.add(key)
        out.append(list(motif))
    return out


def _motif_dihedral_variants(motif: Sequence[str]) -> List[List[str]]:
    return [variant for _, variant in _motif_named_dihedral_variants(motif)]


def _motif_named_dihedral_variants(motif: Sequence[str]) -> List[Tuple[str, List[str]]]:
    grid = [row.split() for row in motif]
    transforms = [
        ("identity", grid),
        ("rotate90", _rotate_motif_grid(grid)),
        ("rotate180", _rotate_motif_grid(_rotate_motif_grid(grid))),
        ("rotate270", _rotate_motif_grid(_rotate_motif_grid(_rotate_motif_grid(grid)))),
    ]
    mirrored = _mirror_motif_grid(grid)
    transforms.extend(
        [
            ("mirror", mirrored),
            ("mirror_rotate90", _rotate_motif_grid(mirrored)),
            ("mirror_rotate180", _rotate_motif_grid(_rotate_motif_grid(mirrored))),
            ("mirror_rotate270", _rotate_motif_grid(_rotate_motif_grid(_rotate_motif_grid(mirrored)))),
        ]
    )
    out: List[Tuple[str, List[str]]] = []
    seen: set[str] = set()
    for name, variant in transforms:
        rows = [" ".join(row) for row in variant]
        key = _motif_key(rows)
        if key not in seen:
            seen.add(key)
            out.append((name, rows))
    return out


def _rotate_motif_grid(grid: Sequence[Sequence[str]]) -> List[List[str]]:
    return [[grid[2 - c][r] for c in range(3)] for r in range(3)]


def _mirror_motif_grid(grid: Sequence[Sequence[str]]) -> List[List[str]]:
    return [list(reversed(row)) for row in grid]


def _motif_key(motif: Sequence[str]) -> str:
    return "|".join(" ".join(row.split()) for row in motif)


def _motif_canonical_key(motif: Sequence[str]) -> str:
    variants = [_motif_key(variant) for _, variant in _motif_named_dihedral_variants(motif)]
    return min(variants) if variants else ""


def _choose_prime_motif_variant(
    motif: Sequence[str],
    rng: random.Random,
) -> Tuple[List[str], Dict[str, Any]]:
    original_key = _motif_key(motif)
    variants = [
        (name, variant)
        for name, variant in _motif_named_dihedral_variants(motif)
        if _motif_key(variant) != original_key
    ]
    if not variants:
        return list(motif), {
            "transform": "identity",
            "exact_same": True,
            "dihedral_equivalent": True,
            "source_motif_key": original_key,
            "prime_motif_key": original_key,
            "canonical_motif_key": _motif_canonical_key(motif),
        }
    name, variant = rng.choice(variants)
    return variant, {
        "transform": name,
        "exact_same": False,
        "dihedral_equivalent": True,
        "source_motif_key": original_key,
        "prime_motif_key": _motif_key(variant),
        "canonical_motif_key": _motif_canonical_key(motif),
    }


def _motif_relation_metadata(
    prime_motif: Sequence[str],
    target_motif: Sequence[str],
) -> Dict[str, Any]:
    prime_key = _motif_key(prime_motif)
    target_key = _motif_key(target_motif)
    prime_canonical = _motif_canonical_key(prime_motif)
    target_canonical = _motif_canonical_key(target_motif)
    return {
        "dihedral_equivalent": bool(prime_canonical and prime_canonical == target_canonical),
        "exact_same": prime_key == target_key,
        "prime_motif_key": prime_key,
        "target_motif_key": target_key,
        "canonical_motif_key": prime_canonical if prime_canonical == target_canonical else None,
        "relation": "same_orientation" if prime_key == target_key else "rotated_or_reflected",
    }


def _random_minesweeper_motif_layout(
    rng: random.Random,
    num_motifs: int,
) -> Tuple[int, int, List[Tuple[int, int]]]:
    if num_motifs <= 4:
        rows = cols = 6
        centers = [(1, 1), (1, 4), (4, 1), (4, 4)]
        rng.shuffle(centers)
        return rows, cols, centers[:num_motifs]
    if num_motifs > 4:
        rows = cols = 11 if num_motifs <= 9 else 15
        centers = [(r, c) for r in range(1, rows - 1, 4) for c in range(1, cols - 1, 4)]
        rng.shuffle(centers)
        return rows, cols, centers[:num_motifs]




def _place_minesweeper_motif(
    motif: Sequence[str],
    center: Tuple[int, int],
    target_kind: str,
    hidden_mines: set[Tuple[int, int]],
    flags: set[Tuple[int, int]],
    rows: int,
    cols: int,
) -> Tuple[int, int]:
    target: Tuple[int, int] | None = None
    for mr, row in enumerate(motif):
        for mc, token in enumerate(row.split()):
            r, c = center[0] + mr - 1, center[1] + mc - 1
            if r < 0 or r >= rows or c < 0 or c >= cols:
                if token == "O":
                    continue
                raise ValueError("Minesweeper motif is out of bounds")
            if token == "O":
                continue
            if token == "F":
                hidden_mines.add((r, c))
                flags.add((r, c))
            elif token == "T":
                target = (r, c)
                if target_kind == "mine":
                    hidden_mines.add((r, c))
    if target is None:
        raise ValueError("Minesweeper motif missing target")
    return target


def _random_mini_sudoku_transfer_target(strategy: str, rng: random.Random) -> Puzzle | None:
    n = 6 if rng.random() < 0.7 else 4
    full = mini_sudoku._random_full_board(n, rng)
    keep_prob = rng.uniform(0.28, 0.48) if n == 6 else rng.uniform(0.34, 0.56)
    board = ["".join(ch if rng.random() < keep_prob else "." for ch in row) for row in full]
    digit = rng.randint(1, n)
    puzzle = mini_sudoku.make_puzzle(board, digit)
    if puzzle is None:
        return None
    puzzle["id"] = f"TRANSFER_SRC_SUD_{stable_hash({'board': board, 'digit': digit, 'strategy': strategy})[:10]}"
    return puzzle


def _build_single_prime_for_target(
    puzzle_type: str,
    strategy: str,
    target_b: Puzzle,
    target_witness: Dict[str, Any],
    rng: random.Random,
    used_primes: set[str],
) -> Puzzle:
    if puzzle_type == "minesweeper_lite":
        for pad in range(1, 8):
            try:
                prime = _build_minesweeper_prime_from_witness(strategy, target_witness, rng, pad=pad)
            except ValueError:
                continue
            prime_witness = _strategy_witness_metadata(prime, strategy, prime["solutions"][0]["coordinate"])
            witness_match = _witness_match_metadata(prime, prime_witness, target_b, target_witness)
            relation = witness_match.get("motif_relation", {})
            if (
                _prime_seen_hash(prime) not in used_primes
                and witness_match["human_strategy_match"]
                and not relation.get("exact_same", True)
            ):
                return prime
        raise ValueError(f"Could not build unique transfer prime for {puzzle_type}/{strategy}")

    n = target_b["machine_readable_instance"].get("n")
    for _ in range(2000):
        prime = _build_mini_sudoku_single_prime(strategy, rng, int(n), target_witness)
        prime_witness = _strategy_witness_metadata(prime, strategy, prime["solutions"][0]["coordinate"])
        if _prime_seen_hash(prime) not in used_primes and _witness_match_metadata(prime, prime_witness, target_b, target_witness)["human_strategy_match"]:
            return prime
    raise ValueError(f"Could not build unique transfer prime for {puzzle_type}/{strategy}")


def _prime_seen_hash(row: Puzzle) -> str:
    return stable_hash(
        {
            "puzzle_type": row["puzzle_type"],
            "machine_readable_instance": row["machine_readable_instance"],
            "strategy": row["features"].get("transfer_prime_strategy"),
        }
    )


def _strategy_target_coordinates(row: Puzzle, strategy: str) -> List[List[int]]:
    if row["puzzle_type"] == "minesweeper_lite":
        inst = row["machine_readable_instance"]
        valid = {tuple(sol["coordinate"]) for sol in row["solutions"]}
        witness_map = _minesweeper_strategy_witness_map(inst["board"], inst["target_kind"], strategy)
        preferred = inst.get("transfer_target_local_motif")
        if preferred:
            coords = sorted(coord for coord, witness in witness_map.items() if witness.get("local_motif") == preferred)
            return [[r, c] for r, c in coords if (r, c) in valid]
        coords = sorted(valid & set(witness_map))
        return [[r, c] for r, c in coords]
    if row["puzzle_type"] == "mini_sudoku":
        return _mini_sudoku_strategy_coordinates(row, strategy)
    return []




def _minesweeper_strategy_witness_map(
    board: Sequence[str],
    target_kind: str,
    strategy: str,
) -> Dict[Tuple[int, int], Dict[str, Any]]:
    if strategy == "safe_by_satisfied_clue" and target_kind != "safe":
        return {}
    if strategy == "mine_by_exact_remaining" and target_kind != "mine":
        return {}
    rows, cols = len(board), len(board[0])
    witnesses: Dict[Tuple[int, int], Dict[str, Any]] = {}
    for r in range(rows):
        for c in range(cols):
            if not board[r][c].isdigit():
                continue
            clue = int(board[r][c])
            neigh = minesweeper_lite.neighbors(rows, cols, (r, c))
            flags = sum(board[rr][cc] == "F" for rr, cc in neigh)
            unknowns = [(rr, cc) for rr, cc in neigh if board[rr][cc] == "?"]
            remaining = clue - flags
            if strategy == "safe_by_satisfied_clue" and remaining != 0:
                continue
            if strategy == "mine_by_exact_remaining" and not (remaining > 0 and remaining == len(unknowns)):
                continue
            if len(unknowns) != 1:
                continue
            target = (unknowns[0][0] + 1, unknowns[0][1] + 1)
            candidate = {
                "target_coordinate": [target[0], target[1]],
                "witness_coordinate": [r + 1, c + 1],
                "witness_kind": "minesweeper_clue",
                "strategy": strategy,
                "clue_value": clue,
                "flag_count": flags,
                "remaining_mines": remaining,
                "adjacent_unknown_count": len(unknowns),
                "target_relative_to_witness": [target[0] - (r + 1), target[1] - (c + 1)],
                "local_motif": _minesweeper_witness_motif(board, (r, c), unknowns[0]),
            }
            if target not in witnesses or _minesweeper_witness_priority(candidate) > _minesweeper_witness_priority(witnesses[target]):
                witnesses[target] = candidate
    return witnesses


def _minesweeper_strategy_witnesses_for_target(
    board: Sequence[str],
    target_kind: str,
    strategy: str,
    target1: Tuple[int, int],
) -> List[Dict[str, Any]]:
    if strategy == "safe_by_satisfied_clue" and target_kind != "safe":
        return []
    if strategy == "mine_by_exact_remaining" and target_kind != "mine":
        return []
    rows, cols = len(board), len(board[0])
    target0 = (target1[0] - 1, target1[1] - 1)
    witnesses: List[Dict[str, Any]] = []
    for r in range(rows):
        for c in range(cols):
            if not board[r][c].isdigit():
                continue
            neigh = minesweeper_lite.neighbors(rows, cols, (r, c))
            if target0 not in neigh or board[target0[0]][target0[1]] != "?":
                continue
            clue = int(board[r][c])
            flags = sum(board[rr][cc] == "F" for rr, cc in neigh)
            unknowns = [(rr, cc) for rr, cc in neigh if board[rr][cc] == "?"]
            remaining = clue - flags
            if unknowns != [target0]:
                continue
            if strategy == "safe_by_satisfied_clue" and remaining != 0:
                continue
            if strategy == "mine_by_exact_remaining" and not (remaining == 1):
                continue
            witnesses.append(
                {
                    "target_coordinate": [target1[0], target1[1]],
                    "witness_coordinate": [r + 1, c + 1],
                    "witness_kind": "minesweeper_clue",
                    "strategy": strategy,
                    "clue_value": clue,
                    "flag_count": flags,
                    "remaining_mines": remaining,
                    "adjacent_unknown_count": len(unknowns),
                    "target_relative_to_witness": [target1[0] - (r + 1), target1[1] - (c + 1)],
                    "local_motif": _minesweeper_witness_motif(board, (r, c), target0),
                }
            )
    witnesses.sort(key=_minesweeper_witness_priority, reverse=True)
    return witnesses


def _minesweeper_witness_priority(witness: Dict[str, Any]) -> Tuple[int, int, int, int]:
    motif = witness.get("local_motif") or []
    o_count = sum(row.split().count("O") for row in motif)
    rel = witness.get("target_relative_to_witness") or [0, 0]
    manhattan = abs(int(rel[0])) + abs(int(rel[1]))
    return (
        int(witness.get("flag_count") or 0),
        int(witness.get("clue_value") or 0),
        -manhattan,
        o_count,
    )


def _minesweeper_witness_motif(
    board: Sequence[str],
    witness0: Tuple[int, int],
    target0: Tuple[int, int],
) -> List[str]:
    rows, cols = len(board), len(board[0])
    wr, wc = witness0
    motif: List[str] = []
    for dr in (-1, 0, 1):
        chars: List[str] = []
        for dc in (-1, 0, 1):
            rr, cc = wr + dr, wc + dc
            if rr < 0 or rr >= rows or cc < 0 or cc >= cols:
                chars.append("O")
            elif (rr, cc) == witness0:
                chars.append(f"C{board[rr][cc]}")
            elif (rr, cc) == target0:
                chars.append("T")
            elif board[rr][cc] == "F":
                chars.append("F")
            elif board[rr][cc] == "?":
                chars.append("?")
            else:
                chars.append(".")
        motif.append(" ".join(chars))
    return motif


def _minesweeper_witness_has_clear_motif(witness: Dict[str, Any] | None) -> bool:
    if not witness or witness.get("witness_kind") != "minesweeper_clue":
        return False
    motif = witness.get("local_motif") or []
    if not (bool(motif) and sum(row.split().count("T") for row in motif) == 1):
        return False
    if witness.get("strategy") == "safe_by_satisfied_clue":
        return int(witness.get("flag_count") or 0) >= 1
    return int(witness.get("clue_value") or 0) >= 1


def _build_minesweeper_prime_from_witness(
    strategy: str,
    target_witness: Dict[str, Any],
    rng: random.Random,
    pad: int = 1,
) -> Puzzle:
    source_motif = target_witness.get("local_motif")
    if not source_motif:
        raise ValueError("Minesweeper transfer witness is missing local_motif")
    motif, motif_transform = _choose_prime_motif_variant(source_motif, rng)
    rows = cols = 6
    centers = [(1, 1), (1, 4), (4, 1), (4, 4)]
    rng.shuffle(centers)
    center = centers[0]
    target_kind = "safe" if strategy == "safe_by_satisfied_clue" else "mine"
    decoy_strategy = _opposite_minesweeper_strategy(strategy)
    decoy_kind = "mine" if target_kind == "safe" else "safe"
    decoy_motifs = _minesweeper_transfer_motifs(decoy_strategy)
    rng.shuffle(decoy_motifs)
    decoy_centers = centers[1 : TRANSFER_MIN_PRIME_UNKNOWN_CELLS]
    if len(decoy_centers) < TRANSFER_MIN_PRIME_UNKNOWN_CELLS - 1:
        raise ValueError("Not enough room for Minesweeper prime decoys")
    specs = [{"role": "valid", "motif": motif, "target_kind": target_kind, "center": center}]
    specs.extend(
        {"role": "invalid_decoy", "motif": decoy_motif, "target_kind": decoy_kind, "center": decoy_center}
        for decoy_motif, decoy_center in zip(decoy_motifs, decoy_centers)
    )
    board, placements = _compose_minesweeper_motif_board(rows, cols, specs)
    target = placements[0]["target"]
    target_coord = [target[0] + 1, target[1] + 1]
    decoys = [placement["target"] for placement in placements if placement["role"] == "invalid_decoy"]
    solutions, num_assignments = minesweeper_lite.forced_cells(board, target_kind)
    if len(solutions) != 1 or solutions[0]["coordinate"] != target_coord:
        raise ValueError("Constructed Minesweeper prime does not have the intended single solution")
    puzzle = _make_minesweeper_prime_puzzle(board, target_kind, solutions, num_assignments, strategy)
    puzzle["machine_readable_instance"]["preferred_transfer_local_motif"] = motif
    puzzle["machine_readable_instance"]["transfer_prime_source_local_motif"] = source_motif
    puzzle["machine_readable_instance"]["transfer_prime_motif_transform"] = motif_transform
    puzzle["machine_readable_instance"]["transfer_decoy_unknown_coordinates"] = [[r + 1, c + 1] for r, c in decoys]
    puzzle["features"]["candidate_unknown_count"] = _minesweeper_unknown_count(board)
    puzzle["features"]["invalid_candidate_count"] = puzzle["features"]["candidate_unknown_count"] - puzzle["num_solutions"]
    return puzzle




def _mini_sudoku_strategy_coordinates(row: Puzzle, strategy: str) -> List[List[int]]:
    coords = [tuple(sol["coordinate"]) for sol in row["solutions"]]
    if not coords:
        return []
    n = row["machine_readable_instance"]["n"]
    br, bc = mini_sudoku.box_shape(n)
    board = row["machine_readable_instance"]["board"]
    row_single = {coord for coord in coords if board[coord[0] - 1].count(".") == 1}
    box_single = {coord for coord in coords if _sudoku_box_blank_count(board, coord, br, bc) == 1}
    if strategy == "sudoku_row_elimination":
        primary, other = row_single, box_single
    elif strategy == "sudoku_box_elimination":
        primary, other = box_single, row_single
    else:
        return []
    targets = primary - other
    if not targets or len(targets) >= len(coords):
        return []
    return [[r, c] for r, c in sorted(targets)]




def _sudoku_box_blank_count(board: Sequence[str], coord: Tuple[int, int], br: int, bc: int) -> int:
    r, c = coord
    r0, c0 = ((r - 1) // br) * br, ((c - 1) // bc) * bc
    return sum(board[rr][cc] == "." for rr in range(r0, r0 + br) for cc in range(c0, c0 + bc))


def _strategy_witness_metadata(row: Puzzle, strategy: str, coord: Sequence[int]) -> Dict[str, Any]:
    target = [int(coord[0]), int(coord[1])]
    if row["puzzle_type"] == "minesweeper_lite":
        inst = row["machine_readable_instance"]
        preferred = inst.get("preferred_transfer_local_motif") or inst.get("transfer_target_local_motif")
        if preferred:
            for witness in _minesweeper_strategy_witnesses_for_target(
                inst["board"], inst["target_kind"], strategy, tuple(target)
            ):
                if witness.get("local_motif") == preferred:
                    return witness
        witness = _minesweeper_strategy_witness_map(inst["board"], inst["target_kind"], strategy).get(tuple(target))
        if witness:
            return witness
        unknown_count = _minesweeper_strategy_witness_unknown_count(inst["board"], tuple(target), strategy)
        return {
            "target_coordinate": target,
            "witness_kind": "minesweeper_clue",
            "strategy": strategy,
            "adjacent_unknown_count": unknown_count,
        }
    if row["puzzle_type"] == "mini_sudoku":
        board = row["machine_readable_instance"]["board"]
        n = row["machine_readable_instance"]["n"]
        br, bc = mini_sudoku.box_shape(n)
        r, c = target
        r0, c0 = ((r - 1) // br) * br + 1, ((c - 1) // bc) * bc + 1
        if strategy == "sudoku_row_elimination":
            return {
                "target_coordinate": target,
                "witness_kind": "sudoku_row",
                "strategy": strategy,
                "n": n,
                "row": r,
                "target_column": c,
                "empty_count": board[r - 1].count("."),
            }
        if strategy == "sudoku_box_elimination":
            return {
                "target_coordinate": target,
                "witness_kind": "sudoku_box",
                "strategy": strategy,
                "n": n,
                "box_origin": [r0, c0],
                "box_rows": br,
                "box_cols": bc,
                "target_local_position": [r - r0, c - c0],
                "empty_count": _sudoku_box_blank_count(board, (r, c), br, bc),
            }
    return {"target_coordinate": target, "strategy": strategy}




def _minesweeper_strategy_witness_unknown_count(
    board: Sequence[str],
    coord1: Tuple[int, int],
    strategy: str,
) -> int | None:
    rows, cols = len(board), len(board[0])
    target0 = (coord1[0] - 1, coord1[1] - 1)
    witness_sizes: List[int] = []
    for r in range(rows):
        for c in range(cols):
            if not board[r][c].isdigit():
                continue
            neigh = minesweeper_lite.neighbors(rows, cols, (r, c))
            if target0 not in neigh:
                continue
            unknowns = [(rr, cc) for rr, cc in neigh if board[rr][cc] == "?"]
            if target0 not in unknowns:
                continue
            clue = int(board[r][c])
            flags = sum(board[rr][cc] == "F" for rr, cc in neigh)
            remaining = clue - flags
            if strategy == "safe_by_satisfied_clue" and remaining == 0:
                witness_sizes.append(len(unknowns))
            elif strategy == "mine_by_exact_remaining" and remaining > 0 and remaining == len(unknowns):
                witness_sizes.append(len(unknowns))
    return min(witness_sizes) if witness_sizes else None


def _make_minesweeper_prime_puzzle(
    board: Sequence[str],
    target_kind: str,
    solutions: Sequence[Dict[str, Any]],
    num_assignments: int,
    strategy: str,
) -> Puzzle:
    features, signature, bucket = minesweeper_lite.compute_features(board, target_kind, solutions, num_assignments)
    features["transfer_prime_strategy"] = strategy
    signature["transfer_prime_strategy"] = strategy
    rendered = "\n".join(" ".join(row) for row in board)
    task = "definitely safe" if target_kind == "safe" else "definitely a mine"
    prompt = (
        "Question {{ID}}:\n"
        f"Choose one hidden cell marked ? that is {task}.\n\n"
        "Rules: each ? is hidden and is either safe or a mine. Each number is already revealed and shows exactly how many mines are in the up to 8 neighboring cells around it, including diagonals. F is a known mine.\n\n"
        f"Board:\n{rendered}\n\nWrite one coordinate as row,column."
    )
    puzzle = annotate_puzzle(
        {
            "id": f"PRIME_SRC_MS_{stable_hash({'board': list(board), 'target_kind': target_kind, 'strategy': strategy})[:10]}",
            "puzzle_type": "minesweeper_lite",
            "prompt_text": prompt,
            "rendered_puzzle": rendered,
            "machine_readable_instance": {
                "board": list(board),
                "target_kind": target_kind,
                "transfer_prime_strategy": strategy,
            },
            "num_solutions": len(solutions),
            "solutions": list(copy.deepcopy(solutions)),
            "features": features,
            "difficulty_bucket": bucket,
            "diversity_signature": signature,
        }
    )
    return puzzle


def _build_mini_sudoku_single_prime(
    strategy: str,
    rng: random.Random,
    n: int,
    target_witness: Dict[str, Any] | None = None,
) -> Puzzle:
    for _ in range(10000):
        full = mini_sudoku._random_full_board(n, rng)
        digit = rng.randint(1, n)
        targets = [(r, c) for r in range(n) for c in range(n) if full[r][c] == str(digit)]
        if target_witness:
            targets = [
                target
                for target in targets
                if _sudoku_prime_target_matches_witness(target, strategy, n, target_witness)
            ]
        rng.shuffle(targets)
        for target in targets:
            board_chars = [list(row) for row in full]
            board_chars[target[0]][target[1]] = "."
            required = _sudoku_prime_required_blanks(full, digit, target, strategy)
            if not required:
                continue
            for r, c in required:
                board_chars[r][c] = "."
            extra_candidates = _sudoku_prime_extra_blank_candidates(full, digit, target, strategy)
            rng.shuffle(extra_candidates)
            extra_count = rng.randint(2, 5 if n == 4 else 8)
            for r, c in extra_candidates[:extra_count]:
                board_chars[r][c] = "."
            board = ["".join(row) for row in board_chars]
            solutions = mini_sudoku.enumerate_placements(board, digit)
            coord = [target[0] + 1, target[1] + 1]
            if len(solutions) == 1 and solutions[0]["coordinate"] == coord:
                puzzle = _make_mini_sudoku_prime_puzzle(board, digit, solutions, strategy)
                witness = _strategy_witness_metadata(puzzle, strategy, coord)
                if target_witness and _human_witness_fingerprint(puzzle, witness) != _human_witness_fingerprint(puzzle, target_witness):
                    continue
                return puzzle
    raise ValueError(f"Could not generate single-solution Mini Sudoku prime for {strategy}/{n}")


def _sudoku_prime_target_matches_witness(
    target0: Tuple[int, int],
    strategy: str,
    n: int,
    target_witness: Dict[str, Any],
) -> bool:
    tr, tc = target0[0] + 1, target0[1] + 1
    if target_witness.get("n") != n:
        return False
    if strategy == "sudoku_row_elimination":
        return target_witness.get("target_column") == tc
    if strategy == "sudoku_box_elimination":
        br, bc = mini_sudoku.box_shape(n)
        return target_witness.get("target_local_position") == [target0[0] % br, target0[1] % bc]
    return True


def _sudoku_prime_required_blanks(
    full: Sequence[str],
    digit: int,
    target: Tuple[int, int],
    strategy: str,
) -> List[Tuple[int, int]]:
    n = len(full)
    br, bc = mini_sudoku.box_shape(n)
    tr, tc = target
    r0, c0 = (tr // br) * br, (tc // bc) * bc
    required: List[Tuple[int, int]] = []
    if strategy == "sudoku_row_elimination":
        box_candidates = [
            (r, c)
            for r in range(r0, r0 + br)
            for c in range(c0, c0 + bc)
            if r != tr and full[r][c] != str(digit)
        ]
        col_candidates = [
            (r, tc)
            for r in range(n)
            if r != tr and full[r][tc] != str(digit)
        ]
        if not box_candidates or not col_candidates:
            return []
        required.extend([box_candidates[0], col_candidates[0]])
    elif strategy == "sudoku_box_elimination":
        row_candidates = [
            (tr, c)
            for c in range(n)
            if not (c0 <= c < c0 + bc) and full[tr][c] != str(digit)
        ]
        col_candidates = [
            (r, tc)
            for r in range(n)
            if not (r0 <= r < r0 + br) and full[r][tc] != str(digit)
        ]
        if not row_candidates or not col_candidates:
            return []
        required.extend([row_candidates[0], col_candidates[0]])
    return sorted(set(required))


def _sudoku_prime_extra_blank_candidates(
    full: Sequence[str],
    digit: int,
    target: Tuple[int, int],
    strategy: str,
) -> List[Tuple[int, int]]:
    n = len(full)
    br, bc = mini_sudoku.box_shape(n)
    tr, tc = target
    r0, c0 = (tr // br) * br, (tc // bc) * bc
    candidates = []
    for r in range(n):
        for c in range(n):
            if (r, c) == target or full[r][c] == str(digit):
                continue
            if strategy == "sudoku_row_elimination" and r == tr:
                continue
            if strategy == "sudoku_box_elimination" and r0 <= r < r0 + br and c0 <= c < c0 + bc:
                continue
            candidates.append((r, c))
    return candidates


def _make_mini_sudoku_prime_puzzle(
    board: Sequence[str],
    digit: int,
    solutions: Sequence[Dict[str, Any]],
    strategy: str,
) -> Puzzle:
    features, signature, bucket = mini_sudoku.compute_features(board, digit, solutions)
    features["transfer_prime_strategy"] = strategy
    signature["transfer_prime_strategy"] = strategy
    n = len(board)
    br, bc = mini_sudoku.box_shape(n)
    rendered = "\n".join(" ".join(row) for row in board)
    prompt = (
        "Question {{ID}}:\n"
        f"Place the digit {digit} in one legal empty cell of this {n} by {n} mini Sudoku.\n\n"
        f"Rules: rows, columns, and each {br} by {bc} box cannot repeat a digit.\n\n"
        f"Board:\n{rendered}\n\nWrite one coordinate as row,column."
    )
    return annotate_puzzle(
        {
            "id": f"PRIME_SRC_SUD_{stable_hash({'board': list(board), 'digit': digit, 'strategy': strategy})[:10]}",
            "puzzle_type": "mini_sudoku",
            "prompt_text": prompt,
            "rendered_puzzle": rendered,
            "machine_readable_instance": {
                "n": n,
                "box_rows": br,
                "box_cols": bc,
                "digit": digit,
                "board": list(board),
                "transfer_prime_strategy": strategy,
            },
            "num_solutions": len(solutions),
            "solutions": list(copy.deepcopy(solutions)),
            "features": features,
            "difficulty_bucket": bucket,
            "diversity_signature": signature,
        }
    )


def _strategy_signature(puzzle_type: str, solution: Dict[str, Any]) -> str:
    comps = solution["probe_features"].get("simplicity_components", {})
    if puzzle_type == "minesweeper_lite":
        local = _bucket(float(comps.get("adjacent_clues", 0)), [1, 3])
        return f"{solution['cell_type']}|local={local}"
    if puzzle_type == "mini_sudoku":
        local = _bucket(float(comps.get("local_visibility", 0)), [3, 6])
        scan = _bucket(float(comps.get("scanning_load", 0.0)), [0.35, 0.6])
        return f"local={local}|scan={scan}"
    return str(solution["probe_features"].get("solution_strategy_class"))








def _families_with_variants(rows: Sequence[Puzzle]) -> Dict[str, Dict[str, Puzzle]]:
    groups: Dict[str, Dict[str, Puzzle]] = defaultdict(dict)
    for row in rows:
        groups[row["family_id"]][row["variant_type"]] = row
    return groups


def _canonical_rows(rows: Sequence[Puzzle]) -> List[Puzzle]:
    return [row for row in rows if row.get("variant_type") == "canonical"]


def _control_family(groups: Dict[str, Dict[str, Puzzle]], selected_ids: List[str], current_id: str, idx: int) -> str:
    available = [
        family_id
        for family_id in sorted(groups)
        if family_id != current_id
        and "canonical" in groups[family_id]
        and solution_count_in_range(int(groups[family_id]["canonical"]["num_solutions"]))
    ]
    if not available:
        raise ValueError("No control family available")
    return available[idx % len(available)]


def _abstract_map(a: Puzzle, b: Puzzle) -> Dict[str, str]:
    a_ids = {sol.get("abstract_solution_id") for sol in a["solutions"]}
    b_ids = {sol.get("abstract_solution_id") for sol in b["solutions"]}
    return {abstract_id: abstract_id for abstract_id in sorted(a_ids & b_ids) if abstract_id}


def _solution_by_coordinate(row: Puzzle, coordinate: Sequence[int]) -> Dict[str, Any] | None:
    coord = [int(coordinate[0]), int(coordinate[1])]
    for solution in row["solutions"]:
        if [int(solution["coordinate"][0]), int(solution["coordinate"][1])] == coord:
            return solution
    return None


def _sequence_block(module: str, sequence_id: str, puzzle_type: str, trial_ids: List[str], condition: str) -> Dict[str, Any]:
    return {
        "module": module,
        "block_id": sequence_id,
        "sequence_id": sequence_id,
        "puzzle_type": puzzle_type,
        "condition": condition,
        "trial_ids": trial_ids,
    }


def _prefix(puzzle_type: str) -> str:
    return "MS" if puzzle_type == "minesweeper_lite" else "SUD"


def _bucket(value: float, cuts: Sequence[float]) -> str:
    labels = ["low", "mid", "high", "very_high"]
    for i, cut in enumerate(cuts):
        if value <= cut:
            return labels[i]
    return labels[len(cuts)]


def _prepare_output_dir(out_dir: str) -> None:
    path = Path(out_dir)
    if path.exists():
        if not path.is_dir():
            raise NotADirectoryError(f"Output path is not a directory: {path}")
        if next(path.iterdir(), None) is not None:
            raise FileExistsError(
                f"Refusing to generate modules into non-empty output directory: {path}. "
                "Choose a new or empty --out_dir."
            )
        return
    path.mkdir(parents=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate focused cue/spatial/pair/transfer/formulation study modules.")
    parser.add_argument("--data", default="data/stimuli/all_puzzles.jsonl")
    parser.add_argument("--out_dir", default="data/stimuli/modules")
    parser.add_argument("--per_type", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    counts = build_study_modules(args.data, args.out_dir, args.per_type, args.seed)
    print(f"Wrote study modules to {args.out_dir}")
    print(counts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
