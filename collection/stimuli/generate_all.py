from __future__ import annotations

import argparse
import os
import random
from collections import Counter
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from puzzles import arithmetic24, grid_placement, maze, minesweeper_lite, mini_sudoku
from puzzles.common import equal_bucket_counts, prune_overgenerated, solution_count_in_range, write_jsonl
from puzzles.diversity import PROBE_PRESSURE_ORDER, choose_final_dataset
from puzzles.probes import rebucket_search_pressure
from puzzles.variants import build_family_variants


def generate_all(num_base_per_type: int, seed: int, out_dir: str) -> list[dict]:
    _prepare_output_dir(out_dir)
    overgenerate_num = max(num_base_per_type, int(round(num_base_per_type * 1.5)))
    specs = [
        ("arithmetic24", "A24", arithmetic24.generate_pool, seed + 11, _quality_arithmetic24),
        ("maze", "MZ", maze.generate_pool, seed + 22, _quality_maze),
        ("grid_placement", "GP", grid_placement.generate_pool, seed + 33, _quality_grid_placement),
        ("minesweeper_lite", "MS", minesweeper_lite.generate_pool, seed + 44, _quality_minesweeper_lite),
        ("mini_sudoku", "SUD", mini_sudoku.generate_pool, seed + 55, _quality_mini_sudoku),
    ]
    all_rows: list[dict] = []
    for puzzle_type, prefix, pool_fn, type_seed, quality_fn in specs:
        print(f"Generating candidate pool for {puzzle_type}...", flush=True)
        pool = pool_fn(type_seed, valid_target=max(220, overgenerate_num * 5), min_attempts=2500)
        pool = [p for p in pool if solution_count_in_range(int(p["num_solutions"]))]
        if len(pool) < overgenerate_num:
            raise RuntimeError(f"{puzzle_type}: only {len(pool)} valid 3-8 solution candidates found")
        rebucket_search_pressure(pool)
        print(
            f"  valid candidates: {len(pool)}; pressure: {dict(Counter(p['features']['search_pressure_bucket'] for p in pool))}; cells: {dict(Counter(p['features']['probe_cell'] for p in pool))}",
            flush=True,
        )
        overselected = choose_final_dataset(pool, overgenerate_num, type_seed)
        print(
            f"  overselected: {len(overselected)}; pressure: {dict(Counter(p['features']['search_pressure_bucket'] for p in overselected))}",
            flush=True,
        )
        chosen = prune_overgenerated(
            overselected,
            num_base_per_type,
            quality_fn,
            random.Random(type_seed + 999),
            desired=equal_bucket_counts(num_base_per_type, PROBE_PRESSURE_ORDER),
            stratify_key="search_pressure_bucket",
            bucket_order=PROBE_PRESSURE_ORDER,
        )
        print(
            f"  pruned to: {len(chosen)}; pressure: {dict(Counter(p['features']['search_pressure_bucket'] for p in chosen))}; human difficulty: {dict(Counter(p['difficulty_bucket'] for p in chosen))}",
            flush=True,
        )
        type_rows: list[dict] = []
        for i, base in enumerate(chosen, 1):
            family_id = f"{prefix}_{i:04d}"
            variants = build_family_variants(
                base,
                family_id,
                random.Random(type_seed * 1000 + i),
                include_formulation=False,
            )
            type_rows.extend(variants)
        print(
            f"  variants: {len(type_rows)}; variant types: {dict(Counter(p['variant_type'] for p in type_rows))}",
            flush=True,
        )
        all_rows.extend(type_rows)
    write_jsonl(os.path.join(out_dir, "all_puzzles.jsonl"), all_rows)
    return all_rows


def _prepare_output_dir(out_dir: str) -> None:
    path = Path(out_dir)
    if path.exists():
        if not path.is_dir():
            raise NotADirectoryError(f"Output path is not a directory: {path}")
        if next(path.iterdir(), None) is not None:
            raise FileExistsError(
                f"Refusing to generate into non-empty output directory: {path}. "
                "Choose a new or empty --out_dir."
            )
        return
    path.mkdir(parents=True)


def _closeness(value: float, target: float, scale: float) -> float:
    return max(0.0, 1.0 - abs(value - target) / scale)


def _quality_arithmetic24(p: dict) -> float:
    f = p["features"]
    score = 0.0
    score += 2.0 * _closeness(p["num_solutions"], 5, 4)
    score += 1.5 * _closeness(f["min_expression_cost"], 4.2, 2.5)
    score += 0.8 if not f.get("classic_overused", False) else -2.0
    score += 0.5 if f.get("number_range", 0) >= 4 else 0.0
    score += 0.4 if len(f.get("operation_pattern_counts", {})) >= 2 else 0.0
    score += 0.6 * _closeness(f.get("simplicity_contrast_score", 0.0), 1.2, 1.5)
    score += 0.4 * _closeness(f.get("search_pressure_score", 50.0), 50.0, 40.0)
    return score


def _quality_maze(p: dict) -> float:
    f = p["features"]
    score = 0.0
    score += 1.5 * _closeness(f["shortest_length"], 14, 6)
    score += 1.2 * _closeness(p["num_solutions"], 5, 4)
    score += 1.0 * f.get("max_pairwise_jaccard_distance", 0.0)
    score += 0.8 * _closeness(f["wall_density"], 0.27, 0.16)
    score += 0.5 if f.get("macroscopically_different_route_classes", False) else 0.0
    score += 1.0 * f.get("route_class_contrast_score", 0.0)
    score += 0.25 * min(3, max(0, f.get("num_route_classes", 1) - 1))
    score -= 0.08 * max(0, f.get("num_dead_ends", 0) - 7)
    score += 0.5 * _closeness(f.get("simplicity_contrast_score", 0.0), 0.8, 1.2)
    score += 0.8 * f.get("spatial_contrast_score", 0.0)
    score += 0.5 * f.get("solution_diversity_score", 0.0)
    score += 0.7 * _closeness(f.get("human_difficulty_score", 50.0), 50.0, 35.0)
    score -= 0.5 if f.get("human_difficulty_score", 50.0) < 25.0 else 0.0
    return score


def _quality_grid_placement(p: dict) -> float:
    f = p["features"]
    score = 0.0
    score += 1.8 * _closeness(p["num_solutions"], 6, 4)
    score += 1.0 * f.get("max_pairwise_jaccard_distance", 0.0)
    score += 0.7 if f.get("geometrically_diverse_solutions", False) else 0.0
    score += 1.1 * f.get("visual_solution_contrast_score", 0.0)
    score += 0.25 * min(3, max(0, f.get("num_visual_solution_classes", 1) - 1))
    score += 0.25 * min(2, max(0, f.get("num_solution_shape_classes", 1) - 1))
    score += 0.6 if (f.get("n"), f.get("m")) == (5, 4) else 0.2
    score -= 0.4 * f.get("num_forced_cells", 0)
    score += 0.8 * _closeness(f.get("simplicity_contrast_score", 0.0), 0.7, 1.0)
    score += 0.7 * f.get("spatial_contrast_score", 0.0)
    score += 0.4 * f.get("solution_diversity_score", 0.0)
    score += 0.8 * _closeness(f.get("human_difficulty_score", 50.0), 48.0, 35.0)
    return score


def _quality_minesweeper_lite(p: dict) -> float:
    f = p["features"]
    score = 0.0
    score += 1.5 * _closeness(p["num_solutions"], 5, 4)
    score += 1.2 * _closeness(f.get("num_unknown_cells", 10), 10, 6)
    score += 0.9 * _closeness(f.get("num_consistent_assignments", 4), 6, 12)
    score += 1.2 * _closeness(f.get("solution_local_clue_contrast", 0), 3, 4)
    score += 0.8 * _closeness(f.get("human_difficulty_score", 45.0), 45.0, 35.0)
    return score


def _quality_mini_sudoku(p: dict) -> float:
    f = p["features"]
    score = 0.0
    score += 1.5 * _closeness(p["num_solutions"], 5, 4)
    score += 1.0 if f.get("n") == 6 else 0.5
    score += 1.2 * _closeness(f.get("solution_visibility_contrast", 0), 5, 5)
    score += 0.8 * _closeness(f.get("clue_density", 0.4), 0.4, 0.25)
    score += 0.8 * _closeness(f.get("human_difficulty_score", 45.0), 45.0, 35.0)
    return score


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate new candidate reasoning-puzzle datasets; frozen study selection is a separate input.",
        epilog="The selected Arithmetic 24 replacement provenance is data/stimuli/selection_provenance/arithmetic24_candidate_30_review.jsonl. A seed alone does not reconstruct that reviewed selection; use the frozen data/stimuli files for paper reproduction.",
    )
    parser.add_argument("--num_base_per_type", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out_dir", default="data/stimuli")
    args = parser.parse_args(argv)
    num_base = args.num_base_per_type
    rows = generate_all(num_base, args.seed, args.out_dir)
    print(f"Done. Wrote {len(rows)} puzzle variants from {num_base * 5} base families to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
