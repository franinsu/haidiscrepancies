from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from puzzles.common import read_jsonl


def build_solution_catalog(data_path: str, out_path: str) -> List[Dict[str, Any]]:
    out = Path(out_path)
    if out.exists() or out.is_symlink():
        raise FileExistsError(f"Output already exists: {out}. Choose a new destination.")
    rows = read_jsonl(data_path)
    catalog: List[Dict[str, Any]] = []
    for puzzle in rows:
        for idx, solution in enumerate(puzzle["solutions"], 1):
            solution_id = solution.get("solution_id", f"{puzzle['id']}_sol_{idx:02d}")
            features = _solution_features(puzzle, solution)
            catalog.append(
                {
                    "solution_id": solution_id,
                    "abstract_solution_id": solution.get("abstract_solution_id"),
                    "family_id": puzzle.get("family_id"),
                    "variant_type": puzzle.get("variant_type"),
                    "solution_index": idx,
                    "puzzle_id": puzzle["id"],
                    "puzzle_type": puzzle["puzzle_type"],
                    "difficulty_bucket": puzzle["difficulty_bucket"],
                    "num_solutions": puzzle["num_solutions"],
                    "canonical_answer": _canonical_answer(puzzle["puzzle_type"], solution),
                    "solution": solution,
                    "solution_features": features,
                    "solution_cluster": _solution_cluster(puzzle["puzzle_type"], features),
                }
            )
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x", encoding="utf-8") as stream:
        for row in catalog:
            stream.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
    return catalog


def _canonical_answer(puzzle_type: str, solution: Dict[str, Any]) -> Any:
    if puzzle_type == "arithmetic24":
        return solution["canonical_expression"]
    if puzzle_type == "maze":
        return solution["moves"]
    if puzzle_type == "grid_placement":
        return solution["coordinates"]
    if puzzle_type in {"minesweeper_lite", "mini_sudoku"}:
        return solution["coordinate"]
    raise ValueError(f"Unknown puzzle type: {puzzle_type}")


def _solution_features(puzzle: Dict[str, Any], solution: Dict[str, Any]) -> Dict[str, Any]:
    probe = solution.get("probe_features", {})
    ptype = puzzle["puzzle_type"]
    if ptype == "arithmetic24":
        cost = float(solution["cost"])
        return _merge_probe_features(
            {
            "expression_cost": cost,
            "cost_bucket": _bucket(cost, [3.5, 5.5]),
            "uses_division": solution["uses_division"],
            "has_fraction_intermediate": solution["has_fraction_intermediate"],
            "operation_pattern": solution["operation_pattern"],
            "operation_family": _arithmetic_operation_family(solution),
            "tree_shape": solution["tree_shape"],
            },
            probe,
        )
    if ptype == "maze":
        coords = solution["coordinates"]
        grid = puzzle["machine_readable_instance"]["grid"]
        return _merge_probe_features(
            {
            "path_length": len(solution["moves"]),
            "length_bucket": _bucket(len(solution["moves"]), [10, 14, 18]),
            "route_region": _route_region(coords, grid),
            "uses_bottleneck": _uses_bottleneck(puzzle, coords),
            "edge_touch_profile": _edge_touch_profile(coords, grid),
            "turn_count": _turn_count(solution["moves"]),
            "turn_bucket": _bucket(_turn_count(solution["moves"]), [2, 5, 8]),
            },
            probe,
        )
    if ptype == "grid_placement":
        n = puzzle["machine_readable_instance"]["n"]
        coords = [tuple(c) for c in solution["coordinates"]]
        corners = {(1, 1), (1, n), (n, 1), (n, n)}
        center_cells = {(n // 2 + 1, n // 2 + 1)} if n % 2 == 1 else set()
        mean_row = sum(r for r, _ in coords) / len(coords)
        mean_col = sum(c for _, c in coords) / len(coords)
        spread = sum(((r - mean_row) ** 2 + (c - mean_col) ** 2) ** 0.5 for r, c in coords) / len(coords)
        return _merge_probe_features(
            {
            "row_set": sorted(r for r, _ in coords),
            "col_set": sorted(c for _, c in coords),
            "corner_count": sum(1 for c in coords if c in corners),
            "center_count": sum(1 for c in coords if c in center_cells),
            "edge_count": sum(1 for r, c in coords if r in {1, n} or c in {1, n}),
            "mean_row": round(mean_row, 3),
            "mean_col": round(mean_col, 3),
            "spread": round(spread, 3),
            "spread_bucket": _bucket(spread, [1.0, 1.6, 2.2]),
            "shape_class": _placement_shape_class(coords, n, spread),
            },
            probe,
        )
    if ptype in {"minesweeper_lite", "mini_sudoku"}:
        inst = puzzle["machine_readable_instance"]
        if ptype == "minesweeper_lite":
            rows, cols = len(inst["board"]), len(inst["board"][0])
            kind = solution["cell_type"]
            coord = tuple(solution["coordinate"])
            return _merge_probe_features(
                {
                    "coordinate": list(coord),
                    "cell_type": kind,
                    "row": coord[0],
                    "col": coord[1],
                    "region": _single_coord_region(coord, rows, cols),
                },
                probe,
            )
        n = inst["n"]
        coord = tuple(solution["coordinate"])
        return _merge_probe_features(
            {
                "coordinate": list(coord),
                "digit": solution["digit"],
                "row": coord[0],
                "col": coord[1],
                "region": _single_coord_region(coord, n, n),
            },
            probe,
        )
    raise ValueError(f"Unknown puzzle type: {ptype}")


def _merge_probe_features(base: Dict[str, Any], probe: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    out.update(
        {
            "simplicity_score": probe.get("simplicity_score"),
            "simplicity_rank_within_puzzle": probe.get("simplicity_rank_within_puzzle"),
            "simplicity_percentile_within_puzzle": probe.get("simplicity_percentile_within_puzzle"),
            "simplicity_bucket": probe.get("simplicity_bucket"),
            "simplicity_components": probe.get("simplicity_components", {}),
            "simplicity_drivers": probe.get("simplicity_drivers", []),
            "spatial_features": probe.get("spatial_features", {}),
            "optimality_features": probe.get("optimality_features", {}),
            "solution_strategy_class": probe.get("solution_strategy_class"),
        }
    )
    spatial = probe.get("spatial_features", {})
    optimality = probe.get("optimality_features", {})
    for key, value in spatial.items():
        out[f"spatial_{key}"] = value
    for key, value in optimality.items():
        out[f"optimality_{key}"] = value
    return out


def _solution_cluster(puzzle_type: str, features: Dict[str, Any]) -> str:
    if puzzle_type == "arithmetic24":
        return "|".join(
            [
                "arith",
                features["operation_family"],
                f"cost={features['cost_bucket']}",
                f"division={int(features['uses_division'])}",
                f"fraction={int(features['has_fraction_intermediate'])}",
                f"simplicity={features.get('simplicity_bucket')}",
            ]
        )
    if puzzle_type == "maze":
        return "|".join(
            [
                "maze",
                f"strategy={features.get('solution_strategy_class')}",
                f"route={features['route_region']}",
                f"length={features['length_bucket']}",
                f"turns={features['turn_bucket']}",
                f"bottleneck={int(features['uses_bottleneck'])}",
                f"simplicity={features.get('simplicity_bucket')}",
            ]
        )
    if puzzle_type == "grid_placement":
        return "|".join(
            [
                "grid",
                f"strategy={features.get('solution_strategy_class')}",
                f"shape={features['shape_class']}",
                f"corners={features['corner_count']}",
                f"center={features['center_count']}",
                f"spread={features['spread_bucket']}",
                f"simplicity={features.get('simplicity_bucket')}",
            ]
        )
    if puzzle_type == "minesweeper_lite":
        return "|".join(
            [
                "mine",
                f"strategy={features.get('solution_strategy_class')}",
                f"cell_type={features.get('cell_type')}",
                f"region={features.get('region')}",
                f"simplicity={features.get('simplicity_bucket')}",
            ]
        )
    if puzzle_type == "mini_sudoku":
        return "|".join(
            [
                "sudoku",
                f"strategy={features.get('solution_strategy_class')}",
                f"region={features.get('region')}",
                f"simplicity={features.get('simplicity_bucket')}",
            ]
        )
    raise ValueError(f"Unknown puzzle type: {puzzle_type}")


def _bucket(value: float, cuts: List[float]) -> str:
    labels = ["low", "mid", "high", "very_high"]
    for i, cut in enumerate(cuts):
        if value <= cut:
            return labels[i]
    return labels[len(cuts)]


def _arithmetic_operation_family(solution: Dict[str, Any]) -> str:
    pattern = solution["operation_pattern"]
    if solution["has_fraction_intermediate"]:
        return "fraction_trick"
    if "/" in pattern:
        return "division"
    if "*" in pattern and "-" in pattern:
        return "difference_product"
    if "*" in pattern and "+" in pattern:
        return "additive_product"
    if pattern.count("+") + pattern.count("-") == len(pattern):
        return "additive"
    if pattern.count("*") == len(pattern):
        return "multiplicative"
    return "mixed"


def _route_region(coords: List[List[int]], grid: List[str]) -> str:
    rows, cols = len(grid), len(grid[0])
    mean_r = sum(r for r, _ in coords) / len(coords)
    mean_c = sum(c for _, c in coords) / len(coords)
    if mean_r <= rows * 0.38:
        vertical = "top"
    elif mean_r >= rows * 0.62:
        vertical = "bottom"
    else:
        vertical = "middle"
    if mean_c <= cols * 0.38:
        horizontal = "left"
    elif mean_c >= cols * 0.62:
        horizontal = "right"
    else:
        horizontal = "center"
    return f"{vertical}_{horizontal}"


def _single_coord_region(coord: tuple[int, int], rows: int, cols: int) -> str:
    r, c = coord
    if r <= rows * 0.38:
        vertical = "top"
    elif r >= rows * 0.62:
        vertical = "bottom"
    else:
        vertical = "middle"
    if c <= cols * 0.38:
        horizontal = "left"
    elif c >= cols * 0.62:
        horizontal = "right"
    else:
        horizontal = "center"
    return f"{vertical}_{horizontal}"


def _uses_bottleneck(puzzle: Dict[str, Any], coords: List[List[int]]) -> bool:
    all_solutions = puzzle["solutions"]
    if len(all_solutions) < 2:
        return False
    common = {tuple(c) for c in all_solutions[0]["coordinates"]}
    for sol in all_solutions[1:]:
        common &= {tuple(c) for c in sol["coordinates"]}
    endpoints = {
        tuple(all_solutions[0]["coordinates"][0]),
        tuple(all_solutions[0]["coordinates"][-1]),
    }
    bottlenecks = common - endpoints
    return bool({tuple(c) for c in coords} & bottlenecks)


def _edge_touch_profile(coords: List[List[int]], grid: List[str]) -> str:
    rows, cols = len(grid), len(grid[0])
    touches = {
        "top": any(r == 1 for r, _ in coords),
        "bottom": any(r == rows for r, _ in coords),
        "left": any(c == 1 for _, c in coords),
        "right": any(c == cols for _, c in coords),
    }
    active = [name for name, value in touches.items() if value]
    return "_".join(active) if active else "interior"


def _turn_count(moves: str) -> int:
    return sum(1 for a, b in zip(moves, moves[1:]) if a != b)


def _placement_shape_class(coords: List[tuple[int, int]], n: int, spread: float) -> str:
    corners = {(1, 1), (1, n), (n, 1), (n, n)}
    corner_count = sum(1 for coord in coords if coord in corners)
    edge_count = sum(1 for r, c in coords if r in {1, n} or c in {1, n})
    same = sum(1 for r, c in coords if r == c)
    anti = sum(1 for r, c in coords if r + c == n + 1)
    if same >= len(coords) - 1:
        return "main_diagonal_like"
    if anti >= len(coords) - 1:
        return "anti_diagonal_like"
    if corner_count >= 2:
        return "corner_heavy"
    if edge_count >= max(2, len(coords) - 1):
        return "edge_heavy"
    if spread <= 1.1:
        return "compact"
    return "spread"


def _is_diagonal_like(coords: List[tuple[int, int]], n: int) -> bool:
    if len(coords) < 2:
        return False
    same = sum(1 for r, c in coords if r == c)
    anti = sum(1 for r, c in coords if r + c == n + 1)
    return same >= len(coords) - 1 or anti >= len(coords) - 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Build stable solution IDs and per-solution features.")
    parser.add_argument("--data", default="data/stimuli/all_puzzles.jsonl")
    parser.add_argument("--out", required=True, help="New solution catalog file; must not exist.")
    args = parser.parse_args()
    catalog = build_solution_catalog(args.data, args.out)
    print(f"Wrote {len(catalog)} solution rows to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
