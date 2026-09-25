from __future__ import annotations

import itertools
import random
from collections import Counter
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from .common import MAX_MULTI_SOLUTIONS, MIN_MULTI_SOLUTIONS, Puzzle, solution_count_in_range, stable_hash
from .probes import annotate_puzzle


Coord = Tuple[int, int]


def enumerate_placements(board: Sequence[str], m: int) -> List[Dict[str, Any]]:
    n = len(board)
    rows = range(n)
    cols = range(n)
    solutions: List[Dict[str, Any]] = []
    for chosen_rows in itertools.combinations(rows, m):
        for chosen_cols in itertools.combinations(cols, m):
            for perm in itertools.permutations(chosen_cols):
                coords = tuple(sorted((r, c) for r, c in zip(chosen_rows, perm)))
                if all(board[r][c] != "X" for r, c in coords):
                    solutions.append(_solution_record(board, coords))
    solutions.sort(key=lambda s: s["coordinates"])
    return solutions


def _solution_record(board: Sequence[str], coords0: Sequence[Coord]) -> Dict[str, Any]:
    coords1 = [[r + 1, c + 1] for r, c in coords0]
    coord_set = set(coords0)
    rows = []
    for r, row in enumerate(board):
        chars = []
        for c, ch in enumerate(row):
            chars.append("O" if (r, c) in coord_set else ch)
        rows.append(" ".join(chars))
    return {"coordinates": coords1, "board": "\n".join(rows)}


def make_puzzle(
    board: Sequence[str],
    m: int,
    min_solutions: int = MIN_MULTI_SOLUTIONS,
    max_solutions: int = MAX_MULTI_SOLUTIONS,
) -> Puzzle | None:
    solutions = enumerate_placements(board, m)
    if not solution_count_in_range(len(solutions), min_solutions, max_solutions):
        return None
    features, signature, bucket = compute_features(board, m, solutions)
    rendered = "\n".join(" ".join(row) for row in board)
    n = len(board)
    prompt = (
        "Question {{ID}}:\n"
        f"Place {m} tokens on the {n} by {n} board below.\n\n"
        "Rules: tokens cannot be on X, and no two tokens can share a row or column.\n\n"
        f"Board:\n{rendered}\n\nWrite your answer as coordinates using row and column numbers from 1 to {n}."
    )
    return annotate_puzzle({
        "id": "",
        "puzzle_type": "grid_placement",
        "prompt_text": prompt,
        "rendered_puzzle": rendered,
        "machine_readable_instance": {"n": n, "m": m, "board": list(board)},
        "num_solutions": len(solutions),
        "solutions": solutions,
        "features": features,
        "difficulty_bucket": bucket,
        "diversity_signature": signature,
    })


def compute_features(
    board: Sequence[str], m: int, solutions: Sequence[Dict[str, Any]]
) -> Tuple[Dict[str, Any], Dict[str, Any], str]:
    n = len(board)
    available = [(r, c) for r in range(n) for c in range(n) if board[r][c] != "X"]
    row_counts = [sum(1 for c in range(n) if board[r][c] != "X") for r in range(n)]
    col_counts = [sum(1 for r in range(n) if board[r][c] != "X") for c in range(n)]
    sol_sets = [{tuple(x) for x in s["coordinates"]} for s in solutions]
    avg_j, max_j = _pairwise_jaccard(sol_sets)
    forced = _forced_cells(solutions)
    center = (n + 1) / 2.0
    centers = []
    corner_hits = 0
    center_hits = 0
    corners = {(1, 1), (1, n), (n, 1), (n, n)}
    center_cells = {(n // 2 + 1, n // 2 + 1)} if n % 2 == 1 else set()
    for sol in solutions:
        coords = [tuple(c) for c in sol["coordinates"]]
        centers.append((sum(r for r, _ in coords) / len(coords), sum(c for _, c in coords) / len(coords)))
        corner_hits += sum(1 for c in coords if c in corners)
        center_hits += sum(1 for c in coords if c in center_cells)
    spread = _center_spread(centers)
    visual = _visual_solution_stats(solutions, n, m, max_j, spread)
    symmetry = _symmetry_type(board)
    features = {
        "n": n,
        "m": m,
        "num_forbidden_cells": n * n - len(available),
        "num_solutions": len(solutions),
        "row_available_counts": row_counts,
        "col_available_counts": col_counts,
        "rows_with_only_1_available_cell": sum(1 for x in row_counts if x == 1),
        "columns_with_only_1_available_cell": sum(1 for x in col_counts if x == 1),
        "has_forced_placements": bool(forced),
        "num_forced_cells": len(forced),
        "avg_pairwise_jaccard_distance": round(avg_j, 3),
        "max_pairwise_jaccard_distance": round(max_j, 3),
        "solution_center_of_mass_spread": round(spread, 3),
        "corner_usage_frequency": round(corner_hits / (len(solutions) * m), 3),
        "center_usage_frequency": round(center_hits / (len(solutions) * m), 3),
        "approximate_symmetry": symmetry != "none",
        "solution_shape_class_counts": visual["solution_shape_class_counts"],
        "visual_solution_class_counts": visual["visual_solution_class_counts"],
        "num_solution_shape_classes": visual["num_solution_shape_classes"],
        "num_visual_solution_classes": visual["num_visual_solution_classes"],
        "corner_fraction_range": round(visual["corner_fraction_range"], 3),
        "edge_fraction_range": round(visual["edge_fraction_range"], 3),
        "center_fraction_range": round(visual["center_fraction_range"], 3),
        "solution_spread_range": round(visual["solution_spread_range"], 3),
        "visual_solution_contrast_score": round(visual["visual_solution_contrast_score"], 3),
        "geometrically_diverse_solutions": visual["visual_solution_contrast_score"] >= 0.38,
        "canonical_hash": canonical_board_hash(board, m),
    }
    signature = {
        "n_m": f"{n}x{n}_{m}",
        "forbidden_bucket": _bucket(features["num_forbidden_cells"], [6, 10, 15, 21]),
        "row_pattern": sorted(row_counts),
        "col_pattern": sorted(col_counts),
        "symmetry_type": symmetry,
        "num_solutions_bucket": _bucket(len(solutions), [3, 6, 9]),
        "shape_classes": tuple(sorted(visual["solution_shape_class_counts"])),
        "visual_solution_class_count_bucket": _bucket(visual["num_visual_solution_classes"], [1, 2, 4]),
        "visual_solution_contrast_bucket": _bucket(visual["visual_solution_contrast_score"], [0.3, 0.5, 0.7]),
        "corner_heavy": features["corner_usage_frequency"] >= 0.25,
        "center_heavy": features["center_usage_frequency"] >= 0.15,
        "forced_structure": bool(forced),
    }
    return features, signature, assign_difficulty(features)


def _forced_cells(solutions: Sequence[Dict[str, Any]]) -> set[Tuple[int, int]]:
    if not solutions:
        return set()
    common = {tuple(x) for x in solutions[0]["coordinates"]}
    for sol in solutions[1:]:
        common &= {tuple(x) for x in sol["coordinates"]}
    return common


def _pairwise_jaccard(sets: Sequence[set]) -> Tuple[float, float]:
    vals = []
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            vals.append(1 - len(sets[i] & sets[j]) / len(sets[i] | sets[j]))
    return (sum(vals) / len(vals), max(vals)) if vals else (0.0, 0.0)


def _center_spread(centers: Sequence[Tuple[float, float]]) -> float:
    if not centers:
        return 0.0
    ar = sum(r for r, _ in centers) / len(centers)
    ac = sum(c for _, c in centers) / len(centers)
    return sum(((r - ar) ** 2 + (c - ac) ** 2) ** 0.5 for r, c in centers) / len(centers)


def _visual_solution_stats(
    solutions: Sequence[Dict[str, Any]],
    n: int,
    m: int,
    max_jaccard: float,
    center_spread: float,
) -> Dict[str, Any]:
    shapes: Counter[str] = Counter()
    classes: Counter[str] = Counter()
    corner_fracs: List[float] = []
    edge_fracs: List[float] = []
    center_fracs: List[float] = []
    spreads: List[float] = []
    for sol in solutions:
        coords = [tuple(c) for c in sol["coordinates"]]
        spatial = _placement_spatial_features(coords, n)
        shapes[spatial["shape_class"]] += 1
        classes[_placement_visual_class(coords, n, spatial)] += 1
        corner_fracs.append(spatial["corner_fraction"])
        edge_fracs.append(spatial["edge_fraction"])
        center_fracs.append(spatial["center_fraction"])
        spreads.append(spatial["spread"])
    corner_range = max(corner_fracs) - min(corner_fracs)
    edge_range = max(edge_fracs) - min(edge_fracs)
    center_range = max(center_fracs) - min(center_fracs)
    spread_range = max(spreads) - min(spreads)
    class_factor = min(1.0, max(0, len(classes) - 1) / 2.0)
    shape_factor = min(1.0, max(0, len(shapes) - 1) / 2.0)
    visual_range = min(1.0, corner_range + edge_range + center_range + spread_range / max(1, n))
    contrast = (
        0.30 * max_jaccard
        + 0.25 * class_factor
        + 0.20 * shape_factor
        + 0.15 * visual_range
        + 0.10 * min(1.0, center_spread / max(1.0, n * 0.2))
    )
    return {
        "solution_shape_class_counts": dict(sorted(shapes.items())),
        "visual_solution_class_counts": dict(sorted(classes.items())),
        "num_solution_shape_classes": len(shapes),
        "num_visual_solution_classes": len(classes),
        "corner_fraction_range": corner_range,
        "edge_fraction_range": edge_range,
        "center_fraction_range": center_range,
        "solution_spread_range": spread_range,
        "visual_solution_contrast_score": contrast,
    }


def _placement_spatial_features(coords: Sequence[Coord], n: int) -> Dict[str, Any]:
    corners = {(1, 1), (1, n), (n, 1), (n, n)}
    center_cells = {(n // 2 + 1, n // 2 + 1)} if n % 2 == 1 else set()
    mean_row = sum(r for r, _ in coords) / len(coords)
    mean_col = sum(c for _, c in coords) / len(coords)
    corner_count = sum(1 for coord in coords if coord in corners)
    edge_count = sum(1 for r, c in coords if r in {1, n} or c in {1, n})
    center_count = sum(1 for coord in coords if coord in center_cells)
    spread = sum(((r - mean_row) ** 2 + (c - mean_col) ** 2) ** 0.5 for r, c in coords) / len(coords)
    shape = _placement_shape_class(coords, n, spread)
    return {
        "shape_class": shape,
        "mean_row": mean_row,
        "mean_col": mean_col,
        "corner_fraction": corner_count / len(coords),
        "edge_fraction": edge_count / len(coords),
        "center_fraction": center_count / len(coords),
        "spread": spread,
    }


def _placement_visual_class(coords: Sequence[Coord], n: int, spatial: Dict[str, Any]) -> str:
    region = _center_region(spatial["mean_row"], spatial["mean_col"], n)
    corner_band = _count_band(round(spatial["corner_fraction"] * len(coords)), [0, 1])
    edge_band = _feature_band(spatial["edge_fraction"], [0.35, 0.65])
    center_band = "uses_center" if spatial["center_fraction"] > 0 else "no_center"
    spread_band = _feature_band(spatial["spread"] / max(1, n), [0.25, 0.4])
    return (
        f"{spatial['shape_class']}|{region}|corner={corner_band}|"
        f"edge={edge_band}|{center_band}|spread={spread_band}"
    )


def _placement_shape_class(coords: Sequence[Coord], n: int, spread: float) -> str:
    corners = {(1, 1), (1, n), (n, 1), (n, n)}
    corner_count = sum(1 for coord in coords if coord in corners)
    edge_count = sum(1 for r, c in coords if r in {1, n} or c in {1, n})
    diagonal = _diagonal_kind(coords, n)
    if diagonal != "none":
        return diagonal
    if corner_count >= 2:
        return "corner_heavy"
    if edge_count >= max(2, len(coords) - 1):
        return "edge_heavy"
    if spread <= 1.1:
        return "compact"
    return "spread"


def _diagonal_kind(coords: Sequence[Coord], n: int) -> str:
    same = sum(1 for r, c in coords if r == c)
    anti = sum(1 for r, c in coords if r + c == n + 1)
    if same >= len(coords) - 1:
        return "main_diagonal_like"
    if anti >= len(coords) - 1:
        return "anti_diagonal_like"
    return "none"


def _center_region(mean_row: float, mean_col: float, n: int) -> str:
    if mean_row <= n * 0.38:
        row = "top"
    elif mean_row >= n * 0.62:
        row = "bottom"
    else:
        row = "middle"
    if mean_col <= n * 0.38:
        col = "left"
    elif mean_col >= n * 0.62:
        col = "right"
    else:
        col = "center"
    return f"{row}_{col}"


def _feature_band(value: float, cuts: Sequence[float]) -> str:
    labels = ["low", "mid", "high"]
    for i, cut in enumerate(cuts):
        if value <= cut:
            return labels[i]
    return labels[-1]


def _count_band(value: int, cuts: Sequence[int]) -> str:
    labels = ["none", "one", "many"]
    for i, cut in enumerate(cuts):
        if value <= cut:
            return labels[i]
    return labels[-1]


def _bucket(value: float, cuts: Sequence[float]) -> str:
    labels = ["low", "mid", "high", "very_high", "extreme"]
    for i, cut in enumerate(cuts):
        if value <= cut:
            return labels[i]
    return labels[len(cuts)]


def _symmetry_type(board: Sequence[str]) -> str:
    if list(board) == [row[::-1] for row in board]:
        return "vertical"
    if list(board) == list(reversed(board)):
        return "horizontal"
    return "none"


def assign_difficulty(features: Dict[str, Any]) -> str:
    n, m = features["n"], features["m"]
    if n == 4 and m == 3:
        return "easy"
    if n == 6 or features["num_forced_cells"] >= 2 or features["num_forbidden_cells"] >= 15:
        return "hard"
    return "medium"


def canonical_board_hash(board: Sequence[str], m: int) -> str:
    return stable_hash({"m": m, "board": canonical_board_under_row_col_relabeling(board)})


def canonical_board_under_row_col_relabeling(board: Sequence[str]) -> List[str]:
    """Canonicalize a board under arbitrary row and column relabeling.

    Rows and columns are labels in the human prompt, but two boards that only
    rename rows/columns create the same constraint structure. For each row
    permutation, sorting the induced column vectors gives the lexicographically
    smallest column relabeling for that row order. Taking the best over all row
    permutations is exact for these small n. We also compare the transpose so
    rotation/reflection-style equivalences are covered by the same hash.
    """
    mats = [_rows_to_matrix(board), _transpose(_rows_to_matrix(board))]
    return min(_canonical_matrix(mat) for mat in mats)


def _canonical_matrix(mat: List[List[str]]) -> List[str]:
    n = len(mat)
    best: List[str] | None = None
    for row_perm in itertools.permutations(range(n)):
        row_ordered = [mat[r] for r in row_perm]
        col_vectors = []
        for c in range(n):
            col_vectors.append(tuple(row_ordered[r][c] for r in range(n)))
        col_perm = sorted(range(n), key=lambda c: col_vectors[c])
        rows = ["".join(row[c] for c in col_perm) for row in row_ordered]
        if best is None or rows < best:
            best = rows
    assert best is not None
    return best


def _rows_to_matrix(board: Sequence[str]) -> List[List[str]]:
    return [list(row) for row in board]


def _transpose(mat: List[List[str]]) -> List[List[str]]:
    return [list(row) for row in zip(*mat)]




def random_board(rng: random.Random) -> Tuple[List[str], int]:
    choice = rng.random()
    if choice < 0.24:
        n, m, lo, hi = 4, 3, 5, 9
    elif choice < 0.82:
        n, m, lo, hi = 5, 4, 9, 15
    else:
        n, m, lo, hi = 6, 5, 17, 26
    forbidden_count = rng.randint(lo, hi)
    cells = [(r, c) for r in range(n) for c in range(n)]
    forbidden = set(rng.sample(cells, forbidden_count))
    rows = []
    for r in range(n):
        rows.append("".join("X" if (r, c) in forbidden else "." for c in range(n)))
    return rows, m


def generate_pool(seed: int, valid_target: int = 500, min_attempts: int = 5000) -> List[Puzzle]:
    rng = random.Random(seed)
    pool: List[Puzzle] = []
    seen: set[str] = set()
    attempts = 0
    max_attempts = max(min_attempts, valid_target * 1000)
    while (attempts < min_attempts or len(pool) < valid_target) and attempts < max_attempts:
        attempts += 1
        board, m = random_board(rng)
        p = make_puzzle(board, m)
        if p is None:
            continue
        h = p["features"]["canonical_hash"]
        if h in seen:
            continue
        if not _passes_visual_solution_quality(p):
            continue
        seen.add(h)
        pool.append(p)
    return pool


def validate_placement(board: Sequence[str], m: int, coordinates: Sequence[Sequence[int]]) -> bool:
    n = len(board)
    coords = [(r - 1, c - 1) for r, c in coordinates]
    if len(coords) != m or len(set(coords)) != m:
        return False
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    if len(set(rows)) != m or len(set(cols)) != m:
        return False
    for r, c in coords:
        if not (0 <= r < n and 0 <= c < n) or board[r][c] == "X":
            return False
    return True


def _passes_visual_solution_quality(puzzle: Puzzle) -> bool:
    f = puzzle["features"]
    if f.get("num_visual_solution_classes", 0) < 2:
        return False
    if f.get("visual_solution_contrast_score", 0.0) < 0.34:
        return False
    if (
        f.get("num_solution_shape_classes", 0) < 2
        and f.get("max_pairwise_jaccard_distance", 0.0) < 0.55
        and f.get("solution_center_of_mass_spread", 0.0) < 0.35
    ):
        return False
    return True
