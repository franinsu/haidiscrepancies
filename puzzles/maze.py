from __future__ import annotations

import random
from collections import Counter, deque
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from .common import MAX_MULTI_SOLUTIONS, MIN_MULTI_SOLUTIONS, Puzzle, solution_count_in_range, stable_hash
from .probes import annotate_puzzle


Coord = Tuple[int, int]
MOVES = [("U", (-1, 0)), ("D", (1, 0)), ("L", (0, -1)), ("R", (0, 1))]


def find_points(grid: Sequence[str]) -> Tuple[Coord, Coord]:
    start = goal = None
    for r, row in enumerate(grid):
        for c, ch in enumerate(row):
            if ch == "S":
                start = (r, c)
            elif ch == "G":
                goal = (r, c)
    if start is None or goal is None:
        raise ValueError("Grid must contain S and G")
    return start, goal


def neighbors(grid: Sequence[str], cell: Coord) -> Iterable[Tuple[str, Coord]]:
    rows, cols = len(grid), len(grid[0])
    r, c = cell
    for move, (dr, dc) in MOVES:
        nr, nc = r + dr, c + dc
        if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] != "#":
            yield move, (nr, nc)


def bfs_distances(grid: Sequence[str], start: Coord) -> Dict[Coord, int]:
    q = deque([start])
    dist = {start: 0}
    while q:
        cell = q.popleft()
        for _, nxt in neighbors(grid, cell):
            if nxt not in dist:
                dist[nxt] = dist[cell] + 1
                q.append(nxt)
    return dist


def enumerate_shortest_paths(
    grid: Sequence[str], max_paths: int | None = None
) -> Tuple[int | None, List[Dict[str, Any]], bool]:
    start, goal = find_points(grid)
    ds = bfs_distances(grid, start)
    if goal not in ds:
        return None, [], False
    dg = bfs_distances(grid, goal)
    length = ds[goal]
    paths: List[Dict[str, Any]] = []
    exceeded = False

    def dfs(cell: Coord, moves: List[str], coords: List[Coord]) -> None:
        nonlocal exceeded
        if exceeded:
            return
        if cell == goal:
            paths.append(
                {
                    "moves": "".join(moves),
                    "coordinates": [[r + 1, c + 1] for r, c in coords],
                }
            )
            if max_paths is not None and len(paths) > max_paths:
                exceeded = True
            return
        for move, nxt in neighbors(grid, cell):
            if ds.get(nxt) == ds[cell] + 1 and ds[nxt] + dg.get(nxt, 10**9) == length:
                moves.append(move)
                coords.append(nxt)
                dfs(nxt, moves, coords)
                coords.pop()
                moves.pop()

    dfs(start, [], [start])
    paths.sort(key=lambda p: p["moves"])
    return length, paths, exceeded


def make_puzzle(
    grid: Sequence[str],
    min_solutions: int = MIN_MULTI_SOLUTIONS,
    max_solutions: int = MAX_MULTI_SOLUTIONS,
) -> Puzzle | None:
    length, paths, exceeded = enumerate_shortest_paths(grid, max_paths=max_solutions)
    if length is None or exceeded or not solution_count_in_range(len(paths), min_solutions, max_solutions):
        return None
    features, signature, bucket = compute_features(grid, paths, length)
    rendered = "\n".join(" ".join(row) for row in grid)
    prompt = (
        "Question {{ID}}:\n"
        "Find one shortest path from S to G. Move only U, D, L, or R. You cannot pass through #.\n\n"
        f"Grid:\n{rendered}\n\nWrite your path as a string of moves using U, D, L, R."
    )
    return annotate_puzzle({
        "id": "",
        "puzzle_type": "maze",
        "prompt_text": prompt,
        "rendered_puzzle": rendered,
        "machine_readable_instance": {"grid": list(grid)},
        "num_solutions": len(paths),
        "solutions": paths,
        "features": features,
        "difficulty_bucket": bucket,
        "diversity_signature": signature,
    })


def compute_features(
    grid: Sequence[str], paths: Sequence[Dict[str, Any]], shortest_length: int
) -> Tuple[Dict[str, Any], Dict[str, Any], str]:
    rows, cols = len(grid), len(grid[0])
    open_cells = [(r, c) for r in range(rows) for c in range(cols) if grid[r][c] != "#"]
    wall_count = rows * cols - len(open_cells)
    wall_density = wall_count / (rows * cols)
    start, goal = find_points(grid)
    manhattan = abs(start[0] - goal[0]) + abs(start[1] - goal[1])
    detour_over_manhattan = shortest_length - manhattan
    reachable = set(bfs_distances(grid, start))
    dead_ends = sum(1 for cell in reachable if len(list(neighbors(grid, cell))) == 1 and cell not in (start, goal))
    branch_points = sum(1 for cell in reachable if len(list(neighbors(grid, cell))) >= 3)
    path_sets = [
        {tuple(coord) for coord in p["coordinates"]}
        for p in paths
    ]
    avg_j, max_j = _pairwise_jaccard(path_sets)
    route_stats = _route_structure_stats(paths, rows, cols)
    all_path_counts = Counter(cell for s in path_sets for cell in s)
    bottleneck = any(count == len(paths) for cell, count in all_path_counts.items() if cell not in {(start[0]+1, start[1]+1), (goal[0]+1, goal[1]+1)})
    branch_on_path = max(
        sum(1 for coord in p["coordinates"] if len(list(neighbors(grid, (coord[0] - 1, coord[1] - 1)))) >= 3)
        for p in paths
    )
    symmetric = _symmetry_type(grid) != "none"
    macros = (
        max_j >= 0.35
        or route_stats["num_route_classes"] >= 2
        or route_stats["path_center_spread"] >= 1.0
    )
    features = {
        "grid_size": rows,
        "wall_density": round(wall_density, 3),
        "shortest_length": shortest_length,
        "direct_manhattan_distance": manhattan,
        "shortest_detour_over_manhattan": detour_over_manhattan,
        "num_shortest_paths": len(paths),
        "num_dead_ends": dead_ends,
        "num_branch_points_reachable_from_start": branch_points,
        "branch_points_on_shortest_path": branch_on_path,
        "avg_pairwise_jaccard_distance": round(avg_j, 3),
        "max_pairwise_jaccard_distance": round(max_j, 3),
        "approximate_symmetry": symmetric,
        "has_bottleneck_cell": bottleneck,
        "macroscopically_different_route_classes": macros,
        "route_region_counts": route_stats["route_region_counts"],
        "route_class_counts": route_stats["route_class_counts"],
        "num_route_regions": route_stats["num_route_regions"],
        "num_route_classes": route_stats["num_route_classes"],
        "path_center_spread": round(route_stats["path_center_spread"], 3),
        "path_mean_row_range": round(route_stats["mean_row_range"], 3),
        "path_mean_col_range": round(route_stats["mean_col_range"], 3),
        "path_edge_fraction_range": round(route_stats["edge_fraction_range"], 3),
        "path_center_fraction_range": round(route_stats["center_fraction_range"], 3),
        "path_turn_count_range": route_stats["turn_count_range"],
        "route_class_contrast_score": round(route_stats["route_class_contrast_score"], 3),
        "has_meaningful_route_contrast": route_stats["route_class_contrast_score"] >= 0.42,
        "canonical_hash": canonical_grid_hash(grid),
    }
    signature = {
        "grid_size": rows,
        "wall_density_bucket": _bucket(wall_density, [0.2, 0.3]),
        "sg_relative_position": _sg_relation(start, goal, rows, cols),
        "route_type": _route_type(start, goal, rows, cols, bottleneck, symmetric),
        "route_class_count_bucket": _bucket(route_stats["num_route_classes"], [1, 2, 4]),
        "route_contrast_bucket": _bucket(route_stats["route_class_contrast_score"], [0.35, 0.55, 0.75]),
        "shortest_length_bucket": _bucket(shortest_length, [10, 15, 19]),
        "num_shortest_paths_bucket": _bucket(len(paths), [3, 6, 9]),
        "path_diversity_bucket": _bucket(max_j, [0.25, 0.45, 0.65]),
    }
    return features, signature, assign_difficulty(features)


def _pairwise_jaccard(sets: Sequence[set]) -> Tuple[float, float]:
    vals = []
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            union = sets[i] | sets[j]
            vals.append(0.0 if not union else 1 - len(sets[i] & sets[j]) / len(union))
    return (sum(vals) / len(vals), max(vals)) if vals else (0.0, 0.0)


def _route_structure_stats(paths: Sequence[Dict[str, Any]], rows: int, cols: int) -> Dict[str, Any]:
    regions: Counter[str] = Counter()
    classes: Counter[str] = Counter()
    centers: List[Tuple[float, float]] = []
    mean_rows: List[float] = []
    mean_cols: List[float] = []
    edge_fracs: List[float] = []
    center_fracs: List[float] = []
    turn_counts: List[int] = []
    for path in paths:
        coords = [(int(r), int(c)) for r, c in path["coordinates"]]
        stats = _path_shape_stats(coords, rows, cols, path["moves"])
        regions[stats["region"]] += 1
        classes[stats["route_class"]] += 1
        centers.append((stats["mean_row"], stats["mean_col"]))
        mean_rows.append(stats["mean_row"])
        mean_cols.append(stats["mean_col"])
        edge_fracs.append(stats["edge_fraction"])
        center_fracs.append(stats["center_fraction"])
        turn_counts.append(stats["turn_count"])
    center_spread = _center_spread(centers)
    class_factor = min(1.0, max(0, len(classes) - 1) / 2.0)
    region_factor = min(1.0, max(0, len(regions) - 1) / 2.0)
    row_col_range = (max(mean_rows) - min(mean_rows) + max(mean_cols) - min(mean_cols)) / max(1, rows + cols)
    edge_center_range = (max(edge_fracs) - min(edge_fracs)) + (max(center_fracs) - min(center_fracs))
    turn_range = max(turn_counts) - min(turn_counts)
    contrast = (
        0.28 * class_factor
        + 0.18 * region_factor
        + 0.24 * min(1.0, center_spread / max(1.0, rows * 0.18))
        + 0.18 * min(1.0, edge_center_range)
        + 0.12 * min(1.0, turn_range / 4.0)
    )
    return {
        "route_region_counts": dict(sorted(regions.items())),
        "route_class_counts": dict(sorted(classes.items())),
        "num_route_regions": len(regions),
        "num_route_classes": len(classes),
        "path_center_spread": center_spread,
        "mean_row_range": max(mean_rows) - min(mean_rows),
        "mean_col_range": max(mean_cols) - min(mean_cols),
        "edge_fraction_range": max(edge_fracs) - min(edge_fracs),
        "center_fraction_range": max(center_fracs) - min(center_fracs),
        "turn_count_range": turn_range,
        "route_class_contrast_score": contrast,
    }


def _path_shape_stats(coords: Sequence[Coord], rows: int, cols: int, moves: str) -> Dict[str, Any]:
    mean_row = sum(r for r, _ in coords) / len(coords)
    mean_col = sum(c for _, c in coords) / len(coords)
    edge_fraction = sum(1 for r, c in coords if r in {1, rows} or c in {1, cols}) / len(coords)
    center_fraction = sum(
        1
        for r, c in coords
        if rows * 0.35 <= r <= rows * 0.65 and cols * 0.35 <= c <= cols * 0.65
    ) / len(coords)
    region = _route_region(mean_row, mean_col, rows, cols)
    edge_band = _feature_band(edge_fraction, [0.25, 0.5])
    center_band = _feature_band(center_fraction, [0.2, 0.45])
    turn_band = _feature_band(_turn_count(moves), [3, 6])
    return {
        "mean_row": mean_row,
        "mean_col": mean_col,
        "edge_fraction": edge_fraction,
        "center_fraction": center_fraction,
        "turn_count": _turn_count(moves),
        "region": region,
        "route_class": f"{region}|edge={edge_band}|center={center_band}|turns={turn_band}",
    }


def _route_region(mean_r: float, mean_c: float, rows: int, cols: int) -> str:
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


def _feature_band(value: float, cuts: Sequence[float]) -> str:
    labels = ["low", "mid", "high"]
    for i, cut in enumerate(cuts):
        if value <= cut:
            return labels[i]
    return labels[-1]


def _turn_count(moves: str) -> int:
    return sum(1 for a, b in zip(moves, moves[1:]) if a != b)


def _center_spread(centers: Sequence[Tuple[float, float]]) -> float:
    if not centers:
        return 0.0
    ar = sum(r for r, _ in centers) / len(centers)
    ac = sum(c for _, c in centers) / len(centers)
    return sum(((r - ar) ** 2 + (c - ac) ** 2) ** 0.5 for r, c in centers) / len(centers)


def _bucket(value: float, cuts: Sequence[float]) -> str:
    labels = ["low", "mid", "high", "very_high"]
    for i, cut in enumerate(cuts):
        if value <= cut:
            return labels[i]
    return labels[len(cuts)]


def _symmetry_type(grid: Sequence[str]) -> str:
    plain = [row.replace("S", ".").replace("G", ".") for row in grid]
    if plain == [row[::-1] for row in plain]:
        return "vertical"
    if plain == list(reversed(plain)):
        return "horizontal"
    return "none"


def _sg_relation(start: Coord, goal: Coord, rows: int, cols: int) -> str:
    sr, sc = start
    gr, gc = goal
    if abs(sr - gr) > abs(sc - gc) * 1.5:
        return "top_bottom"
    if abs(sc - gc) > abs(sr - gr) * 1.5:
        return "left_right"
    if (sr in (0, rows - 1) and sc in (0, cols - 1) and gr in (0, rows - 1) and gc in (0, cols - 1)):
        return "corner_corner"
    return "diagonal_or_central"


def _route_type(start: Coord, goal: Coord, rows: int, cols: int, bottleneck: bool, symmetric: bool) -> str:
    if symmetric:
        return "symmetric"
    if bottleneck:
        return "bottleneck"
    return _sg_relation(start, goal, rows, cols)


def assign_difficulty(features: Dict[str, Any]) -> str:
    length = features["shortest_length"]
    branch = features["num_branch_points_reachable_from_start"]
    dead = features["num_dead_ends"]
    if length <= 12 and branch <= 8 and dead <= 5:
        return "easy"
    if length >= 19 or branch >= 15 or dead >= 9:
        return "hard"
    return "medium"


def canonical_grid_hash(grid: Sequence[str]) -> str:
    variants = _grid_variants(grid)
    return stable_hash(min(variants))


def _grid_variants(grid: Sequence[str]) -> List[List[str]]:
    mats = [[ch for ch in row] for row in grid]

    def to_rows(mat: List[List[str]]) -> List[str]:
        return ["".join(row) for row in mat]

    def rot(mat: List[List[str]]) -> List[List[str]]:
        return [list(row) for row in zip(*mat[::-1])]

    def flip(mat: List[List[str]]) -> List[List[str]]:
        return [list(reversed(row)) for row in mat]

    out = []
    cur = mats
    for _ in range(4):
        out.append(to_rows(cur))
        out.append(to_rows(flip(cur)))
        cur = rot(cur)
    return out


def random_grid(rng: random.Random) -> List[str]:
    n = rng.choice([6, 7, 8])
    wall_p = rng.uniform(0.22, 0.38)
    start, goal = _choose_start_goal(rng, n)
    rows: List[List[str]] = []
    for r in range(n):
        row = []
        for c in range(n):
            row.append("#" if rng.random() < wall_p else ".")
        rows.append(row)
    rows[start[0]][start[1]] = "S"
    rows[goal[0]][goal[1]] = "G"
    return ["".join(row) for row in rows]


def _choose_start_goal(rng: random.Random, n: int) -> Tuple[Coord, Coord]:
    border = [(0, c) for c in range(n)] + [(n - 1, c) for c in range(n)] + [(r, 0) for r in range(1, n - 1)] + [(r, n - 1) for r in range(1, n - 1)]
    pattern = rng.choice(["corners", "left_right", "top_bottom", "mixed"])
    if pattern == "corners":
        pairs = [((0, 0), (n - 1, n - 1)), ((0, n - 1), (n - 1, 0))]
        return rng.choice(pairs)
    if pattern == "left_right":
        return (rng.randrange(n), 0), (rng.randrange(n), n - 1)
    if pattern == "top_bottom":
        return (0, rng.randrange(n)), (n - 1, rng.randrange(n))
    s = rng.choice(border)
    far = [p for p in border if abs(p[0] - s[0]) + abs(p[1] - s[1]) >= n - 1]
    return s, rng.choice(far)


def generate_pool(seed: int, valid_target: int = 500, min_attempts: int = 5000) -> List[Puzzle]:
    rng = random.Random(seed)
    pool: List[Puzzle] = []
    seen: set[str] = set()
    near_seen: Dict[Tuple[Any, ...], List[List[str]]] = {}
    attempts = 0
    max_attempts = max(min_attempts, valid_target * 1000)
    while (attempts < min_attempts or len(pool) < valid_target) and attempts < max_attempts:
        attempts += 1
        grid = random_grid(rng)
        p = make_puzzle(grid)
        if p is None:
            continue
        h = p["features"]["canonical_hash"]
        if h in seen:
            continue
        if not (8 <= p["features"]["shortest_length"] <= 22):
            continue
        if p["features"]["wall_density"] < 0.2:
            continue
        if not _passes_route_quality_filter(p, strict=True):
            continue
        near_key = _near_duplicate_key(p)
        if _has_near_duplicate(grid, near_seen.get(near_key, [])):
            continue
        seen.add(h)
        near_seen.setdefault(near_key, []).append(list(grid))
        pool.append(p)
    return pool


def validate_path(grid: Sequence[str], moves: str) -> bool:
    length, paths, exceeded = enumerate_shortest_paths(grid, max_paths=None)
    if length is None or exceeded:
        return False
    shortest = {p["moves"] for p in paths}
    return moves in shortest


def _near_duplicate_key(puzzle: Puzzle) -> Tuple[Any, ...]:
    sig = puzzle["diversity_signature"]
    return (
        sig["grid_size"],
        sig["wall_density_bucket"],
        sig["sg_relative_position"],
        sig["route_contrast_bucket"],
    )


def _passes_route_quality_filter(puzzle: Puzzle, strict: bool) -> bool:
    f = puzzle["features"]
    contrast = f.get("route_class_contrast_score", 0.0)
    max_j = f.get("max_pairwise_jaccard_distance", 0.0)
    classes = f.get("num_route_classes", 1)
    regions = f.get("num_route_regions", 1)
    turn_range = f.get("path_turn_count_range", 0)
    human_difficulty = f.get("human_difficulty_score", 50.0)
    if human_difficulty < (24.0 if strict else 22.0):
        return False
    if strict:
        return contrast >= 0.42 and (classes >= 2 or regions >= 2 or max_j >= 0.5)
    return contrast >= 0.34 and (classes >= 2 or regions >= 2 or max_j >= 0.45 or turn_range >= 2)


def _has_near_duplicate(grid: Sequence[str], previous: Sequence[Sequence[str]], threshold: float = 0.16) -> bool:
    if not previous:
        return False
    for other in previous:
        if len(grid) != len(other) or len(grid[0]) != len(other[0]):
            continue
        if _min_variant_hamming(grid, other) <= threshold:
            return True
    return False


def _min_variant_hamming(a: Sequence[str], b: Sequence[str]) -> float:
    total = len(a) * len(a[0])
    best = 1.0
    flat_a = "".join(a)
    for variant in _grid_variants(b):
        if len(variant) != len(a) or len(variant[0]) != len(a[0]):
            continue
        diff = sum(ch1 != ch2 for ch1, ch2 in zip(flat_a, "".join(variant)))
        best = min(best, diff / total)
    return best
