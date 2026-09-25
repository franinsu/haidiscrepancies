from __future__ import annotations

import math
from typing import Any, Dict, List, Sequence, Tuple


Coord = Tuple[int, int]


REASONING_FAMILIES = {
    "arithmetic24": "symbolic_arithmetic",
    "maze": "optimal_path_planning",
    "grid_placement": "visual_constraint_satisfaction",
    "minesweeper_lite": "local_visual_constraint_reasoning",
    "mini_sudoku": "visual_symbolic_constraint_reasoning",
}


def rebucket_search_pressure(puzzles: Sequence[Dict[str, Any]]) -> None:
    """Assign low/medium/high pressure buckets by within-type score tertiles."""
    if not puzzles:
        return
    ordered = sorted(
        puzzles,
        key=lambda p: (
            p["features"].get("search_pressure_score", 0.0),
            p["features"].get("canonical_hash", ""),
        ),
    )
    n = len(ordered)
    for i, puzzle in enumerate(ordered):
        if i < n / 3:
            bucket = "low"
        elif i < 2 * n / 3:
            bucket = "medium"
        else:
            bucket = "high"
        _set_search_pressure_bucket(puzzle, bucket)


def annotate_puzzle(puzzle: Dict[str, Any]) -> Dict[str, Any]:
    """Attach probe-oriented metadata used by the study and analysis tools."""
    ptype = puzzle["puzzle_type"]
    if ptype == "arithmetic24":
        _annotate_arithmetic(puzzle)
    elif ptype == "maze":
        _annotate_maze(puzzle)
    elif ptype == "grid_placement":
        _annotate_grid_placement(puzzle)
    elif ptype == "minesweeper_lite":
        _annotate_minesweeper_lite(puzzle)
    elif ptype == "mini_sudoku":
        _annotate_mini_sudoku(puzzle)
    else:
        raise ValueError(f"Unknown puzzle type: {ptype}")

    _rank_solution_simplicity(puzzle)
    _attach_common_probe_metadata(puzzle)
    return puzzle


def _set_search_pressure_bucket(puzzle: Dict[str, Any], bucket: str) -> None:
    features = puzzle["features"]
    metadata = puzzle.get("probe_metadata", {})
    signature = puzzle["diversity_signature"]
    solution_count_bucket = features["solution_count_bucket"]
    simplicity_contrast_bucket = features["simplicity_contrast_bucket"]
    spatial_contrast_bucket = features["spatial_contrast_bucket"]
    metadata["search_pressure_bucket"] = bucket
    metadata["probe_cell"] = f"{bucket}_{solution_count_bucket}"
    metadata["probe_tags"] = _probe_tags(puzzle, bucket, simplicity_contrast_bucket, spatial_contrast_bucket)
    puzzle["probe_metadata"] = metadata
    features["search_pressure_bucket"] = bucket
    features["probe_cell"] = metadata["probe_cell"]
    signature["search_pressure_bucket"] = bucket
    signature["probe_cell"] = metadata["probe_cell"]


def _attach_common_probe_metadata(puzzle: Dict[str, Any]) -> None:
    ptype = puzzle["puzzle_type"]
    features = puzzle["features"]
    solutions = puzzle["solutions"]
    simplicity_scores = [float(s["probe_features"]["simplicity_score"]) for s in solutions]
    simplicity_contrast = max(simplicity_scores) - min(simplicity_scores) if simplicity_scores else 0.0
    spatial_contrast = _spatial_contrast_score(solutions)
    solution_diversity = _solution_diversity_score(puzzle, simplicity_contrast, spatial_contrast)
    solution_count_bucket = _solution_count_bucket(puzzle["num_solutions"])
    search_pressure_score = _search_pressure_score(puzzle)
    search_pressure_bucket = _score_bucket(search_pressure_score)
    human_difficulty_score, human_difficulty_drivers = _human_difficulty_score(puzzle)
    human_difficulty_bucket = _human_difficulty_bucket(ptype, human_difficulty_score)
    simplicity_contrast_bucket = _contrast_bucket(simplicity_contrast)
    spatial_contrast_bucket = _contrast_bucket(spatial_contrast)
    metadata = {
        "reasoning_family": REASONING_FAMILIES[ptype],
        "search_pressure_score": round(search_pressure_score, 3),
        "search_pressure_bucket": search_pressure_bucket,
        "human_difficulty_score": round(human_difficulty_score, 3),
        "human_difficulty_bucket": human_difficulty_bucket,
        "human_difficulty_drivers": human_difficulty_drivers,
        "solution_count_bucket": solution_count_bucket,
        "simplicity_contrast_score": round(simplicity_contrast, 3),
        "simplicity_contrast_bucket": simplicity_contrast_bucket,
        "spatial_contrast_score": round(spatial_contrast, 3),
        "spatial_contrast_bucket": spatial_contrast_bucket,
        "solution_diversity_score": round(solution_diversity, 3),
        "solution_diversity_bucket": _contrast_bucket(solution_diversity),
        "entropy_capacity_bits": round(math.log2(puzzle["num_solutions"]), 3),
        "visual_spatial_level": _visual_spatial_level(ptype),
        "format_burden_level": _format_burden_level(ptype),
        "probe_cell": f"{search_pressure_bucket}_{solution_count_bucket}",
        "probe_tags": _probe_tags(puzzle, search_pressure_bucket, simplicity_contrast_bucket, spatial_contrast_bucket),
    }
    puzzle["probe_metadata"] = metadata
    features.update(
        {
            "reasoning_family": metadata["reasoning_family"],
            "search_pressure_score": metadata["search_pressure_score"],
            "search_pressure_bucket": metadata["search_pressure_bucket"],
            "human_difficulty_score": metadata["human_difficulty_score"],
            "human_difficulty_bucket": metadata["human_difficulty_bucket"],
            "human_difficulty_drivers": metadata["human_difficulty_drivers"],
            "solution_count_bucket": metadata["solution_count_bucket"],
            "simplicity_contrast_score": metadata["simplicity_contrast_score"],
            "simplicity_contrast_bucket": metadata["simplicity_contrast_bucket"],
            "spatial_contrast_score": metadata["spatial_contrast_score"],
            "spatial_contrast_bucket": metadata["spatial_contrast_bucket"],
            "solution_diversity_score": metadata["solution_diversity_score"],
            "solution_diversity_bucket": metadata["solution_diversity_bucket"],
            "entropy_capacity_bits": metadata["entropy_capacity_bits"],
            "probe_cell": metadata["probe_cell"],
        }
    )
    puzzle["difficulty_bucket"] = metadata["human_difficulty_bucket"]
    puzzle["diversity_signature"].update(
        {
            "reasoning_family": metadata["reasoning_family"],
            "search_pressure_bucket": metadata["search_pressure_bucket"],
            "human_difficulty_bucket": metadata["human_difficulty_bucket"],
            "solution_count_bucket": metadata["solution_count_bucket"],
            "simplicity_contrast_bucket": metadata["simplicity_contrast_bucket"],
            "spatial_contrast_bucket": metadata["spatial_contrast_bucket"],
            "solution_diversity_bucket": metadata["solution_diversity_bucket"],
            "probe_cell": metadata["probe_cell"],
        }
    )


def _annotate_arithmetic(puzzle: Dict[str, Any]) -> None:
    for sol in puzzle["solutions"]:
        cost = float(sol["cost"])
        uses_division = bool(sol["uses_division"])
        has_fraction = bool(sol["has_fraction_intermediate"])
        description = len(sol["canonical_expression"]) / 28.0
        computational = cost + (0.8 if uses_division else 0.0) + (1.0 if has_fraction else 0.0)
        sol["probe_features"] = {
            "simplicity_score": round(computational + 0.25 * description, 3),
            "simplicity_components": {
                "computational": round(computational, 3),
                "description": round(description, 3),
                "action": 0.0,
                "perceptual": 0.0,
                "constraint": 0.0,
            },
            "spatial_features": {},
            "optimality_features": {},
            "simplicity_drivers": [
                driver
                for driver, active in (
                    ("low_expression_cost", cost <= 3.5),
                    ("uses_division", uses_division),
                    ("fraction_intermediate", has_fraction),
                )
                if active
            ],
        }


def _annotate_maze(puzzle: Dict[str, Any]) -> None:
    grid = puzzle["machine_readable_instance"]["grid"]
    shortest = puzzle["machine_readable_instance"].get("shortest_length")
    valid_edges = _valid_solution_edges(puzzle["solutions"])
    for sol in puzzle["solutions"]:
        moves = sol["moves"]
        length = int(sol.get("length", len(moves)))
        turns = _turn_count(moves)
        runs = turns + 1 if moves else 0
        over_shortest = length - shortest if shortest is not None else 0
        spatial = _path_spatial_features(sol["coordinates"], grid)
        human = _path_human_components(grid, sol["coordinates"], moves, shortest, valid_edges)
        action_complexity = (
            length
            + 2.6 * max(0, over_shortest)
            + 1.9 * turns
            + 1.8 * human["moves_away_from_goal"]
            + 1.4 * human["decision_points"]
            + 1.1 * human["tempting_branch_count"]
        )
        strategy_class = _maze_strategy_class(spatial, human, over_shortest)
        sol["probe_features"] = {
            "simplicity_score": round(action_complexity, 3),
            "simplicity_components": {
                "action": round(action_complexity, 3),
                "path_length": length,
                "length_over_shortest": over_shortest,
                "turn_count": turns,
                "direction_runs": runs,
                "moves_away_from_goal": human["moves_away_from_goal"],
                "decision_points": human["decision_points"],
                "side_branch_count": human["side_branch_count"],
                "tempting_branch_count": human["tempting_branch_count"],
                "valid_alternative_branch_count": human["valid_alternative_branch_count"],
                "detour_over_manhattan": human["detour_over_manhattan"],
                "description": len(moves),
                "perceptual": 0.0,
                "computational": 0.0,
                "constraint": 0.0,
            },
            "spatial_features": spatial,
            "optimality_features": {
                "is_shortest": over_shortest == 0,
                "length_over_shortest": over_shortest,
                "path_length": length,
                "turn_count": turns,
                "moves_away_from_goal": human["moves_away_from_goal"],
                "decision_points": human["decision_points"],
                "tempting_branch_count": human["tempting_branch_count"],
                "valid_alternative_branch_count": human["valid_alternative_branch_count"],
            },
            "solution_strategy_class": strategy_class,
            "simplicity_drivers": [
                driver
                for driver, active in (
                    ("shortest", over_shortest == 0),
                    ("few_turns", turns <= 3),
                    ("many_turns", turns >= 7),
                    ("few_decisions", human["decision_points"] <= 2),
                    ("many_decisions", human["decision_points"] >= 5),
                    ("moves_away_from_goal", human["moves_away_from_goal"] > 0),
                )
                if active
            ],
        }


def _annotate_grid_placement(puzzle: Dict[str, Any]) -> None:
    n = puzzle["machine_readable_instance"]["n"]
    row_counts = puzzle["features"].get("row_available_counts", [])
    col_counts = puzzle["features"].get("col_available_counts", [])
    forced_cells = _forced_cells_from_solutions(puzzle["solutions"])
    for sol in puzzle["solutions"]:
        coords = [tuple(c) for c in sol["coordinates"]]
        spatial = _placement_spatial_features(coords, n)
        human = _placement_human_components(coords, n, row_counts, col_counts, forced_cells, spatial)
        geometric_complexity = max(
            0.0,
            2.4 * human["monotonicity_penalty"]
            + 1.1 * human["jumpiness_penalty"]
            + 0.8 * human["gap_penalty"]
            + 1.6 * human["choice_cost"]
            + 0.6 * human["center_imbalance"]
            - 1.2 * human["forced_fraction"]
            - 1.0 * human["diagonal_alignment"],
        )
        strategy_class = _placement_strategy_class(spatial, human)
        sol["probe_features"] = {
            "simplicity_score": round(geometric_complexity, 3),
            "simplicity_components": {
                "geometric": round(geometric_complexity, 3),
                "monotonicity_penalty": round(human["monotonicity_penalty"], 3),
                "jumpiness_penalty": round(human["jumpiness_penalty"], 3),
                "gap_penalty": round(human["gap_penalty"], 3),
                "choice_cost": round(human["choice_cost"], 3),
                "forced_fraction": round(human["forced_fraction"], 3),
                "diagonal_alignment": round(human["diagonal_alignment"], 3),
                "center_imbalance": round(human["center_imbalance"], 3),
                "perceptual": round(geometric_complexity, 3),
                "constraint": round(human["choice_cost"] - human["forced_fraction"], 3),
                "action": 0.0,
                "description": len(coords),
                "computational": 0.0,
            },
            "spatial_features": spatial,
            "optimality_features": {},
            "solution_strategy_class": strategy_class,
            "simplicity_drivers": [
                driver
                for driver, active in (
                    ("diagonal_like", _is_diagonal_like(coords, n)),
                    ("monotonic", human["monotonicity_penalty"] <= 0.15),
                    ("forced_anchor", human["forced_fraction"] > 0),
                    ("low_choice_cells", human["choice_cost"] <= 0.35),
                    ("corner_using", spatial["corner_fraction"] > 0),
                    ("center_using", spatial["center_fraction"] > 0),
                    ("edge_heavy", spatial["edge_fraction"] >= 0.6),
                )
                if active
            ],
        }


def _annotate_minesweeper_lite(puzzle: Dict[str, Any]) -> None:
    rows = int(puzzle["features"].get("rows", 0))
    cols = int(puzzle["features"].get("cols", 0))
    for sol in puzzle["solutions"]:
        r, c = sol["coordinate"]
        local_clues = _minesweeper_local_clue_count(puzzle["machine_readable_instance"]["board"], (r, c))
        spatial = _single_cell_spatial_features((r, c), rows, cols)
        # A cell surrounded by explicit clues/flags is easier for humans to
        # verify than one whose status follows from a more global assignment.
        perceptual = max(0.0, 6.0 - local_clues) + 0.35 * spatial["center_distance"]
        strategy_class = f"{sol['cell_type']}|local={_feature_bucket(local_clues, [1, 3])}|region={spatial['region']}"
        sol["probe_features"] = {
            "simplicity_score": round(perceptual, 3),
            "simplicity_components": {
                "perceptual": round(perceptual, 3),
                "adjacent_clues": local_clues,
                "center_distance": round(spatial["center_distance"], 3),
                "constraint": round(max(0.0, 6.0 - local_clues), 3),
                "action": 1.0,
                "description": 1,
                "computational": 0.0,
            },
            "spatial_features": spatial,
            "optimality_features": {},
            "solution_strategy_class": strategy_class,
            "simplicity_drivers": [
                driver
                for driver, active in (
                    ("many_adjacent_clues", local_clues >= 3),
                    ("few_adjacent_clues", local_clues <= 1),
                    ("corner_cell", spatial["corner"]),
                    ("center_region", spatial["region"] == "middle_center"),
                    ("forced_mine", sol["cell_type"] == "mine"),
                    ("forced_safe", sol["cell_type"] == "safe"),
                )
                if active
            ],
        }


def _annotate_mini_sudoku(puzzle: Dict[str, Any]) -> None:
    inst = puzzle["machine_readable_instance"]
    board = inst["board"]
    n = int(inst["n"])
    br = int(inst["box_rows"])
    bc = int(inst["box_cols"])
    for sol in puzzle["solutions"]:
        r, c = sol["coordinate"]
        local_visibility = _sudoku_local_visibility(board, (r, c), br, bc)
        row_blanks = board[r - 1].count(".")
        col_blanks = sum(board[rr][c - 1] == "." for rr in range(n))
        box_r = ((r - 1) // br) * br
        box_c = ((c - 1) // bc) * bc
        box_blanks = sum(board[rr][cc] == "." for rr in range(box_r, box_r + br) for cc in range(box_c, box_c + bc))
        spatial = _single_cell_spatial_features((r, c), n, n)
        scanning = (row_blanks + col_blanks + box_blanks) / max(1, 3 * n)
        perceptual = max(0.0, 8.0 - local_visibility) + 2.0 * scanning + 0.25 * spatial["center_distance"]
        strategy_class = (
            f"digit={sol['digit']}|local={_feature_bucket(local_visibility, [3, 6])}|"
            f"scan={_feature_bucket(scanning, [0.35, 0.6])}|region={spatial['region']}"
        )
        sol["probe_features"] = {
            "simplicity_score": round(perceptual, 3),
            "simplicity_components": {
                "perceptual": round(perceptual, 3),
                "local_visibility": local_visibility,
                "row_blanks": row_blanks,
                "col_blanks": col_blanks,
                "box_blanks": box_blanks,
                "scanning_load": round(scanning, 3),
                "center_distance": round(spatial["center_distance"], 3),
                "constraint": round(2.0 * scanning, 3),
                "action": 1.0,
                "description": 1,
                "computational": 0.0,
            },
            "spatial_features": spatial,
            "optimality_features": {},
            "solution_strategy_class": strategy_class,
            "simplicity_drivers": [
                driver
                for driver, active in (
                    ("high_local_visibility", local_visibility >= 6),
                    ("low_local_visibility", local_visibility <= 2),
                    ("low_scanning_load", scanning <= 0.35),
                    ("high_scanning_load", scanning >= 0.6),
                    ("corner_cell", spatial["corner"]),
                    ("center_region", spatial["region"] == "middle_center"),
                )
                if active
            ],
        }


def _rank_solution_simplicity(puzzle: Dict[str, Any]) -> None:
    ordered = sorted(
        enumerate(puzzle["solutions"]),
        key=lambda item: (
            item[1]["probe_features"]["simplicity_score"],
            _solution_sort_key(puzzle["puzzle_type"], item[1]),
        ),
    )
    total = len(ordered)
    for rank, (_, sol) in enumerate(ordered, 1):
        score = float(sol["probe_features"]["simplicity_score"])
        sol["probe_features"].update(
            {
                "simplicity_rank_within_puzzle": rank,
                "simplicity_percentile_within_puzzle": round((rank - 1) / max(1, total - 1), 3),
                "simplicity_bucket": _rank_bucket(rank, total),
            }
        )


def _human_difficulty_score(puzzle: Dict[str, Any]) -> Tuple[float, List[str]]:
    ptype = puzzle["puzzle_type"]
    f = puzzle["features"]
    solution_scores = [float(s["probe_features"]["simplicity_score"]) for s in puzzle["solutions"]]
    easiest_solution = min(solution_scores) if solution_scores else 0.0
    drivers: List[str] = []
    if ptype == "arithmetic24":
        score = 8.0
        score += _scaled(f.get("min_expression_cost", 0.0), 2.5, 6.5, 45.0)
        score += 12.0 if f.get("fraction_required") else 0.0
        score += 8.0 if f.get("division_required") else 0.0
        score += 8.0 if not f.get("has_direct_factor_pair") else -4.0
        score -= 1.2 * max(0, puzzle["num_solutions"] - 2)
        drivers = [
            name
            for name, active in (
                ("fraction_required", f.get("fraction_required")),
                ("division_required", f.get("division_required")),
                ("no_direct_factor_pair", not f.get("has_direct_factor_pair")),
                ("few_solutions", puzzle["num_solutions"] <= 3),
            )
            if active
        ]
        return max(0.0, min(100.0, score)), drivers

    if ptype == "maze":
        opts = [s["probe_features"]["optimality_features"] for s in puzzle["solutions"]]
        min_turns = min(int(o.get("turn_count", 0)) for o in opts)
        min_decisions = min(int(o.get("decision_points", 0)) for o in opts)
        min_tempting = min(int(o.get("tempting_branch_count", 0)) for o in opts)
        min_away = min(int(o.get("moves_away_from_goal", 0)) for o in opts)
        min_valid_alts = min(int(o.get("valid_alternative_branch_count", 0)) for o in opts)
        detour = float(f.get("shortest_detour_over_manhattan", 0.0))
        direct_easy = detour == 0 and min_away == 0 and min_turns <= 2
        score = 8.0
        score += 0.85 * float(f.get("shortest_length", 0))
        score += 5.0 * detour
        score += 4.0 * min_turns
        score += 5.0 * min_decisions
        score += 3.5 * min_tempting
        score += 2.0 * min_away
        score += 2.0 * min(4, max(0, int(f.get("num_route_classes", 1)) - 2))
        score += 8.0 * min(1.0, float(f.get("wall_density", 0.0)) / 0.42)
        score -= 2.0 * max(0, puzzle["num_solutions"] - 2)
        score -= 1.2 * min(5, min_valid_alts)
        if direct_easy:
            score -= 16.0
        drivers = [
            name
            for name, active in (
                ("longer_path", f.get("shortest_length", 0) >= 14),
                ("detour_from_direct_route", detour >= 3),
                ("many_turns_even_in_easiest_path", min_turns >= 5),
                ("many_decision_points", min_decisions >= 4),
                ("tempting_wrong_branches", min_tempting >= 3),
                ("must_move_away_from_goal", min_away > 0),
                ("direct_monotone_solution", direct_easy),
            )
            if active
        ]
        return max(0.0, min(100.0, score)), drivers

    if ptype == "grid_placement":
        n = int(f.get("n", 0))
        m = int(f.get("m", 1))
        forced_fraction = float(f.get("num_forced_cells", 0)) / max(1, m)
        row_counts = f.get("row_available_counts", [])
        col_counts = f.get("col_available_counts", [])
        ambiguous_lines = sum(1 for x in row_counts + col_counts if 2 <= x <= max(2, n - 2))
        singleton_lines = sum(1 for x in row_counts + col_counts if x == 1)
        ambiguity = ambiguous_lines / max(1, 2 * n)
        singleton_help = singleton_lines / max(1, 2 * n)
        has_obvious_diagonal = any(
            s["probe_features"].get("simplicity_components", {}).get("diagonal_alignment", 0.0) >= 0.75
            for s in puzzle["solutions"]
        )
        score = 18.0
        score += 12.0 * easiest_solution
        score += 4.0 * max(0, n - 4)
        score += 14.0 * ambiguity
        score += 10.0 * (1.0 - forced_fraction)
        score -= 8.0 * singleton_help
        score -= 1.0 * max(0, puzzle["num_solutions"] - 2)
        score -= 5.0 if has_obvious_diagonal else 0.0
        drivers = [
            name
            for name, active in (
                ("large_board", n >= 6),
                ("few_forced_anchors", forced_fraction < 0.25 and not has_obvious_diagonal),
                ("ambiguous_rows_or_columns", ambiguity >= 0.35 and not has_obvious_diagonal),
                ("obvious_diagonal_solution", has_obvious_diagonal),
                ("no_obvious_diagonal_solution", not has_obvious_diagonal),
                ("few_solutions", puzzle["num_solutions"] <= 3),
            )
            if active
        ]
        return max(0.0, min(100.0, score)), drivers

    if ptype == "minesweeper_lite":
        max_local = float(f.get("max_solution_adjacent_clues", 0.0))
        assignments = float(f.get("num_consistent_assignments", 1.0))
        unknowns = float(f.get("num_unknown_cells", 0.0))
        score = 12.0
        score += 2.2 * unknowns
        score += 8.0 * math.log2(max(1.0, assignments))
        score -= 5.0 * max_local
        score -= 1.5 * max(0, puzzle["num_solutions"] - 2)
        drivers = [
            name
            for name, active in (
                ("many_unknowns", unknowns >= 12),
                ("many_consistent_assignments", assignments >= 16),
                ("locally_visible_forced_cell", max_local >= 3),
                ("few_solutions", puzzle["num_solutions"] <= 3),
            )
            if active
        ]
        return max(0.0, min(100.0, score)), drivers

    if ptype == "mini_sudoku":
        max_visibility = float(f.get("max_solution_local_visibility", 0.0))
        n = float(f.get("n", 4.0))
        empty = float(f.get("empty_count", 0.0))
        score = 10.0
        score += 4.0 * max(0.0, n - 4.0)
        score += 1.2 * empty
        score -= 4.0 * max_visibility
        score -= 1.2 * max(0, puzzle["num_solutions"] - 2)
        drivers = [
            name
            for name, active in (
                ("six_by_six", n >= 6),
                ("many_empty_cells", empty >= 22),
                ("highly_visible_solution", max_visibility >= 7),
                ("low_visibility_solutions", max_visibility <= 3),
                ("few_solutions", puzzle["num_solutions"] <= 3),
            )
            if active
        ]
        return max(0.0, min(100.0, score)), drivers

    return 50.0, []


def _human_difficulty_bucket(ptype: str, score: float) -> str:
    cuts = {
        "arithmetic24": (25.0, 40.0),
        "maze": (35.0, 60.0),
        "grid_placement": (35.0, 55.0),
        "minesweeper_lite": (30.0, 55.0),
        "mini_sudoku": (30.0, 55.0),
    }.get(ptype, (35.0, 62.0))
    if score < cuts[0]:
        return "easy"
    if score < cuts[1]:
        return "medium"
    return "hard"


def _path_human_components(
    grid: Sequence[str],
    coords1: Sequence[Sequence[int]],
    moves: str,
    shortest: int | None,
    valid_edges: set[Tuple[Coord, Coord]] | None = None,
) -> Dict[str, Any]:
    valid_edges = valid_edges or set()
    coords = [(int(r) - 1, int(c) - 1) for r, c in coords1]
    start, goal = coords[0], coords[-1]
    direct = abs(start[0] - goal[0]) + abs(start[1] - goal[1])
    moves_away = 0
    for a, b in zip(coords, coords[1:]):
        before = abs(a[0] - goal[0]) + abs(a[1] - goal[1])
        after = abs(b[0] - goal[0]) + abs(b[1] - goal[1])
        if after > before:
            moves_away += 1
    decision_points = 0
    side_branches = 0
    tempting = 0
    valid_alternatives = 0
    for i, cell in enumerate(coords[:-1]):
        prev_cell = coords[i - 1] if i > 0 else None
        next_cell = coords[i + 1]
        options = [nxt for nxt in _grid_neighbors(grid, cell) if nxt != prev_cell]
        side_options = [nxt for nxt in options if nxt != next_cell]
        wrong_side_options = [nxt for nxt in side_options if (cell, nxt) not in valid_edges]
        valid_side_options = [nxt for nxt in side_options if (cell, nxt) in valid_edges]
        if wrong_side_options:
            decision_points += 1
        side_branches += len(wrong_side_options)
        valid_alternatives += len(valid_side_options)
        current_goal_dist = abs(cell[0] - goal[0]) + abs(cell[1] - goal[1])
        for nxt in wrong_side_options:
            if abs(nxt[0] - goal[0]) + abs(nxt[1] - goal[1]) <= current_goal_dist:
                tempting += 1
    length = len(moves)
    return {
        "direct_manhattan_distance": direct,
        "detour_over_manhattan": max(0, length - direct),
        "length_over_shortest": 0 if shortest is None else length - shortest,
        "moves_away_from_goal": moves_away,
        "decision_points": decision_points,
        "side_branch_count": side_branches,
        "tempting_branch_count": tempting,
        "valid_alternative_branch_count": valid_alternatives,
    }


def _valid_solution_edges(solutions: Sequence[Dict[str, Any]]) -> set[Tuple[Coord, Coord]]:
    edges: set[Tuple[Coord, Coord]] = set()
    for sol in solutions:
        coords = [(int(r) - 1, int(c) - 1) for r, c in sol["coordinates"]]
        for a, b in zip(coords, coords[1:]):
            edges.add((a, b))
    return edges


def _grid_neighbors(grid: Sequence[str], cell: Coord) -> List[Coord]:
    rows, cols = len(grid), len(grid[0])
    r, c = cell
    out: List[Coord] = []
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        nr, nc = r + dr, c + dc
        if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] != "#":
            out.append((nr, nc))
    return out


def _maze_strategy_class(spatial: Dict[str, Any], human: Dict[str, Any], over_shortest: int) -> str:
    turns = _feature_bucket(float(human.get("decision_points", 0)), [2, 4])
    away = "away" if human.get("moves_away_from_goal", 0) else "direct"
    detour = "detour" if over_shortest > 0 else "shortest"
    return f"route={spatial.get('route_region')}|decisions={turns}|{away}|{detour}"


def _forced_cells_from_solutions(solutions: Sequence[Dict[str, Any]]) -> set[Coord]:
    if not solutions:
        return set()
    common = {tuple(coord) for coord in solutions[0]["coordinates"]}
    for sol in solutions[1:]:
        common &= {tuple(coord) for coord in sol["coordinates"]}
    return common


def _placement_human_components(
    coords: Sequence[Coord],
    n: int,
    row_counts: Sequence[int],
    col_counts: Sequence[int],
    forced_cells: set[Coord],
    spatial: Dict[str, Any],
) -> Dict[str, float]:
    ordered = sorted(coords)
    cols = [c for _, c in ordered]
    max_inv = max(1, len(cols) * (len(cols) - 1) // 2)
    inversions = sum(1 for i in range(len(cols)) for j in range(i + 1, len(cols)) if cols[i] > cols[j])
    monotonicity_penalty = min(inversions, max_inv - inversions) / max_inv
    if len(cols) > 1:
        jumpiness_penalty = sum(max(0, abs(a - b) - 1) for a, b in zip(cols, cols[1:])) / max(1, (n - 2) * (len(cols) - 1))
    else:
        jumpiness_penalty = 0.0
    rows = sorted(r for r, _ in coords)
    used_cols = sorted(c for _, c in coords)
    row_gap = (max(rows) - min(rows) + 1 - len(rows)) / max(1, n - len(rows) + 1)
    col_gap = (max(used_cols) - min(used_cols) + 1 - len(used_cols)) / max(1, n - len(used_cols) + 1)
    same = sum(1 for r, c in coords if r == c)
    anti = sum(1 for r, c in coords if r + c == n + 1)
    diagonal_alignment = max(same, anti) / len(coords)
    forced_fraction = sum(1 for coord in coords if coord in forced_cells) / len(coords)
    option_values = []
    for r, c in coords:
        rr = row_counts[r - 1] if r - 1 < len(row_counts) else n
        cc = col_counts[c - 1] if c - 1 < len(col_counts) else n
        option_values.append((rr + cc) / max(1, 2 * n))
    choice_cost = sum(option_values) / len(option_values)
    center_imbalance = (
        abs(float(spatial["mean_row"]) - (n + 1) / 2.0)
        + abs(float(spatial["mean_col"]) - (n + 1) / 2.0)
    ) / n
    return {
        "monotonicity_penalty": monotonicity_penalty,
        "jumpiness_penalty": jumpiness_penalty,
        "gap_penalty": (row_gap + col_gap) / 2.0,
        "choice_cost": choice_cost,
        "forced_fraction": forced_fraction,
        "diagonal_alignment": diagonal_alignment,
        "center_imbalance": center_imbalance,
    }


def _placement_strategy_class(spatial: Dict[str, Any], human: Dict[str, float]) -> str:
    monotonic = _feature_bucket(human["monotonicity_penalty"], [0.15, 0.45])
    choice = _feature_bucket(human["choice_cost"], [0.35, 0.55])
    forced = _feature_bucket(human["forced_fraction"], [0.0, 0.4])
    return f"shape={spatial.get('shape_class')}|mono={monotonic}|choice={choice}|forced={forced}"


def _feature_bucket(value: float, cuts: Sequence[float]) -> str:
    labels = ["low", "mid", "high"]
    for i, cut in enumerate(cuts):
        if value <= cut:
            return labels[i]
    return labels[-1]


def _spatial_contrast_score(solutions: Sequence[Dict[str, Any]]) -> float:
    vectors = [
        _spatial_vector(sol["probe_features"].get("spatial_features", {}))
        for sol in solutions
        if sol.get("probe_features", {}).get("spatial_features")
    ]
    if len(vectors) < 2:
        return 0.0
    best = 0.0
    for i in range(len(vectors)):
        for j in range(i + 1, len(vectors)):
            keys = set(vectors[i]) | set(vectors[j])
            dist = math.sqrt(sum((vectors[i].get(k, 0.0) - vectors[j].get(k, 0.0)) ** 2 for k in keys))
            best = max(best, dist)
    return best


def _solution_diversity_score(puzzle: Dict[str, Any], simplicity_contrast: float, spatial_contrast: float) -> float:
    features = puzzle["features"]
    structural = float(features.get("max_pairwise_jaccard_distance", 0.0))
    if puzzle["puzzle_type"] == "arithmetic24":
        structural = len(features.get("operation_pattern_counts", {})) / 4.0
    return 0.45 * structural + 0.35 * spatial_contrast + 0.20 * min(2.0, simplicity_contrast) / 2.0


def _spatial_vector(spatial: Dict[str, Any]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for key, value in spatial.items():
        if isinstance(value, (int, float)):
            scale = 0.25 if key in {"mean_row", "mean_col"} else 1.0
            out[key] = float(value) * scale
        elif isinstance(value, str):
            out[f"{key}={value}"] = 1.0
    return out


def _solution_sort_key(ptype: str, solution: Dict[str, Any]) -> str:
    if ptype == "arithmetic24":
        return solution["canonical_expression"]
    if ptype == "maze":
        return solution["moves"]
    if ptype in {"minesweeper_lite", "mini_sudoku"}:
        return str(solution["coordinate"])
    return str(solution["coordinates"])


def _search_pressure_score(puzzle: Dict[str, Any]) -> float:
    ptype = puzzle["puzzle_type"]
    f = puzzle["features"]
    nsol = puzzle["num_solutions"]
    few_solution_pressure = { "few": 18.0, "some": 9.0, "many": 2.0 }[_solution_count_bucket(nsol)]
    if ptype == "arithmetic24":
        score = 0.0
        score += _scaled(f["min_expression_cost"], 2.5, 6.5, 40)
        score += 18.0 if f.get("fraction_required") else 0.0
        score += 12.0 if f.get("division_required") else 0.0
        score += 10.0 if not f.get("has_direct_factor_pair") else 0.0
        score += few_solution_pressure
        return min(100.0, score)
    if ptype == "maze":
        score = 0.0
        score += _scaled(f["shortest_length"], 7, 22, 35)
        score += _scaled(f["num_branch_points_reachable_from_start"], 2, 18, 25)
        score += _scaled(f["num_dead_ends"], 0, 12, 15)
        score += _scaled(f["wall_density"], 0.15, 0.38, 10)
        score += few_solution_pressure
        return min(100.0, score)
    if ptype == "grid_placement":
        n, m = f["n"], f["m"]
        forced = f.get("num_forced_cells", 0)
        score = 0.0
        score += {4: 10.0, 5: 28.0, 6: 42.0}.get(n, 25.0)
        score += _scaled(f["num_forbidden_cells"] / (n * n), 0.18, 0.65, 20)
        score += _scaled(m, 3, 5, 12)
        score += _scaled(f.get("rows_with_only_1_available_cell", 0) + f.get("columns_with_only_1_available_cell", 0), 0, 6, 10)
        score += 8.0 if forced >= 2 else 0.0
        score += few_solution_pressure
        return min(100.0, score)
    if ptype == "minesweeper_lite":
        score = 0.0
        score += _scaled(f.get("num_unknown_cells", 0), 6, 16, 28)
        score += _scaled(math.log2(max(1, f.get("num_consistent_assignments", 1))), 0, 6, 28)
        score += 18.0 if f.get("max_solution_adjacent_clues", 0) <= 1 else 0.0
        score += few_solution_pressure
        return min(100.0, score)
    if ptype == "mini_sudoku":
        score = 0.0
        score += {4: 10.0, 6: 28.0}.get(f.get("n", 4), 20.0)
        score += _scaled(f.get("empty_count", 0), 8, 28, 28)
        score += 18.0 if f.get("max_solution_local_visibility", 0) <= 3 else 0.0
        score += few_solution_pressure
        return min(100.0, score)
    return 50.0


def _probe_tags(puzzle: Dict[str, Any], pressure: str, simplicity_contrast: str, spatial_contrast: str) -> List[str]:
    ptype = puzzle["puzzle_type"]
    tags = [REASONING_FAMILIES[ptype], f"pressure_{pressure}", f"solutions_{_solution_count_bucket(puzzle['num_solutions'])}"]
    if simplicity_contrast == "high":
        tags.append("simplicity_probe")
    if spatial_contrast == "high":
        tags.append("spatial_bias_probe")
    if ptype in {"maze", "grid_placement", "minesweeper_lite", "mini_sudoku"}:
        tags.append("visual_spatial")
    if ptype in {"minesweeper_lite", "mini_sudoku"}:
        tags.append("local_constraint_probe")
    return tags


def _path_spatial_features(coords: Sequence[Sequence[int]], grid: Sequence[str]) -> Dict[str, Any]:
    rows, cols = len(grid), len(grid[0])
    points = [(int(r), int(c)) for r, c in coords]
    mean_row = sum(r for r, _ in points) / len(points)
    mean_col = sum(c for _, c in points) / len(points)
    corners = {(1, 1), (1, cols), (rows, 1), (rows, cols)}
    edge_count = sum(1 for r, c in points if r in {1, rows} or c in {1, cols})
    center_count = sum(1 for r, c in points if rows * 0.35 <= r <= rows * 0.65 and cols * 0.35 <= c <= cols * 0.65)
    corner_count = sum(1 for p in points if p in corners)
    route_region = _route_region(points, rows, cols)
    spatial_score = (
        corner_count / len(points)
        + edge_count / len(points)
        + center_count / len(points)
        + (0.5 if "middle" in route_region or "center" in route_region else 0.0)
    )
    return {
        "route_region": route_region,
        "mean_row": round(mean_row, 3),
        "mean_col": round(mean_col, 3),
        "corner_cell_fraction": round(corner_count / len(points), 3),
        "edge_cell_fraction": round(edge_count / len(points), 3),
        "center_cell_fraction": round(center_count / len(points), 3),
        "spatial_score": round(spatial_score, 3),
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
    spatial_score = corner_count / len(coords) + edge_count / len(coords) + center_count / len(coords) + spread / max(1, n)
    return {
        "shape_class": shape,
        "mean_row": round(mean_row, 3),
        "mean_col": round(mean_col, 3),
        "corner_fraction": round(corner_count / len(coords), 3),
        "edge_fraction": round(edge_count / len(coords), 3),
        "center_fraction": round(center_count / len(coords), 3),
        "spread": round(spread, 3),
        "spatial_score": round(spatial_score, 3),
    }


def _route_region(coords: Sequence[Coord], rows: int, cols: int) -> str:
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


def _single_cell_spatial_features(coord: Coord, rows: int, cols: int) -> Dict[str, Any]:
    r, c = coord
    center_r = (rows + 1) / 2.0
    center_c = (cols + 1) / 2.0
    center_distance = abs(r - center_r) + abs(c - center_c)
    corner = (r, c) in {(1, 1), (1, cols), (rows, 1), (rows, cols)}
    edge = r in {1, rows} or c in {1, cols}
    region = _route_region([(r, c)], rows, cols)
    return {
        "row": r,
        "col": c,
        "region": region,
        "corner": corner,
        "edge": edge,
        "center_distance": round(center_distance, 3),
        "spatial_score": round((1.0 if corner else 0.0) + (0.5 if edge else 0.0) + center_distance / max(rows, cols), 3),
    }


def _minesweeper_local_clue_count(board: Sequence[str], coord: Coord) -> int:
    rows, cols = len(board), len(board[0])
    r, c = coord[0] - 1, coord[1] - 1
    count = 0
    for rr in range(max(0, r - 1), min(rows, r + 2)):
        for cc in range(max(0, c - 1), min(cols, c + 2)):
            if (rr, cc) != (r, c) and (board[rr][cc].isdigit() or board[rr][cc] == "F"):
                count += 1
    return count


def _sudoku_local_visibility(board: Sequence[str], coord: Coord, br: int, bc: int) -> int:
    n = len(board)
    r, c = coord[0] - 1, coord[1] - 1
    cells = {(r, cc) for cc in range(n)} | {(rr, c) for rr in range(n)}
    r0, c0 = (r // br) * br, (c // bc) * bc
    cells |= {(rr, cc) for rr in range(r0, r0 + br) for cc in range(c0, c0 + bc)}
    return sum(board[rr][cc] != "." for rr, cc in cells)


def _placement_shape_class(coords: Sequence[Coord], n: int, spread: float) -> str:
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


def _is_diagonal_like(coords: Sequence[Coord], n: int) -> bool:
    same = sum(1 for r, c in coords if r == c)
    anti = sum(1 for r, c in coords if r + c == n + 1)
    return same >= len(coords) - 1 or anti >= len(coords) - 1


def _turn_count(moves: str) -> int:
    return sum(1 for a, b in zip(moves, moves[1:]) if a != b)


def _scaled(value: float, low: float, high: float, weight: float) -> float:
    if high <= low:
        return 0.0
    return max(0.0, min(1.0, (value - low) / (high - low))) * weight


def _score_bucket(score: float) -> str:
    if score < 33.4:
        return "low"
    if score < 66.7:
        return "medium"
    return "high"


def _solution_count_bucket(count: int) -> str:
    if count <= 3:
        return "few"
    if count <= 6:
        return "some"
    return "many"


def _contrast_bucket(value: float) -> str:
    if value < 0.35:
        return "low"
    if value < 0.9:
        return "medium"
    return "high"


def _rank_bucket(rank: int, total: int) -> str:
    if rank == 1:
        return "simplest"
    if rank <= max(2, math.ceil(total / 3)):
        return "simple"
    if rank >= max(1, total - math.floor(total / 3) + 1):
        return "complex"
    return "middle"


def _visual_spatial_level(ptype: str) -> str:
    return {
        "arithmetic24": "none",
        "maze": "high",
        "grid_placement": "high",
        "minesweeper_lite": "high",
        "mini_sudoku": "high",
    }[ptype]


def _format_burden_level(ptype: str) -> str:
    return {
        "arithmetic24": "high",
        "maze": "low",
        "grid_placement": "medium",
        "minesweeper_lite": "low",
        "mini_sudoku": "low",
    }[ptype]
