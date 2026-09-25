from __future__ import annotations

import random
from collections import Counter
from typing import Any, Dict, List, Sequence, Tuple

from .common import MAX_MULTI_SOLUTIONS, MIN_MULTI_SOLUTIONS, Puzzle, solution_count_in_range, stable_hash
from .probes import annotate_puzzle


Coord = Tuple[int, int]


def box_shape(n: int) -> Tuple[int, int]:
    if n == 4:
        return 2, 2
    if n == 6:
        return 2, 3
    raise ValueError("Mini Sudoku supports n=4 or n=6")


def enumerate_placements(board: Sequence[str], digit: int) -> List[Dict[str, Any]]:
    n = len(board)
    br, bc = box_shape(n)
    token = str(digit)
    solutions: List[Dict[str, Any]] = []
    for r in range(n):
        for c in range(n):
            if board[r][c] != ".":
                continue
            if token in board[r]:
                continue
            if any(board[rr][c] == token for rr in range(n)):
                continue
            box_r = (r // br) * br
            box_c = (c // bc) * bc
            if any(board[rr][cc] == token for rr in range(box_r, box_r + br) for cc in range(box_c, box_c + bc)):
                continue
            solutions.append(_solution_record(board, digit, (r, c)))
    solutions.sort(key=lambda sol: sol["coordinate"])
    return solutions


def validate_placement(board: Sequence[str], digit: int, coordinate: Sequence[int]) -> bool:
    if len(coordinate) != 2:
        return False
    n = len(board)
    r, c = int(coordinate[0]) - 1, int(coordinate[1]) - 1
    return any(sol["coordinate"] == [r + 1, c + 1] for sol in enumerate_placements(board, digit))


def _solution_record(board: Sequence[str], digit: int, coord0: Coord) -> Dict[str, Any]:
    r, c = coord0
    rows = []
    for rr, row in enumerate(board):
        chars = []
        for cc, ch in enumerate(row):
            chars.append(str(digit) if (rr, cc) == coord0 else ch)
        rows.append(" ".join(chars))
    return {
        "coordinate": [r + 1, c + 1],
        "digit": digit,
        "board": "\n".join(rows),
    }


def make_puzzle(
    board: Sequence[str],
    digit: int,
    min_solutions: int = MIN_MULTI_SOLUTIONS,
    max_solutions: int = MAX_MULTI_SOLUTIONS,
) -> Puzzle | None:
    solutions = enumerate_placements(board, digit)
    if not solution_count_in_range(len(solutions), min_solutions, max_solutions):
        return None
    features, signature, bucket = compute_features(board, digit, solutions)
    n = len(board)
    br, bc = box_shape(n)
    rendered = "\n".join(" ".join(row) for row in board)
    prompt = (
        "Question {{ID}}:\n"
        f"Place the digit {digit} in one legal empty cell of this {n} by {n} mini Sudoku.\n\n"
        f"Rules: choose an empty cell where placing {digit} would not repeat {digit} in that row, column, or outlined box.\n\n"
        f"Board:\n{rendered}\n\nWrite one coordinate as row,column."
    )
    return annotate_puzzle(
        {
            "id": "",
            "puzzle_type": "mini_sudoku",
            "prompt_text": prompt,
            "rendered_puzzle": rendered,
            "machine_readable_instance": {
                "n": n,
                "box_rows": br,
                "box_cols": bc,
                "digit": digit,
                "board": list(board),
            },
            "num_solutions": len(solutions),
            "solutions": solutions,
            "features": features,
            "difficulty_bucket": bucket,
            "diversity_signature": signature,
        }
    )


def compute_features(
    board: Sequence[str], digit: int, solutions: Sequence[Dict[str, Any]]
) -> Tuple[Dict[str, Any], Dict[str, Any], str]:
    n = len(board)
    br, bc = box_shape(n)
    clue_count = sum(ch != "." for row in board for ch in row)
    empty_count = n * n - clue_count
    solution_coords = [tuple(sol["coordinate"]) for sol in solutions]
    local_scores = [_local_visibility(board, coord, br, bc) for coord in solution_coords]
    row_blanks = [row.count(".") for row in board]
    col_blanks = [sum(board[r][c] == "." for r in range(n)) for c in range(n)]
    box_blanks = []
    for r0 in range(0, n, br):
        for c0 in range(0, n, bc):
            box_blanks.append(sum(board[r][c] == "." for r in range(r0, r0 + br) for c in range(c0, c0 + bc)))
    center = (n + 1) / 2.0
    center_distances = [abs(r - center) + abs(c - center) for r, c in solution_coords]
    features = {
        "n": n,
        "box_rows": br,
        "box_cols": bc,
        "target_digit": digit,
        "num_clues": clue_count,
        "empty_count": empty_count,
        "clue_density": round(clue_count / (n * n), 3),
        "num_solutions": len(solutions),
        "row_blank_counts": row_blanks,
        "col_blank_counts": col_blanks,
        "box_blank_counts": box_blanks,
        "min_solution_local_visibility": min(local_scores),
        "max_solution_local_visibility": max(local_scores),
        "solution_visibility_contrast": max(local_scores) - min(local_scores),
        "min_solution_center_distance": round(min(center_distances), 3),
        "max_solution_center_distance": round(max(center_distances), 3),
        "solution_center_distance_contrast": round(max(center_distances) - min(center_distances), 3),
        "canonical_hash": canonical_board_hash(board, digit),
    }
    signature = {
        "n": n,
        "digit": digit,
        "clue_density_bucket": _bucket(features["clue_density"], [0.28, 0.42, 0.58]),
        "num_solutions_bucket": _bucket(len(solutions), [3, 6, 9]),
        "visibility_contrast_bucket": _bucket(features["solution_visibility_contrast"], [2, 5, 8]),
        "center_contrast_bucket": _bucket(features["solution_center_distance_contrast"], [1, 2.5, 4]),
        "row_blank_pattern": sorted(row_blanks),
        "col_blank_pattern": sorted(col_blanks),
    }
    return features, signature, assign_difficulty(features)


def _local_visibility(board: Sequence[str], coord1: Coord, br: int, bc: int) -> int:
    n = len(board)
    r, c = coord1[0] - 1, coord1[1] - 1
    cells = {(r, cc) for cc in range(n)} | {(rr, c) for rr in range(n)}
    r0, c0 = (r // br) * br, (c // bc) * bc
    cells |= {(rr, cc) for rr in range(r0, r0 + br) for cc in range(c0, c0 + bc)}
    return sum(board[rr][cc] != "." for rr, cc in cells)


def assign_difficulty(features: Dict[str, Any]) -> str:
    easiest_visibility = features["max_solution_local_visibility"]
    solutions = features["num_solutions"]
    if easiest_visibility >= 7 and solutions >= 4:
        return "easy"
    if features["n"] == 6 and easiest_visibility <= 3:
        return "hard"
    if features["solution_visibility_contrast"] >= 6:
        return "medium"
    return "medium"


def canonical_board_hash(board: Sequence[str], digit: int) -> str:
    return stable_hash({"mini_sudoku": {"board": list(board), "digit": digit}})


def generate_pool(seed: int, valid_target: int = 500, min_attempts: int = 5000) -> List[Puzzle]:
    rng = random.Random(seed)
    pool: List[Puzzle] = []
    seen: set[str] = set()
    attempts = 0
    while attempts < min_attempts or len(pool) < valid_target:
        attempts += 1
        n = 6 if rng.random() < 0.72 else 4
        full = _random_full_board(n, rng)
        keep_prob = rng.uniform(0.26, 0.48) if n == 6 else rng.uniform(0.32, 0.55)
        board = []
        for row in full:
            board.append("".join(ch if rng.random() < keep_prob else "." for ch in row))
        digit = rng.randint(1, n)
        puzzle = make_puzzle(board, digit)
        if puzzle is None:
            continue
        h = puzzle["features"]["canonical_hash"]
        if h in seen:
            continue
        seen.add(h)
        pool.append(puzzle)
        if len(pool) >= valid_target and attempts >= min_attempts:
            break
        if attempts > min_attempts * 8 and len(pool) >= valid_target:
            break
    return pool


def _random_full_board(n: int, rng: random.Random) -> List[str]:
    br, bc = box_shape(n)
    base = [[((r * bc + r // br + c) % n) + 1 for c in range(n)] for r in range(n)]
    row_order: List[int] = []
    for band in rng.sample(range(n // br), n // br):
        rows = list(range(band * br, band * br + br))
        rng.shuffle(rows)
        row_order.extend(rows)
    col_order: List[int] = []
    for stack in rng.sample(range(n // bc), n // bc):
        cols = list(range(stack * bc, stack * bc + bc))
        rng.shuffle(cols)
        col_order.extend(cols)
    digits = list(range(1, n + 1))
    rng.shuffle(digits)
    mapping = {i + 1: digits[i] for i in range(n)}
    return ["".join(str(mapping[base[r][c]]) for c in col_order) for r in row_order]




def _bucket(value: float, cuts: Sequence[float]) -> str:
    labels = ["low", "mid", "high", "very_high"]
    for i, cut in enumerate(cuts):
        if value <= cut:
            return labels[i]
    return labels[len(cuts)]
