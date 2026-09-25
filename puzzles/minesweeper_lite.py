from __future__ import annotations

import itertools
import random
from typing import Any, Dict, List, Sequence, Tuple

from .common import MAX_MULTI_SOLUTIONS, MIN_MULTI_SOLUTIONS, Puzzle, solution_count_in_range, stable_hash
from .probes import annotate_puzzle


Coord = Tuple[int, int]


def neighbors(rows: int, cols: int, cell: Coord) -> List[Coord]:
    r, c = cell
    out = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                out.append((nr, nc))
    return out


def enumerate_consistent_assignments(board: Sequence[str]) -> List[set[Coord]]:
    rows, cols = len(board), len(board[0])
    unknowns = [(r, c) for r in range(rows) for c in range(cols) if board[r][c] == "?"]
    flags = {(r, c) for r in range(rows) for c in range(cols) if board[r][c] == "F"}
    clues = [
        ((r, c), int(board[r][c]))
        for r in range(rows)
        for c in range(cols)
        if board[r][c].isdigit()
    ]
    if len(unknowns) > 18:
        return []
    assignments: List[set[Coord]] = []
    for bits in itertools.product((0, 1), repeat=len(unknowns)):
        mines = set(flags)
        mines.update(cell for bit, cell in zip(bits, unknowns) if bit)
        ok = True
        for cell, clue in clues:
            if sum(n in mines for n in neighbors(rows, cols, cell)) != clue:
                ok = False
                break
        if ok:
            assignments.append({cell for bit, cell in zip(bits, unknowns) if bit})
    return assignments


def forced_cells(board: Sequence[str], target_kind: str) -> Tuple[List[Dict[str, Any]], int]:
    rows, cols = len(board), len(board[0])
    unknowns = [(r, c) for r in range(rows) for c in range(cols) if board[r][c] == "?"]
    assignments = enumerate_consistent_assignments(board)
    if not assignments:
        return [], 0
    solutions = []
    for cell in unknowns:
        mine_count = sum(cell in assignment for assignment in assignments)
        if target_kind == "mine" and mine_count == len(assignments):
            solutions.append(_solution_record(board, cell, target_kind))
        elif target_kind == "safe" and mine_count == 0:
            solutions.append(_solution_record(board, cell, target_kind))
    solutions.sort(key=lambda sol: sol["coordinate"])
    return solutions, len(assignments)


def validate_cell(board: Sequence[str], target_kind: str, coordinate: Sequence[int]) -> bool:
    if len(coordinate) != 2:
        return False
    solutions, _ = forced_cells(board, target_kind)
    return [int(coordinate[0]), int(coordinate[1])] in [sol["coordinate"] for sol in solutions]


def _solution_record(board: Sequence[str], coord0: Coord, target_kind: str) -> Dict[str, Any]:
    r, c = coord0
    rows = []
    marker = "M" if target_kind == "mine" else "S"
    for rr, row in enumerate(board):
        chars = []
        for cc, ch in enumerate(row):
            chars.append(marker if (rr, cc) == coord0 else ch)
        rows.append(" ".join(chars))
    return {
        "coordinate": [r + 1, c + 1],
        "cell_type": target_kind,
        "board": "\n".join(rows),
    }


def make_puzzle(
    board: Sequence[str],
    target_kind: str,
    min_solutions: int = MIN_MULTI_SOLUTIONS,
    max_solutions: int = MAX_MULTI_SOLUTIONS,
) -> Puzzle | None:
    solutions, num_assignments = forced_cells(board, target_kind)
    if not solution_count_in_range(len(solutions), min_solutions, max_solutions):
        return None
    features, signature, bucket = compute_features(board, target_kind, solutions, num_assignments)
    rendered = "\n".join(" ".join(row) for row in board)
    task = "definitely safe" if target_kind == "safe" else "definitely a mine"
    prompt = (
        "Question {{ID}}:\n"
        f"Choose one hidden cell marked ? that is {task}.\n\n"
        "Rules: each ? is hidden and is either safe or a mine. Each number is already revealed and shows exactly how many mines are in all cells touching it, including diagonals. F is a known mine and counts toward nearby numbers. Choose a cell that has the required status in every mine layout that fits all numbers; do not guess.\n\n"
        f"Board:\n{rendered}\n\nWrite one coordinate as row,column."
    )
    return annotate_puzzle(
        {
            "id": "",
            "puzzle_type": "minesweeper_lite",
            "prompt_text": prompt,
            "rendered_puzzle": rendered,
            "machine_readable_instance": {
                "board": list(board),
                "target_kind": target_kind,
            },
            "num_solutions": len(solutions),
            "solutions": solutions,
            "features": features,
            "difficulty_bucket": bucket,
            "diversity_signature": signature,
        }
    )


def compute_features(
    board: Sequence[str],
    target_kind: str,
    solutions: Sequence[Dict[str, Any]],
    num_assignments: int,
) -> Tuple[Dict[str, Any], Dict[str, Any], str]:
    rows, cols = len(board), len(board[0])
    unknowns = [(r, c) for r in range(rows) for c in range(cols) if board[r][c] == "?"]
    clues = [(r, c) for r in range(rows) for c in range(cols) if board[r][c].isdigit()]
    flags = [(r, c) for r in range(rows) for c in range(cols) if board[r][c] == "F"]
    local_clues = [_local_clue_count(board, tuple(sol["coordinate"])) for sol in solutions]
    center = ((rows + 1) / 2.0, (cols + 1) / 2.0)
    center_distances = [abs(r - center[0]) + abs(c - center[1]) for r, c in (sol["coordinate"] for sol in solutions)]
    features = {
        "rows": rows,
        "cols": cols,
        "target_kind": target_kind,
        "num_unknown_cells": len(unknowns),
        "num_clue_cells": len(clues),
        "num_flagged_mines": len(flags),
        "num_consistent_assignments": num_assignments,
        "num_solutions": len(solutions),
        "min_solution_adjacent_clues": min(local_clues),
        "max_solution_adjacent_clues": max(local_clues),
        "solution_local_clue_contrast": max(local_clues) - min(local_clues),
        "min_solution_center_distance": round(min(center_distances), 3),
        "max_solution_center_distance": round(max(center_distances), 3),
        "solution_center_distance_contrast": round(max(center_distances) - min(center_distances), 3),
        "canonical_hash": canonical_board_hash(board, target_kind),
    }
    signature = {
        "size": f"{rows}x{cols}",
        "target_kind": target_kind,
        "unknown_bucket": _bucket(len(unknowns), [7, 11, 15]),
        "assignment_bucket": _bucket(num_assignments, [1, 4, 16, 64]),
        "num_solutions_bucket": _bucket(len(solutions), [3, 6, 9]),
        "local_clue_contrast_bucket": _bucket(features["solution_local_clue_contrast"], [1, 3, 5]),
    }
    return features, signature, assign_difficulty(features)


def _local_clue_count(board: Sequence[str], coord1: Coord) -> int:
    rows, cols = len(board), len(board[0])
    r, c = coord1[0] - 1, coord1[1] - 1
    return sum(board[rr][cc].isdigit() or board[rr][cc] == "F" for rr, cc in neighbors(rows, cols, (r, c)))


def assign_difficulty(features: Dict[str, Any]) -> str:
    if features["max_solution_adjacent_clues"] >= 3 and features["num_consistent_assignments"] <= 4:
        return "easy"
    if features["num_consistent_assignments"] >= 32 or features["max_solution_adjacent_clues"] <= 1:
        return "hard"
    return "medium"


def canonical_board_hash(board: Sequence[str], target_kind: str) -> str:
    return stable_hash({"minesweeper_lite": {"board": list(board), "target_kind": target_kind}})


def generate_pool(seed: int, valid_target: int = 500, min_attempts: int = 5000) -> List[Puzzle]:
    rng = random.Random(seed)
    pool: List[Puzzle] = []
    seen: set[str] = set()
    attempts = 0
    while attempts < min_attempts or len(pool) < valid_target:
        attempts += 1
        rows = cols = 5 if rng.random() < 0.65 else 4
        mine_count = rng.randint(3, 7 if rows == 5 else 4)
        hidden_mines = set(rng.sample([(r, c) for r in range(rows) for c in range(cols)], mine_count))
        board = _visible_board_from_hidden(rows, cols, hidden_mines, rng)
        if sum(ch == "?" for row in board for ch in row) > 16:
            continue
        target_kind = "safe" if rng.random() < 0.55 else "mine"
        puzzle = make_puzzle(board, target_kind)
        if puzzle is None:
            other = "mine" if target_kind == "safe" else "safe"
            puzzle = make_puzzle(board, other)
        if puzzle is None:
            continue
        h = puzzle["features"]["canonical_hash"]
        if h in seen:
            continue
        seen.add(h)
        pool.append(puzzle)
        if len(pool) >= valid_target and attempts >= min_attempts:
            break
        if attempts > min_attempts * 10 and len(pool) >= valid_target:
            break
    return pool


def _visible_board_from_hidden(rows: int, cols: int, mines: set[Coord], rng: random.Random) -> List[str]:
    reveal_prob = rng.uniform(0.45, 0.72)
    flag_prob = rng.uniform(0.0, 0.25)
    board = []
    for r in range(rows):
        chars = []
        for c in range(cols):
            if (r, c) in mines:
                chars.append("F" if rng.random() < flag_prob else "?")
            elif rng.random() < reveal_prob:
                chars.append(str(sum(n in mines for n in neighbors(rows, cols, (r, c)))))
            else:
                chars.append("?")
        board.append("".join(chars))
    return board




def _bucket(value: float, cuts: Sequence[float]) -> str:
    labels = ["low", "mid", "high", "very_high", "extreme"]
    for i, cut in enumerate(cuts):
        if value <= cut:
            return labels[i]
    return labels[len(cuts)]
