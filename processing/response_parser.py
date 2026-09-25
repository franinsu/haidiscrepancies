from __future__ import annotations

import ast
import re
from fractions import Fraction
from typing import Any, Dict, List, Sequence, Tuple

from puzzles import arithmetic24, grid_placement, maze, minesweeper_lite, mini_sudoku


PARSER_VERSION = "response_parser_v5"


def score_raw_answer(puzzle: Dict[str, Any], raw_answer: str, raw_response: str | None = None) -> Dict[str, Any]:
    """Parse, validate, and map a raw answer to a dataset solution_id."""
    raw_response = (raw_response if raw_response is not None else raw_answer or "").strip()
    raw_answer = (raw_answer or raw_response or "").strip()
    if not raw_answer:
        return _invalid("empty", parse_notes="No answer text found.")
    raw_answer = _extract_final_answer_line(raw_answer)
    raw_answer, ignored_terminal_period = _strip_sentence_terminal_period(raw_answer)
    ptype = puzzle["puzzle_type"]
    if ptype == "arithmetic24":
        result = _score_arithmetic(puzzle, raw_answer)
    elif ptype == "maze":
        result = _score_maze(puzzle, raw_answer)
    elif ptype == "grid_placement":
        result = _score_grid_placement(puzzle, raw_answer)
    elif ptype == "minesweeper_lite":
        result = _score_single_coordinate(puzzle, raw_answer, minesweeper_lite.validate_cell)
    elif ptype == "mini_sudoku":
        result = _score_single_coordinate(puzzle, raw_answer, mini_sudoku.validate_placement)
    else:
        result = _invalid("unknown_puzzle_type")
    if ignored_terminal_period:
        note = "Ignored one sentence-terminal period."
        result["parse_notes"] = " ".join(part for part in (note, result["parse_notes"]) if part)
    return result


def _extract_final_answer_line(raw_answer: str) -> str:
    """Prefer the final explicit ANSWER line when a model includes scratch text."""
    matches = list(re.finditer(r"(?im)^\s*ANSWER\s*:\s*(.+?)\s*$", raw_answer))
    if matches:
        return matches[-1].group(1).strip()
    return raw_answer


def _strip_sentence_terminal_period(raw_answer: str) -> Tuple[str, bool]:
    """Ignore one prose-style final period without accepting ellipses."""
    text = raw_answer.rstrip()
    if len(text) > 1 and text.endswith(".") and not text.endswith(".."):
        return text[:-1].rstrip(), True
    return text, False


def _base_parse_fields(
    extracted_answer: Any = None,
    parse_confidence: str = "high",
    requires_manual_review: bool = False,
    parse_notes: str = "",
) -> Dict[str, Any]:
    return {
        "parser_version": PARSER_VERSION,
        "extracted_answer": extracted_answer,
        "parse_confidence": parse_confidence,
        "requires_manual_review": requires_manual_review,
        "parse_notes": parse_notes,
    }


def _valid(
    solution_id: str,
    parsed_answer: Any,
    canonical_answer: Any,
    extracted_answer: Any = None,
    parse_confidence: str = "high",
    requires_manual_review: bool = False,
    parse_notes: str = "",
) -> Dict[str, Any]:
    return {
        "is_valid": True,
        "invalid_reason": None,
        "solution_id": solution_id,
        "parsed_answer": parsed_answer,
        "canonical_answer": canonical_answer,
        **_base_parse_fields(
            extracted_answer if extracted_answer is not None else parsed_answer,
            parse_confidence,
            requires_manual_review,
            parse_notes,
        ),
    }


def _invalid(
    reason: str,
    parsed_answer: Any = None,
    extracted_answer: Any = None,
    parse_confidence: str = "low",
    requires_manual_review: bool = False,
    parse_notes: str = "",
) -> Dict[str, Any]:
    return {
        "is_valid": False,
        "invalid_reason": reason,
        "solution_id": None,
        "parsed_answer": parsed_answer,
        "canonical_answer": None,
        **_base_parse_fields(
            extracted_answer if extracted_answer is not None else parsed_answer,
            parse_confidence,
            requires_manual_review,
            parse_notes,
        ),
    }


def _solution_id(puzzle: Dict[str, Any], index0: int) -> str:
    return puzzle["solutions"][index0].get("solution_id", f"{puzzle['id']}_sol_{index0 + 1:02d}")


def _score_arithmetic(puzzle: Dict[str, Any], raw_answer: str) -> Dict[str, Any]:
    expression, confidence, manual, notes = _extract_expression(raw_answer)
    if not expression:
        return _invalid("format_error", parse_notes=notes)
    numbers = puzzle["machine_readable_instance"]["numbers"]
    target = puzzle["machine_readable_instance"].get("target", 24)
    if not arithmetic24.validate_expression(expression, numbers, target):
        return _invalid(
            "wrong_arithmetic",
            expression,
            extracted_answer=expression,
            parse_confidence=confidence,
            requires_manual_review=manual,
            parse_notes=notes,
        )
    try:
        canonical = canonicalize_arithmetic_expression(expression)
    except Exception as exc:
        return _invalid("parser_error", expression, expression, "low", True, str(exc))
    for i, solution in enumerate(puzzle["solutions"]):
        if canonical == solution["canonical_expression"]:
            return _valid(_solution_id(puzzle, i), expression, canonical, expression, confidence, manual, notes)
    return _invalid("valid_but_unmatched_solution", expression, expression, confidence, True, notes)


def _extract_expression(raw_answer: str) -> Tuple[str | None, str, bool, str]:
    text = raw_answer.strip()
    code = re.findall(r"`([^`]+)`", text)
    if code:
        text = code[0]
    if "=" in text:
        left, right = text.split("=", 1)
        if "24" in right:
            text = left
    candidates = [m.group(0).strip() for m in re.finditer(r"[-+*/().\d\s]+", text)]
    candidates = [candidate for candidate in candidates if any(ch.isdigit() for ch in candidate)]
    if not candidates:
        return None, "low", False, "No arithmetic-looking expression found."
    expression = max(candidates, key=len)
    multiple = len(candidates) > 1
    confidence = "medium" if multiple or code else "high"
    notes = "Multiple expression-like spans found; selected the longest." if multiple else ""
    return expression, confidence, multiple, notes


def canonicalize_arithmetic_expression(expression: str) -> str:
    tree = ast.parse(expression, mode="eval")
    value, key = _canonicalize_ast(tree.body)
    if value != Fraction(24, 1):
        raise ValueError("Expression does not evaluate to 24")
    return arithmetic24._key_to_string(key)


def _canonicalize_ast(node: ast.AST) -> Tuple[Fraction, Tuple[Any, ...]]:
    if isinstance(node, ast.Constant) and isinstance(node.value, int):
        value = Fraction(int(node.value), 1)
        return value, ("n", int(node.value))
    if isinstance(node, ast.BinOp):
        left_value, left_key = _canonicalize_ast(node.left)
        right_value, right_key = _canonicalize_ast(node.right)
        if isinstance(node.op, ast.Add):
            return left_value + right_value, arithmetic24._canonical_key("+", left_key, right_key)
        if isinstance(node.op, ast.Sub):
            return left_value - right_value, arithmetic24._canonical_key("-", left_key, right_key)
        if isinstance(node.op, ast.Mult):
            return left_value * right_value, arithmetic24._canonical_key("*", left_key, right_key)
        if isinstance(node.op, ast.Div):
            if right_value == 0:
                raise ZeroDivisionError
            return left_value / right_value, arithmetic24._canonical_key("/", left_key, right_key)
    raise ValueError(f"Unsupported expression: {ast.dump(node)}")


def _score_maze(puzzle: Dict[str, Any], raw_answer: str) -> Dict[str, Any]:
    moves, confidence, manual, notes = _extract_moves(raw_answer)
    if not moves:
        return _invalid("format_error", parse_notes=notes)
    grid = puzzle["machine_readable_instance"]["grid"]
    if not _is_path_syntactically_valid(moves):
        return _invalid("format_error", moves, moves, confidence, manual, notes)
    if not maze.validate_path(grid, moves):
        return _invalid(_maze_invalid_reason(grid, moves, must_be_shortest=True), moves, moves, confidence, manual, notes)
    for i, solution in enumerate(puzzle["solutions"]):
        if moves == solution["moves"]:
            return _valid(_solution_id(puzzle, i), moves, moves, moves, confidence, manual, notes)
    return _invalid("valid_but_unmatched_solution", moves, moves, confidence, True, notes)


def _extract_moves(raw_answer: str) -> Tuple[str | None, str, bool, str]:
    text = raw_answer.strip().upper()
    code = re.findall(r"`([^`]+)`", text)
    if code:
        text = code[0].upper()
    explicit = re.search(r"\b(?:PATH|MOVES?)\s*[:=]\s*([UDLR][UDLR\s,;:.-]*)", text)
    if explicit:
        return "".join(re.findall(r"[UDLR]", explicit.group(1))), "high", False, ""
    if re.fullmatch(r"[UDLR\s,;:.-]+", text) and re.search(r"[UDLR]", text):
        return "".join(re.findall(r"[UDLR]", text)), "high", False, ""
    contiguous = re.findall(r"[UDLR]+", text)
    if contiguous:
        moves = max(contiguous, key=len)
        multiple = len(contiguous) > 1
        confidence = "medium" if multiple else "high"
        notes = "Multiple move-like spans found; selected the longest." if multiple else ""
        return moves, confidence, multiple, notes
    tokens = re.findall(r"\b[UDLR]\b", text)
    if tokens:
        return "".join(tokens), "medium", False, "Moves were extracted from separated single-letter tokens."
    return None, "low", False, "No U/D/L/R move string found."


def _is_path_syntactically_valid(moves: str) -> bool:
    return bool(moves) and all(ch in "UDLR" for ch in moves)


def _maze_invalid_reason(
    grid: Sequence[str],
    moves: str,
    must_be_shortest: bool,
    max_length: int | None = None,
) -> str:
    start, goal = maze.find_points(grid)
    cell = start
    seen = {cell}
    move_map = {m: d for m, d in maze.MOVES}
    for move in moves:
        dr, dc = move_map.get(move, (999, 999))
        nxt = (cell[0] + dr, cell[1] + dc)
        if not (0 <= nxt[0] < len(grid) and 0 <= nxt[1] < len(grid[0])):
            return "path_out_of_bounds"
        if grid[nxt[0]][nxt[1]] == "#":
            return "path_hits_wall"
        if not must_be_shortest and nxt in seen:
            return "repeated_cell"
        cell = nxt
        seen.add(cell)
    if cell != goal:
        return "does_not_reach_goal"
    if max_length is not None and len(moves) > max_length:
        return "path_too_long"
    if must_be_shortest:
        shortest = maze.bfs_distances(grid, start).get(goal)
        if shortest is not None and len(moves) != shortest:
            return "not_shortest"
    return "invalid_path"


def _score_grid_placement(puzzle: Dict[str, Any], raw_answer: str) -> Dict[str, Any]:
    coords, confidence, manual, notes = _extract_coordinates(raw_answer)
    if not coords:
        return _invalid("format_error", parse_notes=notes)
    inst = puzzle["machine_readable_instance"]
    board, m = inst["board"], inst["m"]
    if not grid_placement.validate_placement(board, m, coords):
        return _invalid(_grid_invalid_reason(board, m, coords), coords, coords, confidence, manual, notes)
    normalized = sorted([list(c) for c in coords])
    for i, solution in enumerate(puzzle["solutions"]):
        if normalized == sorted(solution["coordinates"]):
            return _valid(_solution_id(puzzle, i), normalized, normalized, coords, confidence, manual, notes)
    return _invalid("valid_but_unmatched_solution", normalized, coords, confidence, True, notes)


def _score_single_coordinate(puzzle: Dict[str, Any], raw_answer: str, validator) -> Dict[str, Any]:
    coords, confidence, manual, notes = _extract_coordinates(raw_answer)
    if not coords:
        return _invalid("format_error", parse_notes=notes)
    if len(coords) != 1:
        return _invalid("wrong_number_of_tokens", coords, coords, confidence, manual, notes)
    coord = coords[0]
    inst = puzzle["machine_readable_instance"]
    if puzzle["puzzle_type"] == "minesweeper_lite":
        ok = validator(inst["board"], inst["target_kind"], coord)
    else:
        ok = validator(inst["board"], inst["digit"], coord)
    if not ok:
        return _invalid("invalid_coordinate", coord, coords, confidence, manual, notes)
    normalized = [int(coord[0]), int(coord[1])]
    for i, solution in enumerate(puzzle["solutions"]):
        if normalized == solution["coordinate"]:
            return _valid(_solution_id(puzzle, i), normalized, normalized, coords, confidence, manual, notes)
    return _invalid("valid_but_unmatched_solution", normalized, coords, confidence, True, notes)


def _extract_coordinates(raw_answer: str) -> Tuple[List[List[int]], str, bool, str]:
    pairs = re.findall(r"\(?\s*(\d+)\s*,\s*(\d+)\s*\)?", raw_answer)
    if pairs:
        return [[int(a), int(b)] for a, b in pairs], "high", False, ""
    ints = [int(x) for x in re.findall(r"\d+", raw_answer)]
    if len(ints) >= 2 and len(ints) % 2 == 0:
        return [[ints[i], ints[i + 1]] for i in range(0, len(ints), 2)], "medium", False, "Coordinates inferred from a flat number list."
    return [], "low", False, "No coordinate pairs found."


def _grid_invalid_reason(board: Sequence[str], m: int, coords: Sequence[Sequence[int]]) -> str:
    n = len(board)
    normalized = [(r, c) for r, c in coords]
    if len(normalized) != m:
        return "wrong_number_of_tokens"
    if len(set(normalized)) != len(normalized):
        return "duplicate_coordinate"
    rows = [r for r, _ in normalized]
    cols = [c for _, c in normalized]
    if len(set(rows)) != len(rows):
        return "row_conflict"
    if len(set(cols)) != len(cols):
        return "column_conflict"
    for r, c in normalized:
        if not (1 <= r <= n and 1 <= c <= n):
            return "coordinate_out_of_bounds"
        if board[r - 1][c - 1] == "X":
            return "forbidden_cell"
    return "invalid_placement"


def split_sequence_response(raw_response: str, expected: int = 2) -> Dict[str, Any]:
    """Split a joint Puzzle A / Puzzle B response into per-puzzle answer strings."""
    text = (raw_response or "").strip()
    answers = [""] * expected
    label_to_index = {label: idx for idx, label in enumerate(("A", "B", "C", "D")[:expected])}
    label_pattern = re.compile(
        r"(?im)(?<![A-Za-z0-9])"
        r"(?:Puzzle|Problem)?\s*([A-D])\s*[:\-]\s*"
    )
    matches = [
        match
        for match in label_pattern.finditer(text)
        if match.group(1).upper() in label_to_index
    ]
    if matches:
        for pos, match in enumerate(matches):
            idx = label_to_index[match.group(1).upper()]
            end = matches[pos + 1].start() if pos + 1 < len(matches) else len(text)
            answer = text[match.end() : end].strip()
            answer = re.sub(r"^[;\s]+|[;\s]+$", "", answer)
            if answer:
                answers[idx] = answer
    if all(answers):
        return {"answers": answers, "parse_confidence": "high", "parse_notes": ""}

    if matches:
        return {
            "answers": answers,
            "parse_confidence": "low",
            "parse_notes": "A labeled sequence answer is missing; left the missing slot empty.",
        }

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) >= expected:
        for idx in range(expected):
            if not answers[idx]:
                answers[idx] = lines[idx]
        return {
            "answers": answers,
            "parse_confidence": "medium",
            "parse_notes": "Sequence answer inferred from line order.",
        }

    return {
        "answers": answers,
        "parse_confidence": "low",
        "parse_notes": "Too few sequence answers; left unanswered slots empty.",
    }
