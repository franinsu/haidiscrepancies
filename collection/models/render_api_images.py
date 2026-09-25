from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from puzzles.common import read_jsonl

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError as exc:  # pragma: no cover - environment guard
    raise SystemExit("Pillow is required to render API PNG images.") from exc


RENDER_VERSION = "api_image_v9"
BG = "#fbfaf7"
PANEL = "#ffffff"
INK = "#121826"
NAVY = "#17206a"
MUTED = "#5d6675"
LINE = "#e1ded6"
GRID_LINE = "#9a9489"
YELLOW_STROKE = "#c98600"
BLUE = "#2563eb"
GREEN = "#16835c"
RED = "#b42318"
WALL = "#24211d"


def main() -> int:
    parser = argparse.ArgumentParser(description="Render formal image inputs for API model collection.")
    parser.add_argument("--main", default="data/stimuli/all_puzzles.jsonl")
    parser.add_argument("--modules", default="data/stimuli/modules/all_module_trials.jsonl")
    parser.add_argument("--blocks", default="data/stimuli/modules/module_blocks.jsonl")
    parser.add_argument("--out_dir", required=True, help="New directory for generated images; must not exist.")
    parser.add_argument("--manifest", required=True, help="New image manifest file; must not exist.")
    args = parser.parse_args()

    counts = render_api_images(
        main_path=args.main,
        modules_path=args.modules,
        blocks_path=args.blocks,
        out_dir=args.out_dir,
        manifest_path=args.manifest,
    )
    for key, value in counts.items():
        print(f"{key}: {value}")
    return 0


def render_api_images(
    main_path: str,
    modules_path: str,
    blocks_path: str,
    out_dir: str,
    manifest_path: str,
) -> Dict[str, int]:
    out = Path(out_dir)
    manifest = Path(manifest_path)
    for path in (out, manifest):
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"Output already exists: {path}. Choose a new destination.")
    if out.resolve() == manifest.resolve():
        raise ValueError("Image directory and manifest must have different destinations.")

    main_rows = read_jsonl(main_path)
    module_rows = read_jsonl(modules_path)
    blocks = read_jsonl(blocks_path)
    rows_by_id = {row["id"]: row for row in [*main_rows, *module_rows]}
    out.mkdir(parents=True)
    manifest.parent.mkdir(parents=True, exist_ok=True)

    records: List[Dict[str, Any]] = []
    counts = {"main_images": 0, "module_single_images": 0, "module_sequence_images": 0}

    for row in main_rows:
        image_id = row["id"]
        image_path = out / f"{_safe_name(image_id)}.png"
        render_single_trial(row, image_path, dataset_name="main")
        records.append(
            _manifest_record(
                image_id=image_id,
                puzzle_id=row["id"],
                sequence_id="",
                dataset_name="main",
                module="main",
                condition=row.get("variant_type", "canonical"),
                puzzle_type=row["puzzle_type"],
                trial_ids=[row["id"]],
                prompt_version="image_prompt_v1",
                prompt_file="collection/models/prompts/single_puzzle_image_prompt_v1.txt",
                image_path=image_path,
            )
        )
        counts["main_images"] += 1

    block_trial_ids = {trial_id for block in blocks for trial_id in block.get("trial_ids", [])}
    sequence_modules = {"pair", "transfer"}
    for row in module_rows:
        module = row.get("module_metadata", {}).get("module", row.get("presentation_metadata", {}).get("module", "module"))
        if module in sequence_modules and row["id"] in block_trial_ids:
            continue
        image_id = row["id"]
        image_path = out / f"{_safe_name(image_id)}.png"
        render_single_trial(row, image_path, dataset_name="modules")
        records.append(
            _manifest_record(
                image_id=image_id,
                puzzle_id=row["id"],
                sequence_id="",
                dataset_name="modules",
                module=module,
                condition=_condition(row),
                puzzle_type=row["puzzle_type"],
                trial_ids=[row["id"]],
                prompt_version="image_prompt_v1",
                prompt_file="collection/models/prompts/single_puzzle_image_prompt_v1.txt",
                image_path=image_path,
            )
        )
        counts["module_single_images"] += 1

    for block in blocks:
        if block.get("module") not in sequence_modules:
            continue
        trials = [rows_by_id[trial_id] for trial_id in block["trial_ids"]]
        image_id = block.get("sequence_id") or block["block_id"]
        image_path = out / f"{_safe_name(image_id)}.png"
        if len(trials) == 1:
            render_single_trial(trials[0], image_path, dataset_name="modules")
            prompt_version = "image_prompt_v1"
            prompt_file = "collection/models/prompts/single_puzzle_image_prompt_v1.txt"
            input_kind = "single"
        else:
            render_sequence_trial(trials, image_path, block=block)
            prompt_version = "image_sequence_prompt_v1"
            prompt_file = "collection/models/prompts/sequence_pair_transfer_image_prompt_v1.txt"
            input_kind = "sequence"
        records.append(
            _manifest_record(
                image_id=image_id,
                puzzle_id=";".join(trial["id"] for trial in trials),
                sequence_id=block.get("sequence_id", image_id),
                dataset_name="modules",
                module=block["module"],
                condition=block.get("condition", ""),
                puzzle_type=block["puzzle_type"],
                trial_ids=[trial["id"] for trial in trials],
                prompt_version=prompt_version,
                prompt_file=prompt_file,
                image_path=image_path,
                input_kind=input_kind,
            )
        )
        counts["module_sequence_images"] += 1

    _remove_unmanifested_pngs(out, records)
    _write_manifest(manifest, records)
    counts["total_images"] = len(records)
    counts["manifest_rows"] = len(records)
    return counts


def render_single_trial(
    row: Dict[str, Any],
    image_path: Path,
    dataset_name: str,
    title_override: str | None = None,
) -> None:
    w, h = 1280, 960
    image, draw = _canvas(w, h)
    fonts = _fonts()
    title = title_override or _type_title(row)
    y = 36
    _text(draw, (64, y), title, fonts["title"], NAVY)
    card_bottom = _draw_instruction_card(draw, row, (64, 84, w - 64), fonts)
    board_top = card_bottom + 26
    board_box = (64, board_top, w - 64, h - 44)
    _draw_puzzle_body(draw, row, board_box, fonts)
    image.save(image_path)


def render_sequence_trial(trials: Sequence[Dict[str, Any]], image_path: Path, block: Dict[str, Any]) -> None:
    w, h = 1720, 1040
    image, draw = _canvas(w, h)
    fonts = _fonts()
    y = 30
    _text(draw, (82, y), _sequence_title(trials), fonts["title"], INK)
    card_bottom = _draw_sequence_instruction_card(draw, trials, (82, 72, w - 82), fonts)
    top = card_bottom + 26
    panel_gap = 28
    panel_w = (w - 164 - panel_gap) // 2
    panel_h = h - top - 54
    labels = ["Puzzle A", "Puzzle B"]
    for i, row in enumerate(trials[:2]):
        x0 = 82 + i * (panel_w + panel_gap)
        _draw_panel(draw, (x0, top, x0 + panel_w, top + panel_h), row, labels[i], fonts)
    image.save(image_path)


def _draw_panel(
    draw: ImageDraw.ImageDraw,
    box: Tuple[int, int, int, int],
    row: Dict[str, Any],
    label: str,
    fonts: Dict[str, ImageFont.FreeTypeFont],
) -> None:
    x0, y0, x1, y1 = box
    draw.rounded_rectangle(box, radius=0, fill=PANEL, outline=LINE, width=2)
    _text(draw, (x0 + 28, y0 + 34), label, fonts["subtitle"], NAVY)
    y = y0 + 68
    task_lines = _wrap_rich(_task_segments(row), fonts["panel_task"], fonts["panel_task_bold"], x1 - x0 - 56, draw)
    for line in task_lines[:6]:
        _draw_rich_line(draw, (x0 + 28, y), line, fonts["panel_task"], fonts["panel_task_bold"], INK)
        y += 31
    board_box = (x0 + 28, y + 10, x1 - 28, y1 - 28)
    _draw_puzzle_body(draw, row, board_box, fonts)


def _draw_puzzle_body(
    draw: ImageDraw.ImageDraw,
    row: Dict[str, Any],
    box: Tuple[int, int, int, int],
    fonts: Dict[str, ImageFont.FreeTypeFont],
) -> None:
    ptype = row["puzzle_type"]
    if ptype == "arithmetic24":
        _draw_arithmetic(draw, row, box, fonts)
    elif ptype == "maze":
        _draw_grid(draw, row["machine_readable_instance"]["grid"], box, fonts, mode="maze", cue_cells=set())
    elif ptype == "grid_placement":
        _draw_grid(draw, row["machine_readable_instance"]["board"], box, fonts, mode="grid_placement", cue_cells=set(), show_coords=True)
    elif ptype == "minesweeper_lite":
        _draw_grid(
            draw,
            row["machine_readable_instance"]["board"],
            box,
            fonts,
            mode="minesweeper_lite",
            cue_cells=_cue_cells(row),
            show_coords=True,
        )
    elif ptype == "mini_sudoku":
        _draw_grid(
            draw,
            row["machine_readable_instance"]["board"],
            box,
            fonts,
            mode="mini_sudoku",
            cue_cells=_cue_cells(row),
            show_coords=True,
            box_rows=int(row["machine_readable_instance"].get("box_rows", 2)),
            box_cols=int(row["machine_readable_instance"].get("box_cols", 3)),
        )
    else:
        _text(draw, (box[0], box[1]), row.get("rendered_puzzle", ""), fonts["body"], INK)


def _draw_instruction_card(
    draw: ImageDraw.ImageDraw,
    row: Dict[str, Any],
    span: Tuple[int, int, int],
    fonts: Dict[str, ImageFont.FreeTypeFont],
) -> int:
    x0, y0, x1 = span
    lines = [
        ("Task", _task_segments(row)),
        ("Rules", _plain_segments(_rules_text(row))),
        ("Output", _plain_segments(_output_text(row))),
    ]
    y = y0 + 22
    row_heights: List[Tuple[str, List[List[Tuple[str, bool]]], int]] = []
    text_font = fonts["instruction"]
    text_bold_font = fonts["instruction_bold"]
    label_font = fonts["label"]
    text_x = x0 + 126
    text_w = x1 - text_x - 24
    line_step = 32
    for label, segments in lines:
        wrapped = _wrap_rich(segments, text_font, text_bold_font, text_w, draw)
        row_heights.append((label, wrapped, max(36, line_step * len(wrapped))))
        y += row_heights[-1][2] + 8
    bottom = y0 + 22 + sum(height + 8 for _, _, height in row_heights) + 14
    draw.rounded_rectangle((x0, y0, x1, bottom), radius=0, fill=PANEL, outline=LINE, width=2)
    y = y0 + 28
    for label, wrapped, height in row_heights:
        _text(draw, (x0 + 24, y), label.upper(), label_font, NAVY)
        ty = y
        for line in wrapped:
            _draw_rich_line(draw, (text_x, ty), line, text_font, text_bold_font, INK)
            ty += line_step
        y += height + 8
    return bottom


def _draw_sequence_instruction_card(
    draw: ImageDraw.ImageDraw,
    trials: Sequence[Dict[str, Any]],
    span: Tuple[int, int, int],
    fonts: Dict[str, ImageFont.FreeTypeFont],
) -> int:
    x0, y0, x1 = span
    lines = [
        ("Task", "Solve Puzzle A and Puzzle B."),
        ("Rules", _sequence_rules_text(trials)),
        ("Output", "Return exactly two lines and nothing else. Replace placeholders with your answers."),
        ("Line 1", f"A: {_sequence_answer_format_text(trials[0])}"),
        ("Line 2", f"B: {_sequence_answer_format_text(trials[1])}"),
    ]
    y = y0 + 22
    row_heights: List[Tuple[str, List[str], int]] = []
    text_font = fonts["instruction"]
    label_font = fonts["label"]
    text_x = x0 + 126
    text_w = x1 - text_x - 24
    line_step = 32
    for label, text in lines:
        wrapped = _wrap(text, text_font, text_w, draw)
        row_heights.append((label, wrapped, max(36, line_step * len(wrapped))))
        y += row_heights[-1][2] + 8
    bottom = y0 + 22 + sum(height + 8 for _, _, height in row_heights) + 14
    draw.rounded_rectangle((x0, y0, x1, bottom), radius=0, fill=PANEL, outline=LINE, width=2)
    y = y0 + 28
    for label, wrapped, height in row_heights:
        _text(draw, (x0 + 24, y), label.upper(), label_font, NAVY)
        ty = y
        for line in wrapped:
            _text(draw, (text_x, ty), line, text_font, INK)
            ty += line_step
        y += height + 8
    return bottom


def _draw_arithmetic(
    draw: ImageDraw.ImageDraw,
    row: Dict[str, Any],
    box: Tuple[int, int, int, int],
    fonts: Dict[str, ImageFont.FreeTypeFont],
) -> None:
    x0, y0, x1, y1 = box
    draw.rounded_rectangle(box, radius=0, fill=PANEL, outline=LINE, width=2)
    numbers = row["machine_readable_instance"].get("display_numbers") or row["machine_readable_instance"]["numbers"]
    _text(draw, ((x0 + x1) // 2, y0 + 92), "Numbers", fonts["subtitle"], MUTED, anchor="mm")
    card_w = min(180, (x1 - x0 - 180) // 4)
    gap = 24
    total = 4 * card_w + 3 * gap
    start = (x0 + x1 - total) // 2
    cy = y0 + 150
    for i, n in enumerate(numbers):
        cx = start + i * (card_w + gap)
        draw.rounded_rectangle((cx, cy, cx + card_w, cy + 128), radius=0, fill="#fff8e8", outline=LINE, width=2)
        _text_centered(draw, (cx, cy, cx + card_w, cy + 128), str(n), fonts["huge"], INK)


def _draw_grid(
    draw: ImageDraw.ImageDraw,
    grid: Sequence[str],
    box: Tuple[int, int, int, int],
    fonts: Dict[str, ImageFont.FreeTypeFont],
    mode: str,
    cue_cells: set[Tuple[int, int]],
    show_coords: bool = False,
    box_rows: int = 0,
    box_cols: int = 0,
) -> None:
    rows, cols = len(grid), len(grid[0])
    x0, y0, x1, y1 = box
    draw.rounded_rectangle(box, radius=0, fill=PANEL, outline=LINE, width=2)
    pad = 34
    label = 46 if show_coords else 0
    cell = int(min((x1 - x0 - 2 * pad - label) / cols, (y1 - y0 - 2 * pad - label) / rows))
    cell = max(42, min(cell, 116))
    total_w = cols * cell + label
    total_h = rows * cell + label
    gx = int(x0 + (x1 - x0 - total_w) / 2 + label)
    gy = int(y0 + (y1 - y0 - total_h) / 2 + label)

    if show_coords:
        for c in range(1, cols + 1):
            _text(draw, (gx + (c - 0.5) * cell, gy - 20), str(c), fonts["coord"], MUTED, anchor="mm")
        for r in range(1, rows + 1):
            _text(draw, (gx - 24, gy + (r - 0.5) * cell), str(r), fonts["coord"], MUTED, anchor="mm")

    for r, line in enumerate(grid, 1):
        for c, ch in enumerate(line, 1):
            left = gx + (c - 1) * cell
            top = gy + (r - 1) * cell
            coord = (r, c)
            fill, text_color, label_text = _cell_style(mode, ch, coord in cue_cells)
            draw.rectangle((left, top, left + cell, top + cell), fill=fill, outline=GRID_LINE, width=1)
            if coord in cue_cells:
                draw.rectangle((left + 4, top + 4, left + cell - 4, top + cell - 4), outline=YELLOW_STROKE, width=4)
            if label_text:
                _text_centered(
                    draw,
                    (left, top, left + cell, top + cell),
                    label_text,
                    _cell_font(fonts, cell, mode),
                    text_color,
                )

    if mode == "mini_sudoku" and box_rows and box_cols:
        for r in range(0, rows + 1, box_rows):
            y = gy + r * cell
            draw.line((gx, y, gx + cols * cell, y), fill=INK, width=4)
        for c in range(0, cols + 1, box_cols):
            x = gx + c * cell
            draw.line((x, gy, x, gy + rows * cell), fill=INK, width=4)
    else:
        draw.rectangle((gx, gy, gx + cols * cell, gy + rows * cell), outline=INK, width=3)


def _cell_style(mode: str, ch: str, highlighted: bool) -> Tuple[str, str, str]:
    base = "#ffffff"
    if mode == "maze":
        if ch == "#":
            return WALL, "#ffffff", ""
        if ch == "S":
            return "#d7f5e6", GREEN, "S"
        if ch == "G":
            return "#dbeafe", BLUE, "G"
        return "#ffffff", INK, ""
    if mode == "grid_placement":
        if ch == "X":
            return WALL, "#ffffff", "X"
        return base, INK, ""
    if mode == "minesweeper_lite":
        if ch == "F":
            return "#e8eeec", INK, "F"
        if ch == "?":
            return "#fffdf8", MUTED, "?"
        return "#e8eeec", INK, ch
    if mode == "mini_sudoku":
        if ch in {".", "0"}:
            return "#fffdf8", INK, ""
        return "#e6ecec", INK, ch
    return base, INK, "" if ch == "." else ch


def _task_text(row: Dict[str, Any]) -> str:
    ptype = row["puzzle_type"]
    inst = row["machine_readable_instance"]
    if ptype == "arithmetic24":
        return "Make exactly 24 using the four displayed numbers."
    if ptype == "maze":
        return "Find one shortest path from S to G."
    if ptype == "grid_placement":
        return f"Place {inst['m']} tokens on the board."
    if ptype == "minesweeper_lite":
        target = "safe" if inst.get("target_kind") == "safe" else "a mine"
        return (
            "Select exactly one hidden cell marked ?. Each number gives the exact count of mines in the up to eight "
            "neighboring cells, including diagonals. F marks a known mine. Your selected ? cell must be forced by the "
            f"clues: in every mine arrangement consistent with all numbers, that cell is {target}."
        )
    if ptype == "mini_sudoku":
        digit = inst["digit"]
        return (
            f"Select exactly one empty cell for digit {digit}. A cell is legal only if {digit} does not already appear "
            "in the same row, the same column, or the same outlined box. Printed numbers are fixed clues and cannot be "
            "selected."
        )
    return row.get("prompt_text", "")


def _task_segments(row: Dict[str, Any]) -> List[Tuple[str, bool]]:
    ptype = row["puzzle_type"]
    inst = row["machine_readable_instance"]
    if ptype == "minesweeper_lite":
        target = "safe" if inst.get("target_kind") == "safe" else "mine"
        segments: List[Tuple[str, bool]] = [
            (
                "Select exactly one hidden cell marked ?. Each number gives the exact count of mines in the up to eight "
                "neighboring cells, including diagonals. F marks a known mine. Your selected ? cell must be forced by the "
                "clues: in every mine arrangement consistent with all numbers, that cell is ",
                False,
            )
        ]
        if target == "mine":
            segments.append(("a ", False))
        segments.extend([(target, True), (".", False)])
        return segments
    if ptype == "mini_sudoku":
        digit = str(inst["digit"])
        return [
            ("Select exactly one empty cell for digit ", False),
            (digit, True),
            (". A cell is legal only if ", False),
            (digit, True),
            (
                " does not already appear in the same row, the same column, or the same outlined box. Printed numbers are "
                "fixed clues and cannot be selected.",
                False,
            ),
        ]
    return _plain_segments(_task_text(row))


def _plain_segments(text: str) -> List[Tuple[str, bool]]:
    return [(text, False)]


def _rules_text(row: Dict[str, Any]) -> str:
    ptype = row["puzzle_type"]
    inst = row["machine_readable_instance"]
    if ptype == "arithmetic24":
        return "Use each number exactly once. Allowed operations: +, -, *, /. Parentheses are allowed. Do not use other numbers. Only one correct answer is needed."
    if ptype == "maze":
        return "Move only U/D/L/R. White cells are open; black cells are walls. S and G are open. The path must be shortest. Only one correct answer is needed."
    if ptype == "grid_placement":
        return f"Use row/column numbers. Place exactly {inst['m']} tokens. Do not place on X. No two tokens may share a row or column. Only one correct answer is needed."
    if ptype == "minesweeper_lite":
        return "Only one correct answer is needed."
    if ptype == "mini_sudoku":
        return "Only one correct answer is needed."
    return "Follow the rules shown in the puzzle."


def _sequence_title(trials: Sequence[Dict[str, Any]]) -> str:
    titles = [_type_title(row) for row in trials[:2]]
    if titles and all(title == titles[0] for title in titles):
        return f"Two Puzzles: {titles[0]}"
    return "Two Puzzles"


def _sequence_rules_text(trials: Sequence[Dict[str, Any]]) -> str:
    ptype = trials[0]["puzzle_type"]
    inst = trials[0]["machine_readable_instance"]
    if ptype == "minesweeper_lite":
        return "Use the rule statement shown above each Minesweeper board. Only one correct answer is needed for each puzzle."
    if ptype == "mini_sudoku":
        return "Use the rule statement shown above each Mini Sudoku board. Only one correct answer is needed for each puzzle."
    return _rules_text(trials[0]).replace("Only one correct answer is needed.", "Only one correct answer is needed for each puzzle.")


def _answer_format_text(row: Dict[str, Any]) -> str:
    ptype = row["puzzle_type"]
    inst = row.get("machine_readable_instance", {})
    if ptype == "arithmetic24":
        return "ANSWER: expression=<expression>"
    if ptype == "maze":
        return "ANSWER: path=<UDLR-moves>"
    if ptype == "grid_placement":
        return "ANSWER: cells=" + _placement_placeholder(int(inst.get("m", 3)))
    if ptype in {"minesweeper_lite", "mini_sudoku"}:
        return "ANSWER: cell=<row,column>"
    return "final answer"


def _sequence_answer_format_text(row: Dict[str, Any]) -> str:
    ptype = row["puzzle_type"]
    inst = row.get("machine_readable_instance", {})
    if ptype == "arithmetic24":
        return "expression=<expression>"
    if ptype == "maze":
        return "path=<UDLR-moves>"
    if ptype == "grid_placement":
        return "cells=" + _placement_placeholder(int(inst.get("m", 3)))
    if ptype in {"minesweeper_lite", "mini_sudoku"}:
        return "cell=<row,column>"
    return "<answer>"


def _output_text(row: Dict[str, Any]) -> str:
    return f"Return exactly one line: {_answer_format_text(row)}. Replace placeholders with your answer."


def _placement_placeholder(m: int) -> str:
    return "; ".join(f"<r{i},c{i}>" for i in range(1, max(1, m) + 1))


def _canvas(w: int, h: int) -> Tuple[Image.Image, ImageDraw.ImageDraw]:
    image = Image.new("RGB", (w, h), BG)
    return image, ImageDraw.Draw(image)


def _fonts() -> Dict[str, ImageFont.FreeTypeFont]:
    return {
        "title": _font(40, bold=True),
        "subtitle": _font(27, bold=True),
        "body": _font(26),
        "body_bold": _font(26, bold=True),
        "label": _font(19, bold=True),
        "instruction": _font(26),
        "instruction_bold": _font(26, bold=True),
        "panel_task": _font(26),
        "panel_task_bold": _font(26, bold=True),
        "small": _font(20),
        "tiny": _font(17),
        "coord": _font(21, bold=True),
        "cell": _font(34, bold=True),
        "cell_small": _font(28, bold=True),
        "huge": _font(58, bold=True),
    }


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial Bold.ttf" if bold else "/Library/Fonts/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for path in candidates:
        if path and Path(path).exists():
            # Match the frozen study images even when Pillow includes RAQM.
            return ImageFont.truetype(path, size=size, layout_engine=ImageFont.Layout.BASIC)
    return ImageFont.load_default()


def _cell_font(fonts: Dict[str, ImageFont.FreeTypeFont], cell: int, mode: str) -> ImageFont.FreeTypeFont:
    if cell < 56 or mode == "mini_sudoku":
        return fonts["cell_small"]
    return fonts["cell"]


def _text(
    draw: ImageDraw.ImageDraw,
    xy: Tuple[float, float],
    text: Any,
    font: ImageFont.FreeTypeFont,
    fill: str,
    anchor: str = "la",
) -> None:
    draw.text(xy, str(text), font=font, fill=fill, anchor=anchor)


def _text_centered(
    draw: ImageDraw.ImageDraw,
    rect: Tuple[float, float, float, float],
    text: Any,
    font: ImageFont.FreeTypeFont,
    fill: str,
) -> None:
    label = str(text)
    box = draw.textbbox((0, 0), label, font=font)
    tw = box[2] - box[0]
    th = box[3] - box[1]
    x0, y0, x1, y1 = rect
    x = x0 + (x1 - x0 - tw) / 2 - box[0]
    y = y0 + (y1 - y0 - th) / 2 - box[1]
    draw.text((x, y), label, font=font, fill=fill)


def _draw_rich_line(
    draw: ImageDraw.ImageDraw,
    xy: Tuple[float, float],
    segments: Sequence[Tuple[str, bool]],
    font: ImageFont.FreeTypeFont,
    bold_font: ImageFont.FreeTypeFont,
    fill: str,
) -> None:
    x, y = xy
    for text, bold in segments:
        if not text:
            continue
        active_font = bold_font if bold else font
        draw.text((x, y), text, font=active_font, fill=fill, anchor="la")
        x += _text_width(draw, text, active_font)


def _wrap_rich(
    segments: Sequence[Tuple[str, bool]],
    font: ImageFont.FreeTypeFont,
    bold_font: ImageFont.FreeTypeFont,
    max_width: int,
    draw: ImageDraw.ImageDraw,
) -> List[List[Tuple[str, bool]]]:
    lines: List[List[Tuple[str, bool]]] = []
    current: List[Tuple[str, bool]] = []
    for text, bold in segments:
        for token in re.findall(r"\s+|\S+", str(text)):
            if not current and token.isspace():
                continue
            candidate = [*current, (token, bold)]
            if _rich_text_width(draw, candidate, font, bold_font) <= max_width or not _has_visible_text(current):
                current = candidate
            else:
                lines.append(_normalize_rich_line(current))
                current = [] if token.isspace() else [(token, bold)]
    if current:
        lines.append(_normalize_rich_line(current))
    return lines


def _has_visible_text(segments: Sequence[Tuple[str, bool]]) -> bool:
    return any(text.strip() for text, _ in segments)


def _normalize_rich_line(segments: Sequence[Tuple[str, bool]]) -> List[Tuple[str, bool]]:
    trimmed = list(segments)
    while trimmed and not trimmed[0][0].strip():
        trimmed.pop(0)
    while trimmed and not trimmed[-1][0].strip():
        trimmed.pop()
    merged: List[Tuple[str, bool]] = []
    for text, bold in trimmed:
        if not text:
            continue
        if merged and merged[-1][1] == bold:
            merged[-1] = (merged[-1][0] + text, bold)
        else:
            merged.append((text, bold))
    return merged


def _rich_text_width(
    draw: ImageDraw.ImageDraw,
    segments: Sequence[Tuple[str, bool]],
    font: ImageFont.FreeTypeFont,
    bold_font: ImageFont.FreeTypeFont,
) -> int:
    return sum(_text_width(draw, text, bold_font if bold else font) for text, bold in segments)


def _wrap(text: str, font: ImageFont.FreeTypeFont, max_width: int, draw: ImageDraw.ImageDraw) -> List[str]:
    words = str(text).split()
    lines: List[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if _text_width(draw, candidate, font) <= max_width or not current:
            current = candidate
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def _text_width(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont) -> int:
    box = draw.textbbox((0, 0), text, font=font)
    return box[2] - box[0]


def _cue_cells(row: Dict[str, Any]) -> set[Tuple[int, int]]:
    cue = _cue_from(row)
    if not cue:
        return set()
    coords = cue.get("coordinates") or []
    if cue.get("coordinate"):
        coords = [*coords, cue["coordinate"]]
    return {tuple(map(int, coord)) for coord in coords}


def _cue_from(row: Dict[str, Any]) -> Dict[str, Any] | None:
    inst = row.get("machine_readable_instance", {})
    pres = row.get("presentation_metadata", {})
    return inst.get("visual_cue") or inst.get("irrelevant_cue") or pres.get("visual_cue") or pres.get("irrelevant_cue")


def _type_title(row: Dict[str, Any]) -> str:
    return {
        "arithmetic24": "Arithmetic 24",
        "maze": "Shortest Path Maze",
        "grid_placement": "Grid Placement",
        "minesweeper_lite": "Minesweeper Lite",
        "mini_sudoku": "Mini Sudoku",
    }.get(row["puzzle_type"], row["puzzle_type"])


def _module(row: Dict[str, Any]) -> str:
    return row.get("module_metadata", {}).get("module") or row.get("presentation_metadata", {}).get("module") or ""


def _condition(row: Dict[str, Any]) -> str:
    return (
        row.get("module_metadata", {}).get("condition")
        or row.get("presentation_metadata", {}).get("module_condition")
        or row.get("variant_type", "")
    )


def _safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _manifest_record(
    image_id: str,
    puzzle_id: str,
    sequence_id: str,
    dataset_name: str,
    module: str,
    condition: str,
    puzzle_type: str,
    trial_ids: Sequence[str],
    prompt_version: str,
    prompt_file: str,
    image_path: Path,
    input_kind: str = "single",
) -> Dict[str, Any]:
    with Image.open(image_path) as img:
        width, height = img.size
    return {
        "image_id": image_id,
        "puzzle_id": puzzle_id,
        "sequence_id": sequence_id,
        "dataset_name": dataset_name,
        "module": module,
        "condition": condition,
        "puzzle_type": puzzle_type,
        "trial_ids": ";".join(trial_ids),
        "input_kind": input_kind,
        "prompt_version": prompt_version,
        "prompt_file": prompt_file,
        "image_path": str(image_path),
        "image_sha256": _sha256(image_path),
        "render_version": RENDER_VERSION,
        "width": width,
        "height": height,
    }


def _write_manifest(path: Path, records: Sequence[Dict[str, Any]]) -> None:
    fields = [
        "image_id",
        "puzzle_id",
        "sequence_id",
        "dataset_name",
        "module",
        "condition",
        "puzzle_type",
        "trial_ids",
        "input_kind",
        "prompt_version",
        "prompt_file",
        "image_path",
        "image_sha256",
        "render_version",
        "width",
        "height",
    ]
    with path.open("x", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)


def _remove_unmanifested_pngs(out_dir: Path, records: Sequence[Dict[str, Any]]) -> None:
    manifest_names = {Path(record["image_path"]).name for record in records}
    for path in out_dir.glob("*.png"):
        if path.name not in manifest_names:
            path.unlink()


if __name__ == "__main__":
    raise SystemExit(main())
