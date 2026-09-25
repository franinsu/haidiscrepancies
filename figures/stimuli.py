"""Six study-board illustrations from the public stimulus catalog."""
from pathlib import Path
from collections import defaultdict
from typing import Any
import json
import re
from matplotlib.patches import Rectangle
from .plot_style import plt, FONT_FAMILY
INK = '#16181B'
MUTED = '#5B6168'

def pagella_text(ax,x,y,text_value,*,size,bold=False,**kwargs):
    return ax.text(x,y,text_value,fontfamily=FONT_FAMILY,fontsize=size,fontweight='bold' if bold else 'normal',**kwargs)


def page_canvas(width: float, height: float) -> tuple[plt.Figure, plt.Axes]:
    fig = plt.figure(figsize=(width / 72, height / 72), dpi=72, facecolor="white")
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, width); ax.set_ylim(0, height); ax.axis("off")
    return fig, ax

def load_project(project: Path) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    def read(path: Path) -> list[dict[str, Any]]:
        with path.open() as stream:
            return [json.loads(line) for line in stream if line.strip()]
    rows = read(project / "all_puzzles.jsonl") + read(project / "modules/all_module_trials.jsonl")
    return {row["id"]: row for row in rows}, read(project / "modules/module_blocks.jsonl")

def cue_cells(row: dict[str, Any]) -> set[tuple[int, int]]:
    cue = row.get("presentation_metadata", {}).get("visual_cue", {})
    return {tuple(cell) for key in ("highlighted_valid_coordinates", "highlighted_invalid_coordinates") for cell in cue.get(key, [])}

def draw_board_page(
    ax: plt.Axes,
    row: dict[str, Any],
    x: float,
    y: float,
    width: float,
    height: float,
    *,
    coordinates: bool = False,
    types_page: bool = False,
    proportional_label_offset: bool = False,
) -> None:
    """Draw one stimulus board in fixed page coordinates."""
    ptype = row["puzzle_type"]
    instance = row["machine_readable_instance"]
    grid = instance.get("grid") or instance.get("board")
    rows, cols = len(grid), len(grid[0])
    cell_width, cell_height = width / cols, height / rows
    highlights = cue_cells(row)
    for row_index, line in enumerate(grid):
        for col_index, value in enumerate(line):
            left = x + col_index * cell_width
            bottom = y + (rows - row_index - 1) * cell_height
            fill, label, color = "#F4F5F7", "", INK
            if ptype == "maze" and value == "#":
                fill = "#3A3E44" if types_page else "#3F444B"
            elif ptype == "grid_placement" and value in {"#", "X"}:
                fill = "#3A3E44" if types_page else "#3F444B"
            elif value in {".", " "}:
                fill = "white"
            elif value == "?":
                fill, label, color = "white", "?", MUTED
            elif value in {"S", "G"}:
                fill, label = "white", value
            else:
                label = str(value)
            ax.add_patch(
                Rectangle(
                    (left, bottom),
                    cell_width,
                    cell_height,
                    facecolor=fill,
                    edgecolor="#B3B9C0",
                    linewidth=0.6,
                )
            )
            if label:
                if types_page and ptype in {"minesweeper_lite", "mini_sudoku"}:
                    label_offset = 0.835612
                elif types_page and ptype == "maze":
                    label_offset = cell_height * 0.06
                elif proportional_label_offset and ptype == "mini_sudoku":
                    label_offset = cell_height * 0.05
                else:
                    label_offset = 0.741186
                pagella_text(
                    ax,
                    left + cell_width / 2,
                    bottom + cell_height / 2 - label_offset,
                    label,
                    size=7.6 if label == "?" else (8.6 if types_page and ptype == "maze" and label in {"S", "G"} else 8.4),
                    bold=label != "?",
                    ha="center",
                    va="center",
                    color=color,
                )
            if (row_index + 1, col_index + 1) in highlights:
                inset = min(cell_width, cell_height) * 0.075
                ax.add_patch(
                    Rectangle(
                        (left + inset, bottom + inset),
                        cell_width - 2 * inset,
                        cell_height - 2 * inset,
                        fill=False,
                        edgecolor="#C8791A",
                        linewidth=1.5,
                        zorder=1.5,
                    )
                )
    if ptype == "mini_sudoku":
        box_rows, box_cols = int(instance.get("box_rows", 2)), int(instance.get("box_cols", 3))
        for row_index in range(0, rows + 1, box_rows):
            line_y = y + row_index * cell_height
            ax.plot([x, x + width], [line_y, line_y], color=INK, lw=1.15, solid_capstyle="butt")
        for col_index in range(0, cols + 1, box_cols):
            line_x = x + col_index * cell_width
            ax.plot([line_x, line_x], [y, y + height], color=INK, lw=1.15, solid_capstyle="butt")
    ax.add_patch(Rectangle((x, y), width, height, fill=False, edgecolor=INK, linewidth=1.15, zorder=2))
    if coordinates:
        column_offset = 5.179559 if types_page else 5.1
        row_offset = 0.835611 if types_page else 0.0
        for col_index in range(cols):
            pagella_text(ax, x + (col_index + .5) * cell_width, y + height + column_offset, str(col_index + 1), size=5.9, ha="center", va="center", color=MUTED)
        for row_index in range(rows):
            pagella_text(ax, x - 5.18, y + height - (row_index + .5) * cell_height - row_offset, str(row_index + 1), size=5.9, ha="center", va="center", color=MUTED)

def draw_number_row(
    ax: plt.Axes,
    numbers: list[int],
    x: float,
    y: float,
    cell: float,
    gap: float,
    *,
    types_page: bool = False,
) -> None:
    for index, number in enumerate(numbers):
        left = x + index * (cell + gap)
        ax.add_patch(Rectangle((left, y), cell, cell, facecolor="white", edgecolor=INK, linewidth=1.0))
        offset = 1.104274 if types_page else 1.475712
        pagella_text(ax, left + cell / 2, y + cell / 2 - offset, str(number), size=10.0, bold=True, ha="center", va="center", color=INK)

def stimulus_figures(project: Path, out: Path, *, only=None) -> None:
    def save_page(fig, path):
        if only is None or path.stem in only:
            fig.set_dpi(300)
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            canvas = fig.bbox.padded(.5)
            clipped = any(not canvas.contains(*point)
                          for ax in fig.axes for text in ax.texts
                          for point in text.get_window_extent(renderer).get_points())
            # Preserve the paper's canvas unless a wider font needs more room.
            fig.savefig(path,facecolor='white',bbox_inches='tight' if clipped else None,
                        pad_inches=.02 if clipped else 0,dpi=300)
        plt.close(fig)

    puzzles, blocks = load_project(project)
    by_module_type: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for block in blocks:
        by_module_type[(block["module"], block["puzzle_type"])].append(block)

    # Five representative task families.
    fig, ax = page_canvas(469.019520, 97.8207375)
    draw_number_row(ax, [3, 4, 4, 8], 5.194533, 47.283979, 18.993517, 3.091968, types_page=True)
    draw_board_page(ax, puzzles["MZ_0001_CAN"], 103.926936, 17.848153, 77.865168, 77.865168, types_page=True)
    draw_board_page(ax, puzzles["GP_0004_CAN"], 209.661545, 18.183269, 66.835443, 66.835443, coordinates=True, types_page=True)
    draw_board_page(ax, puzzles["MS_0001_CAN"], 304.701545, 18.183269, 66.835443, 66.835443, coordinates=True, types_page=True)
    draw_board_page(ax, puzzles["SUD_0003_CAN"], 399.741545, 18.183269, 66.835443, 66.835443, coordinates=True, types_page=True)
    for center, title in zip((47.81952, 142.859519, 243.079267, 338.119267, 433.159267), ("Arithmetic", "Maze", "Rooks", "Minesweeper", "Sudoku")):
        pagella_text(ax, center, 3.65875, title, size=8.6, bold=True, ha="center", va="baseline", color=INK)
    save_page(fig, out / "fBboard_types.png")

    # Cue module.
    fig, ax = page_canvas(372.7145106, 185.04)
    for row_index, ptype in enumerate(("minesweeper_lite", "mini_sudoku")):
        block = by_module_type[("cue", ptype)][0]
        if row_index == 0:
            positions = ((73.177111,93.105),(182.378311,93.105),(291.579511,93.105))
            extent = 78.75
        else:
            positions = (
                (73.022699,2.2305882352941175),
                (182.223899,2.2305882352941175),
                (291.425099,2.2305882352941175),
            )
            extent = 79.05882352941177
        for (board_x,board_y), puzzle_id in zip(positions, block["trial_ids"]):
            draw_board_page(ax, puzzles[puzzle_id], board_x, board_y, extent, extent, proportional_label_offset=True)
    for center, title in zip((112.552111, 221.753311, 330.954511), ("no cue", "cue experiment A", "cue experiment B")):
        pagella_text(ax, center, 177.146875, title, size=8.6, bold=True, ha="center", va="baseline", color=INK)
    pagella_text(ax, 52.393125, 132.48, "Minesweeper", size=8.2, bold=True, ha="right", va="center", color=INK)
    pagella_text(ax, 52.393125, 41.76, "Sudoku", size=8.2, bold=True, ha="right", va="center", color=INK)
    save_page(fig, out / "fBboard_cue.png")

    # Spatial transformations.
    fig, ax = page_canvas(457.2524658, 185.04)
    for row_index, ptype in enumerate(("minesweeper_lite", "mini_sudoku")):
        block = by_module_type[("spatial", ptype)][0]
        if row_index == 0:
            positions = ((70.9527,93.1050),(172.6743,93.1050),(274.3959,93.1050),(376.1175,93.1050))
            extent = 78.75
        else:
            positions = ((70.7983,2.2306),(172.5199,2.2306),(274.2415,2.2306),(375.9631,2.2306))
            extent = 79.0588
        for (board_x,board_y), puzzle_id in zip(positions, block["trial_ids"]):
            draw_board_page(ax, puzzles[puzzle_id], board_x, board_y, extent, extent, proportional_label_offset=True)
    for center, title in zip((110.327666, 212.049266, 313.770866, 415.492466), ("original", "mirror L–R", "mirror T–B", "both mirrors")):
        pagella_text(ax, center, 177.146875, title, size=8.6, bold=True, ha="center", va="baseline", color=INK)
    pagella_text(ax, 52.393125, 132.48, "Minesweeper", size=8.2, bold=True, ha="right", va="center", color=INK)
    pagella_text(ax, 52.393125, 41.76, "Sudoku", size=8.2, bold=True, ha="right", va="center", color=INK)
    save_page(fig, out / "fBboard_spatial.png")

    # Arithmetic formulation/order manipulation.
    block = by_module_type[("formulation", "arithmetic24")][0]
    fig, ax = page_canvas(253.847808, 61.73568)
    for left, puzzle_id, title, center in zip((6.457421, 133.465421), block["trial_ids"], ("original order", "reordered"), (63.419904, 190.427904)):
        instance = puzzles[puzzle_id]["machine_readable_instance"]
        numbers = instance.get("display_numbers") or instance.get("numbers")
        draw_number_row(ax, numbers, left, 10.884557, 25.382246, 4.131994)
        pagella_text(ax, center, 53.842555, title, size=8.6, bold=True, ha="center", va="baseline", color=INK)
    save_page(fig, out / "fBboard_form.png")

    # Pair: B alone, unrelated A/B, related A/B.
    pair_groups: dict[tuple[str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for block in blocks:
        if block["module"] == "pair":
            stem = re.sub(r"_(?:NO_PRIME|REL|CTRL)$", "", str(block["block_id"]))
            pair_groups[(block["puzzle_type"], stem)][str(block["condition"])] = block
    fig, ax = page_canvas(480.703125, 174.96)
    for row_index, ptype in enumerate(("minesweeper_lite", "mini_sudoku")):
        group = next(value for (observed, _), value in pair_groups.items() if observed == ptype)
        ids = [group["no_prime"]["trial_ids"][-1], *group["unrelated_control"]["trial_ids"], *group["related"]["trial_ids"]]
        if row_index == 0:
            positions = ((61.834822,88.005938),(153.263303,88.005938),(233.191943,88.005938),(324.620422,88.005938),(404.549062,88.005938))
            extent = 73.828125
        else:
            positions = ((61.690061,2.181177),(153.118541,2.181177),(233.047181,2.181177),(324.475661,2.181177),(404.404301,2.181177))
            extent = 74.117647
        for (board_x,board_y), puzzle_id in zip(positions, ids):
            draw_board_page(ax, puzzles[puzzle_id], board_x, board_y, extent, extent, proportional_label_offset=True)
    for center, title in ((98.772645, "B alone"), (230.165445, "unrelated A + B"), (401.522565, "related A + B")):
        pagella_text(ax, center, 167.066875, title, size=8.6, bold=True, ha="center", va="baseline", color=INK)
    pagella_text(ax, 52.393125, 124.92, "Minesweeper", size=8.2, bold=True, ha="right", va="center", color=INK)
    pagella_text(ax, 52.393125, 39.24, "Sudoku", size=8.2, bold=True, ha="right", va="center", color=INK)
    save_page(fig, out / "fBboard_pair.png")

    transfer_groups: dict[tuple[str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for block in blocks:
        if block["module"] == "transfer":
            stem = re.sub(r"_(?:PRIME|ALONE)$", "", str(block["block_id"]))
            transfer_groups[(block["puzzle_type"], stem)][str(block["condition"])] = block
    fig, ax = page_canvas(363.706725, 205.2)
    for row_index, ptype in enumerate(("minesweeper_lite", "mini_sudoku")):
        if ptype == "mini_sudoku":
            group = next(value for (observed, stem), value in transfer_groups.items() if observed == ptype and stem.endswith("SUD_0003"))
        else:
            group = next(value for (observed, _), value in transfer_groups.items() if observed == ptype)
        ids = [group["no_prime"]["trial_ids"][-1], *group["strategy_prime"]["trial_ids"]]
        board_y = 103.1294 if row_index == 0 else 2.3294
        for board_x, puzzle_id in zip((61.4617,173.7817,272.4361), ids):
            draw_board_page(ax, puzzles[puzzle_id], board_x, board_y, 88.9412, 88.9412)
    pagella_text(ax, 105.932325, 197.306875, "B alone", size=8.6, bold=True, ha="center", va="baseline", color=INK)
    pagella_text(ax, 267.598245, 197.369375, "primer puzzle A + target puzzle B", size=8.6, bold=True, ha="center", va="baseline", color=INK)
    pagella_text(ax, 52.393125, 147.6, "Minesweeper", size=8.2, bold=True, ha="right", va="center", color=INK)
    pagella_text(ax, 52.393125, 46.8, "Sudoku", size=8.2, bold=True, ha="right", va="center", color=INK)
    save_page(fig, out / "fBboard_transfer.png")
