"""Format computed study results as manuscript tables. No statistical inference."""
from __future__ import annotations
from typing import Any
from .tables_entropy import SOURCE_HEADERS, PUZZLE_TYPES
def _num(value: float) -> str:
    value = 0.0 if abs(value) < 0.005 else value
    return f"{value:.2f}"

def _corr_cell(item: dict[str, Any]) -> str:
    low, high = item["ci95"]
    return f"${_num(item['estimate'])}[{_num(low)},{_num(high)}]$"

def render_correlation_table(data: dict[str, Any]) -> str:
    results = data["correlations"]["results"]
    lines = [
        "% Generated log-effort correlation table.",
        "% Do not edit numerical cells by hand.",
        "\\begin{table}[!htbp]",
        "  \\caption{Pearson correlation of per-puzzle relative-difficulty scores",
        "  by puzzle family.  The Mean column weights the five family correlations",
        "  equally.  Brackets are 95\\% paired-puzzle bootstrap intervals.}",
        "  \\label{tab:effort-correlation}",
        "  \\centering",
        "  \\small",
        "  \\setlength{\\tabcolsep}{1.2pt}",
        "  \\renewcommand{\\arraystretch}{1.08}",
        "  \\begin{tabular}{@{}lrrrrrr@{}}",
        "    \\toprule",
        r"    & \multicolumn{6}{c}{Pearson correlation [95\% CI]} \\",
        "    \\cmidrule(l){2-7}",
        "    Source pair & Mean & Arithmetic & Maze & Rooks & Minesweeper & Sudoku \\\\",
        "    \\midrule",
    ]
    for key in data["correlations"]["pair_order"]:
        item = results[key]
        source_x, source_y = item["sources"]
        cells = [_corr_cell(item["mean"])] + [_corr_cell(item["by_type"][puzzle_type]) for puzzle_type, _ in PUZZLE_TYPES]
        lines.append(f"    {SOURCE_HEADERS[source_x]} vs {SOURCE_HEADERS[source_y]} & " + " & ".join(cells) + r" \\")
    lines.extend(["    \\bottomrule", "  \\end{tabular}", "\\end{table}", ""])
    return "\n".join(lines)
