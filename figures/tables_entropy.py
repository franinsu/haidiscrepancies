"""Format computed study results as manuscript tables. No statistical inference."""
from __future__ import annotations
from typing import Any
SOURCES = ("Human", "GPT 5.6 Sol", "Claude Opus 4.8", "Gemini 3.5 Flash")

SOURCE_HEADERS = {
    "Human": "Human",
    "GPT 5.6 Sol": "GPT",
    "Claude Opus 4.8": "Claude",
    "Gemini 3.5 Flash": "Gemini",
}

PUZZLE_TYPES = (
    ("arithmetic24", "Arithmetic"),
    ("maze", "Maze"),
    ("grid_placement", "Rooks"),
    ("minesweeper_lite", "Minesweeper"),
    ("mini_sudoku", "Sudoku"),
)

def inline_table_cell(item: dict[str, Any]) -> str:
    low, high = item["ci95"]
    return "$%.2f[%.2f,%.2f]$" % (item["mean"], low, high)

def render_table_tex(data: dict[str, Any]) -> str:
    lines = [
        "% Generated normalized-entropy table.",
        "% Do not edit numerical cells by hand.",
        "\\begin{table}[H]",
        "  \\color{black}",
        "  \\captionsetup{font={color=black}}",
        "  \\caption{Normalized Shannon entropy by puzzle family. Same data pipeline as",
        "  for Fig.~\\ref{fig:tv-source-geometry}.}",
        "  \\label{tab:normalized-entropy}",
        "  \\centering",
        "  \\setlength{\\tabcolsep}{2pt}",
        "  \\begin{tabular}{@{}lrrrr@{}}",
        "    \\toprule",
        "    & \\multicolumn{4}{c}{Mean normalized entropy [95\\% CI]} \\\\",
        "    \\cmidrule(l){2-5}",
        "    Puzzle family & Human & GPT & Claude & Gemini \\\\",
        "    \\midrule",
    ]
    for puzzle_type, label in PUZZLE_TYPES:
        item = data["by_type"][puzzle_type]
        cells = [inline_table_cell(item["sources"][source]) for source in SOURCES]
        lines.append(f"    {label} & " + " & ".join(cells) + r" \\")
    mean_cells = [
        inline_table_cell(data["overall"]["sources"][source])
        for source in SOURCES
    ]
    lines.append("    \\midrule")
    lines.append("    Mean & " + " & ".join(mean_cells) + r" \\")
    lines.extend(
        [
            "    \\bottomrule",
            "  \\end{tabular}",
            "\\end{table}",
            "",
        ]
    )
    return "\n".join(lines)

def render_svd_table_tex(data: dict[str, Any]) -> str:
    projection = data["source_svd_projection"]
    family_coefficients = projection["left_singular_vectors_top_two"]
    source_scores = projection["source_scores_top_two"]
    energy_share = projection["frobenius_energy_share"]
    lines = [
        "% Generated centered entropy-profile SVD table.",
        "% Do not edit numerical cells by hand.",
        "\\begin{table}[H]",
        "  \\caption{Centered entropy-profile SVD.\\@  Each source is demeaned across",
        "  puzzle families before decomposition; percentages are shares of the",
        "  centered matrix's Frobenius energy.}",
        "  \\label{tab:entropy-svd}",
        "  \\centering",
        "  \\setlength{\\tabcolsep}{10pt}",
        "  \\begin{adjustbox}{max width=\\linewidth}",
        "  \\begin{tabular}[t]{@{}lrr@{}}",
        "    \\toprule",
        r"    \multicolumn{3}{@{}l}{\textit{Panel A: family coefficients}} \\",
        "    \\addlinespace[2pt]",
        (
            "    Puzzle family & "
            f"SV1 ({100.0 * energy_share[0]:.1f}\\%) & "
            f"SV2 ({100.0 * energy_share[1]:.1f}\\%) \\\\"
        ),
        "    \\midrule",
    ]
    for puzzle_type, label in PUZZLE_TYPES:
        coefficients = family_coefficients[puzzle_type]
        lines.append(
            f"    {label} & ${coefficients[0]:.2f}$ & ${coefficients[1]:.2f}$ \\\\"
        )
    lines.extend(
        [
            "    \\bottomrule",
            "  \\end{tabular}",
            "  \\hspace{0.07\\linewidth}",
            "  \\begin{tabular}[t]{@{}lrr@{}}",
            "    \\toprule",
            r"    \multicolumn{3}{@{}l}{\textit{Panel B: source scores}} \\",
            "    \\addlinespace[2pt]",
            r"    Source & SV1 & SV2 \\",
            "    \\midrule",
        ]
    )
    for source in SOURCES:
        scores = source_scores[source]
        lines.append(
            f"    {SOURCE_HEADERS[source]} & ${scores[0]:.2f}$ & ${scores[1]:.2f}$ \\\\"
        )
    lines.extend(
        [
            "    \\bottomrule",
            "  \\end{tabular}",
            "  \\end{adjustbox}",
            "\\end{table}",
            "",
        ]
    )
    return "\n".join(lines)
