"""Format computed study results as manuscript tables. No statistical inference."""

from __future__ import annotations

from typing import Any, Iterable


SOURCE_ORDER = ("Human", "GPT", "Claude", "Gemini")

PAIRS = (
    (0, 1, "Human vs GPT"),
    (0, 2, "Human vs Claude"),
    (0, 3, "Human vs Gemini"),
    (1, 2, "GPT vs Claude"),
    (1, 3, "GPT vs Gemini"),
    (2, 3, "Claude vs Gemini"),
)

PUZZLE_TYPES = (
    ("arithmetic24", "Arithmetic"),
    ("maze", "Maze"),
    ("grid_placement", "Rooks"),
    ("minesweeper_lite", "Minesweeper"),
    ("mini_sudoku", "Sudoku"),
)

METRICS = (
    ("total_variation_distance", "Total variation", "TV", "distance"),
    (
        "jensen_shannon_distance",
        "Jensen--Shannon distance",
        "JS",
        "distance",
    ),
    ("hellinger_distance", "Hellinger distance", "H", "distance"),
    (
        "log_density_correlation",
        "Log-density correlation",
        r"$\rho_{\log}$",
        "similarity",
    ),
    (
        "tie_aware_modal_agreement",
        "Tie-aware modal agreement",
        "Modal",
        "similarity",
    ),
)

def clean_number(value: float) -> float:
    return 0.0 if abs(value) < 0.005 else value

def result(value: float, values: Iterable[float]) -> str:
    lower, upper = (clean_number(float(entry)) for entry in values)
    return (
        "$"
        + f"{clean_number(value):.2f}[{lower:.2f},{upper:.2f}]"
        + "$"
    )

def metric_block(stats: dict[str, Any]) -> dict[str, Any]:
    comparison = stats["distribution_metric_comparison_primary"]
    if normalized_order(comparison["source_order"]) != SOURCE_ORDER:
        raise ValueError(
            "Unexpected common source order: "
            f"{comparison['source_order']}; expected {list(SOURCE_ORDER)}"
        )
    if comparison.get("condition", {}).get("effort") != "low":
        raise ValueError("Metric comparison is not the low-effort condition")
    if comparison.get("condition", {}).get("prompt") != "plain":
        raise ValueError("Metric comparison is not the plain-prompt condition")
    return comparison

def check_item(item: dict[str, Any], n_puzzles: int) -> None:
    if normalized_order(item["order"]) != SOURCE_ORDER:
        raise ValueError(f"Unexpected source order: {item['order']}")
    if int(item["n_puzzles"]) != n_puzzles:
        raise ValueError(
            f"Unexpected puzzle count: {item['n_puzzles']}; expected {n_puzzles}"
        )

def normalized_order(order: Iterable[str]) -> tuple[str, ...]:
    """Accept the former ``ChatGPT`` display label on old artifacts."""
    return tuple("GPT" if source == "ChatGPT" else source for source in order)

def direction_label(kind: str) -> str:
    if kind == "distance":
        return "lower is closer"
    if kind == "similarity":
        return "higher is closer"
    raise ValueError(f"Unknown metric kind: {kind}")

def family_rows(
    metric_label: str,
    kind: str,
    global_item: dict[str, Any],
    by_type: dict[str, dict[str, Any]],
) -> list[str]:
    check_item(global_item, 100)
    lines = [
        "    \\addlinespace[0.35em]",
        (
            "    \\multicolumn{7}{@{}l}{\\textit{"
            f"{metric_label} ({direction_label(kind)})"
            "}} \\\\"
        ),
    ]
    for left, right, label in PAIRS:
        global_ci = global_item["ci95_puzzle_matrix"][left][right]
        if global_ci is None:
            raise ValueError(f"Missing Mean interval for {metric_label}: {label}")
        entries = [result(float(global_item["matrix"][left][right]), global_ci)]
        for puzzle_type, _ in PUZZLE_TYPES:
            item = by_type[puzzle_type]
            check_item(item, 20)
            ci = item["ci95_puzzle_matrix"][left][right]
            if ci is None:
                raise ValueError(
                    f"Missing family interval for {metric_label}, "
                    f"{puzzle_type}: {label}"
                )
            entries.append(result(float(item["matrix"][left][right]), ci))
        lines.append(f"    {label} & " + " & ".join(entries) + r" \\")
    return lines

def render(stats: dict[str, Any]) -> str:
    metrics = metric_block(stats)['metrics']
    lines = [
        '% Alternative measures by family; global and concordance tables removed.',
        r'\begin{table}[!htbp]',
        r'  \caption{Alternative distribution measures by puzzle family in the primary low-effort, plain-prompt condition; corresponding TV estimates are shown in Figure~\ref{fig:tv-source-geometry}. Each cell is the metric estimate followed by its pointwise 95\% puzzle-bootstrap interval; the Mean column weights the five family estimates equally.}',
        r'  \label{tab:metric-comparison-family}',r'  \centering',r'  \small',
        r'  \setlength{\tabcolsep}{0.8pt}',r'  \renewcommand{\arraystretch}{0.94}',
        r'  \begin{adjustbox}{max width=\linewidth}',r'\begin{tabular}{@{}lcccccc@{}}',r'    \toprule',
        r'    Comparison & Mean & Arithmetic & Maze & Rooks & Minesweeper & Sudoku \\',r'    \midrule']
    for key,label,_,kind in METRICS[1:]:
        lines.extend(family_rows(label,kind,metrics[key]['global'],metrics[key]['by_type']))
    lines.extend([r'    \bottomrule',r'  \end{tabular}',r'\end{adjustbox}',r'\end{table}'])
    return '\n'.join(lines)+'\n'
