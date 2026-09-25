"""Format computed study results as manuscript tables. No statistical inference."""
from __future__ import annotations
from typing import Any
PROVIDERS = ("GPT 5.6 Sol", "Claude Opus 4.8", "Gemini 3.5 Flash")

PROVIDER_HEADERS = {
    "GPT 5.6 Sol": "ChatGPT",
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

CONTRAST_LABELS = (
    ("persona_minus_plain_low", r"\shortstack[l]{Persona $-$ plain\\(low effort)}"),
    ("persona_minus_plain_medium", r"\shortstack[l]{Persona $-$ plain\\(medium effort)}"),
    ("medium_minus_low_plain", r"\shortstack[l]{Medium $-$ low\\(plain prompt)}"),
    ("medium_minus_low_persona", r"\shortstack[l]{Medium $-$ low\\(persona prompt)}"),
)

GLOBAL_PAIRS = (
    (0, 1, "Uniform vs Human"),
    (0, 2, "Uniform vs ChatGPT"),
    (0, 3, "Uniform vs Claude"),
    (0, 4, "Uniform vs Gemini"),
    (1, 2, "Human vs ChatGPT"),
    (1, 3, "Human vs Claude"),
    (1, 4, "Human vs Gemini"),
    (2, 3, "ChatGPT vs Claude"),
    (2, 4, "ChatGPT vs Gemini"),
    (3, 4, "Claude vs Gemini"),
)

EXPECTED_BOOTSTRAP = {
    "method": "design-stratified percentile cluster bootstrap",
    "confidence": 0.95,
    "repeats": 5000,
    "seed": 0,
    "resampling_unit": "core-battery puzzle",
    "strata": ["puzzle_type"],
    "n_clusters": 100,
    "n_puzzles": 100,
}

EXPECTED_GLOBAL_RESPONSE_BOOTSTRAP = {
    "method": "nonparametric percentile bootstrap",
    "confidence": 0.95,
    "repeats": 5000,
    "seed": 0,
    "scope": "primary low-effort plain-prompt Table 1",
    "coverage": "pointwise (marginal), not simultaneous",
    "estimand": (
        "equal-weight mean of same-puzzle total-variation distances "
        "over the balanced 100-puzzle core battery"
    ),
    "trial_only": {
        "puzzles": "fixed observed 100-puzzle core battery",
        "human_resampling_unit": (
            "retained participant response profile across all 100 puzzles"
        ),
        "human_sample_size": 104,
        "human_filter": "correct responses retained after resampling",
        "model_resampling_unit": (
            "original API request within model-source--puzzle cell"
        ),
        "model_sample_size": "100 requests per model-source--puzzle cell",
        "model_filter": "valid responses retained after resampling",
        "sharing": (
            "one participant draw and one source--puzzle distribution "
            "shared across all pairwise comparisons"
        ),
        "uniform_reference": "fixed",
    },
    "joint": {
        "puzzle_resampling": (
            "20 whole puzzles with replacement within each puzzle family"
        ),
        "response_resampling": "same trial resampling as trial-only",
        "repeated_puzzle_occurrences": (
            "weighted by puzzle multiplicity; trial draw reused"
        ),
        "uniform_reference": "fixed",
    },
    "zero_denominator": "redraw entire bootstrap replicate",
}

def estimate_cell(estimate: float) -> str:
    estimate = 0.0 if abs(estimate) < 0.005 else estimate
    return rf"${estimate:.2f}$"

def interval_cell(interval: list[float]) -> str:
    interval = [0.0 if abs(value) < 0.005 else value for value in interval]
    return rf"$[{interval[0]:.2f},{interval[1]:.2f}]$"

def stacked_percentage_point_cell(estimate: float, interval: list[float]) -> str:
    """Emphasize estimates whose unrounded, pointwise interval excludes zero."""
    excludes_zero = interval[0] > 0 or interval[1] < 0
    estimate *= 100
    interval = [100 * value for value in interval]
    estimate = 0.0 if abs(estimate) < 0.005 else estimate
    interval = [0.0 if abs(value) < 0.005 else value for value in interval]
    point = f"{estimate:.2f}"
    if excludes_zero:
        point = rf"\mathbf{{{point}}}"
    return (
        rf"\shortstack{{${point}$\\[1pt]"
        rf"{{\scriptsize$[{interval[0]:.2f},{interval[1]:.2f}]$}}}}"
    )

def validate_bootstrap_contract(stats: dict[str, Any]) -> None:
    bootstrap = stats["tv_bootstrap"]
    if bootstrap != EXPECTED_BOOTSTRAP:
        raise ValueError(f"Unexpected TV bootstrap contract: {bootstrap}")

def validate_global_response_bootstrap_contract(stats: dict[str, Any]) -> None:
    bootstrap = stats["global_tv_response_bootstrap"]
    if bootstrap != EXPECTED_GLOBAL_RESPONSE_BOOTSTRAP:
        raise ValueError(
            f"Unexpected global-TV response bootstrap contract: {bootstrap}"
        )

def render_global(stats: dict[str, Any]) -> str:
    validate_bootstrap_contract(stats)
    validate_global_response_bootstrap_contract(stats)
    global_tv = stats["global_tv_primary"]
    expected_order = ["Uniform", "Human", *PROVIDERS]
    if global_tv["order"] != expected_order or int(global_tv["n_puzzles"]) != 100:
        raise ValueError("Unexpected global-TV source order or puzzle count")

    lines = [
        "% Generated by figure_reproduction_2026-08-25/make_tv_tables.py.",
        "% Do not edit the numerical cells by hand.",
        "\\begin{table}[H]",
        "  \\color{black}",
        "  \\captionsetup{font+={color=black}}",
        "  \\caption{Pairwise mean total-variation distance across the 100-puzzle",
        "  battery.  Uniform assigns equal mass to a puzzle's valid solution classes.}",
        "  \\label{tab:tv-global}",
        "  \\centering",
        "  \\setlength{\\tabcolsep}{4pt}",
        "  \\begin{tabular}{@{}lrrrr@{}}",
        "    \\toprule",
        "    & & \\multicolumn{3}{c}{95\\% bootstrap CI} \\\\",
        "    \\cmidrule(lr){3-5}",
        "    Comparison & Mean TV & Puzzles & Trials & Both \\\\",
        "    \\midrule",
    ]
    for pair_index, (left, right, label) in enumerate(GLOBAL_PAIRS):
        if pair_index in (4, 7):
            lines.append("    \\midrule")
        puzzle_interval = global_tv["ci95_puzzle_matrix"][left][right]
        trial_interval = global_tv["ci95_trial_matrix"][left][right]
        joint_interval = global_tv["ci95_joint_matrix"][left][right]
        if any(
            interval is None
            for interval in (puzzle_interval, trial_interval, joint_interval)
        ):
            raise ValueError(f"Missing global-TV interval for {left}, {right}")
        lines.append(
            f"    {label} & "
            + estimate_cell(float(global_tv["matrix"][left][right]))
            + " & "
            + interval_cell(puzzle_interval)
            + " & "
            + interval_cell(trial_interval)
            + " & "
            + interval_cell(joint_interval)
            + r" \\"
        )
    lines.extend([
        "    \\bottomrule",
        "  \\end{tabular}",
        "\\end{table}",
        "",
    ])
    return "\n".join(lines)

def render_condition_contrasts(stats: dict[str, Any]) -> str:
    """Render family-resolved model--Human TV condition contrasts."""
    validate_bootstrap_contract(stats)
    lines = [
        "% Generated by analysis/wrangling/make_tv_tables.py --contrast-out.",
        "% Main-text effort--prompt contrasts; do not edit numerical cells by hand.",
        "% Contrasts use separate condition cells; no response rows are pooled.",
        "\\begin{table}[H]",
        "  \\caption{Condition contrasts in model--Human mean TV.",
        "  Estimates and 95\\% paired-puzzle bootstrap intervals are multiplied",
        "  by 100 (Section~\\ref{supp:puzzle-bootstrap}). Positive values indicate",
        "  greater distance from Human; Mean weights the five families equally.",
        "  Bold estimates have intervals excluding zero. Intervals are pointwise",
        "  and unadjusted for multiple comparisons.}",
        "  \\label{tab:tv-condition-contrasts}",
        "  \\centering",
        "  \\small",
        "  \\setlength{\\tabcolsep}{3pt}",
        "  \\renewcommand{\\arraystretch}{1.15}",
        "  \\begin{tabular*}{\\linewidth}{@{\\extracolsep{\\fill}}lcccccc@{}}",
        "    \\toprule",
        "    Contrast & Mean & Arithmetic & Maze & Rooks & Minesweeper & Sudoku \\\\",
        "    \\midrule",
    ]
    for provider_index, provider in enumerate(PROVIDERS):
        if provider_index:
            lines.append("    \\midrule")
        lines.append(
            r"    \multicolumn{7}{@{}l}{\textbf{" + PROVIDER_HEADERS[provider] + r"}} \\"
        )
        lines.append(r"    \addlinespace[2pt]")
        for contrast, label in CONTRAST_LABELS:
            item = stats["tv_condition_contrasts"][provider][contrast]
            if int(item["n_puzzles"]) != 100:
                raise ValueError(
                    f"TV contrast does not use 100 puzzles: {provider}, {contrast}"
                )
            cells = [
                stacked_percentage_point_cell(
                    float(item["estimate"]), item["ci95"]
                )
            ]
            for puzzle_type, _ in PUZZLE_TYPES:
                family_item = item["by_type"][puzzle_type]
                if int(family_item["n_puzzles"]) != 20:
                    raise ValueError(
                        f"TV contrast family does not use 20 puzzles: "
                        f"{provider}, {contrast}, {puzzle_type}"
                    )
                cells.append(
                    stacked_percentage_point_cell(
                        float(family_item["estimate"]), family_item["ci95"]
                    )
                )
            lines.append(
                f"    {label} & "
                + " & ".join(cells)
                + r" \\[4pt]"
            )
    lines.extend([
        "    \\bottomrule",
        "  \\end{tabular*}",
        "\\end{table}",
        "",
    ])
    return "\n".join(lines)

def result_cell(estimate: float, interval: list[float]) -> str:
    estimate = 0.0 if abs(estimate) < 0.005 else estimate
    interval = [0.0 if abs(value) < 0.005 else value for value in interval]
    return rf"${estimate:.2f}[{interval[0]:.2f},{interval[1]:.2f}]$"
