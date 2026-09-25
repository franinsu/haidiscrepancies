"""Export only plotted aggregates for the hover-responsive companion figures.

Run after rendering the analysis: python -m figures.website_figures
No participant records or individual model responses are read.
"""

from itertools import combinations
import json
from pathlib import Path

import numpy as np

from analysis.geometry import (
    FAMILY_ORDER,
    centered_metric_kernel,
    eigencoordinates,
    family_geometry_from_metric_matrices,
    orient_source_coordinates,
)

ROOT = Path(__file__).resolve().parents[1]
SOURCES = [
    {"name": "Human", "label": "Human", "color": "#8C1515"},
    {"name": "GPT 5.6 Sol", "label": "ChatGPT", "color": "#482878"},
    {"name": "Claude Opus 4.8", "label": "Claude", "color": "#31688E"},
    {"name": "Gemini 3.5 Flash", "label": "Gemini", "color": "#21918C"},
]
FAMILIES = [
    ("arithmetic24", "Arithmetic"), ("maze", "Maze"),
    ("grid_placement", "Rooks"), ("minesweeper_lite", "Minesweeper"),
    ("mini_sudoku", "Sudoku"),
]
ALTERNATIVE_MEASURES = [
    ("jensen_shannon_distance", "Jensen–Shannon distance", "lower",
     "Square root of Jensen–Shannon divergence, using base-2 logarithms and empirical solution proportions."),
    ("hellinger_distance", "Hellinger distance", "lower",
     "Euclidean distance between square-root solution probabilities, divided by the square root of two."),
    ("log_density_correlation", "Log-density correlation", "higher",
     "Within-puzzle Pearson correlation of log probabilities, after adding half a count to every valid solution."),
    ("tie_aware_modal_agreement", "Tie-aware modal agreement", "higher",
     "Fraction of puzzles where the sources share at least one most-frequent solution; all ties are retained."),
]


def read(name):
    return json.loads((ROOT / "results/source_data" / f"{name}.json").read_text())


def alternative_geometry(metric, scopes):
    """Embed frozen mean matrices with the study's shared source/family rules."""
    assert [family for family, _ in FAMILIES] == list(FAMILY_ORDER)
    matrices = np.asarray([scope["matrix"] for scope in scopes], dtype=float)
    assert matrices.shape == (6, len(SOURCES), len(SOURCES))
    lower, upper = metric["metadata"]["range"]
    assert np.all(np.isfinite(matrices))
    assert np.min(matrices) >= lower - 1e-12
    assert np.max(matrices) <= upper + 1e-12
    kind = metric["metadata"]["kind"]
    source_kernel = centered_metric_kernel(matrices[0], kind=kind)["kernel"]
    coordinates, eigenvalues, shares = eigencoordinates(source_kernel)
    coordinates = orient_source_coordinates(
        coordinates, tuple(source["name"] for source in SOURCES)
    )
    family = family_geometry_from_metric_matrices(
        matrices[1:], kind=kind,
        method="CKA of full PSD-projected family kernels from frozen mean metric matrices",
    )
    diagnostics = family["kernel_projection_diagnostics"]
    return {
        "source": {
            "coordinates": coordinates.tolist(),
            "shares": shares.tolist(),
            "source_indexes": list(range(1, len(SOURCES) + 1)),
            "squared_dissimilarity": "D²" if kind == "distance" else "2(1−S)",
            "distance_description": (
                "Mean pairwise distance" if kind == "distance"
                else "Square root of 2(1−S), where S is the mean pairwise similarity"
            ),
            "negative_eigenvalue_mass": float(-np.sum(eigenvalues[eigenvalues < -1e-12])),
        },
        "family": {
            "coordinates": family["coordinates"].tolist(),
            "shares": family["shares"].tolist(),
            "projection": "Full PSD projection before CKA; negative eigenvalues clipped to zero",
            "projected_families": [
                label for label, diagnostic in diagnostics.items()
                if diagnostic["projection_applied"]
            ],
            "negative_eigenvalue_mass_discarded": {
                label: diagnostic["negative_eigenvalue_mass_discarded"]
                for label, diagnostic in diagnostics.items()
            },
        },
    }


def alternative_measures():
    """Copy full-precision aggregate estimates underlying the supplementary table."""
    stats = json.loads((ROOT / "intermediate/statistics/figure_statistics.json").read_text())
    comparison = stats["distribution_metric_comparison_primary"]
    source_order = [source["label"] for source in SOURCES]
    assert comparison["source_order"] == source_order
    assert comparison["condition"]["effort"] == "low"
    assert comparison["condition"]["prompt"] == "plain"
    measures = []
    for key, label, direction, description in ALTERNATIVE_MEASURES:
        metric = comparison["metrics"][key]
        scopes = [metric["global"]] + [metric["by_type"][family] for family, _ in FAMILIES]
        for scope, count in zip(scopes, [100, 20, 20, 20, 20, 20]):
            assert scope["order"] == source_order
            assert scope["n_puzzles"] == count
        pairs = []
        for a, b in combinations(range(4), 2):
            values = [{"mean": scope["matrix"][a][b],
                       "ci95": scope["ci95_puzzle_matrix"][a][b]} for scope in scopes]
            assert all(value["ci95"] is not None for value in values)
            # TV puts Uniform first; keep the website's shared source indexes.
            pairs.append(dict(a=a + 1, b=b + 1, **values[0], families=values[1:]))
        measures.append(dict(id=key, label=label, direction=direction,
                             domain=metric["metadata"]["range"],
                             description=description, pairs=pairs,
                             geometry=alternative_geometry(metric, scopes)))
    return measures


def build():
    entropy = read("entropy_profile")
    tv = read("tv_source_geometry")
    effects = read("fig_presentation_context")
    difficulty = read("fig5_relative_difficulty")
    family_difficulty = read("effort_difficulty_by_type")
    prompting = read("fig6_effort_prompting")
    correlations = json.loads((ROOT / "intermediate/statistics/effort_difficulty_stats.json").read_text())["correlations"]["results"]
    family_index = {key: i for i, (key, _) in enumerate(FAMILIES)}
    assert len(set(difficulty["puzzle_ids"])) == 100
    summary = {
        "metric": "Mean normalized Shannon entropy",
        "condition": "Low effort, plain prompt; correct/valid answers only",
        "uncertainty": "95% percentile intervals; 5,000 whole-puzzle resamples within family",
        "sources": SOURCES,
        "families": [],
    }
    for key, label in FAMILIES:
        values = []
        for source in SOURCES:
            matches = [r for r in entropy["summary"] if r["family"] == key and r["source"] == source["name"]]
            assert len(matches) == 1
            values.append({k: matches[0][k] for k in ("mean", "ci95")})
        summary["families"].append(dict(id=key, label=label, n_puzzles=20, values=values))
    source_labels = ["Uniform"] + [s["label"] for s in SOURCES]
    pairs = []
    for index, (a, b) in enumerate(combinations(range(5), 2)):
        family_values = []
        for _, family in FAMILIES:
            matches = [r for r in tv["family_intervals"] if r["row"] == index and r["family"] == family]
            assert len(matches) == 1
            family_values.append({k: matches[0][k] for k in ("mean", "ci95")})
        pairs.append(dict(a=a, b=b, mean=tv["matrix"][a][b], ci95=tv["ci95"][a][b], families=family_values))
    assert [r["source"] for r in entropy["density"]["sources"]] == [s["name"] for s in SOURCES]
    return summary, {
        "sources": SOURCES,
        "families": [label for _, label in FAMILIES],
        "density": {
            **{k: entropy["density"][k] for k in ("grid", "bandwidth", "method")},
            "curves": [r["density"] for r in entropy["density"]["sources"]],
        },
        "tv": {"sources": source_labels, "pairs": pairs, "geometry": tv["geometry"]},
        "alternative_measures": alternative_measures(),
        "effects": {key: [{k: r[k] for k in ("metric" if key == "mi" else "module", "source", "estimate", "ci95")}
                          for r in effects[key]] for key in ("tv", "gain", "mi")},
        "difficulty": {
            "density": [{k: difficulty["density_curves"][source["name"]][k] for k in ("grid", "density", "bandwidth")} for source in SOURCES],
            "family_means": [{k: r[k] for k in ("source", "family", "mean", "ci95")} for r in family_difficulty["points"]],
            "puzzles": [{"id": pid, "family": family_index[difficulty["families"][i]],
                         "z": [difficulty["z_scores"][source["name"]][i] for source in SOURCES]}
                        for i, pid in enumerate(difficulty["puzzle_ids"])],
            "correlations": [{"source": source["name"],
                              **{kind: {k: correlations[f"Human|{source['name']}"][kind][k] for k in ("estimate", "ci95")}
                                 for kind in ("mean", "pooled")}} for source in SOURCES[1:]],
        },
        "prompting": {
            "geometry": {k: prompting[k] for k in ("coordinates", "source_order", "shares")},
            "levels": [{k: r[k] for k in ("provider", "condition", "estimate", "ci95")} for r in prompting["levels"]],
            "shifts": [{"condition": r["conditions"][1], **{k: r[k] for k in ("provider", "estimate", "ci95")}}
                       for r in prompting["condition_pairs"]],
            "uniform_reference": {k: prompting["uniform_reference"][k] for k in ("estimate", "ci95")},
        },
        "context_contrasts": [{k: r[k] for k in ("comparison", "source", "context", "estimate_bits", "ci95_low", "ci95_high")}
                              for r in read("context_mi_contrasts")],
    }


def main():
    summary, figures = build()
    target = ROOT / "website/assets"
    target.mkdir(parents=True, exist_ok=True)
    for name, data in [("entropy-summary", summary), ("figure-data", figures)]:
        (target / f"{name}.json").write_text(json.dumps(data, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n")
    print("Exported aggregate entropy, TV, alternative measures, context, difficulty and prompting figure values.")


if __name__ == "__main__":
    main()
