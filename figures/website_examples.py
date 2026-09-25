"""Export puzzle displays and aggregate answer counts for the companion website.

Run after the standard analysis: python -m figures.website_examples
Reads frozen study materials and aggregate statistics only, never human records.
"""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FAMILIES = [
    ("arithmetic24", "Arithmetic"), ("maze", "Maze"),
    ("grid_placement", "Rooks"), ("minesweeper_lite", "Minesweeper"),
    ("mini_sudoku", "Sudoku"),
]
MODULES = [
    ("cue", "Highlighting"), ("spatial", "Reflection"),
    ("formulation", "Number ordering"), ("pair", "Preceding puzzle"),
    ("transfer", "Strategy primer"),
]
SOURCE_KEYS = ["Human", "GPT 5.6 Sol|low|plain", "Claude Opus 4.8|low|plain", "Gemini 3.5 Flash|low|plain"]
SOURCES = [
    {"label": "Human", "color": "#8C1515"},
    {"label": "ChatGPT", "color": "#482878"},
    {"label": "Claude", "color": "#31688E"},
    {"label": "Gemini", "color": "#21918C"},
]
ARM_LABELS = {
    "cue_half_a": "Highlight A", "cue_half_b": "Highlight B",
    "mirror_lr": "Left–right", "mirror_tb": "Top–bottom", "mirror_lr_tb": "Both reflections",
    "reversed": "Reversed order", "related": "Related first puzzle",
    "unrelated_control": "Unrelated first puzzle", "strategy_prime": "With primer",
}


def read_jsonl(path: str) -> list[dict]:
    return [json.loads(line) for line in (ROOT / path).read_text().splitlines() if line]


def display_puzzle(row: dict, classes: list[str], counts: list[list[int]]) -> dict:
    """Allowlist display fields, joining solutions by abstract ID rather than position."""
    raw = row["machine_readable_instance"]
    instance = {key: raw[key] for key in ("grid", "board", "m", "target", "target_kind", "digit", "box_rows", "box_cols") if key in raw}
    if "numbers" in raw:
        instance["numbers"] = raw.get("display_numbers", raw["numbers"])
    presentation = row.get("presentation_metadata", {})
    cue = raw.get("visual_cue") or raw.get("irrelevant_cue") or presentation.get("visual_cue") or presentation.get("irrelevant_cue") or {}
    solutions = {solution["abstract_solution_id"]: solution for solution in row["solutions"]}
    assert len(classes) == len(set(classes)) and set(classes) == set(solutions), row["id"]
    assert len(counts) == 4
    assert all(len(values) == len(classes) and all(isinstance(n, int) and n >= 0 for n in values) for values in counts)
    return {
        "id": row["id"], "family": row["puzzle_type"], "instance": instance,
        "highlights": cue.get("coordinates", []),
        "solutions": [
            {"id": key, **{field: solutions[key][field] for field in ("expression", "coordinate", "coordinates", "moves") if field in solutions[key]}}
            for key in classes
        ],
        "counts": counts,
    }


def build_examples() -> dict:
    main_rows = read_jsonl("data/stimuli/all_puzzles.jsonl")
    module_rows = read_jsonl("data/stimuli/modules/all_module_trials.jsonl")
    main = json.loads((ROOT / "intermediate/statistics/main_solution_probabilities.json").read_text())
    modules = json.loads((ROOT / "intermediate/statistics/perturbation_counts.json").read_text())
    assert main["sources"] == [source["label"] for source in SOURCES]
    main_stats = {row["id"]: row for row in main["puzzles"]}
    puzzles = {}
    for row in main_rows:
        stats = main_stats[row["id"]]
        assert [sum(values) for values in stats["counts"]] == stats["valid_n"]
        puzzles[row["id"]] = display_puzzle(row, stats["classes"], stats["counts"])
    for row in module_rows:
        stats = modules["trials"][row["id"]]
        classes = sorted(stats["ids"])
        counts = []
        for source in SOURCE_KEYS:
            values = stats["counts"].get(source, {})
            assert set(values) <= set(classes), row["id"]
            counts.append([values.get(key, 0) for key in classes])
        puzzles[row["id"]] = display_puzzle(row, classes, counts)

    rows_by_id = {row["id"]: row for row in module_rows}
    primers = {row["module_metadata"]["sequence_id"]: row["id"] for row in module_rows
               if row.get("module_metadata", {}).get("sequence_position") == 1}
    groups = {}
    for comparison in modules["comparisons"]:
        module = comparison["module"]
        group = groups.setdefault(comparison["block"], {
            "id": comparison["block"], "family": comparison["family"], "module": module,
            "variants": [{"label": "Target alone" if module in {"pair", "transfer"} else "Original", "puzzle": comparison["original"]}],
        })
        variant = {"label": ARM_LABELS[comparison["arm"]], "puzzle": comparison["perturbed"]}
        if comparison.get("preceding"):
            variant["preceding"] = comparison["preceding"]
        if module == "transfer":
            sequence = rows_by_id[comparison["perturbed"]]["module_metadata"]["sequence_id"]
            variant["preceding"] = primers[sequence]
        original_ids = [s["id"] for s in puzzles[comparison["original"]]["solutions"]]
        changed_ids = [s["id"] for s in puzzles[comparison["perturbed"]]["solutions"]]
        assert original_ids == changed_ids, comparison["block"]
        group["variants"].append(variant)
    assert len(puzzles) == 270 and len(groups) == 50
    return {
        "sources": SOURCES,
        "condition": "Models use low effort and plain prompts. Counts include correct/valid answers only.",
        "puzzles": puzzles,
        "families": [{"id": key, "label": label, "examples": [row["id"] for row in main_rows if row["puzzle_type"] == key]} for key, label in FAMILIES],
        "modules": [{"id": key, "label": label, "examples": [group for group in groups.values() if group["module"] == key]} for key, label in MODULES],
    }


def main() -> None:
    target = ROOT / "website/assets/examples.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(build_examples(), ensure_ascii=False, separators=(",", ":")) + "\n")
    print(f"Exported 100 main puzzles and 50 stimulus groups ({target.stat().st_size:,} bytes).")


if __name__ == "__main__":
    main()
