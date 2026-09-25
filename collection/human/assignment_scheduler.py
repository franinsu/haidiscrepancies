from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from puzzles.common import read_jsonl, stable_hash
from processing.retention import participant_key, is_bad_response_record, bad_record_block_key, bad_record_exclusion_summary


HUMAN_EXP_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = HUMAN_EXP_ROOT.parent.parent
STIMULI_ROOT = PROJECT_ROOT / "data" / "stimuli"
CONFIG_PATH = HUMAN_EXP_ROOT / "config" / "study_config.json"
TYPE_ORDER = ["arithmetic24", "maze", "grid_placement", "minesweeper_lite", "mini_sudoku"]


def load_study_config(path: str | Path = CONFIG_PATH) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def participant_index(username: str) -> int:
    digits = "".join(ch for ch in username if ch.isdigit())
    return max(1, int(digits)) if digits else 1


def infer_cohort(username, requested=None):
    if requested is not None:
        if requested not in {"main", "module"}:
            raise ValueError("Only the main and module study cohorts are included")
        return requested
    return "module" if username.upper().startswith("E") else "main"


def load_main_assignment(
    username: str,
    data_path: str = str(STIMULI_ROOT / "all_puzzles.jsonl"),
    dataset_path: str = "/data/stimuli/all_puzzles.jsonl",
    cohort: str = "main",
    config_path: str | Path = CONFIG_PATH,
) -> Dict[str, Any]:
    config = load_study_config(config_path)
    rows = list(read_jsonl(data_path))
    random.Random(stable_hash({"username": username, "cohort": cohort, "assignment": "main_order"})).shuffle(rows)
    assignment = _assignment_payload(
        username=username,
        cohort=cohort,
        dataset_path=dataset_path,
        puzzle_ids=[row["id"] for row in rows],
        unit_plan=[
            {
                "unit_id": row.get("family_id", row["id"]),
                "module": "main",
                "condition": row.get("variant_type", "canonical"),
                "trial_ids": [row["id"]],
                "choice_id": row["id"],
            }
            for row in rows
        ],
        config=config,
    )
    return assignment


def load_module_assignment(
    username: str,
    trials_path: str = str(STIMULI_ROOT / "modules/all_module_trials.jsonl"),
    blocks_path: str = str(STIMULI_ROOT / "modules/module_blocks.jsonl"),
    dataset_path: str = "/data/stimuli/modules/all_module_trials.jsonl",
    cohort: str = "module",
    retained_quota: Dict[str, int] | None = None,
    config_path: str | Path = CONFIG_PATH,
    compact: bool = False,
) -> Dict[str, Any]:
    config = load_study_config(config_path)
    trials = {row["id"]: row for row in read_jsonl(trials_path)}
    blocks = read_jsonl(blocks_path)
    units = build_experimental_units(blocks, trials)
    idx = participant_index(username)
    retained_quota = retained_quota or {}

    selected: List[Dict[str, Any]] = []
    if compact:
        selected.extend(_select_compact_modules(units, idx, retained_quota))
    else:
        selected.extend(_select_simple_module("cue", units, idx, retained_quota))
        selected.extend(_select_simple_module("spatial", units, idx, retained_quota))
        selected.extend(_select_simple_module("formulation", units, idx, retained_quota))
        selected.extend(_select_transfer(units, idx, retained_quota))
        selected.extend(_select_pair(units, idx, retained_quota))
    selected = _order_module_choices(selected, idx)

    puzzle_ids = [trial_id for choice in selected for trial_id in choice["trial_ids"]]
    assignment = _assignment_payload(
        username=username,
        cohort=cohort,
        dataset_path=dataset_path,
        puzzle_ids=puzzle_ids,
        unit_plan=selected,
        config=config,
    )
    assignment["module_counts"] = dict(Counter(choice["module"] for choice in selected))
    assignment["module_trial_counts"] = {
        module: sum(len(choice["trial_ids"]) for choice in selected if choice["module"] == module)
        for module in sorted({choice["module"] for choice in selected})
    }
    return assignment


def make_assignment(
    username: str,
    cohort: str,
    retained_quota: Dict[str, int] | None = None,
) -> Dict[str, Any]:
    cohort = infer_cohort(username, cohort)
    if cohort == "module":
        return load_module_assignment(username, retained_quota=retained_quota)
    return load_main_assignment(username)


def build_experimental_units(
    blocks: Sequence[Dict[str, Any]],
    trials: Dict[str, Dict[str, Any]],
) -> Dict[str, List[Dict[str, Any]]]:
    units: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for block in blocks:
        module = block["module"]
        if module in {"cue", "spatial", "formulation"}:
            unit_id = block["block_id"]
            for trial_id in block["trial_ids"]:
                trial = trials[trial_id]
                condition = trial["module_metadata"]["condition"]
                units[f"{module}:{unit_id}"].append(
                    _choice(
                        module=module,
                        unit_id=unit_id,
                        condition=condition,
                        trial_ids=[trial_id],
                        sequence_id=block.get("sequence_id", ""),
                    )
                )
        elif module == "pair":
            unit_id = _unit_id_from_sequence(block["sequence_id"], ("_NO_PRIME", "_REL", "_CTRL"))
            units[f"{module}:{unit_id}"].append(
                _choice(module, unit_id, block["condition"], block["trial_ids"], block["sequence_id"])
            )
        elif module == "transfer":
            unit_id = _unit_id_from_sequence(block["sequence_id"], ("_PRIME", "_ALONE"))
            units[f"{module}:{unit_id}"].append(
                _choice(module, unit_id, block["condition"], block["trial_ids"], block["sequence_id"])
            )
    for choices in units.values():
        choices.sort(key=lambda item: (item["module"], item["unit_id"], item["condition"], ",".join(item["trial_ids"])))
    return dict(units)


def quota_summary(
    responses: Sequence[Dict[str, Any]],
    assignments: Sequence[Dict[str, Any]] | None = None,
    bad_threshold: int | None = None,
    bad_total_limit: int = 30,
    bad_block_limit: int = 14,
) -> Dict[str, Any]:
    if bad_threshold is not None:
        bad_total_limit = max(0, bad_threshold - 1)
    exclusion = bad_record_exclusion_summary(
        responses,
        bad_total_limit=bad_total_limit,
        bad_block_limit=bad_block_limit,
    )
    excluded = set(exclusion["excluded_users"])
    raw: Counter[str] = Counter()
    retained: Counter[str] = Counter()
    correct: Counter[str] = Counter()
    bad: Counter[str] = Counter()
    for row in responses:
        pid = str(row["puzzle_id"])
        raw[pid] += 1
        if is_bad_response_record(row):
            bad[pid] += 1
        if row.get("is_correct"):
            correct[pid] += 1
        if participant_key(row) not in excluded:
            retained[pid] += 1
    assigned: Counter[str] = Counter()
    for row in assignments or []:
        assigned[str(row["puzzle_id"])] += 1
    return {
        "excluded_users": sorted(excluded),
        "raw_by_puzzle": dict(raw),
        "retained_by_puzzle": dict(retained),
        "correct_by_puzzle": dict(correct),
        "bad_by_puzzle": dict(bad),
        "assigned_by_puzzle": dict(assigned),
    }


def retained_quota_from_response_rows(
    response_rows: Sequence[Dict[str, Any]],
    bad_threshold: int | None = None,
    bad_total_limit: int = 30,
    bad_block_limit: int = 14,
) -> Dict[str, int]:
    return quota_summary(
        response_rows,
        bad_threshold=bad_threshold,
        bad_total_limit=bad_total_limit,
        bad_block_limit=bad_block_limit,
    )["retained_by_puzzle"]










def module_condition_cells(trials_path: str = str(STIMULI_ROOT / "modules/all_module_trials.jsonl")) -> Dict[str, Dict[str, Any]]:
    rows = read_jsonl(trials_path)
    out: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        meta = row.get("module_metadata", {})
        out[row["id"]] = {
            "puzzle_id": row["id"],
            "family_id": row.get("family_id"),
            "module": meta.get("module", row.get("presentation_metadata", {}).get("module", "module")),
            "condition": meta.get("condition", row.get("variant_type")),
            "sequence_id": meta.get("sequence_id", ""),
            "sequence_position": meta.get("sequence_position"),
            "puzzle_type": row["puzzle_type"],
        }
    return out


def _assignment_payload(
    username: str,
    cohort: str,
    dataset_path: str,
    puzzle_ids: List[str],
    unit_plan: List[Dict[str, Any]],
    config: Dict[str, Any],
) -> Dict[str, Any]:
    study = config["studies"][cohort]
    payload = {
        "assignment_version": config["assignment_version"],
        "assignment_hash": stable_hash({"username": username, "cohort": cohort, "puzzle_ids": puzzle_ids, "unit_plan": unit_plan}),
        "username": username,
        "cohort": cohort,
        "dataset_path": dataset_path,
        "puzzle_ids": puzzle_ids,
        "unit_plan": unit_plan,
        "payment": {
            "posted_base_reward_usd": study["posted_base_reward_usd"],
            "correct_bonus_per_trial_usd": study["correct_bonus_per_trial_usd"],
            "max_bonus_usd": round(len(puzzle_ids) * study["correct_bonus_per_trial_usd"], 2),
            "max_total_reward_usd": round(study["posted_base_reward_usd"] + len(puzzle_ids) * study["correct_bonus_per_trial_usd"], 2),
        },
        "study": {
            "name": study["name"],
            "estimated_time_min": study["estimated_time_min"],
            "estimated_time_max": study["estimated_time_max"],
            "study_valid_window_minutes": study.get(
                "study_valid_window_minutes",
                config.get("timing", {}).get("study_valid_window_minutes", 100),
            ),
            "assumed_seconds_per_trial": study.get("assumed_seconds_per_trial", 30),
        },
        "practice": config.get("practice", {}),
        "attention_checks": config.get("attention_checks", {}),
        "completion_code": study["completion_code"],
    }
    return payload


def _choice(module: str, unit_id: str, condition: str, trial_ids: Sequence[str], sequence_id: str = "") -> Dict[str, Any]:
    return {
        "choice_id": sequence_id or trial_ids[0],
        "unit_id": unit_id,
        "module": module,
        "condition": condition,
        "trial_ids": list(trial_ids),
        "sequence_id": sequence_id,
        "trial_count": len(trial_ids),
    }


def _select_simple_module(
    module: str,
    units: Dict[str, List[Dict[str, Any]]],
    participant_idx: int,
    retained_quota: Dict[str, int],
) -> List[Dict[str, Any]]:
    selected: List[Dict[str, Any]] = []
    module_units = _module_units(units, module)
    for unit_offset, choices in enumerate(module_units):
        selected.append(_lowest_quota_choice(choices, retained_quota, participant_idx + unit_offset))
    return selected


def _select_transfer(
    units: Dict[str, List[Dict[str, Any]]],
    participant_idx: int,
    retained_quota: Dict[str, int],
) -> List[Dict[str, Any]]:
    module_units = _module_units(units, "transfer")
    no_prime_units = set(_choose_condition_units(module_units, "no_prime", target=5, retained_quota=retained_quota, seed=participant_idx * 37))
    selected = []
    for unit_idx, choices in enumerate(module_units):
        condition = "no_prime" if unit_idx in no_prime_units else "strategy_prime"
        selected.append(_choice_by_condition(choices, condition))
    return selected


def _select_pair(
    units: Dict[str, List[Dict[str, Any]]],
    participant_idx: int,
    retained_quota: Dict[str, int],
) -> List[Dict[str, Any]]:
    module_units = _module_units(units, "pair")
    no_prime_target = 3 + (participant_idx % 2)
    no_prime_units = set(_choose_condition_units(module_units, "no_prime", target=no_prime_target, retained_quota=retained_quota, seed=participant_idx * 53))
    selected: List[Dict[str, Any]] = []
    for unit_idx, choices in enumerate(module_units):
        if unit_idx in no_prime_units:
            selected.append(_choice_by_condition(choices, "no_prime"))
        else:
            selected.append(
                _lowest_quota_choice(
                    [choice for choice in choices if choice["condition"] in {"related", "unrelated_control"}],
                    retained_quota,
                    participant_idx + unit_idx,
                )
            )
    return selected


def _select_compact_modules(
    units: Dict[str, List[Dict[str, Any]]],
    participant_idx: int,
    retained_quota: Dict[str, int],
) -> List[Dict[str, Any]]:
    selected: List[Dict[str, Any]] = []
    for module in ["cue", "spatial", "formulation", "pair", "transfer"]:
        for unit_offset, choices in enumerate(_module_units(units, module)):
            selected.append(_lowest_quota_choice(choices, retained_quota, participant_idx * 101 + unit_offset))
    return selected


def _choose_condition_units(
    module_units: Sequence[List[Dict[str, Any]]],
    condition: str,
    target: int,
    retained_quota: Dict[str, int],
    seed: int,
) -> List[int]:
    candidates = []
    for unit_idx, choices in enumerate(module_units):
        choice = _choice_by_condition(choices, condition)
        score = _quota_score(choice, retained_quota)
        candidates.append((score, random.Random(seed + unit_idx).random(), unit_idx))
    candidates.sort(key=lambda item: (item[0], item[1]))
    return [unit_idx for _, _, unit_idx in candidates[:target]]


def _lowest_quota_choice(
    choices: Sequence[Dict[str, Any]],
    retained_quota: Dict[str, int],
    tie_seed: int,
) -> Dict[str, Any]:
    ranked = [
        (_quota_score(choice, retained_quota), random.Random(tie_seed + i * 997).random(), choice)
        for i, choice in enumerate(choices)
    ]
    ranked.sort(key=lambda item: (item[0], item[1], item[2]["condition"]))
    return ranked[0][2]


def _quota_score(choice: Dict[str, Any], retained_quota: Dict[str, int]) -> float:
    counts = [retained_quota.get(trial_id, 0) for trial_id in choice["trial_ids"]]
    return sum(counts) / max(1, len(counts))


def _choice_by_condition(choices: Sequence[Dict[str, Any]], condition: str) -> Dict[str, Any]:
    for choice in choices:
        if choice["condition"] == condition:
            return choice
    raise KeyError(condition)


def _module_units(units: Dict[str, List[Dict[str, Any]]], module: str) -> List[List[Dict[str, Any]]]:
    return [units[key] for key in sorted(units) if key.startswith(f"{module}:")]


def _order_module_choices(choices: Sequence[Dict[str, Any]], participant_idx: int) -> List[Dict[str, Any]]:
    groups: Dict[str, List[Dict[str, Any]]] = {key: [] for key in ["cue", "spatial", "formulation", "pair", "transfer"]}
    for choice in choices:
        groups.setdefault(choice["module"], []).append(dict(choice))
    for module, module_choices in groups.items():
        random.Random(participant_idx * 1009 + len(module)).shuffle(module_choices)
    ordered: List[Dict[str, Any]] = []
    cycle = ["cue", "spatial", "formulation", "pair", "transfer"]
    while any(groups.values()):
        for module in cycle:
            if groups.get(module):
                ordered.append(groups[module].pop(0))
    return ordered


def _interleave_by_type(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[str, List[Dict[str, Any]]] = {key: [] for key in TYPE_ORDER}
    for row in rows:
        groups.setdefault(row["puzzle_type"], []).append(row)
    out: List[Dict[str, Any]] = []
    while any(groups.values()):
        for key in TYPE_ORDER:
            if groups.get(key):
                out.append(groups[key].pop(0))
    return out


def _unit_id_from_sequence(sequence_id: str, suffixes: tuple[str, ...]) -> str:
    for suffix in suffixes:
        if sequence_id.endswith(suffix):
            return sequence_id[: -len(suffix)]
    return sequence_id.rsplit("_", 1)[0]


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Preview formal human-study assignments.")
    parser.add_argument("--username", default="E0001")
    parser.add_argument("--cohort", choices=["main", "module"], default="")
    args = parser.parse_args(list(argv) if argv is not None else None)
    assignment = make_assignment(args.username, args.cohort or infer_cohort(args.username))
    print(json.dumps(assignment, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
