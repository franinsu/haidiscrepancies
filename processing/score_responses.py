from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from processing.response_parser import score_raw_answer, split_sequence_response
from puzzles.common import read_jsonl, write_jsonl


def score_responses(
    data_path: str,
    responses_path: str,
    out_path: str,
    catalog_path: str = "data/stimuli/solution_catalog.jsonl",
    skip_unknown_puzzle_ids: bool = False,
) -> List[Dict[str, Any]]:
    puzzles, catalog = load_scoring_inputs(data_path, responses_path, out_path, catalog_path)
    responses = read_jsonl(responses_path)
    scored: List[Dict[str, Any]] = []
    response_index = 0
    for response in responses:
        expanded = _expand_response_row(response)
        for response_part in expanded:
            if skip_unknown_puzzle_ids and response_part.get("puzzle_id") not in puzzles:
                continue
            response_index += 1
            scored.append(
                _score_one_response(
                    response_part,
                    response_index=response_index,
                    puzzles=puzzles,
                    catalog=catalog,
                )
            )
    write_jsonl(out_path, scored)
    return scored


def load_scoring_inputs(data_path, responses_path, out_path, catalog_path):
    output = Path(out_path)
    for source_name, source in (("puzzle data", data_path), ("solution catalog", catalog_path), ("raw responses", responses_path)):
        source_path = Path(source)
        if output.resolve() == source_path.resolve() or (
            output.exists() and source_path.exists() and output.samefile(source_path)
        ):
            raise ValueError(f"Scored output must not overwrite source {source_name}: {source}")
    puzzles = {row["id"]: row for row in read_jsonl(data_path)}
    catalog = _load_catalog(catalog_path)
    for puzzle in puzzles.values():
        for solution in puzzle["solutions"]:
            solution_id = solution["solution_id"]
            entry = catalog.get(solution_id)
            expected = solution.get("abstract_solution_id")
            if not expected or entry is None or entry.get("abstract_solution_id") != expected:
                raise ValueError(f"Missing or inconsistent solution catalog mapping for {solution_id}")
    return puzzles, catalog


def _score_one_response(
    response: Dict[str, Any],
    response_index: int,
    puzzles: Dict[str, Dict[str, Any]],
    catalog: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    puzzle_id = response.get("puzzle_id")
    raw_answer = response.get("raw_answer", response.get("answer", ""))
    raw_response = response.get("raw_response", raw_answer)
    if _is_timeout_or_give_up(response):
        parsed = _bad_human_outcome_parse(response)
    elif puzzle_id not in puzzles:
        parsed = {
            "is_valid": False,
            "invalid_reason": "unknown_puzzle_id",
            "solution_id": None,
            "parsed_answer": None,
            "canonical_answer": None,
            "parser_version": None,
            "extracted_answer": None,
            "parse_confidence": "low",
            "requires_manual_review": True,
            "parse_notes": "Puzzle ID was not found in the dataset.",
        }
    else:
        parsed = score_raw_answer(puzzles[puzzle_id], raw_answer, raw_response=raw_response)
    solution_meta = catalog.get(parsed.get("solution_id"), {})
    puzzle_meta = puzzles.get(puzzle_id, {})
    return {
        **response,
        "response_index": response.get("response_index", response_index),
        "raw_answer": raw_answer,
        "raw_response": raw_response,
        "is_valid": parsed["is_valid"],
        "invalid_reason": parsed["invalid_reason"],
        "solution_id": parsed["solution_id"],
        "abstract_solution_id": solution_meta.get("abstract_solution_id"),
        "solution_cluster": solution_meta.get("solution_cluster"),
        "solution_features": solution_meta.get("solution_features"),
        "family_id": response.get("family_id") or puzzle_meta.get("family_id"),
        "base_puzzle_id": response.get("base_puzzle_id") or puzzle_meta.get("base_puzzle_id"),
        "variant_type": response.get("variant_type") or puzzle_meta.get("variant_type"),
        "module": response.get("module") or _puzzle_module(puzzle_meta),
        "condition": response.get("condition") or _puzzle_condition(puzzle_meta),
        "module_role": response.get("module_role") or puzzle_meta.get("module_metadata", {}).get("role"),
        "parser_version": parsed["parser_version"],
        "extracted_answer": parsed["extracted_answer"],
        "parse_confidence": parsed["parse_confidence"],
        "requires_manual_review": parsed["requires_manual_review"],
        "parse_notes": parsed["parse_notes"],
        "parsed_answer": parsed["parsed_answer"],
        "canonical_answer": parsed["canonical_answer"],
    }


def _expand_response_row(response: Dict[str, Any]) -> List[Dict[str, Any]]:
    if response.get("input_kind") != "sequence":
        return [response]
    trial_ids = _split_ids(response.get("trial_ids") or response.get("puzzle_id") or "")
    if len(trial_ids) <= 1:
        return [response]
    raw_response = response.get("raw_response", response.get("raw_answer", ""))
    split = split_sequence_response(raw_response, len(trial_ids))
    rows: List[Dict[str, Any]] = []
    labels = ["A", "B", "C", "D"]
    for idx, trial_id in enumerate(trial_ids):
        part = dict(response)
        part["sequence_parent_puzzle_id"] = response.get("puzzle_id")
        part["puzzle_id"] = trial_id
        part["raw_answer"] = split["answers"][idx] if idx < len(split["answers"]) else ""
        part["raw_response"] = raw_response
        part["sequence_position"] = idx + 1
        part["sequence_label"] = labels[idx] if idx < len(labels) else str(idx + 1)
        part["sequence_parse_confidence"] = split["parse_confidence"]
        part["sequence_parse_notes"] = split["parse_notes"]
        rows.append(part)
    return rows


def _load_catalog(path: str) -> Dict[str, Dict[str, Any]]:
    if not path or not os.path.exists(path):
        raise FileNotFoundError(f"Solution catalog is required: {path}")
    rows = read_jsonl(path)
    catalog = {row["solution_id"]: row for row in rows}
    if len(catalog) != len(rows):
        raise ValueError("Solution catalog contains duplicate solution IDs")
    return catalog


def _split_ids(value: Any) -> List[str]:
    return [part.strip() for part in str(value).split(";") if part.strip()]


def _puzzle_module(puzzle: Dict[str, Any]) -> str:
    return (
        puzzle.get("module_metadata", {}).get("module")
        or puzzle.get("presentation_metadata", {}).get("module")
        or puzzle.get("features", {}).get("module")
        or "main"
    )


def _puzzle_condition(puzzle: Dict[str, Any]) -> str:
    return (
        puzzle.get("module_metadata", {}).get("condition")
        or puzzle.get("features", {}).get("module_condition")
        or puzzle.get("variant_type")
        or "canonical"
    )


def _is_timeout_or_give_up(response: Dict[str, Any]) -> bool:
    return bool(response.get("timeout") or response.get("gave_up") or response.get("skipped"))


def _bad_human_outcome_parse(response: Dict[str, Any]) -> Dict[str, Any]:
    if response.get("timeout") and (response.get("gave_up") or response.get("skipped")):
        reason = "timeout_or_give_up"
    elif response.get("timeout"):
        reason = "timeout"
    else:
        reason = "give_up"
    return {
        "is_valid": False,
        "invalid_reason": reason,
        "solution_id": None,
        "parsed_answer": None,
        "canonical_answer": None,
        "parser_version": "human_outcome_v1",
        "extracted_answer": None,
        "parse_confidence": "high",
        "requires_manual_review": False,
        "parse_notes": "Human trial ended by skip/give-up/timeout; no final answer was scored.",
    }
