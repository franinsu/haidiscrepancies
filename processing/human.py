#!/usr/bin/env python3
"""Reconstruct the retained human analysis rows from the immutable archive.

Only de-identified fields needed for figure reproduction are written.  The
retention rules mirror the July 2026 analysis-policy amendment.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


from processing.retention import retention_by_user


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def payloads(archive: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for outer in read_jsonl(archive / "responses.jsonl"):
        row = json.loads(outer["row_json"])
        # Database columns are canonical for the fields duplicated in row_json.
        row["username"] = outer["username"]
        row["puzzle_id"] = outer["puzzle_id"]
        row["is_correct"] = bool(outer["is_correct"])
        row["bad_record"] = bool(outer["bad_record"])
        row["timeout"] = bool(outer["timeout"])
        row["gave_up"] = bool(outer["gave_up"])
        row["response_time_ms"] = outer.get("response_time_ms")
        rows.append(row)
    return rows


def retained_users(cohort, participants, rows):
    if cohort not in {"main", "module"}:
        raise ValueError(f"unknown cohort: {cohort}")
    summary = retention_by_user([{**row, "cohort": cohort} for row in participants], rows)
    return {name for name, value in summary.items() if value["analysis_retained"]}


def deidentify(
    rows: list[dict[str, Any]],
    participants: list[dict[str, Any]],
    retained: set[str],
) -> list[dict[str, Any]]:
    analysis_ids = {
        str(row["username"]): str(row.get("analysis_id") or f"anonymous-{index:04d}")
        for index, row in enumerate(participants, start=1)
    }
    fields = (
        "puzzle_id",
        "puzzle_type",
        "family_id",
        "block_id",
        "module",
        "condition",
        "sequence_id",
        "sequence_position",
        "is_correct",
        "bad_record",
        "timeout",
        "gave_up",
        "response_time_ms",
        "solution_id",
        "abstract_solution_id",
        "accepted_solution_id",
        "accepted_abstract_solution_id",
    )
    output = []
    for row in rows:
        username = str(row["username"])
        if username not in retained:
            continue
        slim = {field: row.get(field) for field in fields}
        slim["participant_id"] = analysis_ids[username]
        slim["is_valid"] = bool(row.get("is_correct"))
        slim["feedback_mode"] = "human_feedback"
        slim["attempt_index"] = 1
        slim["source"] = "HUMAN"
        # Prefer the accepted/canonical class when both names exist.
        slim["solution_id"] = row.get("accepted_solution_id") or row.get("solution_id")
        slim["abstract_solution_id"] = (
            row.get("accepted_abstract_solution_id")
            or row.get("abstract_solution_id")
            or slim["solution_id"]
        )
        output.append(slim)
    return output
