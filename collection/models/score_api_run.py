from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from processing.score_responses import _expand_response_row, _load_catalog, _score_one_response
from collection.models.run_integrity import (
    assert_no_mock_results,
    atomic_write_jsonl,
    ensure_distinct_output,
    validate_static_assets,
)
from puzzles.common import read_jsonl


def score_api_run(
    run_dir: str,
    raw_path: str = "",
    out_path: str = "",
    main_data_path: str = "data/stimuli/all_puzzles.jsonl",
    modules_data_path: str = "data/stimuli/modules/all_module_trials.jsonl",
    main_catalog_path: str = "data/stimuli/solution_catalog.jsonl",
    modules_catalog_path: str = "data/stimuli/modules/solution_catalog_modules.jsonl",
) -> List[Dict[str, Any]]:
    root = Path(run_dir)
    raw = Path(raw_path) if raw_path else root / "raw.jsonl"
    out = Path(out_path) if out_path else root / "scored.jsonl"

    ensure_distinct_output(
        out,
        {
            "raw responses": raw,
            "main data": main_data_path,
            "module data": modules_data_path,
            "main catalog": main_catalog_path,
            "module catalog": modules_catalog_path,
        },
        output_label="scored output",
    )
    formal_paths = {
        "raw": raw,
        "final_summary": root / "final_summary.jsonl",
        "errors": root / "errors.jsonl",
        "run_summary": root / "run_summary.json",
    }
    assert_no_mock_results(formal_paths, context="API scoring")
    validate_static_assets(
        root,
        queue_path=root / "queue.csv",
        main_data_path=main_data_path,
        modules_data_path=modules_data_path,
    )

    puzzles = _load_puzzles(main_data_path, modules_data_path)
    catalog = _load_combined_catalog(main_catalog_path, modules_catalog_path)
    responses = read_jsonl(str(raw))

    scored: List[Dict[str, Any]] = []
    response_index = 0
    for response in responses:
        for response_part in _expand_response_row(response):
            response_index += 1
            scored.append(
                _score_one_response(
                    response_part,
                    response_index=response_index,
                    puzzles=puzzles,
                    catalog=catalog,
                )
            )

    atomic_write_jsonl(out, scored)
    return scored


def _load_puzzles(main_data_path: str, modules_data_path: str) -> Dict[str, Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if main_data_path and Path(main_data_path).exists():
        rows.extend(read_jsonl(main_data_path))
    if modules_data_path and Path(modules_data_path).exists():
        rows.extend(read_jsonl(modules_data_path))
    return {row["id"]: row for row in rows}


def _load_combined_catalog(main_catalog_path: str, modules_catalog_path: str) -> Dict[str, Dict[str, Any]]:
    catalog: Dict[str, Dict[str, Any]] = {}
    catalog.update(_load_catalog(main_catalog_path))
    catalog.update(_load_catalog(modules_catalog_path))
    return catalog


def main() -> int:
    parser = argparse.ArgumentParser(description="Score a complete API run bundle into run_dir/scored.jsonl.")
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--raw", default="")
    parser.add_argument("--out", default="")
    parser.add_argument("--main_data", default="data/stimuli/all_puzzles.jsonl")
    parser.add_argument("--modules_data", default="data/stimuli/modules/all_module_trials.jsonl")
    parser.add_argument("--main_catalog", default="data/stimuli/solution_catalog.jsonl")
    parser.add_argument("--modules_catalog", default="data/stimuli/modules/solution_catalog_modules.jsonl")
    args = parser.parse_args()
    scored = score_api_run(
        run_dir=args.run_dir,
        raw_path=args.raw,
        out_path=args.out,
        main_data_path=args.main_data,
        modules_data_path=args.modules_data,
        main_catalog_path=args.main_catalog,
        modules_catalog_path=args.modules_catalog,
    )
    valid = sum(1 for row in scored if row["is_valid"])
    review = sum(1 for row in scored if row.get("requires_manual_review"))
    out_path = args.out or str(Path(args.run_dir) / "scored.jsonl")
    print(f"Wrote {len(scored)} scored rows to {out_path}")
    print(f"Valid: {valid}; invalid: {len(scored) - valid}")
    print(f"Manual review suggested: {review}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
