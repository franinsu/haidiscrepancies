from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from puzzles.common import read_jsonl, stable_hash
from processing.response_parser import PARSER_VERSION


EXPECTED_RENDER_VERSION = "api_image_v9"
SUPPORTED_FEEDBACK_MODES = {"one_shot"}
SUPPORTED_PROVIDERS = {"openai", "anthropic", "google"}
SUPPORTED_PROVIDER_APIS = {
    "openai": {"responses"},
    "anthropic": {"messages"},
    "google": {"generate_content"},
}
SUPPORTED_PROMPT_CONDITIONS = {"direct_solve", "human_participant"}
DEFAULT_PROMPT_CONDITIONS = ("direct_solve", "human_participant")
RESPONSE_PARSER_PATH = Path(__file__).resolve().parents[2] / "processing" / "response_parser.py"


def describe_api_run(
    image_manifest_path: str = "collection/models/manifests/image_manifest.csv",
    model_conditions_path: str = "collection/models/config/api_model_conditions.json",
    prompt_manifest_path: str = "collection/models/prompts/prompt_manifest.json",
    main_data_path: str = "data/stimuli/all_puzzles.jsonl",
    modules_data_path: str = "data/stimuli/modules/all_module_trials.jsonl",
    blocks_path: str = "data/stimuli/modules/module_blocks.jsonl",
    dataset_manifest_path: str = "data/stimuli/dataset_manifest.json",
    model_conditions: Sequence[str] | None = None,
    feedback_modes: Sequence[str] = ("one_shot",),
    prompt_conditions: Sequence[str] | None = None,
    samples: int = 100,
    dataset_scope: str = "all",
    expected_render_version: str = EXPECTED_RENDER_VERSION,
) -> Dict[str, Any]:
    """Validate a prospective API run and return counts without writing files."""
    feedback_modes, selected_prompt_conditions = _normalize_run_options(
        feedback_modes=feedback_modes,
        prompt_conditions=prompt_conditions,
        samples=samples,
        dataset_scope=dataset_scope,
    )
    state = _load_and_preflight_run(
        image_manifest_path=image_manifest_path,
        model_conditions_path=model_conditions_path,
        prompt_manifest_path=prompt_manifest_path,
        main_data_path=main_data_path,
        modules_data_path=modules_data_path,
        blocks_path=blocks_path,
        dataset_manifest_path=dataset_manifest_path,
        model_conditions=model_conditions,
        prompt_conditions=selected_prompt_conditions,
        dataset_scope=dataset_scope,
        expected_render_version=expected_render_version,
    )
    queue_rows = (
        len(state["selected_images"])
        * len(state["selected_model_config"])
        * len(selected_prompt_conditions)
        * len(feedback_modes)
        * samples
    )
    return {
        "ok": not state["errors"],
        "displayed_images": len(state["selected_images"]),
        "conditions": len(state["selected_model_config"]),
        "prompts": len(selected_prompt_conditions),
        "samples": samples,
        "queue_rows": queue_rows,
        "errors": state["errors"],
    }


def build_api_run_manifest(
    run_id: str,
    out_root: str = "data/ai/new_runs",
    image_manifest_path: str = "collection/models/manifests/image_manifest.csv",
    model_conditions_path: str = "collection/models/config/api_model_conditions.json",
    prompt_manifest_path: str = "collection/models/prompts/prompt_manifest.json",
    main_data_path: str = "data/stimuli/all_puzzles.jsonl",
    modules_data_path: str = "data/stimuli/modules/all_module_trials.jsonl",
    blocks_path: str = "data/stimuli/modules/module_blocks.jsonl",
    dataset_manifest_path: str = "data/stimuli/dataset_manifest.json",
    model_conditions: Sequence[str] | None = None,
    feedback_modes: Sequence[str] = ("one_shot",),
    prompt_conditions: Sequence[str] | None = None,
    samples: int = 100,
    dataset_scope: str = "all",
    expected_render_version: str = EXPECTED_RENDER_VERSION,
    allow_existing: bool = False,
) -> Dict[str, Any]:
    feedback_modes, selected_prompt_conditions = _normalize_run_options(
        feedback_modes=feedback_modes,
        prompt_conditions=prompt_conditions,
        samples=samples,
        dataset_scope=dataset_scope,
    )

    run_dir = Path(out_root) / run_id
    if run_dir.exists() and not allow_existing:
        raise FileExistsError(f"Run directory already exists: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)

    state = _load_and_preflight_run(
        image_manifest_path=image_manifest_path,
        model_conditions_path=model_conditions_path,
        prompt_manifest_path=prompt_manifest_path,
        main_data_path=main_data_path,
        modules_data_path=modules_data_path,
        blocks_path=blocks_path,
        dataset_manifest_path=dataset_manifest_path,
        model_conditions=model_conditions,
        prompt_conditions=selected_prompt_conditions,
        dataset_scope=dataset_scope,
        expected_render_version=expected_render_version,
    )
    errors = state["errors"]
    if errors:
        _write_json(run_dir / "preflight_errors.json", {"errors": errors})
        raise SystemExit("API run preflight failed:\n- " + "\n- ".join(errors))

    selected_images = state["selected_images"]
    main_rows = state["main_rows"]
    module_rows = state["module_rows"]
    block_rows = state["block_rows"]
    prompt_manifest = state["prompt_manifest"]
    selected_model_names = state["selected_model_names"]
    selected_model_config = state["selected_model_config"]
    prompt_variants = _prompt_variants(prompt_manifest, selected_prompt_conditions)
    prompt_hashes = _prompt_hashes(prompt_variants)
    queue_rows = _build_queue_rows(
        run_id=run_id,
        image_rows=selected_images,
        selected_model_config=selected_model_config,
        feedback_modes=feedback_modes,
        prompt_conditions=selected_prompt_conditions,
        prompt_variants=prompt_variants,
        samples=samples,
        prompt_hashes=prompt_hashes,
    )
    queue_path = run_dir / "queue.csv"
    _write_csv(queue_path, queue_rows)

    run_manifest = {
        "run_id": run_id,
        "created_at_utc": _utc_now(),
        "parser_version": PARSER_VERSION,
        "dataset_scope": dataset_scope,
        "samples_per_displayed_item": samples,
        "feedback_modes": feedback_modes,
        "prompt_conditions": selected_prompt_conditions,
        "model_conditions": selected_model_names,
        "expected_render_version": expected_render_version,
        "paths": {
            "run_dir": str(run_dir),
            "queue": str(queue_path),
            "raw": str(run_dir / "raw.jsonl"),
            "scored": str(run_dir / "scored.jsonl"),
            "final_summary": str(run_dir / "final_summary.jsonl"),
            "errors": str(run_dir / "errors.jsonl"),
            "analysis_dir": str(run_dir / "analysis"),
            "run_summary": str(run_dir / "run_summary.json"),
            "image_manifest": image_manifest_path,
            "model_conditions": model_conditions_path,
            "prompt_manifest": prompt_manifest_path,
            "main_data": main_data_path,
            "modules_data": modules_data_path,
            "blocks": blocks_path,
            "dataset_manifest": dataset_manifest_path,
        },
        "hashes": {
            "main_data_stable_hash": stable_hash(main_rows),
            "modules_data_stable_hash": stable_hash(module_rows),
            "module_blocks_stable_hash": stable_hash(block_rows),
            "dataset_manifest_sha256": _sha256_file(dataset_manifest_path),
            "image_manifest_sha256": _sha256_file(image_manifest_path),
            "prompt_manifest_sha256": _sha256_file(prompt_manifest_path),
            "model_conditions_sha256": _sha256_file(model_conditions_path),
            "response_parser_sha256": _sha256_file(RESPONSE_PARSER_PATH),
            "selected_model_conditions_stable_hash": stable_hash(selected_model_config),
            "prompt_file_sha256": prompt_hashes,
        },
        "counts": {
            "main_puzzles": len(main_rows),
            "module_trials": len(module_rows),
            "module_blocks": len(block_rows),
            "selected_images": len(selected_images),
            "queue_rows": len(queue_rows),
            "images_by_dataset": dict(Counter(row["dataset_name"] for row in selected_images)),
            "images_by_module": dict(Counter(row["module"] for row in selected_images)),
            "images_by_input_kind": dict(Counter(row["input_kind"] for row in selected_images)),
            "requests_by_prompt_condition": dict(Counter(row["prompt_condition"] for row in queue_rows)),
            "requests_by_prompt_version": dict(Counter(row["prompt_version"] for row in queue_rows)),
        },
        "quality_gates": {
            "parser_success_rate_min": 0.95,
            "image_render_failures_max": 0,
            "api_error_rate_max": 0.02,
            "manual_review_rate_max": 0.05,
        },
    }
    _write_json(run_dir / "run_manifest.json", run_manifest)
    _write_json(run_dir / "run_summary.json", _initial_run_summary(run_manifest))
    return run_manifest


def _normalize_run_options(
    feedback_modes: Sequence[str],
    prompt_conditions: Sequence[str] | None,
    samples: int,
    dataset_scope: str,
) -> tuple[List[str], List[str]]:
    if samples < 1:
        raise ValueError("samples must be >= 1")
    selected_feedback_modes = list(feedback_modes)
    bad_feedback = sorted(set(selected_feedback_modes) - SUPPORTED_FEEDBACK_MODES)
    if bad_feedback:
        raise ValueError(f"Unsupported feedback modes: {bad_feedback}")
    if dataset_scope not in {"all", "main", "modules"}:
        raise ValueError("dataset_scope must be all, main, or modules")
    if prompt_conditions is None or not prompt_conditions:
        selected_prompt_conditions = list(DEFAULT_PROMPT_CONDITIONS)
    else:
        selected_prompt_conditions = list(prompt_conditions)
    bad_prompts = sorted(set(selected_prompt_conditions) - SUPPORTED_PROMPT_CONDITIONS)
    if bad_prompts:
        raise ValueError(f"Unsupported prompt conditions: {bad_prompts}")
    return selected_feedback_modes, selected_prompt_conditions


def _load_and_preflight_run(
    image_manifest_path: str,
    model_conditions_path: str,
    prompt_manifest_path: str,
    main_data_path: str,
    modules_data_path: str,
    blocks_path: str,
    dataset_manifest_path: str,
    model_conditions: Sequence[str] | None,
    prompt_conditions: Sequence[str],
    dataset_scope: str,
    expected_render_version: str,
) -> Dict[str, Any]:
    image_rows = _read_csv(image_manifest_path)
    selected_images = _filter_images(image_rows, dataset_scope)
    main_rows = read_jsonl(main_data_path)
    module_rows = read_jsonl(modules_data_path)
    block_rows = read_jsonl(blocks_path)
    model_config = _read_json(model_conditions_path)
    prompt_manifest = _read_json(prompt_manifest_path)
    dataset_manifest = _read_json(dataset_manifest_path)

    if model_conditions is None or not model_conditions:
        selected_model_names = sorted(model_config)
    else:
        selected_model_names = list(model_conditions)
    selected_model_config = {
        name: model_config[name]
        for name in selected_model_names
        if name in model_config
    }

    errors: List[str] = []
    errors.extend(_preflight_model_conditions(model_config, selected_model_names))
    errors.extend(_preflight_images(selected_images, expected_render_version))
    errors.extend(_preflight_prompts(selected_images, prompt_manifest, prompt_conditions))
    errors.extend(_preflight_dataset_manifest(main_rows, dataset_manifest))
    if not selected_images:
        errors.append("No image rows selected for this run.")

    return {
        "selected_images": selected_images,
        "main_rows": main_rows,
        "module_rows": module_rows,
        "block_rows": block_rows,
        "prompt_manifest": prompt_manifest,
        "selected_model_names": selected_model_names,
        "selected_model_config": selected_model_config,
        "errors": errors,
    }


def _build_queue_rows(
    run_id: str,
    image_rows: Sequence[Dict[str, str]],
    selected_model_config: Dict[str, Dict[str, Any]],
    feedback_modes: Sequence[str],
    prompt_conditions: Sequence[str],
    prompt_variants: Dict[tuple[str, str], Dict[str, str]],
    samples: int,
    prompt_hashes: Dict[str, str],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for image in image_rows:
        input_kind = image["input_kind"]
        for prompt_condition in prompt_conditions:
            prompt = prompt_variants[(prompt_condition, input_kind)]
            for model_condition, config in selected_model_config.items():
                for feedback_mode in feedback_modes:
                    for sample_index in range(1, samples + 1):
                        request_key = {
                            "run_id": run_id,
                            "image_id": image["image_id"],
                            "prompt_condition": prompt_condition,
                            "prompt_version": prompt["prompt_version"],
                            "model_condition": model_condition,
                            "feedback_mode": feedback_mode,
                            "sample_index": sample_index,
                        }
                        request_id = hashlib.sha256(
                            json.dumps(request_key, sort_keys=True, separators=(",", ":")).encode("utf-8")
                        ).hexdigest()[:20]
                        rows.append(
                            {
                                "run_id": run_id,
                                "request_id": request_id,
                                "sample_index": sample_index,
                                "feedback_mode": feedback_mode,
                                "max_attempts": 1,
                                "api_model_condition": model_condition,
                                "api_provider": config["provider"],
                                "model": config["model"],
                                "sampling_profile": config.get("sampling_profile", "standard"),
                                "temperature": config.get("temperature", ""),
                                "top_p": config.get("top_p", ""),
                                "output_token_cap": config.get("max_output_tokens", config.get("max_tokens", "")),
                                "max_output_tokens": config.get("max_output_tokens", ""),
                                "max_tokens": config.get("max_tokens", ""),
                                "reasoning_effort": (config.get("reasoning") or {}).get("effort", ""),
                                "thinking_type": (config.get("thinking") or {}).get("type", ""),
                                "thinking_effort": _model_condition_effort(config),
                                "dataset_name": image["dataset_name"],
                                "module": image["module"],
                                "condition": image["condition"],
                                "image_id": image["image_id"],
                                "puzzle_id": image["puzzle_id"],
                                "sequence_id": image["sequence_id"],
                                "trial_ids": image["trial_ids"],
                                "input_kind": input_kind,
                                "puzzle_type": image["puzzle_type"],
                                "prompt_condition": prompt_condition,
                                "prompt_version": prompt["prompt_version"],
                                "prompt_file": prompt["prompt_file"],
                                "prompt_sha256": prompt_hashes[prompt["prompt_file"]],
                                "image_path": image["image_path"],
                                "image_sha256": image["image_sha256"],
                                "render_version": image["render_version"],
                                "image_width": image["width"],
                                "image_height": image["height"],
                            }
                        )
    request_ids = [row["request_id"] for row in rows]
    if len(request_ids) != len(set(request_ids)):
        raise ValueError("Queue request_id collision detected.")
    return rows


def _preflight_images(rows: Sequence[Dict[str, str]], expected_render_version: str) -> List[str]:
    errors: List[str] = []
    required = {
        "image_id",
        "puzzle_id",
        "dataset_name",
        "module",
        "condition",
        "puzzle_type",
        "trial_ids",
        "input_kind",
        "prompt_version",
        "prompt_file",
        "image_path",
        "image_sha256",
        "render_version",
        "width",
        "height",
    }
    seen = set()
    for idx, row in enumerate(rows, 1):
        missing = sorted(required - set(row))
        if missing:
            errors.append(f"image manifest row {idx} missing columns: {missing}")
            continue
        if row["image_id"] in seen:
            errors.append(f"Duplicate image_id in selected manifest: {row['image_id']}")
        seen.add(row["image_id"])
        path = Path(row["image_path"])
        if not path.exists():
            errors.append(f"{row['image_id']}: image file missing: {path}")
            continue
        actual_hash = _sha256_file(path)
        if actual_hash != row["image_sha256"]:
            errors.append(f"{row['image_id']}: image SHA mismatch")
        if row["render_version"] != expected_render_version:
            errors.append(
                f"{row['image_id']}: expected render_version {expected_render_version}, found {row['render_version']}"
            )
        if row["input_kind"] not in {"single", "sequence"}:
            errors.append(f"{row['image_id']}: invalid input_kind {row['input_kind']}")
    return errors


def _preflight_prompts(
    rows: Sequence[Dict[str, str]],
    prompt_manifest: Dict[str, Any],
    prompt_conditions: Sequence[str],
) -> List[str]:
    errors: List[str] = []
    versions = set(prompt_manifest)
    available: set[tuple[str, str]] = set()
    for prompt_version, entry in prompt_manifest.items():
        prompt_condition = entry.get("prompt_condition", "direct_solve")
        input_kind = entry.get("input_kind")
        prompt_file = entry.get("file", "")
        if prompt_condition not in SUPPORTED_PROMPT_CONDITIONS:
            errors.append(f"{prompt_version}: unsupported prompt_condition {prompt_condition}")
        if input_kind not in {"single", "sequence"}:
            errors.append(f"{prompt_version}: input_kind must be single or sequence")
        if prompt_condition in SUPPORTED_PROMPT_CONDITIONS and input_kind in {"single", "sequence"}:
            available.add((prompt_condition, input_kind))
        path = _prompt_path(prompt_file)
        if not path.exists():
            errors.append(f"{prompt_version}: prompt file missing: {prompt_file}")
            continue
        manifest_hash = entry.get("sha256")
        if not manifest_hash:
            errors.append(f"{prompt_version}: prompt_manifest is missing sha256")
        elif _sha256_file(path) != manifest_hash:
            errors.append(f"{prompt_version}: prompt SHA mismatch for {path}")
    for prompt_condition in prompt_conditions:
        for input_kind in {"single", "sequence"}:
            if (prompt_condition, input_kind) not in available:
                errors.append(f"Missing prompt manifest entry for {prompt_condition}/{input_kind}")
    for row in rows:
        version = row["prompt_version"]
        if version not in versions:
            errors.append(f"{row['image_id']}: unknown prompt_version {row['prompt_version']}")
            continue
        manifest_entry = prompt_manifest[version]
        manifest_file = manifest_entry.get("file")
        if manifest_file and Path(row["prompt_file"]).name != Path(manifest_file).name:
            errors.append(
                f"{row['image_id']}: prompt_version {version} expects {manifest_file}, found {Path(row['prompt_file']).name}"
            )
        for prompt_condition in prompt_conditions:
            if (prompt_condition, row["input_kind"]) not in available:
                errors.append(f"{row['image_id']}: no {prompt_condition} prompt for input_kind {row['input_kind']}")
    return errors


def _preflight_model_conditions(config: Dict[str, Any], selected: Sequence[str]) -> List[str]:
    errors: List[str] = []
    for name in selected:
        if name not in config:
            errors.append(f"Unknown model condition: {name}")
            continue
        row = config[name]
        provider = row.get("provider")
        model = str(row.get("model", ""))
        if provider not in SUPPORTED_PROVIDERS:
            errors.append(f"{name}: unsupported provider {provider}")
            continue
        api = row.get("api")
        if api not in SUPPORTED_PROVIDER_APIS.get(provider, set()):
            errors.append(f"{name}: provider {provider} does not support api={api!r} in this manifest builder")
        if not row.get("model"):
            errors.append(f"{name}: missing model")
        if not (row.get("max_output_tokens") or row.get("max_tokens")):
            errors.append(f"{name}: missing max_output_tokens or max_tokens")
        if provider == "openai":
            if row.get("api") != "responses":
                errors.append(f"{name}: OpenAI condition must use responses API")
            if "reasoning" not in row:
                errors.append(f"{name}: missing reasoning config")
            if model == "gpt-5.6-sol":
                if row.get("max_output_tokens") != 8192:
                    errors.append(f"{name}: GPT-5.6 Sol max_output_tokens must be 8192")
                if row.get("image_detail") != "high":
                    errors.append(f"{name}: GPT-5.6 Sol image_detail must be high")
                live_fallback = row.get("live_fallback") or {}
                if live_fallback.get("service_tier") != "flex":
                    errors.append(f"{name}: GPT-5.6 Sol live fallback must use service_tier='flex'")
                if int(live_fallback.get("timeout_seconds", 0)) < 900:
                    errors.append(f"{name}: GPT-5.6 Sol Flex timeout_seconds must be at least 900")
                if int(live_fallback.get("resource_unavailable_retries", 0)) < 1:
                    errors.append(f"{name}: GPT-5.6 Sol Flex resource_unavailable_retries must be positive")
                if (row.get("text") or {}).get("verbosity") != "low":
                    errors.append(f"{name}: GPT-5.6 Sol text.verbosity must be low")
                if row.get("tools") != []:
                    errors.append(f"{name}: GPT-5.6 Sol tools must be an empty list")
                if row.get("stream") is not False:
                    errors.append(f"{name}: GPT-5.6 Sol stream must be false")
            text_format = ((row.get("text") or {}).get("format") or {})
            if not isinstance(text_format, dict) or text_format.get("type") != "text":
                errors.append(f"{name}: OpenAI text.format must be an object with type='text'")
            if model.startswith("gpt-5.6"):
                errors.extend(_validate_omitted_keys(name, row, ("temperature", "top_p")))
            else:
                errors.extend(_validate_numeric_range(name, row, "temperature", 0.0, 2.0))
                errors.extend(_validate_numeric_range(name, row, "top_p", 0.0, 1.0))
        if provider == "anthropic":
            if row.get("api") != "messages":
                errors.append(f"{name}: Anthropic condition must use messages API")
            if "thinking" not in row:
                errors.append(f"{name}: missing thinking config")
            if row.get("thinking", {}).get("type") == "adaptive" and not (row.get("output_config") or {}).get("effort"):
                errors.append(f"{name}: adaptive thinking requires output_config.effort")
            if "effort" in row:
                errors.append(f"{name}: top-level effort is not accepted by Anthropic; use output_config.effort")
            if model == "claude-opus-4-8":
                errors.extend(_validate_omitted_keys(name, row, ("temperature", "top_p", "top_k")))
            else:
                errors.extend(_validate_numeric_range(name, row, "temperature", 0.0, 1.0))
                errors.extend(_validate_numeric_range(name, row, "top_p", 0.0, 1.0))
        if provider == "google":
            thinking = row.get("thinking") or {}
            if thinking.get("type") != "level":
                errors.append(f"{name}: Gemini conditions should use thinking.type='level'")
            if thinking.get("effort") not in {"low", "medium", "high"}:
                errors.append(f"{name}: Gemini thinking.effort should be low, medium, or high")
    return errors


def _model_condition_effort(config: Dict[str, Any]) -> str:
    thinking = config.get("thinking") or {}
    return (
        str(config.get("effort") or "")
        or str((config.get("output_config") or {}).get("effort") or "")
        or str(thinking.get("effort") or "")
        or str((config.get("reasoning") or {}).get("effort") or "")
        or ("enabled" if thinking.get("type") == "enabled" else "")
    )


def _validate_omitted_keys(condition_name: str, row: Dict[str, Any], keys: Sequence[str]) -> List[str]:
    return [f"{condition_name}: {key} must be omitted for this model" for key in keys if key in row]


def _validate_numeric_range(
    condition_name: str,
    row: Dict[str, Any],
    key: str,
    min_value: float,
    max_value: float,
) -> List[str]:
    if key not in row:
        return []
    value = row[key]
    if not isinstance(value, (int, float)):
        return [f"{condition_name}: {key} must be numeric"]
    if not (min_value <= float(value) <= max_value):
        return [f"{condition_name}: {key} must be between {min_value} and {max_value}"]
    return []


def _preflight_dataset_manifest(main_rows: List[Dict[str, Any]], manifest: Dict[str, Any]) -> List[str]:
    current_hash = stable_hash(main_rows)
    if manifest.get("dataset_hash") != current_hash:
        return ["data/stimuli/dataset_manifest.json hash does not match current all_puzzles.jsonl; restore the matching frozen study files and manifest."]
    return []


def _filter_images(rows: Sequence[Dict[str, str]], dataset_scope: str) -> List[Dict[str, str]]:
    if dataset_scope == "all":
        return list(rows)
    if dataset_scope == "main":
        return [row for row in rows if row["dataset_name"] == "main"]
    return [row for row in rows if row["dataset_name"] == "modules"]


def _prompt_variants(
    prompt_manifest: Dict[str, Any],
    prompt_conditions: Sequence[str],
) -> Dict[tuple[str, str], Dict[str, str]]:
    out: Dict[tuple[str, str], Dict[str, str]] = {}
    selected = set(prompt_conditions)
    for prompt_version, entry in prompt_manifest.items():
        prompt_condition = entry.get("prompt_condition", "direct_solve")
        input_kind = entry.get("input_kind")
        if prompt_condition not in selected or input_kind not in {"single", "sequence"}:
            continue
        prompt_file = str(_prompt_path(entry["file"]))
        out[(prompt_condition, input_kind)] = {
            "prompt_version": prompt_version,
            "prompt_file": prompt_file,
        }
    return out


def _prompt_path(prompt_file: str | Path) -> Path:
    path = Path(prompt_file)
    if path.exists():
        return path
    return Path("collection/models/prompts") / path


def _prompt_hashes(prompt_variants: Dict[tuple[str, str], Dict[str, str]]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for prompt_file in sorted({entry["prompt_file"] for entry in prompt_variants.values()}):
        out[prompt_file] = _sha256_file(prompt_file)
    return out


def _read_csv(path: str | Path) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(path: str | Path, rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("Refusing to write empty queue.")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _read_json(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: str | Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True, ensure_ascii=False)
        f.write("\n")


def _sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _initial_run_summary(run_manifest: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "run_id": run_manifest["run_id"],
        "created_at_utc": run_manifest["created_at_utc"],
        "status": "manifest_created",
        "queue_rows": run_manifest["counts"]["queue_rows"],
        "completed_requests": 0,
        "api_error_rate": None,
        "parser_success_rate": None,
        "manual_review_rate": None,
        "one_shot_valid_rate": None,
    }


def _split_arg(value: str | None) -> List[str]:
    if value is None or not value.strip():
        return []
    return [part.strip() for part in value.split(",") if part.strip()]


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build a frozen API run manifest and resumable request queue.")
    parser.add_argument("--run_id", default="", help="Unique run ID. Defaults to timestamped api_YYYYMMDDTHHMMSSZ.")
    parser.add_argument("--out_root", default="data/ai/new_runs")
    parser.add_argument("--image_manifest", default="collection/models/manifests/image_manifest.csv")
    parser.add_argument("--model_conditions_config", default="collection/models/config/api_model_conditions.json")
    parser.add_argument("--prompt_manifest", default="collection/models/prompts/prompt_manifest.json")
    parser.add_argument("--main_data", default="data/stimuli/all_puzzles.jsonl")
    parser.add_argument("--modules_data", default="data/stimuli/modules/all_module_trials.jsonl")
    parser.add_argument("--blocks", default="data/stimuli/modules/module_blocks.jsonl")
    parser.add_argument("--dataset_manifest", default="data/stimuli/dataset_manifest.json")
    parser.add_argument("--api_model_conditions", default="", help="Comma-separated condition names. Defaults to all.")
    parser.add_argument("--feedback_modes", default="one_shot", help="Comma-separated feedback modes. Currently only one_shot is supported.")
    parser.add_argument(
        "--prompt_conditions",
        default=",".join(DEFAULT_PROMPT_CONDITIONS),
        help="Comma-separated prompt conditions. Defaults to direct_solve,human_participant.",
    )
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--dataset_scope", choices=["all", "main", "modules"], default="all")
    parser.add_argument("--expected_render_version", default=EXPECTED_RENDER_VERSION)
    parser.add_argument("--allow_existing", action="store_true")
    parser.add_argument(
        "--describe",
        "--preflight-only",
        dest="preflight_only",
        action="store_true",
        help="Validate inputs and print a JSON count summary without creating a run directory.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    selected_model_conditions = _split_arg(args.api_model_conditions)
    selected_feedback_modes = _split_arg(args.feedback_modes) or ["one_shot"]
    selected_prompt_conditions = _split_arg(args.prompt_conditions) or list(DEFAULT_PROMPT_CONDITIONS)

    if args.preflight_only:
        try:
            summary = describe_api_run(
                image_manifest_path=args.image_manifest,
                model_conditions_path=args.model_conditions_config,
                prompt_manifest_path=args.prompt_manifest,
                main_data_path=args.main_data,
                modules_data_path=args.modules_data,
                blocks_path=args.blocks,
                dataset_manifest_path=args.dataset_manifest,
                model_conditions=selected_model_conditions,
                feedback_modes=selected_feedback_modes,
                prompt_conditions=selected_prompt_conditions,
                samples=args.samples,
                dataset_scope=args.dataset_scope,
                expected_render_version=args.expected_render_version,
            )
        except (OSError, ValueError, KeyError, TypeError, csv.Error) as exc:
            summary = {
                "ok": False,
                "displayed_images": None,
                "conditions": None,
                "prompts": None,
                "samples": args.samples,
                "queue_rows": None,
                "errors": [str(exc)],
            }
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return 0 if summary["ok"] else 1

    run_id = args.run_id or "api_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    manifest = build_api_run_manifest(
        run_id=run_id,
        out_root=args.out_root,
        image_manifest_path=args.image_manifest,
        model_conditions_path=args.model_conditions_config,
        prompt_manifest_path=args.prompt_manifest,
        main_data_path=args.main_data,
        modules_data_path=args.modules_data,
        blocks_path=args.blocks,
        dataset_manifest_path=args.dataset_manifest,
        model_conditions=selected_model_conditions,
        feedback_modes=selected_feedback_modes,
        prompt_conditions=selected_prompt_conditions,
        samples=args.samples,
        dataset_scope=args.dataset_scope,
        expected_render_version=args.expected_render_version,
        allow_existing=args.allow_existing,
    )
    print(f"Wrote {manifest['paths']['run_dir']}/run_manifest.json")
    print(f"Queue rows: {manifest['counts']['queue_rows']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
