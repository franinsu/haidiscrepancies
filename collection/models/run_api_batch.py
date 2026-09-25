from __future__ import annotations

import argparse
import http.client
import json
import os
import re
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path
from typing import Any, Dict, List, Sequence

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from collection.models.run_api_collection import (
    ApiCallError,
    _anthropic_usage,
    _append_jsonl,
    _build_anthropic_request_body,
    _build_gemini_request_body,
    _build_openai_request_body,
    _completed_request_ids,
    _error_row,
    _extract_anthropic_text,
    _extract_gemini_text,
    _extract_openai_text,
    _gemini_usage,
    _final_summary_from_attempts,
    _openai_usage,
    _post_json,
    _provider_api_key,
    _queue_metadata,
    _read_csv,
    _read_json,
    _read_text,
    _resolve_paths,
    _score_attempt,
    _utc_now,
    _write_json,
    _write_run_summary,
)
from collection.models.run_integrity import (
    RunIntegrityError,
    assert_no_mock_results,
    sha256_file,
    stable_fingerprint,
    validate_static_assets,
)
from puzzles.common import read_jsonl


OPENAI_BATCH_ENDPOINT = "/v1/responses"
DEFAULT_BATCH_PROVIDERS = ("openai", "anthropic", "google")
DEFAULT_OPENAI_MAX_REQUESTS = 50_000
DEFAULT_ANTHROPIC_MAX_REQUESTS = 100_000
DEFAULT_GOOGLE_MAX_REQUESTS = 20_000
DEFAULT_OPENAI_MAX_BYTES = 180 * 1024 * 1024
DEFAULT_ANTHROPIC_MAX_BYTES = 220 * 1024 * 1024
DEFAULT_GOOGLE_MAX_BYTES = 18 * 1024 * 1024
GOOGLE_INLINE_MAX_BYTES = 20 * 1024 * 1024
GOOGLE_FILE_MAX_BYTES = 2 * 1024 * 1024 * 1024
GOOGLE_INPUT_MODES = ("inline", "file")
GOOGLE_SUBMISSION_PHASES = ("bootstrap", "standard")
GOOGLE_UPLOAD_CHUNK_BYTES = 64 * 1024 * 1024
GOOGLE_UPLOAD_MAX_RETRIES = 5
DEFAULT_OPENAI_OUTPUT_RETENTION_SECONDS = 7 * 24 * 60 * 60
ANTHROPIC_CUSTOM_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def prepare_api_batches(
    run_dir: str,
    model_conditions_path: str = "collection/models/templates/api_model_conditions.json",
    providers: Sequence[str] = DEFAULT_BATCH_PROVIDERS,
    api_model_conditions: Sequence[str] | None = None,
    limit: int = 0,
    force: bool = False,
    openai_max_bytes: int = DEFAULT_OPENAI_MAX_BYTES,
    anthropic_max_bytes: int = DEFAULT_ANTHROPIC_MAX_BYTES,
    google_max_bytes: int = DEFAULT_GOOGLE_MAX_BYTES,
    openai_max_requests: int = DEFAULT_OPENAI_MAX_REQUESTS,
    anthropic_max_requests: int = DEFAULT_ANTHROPIC_MAX_REQUESTS,
    google_max_requests: int = DEFAULT_GOOGLE_MAX_REQUESTS,
    google_input_mode: str = "inline",
    google_bootstrap_requests_per_condition: int = 0,
    google_bootstrap_max_requests: int = 0,
) -> Dict[str, Any]:
    _validate_google_input_mode(google_input_mode, google_max_bytes)
    _validate_google_bootstrap(
        google_bootstrap_requests_per_condition,
        google_bootstrap_max_requests,
    )
    paths = _resolve_paths(run_dir, "", "", "", "", "")
    assert_no_mock_results(_formal_result_paths(paths), context="API batch prepare")
    validate_static_assets(
        run_dir,
        queue_path=paths["queue"],
        model_conditions_path=model_conditions_path,
    )
    prepare_config = _batch_prepare_config(
        model_conditions_path=model_conditions_path,
        providers=providers,
        api_model_conditions=api_model_conditions,
        limit=limit,
        openai_max_bytes=openai_max_bytes,
        anthropic_max_bytes=anthropic_max_bytes,
        google_max_bytes=google_max_bytes,
        openai_max_requests=openai_max_requests,
        anthropic_max_requests=anthropic_max_requests,
        google_max_requests=google_max_requests,
        google_input_mode=google_input_mode,
        google_bootstrap_requests_per_condition=google_bootstrap_requests_per_condition,
        google_bootstrap_max_requests=google_bootstrap_max_requests,
    )
    fingerprints = _batch_fingerprints(paths["queue"], prepare_config)
    manifest_path = _batch_manifest_path(run_dir)
    if manifest_path.exists():
        old_manifest = _read_json(manifest_path)
        _assert_batch_manifest_matches(old_manifest, fingerprints, model_conditions_path)
        if not force:
            return old_manifest
        if _has_submitted_batches(old_manifest):
            raise ValueError(
                "Refusing to overwrite a batch_manifest.json that already has submitted provider_batch_id values. "
                "Archive the submitted manifest explicitly before preparing a different batch set."
            )

    queue = _read_csv(paths["queue"])
    model_conditions = _read_json(model_conditions_path)
    completed = _completed_request_ids(paths["final_summary"])
    selected_conditions = set(api_model_conditions or [])
    selected_providers = set(providers)
    chunks: Dict[str, _BatchChunk] = {}
    batches: List[Dict[str, Any]] = []
    prepared_count = 0
    prepared_by_condition: Dict[str, int] = {}

    root = Path(run_dir) / "batches"
    root.mkdir(parents=True, exist_ok=True)

    try:
        for queue_row in queue:
            if limit and prepared_count >= limit:
                break
            if queue_row["request_id"] in completed:
                continue
            if queue_row.get("feedback_mode") != "one_shot":
                raise ValueError("Only one_shot API batch collection is supported.")
            condition_name = queue_row["api_model_condition"]
            if selected_conditions and condition_name not in selected_conditions:
                continue
            condition = model_conditions[condition_name]
            provider = condition["provider"]
            if provider not in selected_providers:
                continue
            prompt_text = _read_text(queue_row["prompt_file"])
            request_item = _batch_request_item(queue_row, condition, prompt_text)
            item_bytes = _jsonl_bytes(request_item)
            max_bytes = _provider_max_bytes(provider, openai_max_bytes, anthropic_max_bytes, google_max_bytes)
            max_requests = _provider_max_requests(provider, openai_max_requests, anthropic_max_requests, google_max_requests)
            chunk = chunks.get(condition_name)
            condition_prepared = prepared_by_condition.get(condition_name, 0)
            submission_phase = "standard"
            if (
                provider == "google"
                and google_bootstrap_requests_per_condition
                and condition_prepared < google_bootstrap_requests_per_condition
            ):
                submission_phase = "bootstrap"
            phase_changed = chunk is not None and chunk.submission_phase != submission_phase
            if phase_changed or chunk is None or not chunk.can_accept(item_bytes, max_bytes):
                if chunk is not None:
                    batches.append(chunk.close())
                chunk_index = sum(1 for batch in batches if batch.get("api_model_condition") == condition_name) + 1
                request_limit = max_requests
                if submission_phase == "bootstrap":
                    request_limit = min(
                        google_bootstrap_max_requests,
                        google_bootstrap_requests_per_condition - condition_prepared,
                    )
                chunk = _BatchChunk(
                    run_dir=Path(run_dir),
                    provider=provider,
                    api_model_condition=condition_name,
                    chunk_index=chunk_index,
                    request_limit=request_limit,
                    submission_phase=submission_phase,
                )
                chunks[condition_name] = chunk
            chunk.append(request_item, item_bytes)
            prepared_count += 1
            prepared_by_condition[condition_name] = condition_prepared + 1

        for chunk in chunks.values():
            batches.append(chunk.close())
    except Exception:
        for chunk in chunks.values():
            chunk.abort()
        raise

    for batch in batches:
        if batch["provider"] == "google":
            batch["input_mode"] = google_input_mode
    if google_bootstrap_requests_per_condition:
        google_batches = [batch for batch in batches if batch["provider"] == "google"]
        other_batches = [batch for batch in batches if batch["provider"] != "google"]
        google_batches.sort(
            key=lambda batch: (
                0 if batch.get("submission_phase") == "bootstrap" else 1,
                batch["chunk_index"],
                batch["api_model_condition"],
            )
        )
        batches = [*google_batches, *other_batches]

    manifest = {
        "schema_version": 2,
        "run_dir": str(run_dir),
        "queue_path": paths["queue"],
        "prepared_at_utc": _utc_now(),
        "providers": sorted(selected_providers),
        "api_model_conditions": sorted(selected_conditions) if selected_conditions else "all",
        "openai_batch_endpoint": OPENAI_BATCH_ENDPOINT,
        "limits": {
            "openai_max_requests": openai_max_requests,
            "openai_max_bytes": openai_max_bytes,
            "anthropic_max_requests": anthropic_max_requests,
            "anthropic_max_bytes": anthropic_max_bytes,
            "google_max_requests": google_max_requests,
            "google_max_bytes": google_max_bytes,
            "google_input_mode": google_input_mode,
            "google_bootstrap_requests_per_condition": google_bootstrap_requests_per_condition,
            "google_bootstrap_max_requests": google_bootstrap_max_requests,
        },
        "prepare_config": prepare_config,
        "fingerprints": fingerprints,
        "prepared_request_count": prepared_count,
        "batch_count": len(batches),
        "batches": batches,
    }
    _write_json(manifest_path, manifest)
    return manifest


def submit_api_batches(
    run_dir: str,
    model_conditions_path: str = "collection/models/templates/api_model_conditions.json",
    providers: Sequence[str] = DEFAULT_BATCH_PROVIDERS,
    api_model_conditions: Sequence[str] | None = None,
    limit: int = 0,
    force_prepare: bool = False,
    dry_run: bool = False,
    openai_max_bytes: int = DEFAULT_OPENAI_MAX_BYTES,
    anthropic_max_bytes: int = DEFAULT_ANTHROPIC_MAX_BYTES,
    google_max_bytes: int = DEFAULT_GOOGLE_MAX_BYTES,
    openai_max_requests: int = DEFAULT_OPENAI_MAX_REQUESTS,
    anthropic_max_requests: int = DEFAULT_ANTHROPIC_MAX_REQUESTS,
    google_max_requests: int = DEFAULT_GOOGLE_MAX_REQUESTS,
    max_submit_jobs: int = 0,
    google_input_mode: str = "inline",
    google_bootstrap_requests_per_condition: int = 0,
    google_bootstrap_max_requests: int = 0,
    submission_phases: Sequence[str] | None = None,
) -> Dict[str, Any]:
    manifest_path = _batch_manifest_path(run_dir)
    manifest = prepare_api_batches(
        run_dir=run_dir,
        model_conditions_path=model_conditions_path,
        providers=providers,
        api_model_conditions=api_model_conditions,
        limit=limit,
        force=force_prepare,
        openai_max_bytes=openai_max_bytes,
        anthropic_max_bytes=anthropic_max_bytes,
        google_max_bytes=google_max_bytes,
        openai_max_requests=openai_max_requests,
        anthropic_max_requests=anthropic_max_requests,
        google_max_requests=google_max_requests,
        google_input_mode=google_input_mode,
        google_bootstrap_requests_per_condition=google_bootstrap_requests_per_condition,
        google_bootstrap_max_requests=google_bootstrap_max_requests,
    )
    if dry_run:
        return manifest

    submitted = 0
    selected_phases = set(submission_phases or [])
    for batch in manifest["batches"]:
        if batch.get("provider_batch_id"):
            continue
        if selected_phases and batch.get("submission_phase", "standard") not in selected_phases:
            continue
        if max_submit_jobs and submitted >= max_submit_jobs:
            break
        if batch["provider"] == "openai":
            result = _submit_openai_batch(batch)
        elif batch["provider"] == "anthropic":
            result = _submit_anthropic_batch(batch)
        elif batch["provider"] == "google":
            result = _submit_google_batch(batch)
        else:
            raise ValueError(f"Unsupported provider: {batch['provider']}")
        batch.update(result)
        batch["submitted_at_utc"] = _utc_now()
        batch["status"] = "submitted"
        submitted += 1
        _write_json(manifest_path, manifest)
    manifest["submitted_batch_count"] = sum(1 for batch in manifest["batches"] if batch.get("provider_batch_id"))
    manifest["remaining_unsubmitted_batch_count"] = sum(
        1 for batch in manifest["batches"] if not batch.get("provider_batch_id")
    )
    manifest["remaining_eligible_batch_count"] = sum(
        1
        for batch in manifest["batches"]
        if not batch.get("provider_batch_id")
        and (not selected_phases or batch.get("submission_phase", "standard") in selected_phases)
    )
    manifest["updated_at_utc"] = _utc_now()
    _write_json(manifest_path, manifest)
    return {"submitted_this_run": submitted, **manifest}


def collect_api_batches(
    run_dir: str,
    model_conditions_path: str = "collection/models/templates/api_model_conditions.json",
    main_data_path: str = "data/stimuli/all_puzzles.jsonl",
    modules_data_path: str = "data/stimuli/modules/all_module_trials.jsonl",
    only_completed: bool = True,
) -> Dict[str, Any]:
    manifest_path = _batch_manifest_path(run_dir)
    paths = _resolve_paths(run_dir, "", "", "", "", "")
    assert_no_mock_results(_formal_result_paths(paths), context="API batch collect")
    validate_static_assets(
        run_dir,
        queue_path=paths["queue"],
        model_conditions_path=model_conditions_path,
        main_data_path=main_data_path,
        modules_data_path=modules_data_path,
    )
    manifest = _read_json(manifest_path)
    _assert_stored_batch_manifest_current(manifest, paths["queue"], model_conditions_path)
    queue = {row["request_id"]: row for row in _read_csv(paths["queue"])}
    puzzles = {row["id"]: row for row in [*read_jsonl(main_data_path), *read_jsonl(modules_data_path)]}
    completed = _completed_request_ids(paths["final_summary"])
    collected_success = 0
    collected_errors = 0
    job_statuses: List[Dict[str, Any]] = []

    for batch in manifest["batches"]:
        provider_batch_id = batch.get("provider_batch_id")
        if not provider_batch_id:
            job_statuses.append({**_batch_status_projection(batch), "ready": False, "reason": "not_submitted"})
            continue
        if batch["provider"] == "openai":
            provider_status = _get_openai_batch(provider_batch_id)
            ready = provider_status.get("status") in {"completed", "failed", "expired", "cancelled"}
            output_ref = provider_status.get("output_file_id")
            error_ref = provider_status.get("error_file_id")
        elif batch["provider"] == "anthropic":
            provider_status = _get_anthropic_batch(provider_batch_id)
            ready = provider_status.get("processing_status") in {"ended", "canceling", "canceled", "expired"}
            output_ref = provider_status.get("results_url") or f"{_anthropic_batches_endpoint().rstrip('/')}/{provider_batch_id}/results"
            error_ref = None
        elif batch["provider"] == "google":
            provider_status = _get_google_batch(provider_batch_id)
            state = _google_batch_state(provider_status)
            ready = state in {"JOB_STATE_SUCCEEDED", "JOB_STATE_FAILED", "JOB_STATE_CANCELLED", "JOB_STATE_EXPIRED"}
            output_ref = provider_status
            error_ref = None
        else:
            raise ValueError(f"Unsupported provider: {batch['provider']}")
        batch["last_provider_status"] = provider_status
        batch["last_checked_at_utc"] = _utc_now()
        job_statuses.append({**_batch_status_projection(batch), "ready": ready})
        if only_completed and not ready:
            continue
        if batch.get("collected_at_utc"):
            continue
        if batch["provider"] == "openai":
            if output_ref:
                ok, err = _collect_openai_output(output_ref, queue, puzzles, paths, completed)
                collected_success += ok
                collected_errors += err
            if error_ref:
                collected_errors += _collect_openai_error_file(error_ref, queue, paths, completed)
        elif batch["provider"] == "anthropic":
            ok, err = _collect_anthropic_results(output_ref, queue, puzzles, paths, completed)
            collected_success += ok
            collected_errors += err
        elif batch["provider"] == "google":
            ok, err = _collect_google_results(batch, output_ref, queue, puzzles, paths, completed)
            collected_success += ok
            collected_errors += err
        batch["collected_at_utc"] = _utc_now()
        _write_json(manifest_path, manifest)

    manifest["updated_at_utc"] = _utc_now()
    _write_json(manifest_path, manifest)
    summary = _write_run_summary(paths["run_summary"], list(queue.values()), paths["final_summary"], paths["errors"])
    summary.update(
        {
            "collection_mode": "batch",
            "batch_manifest": str(manifest_path),
            "batch_jobs": job_statuses,
            "batch_success_rows_collected_this_run": collected_success,
            "batch_error_rows_collected_this_run": collected_errors,
        }
    )
    _write_json(paths["run_summary"], summary)
    return summary


class _BatchChunk:
    def __init__(
        self,
        run_dir: Path,
        provider: str,
        api_model_condition: str,
        chunk_index: int,
        request_limit: int,
        submission_phase: str = "standard",
    ):
        self.provider = provider
        self.api_model_condition = api_model_condition
        self.chunk_index = chunk_index
        self.request_limit = request_limit
        self.submission_phase = submission_phase
        self.path = run_dir / "batches" / provider / api_model_condition / f"chunk_{chunk_index:04d}.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.count = 0
        self.bytes = 0
        self._file = tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=self.path.parent,
            prefix=f".{self.path.name}.",
            suffix=".tmp",
            delete=False,
        )
        self._temp_path = Path(self._file.name)
        self._closed = False

    def can_accept(self, item_bytes: int, max_bytes: int) -> bool:
        if self.count == 0:
            return True
        return (self.count + 1 <= self.request_limit) and (self.bytes + item_bytes <= max_bytes)

    def append(self, item: Dict[str, Any], item_bytes: int) -> None:
        self._file.write(json.dumps(item, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n")
        self.count += 1
        self.bytes += item_bytes

    def close(self) -> Dict[str, Any]:
        if self._closed:
            raise RuntimeError(f"Batch chunk already closed: {self.path}")
        self._file.flush()
        os.fsync(self._file.fileno())
        self._file.close()
        os.replace(self._temp_path, self.path)
        self._closed = True
        return {
            "provider": self.provider,
            "api_model_condition": self.api_model_condition,
            "chunk_index": self.chunk_index,
            "input_path": str(self.path),
            "request_count": self.count,
            "request_limit": self.request_limit,
            "input_bytes": self.bytes,
            "input_sha256": sha256_file(self.path),
            "submission_phase": self.submission_phase,
            "status": "prepared",
        }

    def abort(self) -> None:
        if not self._file.closed:
            self._file.close()
        if self._temp_path.exists():
            self._temp_path.unlink()


def _batch_request_item(queue_row: Dict[str, str], condition: Dict[str, Any], prompt_text: str) -> Dict[str, Any]:
    provider = condition["provider"]
    if provider == "openai":
        return {
            "custom_id": queue_row["request_id"],
            "method": "POST",
            "url": OPENAI_BATCH_ENDPOINT,
            "body": _build_openai_request_body(queue_row, condition, prompt_text),
        }
    if provider == "anthropic":
        custom_id = queue_row["request_id"]
        if not ANTHROPIC_CUSTOM_ID_RE.match(custom_id):
            raise ValueError(f"Anthropic batch custom_id is invalid: {custom_id!r}")
        return {
            "custom_id": custom_id,
            "params": _build_anthropic_request_body(queue_row, condition, prompt_text),
        }
    if provider == "google":
        return {
            "key": queue_row["request_id"],
            "request": _build_gemini_request_body(queue_row, condition, prompt_text),
        }
    raise ValueError(f"Unsupported provider for batch: {provider}")


def _provider_max_bytes(
    provider: str,
    openai_max_bytes: int,
    anthropic_max_bytes: int,
    google_max_bytes: int,
) -> int:
    if provider == "openai":
        return openai_max_bytes
    if provider == "anthropic":
        return anthropic_max_bytes
    if provider == "google":
        return google_max_bytes
    raise ValueError(f"Unsupported batch provider: {provider}")


def _provider_max_requests(
    provider: str,
    openai_max_requests: int,
    anthropic_max_requests: int,
    google_max_requests: int,
) -> int:
    if provider == "openai":
        return openai_max_requests
    if provider == "anthropic":
        return anthropic_max_requests
    if provider == "google":
        return google_max_requests
    raise ValueError(f"Unsupported batch provider: {provider}")


def _jsonl_bytes(item: Dict[str, Any]) -> int:
    return len(json.dumps(item, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")) + 1


def _submit_openai_batch(batch: Dict[str, Any]) -> Dict[str, Any]:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ApiCallError("OPENAI_API_KEY is not set")
    file_response = _openai_upload_batch_file(batch["input_path"], api_key)
    input_file_id = file_response["id"]
    response = _post_json(
        _openai_batches_endpoint(),
        {
            "input_file_id": input_file_id,
            "endpoint": OPENAI_BATCH_ENDPOINT,
            "completion_window": "24h",
            "output_expires_after": {
                "anchor": "created_at",
                "seconds": DEFAULT_OPENAI_OUTPUT_RETENTION_SECONDS,
            },
            "metadata": {
                "run_id": Path(batch["input_path"]).parents[3].name,
                "api_model_condition": batch["api_model_condition"],
                "chunk_index": str(batch["chunk_index"]),
            },
        },
        {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )
    return {
        "provider_batch_id": response.get("id"),
        "provider_input_file_id": input_file_id,
        "provider_create_response": response,
    }


def _submit_anthropic_batch(batch: Dict[str, Any]) -> Dict[str, Any]:
    requests = _read_batch_jsonl(batch["input_path"])
    response = _post_json_with_timeout(
        _anthropic_batches_endpoint(),
        {"requests": requests},
        _anthropic_headers(),
        timeout=900,
    )
    return {
        "provider_batch_id": response.get("id"),
        "provider_create_response": response,
    }


def _submit_google_batch(batch: Dict[str, Any]) -> Dict[str, Any]:
    if batch.get("input_mode", "inline") == "file":
        file_response = _google_upload_batch_file(batch["input_path"])
        file = file_response.get("file") or file_response
        input_file_name = file.get("name")
        if not input_file_name:
            raise ApiCallError(f"Google file upload returned no file name: {json.dumps(file_response)[:1000]}")
        response = _post_json_with_timeout(
            _google_batch_endpoint(batch["api_model_condition"]),
            {
                "batch": {
                    "display_name": _batch_display_name(batch),
                    "input_config": {"file_name": input_file_name},
                }
            },
            {
                "x-goog-api-key": _provider_api_key("google"),
                "Content-Type": "application/json",
            },
            timeout=900,
        )
        return {
            "provider_batch_id": response.get("name"),
            "provider_input_file_id": input_file_name,
            "provider_input_file": file,
            "provider_create_response": response,
        }

    requests = _read_batch_jsonl(batch["input_path"])
    response = _post_json_with_timeout(
        _google_batch_endpoint(batch["api_model_condition"]),
        {
            "batch": {
                "display_name": _batch_display_name(batch),
                "input_config": {
                    "requests": {
                        "requests": [
                            {
                                "request": item["request"],
                                "metadata": {"key": item.get("key") or item["custom_id"]},
                            }
                            for item in requests
                        ]
                    }
                },
            }
        },
        {
            "x-goog-api-key": _provider_api_key("google"),
            "Content-Type": "application/json",
        },
        timeout=900,
    )
    return {
        "provider_batch_id": response.get("name"),
        "provider_create_response": response,
    }


def _google_upload_batch_file(path: str) -> Dict[str, Any]:
    file_path = Path(path)
    file_size = file_path.stat().st_size
    api_key = _provider_api_key("google")
    start_request = urllib.request.Request(
        _google_files_upload_endpoint(),
        data=json.dumps({"file": {"display_name": file_path.name}}).encode("utf-8"),
        headers={
            "x-goog-api-key": api_key,
            "X-Goog-Upload-Protocol": "resumable",
            "X-Goog-Upload-Command": "start",
            "X-Goog-Upload-Header-Content-Length": str(file_size),
            "X-Goog-Upload-Header-Content-Type": "application/jsonl",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(start_request, timeout=300) as response:
            upload_url = response.headers.get("X-Goog-Upload-URL")
            chunk_granularity = int(
                response.headers.get("X-Goog-Upload-Chunk-Granularity") or 256 * 1024
            )
            response.read()
    except urllib.error.HTTPError as exc:
        payload = exc.read().decode("utf-8", errors="replace")
        raise ApiCallError(f"Google file upload start HTTP {exc.code}: {payload}") from exc
    except urllib.error.URLError as exc:
        raise ApiCallError(str(exc)) from exc
    if not upload_url:
        raise ApiCallError("Google file upload start returned no X-Goog-Upload-URL header")
    return _google_stream_upload(
        upload_url,
        file_path,
        file_size,
        chunk_granularity=chunk_granularity,
    )


def _google_stream_upload(
    upload_url: str,
    path: Path,
    file_size: int,
    chunk_granularity: int = 256 * 1024,
) -> Dict[str, Any]:
    chunk_size = max(
        chunk_granularity,
        (GOOGLE_UPLOAD_CHUNK_BYTES // chunk_granularity) * chunk_granularity,
    )
    offset = 0
    retry_count = 0
    with path.open("rb") as source:
        while offset < file_size:
            source.seek(offset)
            data = source.read(min(chunk_size, file_size - offset))
            if not data:
                raise ApiCallError(f"Google upload ended before byte {file_size}: {path}")
            final = offset + len(data) == file_size
            try:
                payload, headers = _google_upload_chunk(upload_url, offset, data, final)
            except ApiCallError as exc:
                retry_count += 1
                if retry_count > GOOGLE_UPLOAD_MAX_RETRIES:
                    raise
                time.sleep(min(2 ** (retry_count - 1), 16))
                try:
                    received, status, query_payload = _google_query_upload(upload_url)
                except ApiCallError:
                    if retry_count >= GOOGLE_UPLOAD_MAX_RETRIES:
                        raise exc
                    continue
                if status == "final" and query_payload:
                    return json.loads(query_payload.decode("utf-8"))
                if received < 0 or received > file_size:
                    raise ApiCallError(f"Google upload returned invalid resume offset: {received}")
                offset = received
                continue

            retry_count = 0
            received = int(headers.get("x-goog-upload-size-received") or offset + len(data))
            expected = offset + len(data)
            if received != expected:
                raise ApiCallError(
                    f"Google upload acknowledged byte {received}, expected {expected}"
                )
            offset = received
            print(
                f"Google upload {path.name}: {offset}/{file_size} bytes",
                file=sys.stderr,
                flush=True,
            )
            if final:
                if not payload:
                    raise ApiCallError("Google finalized upload returned an empty response")
                return json.loads(payload.decode("utf-8"))

    raise ApiCallError(f"Google upload did not finalize: {path}")


def _google_upload_chunk(
    upload_url: str,
    offset: int,
    data: bytes,
    final: bool,
) -> tuple[bytes, Dict[str, str]]:
    command = "upload, finalize" if final else "upload"
    return _google_upload_request(
        upload_url,
        data,
        {
            "Content-Length": str(len(data)),
            "X-Goog-Upload-Offset": str(offset),
            "X-Goog-Upload-Command": command,
        },
    )


def _google_query_upload(upload_url: str) -> tuple[int, str, bytes]:
    payload, headers = _google_upload_request(
        upload_url,
        b"",
        {
            "Content-Length": "0",
            "X-Goog-Upload-Command": "query",
        },
    )
    received = int(headers.get("x-goog-upload-size-received") or 0)
    return received, headers.get("x-goog-upload-status", "active"), payload


def _google_upload_request(
    upload_url: str,
    body: bytes,
    headers: Dict[str, str],
) -> tuple[bytes, Dict[str, str]]:
    parsed = urllib.parse.urlparse(upload_url)
    connection_class = http.client.HTTPSConnection if parsed.scheme == "https" else http.client.HTTPConnection
    connection = connection_class(parsed.netloc, timeout=900)
    request_path = parsed.path or "/"
    if parsed.query:
        request_path = f"{request_path}?{parsed.query}"
    try:
        connection.request("POST", request_path, body=body, headers=headers)
        response = connection.getresponse()
        payload = response.read()
        if response.status >= 400:
            raise ApiCallError(
                f"Google file upload HTTP {response.status}: {payload.decode('utf-8', errors='replace')}"
            )
        response_headers = {key.lower(): value for key, value in response.getheaders()}
        return payload, response_headers
    except OSError as exc:
        raise ApiCallError(str(exc)) from exc
    finally:
        connection.close()


def _openai_upload_batch_file(path: str, api_key: str) -> Dict[str, Any]:
    boundary = f"----api-batch-{uuid.uuid4().hex}"
    file_bytes = Path(path).read_bytes()
    file_name = Path(path).name
    parts = [
        f"--{boundary}\r\n"
        'Content-Disposition: form-data; name="purpose"\r\n\r\n'
        "batch\r\n",
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{file_name}"\r\n'
        "Content-Type: application/jsonl\r\n\r\n",
    ]
    body = "".join(parts).encode("utf-8") + file_bytes + f"\r\n--{boundary}--\r\n".encode("utf-8")
    request = urllib.request.Request(
        _openai_files_endpoint(),
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": f"multipart/form-data; boundary={boundary}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        payload = exc.read().decode("utf-8", errors="replace")
        raise ApiCallError(f"OpenAI file upload HTTP {exc.code}: {payload}") from exc
    except urllib.error.URLError as exc:
        raise ApiCallError(str(exc)) from exc


def _get_openai_batch(batch_id: str) -> Dict[str, Any]:
    return _get_json(f"{_openai_batches_endpoint().rstrip('/')}/{batch_id}", {"Authorization": f"Bearer {_require_env('OPENAI_API_KEY')}"})


def _get_anthropic_batch(batch_id: str) -> Dict[str, Any]:
    return _get_json(f"{_anthropic_batches_endpoint().rstrip('/')}/{batch_id}", _anthropic_headers())


def _get_google_batch(batch_id: str) -> Dict[str, Any]:
    name = batch_id.lstrip("/")
    if name.startswith("batches/"):
        name = name[len("batches/") :]
    return _get_json(
        f"{_google_batches_endpoint().rstrip('/')}/{urllib.parse.quote(name, safe='/')}?key={urllib.parse.quote(_provider_api_key('google'), safe='')}",
        {},
    )


def _collect_openai_output(
    output_file_id: str,
    queue: Dict[str, Dict[str, str]],
    puzzles: Dict[str, Dict[str, Any]],
    paths: Dict[str, str],
    completed: set[str],
) -> tuple[int, int]:
    data = _get_bytes(f"{_openai_files_endpoint().rstrip('/')}/{output_file_id}/content", {"Authorization": f"Bearer {_require_env('OPENAI_API_KEY')}"})
    ok = 0
    err = 0
    for line in data.decode("utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        request_id = row.get("custom_id") or ""
        if request_id in completed:
            continue
        queue_row = queue.get(request_id)
        if not queue_row:
            continue
        response_wrapper = row.get("response") or {}
        if row.get("error") or response_wrapper.get("status_code") != 200:
            _append_jsonl(paths["errors"], [{**_error_row(queue_row, 1, ApiCallError(json.dumps(row.get("error") or response_wrapper))), "batch_result": row, "collection_mode": "batch"}])
            err += 1
            continue
        response = response_wrapper.get("body") or {}
        call = {
            "request_start_time_iso": None,
            "response_end_time_iso": _utc_now(),
            "latency_ms": None,
            "retry_count": 0,
            "raw_response": _extract_openai_text(response),
            "provider_response_json": response,
            **_openai_usage(response),
        }
        _append_success(queue_row, call, puzzles, paths, completed)
        ok += 1
    return ok, err


def _collect_openai_error_file(
    error_file_id: str,
    queue: Dict[str, Dict[str, str]],
    paths: Dict[str, str],
    completed: set[str],
) -> int:
    data = _get_bytes(f"{_openai_files_endpoint().rstrip('/')}/{error_file_id}/content", {"Authorization": f"Bearer {_require_env('OPENAI_API_KEY')}"})
    count = 0
    for line in data.decode("utf-8").splitlines():
        row = json.loads(line)
        request_id = row.get("custom_id") or ""
        if request_id in completed:
            continue
        queue_row = queue.get(request_id)
        if queue_row:
            _append_jsonl(paths["errors"], [{**_error_row(queue_row, 1, ApiCallError(json.dumps(row))), "batch_result": row, "collection_mode": "batch"}])
            count += 1
    return count


def _collect_anthropic_results(
    results_url: str,
    queue: Dict[str, Dict[str, str]],
    puzzles: Dict[str, Dict[str, Any]],
    paths: Dict[str, str],
    completed: set[str],
) -> tuple[int, int]:
    data = _get_bytes(_resolve_anthropic_results_url(results_url), _anthropic_headers())
    ok = 0
    err = 0
    for line in data.decode("utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        request_id = row.get("custom_id") or ""
        if request_id in completed:
            continue
        queue_row = queue.get(request_id)
        if not queue_row:
            continue
        result = row.get("result") or {}
        if result.get("type") != "succeeded":
            _append_jsonl(paths["errors"], [{**_error_row(queue_row, 1, ApiCallError(json.dumps(result))), "batch_result": row, "collection_mode": "batch"}])
            err += 1
            continue
        response = result.get("message") or {}
        call = {
            "request_start_time_iso": None,
            "response_end_time_iso": _utc_now(),
            "latency_ms": None,
            "retry_count": 0,
            "raw_response": _extract_anthropic_text(response),
            "provider_response_json": response,
            **_anthropic_usage(response),
        }
        _append_success(queue_row, call, puzzles, paths, completed)
        ok += 1
    return ok, err


def _collect_google_results(
    batch: Dict[str, Any],
    provider_status: Dict[str, Any],
    queue: Dict[str, Dict[str, str]],
    puzzles: Dict[str, Dict[str, Any]],
    paths: Dict[str, str],
    completed: set[str],
) -> tuple[int, int]:
    state = _google_batch_state(provider_status)
    if state != "JOB_STATE_SUCCEEDED":
        err = 0
        for request_id in _batch_request_ids(batch):
            if request_id in completed:
                continue
            queue_row = queue.get(request_id)
            if queue_row:
                _append_jsonl(paths["errors"], [{**_error_row(queue_row, 1, ApiCallError(json.dumps(provider_status))), "batch_result": provider_status, "collection_mode": "batch"}])
                err += 1
        return 0, err

    response = provider_status.get("response") or provider_status.get("dest") or {}
    inline_responses = response.get("inlinedResponses") or response.get("inlined_responses") or []
    if isinstance(inline_responses, dict):
        inline_responses = (
            inline_responses.get("inlinedResponses")
            or inline_responses.get("inlined_responses")
            or []
        )
    request_ids = _batch_request_ids(batch)
    if inline_responses:
        return _collect_google_inline_responses(inline_responses, queue, puzzles, paths, completed, request_ids)

    responses_file = response.get("responsesFile") or response.get("responses_file")
    if responses_file:
        data = _get_bytes(_google_download_url(responses_file), {"x-goog-api-key": _provider_api_key("google")})
        rows = [json.loads(line) for line in data.decode("utf-8").splitlines() if line.strip()]
        return _collect_google_inline_responses(rows, queue, puzzles, paths, completed, request_ids)

    raise ApiCallError(f"Google batch succeeded but no inline responses or responsesFile found: {json.dumps(provider_status)[:1000]}")


def _collect_google_inline_responses(
    inline_responses: Sequence[Dict[str, Any]],
    queue: Dict[str, Dict[str, str]],
    puzzles: Dict[str, Dict[str, Any]],
    paths: Dict[str, str],
    completed: set[str],
    request_ids: Sequence[str],
) -> tuple[int, int]:
    ok = 0
    err = 0
    for index, row in enumerate(inline_responses):
        metadata = row.get("metadata") or {}
        request_id = (
            metadata.get("key")
            or metadata.get("custom_id")
            or row.get("key")
            or row.get("custom_id")
            or ""
        )
        if not request_id and index < len(request_ids):
            request_id = request_ids[index]
        if not request_id:
            continue
        if request_id in completed:
            continue
        queue_row = queue.get(request_id)
        if not queue_row:
            continue
        response = row.get("response") or row.get("generateContentResponse") or row
        if row.get("error") or response.get("error"):
            _append_jsonl(paths["errors"], [{**_error_row(queue_row, 1, ApiCallError(json.dumps(row.get("error") or response.get("error")))), "batch_result": row, "collection_mode": "batch"}])
            err += 1
            continue
        call = {
            "request_start_time_iso": None,
            "response_end_time_iso": _utc_now(),
            "latency_ms": None,
            "retry_count": 0,
            "raw_response": _extract_gemini_text(response),
            "provider_response_json": response,
            **_gemini_usage(response),
        }
        _append_success(queue_row, call, puzzles, paths, completed)
        ok += 1
    return ok, err


def _append_success(
    queue_row: Dict[str, str],
    call: Dict[str, Any],
    puzzles: Dict[str, Dict[str, Any]],
    paths: Dict[str, str],
    completed: set[str],
) -> None:
    parsed = _score_attempt(queue_row, call["raw_response"], puzzles)
    all_valid = all(item["is_valid"] for item in parsed)
    raw_row = {
        **_queue_metadata(queue_row),
        "source": "api",
        "attempt_index": 1,
        "max_attempts": 1,
        "prompt_file_used": queue_row["prompt_file"],
        "request_start_time_iso": call.get("request_start_time_iso"),
        "response_end_time_iso": call.get("response_end_time_iso"),
        "latency_ms": call.get("latency_ms"),
        "input_tokens": call.get("input_tokens"),
        "output_tokens": call.get("output_tokens"),
        "total_tokens": call.get("total_tokens"),
        "cached_input_tokens": call.get("cached_input_tokens"),
        "cache_write_input_tokens": call.get("cache_write_input_tokens"),
        "cache_creation_input_tokens": call.get("cache_creation_input_tokens"),
        "cache_read_input_tokens": call.get("cache_read_input_tokens"),
        "reasoning_tokens": call.get("reasoning_tokens"),
        "thinking_tokens": call.get("thinking_tokens"),
        "hidden_reasoning_tokens": call.get("hidden_reasoning_tokens"),
        "provider_response_id": call.get("provider_response_id"),
        "provider_model": call.get("provider_model"),
        "provider_status": call.get("provider_status"),
        "stop_reason": call.get("stop_reason"),
        "incomplete_reason": call.get("incomplete_reason"),
        "retry_count": call.get("retry_count", 0),
        "raw_response": call["raw_response"],
        "provider_response_json": call.get("provider_response_json"),
        "raw_answer": call["raw_response"],
        "attempt_all_valid": all_valid,
        "attempt_valid_count": sum(1 for item in parsed if item["is_valid"]),
        "attempt_item_count": len(parsed),
        "attempt_solution_ids": ";".join(item.get("solution_id") or "" for item in parsed),
        "attempt_invalid_reasons": ";".join(item.get("invalid_reason") or "" for item in parsed),
        "api_error": None,
        "collection_mode": "batch",
    }
    _append_jsonl(paths["raw"], [raw_row])
    _append_jsonl(paths["final_summary"], [_final_summary_from_attempts(queue_row, [raw_row], status="completed")])
    completed.add(queue_row["request_id"])


def _read_batch_jsonl(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _get_json(url: str, headers: Dict[str, str]) -> Dict[str, Any]:
    return json.loads(_get_bytes(url, headers).decode("utf-8"))


def _post_json_with_timeout(endpoint: str, body: Dict[str, Any], headers: Dict[str, str], timeout: int) -> Dict[str, Any]:
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(body).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        payload = exc.read().decode("utf-8", errors="replace")
        raise ApiCallError(f"HTTP {exc.code}: {payload}") from exc
    except urllib.error.URLError as exc:
        raise ApiCallError(str(exc)) from exc


def _get_bytes(url: str, headers: Dict[str, str]) -> bytes:
    request = urllib.request.Request(url, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            return response.read()
    except urllib.error.HTTPError as exc:
        payload = exc.read().decode("utf-8", errors="replace")
        raise ApiCallError(f"HTTP {exc.code}: {payload}") from exc
    except urllib.error.URLError as exc:
        raise ApiCallError(str(exc)) from exc


def _anthropic_headers() -> Dict[str, str]:
    headers = {
        "x-api-key": _require_env("ANTHROPIC_API_KEY"),
        "anthropic-version": os.environ.get("ANTHROPIC_VERSION", "2023-06-01"),
        "Content-Type": "application/json",
    }
    if os.environ.get("ANTHROPIC_BETA"):
        headers["anthropic-beta"] = os.environ["ANTHROPIC_BETA"]
    return headers


def _require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise ApiCallError(f"{name} is not set")
    return value


def _openai_files_endpoint() -> str:
    return os.environ.get("OPENAI_FILES_ENDPOINT", "https://api.openai.com/v1/files")


def _openai_batches_endpoint() -> str:
    return os.environ.get("OPENAI_BATCHES_ENDPOINT", "https://api.openai.com/v1/batches")


def _anthropic_batches_endpoint() -> str:
    return os.environ.get("ANTHROPIC_MESSAGE_BATCHES_ENDPOINT", "https://api.anthropic.com/v1/messages/batches")


def _google_batch_endpoint(api_model_condition: str) -> str:
    model_conditions = _read_json("collection/models/templates/api_model_conditions.json")
    model = model_conditions[api_model_condition]["model"]
    template = os.environ.get(
        "GEMINI_BATCH_GENERATE_CONTENT_ENDPOINT_TEMPLATE",
        "https://generativelanguage.googleapis.com/v1beta/models/{model}:batchGenerateContent",
    )
    return template.format(model=urllib.parse.quote(model, safe=""))


def _google_batches_endpoint() -> str:
    return os.environ.get("GEMINI_BATCHES_ENDPOINT", "https://generativelanguage.googleapis.com/v1beta/batches")


def _google_files_upload_endpoint() -> str:
    return os.environ.get(
        "GEMINI_FILES_UPLOAD_ENDPOINT",
        "https://generativelanguage.googleapis.com/upload/v1beta/files",
    )


def _google_download_url(file_name: str) -> str:
    name = file_name.lstrip("/")
    return (
        f"{os.environ.get('GEMINI_DOWNLOAD_ENDPOINT', 'https://generativelanguage.googleapis.com/download/v1beta').rstrip('/')}/"
        f"{urllib.parse.quote(name, safe='/')}:download?alt=media"
    )


def _google_batch_state(provider_status: Dict[str, Any]) -> str:
    metadata = provider_status.get("metadata") or {}
    response = provider_status.get("response") or {}
    state = (
        provider_status.get("state")
        or metadata.get("state")
        or response.get("state")
        or ("JOB_STATE_SUCCEEDED" if provider_status.get("done") and response else "")
    )
    if state.startswith("BATCH_STATE_"):
        return f"JOB_STATE_{state.removeprefix('BATCH_STATE_')}"
    return state


def _resolve_anthropic_results_url(results_url: str) -> str:
    if results_url.startswith("http://") or results_url.startswith("https://"):
        return results_url
    base = urllib.parse.urlparse(_anthropic_batches_endpoint())
    if results_url.startswith("/"):
        return f"{base.scheme}://{base.netloc}{results_url}"
    return f"{_anthropic_batches_endpoint().rstrip('/')}/{results_url}"


def _batch_manifest_path(run_dir: str) -> Path:
    return Path(run_dir) / "batch_manifest.json"


def _batch_prepare_config(
    *,
    model_conditions_path: str,
    providers: Sequence[str],
    api_model_conditions: Sequence[str] | None,
    limit: int,
    openai_max_bytes: int,
    anthropic_max_bytes: int,
    google_max_bytes: int,
    openai_max_requests: int,
    anthropic_max_requests: int,
    google_max_requests: int,
    google_input_mode: str,
    google_bootstrap_requests_per_condition: int,
    google_bootstrap_max_requests: int,
) -> Dict[str, Any]:
    selected_conditions = sorted(set(api_model_conditions or []))
    return {
        "providers": sorted(set(providers)),
        "api_model_conditions": selected_conditions if selected_conditions else "all",
        "limit": limit,
        "model_conditions_sha256": sha256_file(model_conditions_path),
        "openai_batch_endpoint": OPENAI_BATCH_ENDPOINT,
        "google_input_mode": google_input_mode,
        "google_bootstrap_requests_per_condition": google_bootstrap_requests_per_condition,
        "google_bootstrap_max_requests": google_bootstrap_max_requests,
        "limits": {
            "openai_max_requests": openai_max_requests,
            "openai_max_bytes": openai_max_bytes,
            "anthropic_max_requests": anthropic_max_requests,
            "anthropic_max_bytes": anthropic_max_bytes,
            "google_max_requests": google_max_requests,
            "google_max_bytes": google_max_bytes,
        },
    }


def _batch_fingerprints(queue_path: str, prepare_config: Dict[str, Any]) -> Dict[str, str]:
    return {
        "queue_sha256": sha256_file(queue_path),
        "config_sha256": stable_fingerprint(prepare_config),
    }


def _assert_batch_manifest_matches(
    manifest: Dict[str, Any],
    expected: Dict[str, str],
    model_conditions_path: str,
) -> None:
    actual = manifest.get("fingerprints")
    if not isinstance(actual, dict) or not actual.get("queue_sha256") or not actual.get("config_sha256"):
        raise RunIntegrityError(
            "Existing batch_manifest.json has no verifiable queue/config fingerprints. "
            "Archive it explicitly and prepare a fresh manifest; it will not be overwritten in place."
        )
    if actual.get("queue_sha256") != expected["queue_sha256"]:
        raise RunIntegrityError(
            "Existing batch_manifest.json queue fingerprint does not match the current queue; "
            "refusing to reuse or overwrite it."
        )
    if actual.get("config_sha256") != expected["config_sha256"]:
        raise RunIntegrityError(
            "Existing batch_manifest.json config fingerprint does not match the requested prepare config; "
            "refusing to reuse or overwrite it."
        )
    prepare_config = manifest.get("prepare_config") or {}
    if prepare_config.get("model_conditions_sha256") != sha256_file(model_conditions_path):
        raise RunIntegrityError(
            "Existing batch_manifest.json model config fingerprint does not match the current config."
        )
    _assert_batch_chunks_current(manifest)


def _assert_stored_batch_manifest_current(
    manifest: Dict[str, Any],
    queue_path: str,
    model_conditions_path: str,
) -> None:
    prepare_config = manifest.get("prepare_config")
    if not isinstance(prepare_config, dict):
        raise RunIntegrityError(
            "Existing batch_manifest.json has no stored prepare config. Archive it explicitly and prepare a fresh manifest."
        )
    expected = _batch_fingerprints(queue_path, prepare_config)
    _assert_batch_manifest_matches(manifest, expected, model_conditions_path)


def _assert_batch_chunks_current(manifest: Dict[str, Any]) -> None:
    for batch in manifest.get("batches", []):
        path = Path(str(batch.get("input_path") or ""))
        expected = batch.get("input_sha256")
        if not path.exists():
            raise RunIntegrityError(f"Prepared batch chunk is missing: {path}")
        if expected and sha256_file(path) != expected:
            raise RunIntegrityError(f"Prepared batch chunk hash mismatch: {path}")


def _has_submitted_batches(manifest: Dict[str, Any]) -> bool:
    return any(batch.get("provider_batch_id") for batch in manifest.get("batches", []))


def _formal_result_paths(paths: Dict[str, str]) -> Dict[str, str]:
    return {key: paths[key] for key in ("raw", "final_summary", "errors", "run_summary")}


def _batch_status_projection(batch: Dict[str, Any]) -> Dict[str, Any]:
    provider_status = batch.get("last_provider_status") or batch.get("provider_create_response") or {}
    return {
        "provider": batch.get("provider"),
        "api_model_condition": batch.get("api_model_condition"),
        "chunk_index": batch.get("chunk_index"),
        "request_count": batch.get("request_count"),
        "input_bytes": batch.get("input_bytes"),
        "provider_batch_id": batch.get("provider_batch_id"),
        "status": provider_status.get("status") or provider_status.get("processing_status") or _google_batch_state(provider_status) or batch.get("status"),
        "collected": bool(batch.get("collected_at_utc")),
    }


def _batch_display_name(batch: Dict[str, Any]) -> str:
    run_id = Path(batch["input_path"]).parents[3].name
    return f"{run_id}_{batch['api_model_condition']}_{batch['chunk_index']:04d}"[:128]


def _batch_request_ids(batch: Dict[str, Any]) -> List[str]:
    request_ids: List[str] = []
    with open(batch["input_path"], "r", encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            row = json.loads(line)
            request_id = row.get("key") or row.get("custom_id")
            if request_id:
                request_ids.append(request_id)
    return request_ids


def _validate_google_input_mode(input_mode: str, max_bytes: int) -> None:
    if input_mode not in GOOGLE_INPUT_MODES:
        raise ValueError(f"Unsupported Google batch input mode: {input_mode}")
    if input_mode == "inline" and max_bytes > GOOGLE_INLINE_MAX_BYTES:
        raise ValueError("Google inline batches must not exceed 20 MB")
    if input_mode == "file" and max_bytes > GOOGLE_FILE_MAX_BYTES:
        raise ValueError("Google file-input batches must not exceed 2 GB")


def _validate_google_bootstrap(requests_per_condition: int, max_requests: int) -> None:
    if requests_per_condition < 0 or max_requests < 0:
        raise ValueError("Google bootstrap request counts must be non-negative")
    if bool(requests_per_condition) != bool(max_requests):
        raise ValueError(
            "Google bootstrap requests per condition and max requests must both be zero or both be positive"
        )


def _parse_csv_arg(value: str) -> List[str]:
    if not value or value == "all":
        return []
    return [part.strip() for part in value.split(",") if part.strip()]


def _mb_to_bytes(value: float) -> int:
    return int(value * 1024 * 1024)


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare, submit, and collect provider batch API jobs for an API run.")
    parser.add_argument("action", choices=["prepare", "submit", "collect"])
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--model_conditions_config", default="collection/models/templates/api_model_conditions.json")
    parser.add_argument("--main_data", default="data/stimuli/all_puzzles.jsonl")
    parser.add_argument("--modules_data", default="data/stimuli/modules/all_module_trials.jsonl")
    parser.add_argument("--providers", default=",".join(DEFAULT_BATCH_PROVIDERS))
    parser.add_argument("--api_model_conditions", default="")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--force_prepare", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--openai_max_mb", type=float, default=180.0)
    parser.add_argument("--anthropic_max_mb", type=float, default=220.0)
    parser.add_argument("--google_max_mb", type=float, default=18.0)
    parser.add_argument(
        "--google_input_mode",
        choices=GOOGLE_INPUT_MODES,
        default="inline",
        help="Use inline requests for small batches or an uploaded JSONL file for large Google batches.",
    )
    parser.add_argument(
        "--google_bootstrap_requests_per_condition",
        type=int,
        default=0,
        help="Prepare this many initial requests per Google condition as bootstrap-phase chunks.",
    )
    parser.add_argument(
        "--google_bootstrap_max_requests",
        type=int,
        default=0,
        help="Maximum requests per Google bootstrap-phase chunk.",
    )
    parser.add_argument("--openai_max_requests", type=int, default=DEFAULT_OPENAI_MAX_REQUESTS)
    parser.add_argument("--anthropic_max_requests", type=int, default=DEFAULT_ANTHROPIC_MAX_REQUESTS)
    parser.add_argument("--google_max_requests", type=int, default=DEFAULT_GOOGLE_MAX_REQUESTS)
    parser.add_argument(
        "--max_submit_jobs",
        type=int,
        default=0,
        help="Maximum new provider jobs to submit in this invocation; 0 submits every prepared job.",
    )
    parser.add_argument(
        "--submission_phases",
        default="",
        help="Optional comma-separated batch phases to submit: bootstrap,standard.",
    )
    parser.add_argument("--collect_incomplete", action="store_true")
    args = parser.parse_args()

    providers = _parse_csv_arg(args.providers) or list(DEFAULT_BATCH_PROVIDERS)
    api_model_conditions = _parse_csv_arg(args.api_model_conditions)
    submission_phases = _parse_csv_arg(args.submission_phases)
    invalid_phases = sorted(set(submission_phases) - set(GOOGLE_SUBMISSION_PHASES))
    if invalid_phases:
        parser.error(f"Unsupported submission phases: {','.join(invalid_phases)}")
    if args.action == "prepare":
        out = prepare_api_batches(
            run_dir=args.run_dir,
            model_conditions_path=args.model_conditions_config,
            providers=providers,
            api_model_conditions=api_model_conditions,
            limit=args.limit,
            force=args.force_prepare,
            openai_max_bytes=_mb_to_bytes(args.openai_max_mb),
            anthropic_max_bytes=_mb_to_bytes(args.anthropic_max_mb),
            google_max_bytes=_mb_to_bytes(args.google_max_mb),
            openai_max_requests=args.openai_max_requests,
            anthropic_max_requests=args.anthropic_max_requests,
            google_max_requests=args.google_max_requests,
            google_input_mode=args.google_input_mode,
            google_bootstrap_requests_per_condition=args.google_bootstrap_requests_per_condition,
            google_bootstrap_max_requests=args.google_bootstrap_max_requests,
        )
    elif args.action == "submit":
        out = submit_api_batches(
            run_dir=args.run_dir,
            model_conditions_path=args.model_conditions_config,
            providers=providers,
            api_model_conditions=api_model_conditions,
            limit=args.limit,
            force_prepare=args.force_prepare,
            dry_run=args.dry_run,
            openai_max_bytes=_mb_to_bytes(args.openai_max_mb),
            anthropic_max_bytes=_mb_to_bytes(args.anthropic_max_mb),
            google_max_bytes=_mb_to_bytes(args.google_max_mb),
            openai_max_requests=args.openai_max_requests,
            anthropic_max_requests=args.anthropic_max_requests,
            google_max_requests=args.google_max_requests,
            max_submit_jobs=args.max_submit_jobs,
            google_input_mode=args.google_input_mode,
            google_bootstrap_requests_per_condition=args.google_bootstrap_requests_per_condition,
            google_bootstrap_max_requests=args.google_bootstrap_max_requests,
            submission_phases=submission_phases,
        )
    else:
        out = collect_api_batches(
            run_dir=args.run_dir,
            model_conditions_path=args.model_conditions_config,
            main_data_path=args.main_data,
            modules_data_path=args.modules_data,
            only_completed=not args.collect_incomplete,
        )
    print(json.dumps(out, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
