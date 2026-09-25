from __future__ import annotations

import argparse
import base64
import csv
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import deque
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from processing.response_parser import score_raw_answer, split_sequence_response
from collection.models.run_integrity import (
    RunIntegrityError,
    assert_no_mock_results,
    atomic_write_json,
    validate_static_assets,
)
from puzzles.common import read_jsonl


class ApiCallError(RuntimeError):
    pass


def run_api_collection(
    run_dir: str = "",
    queue_path: str = "",
    raw_path: str = "",
    final_summary_path: str = "",
    error_log_path: str = "",
    run_summary_path: str = "",
    model_conditions_path: str = "collection/models/templates/api_model_conditions.json",
    main_data_path: str = "data/stimuli/all_puzzles.jsonl",
    modules_data_path: str = "data/stimuli/modules/all_module_trials.jsonl",
    mode: str = "mock",
    mock_policy: str = "ok",
    max_retries: int = 2,
    limit: int = 0,
    workers: int = 1,
) -> Dict[str, Any]:
    if mode not in {"mock", "live"}:
        raise ValueError("mode must be mock or live")
    if workers < 1:
        raise ValueError("workers must be at least 1")
    base_paths = _resolve_paths(run_dir, queue_path, raw_path, final_summary_path, error_log_path, run_summary_path)
    if mode == "mock":
        paths = _resolve_mock_paths(
            run_dir,
            base_paths["queue"],
            raw_path,
            final_summary_path,
            error_log_path,
            run_summary_path,
        )
    else:
        paths = base_paths
        assert_no_mock_results(_formal_result_paths(paths), context="Live API collection")
        validate_static_assets(
            _run_root(run_dir, paths["queue"]),
            queue_path=paths["queue"],
            model_conditions_path=model_conditions_path,
            main_data_path=main_data_path,
            modules_data_path=modules_data_path,
        )
    queue = _read_csv(paths["queue"])
    model_conditions = _read_json(model_conditions_path)
    puzzles = {row["id"]: row for row in [*read_jsonl(main_data_path), *read_jsonl(modules_data_path)]}
    _repair_orphan_final_summaries(paths)
    completed = _completed_request_ids(paths["final_summary"])
    if workers > 1:
        raw_count, error_count = _run_concurrent_collection(
            queue,
            completed,
            paths,
            model_conditions,
            puzzles,
            mode=mode,
            mock_policy=mock_policy,
            max_retries=max_retries,
            limit=limit,
            workers=workers,
        )
        summary = _write_run_summary(paths["run_summary"], queue, paths["final_summary"], paths["errors"])
        summary.update(
            {
                "raw_attempt_rows_written_this_run": raw_count,
                "api_errors_this_run": error_count,
                "collection_mode": mode,
            }
        )
        _write_json(paths["run_summary"], summary)
        return summary

    raw_count = 0
    error_count = 0
    completed_this_run = 0

    for queue_row in queue:
        if limit and completed_this_run >= limit:
            break
        if queue_row["request_id"] in completed:
            continue
        if queue_row.get("feedback_mode") != "one_shot":
            raise ValueError("Only one_shot API collection is supported.")
        attempt_rows: List[Dict[str, Any]] = []
        final_summary: Dict[str, Any] | None = None
        max_attempts = 1
        for attempt_index in range(1, max_attempts + 1):
            prompt_file = queue_row["prompt_file"]
            prompt_text = _read_text(prompt_file)
            try:
                call = _call_with_retries(
                    queue_row,
                    model_conditions[queue_row["api_model_condition"]],
                    prompt_text,
                    mode=mode,
                    mock_policy=mock_policy,
                    max_retries=max_retries,
                    puzzles=puzzles,
                )
            except Exception as exc:
                error_count += 1
                error_row = _error_row(queue_row, attempt_index, exc)
                error_row["collection_mode"] = mode
                _append_jsonl(paths["errors"], [error_row])
                final_summary = _final_summary_from_attempts(
                    queue_row,
                    attempt_rows,
                    status="api_error",
                    api_error=str(exc),
                )
                final_summary["collection_mode"] = mode
                break

            parsed = _score_attempt(queue_row, call["raw_response"], puzzles)
            all_valid = all(item["is_valid"] for item in parsed)
            raw_row = {
                **_queue_metadata(queue_row),
                "source": "api",
                "attempt_index": attempt_index,
                "max_attempts": max_attempts,
                "prompt_file_used": prompt_file,
                "request_start_time_iso": call["request_start_time_iso"],
                "response_end_time_iso": call["response_end_time_iso"],
                "latency_ms": call["latency_ms"],
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
                "collection_mode": mode,
            }
            _append_jsonl(paths["raw"], [raw_row])
            raw_count += 1
            attempt_rows.append(raw_row)
            final_summary = _final_summary_from_attempts(queue_row, attempt_rows, status="completed")
            break

        if final_summary is None:
            final_summary = _final_summary_from_attempts(queue_row, attempt_rows, status="completed")
        _append_jsonl(paths["final_summary"], [final_summary])
        completed_this_run += 1

    summary = _write_run_summary(paths["run_summary"], queue, paths["final_summary"], paths["errors"])
    summary.update(
        {
            "raw_attempt_rows_written_this_run": raw_count,
            "api_errors_this_run": error_count,
            "collection_mode": mode,
        }
    )
    _write_json(paths["run_summary"], summary)
    return summary


def _run_concurrent_collection(
    queue: Sequence[Dict[str, str]],
    completed: set[str],
    paths: Dict[str, str],
    model_conditions: Dict[str, Dict[str, Any]],
    puzzles: Dict[str, Dict[str, Any]],
    mode: str,
    mock_policy: str,
    max_retries: int,
    limit: int,
    workers: int,
) -> Tuple[int, int]:
    pending_rows = _round_robin_pending_rows(queue, completed, limit)

    raw_count = 0
    error_count = 0
    pending = iter(pending_rows)
    in_flight: set[Future] = set()
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for _ in range(min(workers, len(pending_rows))):
            queue_row = next(pending)
            in_flight.add(
                executor.submit(
                    _collect_queue_outcome,
                    queue_row,
                    model_conditions[queue_row["api_model_condition"]],
                    puzzles,
                    mode,
                    mock_policy,
                    max_retries,
                )
            )

        while in_flight:
            finished, in_flight = wait(in_flight, return_when=FIRST_COMPLETED)
            for future in finished:
                outcome = future.result()
                if outcome["error_rows"]:
                    _append_jsonl(paths["errors"], outcome["error_rows"])
                    error_count += len(outcome["error_rows"])
                if outcome["raw_rows"]:
                    _append_jsonl(paths["raw"], outcome["raw_rows"])
                    raw_count += len(outcome["raw_rows"])
                _append_jsonl(paths["final_summary"], [outcome["final_summary"]])

                try:
                    queue_row = next(pending)
                except StopIteration:
                    continue
                in_flight.add(
                    executor.submit(
                        _collect_queue_outcome,
                        queue_row,
                        model_conditions[queue_row["api_model_condition"]],
                        puzzles,
                        mode,
                        mock_policy,
                        max_retries,
                    )
                )
    return raw_count, error_count


def _round_robin_pending_rows(
    queue: Sequence[Dict[str, str]],
    completed: set[str],
    limit: int,
) -> List[Dict[str, str]]:
    if limit < 0:
        return []

    pending_by_condition: Dict[str, deque[Dict[str, str]]] = {}
    for queue_row in queue:
        condition = queue_row["api_model_condition"]
        pending_by_condition.setdefault(condition, deque()).append(queue_row)

    pending_rows: List[Dict[str, str]] = []
    while any(pending_by_condition.values()):
        for condition_rows in pending_by_condition.values():
            if not condition_rows:
                continue
            queue_row = condition_rows.popleft()
            if queue_row["request_id"] in completed:
                continue
            if queue_row.get("feedback_mode") != "one_shot":
                raise ValueError("Only one_shot API collection is supported.")
            pending_rows.append(queue_row)
            if limit and len(pending_rows) >= limit:
                return pending_rows
    return pending_rows


def _collect_queue_outcome(
    queue_row: Dict[str, str],
    condition: Dict[str, Any],
    puzzles: Dict[str, Dict[str, Any]],
    mode: str,
    mock_policy: str,
    max_retries: int,
) -> Dict[str, Any]:
    attempt_rows: List[Dict[str, Any]] = []
    max_attempts = 1
    for attempt_index in range(1, max_attempts + 1):
        prompt_file = queue_row["prompt_file"]
        prompt_text = _read_text(prompt_file)
        try:
            call = _call_with_retries(
                queue_row,
                condition,
                prompt_text,
                mode=mode,
                mock_policy=mock_policy,
                max_retries=max_retries,
                puzzles=puzzles,
            )
        except Exception as exc:
            error_row = _error_row(queue_row, attempt_index, exc)
            error_row["collection_mode"] = mode
            final_summary = _final_summary_from_attempts(
                queue_row,
                attempt_rows,
                status="api_error",
                api_error=str(exc),
            )
            final_summary["collection_mode"] = mode
            return {
                "raw_rows": attempt_rows,
                "error_rows": [error_row],
                "final_summary": final_summary,
            }

        parsed = _score_attempt(queue_row, call["raw_response"], puzzles)
        all_valid = all(item["is_valid"] for item in parsed)
        raw_row = {
            **_queue_metadata(queue_row),
            "source": "api",
            "attempt_index": attempt_index,
            "max_attempts": max_attempts,
            "prompt_file_used": prompt_file,
            "request_start_time_iso": call["request_start_time_iso"],
            "response_end_time_iso": call["response_end_time_iso"],
            "latency_ms": call["latency_ms"],
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
            "collection_mode": mode,
        }
        attempt_rows.append(raw_row)
        return {
            "raw_rows": attempt_rows,
            "error_rows": [],
            "final_summary": _final_summary_from_attempts(queue_row, attempt_rows, status="completed"),
        }

    return {
        "raw_rows": attempt_rows,
        "error_rows": [],
        "final_summary": _final_summary_from_attempts(queue_row, attempt_rows, status="completed"),
    }


def _call_with_retries(
    queue_row: Dict[str, str],
    condition: Dict[str, Any],
    prompt_text: str,
    mode: str,
    mock_policy: str,
    max_retries: int,
    puzzles: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    last_error: Exception | None = None
    flex_resource_retry_limit = _flex_resource_retry_limit(condition) if mode == "live" else 0
    retry_ceiling = max(max_retries, flex_resource_retry_limit)
    for retry_count in range(retry_ceiling + 1):
        start = _utc_now()
        start_time = time.time()
        try:
            if mode == "mock":
                raw_response = _mock_response(queue_row, puzzles, mock_policy)
                output_tokens = len(raw_response.split())
                usage = {
                    "input_tokens": 0,
                    "output_tokens": output_tokens,
                    "total_tokens": output_tokens,
                    "cached_input_tokens": 0,
                    "cache_write_input_tokens": 0,
                    "cache_creation_input_tokens": 0,
                    "cache_read_input_tokens": 0,
                    "reasoning_tokens": 0,
                    "thinking_tokens": 0,
                    "hidden_reasoning_tokens": 0,
                    "provider_status": "mock_completed",
                }
                provider_response = {"mock_policy": mock_policy, "raw_response": raw_response}
            elif mode == "live":
                raw_response, usage, provider_response = _live_response(queue_row, condition, prompt_text)
            else:
                raise ValueError("mode must be mock or live")
            end = _utc_now()
            return {
                "request_start_time_iso": start,
                "response_end_time_iso": end,
                "latency_ms": int(round((time.time() - start_time) * 1000)),
                "retry_count": retry_count,
                "raw_response": raw_response,
                "provider_response_json": provider_response,
                **usage,
            }
        except Exception as exc:  # noqa: PERF203 - retry clarity matters here
            last_error = exc
            retry_limit = (
                flex_resource_retry_limit
                if mode == "live" and _is_flex_resource_unavailable(condition, exc)
                else max_retries
            )
            if retry_count < retry_limit:
                time.sleep(_retry_delay_seconds(condition, exc, retry_count))
            else:
                break
    assert last_error is not None
    raise last_error


def _flex_resource_retry_limit(condition: Dict[str, Any]) -> int:
    fallback = condition.get("live_fallback") or {}
    if fallback.get("service_tier") != "flex":
        return 0
    return max(0, int(fallback.get("resource_unavailable_retries", 0)))


def _is_flex_resource_unavailable(condition: Dict[str, Any], exc: Exception) -> bool:
    fallback = condition.get("live_fallback") or {}
    return fallback.get("service_tier") == "flex" and "HTTP 429" in str(exc)


def _retry_delay_seconds(condition: Dict[str, Any], exc: Exception, retry_count: int) -> float:
    if _is_flex_resource_unavailable(condition, exc):
        fallback = condition.get("live_fallback") or {}
        initial = max(0.0, float(fallback.get("retry_initial_seconds", 2.0)))
        maximum = max(initial, float(fallback.get("retry_max_seconds", 60.0)))
        return min(maximum, initial * (2**retry_count))
    return min(2.0, 0.25 * (2**retry_count))


def _live_response(queue_row: Dict[str, str], condition: Dict[str, Any], prompt_text: str) -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
    provider = condition["provider"]
    if provider == "openai":
        return _openai_response(queue_row, condition, prompt_text)
    if provider == "anthropic":
        return _anthropic_response(queue_row, condition, prompt_text)
    if provider == "google":
        return _gemini_response(queue_row, condition, prompt_text)
    raise ApiCallError(f"Unsupported provider: {provider}")


def _openai_response(queue_row: Dict[str, str], condition: Dict[str, Any], prompt_text: str) -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ApiCallError("OPENAI_API_KEY is not set")
    body = _build_openai_request_body(queue_row, condition, prompt_text, include_live_fallback=True)
    endpoint = os.environ.get("OPENAI_RESPONSES_ENDPOINT", "https://api.openai.com/v1/responses")
    live_fallback = condition.get("live_fallback") or {}
    timeout = float(live_fallback.get("timeout_seconds", 120))
    response = _post_json(
        endpoint,
        body,
        {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        timeout=timeout,
    )
    return _extract_openai_text(response), _openai_usage(response), response


def _build_openai_request_body(
    queue_row: Dict[str, str],
    condition: Dict[str, Any],
    prompt_text: str,
    include_live_fallback: bool = False,
) -> Dict[str, Any]:
    image_data = _image_data_url(queue_row["image_path"])
    body: Dict[str, Any] = {
        "model": condition["model"],
        "input": [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt_text},
                    {
                        "type": "input_image",
                        "image_url": image_data,
                        "detail": condition.get("image_detail", "high"),
                    },
                ],
            }
        ],
        "max_output_tokens": condition.get("max_output_tokens", 512),
        "stream": False,
    }
    for key in ("temperature", "top_p", "reasoning", "text", "tools"):
        if key in condition:
            body[key] = condition[key]
    if include_live_fallback:
        service_tier = (condition.get("live_fallback") or {}).get("service_tier")
        if condition.get("model") == "gpt-5.6-sol" and service_tier != "flex":
            raise ApiCallError("Refusing GPT-5.6 Sol direct collection without service_tier=flex")
        if service_tier:
            body["service_tier"] = service_tier
    return body


def _anthropic_response(queue_row: Dict[str, str], condition: Dict[str, Any], prompt_text: str) -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ApiCallError("ANTHROPIC_API_KEY is not set")
    body = _build_anthropic_request_body(queue_row, condition, prompt_text)
    endpoint = os.environ.get("ANTHROPIC_MESSAGES_ENDPOINT", "https://api.anthropic.com/v1/messages")
    response = _post_json(
        endpoint,
        body,
        {
            "x-api-key": api_key,
            "anthropic-version": os.environ.get("ANTHROPIC_VERSION", "2023-06-01"),
            "Content-Type": "application/json",
        },
    )
    return _extract_anthropic_text(response), _anthropic_usage(response), response


def _build_anthropic_request_body(queue_row: Dict[str, str], condition: Dict[str, Any], prompt_text: str) -> Dict[str, Any]:
    image_b64 = _image_base64(queue_row["image_path"])
    body: Dict[str, Any] = {
        "model": condition["model"],
        "max_tokens": condition.get("max_tokens", 512),
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {"type": "base64", "media_type": "image/png", "data": image_b64},
                    },
                    {"type": "text", "text": prompt_text},
                ],
            }
        ],
    }
    omitted = set(condition.get("omit_parameters", []))
    for key in ("temperature", "top_p", "top_k", "thinking", "output_config", "tools"):
        if key in condition and key not in omitted:
            body[key] = condition[key]
    return body


def _gemini_response(queue_row: Dict[str, str], condition: Dict[str, Any], prompt_text: str) -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
    api_key = _provider_api_key("google")
    endpoint_template = os.environ.get(
        "GEMINI_GENERATE_CONTENT_ENDPOINT_TEMPLATE",
        "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}",
    )
    endpoint = endpoint_template.format(
        model=urllib.parse.quote(condition["model"], safe=""),
        api_key=urllib.parse.quote(api_key, safe=""),
    )
    response = _post_json(
        endpoint,
        _build_gemini_request_body(queue_row, condition, prompt_text),
        {"Content-Type": "application/json"},
    )
    return _extract_gemini_text(response), _gemini_usage(response), response


def _build_gemini_request_body(queue_row: Dict[str, str], condition: Dict[str, Any], prompt_text: str) -> Dict[str, Any]:
    generation_config: Dict[str, Any] = {
        "maxOutputTokens": condition.get("max_output_tokens", condition.get("max_tokens", 512)),
    }
    thinking = condition.get("thinking") or {}
    effort = thinking.get("effort")
    if thinking.get("type") == "level" and effort:
        generation_config["thinkingConfig"] = {"thinkingLevel": str(effort).upper()}
    return {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"text": prompt_text},
                    {
                        "inlineData": {
                            "mimeType": "image/png",
                            "data": _image_base64(queue_row["image_path"]),
                        }
                    },
                ],
            }
        ],
        "generationConfig": generation_config,
    }


def _post_json(
    endpoint: str,
    body: Dict[str, Any],
    headers: Dict[str, str],
    timeout: float = 120,
) -> Dict[str, Any]:
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


def _extract_openai_text(response: Dict[str, Any]) -> str:
    if isinstance(response.get("output_text"), str):
        return response["output_text"].strip()
    parts: List[str] = []
    for item in response.get("output", []) or []:
        for content in item.get("content", []) or []:
            text = content.get("text") or content.get("output_text")
            if isinstance(text, str):
                parts.append(text)
    return "\n".join(parts).strip()


def _extract_anthropic_text(response: Dict[str, Any]) -> str:
    parts: List[str] = []
    for block in response.get("content", []) or []:
        if block.get("type") == "text" and isinstance(block.get("text"), str):
            parts.append(block["text"])
    return "\n".join(parts).strip()


def _extract_gemini_text(response: Dict[str, Any]) -> str:
    parts: List[str] = []
    for candidate in response.get("candidates", []) or []:
        content = candidate.get("content") or {}
        for part in content.get("parts", []) or []:
            if isinstance(part.get("text"), str):
                parts.append(part["text"])
    return "\n".join(parts).strip()


def _openai_usage(response: Dict[str, Any]) -> Dict[str, Any]:
    usage = response.get("usage") or {}
    input_details = usage.get("input_tokens_details") or {}
    output_details = usage.get("output_tokens_details") or {}
    reasoning_tokens = output_details.get("reasoning_tokens")
    incomplete = response.get("incomplete_details") or {}
    return {
        "input_tokens": usage.get("input_tokens"),
        "output_tokens": usage.get("output_tokens"),
        "total_tokens": usage.get("total_tokens"),
        "cached_input_tokens": input_details.get("cached_tokens"),
        "cache_write_input_tokens": input_details.get("cache_write_tokens"),
        "cache_creation_input_tokens": input_details.get("cache_write_tokens"),
        "cache_read_input_tokens": input_details.get("cached_tokens"),
        "reasoning_tokens": reasoning_tokens,
        "thinking_tokens": None,
        "hidden_reasoning_tokens": reasoning_tokens,
        "provider_response_id": response.get("id"),
        "provider_model": response.get("model"),
        "provider_status": response.get("status"),
        "stop_reason": None,
        "incomplete_reason": incomplete.get("reason") if isinstance(incomplete, dict) else None,
    }


def _anthropic_usage(response: Dict[str, Any]) -> Dict[str, Any]:
    usage = response.get("usage") or {}
    output_details = usage.get("output_tokens_details") or {}
    thinking_tokens = output_details.get("thinking_tokens")
    return {
        "input_tokens": usage.get("input_tokens"),
        "output_tokens": usage.get("output_tokens"),
        "total_tokens": _sum_ints(usage.get("input_tokens"), usage.get("output_tokens")),
        "cached_input_tokens": usage.get("cache_read_input_tokens"),
        "cache_creation_input_tokens": usage.get("cache_creation_input_tokens"),
        "cache_read_input_tokens": usage.get("cache_read_input_tokens"),
        "reasoning_tokens": None,
        "thinking_tokens": thinking_tokens,
        "hidden_reasoning_tokens": thinking_tokens,
        "provider_response_id": response.get("id"),
        "provider_model": response.get("model"),
        "provider_status": response.get("type"),
        "stop_reason": response.get("stop_reason"),
        "incomplete_reason": "max_tokens" if response.get("stop_reason") == "max_tokens" else None,
    }


def _gemini_usage(response: Dict[str, Any]) -> Dict[str, Any]:
    usage = response.get("usageMetadata") or {}
    thinking_tokens = usage.get("thoughtsTokenCount")
    candidate = (response.get("candidates") or [{}])[0] if isinstance(response.get("candidates"), list) else {}
    finish_reason = candidate.get("finishReason") if isinstance(candidate, dict) else None
    return {
        "input_tokens": usage.get("promptTokenCount"),
        "output_tokens": usage.get("candidatesTokenCount"),
        "total_tokens": usage.get("totalTokenCount"),
        "cached_input_tokens": usage.get("cachedContentTokenCount"),
        "cache_creation_input_tokens": None,
        "cache_read_input_tokens": usage.get("cachedContentTokenCount"),
        "reasoning_tokens": thinking_tokens,
        "thinking_tokens": thinking_tokens,
        "hidden_reasoning_tokens": thinking_tokens,
        "provider_response_id": response.get("responseId"),
        "provider_model": response.get("modelVersion"),
        "provider_status": response.get("responseId"),
        "stop_reason": finish_reason,
        "incomplete_reason": "MAX_TOKENS" if finish_reason == "MAX_TOKENS" else None,
    }


def _sum_ints(*values: Any) -> int | None:
    total = 0
    seen = False
    for value in values:
        if value in (None, ""):
            continue
        total += int(value)
        seen = True
    return total if seen else None


def _provider_api_key(provider: str) -> str:
    names = {
        "google": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
    }.get(provider, ())
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    raise ApiCallError(f"None of the API key environment variables are set for provider {provider}: {', '.join(names)}")


def _mock_response(queue_row: Dict[str, str], puzzles: Dict[str, Dict[str, Any]], policy: str) -> str:
    if policy == "api_error":
        raise ApiCallError("Mock API error")
    if policy == "malformed":
        return "I am not sure."
    trial_ids = _split_ids(queue_row["trial_ids"])
    if policy == "invalid":
        if queue_row["input_kind"] == "sequence":
            return "A: cell=<99,99>\nB: cell=<99,99>"
        return "ANSWER: cell=<99,99>"
    if queue_row["input_kind"] == "sequence":
        labels = ["A", "B"]
        return "\n".join(
            f"{label}: {_format_answer(puzzles[trial_id], sequence=True)}"
            for label, trial_id in zip(labels, trial_ids)
        )
    puzzle = puzzles[trial_ids[0]]
    return f"ANSWER: {_format_answer(puzzle, sequence=True)}"


def _format_answer(puzzle: Dict[str, Any], sequence: bool = False) -> str:
    solution = puzzle["solutions"][0]
    ptype = puzzle["puzzle_type"]
    if ptype == "arithmetic24":
        return f"expression={solution['expression']}"
    if ptype == "maze":
        return f"path={solution['moves']}"
    if ptype == "grid_placement":
        coords = "; ".join(f"<{r},{c}>" for r, c in solution["coordinates"])
        return f"cells={coords}"
    if ptype in {"minesweeper_lite", "mini_sudoku"}:
        r, c = solution["coordinate"]
        return f"cell=<{r},{c}>"
    raise ValueError(f"Unknown puzzle type: {ptype}")


def _score_attempt(queue_row: Dict[str, str], raw_response: str, puzzles: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    trial_ids = _split_ids(queue_row["trial_ids"])
    if queue_row["input_kind"] == "sequence":
        split = split_sequence_response(raw_response, len(trial_ids))
        out = []
        for idx, trial_id in enumerate(trial_ids):
            answer = split["answers"][idx] if idx < len(split["answers"]) else ""
            parsed = score_raw_answer(puzzles[trial_id], answer, raw_response=raw_response)
            out.append(parsed)
        return out
    trial_id = trial_ids[0]
    return [score_raw_answer(puzzles[trial_id], raw_response, raw_response=raw_response)]


def _final_summary_from_attempts(
    queue_row: Dict[str, str],
    attempts: Sequence[Dict[str, Any]],
    status: str,
    api_error: str | None = None,
) -> Dict[str, Any]:
    first = attempts[0] if attempts else {}
    final = attempts[-1] if attempts else {}
    return {
        **_queue_metadata(queue_row),
        "status": status,
        "num_attempts": len(attempts),
        "first_attempt_valid": first.get("attempt_all_valid") if first else False,
        "final_valid": final.get("attempt_all_valid") if final else False,
        "failed_after_max_attempts": status == "failed_after_max_attempts",
        "first_attempt_solution_ids": first.get("attempt_solution_ids"),
        "final_solution_ids": final.get("attempt_solution_ids"),
        "first_attempt_invalid_reasons": first.get("attempt_invalid_reasons"),
        "final_invalid_reasons": final.get("attempt_invalid_reasons"),
        "input_tokens": final.get("input_tokens"),
        "output_tokens": final.get("output_tokens"),
        "total_tokens": final.get("total_tokens"),
        "cached_input_tokens": final.get("cached_input_tokens"),
        "cache_write_input_tokens": final.get("cache_write_input_tokens"),
        "cache_creation_input_tokens": final.get("cache_creation_input_tokens"),
        "cache_read_input_tokens": final.get("cache_read_input_tokens"),
        "reasoning_tokens": final.get("reasoning_tokens"),
        "thinking_tokens": final.get("thinking_tokens"),
        "hidden_reasoning_tokens": final.get("hidden_reasoning_tokens"),
        "provider_response_id": final.get("provider_response_id"),
        "provider_model": final.get("provider_model"),
        "provider_status": final.get("provider_status"),
        "stop_reason": final.get("stop_reason"),
        "incomplete_reason": final.get("incomplete_reason"),
        "api_error": api_error,
        "collection_mode": final.get("collection_mode") or first.get("collection_mode"),
        "completed_at_utc": _utc_now(),
    }


def _queue_metadata(queue_row: Dict[str, str]) -> Dict[str, Any]:
    keys = [
        "run_id",
        "request_id",
        "sample_index",
        "feedback_mode",
        "api_model_condition",
        "api_provider",
        "model",
        "sampling_profile",
        "temperature",
        "top_p",
        "output_token_cap",
        "max_output_tokens",
        "max_tokens",
        "reasoning_effort",
        "thinking_type",
        "thinking_effort",
        "dataset_name",
        "module",
        "condition",
        "image_id",
        "puzzle_id",
        "sequence_id",
        "trial_ids",
        "input_kind",
        "puzzle_type",
        "prompt_condition",
        "prompt_version",
        "prompt_file",
        "prompt_sha256",
        "image_path",
        "image_sha256",
        "render_version",
    ]
    out = {key: queue_row.get(key) for key in keys}
    for int_key in ("sample_index",):
        if out.get(int_key) not in (None, ""):
            out[int_key] = int(out[int_key])
    return out


def _error_row(queue_row: Dict[str, str], attempt_index: int, exc: Exception) -> Dict[str, Any]:
    return {
        **_queue_metadata(queue_row),
        "attempt_index": attempt_index,
        "error_type": exc.__class__.__name__,
        "error_message": str(exc),
        "logged_at_utc": _utc_now(),
    }


def _write_run_summary(
    path: str,
    queue: Sequence[Dict[str, str]],
    final_summary_path: str,
    errors_path: str,
) -> Dict[str, Any]:
    summaries = _read_jsonl_if_exists(final_summary_path)
    errors = _read_jsonl_if_exists(errors_path)
    completed_by_request = _latest_completed_by_request(summaries)
    completed = len(completed_by_request)
    one_shot = [row for row in completed_by_request.values() if row.get("feedback_mode") == "one_shot"]
    error_request_ids = {str(row.get("request_id") or "") for row in errors if row.get("request_id")}
    unresolved_error_request_ids = error_request_ids - set(completed_by_request)
    return {
        "status": "collection_started" if completed < len(queue) else "collection_complete",
        "queue_rows": len(queue),
        "completed_requests": completed,
        "remaining_requests": max(0, len(queue) - completed),
        "api_error_count": len(errors),
        "api_error_request_count": len(error_request_ids),
        "unresolved_api_error_request_count": len(unresolved_error_request_ids),
        "api_error_rate": _safe_rate(len(unresolved_error_request_ids), max(1, len(queue))),
        "one_shot_valid_rate": _mean_bool(row.get("final_valid") for row in one_shot),
        "updated_at_utc": _utc_now(),
    }


def _repair_orphan_final_summaries(paths: Dict[str, str]) -> int:
    """Recover requests that wrote raw rows before final summary rows.

    This makes resume idempotent after an interruption between appending
    `raw.jsonl` and appending `final_summary.jsonl`.
    """
    completed = _completed_request_ids(paths["final_summary"])
    raw_rows = _read_jsonl_if_exists(paths["raw"])
    by_request: Dict[str, List[Dict[str, Any]]] = {}
    for row in raw_rows:
        request_id = str(row.get("request_id") or "")
        if not request_id or request_id in completed:
            continue
        by_request.setdefault(request_id, []).append(row)
    repaired: List[Dict[str, Any]] = []
    for request_id, attempts in sorted(by_request.items()):
        attempts.sort(key=lambda row: int(row.get("attempt_index") or 0))
        repaired.append(_final_summary_from_attempts(attempts[0], attempts, status="completed"))
        completed.add(request_id)
    if repaired:
        _append_jsonl(paths["final_summary"], repaired)
    return len(repaired)


def _completed_request_ids(path: str) -> set[str]:
    return set(_latest_completed_by_request(_read_jsonl_if_exists(path)))


def _latest_completed_by_request(rows: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    completed: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        request_id = str(row.get("request_id") or "")
        if request_id and row.get("status") == "completed":
            completed[request_id] = row
    return completed


def _resolve_paths(
    run_dir: str,
    queue_path: str,
    raw_path: str,
    final_summary_path: str,
    error_log_path: str,
    run_summary_path: str,
) -> Dict[str, str]:
    if run_dir:
        root = Path(run_dir)
        return {
            "queue": queue_path or str(root / "queue.csv"),
            "raw": raw_path or str(root / "raw.jsonl"),
            "final_summary": final_summary_path or str(root / "final_summary.jsonl"),
            "errors": error_log_path or str(root / "errors.jsonl"),
            "run_summary": run_summary_path or str(root / "run_summary.json"),
        }
    if not queue_path:
        raise ValueError("Either --run_dir or --queue is required.")
    root = Path(queue_path).parent
    return {
        "queue": queue_path,
        "raw": raw_path or str(root / "raw.jsonl"),
        "final_summary": final_summary_path or str(root / "final_summary.jsonl"),
        "errors": error_log_path or str(root / "errors.jsonl"),
        "run_summary": run_summary_path or str(root / "run_summary.json"),
    }


def _resolve_mock_paths(
    run_dir: str,
    queue_path: str,
    raw_path: str,
    final_summary_path: str,
    error_log_path: str,
    run_summary_path: str,
) -> Dict[str, str]:
    root = _run_root(run_dir, queue_path)
    mock_root = root / "mock_preflight"
    if mock_root.is_symlink():
        raise RunIntegrityError(f"Mock output directory must not be a symlink: {mock_root}")
    requested = {
        "raw": raw_path,
        "final_summary": final_summary_path,
        "errors": error_log_path,
        "run_summary": run_summary_path,
    }
    defaults = {
        "raw": mock_root / "raw.jsonl",
        "final_summary": mock_root / "final_summary.jsonl",
        "errors": mock_root / "errors.jsonl",
        "run_summary": mock_root / "run_summary.json",
    }
    resolved: Dict[str, str] = {"queue": queue_path}
    canonical_root = mock_root.resolve(strict=False)
    for key, requested_path in requested.items():
        candidate = Path(requested_path) if requested_path else defaults[key]
        try:
            candidate.resolve(strict=False).relative_to(canonical_root)
        except ValueError as exc:
            raise RunIntegrityError(
                f"Mock {key} output must stay under {mock_root}; received {candidate}."
            ) from exc
        resolved[key] = str(candidate)
    return resolved


def _run_root(run_dir: str, queue_path: str) -> Path:
    return Path(run_dir) if run_dir else Path(queue_path).parent


def _formal_result_paths(paths: Dict[str, str]) -> Dict[str, str]:
    return {key: paths[key] for key in ("raw", "final_summary", "errors", "run_summary")}


def _read_csv(path: str | Path) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _read_json(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: str | Path, obj: Any) -> None:
    atomic_write_json(path, obj)


def _read_text(path: str | Path) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def _append_jsonl(path: str | Path, rows: Sequence[Dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")


def _read_jsonl_if_exists(path: str | Path) -> List[Dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _split_ids(value: str) -> List[str]:
    return [part.strip() for part in str(value).split(";") if part.strip()]


@lru_cache(maxsize=None)
def _image_base64(path: str | Path) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")


@lru_cache(maxsize=None)
def _image_data_url(path: str | Path) -> str:
    return "data:image/png;base64," + _image_base64(path)


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _safe_rate(numer: int, denom: int) -> float | None:
    if denom == 0:
        return None
    return numer / denom


def _mean_bool(values: Sequence[Any]) -> float | None:
    vals = [bool(value) for value in values if value is not None]
    if not vals:
        return None
    return sum(1 for value in vals if value) / len(vals)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run or mock API collection from an API queue.")
    parser.add_argument("--run_dir", default="")
    parser.add_argument("--queue", default="")
    parser.add_argument("--raw", default="")
    parser.add_argument("--final_summary", default="")
    parser.add_argument("--errors", default="")
    parser.add_argument("--run_summary", default="")
    parser.add_argument("--model_conditions_config", default="collection/models/templates/api_model_conditions.json")
    parser.add_argument("--main_data", default="data/stimuli/all_puzzles.jsonl")
    parser.add_argument("--modules_data", default="data/stimuli/modules/all_module_trials.jsonl")
    parser.add_argument("--mode", choices=["mock", "live"], default="mock")
    parser.add_argument("--mock_policy", choices=["ok", "invalid", "malformed", "api_error"], default="ok")
    parser.add_argument("--max_retries", type=int, default=2)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    summary = run_api_collection(
        run_dir=args.run_dir,
        queue_path=args.queue,
        raw_path=args.raw,
        final_summary_path=args.final_summary,
        error_log_path=args.errors,
        run_summary_path=args.run_summary,
        model_conditions_path=args.model_conditions_config,
        main_data_path=args.main_data,
        modules_data_path=args.modules_data,
        mode=args.mode,
        mock_policy=args.mock_policy,
        max_retries=args.max_retries,
        limit=args.limit,
        workers=args.workers,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
