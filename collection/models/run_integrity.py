from __future__ import annotations

import csv
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from puzzles.common import read_jsonl, stable_hash


class RunIntegrityError(ValueError):
    pass


REQUIRED_MANIFEST_HASHES = {
    "main_data_stable_hash",
    "modules_data_stable_hash",
    "module_blocks_stable_hash",
    "dataset_manifest_sha256",
    "image_manifest_sha256",
    "prompt_manifest_sha256",
    "model_conditions_sha256",
    "response_parser_sha256",
    "selected_model_conditions_stable_hash",
    "prompt_file_sha256",
}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_fingerprint(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def atomic_write_json(path: str | Path, value: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temp_path = Path(handle.name)
            json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, destination)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()


def atomic_write_jsonl(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temp_path = Path(handle.name)
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, destination)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()


def ensure_distinct_output(
    output_path: str | Path,
    source_paths: Mapping[str, str | Path],
    *,
    output_label: str = "output",
) -> None:
    output = _canonical_path(output_path)
    for label, source_path in source_paths.items():
        if output == _canonical_path(source_path):
            raise RunIntegrityError(
                f"Refusing to overwrite source {label}: {output_path}. "
                f"Choose a distinct {output_label} path."
            )


def is_mock_row(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    if str(value.get("collection_mode") or "").lower() == "mock":
        return True
    if str(value.get("mode") or "").lower() == "mock" or str(value.get("source") or "").lower() == "mock":
        return True
    if value.get("is_mock") is True or "mock_policy" in value:
        return True
    provider_status = str(value.get("provider_status") or "").lower()
    if provider_status == "mock" or provider_status.startswith("mock_"):
        return True
    legacy_error = " ".join(str(value.get(key) or "") for key in ("api_error", "error_message")).lower()
    if "mock api" in legacy_error:
        return True
    provider_response = value.get("provider_response_json")
    return isinstance(provider_response, dict) and "mock_policy" in provider_response


def assert_no_mock_results(paths: Mapping[str, str | Path], *, context: str) -> None:
    for label, raw_path in paths.items():
        path = Path(raw_path)
        if not path.exists():
            continue
        try:
            if path.suffix == ".jsonl":
                with path.open("r", encoding="utf-8") as handle:
                    for line_number, line in enumerate(handle, 1):
                        if line.strip() and is_mock_row(json.loads(line)):
                            raise RunIntegrityError(
                                f"{context} refused: mock row found in formal {label} at "
                                f"{path}:{line_number}. Move mock output to mock_preflight."
                            )
            else:
                with path.open("r", encoding="utf-8") as handle:
                    value = json.load(handle)
                if _contains_mock_marker(value):
                    raise RunIntegrityError(
                        f"{context} refused: mock marker found in formal {label} at {path}. "
                        "Move mock output to mock_preflight."
                    )
        except RunIntegrityError:
            raise
        except (OSError, json.JSONDecodeError) as exc:
            raise RunIntegrityError(f"{context} could not validate formal {label} at {path}: {exc}") from exc


def validate_static_assets(
    run_dir: str | Path,
    *,
    queue_path: str | Path | None = None,
    model_conditions_path: str | Path | None = None,
    main_data_path: str | Path | None = None,
    modules_data_path: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(run_dir)
    manifest_path = root / "run_manifest.json"
    if not manifest_path.exists():
        raise RunIntegrityError(
            f"Static asset validation requires {manifest_path}."
        )
    try:
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise RunIntegrityError(f"Could not read run manifest {manifest_path}: {exc}") from exc

    hashes = manifest.get("hashes")
    if not isinstance(hashes, dict):
        hashes = {}
    missing_hashes = sorted(REQUIRED_MANIFEST_HASHES - set(hashes))
    if missing_hashes:
        raise RunIntegrityError(
            "Run manifest is too old for static asset validation; missing hashes: "
            + ", ".join(missing_hashes)
            + ". Rebuild the run manifest."
        )

    paths = manifest.get("paths") if isinstance(manifest.get("paths"), dict) else {}
    overrides: dict[str, str | Path | None] = {
        "queue": queue_path,
        "model_conditions": model_conditions_path,
        "main_data": main_data_path,
        "modules_data": modules_data_path,
    }
    errors: list[str] = []

    stable_specs = (
        ("main_data_stable_hash", "main_data"),
        ("modules_data_stable_hash", "modules_data"),
        ("module_blocks_stable_hash", "blocks"),
    )
    for hash_key, path_key in stable_specs:
        if hash_key not in hashes:
            continue
        path = _asset_path(overrides.get(path_key) or paths.get(path_key), root)
        _check_hash(
            errors,
            path_key,
            path,
            hashes[hash_key],
            lambda current: stable_hash(read_jsonl(str(current))),
        )

    sha_specs = (
        ("dataset_manifest_sha256", "dataset_manifest"),
        ("image_manifest_sha256", "image_manifest"),
        ("prompt_manifest_sha256", "prompt_manifest"),
        ("model_conditions_sha256", "model_conditions"),
    )
    for hash_key, path_key in sha_specs:
        if hash_key not in hashes:
            continue
        path = _asset_path(overrides.get(path_key) or paths.get(path_key), root)
        _check_hash(errors, path_key, path, hashes[hash_key], sha256_file)

    if "response_parser_sha256" in hashes:
        parser_path = Path(__file__).resolve().parents[2] / "processing" / "response_parser.py"
        _check_hash(errors, "response_parser", parser_path, hashes["response_parser_sha256"], sha256_file)

    model_path = _asset_path(overrides.get("model_conditions") or paths.get("model_conditions"), root)
    if "selected_model_conditions_stable_hash" in hashes and model_path is not None:
        try:
            with model_path.open("r", encoding="utf-8") as handle:
                model_config = json.load(handle)
            selected_names = manifest.get("model_conditions") or []
            selected = {name: model_config[name] for name in selected_names if name in model_config}
            actual = stable_hash(selected)
            if actual != hashes["selected_model_conditions_stable_hash"]:
                errors.append(f"selected model conditions hash mismatch for {model_path}")
        except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
            errors.append(f"could not validate selected model conditions at {model_path}: {exc}")

    prompt_hashes = hashes.get("prompt_file_sha256")
    if isinstance(prompt_hashes, dict):
        for prompt_path, expected in prompt_hashes.items():
            path = _asset_path(prompt_path, root)
            _check_hash(errors, "prompt file", path, expected, sha256_file)
    elif "prompt_file_sha256" in hashes:
        errors.append("manifest prompt_file_sha256 must be an object mapping paths to hashes")

    resolved_queue = _asset_path(overrides.get("queue") or paths.get("queue") or root / "queue.csv", root)
    if resolved_queue is not None:
        _validate_queue_assets(resolved_queue, root, errors)

    if errors:
        raise RunIntegrityError("Static asset drift detected:\n- " + "\n- ".join(errors))
    return manifest


def _validate_queue_assets(
    queue_path: Path,
    run_dir: Path,
    errors: list[str],
) -> None:
    if not queue_path.exists():
        errors.append(f"queue is missing: {queue_path}")
        return
    try:
        with queue_path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
    except OSError as exc:
        errors.append(f"could not read queue {queue_path}: {exc}")
        return

    assets: dict[tuple[str, str], str] = {}
    for row_number, row in enumerate(rows, 2):
        for label, path_key, hash_key in (
            ("image", "image_path", "image_sha256"),
            ("prompt", "prompt_file", "prompt_sha256"),
        ):
            raw_path = row.get(path_key)
            expected = row.get(hash_key)
            if not raw_path or not expected:
                errors.append(f"queue row {row_number} is missing {path_key} or {hash_key}")
                continue
            key = (label, str(_asset_path(raw_path, run_dir)))
            previous = assets.get(key)
            if previous is not None and previous != expected:
                errors.append(f"queue contains conflicting {hash_key} values for {raw_path}")
            assets[key] = expected
    for (label, raw_path), expected in assets.items():
        _check_hash(errors, f"queue {label}", Path(raw_path), expected, sha256_file)


def _check_hash(
    errors: list[str],
    label: str,
    path: Path | None,
    expected: Any,
    compute,
) -> None:
    if path is None:
        errors.append(f"manifest path is missing for {label}")
        return
    if not path.exists():
        errors.append(f"{label} is missing: {path}")
        return
    try:
        actual = compute(path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        errors.append(f"could not hash {label} at {path}: {exc}")
        return
    if actual != expected:
        errors.append(f"{label} hash mismatch for {path}")


def _asset_path(value: str | Path | None, run_dir: Path) -> Path | None:
    if value in (None, ""):
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    candidates = (path, Path(__file__).resolve().parents[2] / path, run_dir / path)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _contains_mock_marker(value: Any) -> bool:
    if is_mock_row(value):
        return True
    if isinstance(value, dict):
        return any(_contains_mock_marker(item) for item in value.values() if isinstance(item, (dict, list)))
    if isinstance(value, list):
        return any(_contains_mock_marker(item) for item in value)
    return False


def _canonical_path(path: str | Path) -> Path:
    return Path(path).expanduser().resolve(strict=False)
