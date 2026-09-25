from __future__ import annotations

import argparse
import base64
import hashlib
import json
import mimetypes
import os
import posixpath
import re
import secrets
import sqlite3
import sys
import threading
import time
from collections import Counter
from contextlib import closing, contextmanager
from http.cookies import SimpleCookie
from datetime import datetime, timezone
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Iterator, List
from urllib.parse import parse_qs, quote, unquote, urlparse
from urllib import error as urllib_error
from urllib import request as urllib_request

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from processing.response_parser import score_raw_answer
from processing.retention import retention_by_user
from collection.human import assignment_scheduler
from puzzles import grid_placement, minesweeper_lite, mini_sudoku


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB = ROOT / "data" / "private" / "new_runs" / "study_state.sqlite3"
QUALITY_CHECKS_PATH = Path(
    os.environ.get(
        "STUDY_QUALITY_CHECKS_PRIVATE_PATH",
        ROOT / "collection" / "human" / "quality_checks_private.json",
    )
)
PUBLIC_QUALITY_CHECKS_PATH = ROOT / "collection" / "human" / "study_web" / "app" / "quality_checks.json"
USERNAME_RE = re.compile(r"^[A-Za-z0-9_]+$")
ADMIN_USERNAME = os.environ.get("STUDY_ADMIN_USERNAME", "local_admin")
ADMIN_PASSWORD = os.environ.get("STUDY_ADMIN_PASSWORD", "local_dev_password")
ADMIN_USES_DEV_DEFAULT = not os.environ.get("STUDY_ADMIN_USERNAME") or not os.environ.get("STUDY_ADMIN_PASSWORD")
ADMIN_COOKIE = "study_admin_session"
ADMIN_SESSION_TTL_SECONDS = int(os.environ.get("STUDY_ADMIN_SESSION_TTL_SECONDS", str(12 * 60 * 60)))
LOGIN_FAILURE_LIMIT = int(os.environ.get("STUDY_LOGIN_FAILURE_LIMIT", "8"))
LOGIN_FAILURE_WINDOW_SECONDS = int(os.environ.get("STUDY_LOGIN_FAILURE_WINDOW_SECONDS", str(15 * 60)))
TESTER_USERNAMES_ENV = os.environ.get("STUDY_TESTER_USERNAMES", "")
TESTER_PASSWORD_SHA256_ENV = os.environ.get("STUDY_TESTER_PASSWORD_SHA256", "")
TESTER_LOGIN_ENABLED = bool(TESTER_USERNAMES_ENV and TESTER_PASSWORD_SHA256_ENV)
TESTER_USERNAMES = (
    {username.strip() for username in TESTER_USERNAMES_ENV.split(",") if username.strip()}
    if TESTER_LOGIN_ENABLED
    else set()
)
TESTER_PASSWORD_SHA256 = TESTER_PASSWORD_SHA256_ENV
RESEARCHER_TEST_USERNAMES_ENV = os.environ.get("STUDY_RESEARCHER_TEST_USERNAMES", "")
RESEARCHER_TEST_USERNAMES = {username.strip() for username in RESEARCHER_TEST_USERNAMES_ENV.split(",") if username.strip()}
RESEARCHER_TEST_USERNAMES.add(ADMIN_USERNAME)
STUDY_CONFIG = assignment_scheduler.load_study_config()
BAD_RECORD_TOTAL_LIMIT = int(STUDY_CONFIG.get("bad_record_total_retention_limit", 30))
BAD_RECORD_BLOCK_LIMIT = int(STUDY_CONFIG.get("bad_record_block_retention_limit", 14))
QUOTA_TARGET = int(STUDY_CONFIG.get("quota_target_retained", 100))
MODULE_RETENTION_MIN_CORRECT = int(STUDY_CONFIG.get("module_retention_min_correct", 40))
MODULE_RETENTION_MIN_MODULE_CORRECT = int(STUDY_CONFIG.get("module_retention_min_module_correct", 3))
MODULE_RETENTION_MODULES = ["cue", "spatial", "formulation", "pair", "transfer"]
FAST_RESPONSE_MS = int(STUDY_CONFIG.get("attention_checks", {}).get("fast_response_ms", 2000))
DEFAULT_STUDY_VALID_WINDOW_MINUTES = int(STUDY_CONFIG.get("timing", {}).get("study_valid_window_minutes", 100))
TRIAL_TIMEOUT_MS = int(STUDY_CONFIG.get("timing", {}).get("trial_timeout_ms", 120000))
REQUIRE_PROLIFIC = os.environ.get("STUDY_REQUIRE_PROLIFIC") == "1"
PUBLIC_ACCESS_TOKEN = os.environ.get("STUDY_PUBLIC_ACCESS_TOKEN", "")
REQUIRE_PROLIFIC_API = os.environ.get("STUDY_REQUIRE_PROLIFIC_API") == "1"
PROLIFIC_API_TOKEN = os.environ.get("PROLIFIC_API_TOKEN", "")
PROLIFIC_API_BASE = os.environ.get("PROLIFIC_API_BASE", "https://api.prolific.com/api/v1").rstrip("/")
PROLIFIC_API_TIMEOUT_SECONDS = float(os.environ.get("PROLIFIC_API_TIMEOUT_SECONDS", "8"))
PROLIFIC_API_BLOCK_ON_FAILURE = os.environ.get("STUDY_PROLIFIC_API_BLOCK_ON_FAILURE") == "1"
MAX_JSON_BODY_BYTES = int(os.environ.get("STUDY_MAX_JSON_BODY_BYTES", str(1024 * 1024)))
PROLIFIC_ALLOWED_STATUSES = {
    status.strip().upper().replace(" ", "_").replace("-", "_")
    for status in os.environ.get("PROLIFIC_ALLOWED_STATUSES", "ACTIVE,AWAITING_REVIEW,APPROVED,RESERVED").split(",")
    if status.strip()
}
REQUIRE_PROLIFIC_JWT = os.environ.get("STUDY_REQUIRE_PROLIFIC_JWT") == "1"
PROLIFIC_JWKS_URL = os.environ.get("PROLIFIC_JWKS_URL", "https://api.prolific.com/.well-known/study/jwks.json")
PROLIFIC_EXPECTED_AUDIENCE = os.environ.get("PROLIFIC_EXPECTED_AUDIENCE", "")
RAW_RENDER_HOSTS = set(filter(None, os.environ.get("STUDY_REDIRECT_HOSTS", "").split(",")))
CANONICAL_STUDY_HOSTS = {"main": os.environ.get("STUDY_MAIN_HOST", ""),
                         "module": os.environ.get("STUDY_MODULE_HOST", "")}
PUBLIC_STATIC_PREFIXES = ("study_web/app/", "study_web/admin/", "study_web/admin_test/")
PUBLIC_STATIC_FILES = {"study_web/favicon.svg"}
ADMIN_PROTECTED_STATIC_PREFIXES = ("data/stimuli/", "collection/models/images/")
SENSITIVE_STATIC_SUFFIXES = {
    ".db",
    ".sqlite",
    ".sqlite3",
    ".py",
    ".pyc",
    ".doc",
    ".docx",
    ".pdf",
    ".zip",
}
SAFE_INSTANCE_KEYS = {
    "board",
    "box_cols",
    "box_rows",
    "digit",
    "display_numbers",
    "grid",
    "m",
    "n",
    "numbers",
    "target",
    "target_kind",
}
ANALYSIS_EXPORT_REMOVED_KEYS = {
    "platform_id",
    "prolific_pid",
    "prolific_study_id",
    "prolific_session_id",
    "session_id",
    "tester_login",
    "participant_session_token",
    "session_token",
    "study_start_time_iso",
    "trial_start_time_iso",
    "trial_end_time_iso",
}
PUBLIC_EXISTING_ROW_KEYS = {
    "puzzle_id",
    "is_correct",
    "server_scored",
    "timeout",
    "gave_up",
    "skipped",
    "bad_record",
    "is_practice",
    "is_attention_check",
    "response_time_ms",
    "wrong_submit_attempts",
    "wrong_submit_attempts_before_correct",
    "first_action_time_ms",
    "quota_counted",
    "bonus_earned",
    "bonus_usd",
    "too_fast_response_flag",
}
SAFE_MODULE_META_KEYS: set[str] = set()
PUBLIC_ASSIGNMENT_KEYS = {
    "assignment_version",
    "assignment_hash",
    "participant_session_token",
    "username",
    "cohort",
    "dataset_path",
    "puzzle_ids",
    "payment",
    "study",
    "practice",
    "attention_checks",
    "resume",
    "existing_responses",
    "existing_quality_checks",
}
_JWKS_CACHE: Dict[str, Any] | None = None


def b64url_decode(value: str) -> bytes:
    padding = "=" * (-len(value) % 4)
    return base64.urlsafe_b64decode((value + padding).encode("ascii"))


def normalize_prolific_status(status: Any) -> str:
    return str(status or "").strip().upper().replace(" ", "_").replace("-", "_")


def prolific_claim_value(prolific: Dict[str, Any], *keys: str) -> str:
    lowered = {str(key).lower(): value for key, value in prolific.items()}
    for key in keys:
        if key in prolific:
            return str(prolific[key])
        value = lowered.get(key.lower())
        if value is not None:
            return str(value)
    return ""


def load_prolific_jwks() -> Dict[str, Any]:
    global _JWKS_CACHE
    if _JWKS_CACHE is not None:
        return _JWKS_CACHE
    request = urllib_request.Request(PROLIFIC_JWKS_URL, headers={"Accept": "application/json"})
    try:
        with urllib_request.urlopen(request, timeout=PROLIFIC_API_TIMEOUT_SECONDS) as response:
            _JWKS_CACHE = json.loads(response.read().decode("utf-8"))
            return _JWKS_CACHE
    except Exception as exc:
        raise ValueError("prolific_jwks_fetch_failed") from exc


def verify_rs256_jwt(token: str, jwks: Dict[str, Any]) -> Dict[str, Any]:
    parts = token.split(".")
    if len(parts) != 3:
        raise ValueError("prolific_token_invalid")
    header_b64, payload_b64, signature_b64 = parts
    try:
        header = json.loads(b64url_decode(header_b64))
        payload = json.loads(b64url_decode(payload_b64))
        signature = b64url_decode(signature_b64)
    except Exception as exc:
        raise ValueError("prolific_token_invalid") from exc
    if header.get("alg") != "RS256" or not header.get("kid"):
        raise ValueError("prolific_token_invalid")
    key = next((item for item in jwks.get("keys", []) if item.get("kid") == header["kid"]), None)
    if not key or key.get("kty") != "RSA":
        raise ValueError("prolific_token_key_not_found")
    try:
        n = int.from_bytes(b64url_decode(key["n"]), "big")
        e = int.from_bytes(b64url_decode(key["e"]), "big")
    except Exception as exc:
        raise ValueError("prolific_token_key_invalid") from exc
    signature_int = int.from_bytes(signature, "big")
    key_len = (n.bit_length() + 7) // 8
    decrypted = pow(signature_int, e, n).to_bytes(key_len, "big")
    digest = hashlib.sha256(f"{header_b64}.{payload_b64}".encode("ascii")).digest()
    digest_info = bytes.fromhex("3031300d060960864801650304020105000420") + digest
    expected_prefix = b"\x00\x01"
    if not decrypted.startswith(expected_prefix):
        raise ValueError("prolific_token_signature_invalid")
    separator = decrypted.find(b"\x00", 2)
    if separator < 10 or decrypted[2:separator] != b"\xff" * (separator - 2):
        raise ValueError("prolific_token_signature_invalid")
    if not secrets.compare_digest(decrypted[separator + 1 :], digest_info):
        raise ValueError("prolific_token_signature_invalid")
    return payload


def verify_prolific_jwt(
    token: str,
    prolific_pid: str,
    study_id: str,
    session_id: str,
) -> Dict[str, Any]:
    if not token:
        raise ValueError("prolific_token_required")
    payload = verify_rs256_jwt(token, load_prolific_jwks())
    now = int(time.time())
    if int(payload.get("exp") or 0) < now:
        raise ValueError("prolific_token_expired")
    if payload.get("iss") != "https://www.prolific.com":
        raise ValueError("prolific_token_issuer_invalid")
    if PROLIFIC_EXPECTED_AUDIENCE and payload.get("aud") != PROLIFIC_EXPECTED_AUDIENCE:
        raise ValueError("prolific_token_audience_invalid")
    prolific = payload.get("prolific")
    if not isinstance(prolific, dict):
        raise ValueError("prolific_token_missing_claims")
    token_pid = prolific_claim_value(prolific, "PROLIFIC_PID", "participant", "participant_id")
    token_study = prolific_claim_value(prolific, "STUDY_ID", "study", "study_id")
    token_session = prolific_claim_value(prolific, "SESSION_ID", "session", "session_id")
    if token_pid != prolific_pid or token_study != study_id or token_session != session_id:
        raise ValueError("prolific_token_claim_mismatch")
    if payload.get("sub") and str(payload["sub"]) != session_id:
        raise ValueError("prolific_token_claim_mismatch")
    return {
        "verified": True,
        "method": "secure_external_url_jwt",
        "session_id": session_id,
        "study_id": study_id,
        "participant": prolific_pid,
        "expires_at": payload.get("exp"),
    }


def validate_prolific_submission(
    submission: Dict[str, Any],
    prolific_pid: str,
    study_id: str,
    session_id: str,
) -> Dict[str, Any]:
    status = normalize_prolific_status(submission.get("status"))
    participant = str(submission.get("participant") or submission.get("participant_id") or submission.get("prolific_pid") or "")
    returned_study = str(submission.get("study_id") or submission.get("study") or "")
    returned_session = str(submission.get("id") or submission.get("session_id") or "")
    if returned_session != session_id or participant != prolific_pid or returned_study != study_id:
        raise ValueError("prolific_submission_mismatch")
    if status not in PROLIFIC_ALLOWED_STATUSES:
        raise ValueError("prolific_submission_status_invalid")
    return {
        "verified": True,
        "method": "submission_api",
        "submission_id": returned_session,
        "study_id": returned_study,
        "participant": participant,
        "status": status,
    }


def verify_prolific_submission_api(prolific_pid: str, study_id: str, session_id: str) -> Dict[str, Any]:
    if not PROLIFIC_API_TOKEN:
        raise ValueError("prolific_api_token_missing")
    request = urllib_request.Request(
        f"{PROLIFIC_API_BASE}/submissions/{quote(session_id)}/",
        headers={
            "Accept": "application/json",
            "Authorization": f"Token {PROLIFIC_API_TOKEN}",
        },
    )
    try:
        with urllib_request.urlopen(request, timeout=PROLIFIC_API_TIMEOUT_SECONDS) as response:
            submission = json.loads(response.read().decode("utf-8"))
    except urllib_error.HTTPError as exc:
        if exc.code == 404:
            raise ValueError("prolific_submission_not_found") from exc
        raise ValueError("prolific_verification_error") from exc
    except Exception as exc:
        raise ValueError("prolific_verification_error") from exc
    return validate_prolific_submission(submission, prolific_pid, study_id, session_id)


def verify_prolific_access(
    prolific_pid: str,
    study_id: str,
    session_id: str,
    access_token: str,
    prolific_token: str,
) -> Dict[str, Any]:
    if REQUIRE_PROLIFIC and not (prolific_pid and study_id and session_id):
        raise ValueError("prolific_required")
    if PUBLIC_ACCESS_TOKEN and not secrets.compare_digest(access_token, PUBLIC_ACCESS_TOKEN):
        raise ValueError("invalid_access_token")
    if (REQUIRE_PROLIFIC_API or PROLIFIC_API_TOKEN or REQUIRE_PROLIFIC_JWT or prolific_token) and not (
        prolific_pid and study_id and session_id
    ):
        raise ValueError("prolific_required")
    verification: Dict[str, Any] = {}
    if REQUIRE_PROLIFIC_JWT or prolific_token:
        verification["secure_external_url"] = verify_prolific_jwt(prolific_token, prolific_pid, study_id, session_id)
    if REQUIRE_PROLIFIC_API or PROLIFIC_API_TOKEN:
        try:
            verification["submission_api"] = verify_prolific_submission_api(prolific_pid, study_id, session_id)
        except ValueError as exc:
            if str(exc) == "prolific_api_token_missing" or PROLIFIC_API_BLOCK_ON_FAILURE:
                raise
            verification["submission_api"] = {
                "verified": False,
                "method": "submission_api",
                "soft_failed": True,
                "error": str(exc),
                "session_id": session_id,
                "study_id": study_id,
                "participant": prolific_pid,
            }
    return verification


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def parse_iso_datetime(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_puzzle_index() -> Dict[str, Dict[str, Any]]:
    index: Dict[str, Dict[str, Any]] = {}
    for path in [
        ROOT / "data" / "stimuli" / "all_puzzles.jsonl",
        ROOT / "data" / "stimuli" / "modules" / "all_module_trials.jsonl",
    ]:
        if path.exists():
            for row in read_jsonl(path):
                index[row["id"]] = row
    return index


def load_quality_check_index() -> Dict[str, Dict[str, Any]]:
    path = QUALITY_CHECKS_PATH if QUALITY_CHECKS_PATH.exists() else PUBLIC_QUALITY_CHECKS_PATH
    with path.open("r", encoding="utf-8") as f:
        catalog = json.load(f)
    checks: Dict[str, Dict[str, Any]] = {}
    for group in ("practice", "attention"):
        for row in catalog.get(group, []):
            checks[row["id"]] = quality_check_with_solutions(row)
    return checks


def quality_check_with_solutions(row: Dict[str, Any]) -> Dict[str, Any]:
    if row.get("solutions"):
        return row
    enriched = dict(row)
    inst = enriched.get("machine_readable_instance", {})
    puzzle_type = enriched.get("puzzle_type")
    if puzzle_type == "mini_sudoku":
        solutions = mini_sudoku.enumerate_placements(inst["board"], int(inst["digit"]))
    elif puzzle_type == "grid_placement":
        solutions = grid_placement.enumerate_placements(inst["board"], int(inst["m"]))
    elif puzzle_type == "minesweeper_lite":
        solutions, _ = minesweeper_lite.forced_cells(inst["board"], str(inst["target_kind"]))
    else:
        raise ValueError(f"unsupported_quality_check_type:{puzzle_type}")
    enriched["solutions"] = [
        {
            **solution,
            "solution_id": solution.get("solution_id") or f"{enriched['id']}_SOL_{idx + 1:02d}",
            "abstract_solution_id": solution.get("abstract_solution_id") or f"{enriched['id'].lower()}_sol_{idx + 1:02d}",
        }
        for idx, solution in enumerate(solutions)
    ]
    enriched["num_solutions"] = len(enriched["solutions"])
    return enriched


def expected_quality_check_count(assignment: Dict[str, Any]) -> int:
    total = 0
    practice = assignment.get("practice", {})
    if practice.get("enabled"):
        total += int(practice.get("num_trials") or 0)
    attention = assignment.get("attention_checks", {})
    if attention.get("enabled"):
        cohort = str(assignment.get("cohort") or "")
        key = "module_count" if cohort == "module" else "main_count"
        total += int(attention.get(key) or 0)
    return total


def public_cue_payload(cue: Any) -> Dict[str, Any] | None:
    if not isinstance(cue, dict):
        return None
    coordinates = cue.get("coordinates") or ([cue.get("coordinate")] if cue.get("coordinate") else [])
    coordinates = [coord for coord in coordinates if isinstance(coord, list) and len(coord) == 2]
    if not coordinates:
        return None
    return {
        "coordinates": coordinates,
        "kind": cue.get("kind", "visual_highlight"),
        "cue_strength": cue.get("cue_strength", "subtle"),
    }


def public_trial_aliases(puzzle_ids: List[str]) -> Dict[str, str]:
    return {puzzle_id: f"T{idx + 1:03d}" for idx, puzzle_id in enumerate(puzzle_ids)}


def public_puzzle_payload(puzzle: Dict[str, Any], public_id: str | None = None) -> Dict[str, Any]:
    instance = puzzle.get("machine_readable_instance", {})
    public_instance = {key: instance[key] for key in SAFE_INSTANCE_KEYS if key in instance}
    for cue_key in ("visual_cue", "irrelevant_cue"):
        cue = public_cue_payload(instance.get(cue_key))
        if cue:
            public_instance[cue_key] = cue

    presentation = {}
    for cue_key in ("visual_cue", "irrelevant_cue"):
        cue = public_cue_payload(puzzle.get("presentation_metadata", {}).get(cue_key))
        if cue:
            presentation[cue_key] = cue

    module_meta = {
        key: puzzle.get("module_metadata", {}).get(key)
        for key in SAFE_MODULE_META_KEYS
        if key in puzzle.get("module_metadata", {})
    }
    return {
        "id": public_id or puzzle["id"],
        "puzzle_type": puzzle["puzzle_type"],
        "prompt_text": "",
        "rendered_puzzle": "",
        "machine_readable_instance": public_instance,
        "presentation_metadata": presentation,
        "module_metadata": module_meta,
    }


def public_puzzles_for_assignment(
    puzzle_ids: List[str],
    aliases: Dict[str, str] | None = None,
) -> List[Dict[str, Any]]:
    puzzle_index = load_puzzle_index()
    aliases = aliases or {}
    return [
        public_puzzle_payload(puzzle_index[puzzle_id], public_id=aliases.get(puzzle_id))
        for puzzle_id in puzzle_ids
        if puzzle_id in puzzle_index
    ]


def public_unit_plan(assignment: Dict[str, Any], aliases: Dict[str, str]) -> List[Dict[str, Any]]:
    return [
        {"trial_ids": [aliases.get(trial_id, trial_id) for trial_id in choice.get("trial_ids", [])]}
        for choice in assignment.get("unit_plan", [])
        if choice.get("trial_ids")
    ]


def public_existing_rows(rows: List[Dict[str, Any]], aliases: Dict[str, str]) -> List[Dict[str, Any]]:
    safe_rows = []
    for row in rows:
        puzzle_id = str(row.get("puzzle_id") or row.get("check_id") or "")
        if not puzzle_id:
            continue
        safe = {key: row[key] for key in PUBLIC_EXISTING_ROW_KEYS if key in row}
        safe["puzzle_id"] = aliases.get(puzzle_id, puzzle_id)
        safe_rows.append(safe)
    return safe_rows


def public_assignment_payload(
    assignment: Dict[str, Any],
    include_completion_code: bool = False,
) -> Dict[str, Any]:
    aliases = public_trial_aliases(assignment.get("puzzle_ids", []))
    public = {key: assignment[key] for key in PUBLIC_ASSIGNMENT_KEYS if key in assignment}
    public["puzzle_ids"] = [aliases.get(puzzle_id, puzzle_id) for puzzle_id in assignment.get("puzzle_ids", [])]
    if "unit_plan" in assignment:
        public["unit_plan"] = public_unit_plan(assignment, aliases)
    if "existing_responses" in public:
        public["existing_responses"] = public_existing_rows(public["existing_responses"], aliases)
    if "existing_quality_checks" in public:
        public["existing_quality_checks"] = public_existing_rows(public["existing_quality_checks"], aliases)
    if include_completion_code:
        public["completion_code"] = assignment.get("completion_code", "")
    public["public_puzzles"] = public_puzzles_for_assignment(assignment.get("puzzle_ids", []), aliases)
    return public


def json_dumps_compact(value: Dict[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def make_completion_code(cohort: str) -> str:
    prefix = STUDY_CONFIG["studies"][cohort]["completion_code"]
    static_code = os.environ.get(f"STUDY_COMPLETION_CODE_{cohort.upper()}", "").strip()
    if static_code:
        return static_code
    token = secrets.token_urlsafe(8).replace("-", "").replace("_", "").upper()[:10]
    return f"{prefix}_{token}"


def make_participant_session_token() -> str:
    return secrets.token_urlsafe(32)


def hash_session_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def analysis_participant_id(row: Dict[str, Any]) -> str:
    existing = row.get("analysis_participant_id") or row.get("analysis_id")
    if existing:
        return str(existing)
    seed = str(row.get("username") or row.get("participant_id") or row.get("prolific_pid") or "")
    digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()[:16].upper()
    return f"A_{digest}"


def make_analysis_id() -> str:
    return f"A_{secrets.token_hex(12).upper()}"


def password_sha256(password: str) -> str:
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def valid_tester_credentials(username: str, password: str) -> bool:
    if not TESTER_LOGIN_ENABLED:
        return False
    return username in TESTER_USERNAMES and secrets.compare_digest(password_sha256(password), TESTER_PASSWORD_SHA256)


def valid_researcher_test_credentials(username: str, password: str) -> bool:
    return valid_tester_credentials(username, password) or (
        username in RESEARCHER_TEST_USERNAMES and password == ADMIN_PASSWORD
    )


def tester_participant_username(tester_login: str, cohort: str, session_id: str = "") -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_]", "_", tester_login).strip("_")
    session = re.sub(r"[^A-Za-z0-9_]", "_", session_id).strip("_")
    suffix = f"_{session}" if session else ""
    return f"TEST_{cleaned}_{cohort}{suffix}".upper()


def zero_payment_for_tester(assignment: Dict[str, Any], cohort: str) -> None:
    assignment["payment"] = {
        "posted_base_reward_usd": 0.0,
        "correct_bonus_per_trial_usd": 0.0,
        "max_bonus_usd": 0.0,
        "max_total_reward_usd": 0.0,
    }
    assignment["study"] = {
        **assignment.get("study", {}),
        "name": f"Researcher Test - {assignment.get('study', {}).get('name', cohort)}",
    }
    assignment["completion_code"] = f"TESTER_{cohort.upper()}_{secrets.token_urlsafe(6).replace('-', '').replace('_', '').upper()[:8]}"


def infer_cohort(username, requested=None):
    return assignment_scheduler.infer_cohort(username, requested)


def sanitize_username_part(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_]", "_", str(value or "")).strip("_")
    return cleaned or "participant"


def expected_prolific_username(prolific_pid: str, cohort: str) -> str:
    prefix = "E" if cohort == "module" else "M"
    return f"{prefix}_{sanitize_username_part(prolific_pid)}"


def study_valid_window_minutes(cohort: str, assignment: Dict[str, Any] | None = None) -> int:
    if assignment:
        try:
            value = int(assignment.get("study", {}).get("study_valid_window_minutes") or 0)
        except (TypeError, ValueError):
            value = 0
        if value > 0:
            return value
    study = STUDY_CONFIG.get("studies", {}).get(cohort, {})
    try:
        value = int(study.get("study_valid_window_minutes") or 0)
    except (TypeError, ValueError):
        value = 0
    return value if value > 0 else DEFAULT_STUDY_VALID_WINDOW_MINUTES


def make_assignment(username: str, cohort: str, retained_quota: Dict[str, int] | None = None) -> Dict[str, Any]:
    return assignment_scheduler.make_assignment(username, cohort, retained_quota=retained_quota or {})


class StudyDatabase:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.init_schema()

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        # The SQLite transaction context alone does not close its connection.
        with closing(sqlite3.connect(self.path, timeout=30)) as conn:
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA foreign_keys = ON")
            conn.execute("PRAGMA busy_timeout = 30000")
            conn.execute("PRAGMA synchronous = NORMAL")
            with conn:
                yield conn

    def init_schema(self) -> None:
        with self.connect() as conn:
            conn.execute("PRAGMA journal_mode = WAL")
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS participants (
                    username TEXT PRIMARY KEY,
                    cohort TEXT NOT NULL,
                    registered_at TEXT NOT NULL,
                    completed_at TEXT,
                    assignment_dataset TEXT NOT NULL,
                    assignment_json TEXT NOT NULL,
                    assigned_count INTEGER NOT NULL,
                    response_count INTEGER NOT NULL DEFAULT 0,
                    bad_record_count INTEGER NOT NULL DEFAULT 0,
                    excluded INTEGER NOT NULL DEFAULT 0
                );

                CREATE TABLE IF NOT EXISTS responses (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT NOT NULL,
                    puzzle_id TEXT NOT NULL,
                    row_json TEXT NOT NULL,
                    is_correct INTEGER NOT NULL DEFAULT 0,
                    bad_record INTEGER NOT NULL DEFAULT 0,
                    timeout INTEGER NOT NULL DEFAULT 0,
                    gave_up INTEGER NOT NULL DEFAULT 0,
                    response_time_ms INTEGER,
                    trial_start_time_iso TEXT,
                    trial_end_time_iso TEXT,
                    created_at TEXT NOT NULL,
                    UNIQUE(username, puzzle_id),
                    FOREIGN KEY(username) REFERENCES participants(username)
                );

                CREATE TABLE IF NOT EXISTS assignments (
                    username TEXT NOT NULL,
                    trial_order INTEGER NOT NULL,
                    puzzle_id TEXT NOT NULL,
                    family_id TEXT,
                    puzzle_type TEXT,
                    module TEXT,
                    condition TEXT,
                    unit_id TEXT,
                    choice_id TEXT,
                    sequence_id TEXT,
                    sequence_position INTEGER,
                    primary_trial INTEGER NOT NULL DEFAULT 1,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY(username, trial_order),
                    UNIQUE(username, puzzle_id),
                    FOREIGN KEY(username) REFERENCES participants(username)
                );

                CREATE TABLE IF NOT EXISTS payment_ledger (
                    username TEXT PRIMARY KEY,
                    assigned_count INTEGER NOT NULL,
                    correct_count INTEGER NOT NULL DEFAULT 0,
                    posted_base_reward_usd REAL NOT NULL DEFAULT 0,
                    correct_bonus_per_trial_usd REAL NOT NULL DEFAULT 0,
                    bonus_reward_usd REAL NOT NULL DEFAULT 0,
                    total_expected_reward_usd REAL NOT NULL DEFAULT 0,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY(username) REFERENCES participants(username)
                );

                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT,
                    event_type TEXT NOT NULL,
                    event_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS audit_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT,
                    event_type TEXT NOT NULL,
                    event_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS quality_checks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT NOT NULL,
                    check_id TEXT NOT NULL,
                    row_json TEXT NOT NULL,
                    is_correct INTEGER NOT NULL DEFAULT 0,
                    wrong_submit_attempts INTEGER NOT NULL DEFAULT 0,
                    response_time_ms INTEGER,
                    created_at TEXT NOT NULL,
                    UNIQUE(username, check_id),
                    FOREIGN KEY(username) REFERENCES participants(username)
                );
                """
            )
            self.ensure_columns(
                conn,
                "participants",
                {
                    "prolific_pid": "TEXT",
                    "prolific_study_id": "TEXT",
                    "prolific_session_id": "TEXT",
                    "consent_at": "TEXT",
                    "assignment_version": "TEXT",
                    "assignment_hash": "TEXT",
                    "completion_code": "TEXT",
                    "participant_session_hash": "TEXT",
                    "bonus_correct_count": "INTEGER NOT NULL DEFAULT 0",
                    "bonus_reward_usd": "REAL NOT NULL DEFAULT 0",
                    "base_reward_usd": "REAL NOT NULL DEFAULT 0",
                    "total_expected_reward_usd": "REAL NOT NULL DEFAULT 0",
                    "low_effort_flag_count": "INTEGER NOT NULL DEFAULT 0",
                    "participant_source": "TEXT NOT NULL DEFAULT 'prolific_or_local'",
                    "tester_login": "TEXT",
                    "analysis_id": "TEXT",
                    "study_duration_ms": "INTEGER",
                    "study_time_limit_exceeded": "INTEGER NOT NULL DEFAULT 0",
                },
            )
            self.ensure_analysis_ids(conn)

    def ensure_columns(self, conn: sqlite3.Connection, table: str, columns: Dict[str, str]) -> None:
        existing = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
        for name, spec in columns.items():
            if name not in existing:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {spec}")

    def ensure_analysis_ids(self, conn: sqlite3.Connection) -> None:
        existing = {
            str(row["analysis_id"])
            for row in conn.execute(
                "SELECT analysis_id FROM participants WHERE analysis_id IS NOT NULL AND analysis_id != ''",
            ).fetchall()
        }
        for row in conn.execute("SELECT username FROM participants WHERE analysis_id IS NULL OR analysis_id = ''").fetchall():
            analysis_id = make_analysis_id()
            while analysis_id in existing:
                analysis_id = make_analysis_id()
            existing.add(analysis_id)
            conn.execute("UPDATE participants SET analysis_id = ? WHERE username = ?", (analysis_id, row["username"]))
        conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_participants_analysis_id ON participants(analysis_id)")

    def log_event(self, conn: sqlite3.Connection, event_type: str, username: str = "", payload: Dict[str, Any] | None = None) -> None:
        event_json = json_dumps_compact(payload or {})
        created_at = now_iso()
        for table in ("events", "audit_events"):
            conn.execute(
                f"INSERT INTO {table} (username, event_type, event_json, created_at) VALUES (?, ?, ?, ?)",
                (username, event_type, event_json, created_at),
            )

    def response_payloads(self, conn: sqlite3.Connection) -> List[Dict[str, Any]]:
        rows = []
        for row in conn.execute("SELECT username, puzzle_id, row_json, is_correct, bad_record, timeout, gave_up FROM responses").fetchall():
            payload = json.loads(row["row_json"])
            payload.setdefault("username", row["username"])
            payload.setdefault("puzzle_id", row["puzzle_id"])
            payload["is_correct"] = bool(row["is_correct"])
            payload["bad_record"] = bool(row["bad_record"])
            payload["timeout"] = bool(row["timeout"])
            payload["gave_up"] = bool(row["gave_up"])
            rows.append(payload)
        return rows

    def analysis_retention_by_user(self, conn: sqlite3.Connection) -> Dict[str, Dict[str, Any]]:
        participants = {
            row["username"]: dict(row)
            for row in conn.execute("SELECT * FROM participants").fetchall()
        }
        assignment_meta: Dict[tuple[str, str], Dict[str, Any]] = {}
        for row in conn.execute("SELECT username, puzzle_id, module, puzzle_type FROM assignments").fetchall():
            assignment_meta[(row["username"], row["puzzle_id"])] = {
                "module": row["module"] or "",
                "puzzle_type": row["puzzle_type"] or "",
            }

        response_payloads: List[Dict[str, Any]] = []
        responses_by_user: Dict[str, List[Dict[str, Any]]] = {}
        for row in conn.execute(
            """
            SELECT username, puzzle_id, row_json, is_correct, bad_record, timeout, gave_up
            FROM responses
            """
        ).fetchall():
            meta = assignment_meta.get((row["username"], row["puzzle_id"]), {})
            try:
                payload = json.loads(row["row_json"])
            except json.JSONDecodeError:
                payload = {}
            payload.setdefault("username", row["username"])
            payload.setdefault("puzzle_id", row["puzzle_id"])
            payload.setdefault("module", meta.get("module", ""))
            payload.setdefault("puzzle_type", meta.get("puzzle_type", ""))
            payload["is_correct"] = bool(row["is_correct"])
            payload["bad_record"] = bool(row["bad_record"])
            payload["timeout"] = bool(row["timeout"])
            payload["gave_up"] = bool(row["gave_up"])
            response_payloads.append(payload)
            responses_by_user.setdefault(row["username"], []).append(payload)

        return retention_by_user(
            list(participants.values()), response_payloads,
            bad_total_limit=BAD_RECORD_TOTAL_LIMIT, bad_block_limit=BAD_RECORD_BLOCK_LIMIT,
            module_min_correct=MODULE_RETENTION_MIN_CORRECT,
            module_min_per_module=MODULE_RETENTION_MIN_MODULE_CORRECT,
        )

    def retained_quota(self, conn: sqlite3.Connection) -> Dict[str, int]:
        retention = self.analysis_retention_by_user(conn)
        retained_payloads = [
            payload
            for payload in self.response_payloads(conn)
            if retention.get(str(payload.get("username") or ""), {}).get("analysis_retained")
        ]
        return assignment_scheduler.retained_quota_from_response_rows(
            retained_payloads,
            bad_total_limit=BAD_RECORD_TOTAL_LIMIT,
            bad_block_limit=BAD_RECORD_BLOCK_LIMIT,
        )

    def scheduling_quota_load(self, conn: sqlite3.Connection) -> Dict[str, int]:
        """Use SQL-only finished responses + pending assignments for live scheduling.

        Full retained-quota analysis parses every response JSON blob, which is too
        heavy to run on each registration while Prolific participants arrive in
        bursts. For live assignment balancing, treat non-tester participants who
        finished all assigned trials as usable, then add unanswered assignments
        from in-progress participants as pending load.
        """
        load: Counter[str] = Counter()
        for row in conn.execute(
            """
            SELECT r.puzzle_id, COUNT(*) AS n
            FROM responses r
            JOIN participants p ON p.username = r.username
            WHERE COALESCE(p.participant_source, '') != 'tester'
              AND p.assigned_count > 0
              AND p.response_count >= p.assigned_count
            GROUP BY r.puzzle_id
            """
        ).fetchall():
            load[row["puzzle_id"]] += int(row["n"] or 0)

        for row in conn.execute(
            """
            SELECT a.puzzle_id, COUNT(*) AS n
            FROM assignments a
            JOIN participants p ON p.username = a.username
            LEFT JOIN responses r
              ON r.username = a.username AND r.puzzle_id = a.puzzle_id
            WHERE COALESCE(p.participant_source, '') != 'tester'
              AND NOT (p.assigned_count > 0 AND p.response_count >= p.assigned_count)
              AND r.puzzle_id IS NULL
            GROUP BY a.puzzle_id
            """
        ).fetchall():
            load[row["puzzle_id"]] += int(row["n"] or 0)
        return dict(load)

    def assignment_rows_from_payload(self, username: str, assignment: Dict[str, Any]) -> List[Dict[str, Any]]:
        puzzles = load_puzzle_index()
        rows: List[Dict[str, Any]] = []
        order = 0
        for choice in assignment.get("unit_plan", []):
            trial_ids = choice.get("trial_ids", [])
            for pos, puzzle_id in enumerate(trial_ids, start=1):
                order += 1
                puzzle = puzzles.get(puzzle_id, {})
                meta = puzzle.get("module_metadata", {})
                rows.append(
                    {
                        "username": username,
                        "trial_order": order,
                        "puzzle_id": puzzle_id,
                        "family_id": puzzle.get("family_id") or puzzle.get("base_puzzle_id"),
                        "puzzle_type": puzzle.get("puzzle_type"),
                        "module": choice.get("module") or meta.get("module") or "main",
                        "condition": choice.get("condition") or meta.get("condition") or puzzle.get("variant_type") or "canonical",
                        "unit_id": choice.get("unit_id") or puzzle.get("family_id") or puzzle_id,
                        "choice_id": choice.get("choice_id") or choice.get("sequence_id") or puzzle_id,
                        "sequence_id": choice.get("sequence_id") or meta.get("sequence_id") or "",
                        "sequence_position": meta.get("sequence_position"),
                        "primary_trial": 1,
                    }
                )
        return rows

    def refresh_payment_ledger(self, conn: sqlite3.Connection, username: str) -> Dict[str, Any]:
        participant = conn.execute("SELECT * FROM participants WHERE username = ?", (username,)).fetchone()
        if not participant:
            raise ValueError("unknown_username")
        assignment = json.loads(participant["assignment_json"])
        payment = assignment.get("payment", {})
        correct_count = int(
            conn.execute("SELECT COALESCE(SUM(is_correct), 0) AS n FROM responses WHERE username = ?", (username,)).fetchone()["n"]
        )
        base = float(payment.get("posted_base_reward_usd") or 0)
        bonus_per = float(payment.get("correct_bonus_per_trial_usd") or 0)
        bonus = round(correct_count * bonus_per, 2)
        total = round(base + bonus, 2)
        conn.execute(
            """
            INSERT INTO payment_ledger
            (username, assigned_count, correct_count, posted_base_reward_usd,
             correct_bonus_per_trial_usd, bonus_reward_usd, total_expected_reward_usd, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(username) DO UPDATE SET
                assigned_count = excluded.assigned_count,
                correct_count = excluded.correct_count,
                posted_base_reward_usd = excluded.posted_base_reward_usd,
                correct_bonus_per_trial_usd = excluded.correct_bonus_per_trial_usd,
                bonus_reward_usd = excluded.bonus_reward_usd,
                total_expected_reward_usd = excluded.total_expected_reward_usd,
                updated_at = excluded.updated_at
            """,
            (username, int(participant["assigned_count"]), correct_count, base, bonus_per, bonus, total, now_iso()),
        )
        conn.execute(
            """
            UPDATE participants
            SET bonus_correct_count = ?, bonus_reward_usd = ?, base_reward_usd = ?, total_expected_reward_usd = ?
            WHERE username = ?
            """,
            (correct_count, bonus, base, total, username),
        )
        return {
            "username": username,
            "assigned_count": int(participant["assigned_count"]),
            "correct_count": correct_count,
            "posted_base_reward_usd": base,
            "correct_bonus_per_trial_usd": bonus_per,
            "bonus_reward_usd": bonus,
            "total_expected_reward_usd": total,
        }

    def register(
        self,
        username: str,
        cohort: str,
        prolific_pid: str = "",
        prolific_study_id: str = "",
        prolific_session_id: str = "",
        consent_at: str = "",
        participant_source: str = "",
        tester_login: str = "",
    ) -> Dict[str, Any]:
        with self.connect() as conn:
            existing = conn.execute("SELECT * FROM participants WHERE username = ?", (username,)).fetchone()
            if existing:
                existing_source = str(existing["participant_source"] or "")
                if (
                    (prolific_pid and existing["prolific_pid"] == prolific_pid)
                    or (
                        participant_source == "tester"
                        and existing_source == "tester"
                        and str(existing["tester_login"] or "") == tester_login
                    )
                ):
                    session_token = make_participant_session_token()
                    conn.execute(
                        "UPDATE participants SET participant_session_hash = ? WHERE username = ?",
                        (hash_session_token(session_token), username),
                    )
                    assignment = json.loads(existing["assignment_json"])
                    existing_rows = [
                        json.loads(row["row_json"])
                        for row in conn.execute(
                            "SELECT row_json FROM responses WHERE username = ? ORDER BY id",
                            (username,),
                        ).fetchall()
                    ]
                    existing_quality = [
                        json.loads(row["row_json"])
                        for row in conn.execute(
                            "SELECT row_json FROM quality_checks WHERE username = ? ORDER BY id",
                            (username,),
                        ).fetchall()
                    ]
                    assignment["resume"] = True
                    assignment["participant_session_token"] = session_token
                    assignment["existing_responses"] = existing_rows
                    assignment["existing_quality_checks"] = existing_quality
                    self.log_event(
                        conn,
                        "resume_register",
                        username,
                        {"existing_response_count": len(existing_rows), "existing_quality_count": len(existing_quality)},
                    )
                    return assignment
                raise ValueError("username_exists")
            if prolific_pid:
                existing_pid_rows = conn.execute(
                    """
                    SELECT username, cohort, prolific_study_id, prolific_session_id
                    FROM participants
                    WHERE prolific_pid = ?
                    """,
                    (prolific_pid,),
                ).fetchall()
                if cohort != "module" and existing_pid_rows:
                    raise ValueError("prolific_pid_exists")
                for existing_pid in existing_pid_rows:
                    same_submission = (
                        prolific_session_id
                        and str(existing_pid["prolific_session_id"] or "") == prolific_session_id
                    )
                    same_prolific_study = (
                        prolific_study_id
                        and str(existing_pid["prolific_study_id"] or "") == prolific_study_id
                    )
                    same_cohort_without_study_id = (
                        not prolific_study_id
                        and str(existing_pid["cohort"] or "") == cohort
                    )
                    if same_submission or same_prolific_study or same_cohort_without_study_id:
                        raise ValueError("prolific_pid_exists")
            if not participant_source:
                participant_source = "prolific" if prolific_pid else "local"
            assignment = make_assignment(username, cohort, self.scheduling_quota_load(conn))
            assignment["completion_code"] = make_completion_code(cohort)
            if participant_source == "tester":
                zero_payment_for_tester(assignment, cohort)
            session_token = make_participant_session_token()
            assignment_for_storage = dict(assignment)
            assignment_for_storage.pop("participant_session_token", None)
            assignment["participant_session_token"] = session_token
            assignment_rows = self.assignment_rows_from_payload(username, assignment)
            registered_at = now_iso()
            analysis_id = make_analysis_id()
            while conn.execute("SELECT 1 FROM participants WHERE analysis_id = ?", (analysis_id,)).fetchone():
                analysis_id = make_analysis_id()
            conn.execute(
                """
                INSERT INTO participants
                (username, cohort, registered_at, assignment_dataset, assignment_json, assigned_count,
                 prolific_pid, prolific_study_id, prolific_session_id, consent_at,
                 assignment_version, assignment_hash, completion_code, participant_session_hash,
                 base_reward_usd, total_expected_reward_usd, participant_source, tester_login, analysis_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    username,
                    cohort,
                    registered_at,
                    assignment["dataset_path"],
                    json_dumps_compact(assignment_for_storage),
                    len(assignment["puzzle_ids"]),
                    prolific_pid,
                    prolific_study_id,
                    prolific_session_id,
                    consent_at,
                    assignment.get("assignment_version"),
                    assignment.get("assignment_hash"),
                    assignment.get("completion_code"),
                    hash_session_token(session_token),
                    float(assignment.get("payment", {}).get("posted_base_reward_usd") or 0),
                    float(assignment.get("payment", {}).get("posted_base_reward_usd") or 0),
                    participant_source,
                    tester_login,
                    analysis_id,
                ),
            )
            for row in assignment_rows:
                conn.execute(
                    """
                    INSERT INTO assignments
                    (username, trial_order, puzzle_id, family_id, puzzle_type, module, condition,
                     unit_id, choice_id, sequence_id, sequence_position, primary_trial, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        row["username"],
                        row["trial_order"],
                        row["puzzle_id"],
                        row["family_id"],
                        row["puzzle_type"],
                        row["module"],
                        row["condition"],
                        row["unit_id"],
                        row["choice_id"],
                        row["sequence_id"],
                        row["sequence_position"],
                        row["primary_trial"],
                        now_iso(),
                    ),
                )
            self.refresh_payment_ledger(conn, username)
            self.log_event(
                conn,
                "register",
                username,
                {
                    "cohort": cohort,
                    "assigned_count": len(assignment["puzzle_ids"]),
                    "assignment_version": assignment.get("assignment_version"),
                    "assignment_hash": assignment.get("assignment_hash"),
                    "prolific_pid_present": bool(prolific_pid),
                    "participant_source": participant_source,
                    "tester_login": tester_login if participant_source == "tester" else "",
                },
            )
        return assignment

    def save_response(self, row: Dict[str, Any]) -> Dict[str, Any]:
        username = str(row.get("username") or row.get("participant_id") or "")
        submitted_puzzle_id = str(row.get("puzzle_id") or "")
        if not username or not submitted_puzzle_id:
            raise ValueError("missing username or puzzle_id")
        row["username"] = username
        row.setdefault("participant_id", username)
        with self.connect() as conn:
            participant = conn.execute("SELECT username, participant_source, tester_login FROM participants WHERE username = ?", (username,)).fetchone()
            if not participant:
                raise ValueError("unknown_username")
            row["participant_source"] = participant["participant_source"] or "prolific_or_local"
            if participant["tester_login"]:
                row["tester_login"] = participant["tester_login"]
            assigned = conn.execute(
                "SELECT * FROM assignments WHERE username = ? AND puzzle_id = ?",
                (username, submitted_puzzle_id),
            ).fetchone()
            if not assigned:
                token = re.fullmatch(r"T(\d{3})", submitted_puzzle_id)
                if token:
                    assigned = conn.execute(
                        "SELECT * FROM assignments WHERE username = ? AND trial_order = ?",
                        (username, int(token.group(1))),
                    ).fetchone()
            if not assigned:
                raise ValueError("puzzle_not_assigned")
            puzzle_id = assigned["puzzle_id"]
            if submitted_puzzle_id != puzzle_id:
                row["submitted_puzzle_id"] = submitted_puzzle_id
            row["puzzle_id"] = puzzle_id
            row.update(
                {
                    "family_id": assigned["family_id"],
                    "puzzle_type": assigned["puzzle_type"],
                    "module": assigned["module"],
                    "condition": assigned["condition"],
                    "module_unit_id": assigned["unit_id"],
                    "module_choice_id": assigned["choice_id"],
                    "sequence_id": assigned["sequence_id"],
                    "sequence_position": assigned["sequence_position"],
                }
            )
            response_time_ms = int(row.get("response_time_ms") or 0)
            timed_out = bool(row.get("timeout")) or (response_time_ms >= TRIAL_TIMEOUT_MS)
            gave_up = bool(row.get("gave_up") or row.get("skipped")) and not timed_out
            if timed_out:
                response_time_ms = TRIAL_TIMEOUT_MS
                row["timeout"] = True
                row["gave_up"] = False
                row["skipped"] = False
                row["raw_answer"] = "timeout"
                row["raw_response"] = "timeout"
                row["parsed_answer"] = "timeout"
            else:
                row["timeout"] = False
                row["gave_up"] = gave_up
                row["skipped"] = gave_up
            row["response_time_ms"] = response_time_ms
            bad_record = bool(gave_up or timed_out)
            scored: Dict[str, Any] = {}
            if not bad_record:
                puzzle = load_puzzle_index().get(puzzle_id)
                if not puzzle:
                    raise ValueError("unknown_puzzle")
                scored = score_raw_answer(puzzle, str(row.get("raw_answer") or row.get("raw_response") or ""))
                if not scored.get("is_valid"):
                    raise ValueError(f"invalid_final_answer:{scored.get('invalid_reason')}")
                row["is_correct"] = True
                row["server_scored"] = True
                row["solution_id"] = scored.get("solution_id")
                matched_solution = next(
                    (solution for solution in puzzle.get("solutions", []) if solution.get("solution_id") == scored.get("solution_id")),
                    {},
                )
                row["abstract_solution_id"] = matched_solution.get("abstract_solution_id") or row.get("abstract_solution_id") or scored.get("solution_id")
                row["accepted_solution_id"] = scored.get("solution_id")
                row["accepted_abstract_solution_id"] = row["abstract_solution_id"]
                row["parsed_answer"] = scored.get("parsed_answer")
                row["canonical_answer"] = scored.get("canonical_answer")
                row["parse_confidence"] = scored.get("parse_confidence")
                row["requires_manual_review"] = scored.get("requires_manual_review", False)
            else:
                row["is_correct"] = False
                row["solution_id"] = None
                row["abstract_solution_id"] = None
            row["bad_record"] = bad_record
            row.pop("participant_session_token", None)
            row.pop("session_token", None)
            conn.execute(
                """
                INSERT INTO responses
                (username, puzzle_id, row_json, is_correct, bad_record, timeout, gave_up,
                 response_time_ms, trial_start_time_iso, trial_end_time_iso, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(username, puzzle_id) DO UPDATE SET
                    row_json = excluded.row_json,
                    is_correct = excluded.is_correct,
                    bad_record = excluded.bad_record,
                    timeout = excluded.timeout,
                    gave_up = excluded.gave_up,
                    response_time_ms = excluded.response_time_ms,
                    trial_start_time_iso = excluded.trial_start_time_iso,
                    trial_end_time_iso = excluded.trial_end_time_iso
                """,
                (
                    username,
                    puzzle_id,
                    json_dumps_compact(row),
                    int(bool(row.get("is_correct"))),
                    int(bad_record),
                    int(timed_out),
                    int(gave_up),
                    response_time_ms,
                    row.get("trial_start_time_iso"),
                    row.get("trial_end_time_iso"),
                    now_iso(),
                ),
            )
            self.refresh_participant(conn, username)
            payment = self.refresh_payment_ledger(conn, username)
            self.log_event(
                conn,
                "response_save",
                username,
                {
                    "puzzle_id": puzzle_id,
                    "is_correct": bool(row.get("is_correct")),
                    "bad_record": bad_record,
                    "timeout": timed_out,
                    "gave_up": gave_up,
                    "response_time_ms": response_time_ms,
                },
            )
        return {
            "ok": True,
            "is_correct": bool(row.get("is_correct")),
            "solution_id": row.get("solution_id"),
            "abstract_solution_id": row.get("abstract_solution_id"),
            "accepted_solution_id": row.get("accepted_solution_id"),
            "accepted_abstract_solution_id": row.get("accepted_abstract_solution_id"),
            "parsed_answer": row.get("parsed_answer"),
            "canonical_answer": row.get("canonical_answer"),
            "parse_confidence": row.get("parse_confidence"),
            "requires_manual_review": row.get("requires_manual_review", False),
            "payment": payment,
        }

    def save_quality_check(self, row: Dict[str, Any]) -> Dict[str, Any]:
        username = str(row.get("username") or row.get("participant_id") or "")
        check_id = str(row.get("puzzle_id") or "")
        if not username or not check_id:
            raise ValueError("missing username or quality check id")
        quality_checks = load_quality_check_index()
        check = quality_checks.get(check_id)
        if not check:
            raise ValueError("invalid_quality_check_id")
        scored = score_raw_answer(check, str(row.get("raw_answer") or row.get("raw_response") or ""))
        if not scored.get("is_valid"):
            raise ValueError(f"invalid_quality_check_answer:{scored.get('invalid_reason')}")
        matched_solution = next(
            (solution for solution in check.get("solutions", []) if solution.get("solution_id") == scored.get("solution_id")),
            {},
        )
        row["is_correct"] = True
        row["server_scored"] = True
        row["solution_id"] = scored.get("solution_id")
        row["abstract_solution_id"] = matched_solution.get("abstract_solution_id") or scored.get("solution_id")
        row["accepted_solution_id"] = scored.get("solution_id")
        row["accepted_abstract_solution_id"] = row["abstract_solution_id"]
        row["parsed_answer"] = scored.get("parsed_answer")
        row["canonical_answer"] = scored.get("canonical_answer")
        row["parse_confidence"] = scored.get("parse_confidence")
        row["requires_manual_review"] = scored.get("requires_manual_review", False)
        with self.connect() as conn:
            participant = conn.execute("SELECT username, participant_source, tester_login FROM participants WHERE username = ?", (username,)).fetchone()
            if not participant:
                raise ValueError("unknown_username")
            row["participant_source"] = participant["participant_source"] or "prolific_or_local"
            if participant["tester_login"]:
                row["tester_login"] = participant["tester_login"]
            row["quota_counted"] = False
            row["bonus_earned"] = False
            row["bonus_usd"] = 0
            row.pop("participant_session_token", None)
            row.pop("session_token", None)
            conn.execute(
                """
                INSERT INTO quality_checks
                (username, check_id, row_json, is_correct, wrong_submit_attempts, response_time_ms, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(username, check_id) DO UPDATE SET
                    row_json = excluded.row_json,
                    is_correct = excluded.is_correct,
                    wrong_submit_attempts = excluded.wrong_submit_attempts,
                    response_time_ms = excluded.response_time_ms
                """,
                (
                    username,
                    check_id,
                    json_dumps_compact(row),
                    int(bool(row.get("is_correct"))),
                    int(row.get("wrong_submit_attempts") or 0),
                    int(row.get("response_time_ms") or 0),
                    now_iso(),
                ),
            )
            self.refresh_participant(conn, username)
            self.log_event(
                conn,
                "quality_check_save",
                username,
                {
                    "check_id": check_id,
                    "is_practice": bool(row.get("is_practice")),
                    "is_attention_check": bool(row.get("is_attention_check")),
                    "wrong_submit_attempts": int(row.get("wrong_submit_attempts") or 0),
                    "response_time_ms": int(row.get("response_time_ms") or 0),
                },
            )
        return {
            "ok": True,
            "is_correct": True,
            "solution_id": row.get("solution_id"),
            "abstract_solution_id": row.get("abstract_solution_id"),
            "accepted_solution_id": row.get("accepted_solution_id"),
            "accepted_abstract_solution_id": row.get("accepted_abstract_solution_id"),
            "parsed_answer": row.get("parsed_answer"),
            "canonical_answer": row.get("canonical_answer"),
            "parse_confidence": row.get("parse_confidence"),
            "requires_manual_review": row.get("requires_manual_review", False),
        }

    def study_duration_ms(self, conn: sqlite3.Connection, username: str, finish_payload: Dict[str, Any] | None = None) -> int | None:
        starts: List[datetime] = []
        ends: List[datetime] = []

        def add_payload_times(payload: Dict[str, Any]) -> None:
            start = parse_iso_datetime(payload.get("study_start_time_iso"))
            end = parse_iso_datetime(payload.get("trial_end_time_iso") or payload.get("study_end_time_iso"))
            if start:
                starts.append(start)
            if end:
                ends.append(end)

        for table in ("responses", "quality_checks"):
            for row in conn.execute(f"SELECT row_json FROM {table} WHERE username = ?", (username,)).fetchall():
                try:
                    add_payload_times(json.loads(row["row_json"]))
                except (TypeError, json.JSONDecodeError):
                    continue
        if finish_payload:
            add_payload_times(finish_payload)
            elapsed = finish_payload.get("study_elapsed_ms")
            if not starts or not ends:
                try:
                    parsed_elapsed = int(elapsed)
                except (TypeError, ValueError):
                    parsed_elapsed = 0
                if parsed_elapsed > 0:
                    return parsed_elapsed
        if not starts or not ends:
            return None
        duration = max(ends) - min(starts)
        return max(0, int(duration.total_seconds() * 1000))

    def finish(self, username: str, finish_payload: Dict[str, Any] | None = None) -> Dict[str, Any]:
        with self.connect() as conn:
            participant = conn.execute("SELECT * FROM participants WHERE username = ?", (username,)).fetchone()
            if not participant:
                raise ValueError("unknown_username")
            response_count = int(
                conn.execute(
                    "SELECT COUNT(DISTINCT puzzle_id) AS n FROM responses WHERE username = ?",
                    (username,),
                ).fetchone()["n"]
            )
            assigned_count = int(participant["assigned_count"])
            if response_count < assigned_count:
                raise ValueError(f"incomplete_assignment:{response_count}/{assigned_count}")
            assignment = json.loads(participant["assignment_json"])
            required_quality = expected_quality_check_count(assignment)
            quality_count = int(
                conn.execute(
                    "SELECT COUNT(DISTINCT check_id) AS n FROM quality_checks WHERE username = ?",
                    (username,),
                ).fetchone()["n"]
            )
            if quality_count < required_quality:
                raise ValueError(f"incomplete_quality_checks:{quality_count}/{required_quality}")
            self.refresh_participant(conn, username)
            payment = self.refresh_payment_ledger(conn, username)
            duration_ms = self.study_duration_ms(conn, username, finish_payload)
            valid_window_minutes = study_valid_window_minutes(str(participant["cohort"] or ""), assignment)
            valid_window_ms = valid_window_minutes * 60 * 1000
            if duration_ms is not None:
                conn.execute("UPDATE participants SET study_duration_ms = ? WHERE username = ?", (duration_ms, username))
            if (
                duration_ms is not None
                and duration_ms > valid_window_ms
                and str(participant["participant_source"] or "") != "tester"
            ):
                conn.execute(
                    "UPDATE participants SET excluded = 1, study_time_limit_exceeded = 1 WHERE username = ?",
                    (username,),
                )
                self.log_event(
                    conn,
                    "finish_blocked_time_limit",
                    username,
                    {"study_duration_ms": duration_ms, "study_valid_window_minutes": valid_window_minutes},
                )
                conn.commit()
                raise ValueError(f"study_time_limit_exceeded:{valid_window_minutes}")
            conn.execute("UPDATE participants SET completed_at = COALESCE(completed_at, ?) WHERE username = ?", (now_iso(), username))
            conn.execute("UPDATE participants SET study_time_limit_exceeded = 0 WHERE username = ?", (username,))
            completion = str(participant["completion_code"] or "")
            self.log_event(conn, "finish", username, {"completion_code": completion, "study_duration_ms": duration_ms})
        return {"ok": True, "completion_code": completion, "payment": payment, "study_duration_ms": duration_ms}

    def refresh_participant(self, conn: sqlite3.Connection, username: str) -> None:
        participant = conn.execute("SELECT participant_source, study_time_limit_exceeded FROM participants WHERE username = ?", (username,)).fetchone()
        counts = conn.execute(
            """
            SELECT COUNT(*) AS response_count, COALESCE(SUM(bad_record), 0) AS bad_record_count
            FROM responses WHERE username = ?
            """,
            (username,),
        ).fetchone()
        bad_count = int(counts["bad_record_count"])
        fast_count = int(
            conn.execute(
                """
                SELECT COUNT(*) AS n
                FROM responses
                WHERE username = ? AND is_correct = 1 AND bad_record = 0
                  AND response_time_ms > 0 AND response_time_ms <= ?
                """,
                (username, FAST_RESPONSE_MS),
            ).fetchone()["n"]
        )
        quality_flags = int(
            conn.execute(
                """
                SELECT COUNT(*) AS n
                FROM quality_checks
                WHERE username = ? AND (wrong_submit_attempts > 0 OR (response_time_ms > 0 AND response_time_ms <= ?))
                """,
                (username, FAST_RESPONSE_MS),
            ).fetchone()["n"]
        )
        rows = [
            json.loads(row["row_json"])
            for row in conn.execute("SELECT row_json FROM responses WHERE username = ?", (username,)).fetchall()
        ]
        exclusion = assignment_scheduler.bad_record_exclusion_summary(
            rows,
            bad_total_limit=BAD_RECORD_TOTAL_LIMIT,
            bad_block_limit=BAD_RECORD_BLOCK_LIMIT,
        )
        excluded_by_bad_records = username in set(exclusion["excluded_users"])
        participant_source = str(participant["participant_source"] or "") if participant else ""
        excluded = (
            bool(participant and participant["study_time_limit_exceeded"])
            or (excluded_by_bad_records and participant_source != "tester")
        )
        conn.execute(
            """
            UPDATE participants
            SET response_count = ?, bad_record_count = ?, excluded = ?, low_effort_flag_count = ?
            WHERE username = ?
            """,
            (int(counts["response_count"]), bad_count, int(excluded), fast_count + quality_flags, username),
        )


    def status(self) -> Dict[str, Any]:
        with self.connect() as conn:
            counts = {table: conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                      for table in ("participants", "responses", "quality_checks")}
        return {"ok": True, "database_path": str(self.path), "counts": counts}

    def export_rows_page(
        self,
        table: str,
        cohort: str = "",
        source: str = "",
        since: str = "",
        limit: int = 1000,
        offset: int = 0,
        include_global_events: bool = False,
    ) -> Dict[str, Any]:
        table = str(table or "").strip()
        cohort = str(cohort or "").strip()
        source = str(source or "").strip()
        since = str(since or "").strip()
        limit = max(1, min(int(limit), 5000))
        offset = max(0, int(offset))

        table_specs = {
            "participants": ("participants p", "p.*", "p", "p.registered_at, p.username"),
            "assignments": (
                "assignments a JOIN participants p ON p.username = a.username",
                "a.*",
                "p",
                "a.username, a.trial_order",
            ),
            "responses": (
                "responses r JOIN participants p ON p.username = r.username",
                "r.*",
                "p",
                "r.username, r.id",
            ),
            "quality_checks": (
                "quality_checks q JOIN participants p ON p.username = q.username",
                "q.*",
                "p",
                "q.username, q.id",
            ),
            "payment_ledger": (
                "payment_ledger pl JOIN participants p ON p.username = pl.username",
                "pl.*",
                "p",
                "pl.username",
            ),
        }
        event_specs = {
            "events": ("events e", "e.*", "e", "e.id"),
            "audit_events": ("audit_events e", "e.*", "e", "e.id"),
        }

        params: list[Any] = []
        if table in table_specs:
            from_clause, select_clause, participant_alias, order_clause = table_specs[table]
            where: list[str] = []
            if cohort:
                where.append(f"{participant_alias}.cohort = ?")
                params.append(cohort)
            if source:
                where.append(f"COALESCE({participant_alias}.participant_source, '') = ?")
                params.append(source)
            if since:
                where.append(f"{participant_alias}.registered_at >= ?")
                params.append(since)
            where_sql = f"WHERE {' AND '.join(where)}" if where else ""
        elif table in event_specs:
            from_clause, select_clause, event_alias, order_clause = event_specs[table]
            participant_filters: list[str] = []
            participant_params: list[Any] = []
            if cohort:
                participant_filters.append("p.cohort = ?")
                participant_params.append(cohort)
            if source:
                participant_filters.append("COALESCE(p.participant_source, '') = ?")
                participant_params.append(source)
            if since:
                participant_filters.append("p.registered_at >= ?")
                participant_params.append(since)
            where: list[str] = []
            if participant_filters:
                where.append(
                    f"{event_alias}.username IN (SELECT p.username FROM participants p WHERE {' AND '.join(participant_filters)})"
                )
                params.extend(participant_params)
                if include_global_events:
                    where.append(f"COALESCE({event_alias}.username, '') = ''")
                where_sql = f"WHERE ({' OR '.join(where)})"
            else:
                where_sql = ""
        else:
            raise ValueError("invalid_export_table")

        count_sql = f"SELECT COUNT(*) AS n FROM {from_clause} {where_sql}"
        page_sql = f"SELECT {select_clause} FROM {from_clause} {where_sql} ORDER BY {order_clause} LIMIT ? OFFSET ?"
        with self.connect() as conn:
            total = int(conn.execute(count_sql, params).fetchone()["n"] or 0)
            rows = [
                dict(row)
                for row in conn.execute(page_sql, [*params, limit, offset]).fetchall()
            ]
        next_offset = offset + len(rows)
        return {
            "ok": True,
            "table": table,
            "cohort": cohort or None,
            "source": source or None,
            "since": since or None,
            "limit": limit,
            "offset": offset,
            "row_count": len(rows),
            "total_count": total,
            "next_offset": next_offset if next_offset < total else None,
            "has_more": next_offset < total,
            "rows": rows,
        }


    def backup_database(self) -> Dict[str, Any]:
        backup_dir = self.path.parent / "backups"
        backup_dir.mkdir(parents=True, exist_ok=True)
        stamp = now_iso().replace(":", "").replace(".", "_").replace("Z", "")
        out = backup_dir / f"study_state_{stamp}.sqlite3"
        with self.connect() as source, closing(sqlite3.connect(out)) as target:
            source.backup(target)
        with self.connect() as conn:
            self.log_event(conn, "backup", "", {"backup_path": str(out)})
        return {"ok": True, "backup_path": str(out)}


class StudyHandler(SimpleHTTPRequestHandler):
    db: StudyDatabase
    admin_sessions: Dict[str, float] = {}
    login_failures: Dict[str, tuple[int, float]] = {}
    login_failure_lock = threading.Lock()

    def request_host(self) -> str:
        return (self.headers.get("Host") or "").split(":", 1)[0].lower()

    def cohort_from_host(self) -> str:
        host = self.request_host()
        if host.startswith("main."):
            return "main"
        if host.startswith("module."):
            return "module"
        return ""

    def is_public_deployment_host(self) -> bool:
        host = self.request_host()
        return bool(host) and host not in {"localhost", "127.0.0.1", "["}

    def canonical_render_redirect(self, parsed: Any) -> str:
        host = self.request_host()
        if host not in RAW_RENDER_HOSTS:
            return ""
        params = parse_qs(parsed.query)
        requested = ((params.get("study") or params.get("cohort") or [""])[0] or "").lower()
        cohort = "module" if requested == "module" else "main"
        target = CANONICAL_STUDY_HOSTS[cohort]
        if not target:
            return ""
        path = parsed.path or "/"
        query = f"?{parsed.query}" if parsed.query else ""
        return f"https://{target}{path}{query}"

    def send_redirect(self, location: str) -> None:
        self.send_response(302)
        self.send_header("Location", location)
        self.send_header("Cache-Control", "no-store")
        self.end_headers()

    def handle_read_request(self) -> None:
        parsed = urlparse(self.path)
        canonical_location = self.canonical_render_redirect(parsed)
        if canonical_location:
            self.send_redirect(canonical_location)
            return
        if parsed.path in {"", "/"}:
            cohort = self.cohort_from_host()
            if cohort:
                self.send_redirect(f"/study_web/app/?formal=1&study={cohort}")
            else:
                self.send_redirect("/study_web/app/?formal=1&study=main")
            return
        if parsed.path == "/study_web/app/" and not parsed.query:
            cohort = self.cohort_from_host()
            if cohort:
                self.send_redirect(f"/study_web/app/?formal=1&study={cohort}")
                return
        if parsed.path in {"/api/status", "/api/admin/status"}:
            if not self.require_admin():
                return
            self.send_json(self.db.status())
            return
        if parsed.path == "/api/admin/export_rows":
            if not self.require_admin():
                return
            params = parse_qs(parsed.query)
            try:
                limit = int((params.get("limit") or ["1000"])[0])
            except ValueError:
                limit = 1000
            try:
                offset = int((params.get("offset") or ["0"])[0])
            except ValueError:
                offset = 0
            include_global_events = str((params.get("include_global_events") or ["0"])[0]).lower() in {"1", "true", "yes", "on"}
            self.send_json(
                self.db.export_rows_page(
                    table=str((params.get("table") or [""])[0]),
                    cohort=str((params.get("cohort") or [""])[0]),
                    source=str((params.get("source") or [""])[0]),
                    since=str((params.get("since") or [""])[0]),
                    limit=limit,
                    offset=offset,
                    include_global_events=include_global_events,
                )
            )
            return
        if self.is_admin_entry_path(parsed.path):
            self.invalidate_admin_session()
        self.serve_static(parsed.path)

    def do_GET(self) -> None:
        self.handle_read_request()

    def do_HEAD(self) -> None:
        self.handle_read_request()

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        try:
            body = self.read_json()
            if parsed.path == "/api/register":
                self.handle_register(body)
            elif parsed.path == "/api/tester/login":
                self.handle_tester_login(body)
            elif parsed.path == "/api/response":
                self.require_participant_session(body)
                self.send_json(self.db.save_response(body))
            elif parsed.path == "/api/quality_check":
                self.require_participant_session(body)
                self.send_json(self.db.save_quality_check(body))
            elif parsed.path == "/api/finish":
                self.require_participant_session(body)
                username = str(body.get("username") or "")
                self.send_json(self.db.finish(username, body))
            elif parsed.path == "/api/admin/login":
                self.handle_admin_login(body)
            elif parsed.path == "/api/admin/logout":
                self.handle_admin_logout()
            elif parsed.path == "/api/admin/backup":
                if not self.require_admin():
                    return
                self.send_json(self.db.backup_database())
            else:
                self.send_error(404, "Unknown API endpoint")
        except ValueError as error:
            message = str(error)
            status = 409 if message == "username_exists" else 400
            self.send_json({"ok": False, "error": message}, status=status)
        except Exception as error:  # pragma: no cover - defensive server boundary
            sys.stderr.write(f"Internal API error at {parsed.path}: {type(error).__name__}: {error}\n")
            self.send_json({"ok": False, "error": "internal_server_error"}, status=500)

    def handle_register(self, body: Dict[str, Any]) -> None:
        username = str(body.get("username") or "").strip()
        if not USERNAME_RE.match(username):
            raise ValueError("invalid_username")
        prolific_pid = str(body.get("prolific_pid") or "").strip()
        prolific_study_id = str(body.get("prolific_study_id") or body.get("study_id") or "").strip()
        prolific_session_id = str(body.get("prolific_session_id") or body.get("session_id") or "").strip()
        access_token = str(body.get("access_token") or "").strip()
        prolific_token = str(body.get("prolific_token") or "").strip()
        prolific_verification = verify_prolific_access(
            prolific_pid,
            prolific_study_id,
            prolific_session_id,
            access_token,
            prolific_token,
        )
        cohort = infer_cohort(username, body.get("cohort"))
        if prolific_pid:
            expected_username = expected_prolific_username(prolific_pid, cohort)
            if username != expected_username:
                raise ValueError("username_prolific_pid_mismatch")
        assignment = self.db.register(
            username,
            cohort,
            prolific_pid=prolific_pid,
            prolific_study_id=prolific_study_id,
            prolific_session_id=prolific_session_id,
            consent_at=str(body.get("consent_at") or "").strip(),
        )
        if prolific_verification:
            with self.db.connect() as conn:
                self.db.log_event(conn, "prolific_verify", username, prolific_verification)
        include_completion_code = False
        if assignment.get("resume"):
            with self.db.connect() as conn:
                completed = conn.execute(
                    "SELECT completed_at FROM participants WHERE username = ?",
                    (username,),
                ).fetchone()
                include_completion_code = bool(completed and completed["completed_at"])
        self.send_json(
            {
                "ok": True,
                "username": username,
                "cohort": cohort,
                "assignment": public_assignment_payload(assignment, include_completion_code=include_completion_code),
            }
        )

    def handle_tester_login(self, body: Dict[str, Any]) -> None:
        tester_login = str(body.get("tester_username") or body.get("username") or "").strip()
        password = str(body.get("password") or "")
        requested_cohort = str(body.get("cohort") or "main").strip()
        cohort = infer_cohort("", requested_cohort)
        requested_session_id = str(body.get("tester_session_id") or "").strip()
        if cohort not in {"main", "module"}:
            raise ValueError("invalid_tester_cohort")
        throttle_key = self.login_throttle_key("tester", tester_login)
        if self.reject_login_if_throttled(throttle_key):
            return
        if ADMIN_USES_DEV_DEFAULT and self.is_public_deployment_host():
            self.send_json({"ok": False, "error": "tester_login_not_configured"}, status=503)
            return
        if not valid_researcher_test_credentials(tester_login, password):
            self.record_failed_login(throttle_key)
            self.send_json({"ok": False, "error": "invalid_tester_login"}, status=401)
            return
        self.clear_login_failures(throttle_key)
        username = tester_participant_username(tester_login, cohort, requested_session_id)
        assignment = self.db.register(
            username,
            cohort,
            consent_at=str(body.get("consent_at") or "").strip(),
            participant_source="tester",
            tester_login=tester_login,
        )
        include_completion_code = False
        if assignment.get("resume"):
            with self.db.connect() as conn:
                completed = conn.execute(
                    "SELECT completed_at FROM participants WHERE username = ?",
                    (username,),
                ).fetchone()
                include_completion_code = bool(completed and completed["completed_at"])
        with self.db.connect() as conn:
            self.db.log_event(
                conn,
                "tester_login",
                username,
                {"tester_login": tester_login, "cohort": cohort, "resume": bool(assignment.get("resume"))},
            )
        self.send_json(
            {
                "ok": True,
                "username": username,
                "tester_login": tester_login,
                "cohort": cohort,
                "assignment": public_assignment_payload(assignment, include_completion_code=include_completion_code),
            }
        )

    def require_participant_session(self, body: Dict[str, Any]) -> None:
        username = str(body.get("username") or body.get("participant_id") or "").strip()
        token = str(body.get("participant_session_token") or body.get("session_token") or "").strip()
        if not username or not token:
            raise ValueError("participant_session_required")
        with self.db.connect() as conn:
            row = conn.execute("SELECT participant_session_hash FROM participants WHERE username = ?", (username,)).fetchone()
        if not row:
            raise ValueError("unknown_username")
        expected = str(row["participant_session_hash"] or "")
        if not expected or not secrets.compare_digest(hash_session_token(token), expected):
            raise ValueError("invalid_participant_session")

    def handle_admin_login(self, body: Dict[str, Any]) -> None:
        username = str(body.get("username") or "")
        password = str(body.get("password") or "")
        throttle_key = self.login_throttle_key("admin", username)
        if self.reject_login_if_throttled(throttle_key):
            return
        if ADMIN_USES_DEV_DEFAULT and self.is_public_deployment_host():
            self.send_json({"ok": False, "error": "admin_credentials_not_configured"}, status=503)
            return
        valid_admin = username == ADMIN_USERNAME and password == ADMIN_PASSWORD
        if not valid_admin:
            self.record_failed_login(throttle_key)
            self.send_json({"ok": False, "error": "invalid_admin_login"}, status=401)
            return
        self.clear_login_failures(throttle_key)
        token = secrets.token_urlsafe(32)
        self.admin_sessions[token] = time.time()
        with self.db.connect() as conn:
            self.db.log_event(
                conn,
                "admin_login",
                username,
                {"uses_dev_default": ADMIN_USES_DEV_DEFAULT},
            )
        self.send_json(
            {"ok": True, "username": username},
            headers={
                "Set-Cookie": f"{ADMIN_COOKIE}={token}; {self.admin_cookie_attributes()}",
            },
        )

    def login_throttle_key(self, scope: str, username: str) -> str:
        forwarded_for = (self.headers.get("X-Forwarded-For") or "").split(",", 1)[0].strip()
        client = forwarded_for
        if not client:
            client_address = getattr(self, "client_address", ("unknown",))
            client = str(client_address[0] if client_address else "unknown")
        normalized_username = (username or "-").strip().lower()
        return f"{scope}:{client}:{normalized_username}"

    @classmethod
    def login_attempt_is_blocked(cls, key: str) -> bool:
        if LOGIN_FAILURE_LIMIT <= 0:
            return False
        now = time.time()
        with cls.login_failure_lock:
            count, first_failure_at = cls.login_failures.get(key, (0, now))
            if now - first_failure_at > LOGIN_FAILURE_WINDOW_SECONDS:
                cls.login_failures.pop(key, None)
                return False
            return count >= LOGIN_FAILURE_LIMIT

    @classmethod
    def record_failed_login(cls, key: str) -> None:
        if LOGIN_FAILURE_LIMIT <= 0:
            return
        now = time.time()
        with cls.login_failure_lock:
            count, first_failure_at = cls.login_failures.get(key, (0, now))
            if now - first_failure_at > LOGIN_FAILURE_WINDOW_SECONDS:
                cls.login_failures[key] = (1, now)
            else:
                cls.login_failures[key] = (count + 1, first_failure_at)

    @classmethod
    def clear_login_failures(cls, key: str) -> None:
        with cls.login_failure_lock:
            cls.login_failures.pop(key, None)

    def reject_login_if_throttled(self, key: str) -> bool:
        if not self.login_attempt_is_blocked(key):
            return False
        self.send_json({"ok": False, "error": "too_many_login_attempts"}, status=429)
        return True

    def handle_admin_logout(self) -> None:
        token = self.admin_token()
        if token:
            self.admin_sessions.pop(token, None)
        self.send_json(
            {"ok": True},
            headers={
                "Set-Cookie": f"{ADMIN_COOKIE}=; {self.admin_cookie_attributes()}; Max-Age=0",
            },
        )

    def admin_cookie_attributes(self) -> str:
        attrs = "HttpOnly; SameSite=Strict; Path=/"
        forwarded_proto = (self.headers.get("X-Forwarded-Proto") or "").split(",", 1)[0].strip().lower()
        host = (self.headers.get("Host") or "").split(":", 1)[0].lower()
        if forwarded_proto == "https":
            attrs += "; Secure"
        return attrs

    def admin_token(self) -> str | None:
        cookie_header = self.headers.get("Cookie")
        if not cookie_header:
            return None
        cookies = SimpleCookie(cookie_header)
        morsel = cookies.get(ADMIN_COOKIE)
        return morsel.value if morsel else None

    def invalidate_admin_session(self) -> None:
        token = self.admin_token()
        if token:
            self.admin_sessions.pop(token, None)

    @staticmethod
    def is_admin_entry_path(path: str) -> bool:
        clean = posixpath.normpath(unquote(path)).lstrip("/")
        return clean in {"study_web/admin", "study_web/admin/index.html"}

    def require_admin(self) -> bool:
        if self.admin_is_authenticated():
            return True
        self.send_json({"ok": False, "error": "admin_login_required"}, status=401)
        return False

    def admin_is_authenticated(self) -> bool:
        token = self.admin_token()
        issued_at = self.admin_sessions.get(token or "")
        if issued_at and time.time() - issued_at <= ADMIN_SESSION_TTL_SECONDS:
            return True
        if token:
            self.admin_sessions.pop(token, None)
        return False

    def read_json(self) -> Dict[str, Any]:
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            raise ValueError("invalid_content_length") from None
        if length > MAX_JSON_BODY_BYTES:
            raise ValueError("request_body_too_large")
        raw = self.rfile.read(length).decode("utf-8")
        try:
            payload = json.loads(raw or "{}")
        except json.JSONDecodeError:
            raise ValueError("invalid_json") from None
        if not isinstance(payload, dict):
            raise ValueError("invalid_json_object")
        return payload

    def send_json(self, payload: Dict[str, Any], status: int = 200, headers: Dict[str, str] | None = None) -> None:
        data = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        response_headers = {
            "Cache-Control": "no-store",
            "X-Content-Type-Options": "nosniff",
            **(headers or {}),
        }
        for key, value in response_headers.items():
            self.send_header(key, value)
        self.end_headers()
        if getattr(self, "command", "GET") != "HEAD":
            self.wfile.write(data)

    def serve_static(self, request_path: str) -> None:
        clean = posixpath.normpath(unquote(request_path)).lstrip("/")
        if clean in {"", "."}:
            clean = "study_web/app/index.html"
        public = self.static_path_matches(clean, PUBLIC_STATIC_PREFIXES) or clean in PUBLIC_STATIC_FILES
        protected = (not public) and self.static_path_matches(clean, ADMIN_PROTECTED_STATIC_PREFIXES)
        allowed = public or protected
        if not allowed or clean.startswith(".git/") or "/.git/" in clean:
            self.send_error(403)
            return
        if protected and not self.admin_is_authenticated():
            self.send_static_auth_required(request_path)
            return
        path = (ROOT / "collection" / "human" / clean) if clean.startswith("study_web/") else ROOT / clean
        if request_path.endswith("/") or path.is_dir():
            path = path / "index.html"
        try:
            path.relative_to(ROOT)
        except ValueError:
            self.send_error(403)
            return
        if path.suffix.lower() in SENSITIVE_STATIC_SUFFIXES:
            self.send_error(403)
            return
        if not path.exists() or not path.is_file():
            self.send_error(404)
            return
        data = path.read_bytes()
        content_type = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        if path.suffix in {".jsonl", ".md"}:
            content_type = "text/plain; charset=utf-8"
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store, max-age=0")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        if getattr(self, "command", "GET") != "HEAD":
            self.wfile.write(data)

    @staticmethod
    def static_path_matches(clean: str, prefixes: tuple[str, ...]) -> bool:
        return any(clean == prefix.rstrip("/") or clean.startswith(prefix) for prefix in prefixes)

    def send_static_auth_required(self, request_path: str) -> None:
        clean = posixpath.normpath(unquote(request_path)).lstrip("/")
        suffix = Path(clean).suffix.lower()
        if request_path.endswith("/") or suffix in {"", ".html"}:
            next_path = quote(getattr(self, "path", request_path), safe="/?=&%")
            self.send_redirect(f"/study_web/admin/?next={next_path}")
            return
        self.send_json({"ok": False, "error": "admin_login_required"}, status=401)

    @staticmethod
    def sanitize_log_request_line(request_line: str) -> str:
        parts = str(request_line or "").split(" ")
        if len(parts) < 2:
            return str(request_line or "")
        parsed = urlparse(parts[1])
        if parsed.query:
            clean_target = parsed.path or "/"
            parts[1] = f"{clean_target}?[query redacted]"
        return " ".join(parts)

    def log_message(self, format: str, *args: Any) -> None:
        safe_args = list(args)
        if safe_args and isinstance(safe_args[0], str):
            safe_args[0] = self.sanitize_log_request_line(safe_args[0])
        sys.stderr.write("%s - - [%s] %s\n" % (self.address_string(), self.log_date_time_string(), format % tuple(safe_args)))


def main() -> int:
    parser = argparse.ArgumentParser(description="Serve the study web app with a lightweight SQLite backend.")
    parser.add_argument("--port", type=int, default=8015)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--db", default=str(DEFAULT_DB))
    args = parser.parse_args()

    db = StudyDatabase(Path(args.db))
    handler = type("ConfiguredStudyHandler", (StudyHandler,), {"db": db})
    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(f"Study server: http://{args.host}:{args.port}/study_web/app/")
    print(f"Study administration: http://{args.host}:{args.port}/study_web/admin/")
    print(f"SQLite database: {args.db}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
