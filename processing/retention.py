"""Pure study retention policy shared by collection and offline processing."""
from collections import Counter, defaultdict
from typing import Any, Dict, List, Sequence
MODULES = ("cue", "spatial", "formulation", "pair", "transfer")

def participant_key(row: Dict[str, Any]) -> str:
    return str(row.get("username") or row.get("participant_id") or "UNKNOWN")

def is_bad_response_record(row: Dict[str, Any]) -> bool:
    if row.get("bad_record") is not None:
        return bool(row.get("bad_record"))
    return bool(row.get("timeout") or row.get("gave_up") or row.get("skipped"))

def bad_record_block_key(row: Dict[str, Any]) -> str:
    puzzle_type = str(row.get("puzzle_type") or "").strip()
    if puzzle_type:
        return f"puzzle_type:{puzzle_type}"
    module = str(row.get("module") or "").strip()
    if module:
        return f"module:{module}"
    return "unknown"

def bad_record_exclusion_summary(
    responses: Sequence[Dict[str, Any]],
    bad_total_limit: int = 30,
    bad_block_limit: int = 14,
) -> Dict[str, Any]:
    bad_by_user: Counter[str] = Counter()
    bad_by_user_block: Dict[str, Counter[str]] = defaultdict(Counter)
    for row in responses:
        if not is_bad_response_record(row):
            continue
        username = participant_key(row)
        block = bad_record_block_key(row)
        bad_by_user[username] += 1
        bad_by_user_block[username][block] += 1

    excluded = set()
    reasons: Dict[str, List[str]] = {}
    for username, total in bad_by_user.items():
        user_reasons = []
        if total > bad_total_limit:
            user_reasons.append(f"total_bad_records>{bad_total_limit}")
        over_limit_blocks = [
            f"{block}>{bad_block_limit}"
            for block, count in sorted(bad_by_user_block.get(username, {}).items())
            if count > bad_block_limit
        ]
        user_reasons.extend(over_limit_blocks)
        if user_reasons:
            excluded.add(username)
            reasons[username] = user_reasons

    return {
        "excluded_users": sorted(excluded),
        "bad_by_user": dict(bad_by_user),
        "bad_by_user_block": {username: dict(counts) for username, counts in bad_by_user_block.items()},
        "exclusion_reasons": reasons,
        "bad_total_limit": bad_total_limit,
        "bad_block_limit": bad_block_limit,
    }

def retention_by_user(participant_rows, response_payloads, *, bad_total_limit=30, bad_block_limit=14, module_min_correct=40, module_min_per_module=2):
    participants = {str(row["username"]): row for row in participant_rows}
    responses_by_user = defaultdict(list)
    for row in response_payloads:
        responses_by_user[str(row["username"])].append(row)
    bad_summary = bad_record_exclusion_summary(
        response_payloads,
        bad_total_limit=bad_total_limit,
        bad_block_limit=bad_block_limit,
    )
    bad_excluded = set(bad_summary["excluded_users"])
    bad_counts = Counter(bad_summary["bad_by_user"])
    bad_blocks = bad_summary["bad_by_user_block"]
    out: Dict[str, Dict[str, Any]] = {}
    for username, participant in participants.items():
        cohort = str(participant.get("cohort") or "")
        source = str(participant.get("participant_source") or "")
        assigned_count = int(participant.get("assigned_count") or 0)
        user_responses = responses_by_user.get(username, [])
        response_count = len({str(row.get("puzzle_id") or "") for row in user_responses})
        finished = assigned_count > 0 and response_count >= assigned_count
        correct_count = sum(1 for row in user_responses if row.get("is_correct"))
        module_correct = {
            module: 0
            for module in MODULES
        }
        for row in user_responses:
            module = str(row.get("module") or "")
            if module in module_correct and row.get("is_correct"):
                module_correct[module] += 1

        reasons: List[str] = []
        if source == "tester":
            reasons.append("tester")
        if not finished:
            reasons.append("not_finished")
        if cohort == "module":
            if finished and correct_count < module_min_correct:
                reasons.append(f"total_correct<{module_min_correct}")
            if finished:
                for module in MODULES:
                    if module_correct.get(module, 0) < module_min_per_module:
                        reasons.append(f"{module}_correct<{module_min_per_module}")
        elif cohort == "main":
            if username in bad_excluded:
                reasons.extend(bad_summary["exclusion_reasons"].get(username, ["bad_record_rule"]))
        elif username in bad_excluded:
            reasons.extend(bad_summary["exclusion_reasons"].get(username, ["bad_record_rule"]))

        retained = not reasons
        out[username] = {
            "username": username,
            "cohort": cohort,
            "participant_source": source,
            "finished": finished,
            "analysis_retained": retained,
            "analysis_excluded": not retained,
            "analysis_exclusion_reasons": reasons,
            "assigned_count": assigned_count,
            "response_count": response_count,
            "correct_count": correct_count,
            "bad_record_count": bad_counts.get(username, 0),
            "bad_record_block_counts": bad_blocks.get(username, {}),
            "module_correct": module_correct,
            "time_limit_ignored": True,
        }
    return out
