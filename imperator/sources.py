"""I/O layer: read the world (Redis via PersistenceManager + held NATS stream state)
into plain dicts for the pure compute modules. fakeredis-testable; no asyncio.
"""

from __future__ import annotations

from datetime import datetime


def _to_epoch(ts) -> float:
    """Coerce an ISO-8601 string or numeric epoch to a float epoch (0.0 on failure)."""
    if ts is None:
        return 0.0
    if isinstance(ts, (int, float)):
        return float(ts)
    try:
        return datetime.fromisoformat(str(ts)).timestamp()
    except (ValueError, TypeError):
        return 0.0


def _current_sid(pm) -> str | None:
    """Extract the session_id from the current-session DICT (not a bare SID)."""
    cur = pm.load_current_session()
    return cur.get("session_id") if isinstance(cur, dict) else None


def _recent_session_ids(pm, limit: int = 20) -> list[str]:
    """Newest-first feedback session ids (list_feedback_sessions does NOT exist)."""
    return [
        r.get("session_id")
        for r in (pm.get_all_feedback(limit=limit) or [])
        if isinstance(r, dict) and r.get("session_id")
    ]


def resolve_latest_reflection(pm) -> dict | None:
    """Newest reflection report that ACTUALLY EXISTS (reflection lags feedback).

    Try current session, then iterate recent SIDs; rank by report timestamp
    (ISO string -> epoch). None if none exist.
    """
    candidates: list[str] = []
    cur = _current_sid(pm)
    if cur:
        candidates.append(cur)
    for sid in _recent_session_ids(pm):
        if sid not in candidates:
            candidates.append(sid)

    best, best_ts = None, float("-inf")
    for sid in candidates:
        report = pm.load_reflection(sid)
        if report is None:
            continue
        ts = _to_epoch(report.get("timestamp"))
        if ts >= best_ts:
            best, best_ts = report, ts
    return best


def resolve_latest_decision(pm) -> dict | None:
    """Fuse latest FIRED advice (ISO timestamp) + latest SUPPRESSION (numeric ts)."""
    advice = pm.load_last_advice()
    silences = pm.load_silence_records(limit=1)
    silence = silences[0] if silences else None
    if advice is None and silence is None:
        return None
    advice_ts = _to_epoch(advice.get("timestamp")) if advice else float("-inf")
    silence_ts = float(silence.get("ts", float("-inf"))) if silence else float("-inf")
    if silence is not None and silence_ts >= advice_ts:
        return {
            "decision": "suppressed",
            "arm": silence.get("arm"),
            "reason": silence.get("reason"),
            "decision_as_of": silence_ts,
        }
    return {
        "decision": "fired",
        "arm": None,
        "reason": None,
        "decision_as_of": advice_ts,
        "decision_id": (advice or {}).get("decision_id"),
    }


def resolve_reception(pm, last_advice) -> dict | None:
    """Reception of the last fired advice, matched by decision_id in current feedback."""
    if not last_advice or not last_advice.get("decision_id"):
        return (
            None  # no decision_id -> cannot safely join (avoid false None==None match)
        )
    sid = _current_sid(pm)
    if not sid:
        return None
    fb = pm.get_feedback(sid)
    if not fb:
        return None
    target = last_advice.get("decision_id")
    for ev in reversed(fb.get("advice_events", [])):
        if ev.get("decision_id") == target:
            return {
                "explicit_rating": ev.get("explicit_rating"),
                "behavioral_score": ev.get("behavioral_score"),
                "finalized": ev.get("behavioral_finalized"),
                "unmeasurable": ev.get("unmeasurable"),
            }
    return None
