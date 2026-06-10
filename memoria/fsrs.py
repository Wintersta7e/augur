"""Pure FSRS-exponential decay/review math for Memoria (Lane 2).

No Redis, no AugurConfig mutation. Functions take plain state dicts + the
config (for knobs) and return new state. The decay clock is in ACTIVE-SESSION
units. See docs/superpowers/specs/2026-06-10-memoria-memory-spine.md §4.
"""

from __future__ import annotations

import hashlib
import json

_VALID_SEVERITIES = {"LOW", "MEDIUM", "HIGH"}


def normalize_severity(raw: str | None) -> str:
    """Upper-case a severity label; unknown/None → 'LOW' (conservative)."""
    s = (raw or "").strip().upper()
    return s if s in _VALID_SEVERITIES else "LOW"


def make_memory_id(pattern: dict) -> str:
    """Deterministic SHA-256 id for a correlation edge-pattern.

    Identity = {kind, sorted domains, rule_key, severity}; timestamps and
    session ids are excluded so the same pattern in a later session maps to
    the same id (recurrence == review).
    """
    canonical = {
        "kind": pattern["kind"],
        "domains": sorted(pattern["domains"]),
        "rule_key": pattern.get("rule_key"),
        "severity": pattern["severity"],
    }
    blob = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()


def retrievability(state: dict, active_session: int, cfg) -> float:
    """R(t,S) = 0.9 ** (t / max(S, S_MIN)); t in active-session units."""
    t = max(0, active_session - state["last_review_session"])
    s = max(state["S"], cfg.memory_s_min)
    return 0.9 ** (t / s)


def review(state: dict, active_session: int, session_id: str, cfg) -> dict:
    """Apply a recurrence review (spacing effect). Idempotent per session:
    returns the SAME object unchanged if session_id already reviewed it."""
    if session_id in state.get("source_sessions", []):
        return state
    new = dict(state)
    new["S"] = min(state["S"] * (1.0 + cfg.memory_s_growth_factor), cfg.memory_s_max)
    new["last_review_session"] = active_session
    new["source_sessions"] = [*state.get("source_sessions", []), session_id]
    return new
