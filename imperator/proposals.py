"""Pure proposal logic: schema, deterministic classification, ranking, gate. No I/O/LLM."""

from __future__ import annotations

import hashlib
import json
import uuid

from consilium.prompt_safety import is_prompt_acceptable

_KIND_KLASS = {
    "escalation_rule": "safe",
    "prompt_strategy": "safe",
    "sigma": "safe",
    "gate_calibration": "safe",
    "observe_more": "safe",
    "code": "gated",
    "structural": "gated",
}
_AUTO_APPLY_KINDS = {"escalation_rule", "prompt_strategy"}


def dedupe_key(kind: str, target: str) -> str:
    return hashlib.sha256(f"{kind}\x00{target}".encode()).hexdigest()[:16]


def make_proposal(
    *,
    kind: str,
    target: str,
    action: dict,
    rationale: str,
    klass: str = "safe",
    rank: int = 100,
    source_blind_spot=None,
    now: float = 0.0,
) -> dict:
    return {
        "proposal_id": uuid.uuid4().hex,
        "dedupe_key": dedupe_key(kind, target),
        "ts": now,
        "klass": klass,
        "kind": kind,
        "target": target,
        "rank": rank,
        "rationale": rationale,
        "action": action,
        "status": "logged",
        "applied_session": None,
        "source_blind_spot": source_blind_spot,
    }


def normalize_klass(p: dict) -> dict:
    """Deterministically set klass from kind (override any LLM claim)."""
    p["klass"] = _KIND_KLASS.get(p["kind"], "gated")
    return p


def rank(props: list[dict]) -> list[dict]:
    return sorted(props, key=lambda p: p.get("rank", 100))


def is_auto_applicable(p: dict) -> bool:
    return (
        p["klass"] == "safe"
        and p["kind"] in _AUTO_APPLY_KINDS
        and p["status"] == "logged"
    )


def gate(p: dict, *, cfg, recent_self_tuning: dict, applied_keys: set) -> dict:
    """Deterministic floor. (Reversibility for prompts needs I/O -> enforced in apply.py.)"""
    if p["klass"] == "gated":
        p["status"] = "logged"
        return p
    if p["kind"] == "prompt_strategy" and not is_prompt_acceptable(
        (p.get("action") or {}).get("text", ""), cfg
    ):
        p["status"] = "logged"
        return p
    if p["dedupe_key"] in applied_keys or _matches_recent(p, recent_self_tuning):
        p["status"] = "skipped"
        return p
    p["status"] = "logged"
    return p


def _matches_recent(p: dict, recent: dict) -> bool:
    return bool(recent) and p["target"] in json.dumps(recent)
