"""Pure proposal logic: schema, deterministic classification, ranking, gate. No I/O/LLM."""

from __future__ import annotations

import hashlib
import uuid

from consilium.prompt_safety import is_prompt_acceptable

_KIND_KLASS = {
    "escalation_rule": "safe",
    "prompt_strategy": "safe",
    "sigma": "safe",
    "gate_calibration": "safe",
    "observe_more": "safe",
    "context_directive": "safe",
    "semantic_fact": "safe",
    "code": "gated",
    "structural": "gated",
}
_AUTO_APPLY_KINDS = {"escalation_rule", "prompt_strategy"}

# Kinds a human-confirmed dialogue act may apply (all SAFE, actionable kinds).
# observe_more is safe but a no-op suggestion; gated kinds are excluded by design.
_CONFIRMED_APPLY_KINDS = {
    "escalation_rule",
    "prompt_strategy",
    "sigma",
    "gate_calibration",
    "context_directive",
    "semantic_fact",
}


def dedupe_key(kind: str, target: str) -> str:
    """Idempotency/anti-thrash identity: (kind, target) only — NOT the action.

    Deliberate: once any change to a target is applied, the applied-TTL key
    (augur:imperator:applied:{key}, imperator_ii_dedupe_staleness_s) blocks EVERY
    further proposal for that target — including an opposite-direction correction
    — until the window expires. This is "one move per target per staleness window"
    anti-oscillation; the trade-off is that a wrong move waits out the TTL rather
    than being walked back through the normal gate.
    """
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
    source: str = "imperator_ii",
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
        "source": source,
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
    """True if Disciplina (rung-0 autonomic tuning) just changed the same surface
    this proposal targets, so II defers a cycle to avoid thrashing a fresh change.

    `recent` is the reflection's `adjustments` block — coarse booleans + per-domain
    sigma, no rule-key granularity — so escalation/window proposals defer when the
    matrix/windows were tuned this session, and prompt proposals defer when a prompt
    was mutated *for their own domain* or the target domain's sigma moved. Structured
    field checks, not a substring scan of the serialized blob (which both over- and
    under-matched).

    Domain-scoping of `prompt_mutated`: the flag is a bare session-wide boolean with
    no domain, but the reflection only ever mutates the prompt for its single derived
    session domain — and that domain always has a `sigma_values` entry (it is drawn
    from the same advice events that populate the per-domain sigma map). So a prompt
    proposal defers only when its OWN target domain is one the reflection actually
    touched this session (a `sigma_values` member). A blanket defer on `prompt_mutated`
    alone wrongly held back prompt proposals for *unrelated* domains — domains the
    reflection never looked at, which therefore have no fresh change to thrash.
    """
    if not recent:
        return False
    kind = p.get("kind")
    if kind == "escalation_rule":
        if "window" in (p.get("action") or {}):
            return bool(recent.get("windows_tuned"))
        return bool(recent.get("matrix_mutated"))
    if kind == "prompt_strategy":
        domain = (p.get("action") or {}).get("domain", p.get("target"))
        # Scope to the proposal's OWN domain: defer only when the reflection
        # actually touched this domain this session (a `sigma_values` member).
        # The reflection mutates the prompt for exactly its single derived session
        # domain, which is always such a member, so a touched domain captures both
        # "sigma moved here" and "prompt was mutated here". This is the fix: a
        # session-wide `prompt_mutated` no longer defers *unrelated* domains that
        # the reflection never looked at — they have no fresh change to thrash.
        return domain in (recent.get("sigma_values") or {})
    return False
