"""Rubric verdicts over Imperator II's gated-proposal log (spec S5).

Folds the research tree's four-part self-modification contract into checks:
tests-100% (tests_verifiable), bounded change (bounded), rollback
(reversible), human review (the recommendation itself). v1 never emits
"approve": reversibility and test verification are structurally
unsatisfiable pre-application for code/structural changes, so every
non-rejected gated proposal needs the human (spec D11)."""

from __future__ import annotations

import json
import time

from conscientia import charter

_MAX_ACTION_BYTES = 4096
_MAX_NESTED_PAYLOAD_BYTES = 2048


def _max_nested_payload_len(action: dict) -> int:
    """Longest serialized string (value OR key) anywhere inside *action* —
    spec R4's nested clause: a total under 4 KiB can still smuggle one big
    code blob, which is the shape this catches.

    Iterative walk, measured with the same escaped-serialization convention
    as the total bound. Only called after ``json.dumps`` succeeded on the
    whole action (see _checks), so the structure is known acyclic and the
    walk terminates.
    """
    longest = 0
    stack: list[object] = [action]
    while stack:
        node = stack.pop()
        if isinstance(node, dict):
            stack.extend(node.keys())
            stack.extend(node.values())
        elif isinstance(node, (list, tuple)):
            stack.extend(node)
        elif isinstance(node, str):
            longest = max(longest, len(json.dumps(node)))
    return longest


def _checks(p: dict) -> list[dict]:
    klass_ok = p.get("klass") == "gated"
    target = str(p.get("target") or "").lower()
    surface_ok = not any(
        target.startswith(prefix.lower()) for prefix in charter.PROTECTED_SURFACES
    )
    action = p.get("action")
    try:
        # Total bound first: json.dumps raising on a cyclic action fails the
        # check before _max_nested_payload_len would walk the cycle.
        if isinstance(action, dict):
            total_ok = len(json.dumps(action, default=str)) <= _MAX_ACTION_BYTES
            nested_ok = (
                total_ok
                and _max_nested_payload_len(action) <= _MAX_NESTED_PAYLOAD_BYTES
            )
        else:
            total_ok = nested_ok = False
    except (TypeError, ValueError):
        total_ok = nested_ok = False
    bounded_ok = total_ok and nested_ok
    return [
        {
            "check": "klass_gated",
            "ok": klass_ok,
            "note": "record is gated-class" if klass_ok else "record is not gated",
        },
        {
            "check": "reversible",
            "ok": False,
            "note": "no sanctioned rollback path for code/structural changes",
        },
        {
            "check": "protected_surface",
            "ok": surface_ok,
            "note": "target clear"
            if surface_ok
            else f"target under a protected surface: {p.get('target')!r}",
        },
        {
            "check": "bounded",
            "ok": bounded_ok,
            "note": "action within bounds"
            if bounded_ok
            else (
                "nested payload exceeds 2KiB"
                if total_ok
                else "action missing/oversized (>4KiB)"
            ),
        },
        {
            "check": "tests_verifiable",
            "ok": False,
            "note": "requires human-run verification (contract part 1)",
        },
    ]


def review_gated(p: dict, cfg) -> dict:
    checks = _checks(p)
    by = {c["check"]: c["ok"] for c in checks}
    recommendation = (
        "reject"
        if not by["klass_gated"] or not by["protected_surface"]
        else "needs_human"
    )
    return {
        "proposal_id": p.get("proposal_id"),
        "dedupe_key": p.get("dedupe_key"),
        "kind": p.get("kind"),
        "target": p.get("target"),
        "ts": p.get("ts"),
        "charter_version": charter.CHARTER_VERSION,
        "checks": checks,
        "recommendation": recommendation,
        "reviewed_at": time.time(),
    }
