"""Route a validated intent to a reversible faculty surface.

Pure translation layer between Task 12's intent taxonomy and the sanctioned
apply surface (imperator/apply.py, imperator/proposals.py): a validated
teaching intent becomes a *pending* proposal awaiting user confirmation
(route), a confirmed pending becomes an applied record (apply_confirmed),
and an applied record can be inverted back to its prior state (build_inverse,
apply_undo). No direct Redis access -- all durable writes go through
apply_proposal's sanctioned paths.
"""

from __future__ import annotations

import math
from typing import Any, cast

from imperator import apply as A, proposals as P

_HEAVY_KINDS = {"tune_rule"}

# Numeric bounds mirrored from imperator/apply.py and imperator/dialogue/
# intents.py -- the sources of truth for what the apply layer will actually
# accept. A rollback anchor recorded before those bounds tightened (e.g. a
# historical prior_sigma from a wider [sigma_min, sigma_max] window) would
# otherwise just fail closed inside apply_proposal with the same generic
# "logged" status as any other rejected proposal; apply_undo() checks the
# same bounds up front so it can report a truthful, distinct reason instead.
_SIGMA_MIN, _SIGMA_MAX = 1.5, 5.0
_FLOOR_MIN, _FLOOR_MAX = 0.0, 0.6


def _arm_for_silence(ctx, state_key: str) -> dict:
    """Pick the gate_calibration op that reverses whichever arm most recently
    suppressed this state_key, per the suppression record's ``arm`` field.

    habituation -> reset the habituation floor to 0 (speak up again).
    central_tolerance -> drop the permanent self-tolerance dismissal (the only
    deciding_arm value limen/gate.py ever emits for that suppression path).
    Anything else -- a different arm, or no matching suppression record at
    all -- falls back to self_tolerance_remove, the safe universal "make this
    fireable again" default.
    """
    for s in ctx.recent_suppressions:
        if s.get("state_key") == state_key:
            arm = s.get("arm")
            if arm == "habituation":
                return {"op": "floor_set", "state_key": state_key, "value": 0.0}
            if arm == "central_tolerance":
                return {"op": "self_tolerance_remove", "state_key": state_key}
            break
    return {"op": "self_tolerance_remove", "state_key": state_key}


def route(intent: dict, ctx, *, pm, cfg) -> dict:
    """Translate a validated teaching intent into a pending proposal awaiting
    confirmation. Returns {proposal, tier, echo, confirm_phrase, inverse}.

    ``pm``/``cfg`` are accepted for interface symmetry with the rest of the
    confirm/apply/undo surface; this deterministic mapping needs neither --
    all state comes from ``ctx`` (recent_suppressions).
    """
    kind = intent["kind"]
    # Cast (not assert): _REQUIRE_TARGET-gated kinds always carry a target by
    # the time a validated intent reaches route() (Task 12's validate_intent);
    # kinds that don't require one (query/undo) never dispatch past the
    # `else: raise ValueError` branch below, so they never read this value.
    target = cast(str, intent.get("target"))
    action = intent.get("action", {})
    rationale = intent.get("rationale", "")
    tier = "heavy" if kind in _HEAVY_KINDS else "light"
    confirm_phrase = None

    if kind == "correct_silence":
        p = P.make_proposal(
            kind="gate_calibration",
            target=target,
            action=_arm_for_silence(ctx, target),
            rationale=rationale,
            source="dialogue",
        )
        echo = f"I'll speak up for {target} next time."
    elif kind == "correct_noise":
        p = P.make_proposal(
            kind="gate_calibration",
            target=target,
            action={"op": "self_tolerance_add", "state_key": target},
            rationale=rationale,
            source="dialogue",
        )
        echo = f"I'll stop flagging {target}."
    elif kind == "tune_rule":
        p = P.make_proposal(
            kind="escalation_rule",
            target=target,
            action={"target": action.get("target")},
            rationale=rationale,
            source="dialogue",
        )
        echo = f"I'll set rule {target} → {action.get('target')}."
        confirm_phrase = "change the matrix"
    elif kind == "teach_context_directive":
        p = P.make_proposal(
            kind="context_directive",
            target=target,
            action=action,
            rationale=rationale,
            source="dialogue",
        )
        echo = f"I'll stay quiet when {target} applies."
    elif kind == "teach_semantic_fact":
        pattern = action.get("pattern") or {
            "kind": "semantic",
            "domains": action.get("domains", []),
            "rule_key": action.get("rule_key"),
            "severity": action.get("severity", "LOW"),
        }
        p = P.make_proposal(
            kind="semantic_fact",
            target=target,
            action={"pattern": pattern},
            rationale=rationale,
            source="dialogue",
        )
        echo = f"I'll remember: {rationale}."
    elif kind == "correct_advice_quality":
        # light: credibility down (gate_calibration); heavy variant (prompt
        # rewrite) routed only when the intent explicitly asks for a rewrite.
        if action.get("rewrite"):
            tier, confirm_phrase = "heavy", "rewrite the prompt"
            p = P.make_proposal(
                kind="prompt_strategy",
                target=target,
                action=action,
                rationale=rationale,
                source="dialogue",
            )
            echo = f"I'll rewrite the {target} prompt."
        else:
            p = P.make_proposal(
                kind="gate_calibration",
                target=target,
                action={"op": "self_tolerance_add", "state_key": target},
                rationale=rationale,
                source="dialogue",
            )
            echo = f"I'll trust advice on {target} less."
    else:
        raise ValueError(f"router cannot route kind {kind}")

    P.normalize_klass(p)
    return {
        "proposal": p,
        "tier": tier,
        "echo": echo,
        "confirm_phrase": confirm_phrase,
        "inverse": None,
    }


def apply_confirmed(pending: dict, *, pm, cfg, session_id: str) -> dict:
    """Apply a user-confirmed pending proposal via the sanctioned confirmed
    path (apply.apply_proposal(..., confirmed=True)) and report a small
    result shaped for the dialogue reply."""
    p = A.apply_proposal(
        pm, pending["proposal"], cfg=cfg, session_id=session_id, confirmed=True
    )
    return {"proposal": p, "status": p["status"], "echo": pending["echo"]}


def _undo_proposal(kind: str, target: str, action: dict) -> dict:
    """Build a normalized undo proposal. Every build_inverse branch below
    varies kind/target/action, but rationale="undo" and source="dialogue"
    are constant across all of them."""
    return P.normalize_klass(
        P.make_proposal(
            kind=kind,
            target=target,
            action=action,
            rationale="undo",
            source="dialogue",
        )
    )


def build_inverse(applied: dict) -> dict | None:
    """Construct an inverse proposal from an applied record's rollback anchor.

    None means "not (safely) invertible": either the apply left no real prior
    to restore (e.g. self_tolerance_add on an already-tolerant target, or a
    brand-new escalation rule/sigma domain with no prior value at all -- the
    apply layer has no "remove" op for those, only "set"), or the kind
    carries no rollback anchor to begin with.
    """
    p = applied["proposal"]
    a = p.get("action") or {}
    kind = p["kind"]
    if kind == "escalation_rule" and a.get("prior_target") is not None:
        return _undo_proposal(
            "escalation_rule", p["target"], {"target": a["prior_target"]}
        )
    if kind == "sigma" and a.get("prior_sigma") is not None:
        return _undo_proposal(
            "sigma",
            p["target"],
            {"domain": a.get("domain", p["target"]), "sigma": a["prior_sigma"]},
        )
    if kind == "gate_calibration":
        op = cast(str, a.get("op"))
        if op == "floor_set" and "prior" in a:
            prior = a.get("prior") or {}
            return _undo_proposal(
                "gate_calibration",
                p["target"],
                {
                    "op": "floor_set",
                    "state_key": a.get("state_key", p["target"]),
                    "value": prior.get("floor", 0.0),
                },
            )
        inv_op = {
            "self_tolerance_add": "self_tolerance_remove",
            "self_tolerance_remove": "self_tolerance_add",
        }.get(op)
        # Only invert if membership actually changed: self_tolerance_add
        # changed the state iff it was NOT already a member beforehand;
        # self_tolerance_remove changed it iff it WAS a member beforehand.
        changed = "prior" in a and (op == "self_tolerance_add") != bool(a["prior"])
        if inv_op and changed:
            return _undo_proposal(
                "gate_calibration",
                p["target"],
                {"op": inv_op, "state_key": a.get("state_key", p["target"])},
            )
    if kind == "prompt_strategy" and "prior_text" in a:
        return _undo_proposal(
            "prompt_strategy",
            p["target"],
            {"domain": a.get("domain", p["target"]), "text": a["prior_text"]},
        )
    if kind == "context_directive":
        # Complete restore semantics (Task 20 decision A), designed once and
        # mirrored by the semantic_fact case below: a create/upsert apply
        # records the FULL pre-write content as prior_directive (None for a
        # brand-new id); an applied remove records the FULL pre-delete
        # content the same way. Whichever direction has a real prior
        # restores it; the "no prior" case falls back to the simpler
        # create-inverts-to-remove / remove-with-nothing-to-restore shape.
        prior_directive = a.get("prior_directive")
        if a.get("op") == "remove":
            if prior_directive is None:
                # No prior recorded (id never existed when removed): no
                # inverse -- undo must report unavailable rather than
                # re-add fabricated content and claim a false success.
                return None
            return _undo_proposal(
                "context_directive",
                p["target"],
                {
                    "directive_id": prior_directive.get(
                        "directive_id", a.get("directive_id")
                    ),
                    "predicate": prior_directive.get("predicate", {}),
                    "action": prior_directive.get("action", "suppress"),
                    "scope": prior_directive.get("scope", "all"),
                    # The directive's OWN rationale, not this undo's audit
                    # rationale -- a re-add must not lose the original
                    # user-facing explanation.
                    "rationale": prior_directive.get("rationale", ""),
                },
            )
        if not a.get("directive_id"):
            # No directive_id anchor means the write never stored anything
            # (e.g. refused at cap): no inverse -- undo must report
            # unavailable rather than hdel/re-add a never-stored id and
            # claim a false success.
            return None
        if prior_directive is not None:
            # Upsert-with-prior: inverse RESTORES the prior content.
            return _undo_proposal(
                "context_directive",
                p["target"],
                {
                    "directive_id": a["directive_id"],
                    "predicate": prior_directive.get("predicate", {}),
                    "action": prior_directive.get("action", "suppress"),
                    "scope": prior_directive.get("scope", "all"),
                    "rationale": prior_directive.get("rationale", ""),
                },
            )
        # Create-with-no-prior: inverse is removal (current behavior).
        return _undo_proposal(
            "context_directive",
            p["target"],
            {"op": "remove", "directive_id": a["directive_id"]},
        )
    if kind == "semantic_fact":
        # Mirrors the context_directive case above (Task 20 decision A):
        # prior_fact is the FULL memory state read before the write, on
        # BOTH the create/upsert and remove sides.
        prior_fact = a.get("prior_fact")
        if a.get("op") == "remove":
            if prior_fact is None:
                # No prior recorded (id never existed when removed): no
                # inverse -- undo must report unavailable rather than
                # re-teach fabricated content and claim a false success.
                return None
            return _undo_proposal(
                "semantic_fact", p["target"], {"pattern": prior_fact.get("pattern")}
            )
        if not a.get("memory_id"):
            return None
        if prior_fact is not None:
            # Upsert-with-prior (re-teach): inverse RESTORES the prior
            # content by re-teaching it -- itself a review, per decision B:
            # even an undo is forward decay, never a raw state rollback.
            return _undo_proposal(
                "semantic_fact", p["target"], {"pattern": prior_fact.get("pattern")}
            )
        # Create-with-no-prior: inverse is removal (current behavior).
        return _undo_proposal(
            "semantic_fact",
            p["target"],
            {"op": "remove", "memory_id": a["memory_id"]},
        )
    return None


def _inverse_out_of_bounds(inv: dict, cfg) -> bool:
    """True iff applying ``inv`` would try to restore a numeric that violates
    the CURRENT bounds imperator/apply.py enforces -- i.e. a rollback anchor
    recorded before those bounds tightened. Checked up front so the caller
    can report a truthful, distinct reason instead of letting the apply fail
    generically."""
    a = inv.get("action") or {}
    if inv["kind"] == "sigma":
        try:
            sigma = float(cast(Any, a.get("sigma")))
        except (TypeError, ValueError):
            return True
        lo = getattr(cfg, "sigma_min", _SIGMA_MIN)
        hi = getattr(cfg, "sigma_max", _SIGMA_MAX)
        return not (math.isfinite(sigma) and lo <= sigma <= hi)
    if inv["kind"] == "gate_calibration" and a.get("op") == "floor_set":
        try:
            value = float(cast(Any, a.get("value")))
        except (TypeError, ValueError):
            return True
        return not (math.isfinite(value) and _FLOOR_MIN <= value <= _FLOOR_MAX)
    return False


def apply_undo(applied: dict, *, pm, cfg, session_id: str) -> dict:
    """Build and apply the inverse of an applied dialogue proposal, via the
    same sanctioned confirmed path as apply_confirmed. Returns
    {proposal, status, reason} with a truthful, distinct reason for every
    non-applied outcome:

    - status="unavailable": build_inverse found no safe inverse to construct.
    - status="blocked": an inverse exists but would restore a value the
      CURRENT apply-layer bounds reject (e.g. a prior_sigma recorded before
      the configured [sigma_min, sigma_max] window narrowed) -- reported
      before even attempting the apply, distinct from a generic apply
      failure.
    - status="logged": the inverse was attempted but apply_proposal did not
      report "applied" (dialogue_confirmed_apply_enabled off, non-safe
      klass, kind outside _CONFIRMED_APPLY_KINDS, or an internal apply
      error) -- apply.py's own generic terminal status, surfaced as-is.
    - status="applied": the undo succeeded.
    """
    inv = build_inverse(applied)
    if inv is None:
        return {
            "proposal": None,
            "status": "unavailable",
            "reason": "cannot undo: no inverse available for this change",
        }
    if _inverse_out_of_bounds(inv, cfg):
        return {
            "proposal": inv,
            "status": "blocked",
            "reason": "cannot restore: prior value outside current bounds",
        }
    out = A.apply_proposal(pm, inv, cfg=cfg, session_id=session_id, confirmed=True)
    reason = None if out["status"] == "applied" else "could not apply undo"
    return {"proposal": out, "status": out["status"], "reason": reason}
