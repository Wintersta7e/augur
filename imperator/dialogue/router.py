"""Route a validated intent to a reversible faculty surface.

Pure translation layer between the intents.py taxonomy (spec §7.1) and the
sanctioned apply surface (imperator/apply.py, imperator/proposals.py): a validated
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


def _directive_id_from_reason(reason: Any) -> str | None:
    """Parse the directive id out of a taught-directive suppression's
    ``"taught_directive:<id>"`` reason string. Returns None on anything that
    doesn't match -- absent reason, wrong prefix, or an empty id -- so the
    caller can reject with a clarification instead of building a removal
    proposal that targets nothing real."""
    prefix = "taught_directive:"
    if not isinstance(reason, str) or not reason.startswith(prefix):
        return None
    directive_id = reason[len(prefix) :]
    return directive_id or None


def _arm_for_silence(ctx, state_key: str) -> tuple[str, dict]:
    """Pick the (proposal kind, action) pair that reverses whichever arm most
    recently suppressed this state_key, per the suppression record's ``arm``
    field.

    habituation -> reset the habituation floor to 0 (speak up again);
    gate_calibration floor_set.
    central_tolerance -> drop the permanent self-tolerance dismissal;
    gate_calibration self_tolerance_remove.
    taught_directive -> remove the offending taught directive (spec §7.2/§9):
    limen/gate.py's directive pre-check is the ONLY source of this arm value,
    and it always stamps the reason "taught_directive:<id>", so the id is
    parsed straight from the suppression record rather than re-derived;
    context_directive op="remove".
    Anything else -- a different arm, or no matching suppression record at
    all -- falls back to gate_calibration self_tolerance_remove, the safe
    universal "make this fireable again" default.

    Raises ValueError when a taught_directive suppression's reason carries no
    parseable directive id: correct_silence must reject with a clarification
    rather than build a proposal that removes nothing real.
    """
    for s in ctx.recent_suppressions:
        if s.get("state_key") == state_key:
            arm = s.get("arm")
            if arm == "habituation":
                return "gate_calibration", {
                    "op": "floor_set",
                    "state_key": state_key,
                    "value": 0.0,
                }
            if arm == "central_tolerance":
                return "gate_calibration", {
                    "op": "self_tolerance_remove",
                    "state_key": state_key,
                }
            if arm == "taught_directive":
                directive_id = _directive_id_from_reason(s.get("reason"))
                if not directive_id:
                    raise ValueError(
                        f"can't tell which taught rule silenced {state_key}"
                    )
                return "context_directive", {
                    "op": "remove",
                    "directive_id": directive_id,
                }
            break
    return "gate_calibration", {"op": "self_tolerance_remove", "state_key": state_key}


def _normalize_scope(scope: Any) -> Any:
    """Canonicalize a taught directive's ``scope`` to ``"all"`` or a cleaned
    list of domain strings (spec §7.2), so a malformed value can never reach
    the gate and silently widen suppression to every domain.

    - a list -> its string, non-empty members (an all-empty list becomes "all")
    - "all" / None / absent -> "all" (silence everything in the app)
    - a bare domain string ("typing") -> ["typing"], honoring the narrow intent
      the user expressed rather than widening it to every domain
    - anything else (dict, number, bool) -> "all"; the gate's fail-closed
      _directive_scope_allows is the backstop for anything unexpected.
    """
    if isinstance(scope, list):
        cleaned = [d for d in scope if isinstance(d, str) and d.strip()]
        return cleaned or "all"
    if isinstance(scope, str) and scope != "all" and scope.strip():
        return [scope.strip()]
    return "all"


def route(intent: dict, ctx, *, pm, cfg) -> dict:
    """Translate a validated teaching intent into a pending proposal awaiting
    confirmation. Returns {proposal, tier, echo, confirm_phrase, inverse}.

    ``pm``/``cfg`` are accepted for interface symmetry with the rest of the
    confirm/apply/undo surface; this deterministic mapping needs neither --
    all state comes from ``ctx`` (recent_suppressions).
    """
    kind = intent["kind"]
    # Cast (not assert): _REQUIRE_TARGET-gated kinds always carry a target by
    # the time a validated intent reaches route() (intents.validate_intent
    # enforces it); kinds that don't require one (query/undo) never dispatch
    # past the `else: raise ValueError` branch below, so they never read this
    # value.
    target = cast(str, intent.get("target"))
    action = intent.get("action", {})
    rationale = intent.get("rationale", "")
    tier = "heavy" if kind in _HEAVY_KINDS else "light"
    confirm_phrase = None

    if kind == "correct_silence":
        proposal_kind, silence_action = _arm_for_silence(ctx, target)
        p = P.make_proposal(
            kind=proposal_kind,
            target=target,
            action=silence_action,
            rationale=rationale,
            source="dialogue",
        )
        echo = (
            f"I'll remove the rule that's keeping me quiet about {target}."
            if proposal_kind == "context_directive"
            else f"I'll speak up for {target} next time."
        )
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
        # Server-authoritative predicate (spec §7.2): the gate matches a
        # directive only when predicate.match == the LIVE focused app, so match
        # is filled from ctx.focused_app (the same load_focused_app source the
        # gate reads), NOT trusted from the LLM -- the model is never given the
        # app string, so a guessed predicate would never match. No current
        # focused app -> reject truthfully rather than store a directive that
        # can never fire while replying "Done — applied" (invariant 7).
        focused = getattr(ctx, "focused_app", None)
        if not focused:
            raise ValueError("I can't tell which app you're in right now")
        directive_action = action.get("action", "suppress")
        if directive_action not in ("suppress", "downgrade"):
            raise ValueError(
                "a context directive must suppress or downgrade, "
                f"not {directive_action!r}"
            )
        scope = _normalize_scope(action.get("scope"))
        p = P.make_proposal(
            kind="context_directive",
            target=focused,
            action={
                "predicate": {"context": "focused_app", "match": focused},
                "action": directive_action,
                "scope": scope,
            },
            rationale=rationale,
            source="dialogue",
        )
        verb = "stay quiet" if directive_action == "suppress" else "speak more softly"
        where = "" if scope == "all" else f" about {', '.join(scope)}"
        echo = f"I'll {verb}{where} while you're in {focused}."
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
        # Complete restore semantics (mirrors apply.py's rollback-anchor
        # discipline, spec §8), designed once and mirrored by the
        # semantic_fact case below: a create/upsert apply
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
        # Mirrors the context_directive case above (same rollback-anchor
        # discipline, spec §8): prior_fact is the FULL memory state read
        # before the write, on
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
