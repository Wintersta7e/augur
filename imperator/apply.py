"""Sanctioned, reversible, gated application of safe proposals. Honors apply_enabled."""

from __future__ import annotations

import logging
import math
import time
import uuid as _uuid

from consilium.prompt_safety import is_prompt_acceptable
from conscientia import screens
from conscientia.recording import record_violation_best_effort
from memoria.fsrs import make_memory_id
from nexus import matrix_ops
from imperator import proposals as P

log = logging.getLogger("imperator.apply")

# Habituation-floor bounds, mirroring Disciplina's floor sweep
# [0.0, GATE_FLOOR_MAX=0.6] (disciplina/reflection_engine.py — not imported:
# that module pulls in httpx/nats and mutates sys.path at import time).
_FLOOR_MIN, _FLOOR_MAX = 0.0, 0.6

# Valid values for a context directive's inner "action" field (what the
# directive does when its predicate matches -- distinct from the outer
# proposal's p["action"] dict, which carries this among other keys).
_DIRECTIVE_ACTIONS = ("suppress", "downgrade")


def _arm_gate(pm, p: dict, *, cfg, ctx) -> bool:
    """Write the durable applied-marker that arms the one-move-per-(kind,target)
    anti-thrash gate. Returns True if armed, False if the write failed.

    ``ctx`` is REQUIRED, not defaulted: the marker is a learned write, so a
    caller that forgets it makes this fail closed under ENFORCE — and because
    the failure is caught below, the whole armed-apply path would die behind a
    single log line. A missing keyword must break loudly at the call instead.

    The marker is the ONLY thing closing the gate, so it is written BEFORE the
    primary (matrix/prompt) write: a marker failure must abort the apply rather
    than leave a committed change behind an open gate, where a DIFFERENT-text
    proposal for the same target could re-apply in-window and bury the rollback
    anchor. There is no retry path here — a failed arm means the apply does not
    happen, and the unchanged proposal is reconsidered on a later cycle.
    """
    try:
        pm.mark_proposal_applied(
            p["dedupe_key"],
            ttl_s=int(cfg.imperator_ii_dedupe_staleness_s),
            ctx=ctx,
        )
        return True
    except Exception:
        log.warning(
            "imperator apply: idempotency marker write failed for %s; failing "
            "closed (not applied) so the anti-thrash gate is never left open for "
            "a different action on the same target",
            p["dedupe_key"],
            exc_info=True,
        )
        return False


def _conscientia_refuses(pm, p: dict, *, cfg, session_id: str | None = None) -> bool:
    """S4 pre-apply value screen (spec D6: fail-CLOSED — for self-modification
    the safe direction is not applying). Returns True when the apply must be
    refused; records the violation best-effort.
    """
    try:
        v = screens.screen_proposal(p, cfg)
    except Exception:
        log.warning(
            "conscientia proposal screen failed for %s/%s; refusing apply "
            "(fail-closed)",
            p.get("kind"),
            p.get("target"),
            exc_info=True,
        )
        return True
    if v.ok:
        return False
    record_violation_best_effort(
        pm,
        screens.make_violation(
            "apply",
            v.code or "refused",
            v.detail or "",
            v.principle or "",
            state_key=f"{p.get('kind')}:{p.get('target')}",
            session_id=session_id,
        ),
    )
    log.info(
        "conscientia refused apply for %s/%s: %s",
        p.get("kind"),
        p.get("target"),
        v.detail,
    )
    return True


def _escalation_patch_error(p: dict, *, cfg) -> str | None:
    """Validate the isolated escalation patch BEFORE the anti-thrash gate arms
    (validate -> arm -> write, the same contract every other handler follows).

    matrix_ops commits nothing on a validation error (it unwatches before
    MULTI), so there is no committed-change-behind-an-open-gate risk to
    justify arming first; arming on a patch that can never land would poison
    the (kind, target) dedupe slot for the staleness window and block the
    corrected retry (spec: only *applied* entries block).

    Returns None when the patch is well-formed; the validator error otherwise.
    """
    action = p.get("action") or {}
    if "window" in action:
        return matrix_ops.validate_patch(
            rule_windows={p["target"]: action["window"]}, config=cfg
        )
    return matrix_ops.validate_patch(rules={p["target"]: action.get("target")})


def _apply_escalation_rule(pm, p: dict, *, ctx=None) -> bool:
    """Apply an escalation_rule proposal by updating the matrix and recording the
    rollback anchor. Returns True on success, False on validation error or write failure.

    The rollback anchor is recorded in p["action"]["prior_target"] or
    p["action"]["prior_window"], read from the committed CAS snapshot (not a separate read).
    """
    action = p.get("action") or {}
    if "window" in action:
        res = matrix_ops.apply_matrix_update(
            pm, rule_windows={p["target"]: action["window"]}, mode="patch", ctx=ctx
        )
    else:
        res = matrix_ops.apply_matrix_update(
            pm, rules={p["target"]: action.get("target")}, mode="patch", ctx=ctx
        )
    if "error" in res:
        return False
    if "window" in action:
        action["prior_window"] = (res.get("prior_rule_windows") or {}).get(p["target"])
    else:
        action["prior_target"] = (res.get("prior_rules") or {}).get(p["target"])
    return True


def _apply_prompt_strategy(pm, p: dict, *, cfg, ctx=None) -> bool:
    """Apply a prompt_strategy proposal: validate, arm the gate, save the new
    prompt text, and record the rollback anchor. Returns True on success, False
    if validation fails, no current prompt exists, or the gate cannot be armed.

    Self-validating with a SINGLE load_prompt read shared by the precondition
    check and the rollback anchor (a second read would open a TOCTOU window
    where the anchor comes from a value that was never validated). Ordering is
    validate -> arm -> save -> record anchor: a validation failure must not arm
    the gate (a corrected proposal for the same target can still apply
    in-window), and a marker failure aborts BEFORE save_prompt runs, so the
    prior text is never archived — the rollback anchor stays intact and no
    different-text proposal for this target can re-apply in-window off an
    unarmed gate.

    The rollback anchor p["action"]["prior_text"] is recorded only AFTER the
    (possible) save_prompt returns without raising: if the write raises,
    apply_proposal's except-Exception wrapper logs the proposal (status
    "logged", not "applied") with NO anchor, so a follow-on "undo that" cannot
    invert a rewrite that never persisted and reply a false "Reversed."
    (invariant 7). Idempotent: only re-saves (save_prompt archives the prior
    into rollback history) when the text actually changes — a re-apply of
    identical text must NOT re-archive and corrupt the rollback anchor.
    """
    action = p.get("action") or {}
    domain, text = action.get("domain", p["target"]), action.get("text", "")
    current = pm.load_prompt(domain)
    if not is_prompt_acceptable(text, cfg) or current is None:
        return False
    if not _arm_gate(pm, p, cfg=cfg, ctx=ctx):
        return False
    if current != text:
        pm.save_prompt(domain, text, ctx=ctx)
    action["prior_text"] = current  # rollback anchor, after the write lands
    return True


def _apply_sigma(pm, p: dict, *, cfg, ctx=None) -> bool:
    """Apply a sigma proposal: update the domain's detection sensitivity and
    record the rollback anchor. Returns True on success, False if validation
    fails or the gate cannot be armed.

    Writes the "sigma_threshold" key — the key vigil's anomaly gating actually
    reads (DEFAULT_THRESHOLDS + the deviation comparison) and Disciplina's
    precision pass tunes. Other stored threshold fields are preserved.

    Self-validating and SELF-ARMING (this handler arms; the caller must not):
    ordering is validate -> arm -> write, so a validation failure (missing,
    non-numeric, non-finite, or out-of-range sigma) never arms the anti-thrash
    gate — fail closed, no write. A NaN sigma would make vigil's
    `deviation >= sigma_threshold` comparison silently always-False (disabling
    detection for the domain); range bounds mirror Disciplina's own tuning
    clamps [cfg.sigma_min, cfg.sigma_max].

    The rollback anchor is recorded in p["action"]["prior_sigma"] (the true
    prior sigma_threshold value, None if the domain had none stored).
    """
    a = p.get("action") or {}
    domain = a.get("domain", p["target"])
    try:
        sigma = float(a["sigma"])
    except (KeyError, TypeError, ValueError):
        return False
    sigma_min = getattr(cfg, "sigma_min", 1.5)
    sigma_max = getattr(cfg, "sigma_max", 5.0)
    if not math.isfinite(sigma) or not (sigma_min <= sigma <= sigma_max):
        return False
    current = pm.load_thresholds(domain) or {}
    if not _arm_gate(pm, p, cfg=cfg, ctx=ctx):
        return False
    a["prior_sigma"] = current.get("sigma_threshold")
    pm.save_thresholds(domain, {**current, "sigma_threshold": sigma}, ctx=ctx)
    return True


def _apply_gate_calibration(pm, p: dict, *, cfg, ctx=None) -> bool:
    """Apply a gate_calibration proposal: self_tolerance_add/remove or floor_set.
    Returns True on success, False if validation fails or the gate cannot be armed.

    Handles three ops:
    - self_tolerance_add: adds state_key to self-tolerance set, records prior membership
    - self_tolerance_remove: removes state_key from self-tolerance set, records prior membership
    - floor_set: updates the habituation_floor entry ({"floor": float, "last_ts":
      time.time()}, merged over the prior entry; other signatures' entries are
      untouched — per-field HSET), records the prior entry

    Self-validating and SELF-ARMING (this handler arms; the caller must not):
    ordering is validate -> arm -> write -> record anchor, so a validation
    failure (unknown op; missing, non-numeric, non-finite, or out-of-range floor
    value) never arms the anti-thrash gate — fail closed, no write. The prior
    state (membership bool or prior floor entry) is recorded in p["action"]
    ["prior"] as the rollback anchor ONLY after the write returns without
    raising: a write that raises ends the proposal "logged" with no anchor, so a
    follow-on "undo that" cannot invert a change that never landed (invariant 7).
    All Redis writes go through PersistenceManager.
    """
    a = p.get("action") or {}
    op, sk = a.get("op"), a.get("state_key", p["target"])
    if op in ("self_tolerance_add", "self_tolerance_remove"):
        prior = pm.is_self_tolerant(sk)
        if not _arm_gate(pm, p, cfg=cfg, ctx=ctx):
            return False
        if op == "self_tolerance_add":
            pm.add_self_tolerance(sk, ctx=ctx)
        else:
            pm.remove_self_tolerance(sk, ctx=ctx)
        a["prior"] = prior  # anchor after the write lands (invariant 7)
        return True
    if op == "floor_set":
        # Reject bools before conversion: float(False) == 0.0 is in-range and
        # would otherwise slip through (mirrors the intent-layer check).
        if isinstance(a.get("value"), bool):
            return False
        try:
            value = float(a["value"])
        except (KeyError, TypeError, ValueError):
            return False
        # Bounds mirror Disciplina's floor sweep [0.0, GATE_FLOOR_MAX=0.6]
        # (disciplina/reflection_engine.py); a floor > 1.0 produces a negative
        # habituation cap downstream, and NaN poisons the gate arithmetic.
        if not math.isfinite(value) or not (_FLOOR_MIN <= value <= _FLOOR_MAX):
            return False
        prior_entry = pm.load_habituation_floor(sk) or {}
        if not _arm_gate(pm, p, cfg=cfg, ctx=ctx):
            return False
        new_entry = {**prior_entry, "floor": value, "last_ts": time.time()}
        pm.save_gate_tuning_state(floors={sk: new_entry}, ctx=ctx)
        a["prior"] = prior_entry  # anchor after the write lands (invariant 7)
        return True
    return False


def apply_proposal(
    pm, p: dict, *, cfg, session_id: str | None, confirmed: bool = False
) -> dict:
    """Apply p iff auto-applicable AND apply enabled AND not already applied AND a
    reversibility anchor exists; else leave 'logged' ('skipped' if already applied).
    Sets terminal status. Never raises on apply failure.

    confirmed=True routes to the human-confirmed dialogue path (_apply_confirmed),
    gated independently by cfg.dialogue_confirmed_apply_enabled rather than
    cfg.imperator_ii_apply_enabled -- the two flags are orthogonal."""
    if confirmed:
        return _apply_confirmed(pm, p, cfg=cfg, session_id=session_id)
    if not getattr(
        cfg, "imperator_ii_apply_enabled", False
    ) or not P.is_auto_applicable(p):
        p["status"] = "logged"
        return p
    # Idempotency enforced HERE, not only in run_cycle: a direct call or a second
    # cycle must never re-apply a change already committed within the staleness window.
    if pm.is_proposal_applied(p["dedupe_key"]):
        p["status"] = "skipped"
        return p
    if _conscientia_refuses(pm, p, cfg=cfg, session_id=session_id):
        p["status"] = "logged"
        return p
    learn_ctx = pm.resolve_learn_context(session_id)
    try:
        if p["kind"] == "escalation_rule":
            # Validate the isolated patch FIRST: a malformed patch commits
            # nothing, so it must not consume the anti-thrash gate either
            # (a corrected retry for the same target can still apply
            # in-window — mirrors _apply_prompt_strategy's contract).
            err = _escalation_patch_error(p, cfg=cfg)
            if err:
                log.info(
                    "imperator apply: escalation patch for %s refused before "
                    "arming: %s",
                    p["target"],
                    err,
                )
                p["status"] = "logged"
                return p
            # Arm the anti-thrash gate before the committing matrix write so the
            # write can never land behind an open gate (a failed arm aborts here).
            if not _arm_gate(pm, p, cfg=cfg, ctx=learn_ctx):
                p["status"] = "logged"
                return p
            if not _apply_escalation_rule(pm, p, ctx=learn_ctx):
                p["status"] = "logged"
                return p
        elif p["kind"] == "prompt_strategy":
            # Self-validating helper: validate -> arm gate -> save, with a single
            # load_prompt read shared by the check and the rollback anchor.
            if not _apply_prompt_strategy(pm, p, cfg=cfg, ctx=learn_ctx):
                p["status"] = "logged"
                return p
        else:
            p["status"] = "logged"
            return p
    except Exception:
        p["status"] = "logged"
        return p
    p["status"] = "applied"
    p["applied_session"] = session_id
    return p


def _apply_context_directive(pm, p: dict, *, cfg, ctx=None) -> bool:
    """Apply a context_directive proposal: validate, arm the gate, write, and
    record the rollback anchor. Returns True on success, False if validation
    fails, the gate cannot be armed, or the store refuses a NEW id at cap.

    Handles two operations:
    - op="remove": validates directive_id (False if missing/empty, no arm),
      arms the gate, then removes the directive. Removal arms like every other
      dispatched apply (self_tolerance_remove arms too). The rollback anchor
      p["action"]["prior_directive"] records the FULL directive content read
      BEFORE the delete (None if the id never existed), so a follow-on undo
      can re-add the exact removed content instead of reporting "unavailable"
      whenever a real prior exists (spec §8's rollback-anchor table).
    - create/upsert (default): validates the directive's inner "action" field
      against the suppress|downgrade enum (False if neither, no arm -- fail
      closed like every other enum/range check in this module), mints a
      directive_id (uuid4 hex) if not provided, arms the gate, then stores via
      add_dialogue_directive and propagates its bool: True = written (upserts
      of existing ids succeed even at cap), False = NEW id refused at
      MAX_DIALOGUE_DIRECTIVES, nothing stored. An at-cap refusal is a truthful
      failure -- the proposal ends "logged" (armed-but-refused is acceptable,
      like the arm-then-save-raises path in _apply_prompt_strategy). The
      rollback anchor p["action"]["prior_directive"] records the FULL
      pre-existing content read BEFORE the write (None for a brand-new
      directive_id), recorded ONLY when the write actually stored the
      directive: an anchor on a refused write would let a follow-on undo
      hdel/re-add off a never-stored id and report a false success.

    Self-validating and SELF-ARMING (validate -> arm -> write), so a validation
    failure never arms the gate.
    """
    a = p.get("action") or {}
    if a.get("op") == "remove":
        # Removal path: validate directive_id -> read prior -> arm -> remove
        directive_id = a.get("directive_id")
        if not directive_id:
            return False
        prior = pm.get_dialogue_directive(directive_id)  # rollback anchor, pre-delete
        if not _arm_gate(pm, p, cfg=cfg, ctx=ctx):
            return False
        pm.remove_dialogue_directive(directive_id, ctx=ctx)
        a["prior_directive"] = prior
        return True
    # Creation path: validate the directive action enum -> mint directive_id if
    # not provided -> read prior -> arm -> write
    directive_action = a.get("action", "suppress")
    if directive_action not in _DIRECTIVE_ACTIONS:
        return False
    directive_id = a.get("directive_id") or _uuid.uuid4().hex
    directive = {
        "directive_id": directive_id,
        "predicate": a.get("predicate") or {},
        "action": directive_action,
        "scope": a.get("scope", "all"),
        # A restore inverse carries the ORIGINAL directive's own rationale in
        # the action payload (so a re-add doesn't overwrite it with the
        # outer proposal's generic "undo" audit rationale); the normal teach
        # path never sets a["rationale"], so it falls back to the proposal's.
        "rationale": a.get("rationale", p.get("rationale", "")),
    }
    prior = pm.get_dialogue_directive(directive_id)  # rollback anchor, pre-write
    if not _arm_gate(pm, p, cfg=cfg, ctx=ctx):
        return False
    # Propagate the write's bool (True = written, False = NEW id refused at cap);
    # record the rollback anchor only when the directive was actually stored.
    result = pm.add_dialogue_directive(directive, ctx=ctx)
    if result:
        a["directive_id"] = directive_id
        a["prior_directive"] = prior
    return result


def _apply_semantic_fact(pm, p: dict, *, cfg, session_id: str | None) -> bool:
    """Apply a semantic_fact proposal: teach (create/re-teach) a taught memory,
    or archive one for undo. Returns True on success, False if validation
    fails or the gate cannot be armed.

    Handles two operations:
    - op="remove": validates memory_id (False if missing, no arm), arms the
      gate, then archives via a status flip on the live augur:memoria:dsr
      record (mirrors the sweep's archive path; a fuller sweep-integrated
      move to the separate archive namespace is out of scope here -- see
      pm.apply_memory_sweep's prune handling). The rollback anchor
      p["action"]["prior_fact"] records the FULL pre-removal state read
      BEFORE the flip (None if the id never existed), so a follow-on undo can
      re-teach the exact removed content instead of reporting "unavailable"
      whenever a real prior exists (spec §8's rollback-anchor table).
    - create/upsert (default): requires a pattern with kind="semantic" (False
      otherwise, no arm -- defense-in-depth mirror of the persistence-layer
      check in create_user_taught_memory, kept here so the SAME validate-then-
      arm ordering as every other handler holds even though the write would
      also refuse). Arms the gate, then creates OR FSRS-reviews the taught
      memory via pm.create_user_taught_memory: re-teaching an EXISTING fact
      strengthens it via a real recurrence review -- it never resets decay
      progress, and reactivates a previously-archived fact (Memoria's FSRS
      decay model, tabula/persistence.py's create_user_taught_memory). The
      rollback anchor p["action"]["prior_fact"] records the FULL pre-apply
      state read BEFORE the write (None for a brand-new memory_id), so an
      undo of a re-teach restores the prior CONTENT via the same
      create/upsert path -- itself a review: even an undo is forward decay,
      never a raw state rollback.

    Self-validating and SELF-ARMING (validate -> arm -> write), matching every
    other _dispatch_confirmed handler's contract.
    """
    a = p.get("action") or {}
    ctx = pm.resolve_learn_context(session_id)
    if a.get("op") == "remove":
        mid = a.get("memory_id")
        if not mid:
            return False
        prior = pm.load_memory_state(mid)  # rollback anchor, read before the flip
        if not _arm_gate(pm, p, cfg=cfg, ctx=ctx):
            return False
        if prior is not None:
            pm.save_memory_state(mid, {**prior, "status": "archived"}, ctx=ctx)
        a["prior_fact"] = prior
        return True
    pattern = a.get("pattern")
    if not pattern or pattern.get("kind") != "semantic":
        return False
    mid = make_memory_id(pattern)
    prior = pm.load_memory_state(mid)  # rollback anchor, read before the write
    if not _arm_gate(pm, p, cfg=cfg, ctx=ctx):
        return False
    # Rationale precedence: action-level first (set by the undo-inverse
    # builder restoring a prior fact's own rationale), then proposal-level
    # (the forward teach, where router.route carried the intent rationale).
    pm.create_user_taught_memory(
        pattern,
        source="dialogue",
        protect=True,
        session_id=session_id,
        cfg=cfg,
        rationale=a.get("rationale") or p.get("rationale") or None,
    )
    a["memory_id"] = mid
    a["prior_fact"] = prior
    return True


def _apply_confirmed(pm, p: dict, *, cfg, session_id: str | None) -> dict:
    """Human-confirmed reversible apply (dialogue teaching). Gated independently by
    cfg.dialogue_confirmed_apply_enabled (default True) -- NOT by
    imperator_ii_apply_enabled (the watch-first autonomous gate, default False); the
    two flags are orthogonal. Only SAFE kinds in P._CONFIRMED_APPLY_KINDS are
    eligible -- GATED (code/structural) kinds are never applied, confirmed or not.

    Bypasses the autonomous path's is_auto_applicable AUTO-kind restriction and its
    is_proposal_applied dedupe pre-check: a human explicitly confirming a change is
    not subject to the anti-thrash staleness window that exists to stop the LLM
    reasoner from re-proposing the same target unattended. The per-kind helper
    still records its own reversibility anchor (prior_target/prior_window/
    prior_text/prior_sigma/prior for gate_calibration/prior_directive for
    context_directive/prior_fact for semantic_fact) so a confirmed change
    remains rollback-able like any other. Sets terminal status. Never raises."""
    if not getattr(cfg, "dialogue_confirmed_apply_enabled", False):
        p["status"] = "logged"
        return p
    if p.get("klass") != "safe" or p.get("kind") not in P._CONFIRMED_APPLY_KINDS:
        p["status"] = "logged"
        return p
    if _conscientia_refuses(pm, p, cfg=cfg, session_id=session_id):
        p["status"] = "logged"
        return p
    try:
        ok = _dispatch_confirmed(pm, p, cfg=cfg, session_id=session_id)
    except Exception:
        # Fail to "logged" (truthful: the caller reports "I couldn't apply
        # that."), but LOG the cause -- otherwise a swallowed ConnectionError, a
        # handler KeyError, and an ordinary validation rejection all produce the
        # same silent "logged" with no operator signal (matches _arm_gate).
        log.warning(
            "confirmed apply failed for kind=%s target=%s; failing to logged",
            p.get("kind"),
            p.get("target"),
            exc_info=True,
        )
        p["status"] = "logged"
        return p
    p["status"] = "applied" if ok else "logged"
    if ok:
        p["applied_session"] = session_id
    return p


def _dispatch_confirmed(pm, p: dict, *, cfg, session_id: str | None = None) -> bool:
    """Route a confirmed proposal to its kind helper, honoring each helper's own
    arming contract: escalation_rule does NOT self-arm, so this caller arms the
    anti-thrash gate before invoking it, same as the autonomous path -- a failed
    arm aborts before the matrix write ever runs. prompt_strategy is
    self-validating and self-arming, so it is called directly.

    sigma, gate_calibration, context_directive, and semantic_fact are likewise
    self-validating and SELF-ARMING (validate -> arm -> write inside the
    handler), so a validation failure never leaves the dedupe marker set.
    semantic_fact is the only handler that reads session_id (threaded through
    to pm.create_user_taught_memory's FSRS review on re-teach)."""
    learn_ctx = pm.resolve_learn_context(session_id)
    kind = p["kind"]
    if kind == "escalation_rule":
        # Same validate -> arm -> write ordering as the autonomous path: a
        # malformed patch must not consume the anti-thrash gate.
        err = _escalation_patch_error(p, cfg=cfg)
        if err:
            log.info(
                "confirmed apply: escalation patch for %s refused before arming: %s",
                p["target"],
                err,
            )
            return False
        if not _arm_gate(pm, p, cfg=cfg, ctx=learn_ctx):
            return False
        return _apply_escalation_rule(pm, p, ctx=learn_ctx)
    if kind == "prompt_strategy":
        return _apply_prompt_strategy(pm, p, cfg=cfg, ctx=learn_ctx)
    if kind == "sigma":
        return _apply_sigma(pm, p, cfg=cfg, ctx=learn_ctx)
    if kind == "gate_calibration":
        return _apply_gate_calibration(pm, p, cfg=cfg, ctx=learn_ctx)
    if kind == "context_directive":
        return _apply_context_directive(pm, p, cfg=cfg, ctx=learn_ctx)
    if kind == "semantic_fact":
        return _apply_semantic_fact(pm, p, cfg=cfg, session_id=session_id)
    return False
