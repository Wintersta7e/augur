"""Sanctioned, reversible, gated application of safe proposals. Honors apply_enabled."""

from __future__ import annotations

import logging
import math
import time

from consilium.prompt_safety import is_prompt_acceptable
from nexus import matrix_ops
from imperator import proposals as P

log = logging.getLogger("imperator.apply")

# Habituation-floor bounds, mirroring Disciplina's floor sweep
# [0.0, GATE_FLOOR_MAX=0.6] (disciplina/reflection_engine.py — not imported:
# that module pulls in httpx/nats and mutates sys.path at import time).
_FLOOR_MIN, _FLOOR_MAX = 0.0, 0.6


def _arm_gate(pm, p: dict, *, cfg) -> bool:
    """Write the durable applied-marker that arms the one-move-per-(kind,target)
    anti-thrash gate. Returns True if armed, False if the write failed.

    The marker is the ONLY thing closing the gate, so it is written BEFORE the
    primary (matrix/prompt) write: a marker failure must abort the apply rather
    than leave a committed change behind an open gate, where a DIFFERENT-text
    proposal for the same target could re-apply in-window and bury the rollback
    anchor. There is no retry path here — a failed arm means the apply does not
    happen, and the unchanged proposal is reconsidered on a later cycle.
    """
    try:
        pm.mark_proposal_applied(
            p["dedupe_key"], ttl_s=int(cfg.imperator_ii_dedupe_staleness_s)
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


def _apply_escalation_rule(pm, p: dict) -> bool:
    """Apply an escalation_rule proposal by updating the matrix and recording the
    rollback anchor. Returns True on success, False on validation error or write failure.

    The rollback anchor is recorded in p["action"]["prior_target"] or
    p["action"]["prior_window"], read from the committed CAS snapshot (not a separate read).
    """
    action = p.get("action") or {}
    if "window" in action:
        res = matrix_ops.apply_matrix_update(
            pm, rule_windows={p["target"]: action["window"]}, mode="patch"
        )
    else:
        res = matrix_ops.apply_matrix_update(
            pm, rules={p["target"]: action.get("target")}, mode="patch"
        )
    if "error" in res:
        return False
    if "window" in action:
        action["prior_window"] = (res.get("prior_rule_windows") or {}).get(p["target"])
    else:
        action["prior_target"] = (res.get("prior_rules") or {}).get(p["target"])
    return True


def _apply_prompt_strategy(pm, p: dict, *, cfg) -> bool:
    """Apply a prompt_strategy proposal: validate, arm the gate, save the new
    prompt text, and record the rollback anchor. Returns True on success, False
    if validation fails, no current prompt exists, or the gate cannot be armed.

    Self-validating with a SINGLE load_prompt read shared by the precondition
    check and the rollback anchor (a second read would open a TOCTOU window
    where the anchor comes from a value that was never validated). Ordering is
    validate -> arm -> save: a validation failure must not arm the gate (a
    corrected proposal for the same target can still apply in-window), and a
    marker failure aborts BEFORE save_prompt runs, so the prior text is never
    archived — the rollback anchor stays intact and no different-text proposal
    for this target can re-apply in-window off an unarmed gate.

    The rollback anchor is recorded in p["action"]["prior_text"] on the success
    path only. Idempotent: only re-saves (save_prompt archives the prior into
    rollback history) when the text actually changes — a re-apply of identical
    text must NOT re-archive and corrupt the rollback anchor.
    """
    action = p.get("action") or {}
    domain, text = action.get("domain", p["target"]), action.get("text", "")
    current = pm.load_prompt(domain)
    if not is_prompt_acceptable(text, cfg) or current is None:
        return False
    if not _arm_gate(pm, p, cfg=cfg):
        return False
    action["prior_text"] = current  # rollback anchor
    if current != text:
        pm.save_prompt(domain, text)
    return True


def _apply_sigma(pm, p: dict, *, cfg) -> bool:
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
    if not _arm_gate(pm, p, cfg=cfg):
        return False
    a["prior_sigma"] = current.get("sigma_threshold")
    pm.save_thresholds(domain, {**current, "sigma_threshold": sigma})
    return True


def _apply_gate_calibration(pm, p: dict, *, cfg) -> bool:
    """Apply a gate_calibration proposal: self_tolerance_add/remove or floor_set.
    Returns True on success, False if validation fails or the gate cannot be armed.

    Handles three ops:
    - self_tolerance_add: adds state_key to self-tolerance set, records prior membership
    - self_tolerance_remove: removes state_key from self-tolerance set, records prior membership
    - floor_set: updates the habituation_floor entry ({"floor": float, "last_ts":
      time.time()}, merged over the prior entry; other signatures' entries are
      untouched — per-field HSET), records the prior entry

    Self-validating and SELF-ARMING (this handler arms; the caller must not):
    ordering is validate -> arm -> write, so a validation failure (unknown op;
    missing, non-numeric, non-finite, or out-of-range floor value) never arms
    the anti-thrash gate — fail closed, no write. The prior state (membership
    bool or prior floor entry) is recorded in p["action"]["prior"] as the
    rollback anchor.
    All Redis writes go through PersistenceManager.
    """
    a = p.get("action") or {}
    op, sk = a.get("op"), a.get("state_key", p["target"])
    if op in ("self_tolerance_add", "self_tolerance_remove"):
        prior = pm.is_self_tolerant(sk)
        if not _arm_gate(pm, p, cfg=cfg):
            return False
        a["prior"] = prior
        if op == "self_tolerance_add":
            pm.add_self_tolerance(sk)
        else:
            pm.remove_self_tolerance(sk)
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
        if not _arm_gate(pm, p, cfg=cfg):
            return False
        a["prior"] = prior_entry
        new_entry = {**prior_entry, "floor": value, "last_ts": time.time()}
        pm.save_gate_tuning_state(floors={sk: new_entry})
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
    try:
        if p["kind"] == "escalation_rule":
            # Arm the anti-thrash gate before the committing matrix write so the
            # write can never land behind an open gate (a failed arm aborts here).
            if not _arm_gate(pm, p, cfg=cfg):
                p["status"] = "logged"
                return p
            if not _apply_escalation_rule(pm, p):
                p["status"] = "logged"
                return p
        elif p["kind"] == "prompt_strategy":
            # Self-validating helper: validate -> arm gate -> save, with a single
            # load_prompt read shared by the check and the rollback anchor.
            if not _apply_prompt_strategy(pm, p, cfg=cfg):
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
    prior_text) so a confirmed change remains rollback-able like any other.
    Sets terminal status. Never raises."""
    if not getattr(cfg, "dialogue_confirmed_apply_enabled", False):
        p["status"] = "logged"
        return p
    if p.get("klass") != "safe" or p.get("kind") not in P._CONFIRMED_APPLY_KINDS:
        p["status"] = "logged"
        return p
    try:
        ok = _dispatch_confirmed(pm, p, cfg=cfg)
    except Exception:
        p["status"] = "logged"
        return p
    p["status"] = "applied" if ok else "logged"
    if ok:
        p["applied_session"] = session_id
    return p


def _dispatch_confirmed(pm, p: dict, *, cfg) -> bool:
    """Route a confirmed proposal to its kind helper, honoring each helper's arming
    contract (Task 9): escalation_rule does NOT self-arm, so this caller arms the
    anti-thrash gate before invoking it, same as the autonomous path -- a failed
    arm aborts before the matrix write ever runs. prompt_strategy is
    self-validating and self-arming, so it is called directly.

    sigma and gate_calibration are likewise self-validating and SELF-ARMING
    (validate -> arm -> write inside the handler), so a validation failure never
    leaves the dedupe marker set. (context_directive, semantic_fact land in
    later tasks)"""
    kind = p["kind"]
    if kind == "escalation_rule":
        if not _arm_gate(pm, p, cfg=cfg):
            return False
        return _apply_escalation_rule(pm, p)
    if kind == "prompt_strategy":
        return _apply_prompt_strategy(pm, p, cfg=cfg)
    if kind == "sigma":
        return _apply_sigma(pm, p, cfg=cfg)
    if kind == "gate_calibration":
        return _apply_gate_calibration(pm, p, cfg=cfg)
    return False
