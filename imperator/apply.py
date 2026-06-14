"""Sanctioned, reversible, gated application of safe proposals. Honors apply_enabled."""

from __future__ import annotations

import logging

from consilium.prompt_safety import is_prompt_acceptable
from nexus import matrix_ops
from imperator import proposals as P

log = logging.getLogger("imperator.apply")


def apply_proposal(pm, p: dict, *, cfg, session_id: str | None) -> dict:
    """Apply p iff auto-applicable AND apply enabled AND not already applied AND a
    reversibility anchor exists; else leave 'logged' ('skipped' if already applied).
    Sets terminal status. Never raises on apply failure."""
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
                p["status"] = "logged"
                return p
            # Anchor rollback to the prior value from the COMMITTED CAS snapshot,
            # recorded only on the success path (no anchor on a failed write).
            if "window" in action:
                action["prior_window"] = (res.get("prior_rule_windows") or {}).get(
                    p["target"]
                )
            else:
                action["prior_target"] = (res.get("prior_rules") or {}).get(p["target"])
        elif p["kind"] == "prompt_strategy":
            action = p.get("action") or {}
            domain, text = action.get("domain", p["target"]), action.get("text", "")
            current = pm.load_prompt(domain)
            if not is_prompt_acceptable(text, cfg) or current is None:
                p["status"] = "logged"
                return p
            # Idempotent: only re-save (save_prompt archives the prior into rollback
            # history) when the text actually changes. A re-apply of identical text
            # after a swallowed marker failure must NOT re-archive and corrupt the
            # rollback anchor.
            if current != text:
                pm.save_prompt(domain, text)
        else:
            p["status"] = "logged"
            return p
    except Exception:
        p["status"] = "logged"
        return p
    p["status"] = "applied"
    p["applied_session"] = session_id
    # Best-effort idempotency marker: the write already committed, so a marker
    # failure must not flip a real apply back to an error (re-applying the same
    # reversible patch next cycle is harmless).
    try:
        pm.mark_proposal_applied(
            p["dedupe_key"], ttl_s=int(cfg.imperator_ii_dedupe_staleness_s)
        )
    except Exception:
        log.warning(
            "imperator apply: idempotency marker write failed for %s; a re-apply "
            "is a clean no-op (apply is idempotent), but the marker will be retried",
            p["dedupe_key"],
            exc_info=True,
        )
    return p
