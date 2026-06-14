"""Sanctioned, reversible, gated application of safe proposals. Honors apply_enabled."""

from __future__ import annotations

from consilium.prompt_safety import is_prompt_acceptable
from nexus import matrix_ops
from imperator import proposals as P


def apply_proposal(pm, p: dict, *, cfg, session_id: str | None) -> dict:
    """Apply p iff auto-applicable AND apply enabled AND a reversibility anchor exists;
    else leave 'logged'. Sets terminal status. Never raises on apply failure."""
    if not getattr(
        cfg, "imperator_ii_apply_enabled", False
    ) or not P.is_auto_applicable(p):
        p["status"] = "logged"
        return p
    try:
        if p["kind"] == "escalation_rule":
            action = p.get("action") or {}
            cur = pm.load_escalation_matrix() or {}
            if "window" in action:
                action["prior_window"] = (cur.get("rule_windows") or {}).get(
                    p["target"]
                )
                res = matrix_ops.apply_matrix_update(
                    pm, rule_windows={p["target"]: action["window"]}, mode="patch"
                )
            else:
                action["prior_target"] = (cur.get("rules") or {}).get(p["target"])
                res = matrix_ops.apply_matrix_update(
                    pm, rules={p["target"]: action.get("target")}, mode="patch"
                )
            if "error" in res:
                p["status"] = "logged"
                return p
        elif p["kind"] == "prompt_strategy":
            action = p.get("action") or {}
            domain, text = action.get("domain", p["target"]), action.get("text", "")
            if not is_prompt_acceptable(text, cfg) or pm.load_prompt(domain) is None:
                p["status"] = "logged"
                return p
            pm.save_prompt(domain, text)
        else:
            p["status"] = "logged"
            return p
    except Exception:
        p["status"] = "logged"
        return p
    p["status"] = "applied"
    p["applied_session"] = session_id
    pm.mark_proposal_applied(
        p["dedupe_key"], ttl_s=int(cfg.imperator_ii_dedupe_staleness_s)
    )
    if session_id:
        pm.mark_tuning_applied(session_id, pass_name="imperator")
    return p
