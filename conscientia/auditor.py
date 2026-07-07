"""Offline review sweep, invoked from Disciplina's reflection (Memoria
precedent). Idempotent per proposal_id; publish failures never fail the
sweep (a reflection must not die on an audit pass)."""

from __future__ import annotations

import json
import logging

from conscientia.review import review_gated

log = logging.getLogger("conscientia.auditor")

VERDICT_SUBJECT = "augur.conscientia.verdict"


async def run_conscientia_review(pm, nc, cfg) -> dict:
    counts = {"reject": 0, "needs_human": 0}
    if not getattr(cfg, "conscientia_enabled", True):
        return {"reviewed": 0, "recommendations": counts}
    reviewed_ids = {
        v.get("proposal_id") for v in pm.load_conscientia_verdicts(limit=200)
    }
    gated = [
        p
        for p in pm.load_proposals(limit=200)
        if p.get("klass") == "gated" and p.get("proposal_id") not in reviewed_ids
    ]
    for p in gated:
        rec = review_gated(p, cfg)
        pm.save_conscientia_verdict(rec)
        counts[rec["recommendation"]] = counts.get(rec["recommendation"], 0) + 1
        if nc is not None:
            try:
                await nc.publish(VERDICT_SUBJECT, json.dumps(rec, default=str).encode())
            except Exception as exc:
                log.warning("verdict publish failed (non-fatal): %s", exc)
    return {"reviewed": len(gated), "recommendations": counts}
