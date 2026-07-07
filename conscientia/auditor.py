"""Offline review sweep, invoked from Disciplina's reflection (Memoria
precedent). Idempotent per proposal_id; publish failures never fail the
sweep (a reflection must not die on an audit pass)."""

from __future__ import annotations

import json
import logging

from conscientia.review import review_gated
from tabula.persistence import MAX_CONSCIENTIA_VERDICTS, MAX_IMPERATOR_PROPOSALS

log = logging.getLogger("conscientia.auditor")

# Idempotency invariant: this sweep reads the *entire* verdict store to
# build reviewed_ids, then reads the *entire* proposal store to find gated
# proposals not yet in it. Proposals are a superset stream of verdicts (one
# verdict per reviewed gated proposal), so "entire" only holds if the
# verdict store's cap is at least as large as the proposal store's cap.
# MAX_CONSCIENTIA_VERDICTS must stay >= MAX_IMPERATOR_PROPOSALS or a
# proposal could scroll out of view before its verdict does, breaking the
# no-re-review / no-double-publish guarantee.
assert MAX_CONSCIENTIA_VERDICTS >= MAX_IMPERATOR_PROPOSALS, (
    "MAX_CONSCIENTIA_VERDICTS must stay >= MAX_IMPERATOR_PROPOSALS -- "
    "run_conscientia_review's idempotency depends on it"
)

VERDICT_SUBJECT = "augur.conscientia.verdict"


async def run_conscientia_review(pm, nc, cfg) -> dict:
    counts = {"reject": 0, "needs_human": 0}
    if not getattr(cfg, "conscientia_enabled", True):
        return {"reviewed": 0, "recommendations": counts}
    reviewed_ids = {
        v.get("proposal_id")
        for v in pm.load_conscientia_verdicts(limit=MAX_CONSCIENTIA_VERDICTS)
    }
    gated = [
        p
        for p in pm.load_proposals(limit=MAX_IMPERATOR_PROPOSALS)
        if p.get("klass") == "gated" and p.get("proposal_id") not in reviewed_ids
    ]
    for p in gated:
        rec = review_gated(p, cfg)
        pm.save_conscientia_verdict(rec)
        counts[rec["recommendation"]] = counts.get(rec["recommendation"], 0) + 1
        if nc is not None:
            try:
                await nc.publish(VERDICT_SUBJECT, json.dumps(rec).encode())
            except Exception as exc:
                log.warning("verdict publish failed (non-fatal): %s", exc)
    return {"reviewed": len(gated), "recommendations": counts}
