"""Praesagium's offline mining sweep -- ``run_praesagium_mining``.

Rides Disciplina's reflection (pass 9, wired in a later task). Orchestrates the
pure math of :mod:`praesagium.patterns` over a :class:`PersistenceManager`:

1. Load the episode corpus (newest ``mine_max_sessions``, skip <2-episode
   sessions).
2. Mine fresh A->B candidates (``mine_corpus``).
3. Fold resolved-prediction outcomes into per-pattern EWMA hit rates past a
   watermark, and merge the lifecycle -- probation / retire / reactivate / bound
   (``merge_blob``).
4. Expire stale open predictions (deadline past) via the atomic resolve op.
5. Persist the single patterns blob and mark the session processed.

This module is the SINGLE WRITER of ``augur:praesagium:patterns`` (invariant
PR8) and does NO NATS (synchronous). It self-gates on ``praesagium_enabled``
(zero reads/writes when off -- PR3), ``is_tuning_applied`` (idempotency), and a
mine-interval rate limit.

Spec: docs/superpowers/specs/2026-07-09-praesagium-design.md Sec 4.1, 4.6-4.7.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from praesagium.patterns import merge_blob, mine_corpus

log = logging.getLogger("praesagium.miner")

_RECOGNIZED_OUTCOMES = ("fulfilled", "expired")


def _number(value: Any) -> float | None:
    """A finite-agnostic numeric coercion: bool and non-numerics -> None."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _resolved_ts(rec: Any) -> float | None:
    if not isinstance(rec, dict):
        return None
    return _number(rec.get("resolved_ts"))


def run_praesagium_mining(session_id: str, pm: Any, config: Any) -> dict:
    """Run one offline mining sweep for *session_id*; return the Sec 4.7 report.

    Returns ``{"skipped": True, "reason": ...}`` for the three gated exits
    (``disabled`` / ``already_processed`` / ``recently_mined``) or the full
    report dict ``{"active", "provisional", "retired", "promoted",
    "reactivated", "corpus_sessions", "resolutions_folded", "expired_open"}``.
    """
    # Gate 1: kill-switch -- ZERO reads/writes when disabled (invariant PR3).
    if not config.praesagium_enabled:
        return {"skipped": True, "reason": "disabled"}

    # Gate 2: idempotency marker (memory-sweep precedent).
    if pm.is_tuning_applied(session_id, pass_name="praesagium"):
        return {"skipped": True, "reason": "already_processed"}

    now = time.time()

    # Gate 3: rate limit -- a freshly-mined blob short-circuits WITHOUT marking
    # (skipped runs neither mark nor fold; the watermark keeps late folds safe).
    prev = pm.load_praesagium_patterns()
    prev_mined_at = _number(prev.get("mined_at")) if isinstance(prev, dict) else None
    if (
        prev_mined_at is not None
        and prev_mined_at > now - config.praesagium_mine_min_interval_s
    ):
        return {"skipped": True, "reason": "recently_mined"}

    # --- Corpus (skip <2-episode sessions; track the newest episode ts). -----
    sids = pm.list_praesagium_episode_sessions(
        limit=config.praesagium_mine_max_sessions
    )
    corpus: dict[str, list[dict]] = {}
    corpus_newest_ts = 0.0
    for sid in sids:
        episodes = pm.load_praesagium_episodes(sid)
        if len(episodes) < 2:
            continue
        corpus[sid] = episodes
        for e in episodes:
            t = _number(e.get("t")) if isinstance(e, dict) else None
            if t is not None and t > corpus_newest_ts:
                corpus_newest_ts = t

    candidates = mine_corpus(corpus, config)

    # --- Resolved-outcome history (newest-first log -> time order). ----------
    resolved = pm.load_praesagium_resolved(limit=config.praesagium_predictions_cap)
    time_ordered = list(reversed(resolved))

    prev_watermark = 0.0
    prev_watermark_ids: set[str] = set()
    watermark_ids_known = True  # no prior blob at all -> nothing to migrate
    if isinstance(prev, dict):
        wm = _number(prev.get("hit_rate_watermark"))
        if wm is not None:
            prev_watermark = wm
        wm_ids = prev.get("hit_rate_watermark_ids")
        # A blob written before the exact-fold rule existed never carried this
        # field at all -- distinct from a present-but-empty list, and must
        # mirror patterns.py's merge_blob so this recount agrees with the
        # actual fold. See praesagium/patterns.py::_fold_resolutions docstring.
        watermark_ids_known = wm_ids is not None
        if isinstance(wm_ids, list):
            prev_watermark_ids = {x for x in wm_ids if isinstance(x, str)}

    # WARN when LTRIM has eaten unfolded resolutions: the oldest retained entry
    # postdates the previous watermark, so outcomes in (watermark, oldest) were
    # dropped from the capped log before they could fold (Sec 4.6-3).
    if prev is not None and time_ordered:
        oldest_ts = _resolved_ts(time_ordered[0])
        if oldest_ts is not None and oldest_ts > prev_watermark:
            log.warning(
                "Praesagium resolved-log trim loss: oldest retained resolved_ts "
                "%.3f > watermark %.3f -- some outcomes were never folded",
                oldest_ts,
                prev_watermark,
            )

    blob = merge_blob(
        prev,
        candidates,
        time_ordered,
        now,
        config,
        corpus_newest_ts=corpus_newest_ts,
    )

    # --- Expire stale open predictions (deadline past). ----------------------
    # Runs AFTER the fold: these expiries (resolved_ts=now) fold on the NEXT
    # mine, never this one -- the watermark makes that safe (no double count).
    expired_open = 0
    for rec in pm.load_praesagium_open_predictions():
        if not isinstance(rec, dict):
            continue
        deadline = _number(rec.get("deadline_ts"))
        if deadline is None or deadline >= now - config.praesagium_expiry_grace_s:
            continue
        pid_rec = rec.get("prediction_id")
        if not pid_rec:
            continue
        resolved_rec = {
            "prediction_id": pid_rec,
            "pattern_id": rec.get("pattern_id"),
            "outcome": "expired",
            "resolved_ts": now,
            "created_ts": rec.get("created_ts"),
            "deadline_ts": deadline,
            "lag_s": None,
            "session_id": rec.get("session_id"),
            "resolved_by": "miner_expiry_sweep",
        }
        try:
            if pm.resolve_praesagium_prediction(
                pid_rec, resolved_rec, cap=config.praesagium_predictions_cap
            ):
                expired_open += 1
        except Exception as exc:  # one bad open must not abort the sweep
            log.warning("Praesagium expiry resolve failed for %s: %s", pid_rec, exc)

    # --- Persist + mark. -----------------------------------------------------
    pm.save_praesagium_patterns(blob)
    pm.mark_tuning_applied(session_id, pass_name="praesagium")

    # --- Report (Sec 4.7). ---------------------------------------------------
    patterns = blob["patterns"]
    prev_patterns = prev.get("patterns", {}) if isinstance(prev, dict) else {}
    if not isinstance(prev_patterns, dict):
        prev_patterns = {}
    # merge_blob folds a resolution iff (ts > watermark) OR (ts == watermark and
    # its prediction_id was not already folded there AND watermark_ids_known),
    # its outcome is recognised, and its pid is in union(prev, candidates);
    # recomputing that exact filter here keeps the report honest without
    # re-running the fold.
    merged_pids = set(prev_patterns) | set(candidates)
    resolutions_folded = 0
    for r in time_ordered:
        ts = _resolved_ts(r)
        if ts is None or ts < prev_watermark:
            continue
        if ts == prev_watermark:
            pred_id = r.get("prediction_id")
            if (
                not watermark_ids_known
                or not isinstance(pred_id, str)
                or pred_id in prev_watermark_ids
            ):
                continue
        if (
            r.get("outcome") in _RECOGNIZED_OUTCOMES
            and r.get("pattern_id") in merged_pids
        ):
            resolutions_folded += 1

    def _prev_status(pid: str) -> Any:
        entry = prev_patterns.get(pid)
        return entry.get("status") if isinstance(entry, dict) else None

    return {
        "active": sum(1 for p in patterns.values() if p.get("status") == "active"),
        "provisional": sum(
            1 for p in patterns.values() if p.get("status") == "provisional"
        ),
        "retired": sum(1 for p in patterns.values() if p.get("status") == "retired"),
        "promoted": sum(
            1
            for pid, p in patterns.items()
            if p.get("status") == "active" and _prev_status(pid) == "provisional"
        ),
        "reactivated": sum(
            1
            for pid, p in patterns.items()
            if p.get("status") in ("active", "provisional")
            and _prev_status(pid) == "retired"
        ),
        "corpus_sessions": len(corpus),
        "resolutions_folded": resolutions_folded,
        "expired_open": expired_open,
    }
