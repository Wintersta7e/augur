"""I/O layer: read the world (Redis via PersistenceManager + held NATS stream state)
into plain dicts for the pure compute modules. fakeredis-testable; no asyncio.
"""

from __future__ import annotations

from tabula.persistence import _to_epoch


def _current_sid(pm) -> str | None:
    """Extract the session_id from the current-session DICT (not a bare SID)."""
    cur = pm.load_current_session()
    return cur.get("session_id") if isinstance(cur, dict) else None


def _recent_session_ids(pm, limit: int = 20) -> list[str]:
    """Newest-first feedback session ids (list_feedback_sessions does NOT exist)."""
    out: list[str] = []
    for r in pm.get_all_feedback(limit=limit) or []:
        sid = r.get("session_id") if isinstance(r, dict) else None
        if sid:
            out.append(sid)
    return out


def resolve_latest_reflection(pm) -> dict | None:
    """Newest reflection report that ACTUALLY EXISTS (reflection lags feedback).

    Try current session, then iterate recent SIDs; rank by report timestamp
    (ISO string -> epoch). None if none exist.
    """
    candidates: list[str] = []
    cur = _current_sid(pm)
    if cur:
        candidates.append(cur)
    for sid in _recent_session_ids(pm):
        if sid not in candidates:
            candidates.append(sid)

    best, best_ts = None, float("-inf")
    for sid in candidates:
        report = pm.load_reflection(sid)
        if report is None:
            continue
        ts = _to_epoch(report.get("timestamp"))
        if ts >= best_ts:
            best, best_ts = report, ts
    return best


def resolve_latest_decision(pm) -> dict | None:
    """Fuse latest FIRED advice (ISO timestamp) + latest SUPPRESSION (numeric ts)."""
    advice = pm.load_last_advice()
    silences = pm.load_silence_records(limit=1)
    silence = silences[0] if silences else None
    if advice is None and silence is None:
        return None
    advice_ts = _to_epoch(advice.get("timestamp")) if advice else float("-inf")
    silence_ts = float(silence.get("ts", float("-inf"))) if silence else float("-inf")
    if silence is not None and silence_ts >= advice_ts:
        return {
            "decision": "suppressed",
            "arm": silence.get("arm"),
            "reason": silence.get("reason"),
            "decision_as_of": silence_ts,
        }
    return {
        "decision": "fired",
        "arm": None,
        "reason": None,
        "decision_as_of": advice_ts,
        "decision_id": (advice or {}).get("decision_id"),
    }


def resolve_reception(pm, last_advice) -> dict | None:
    """Reception of the last fired advice, matched by decision_id in current feedback."""
    if not last_advice or not last_advice.get("decision_id"):
        return (
            None  # no decision_id -> cannot safely join (avoid false None==None match)
        )
    sid = _current_sid(pm)
    if not sid:
        return None
    fb = pm.get_feedback(sid)
    if not fb:
        return None
    target = last_advice.get("decision_id")
    for ev in reversed(fb.get("advice_events", [])):
        if ev.get("decision_id") == target:
            return {
                "explicit_rating": ev.get("explicit_rating"),
                "behavioral_score": ev.get("behavioral_score"),
                "finalized": ev.get("behavioral_finalized"),
                "unmeasurable": ev.get("unmeasurable"),
            }
    return None


_GATE_LOG_CAP = 2000  # MAX_GATE_SILENCES / MAX_GATE_EMISSIONS in tabula/persistence.py


def windowed_rates(pm, now: float, window_s: float) -> dict:
    """True windowed suppression rate + advice volume. Read cap-sized (default 100 < 2000 cap).
    Exclude probe / audit_only emissions from 'genuine delivered'.
    """
    lo = now - window_s
    silences = [
        s
        for s in pm.load_silence_records(limit=_GATE_LOG_CAP)
        if float(s.get("ts", 0.0)) >= lo
    ]
    emissions = [
        e
        for e in pm.load_emissions(limit=_GATE_LOG_CAP)
        if float(e.get("ts", 0.0)) >= lo
        and not e.get("probe")
        and not e.get("audit_only")
    ]
    n_sup, n_del = len(silences), len(emissions)
    denom = n_sup + n_del
    return {
        "suppression_rate": (n_sup / denom) if denom else 0.0,
        "advice_volume": {
            "delivered": n_del,
            "suppressed": n_sup,
            "total_decisions": denom,
        },
    }


def _build_blind_spots(pm, baselines: dict, cfg) -> list[dict]:
    """The spec's five structured, addressable weakness kinds (all from existing keys)."""
    spots: list[dict] = []
    confidence = pm.load_rule_confidence() or {}
    matrix = pm.load_escalation_matrix() or {}
    matrix_rules = matrix.get("rules") or {}
    rule_windows = matrix.get("rule_windows") or {}
    window_state = pm.load_rule_window_state() or {}

    # (a) low-confidence rules + never-evaluated (in matrix, absent from confidence)
    for rk, info in confidence.items():
        conf = info.get("confidence", 1.0) if isinstance(info, dict) else 1.0
        if conf < 0.5:
            spots.append(
                {
                    "kind": "low_confidence_rule",
                    "detail": f"{rk} conf={conf:.2f}",
                    "evidence": rk,
                }
            )
    for rk in matrix_rules:
        if rk not in confidence:
            spots.append(
                {
                    "kind": "never_evaluated_rule",
                    "detail": f"{rk} has no confidence data",
                    "evidence": rk,
                }
            )

    # (b) mis-sized windows — derived target vs current window (Disciplina's formula)
    mult = cfg.correlation_window_lag_multiplier
    lo, hi = cfg.correlation_window_min_s, cfg.correlation_window_max_s
    for rk, st in window_state.items():
        lag = st.get("ewma_lag") if isinstance(st, dict) else None
        if lag is None:
            continue
        target = max(lo, min(lag * mult, hi))
        current = rule_windows.get(rk, cfg.correlation_window_s)
        if (
            current
            and abs(target - current) / current
            >= cfg.correlation_window_tuning_hysteresis_pct
        ):
            spots.append(
                {
                    "kind": "mis_sized_window",
                    "detail": f"{rk} window={current:.1f}s target~{target:.1f}s",
                    "evidence": rk,
                }
            )

    # (c) muted channels
    for muted in pm.load_self_tolerance() or []:
        spots.append({"kind": "muted_channel", "detail": str(muted), "evidence": muted})

    # (d) starving channels — nearing the configured cap
    near = max(1, cfg.gate_max_consecutive_suppressions - 2)
    for sk, stats in pm.load_all_channel_stats().items():
        cs = stats.get("consecutive_suppressions", 0) if isinstance(stats, dict) else 0
        if cs >= near:
            spots.append(
                {
                    "kind": "starving_channel",
                    "detail": f"{sk} consec={cs}/{cfg.gate_max_consecutive_suppressions}",
                    "evidence": sk,
                }
            )

    # (e) undertrained baselines
    if baselines.get("untrained", 0) > 0:
        spots.append(
            {
                "kind": "undertrained_baselines",
                "detail": f"{baselines['untrained']} baselines < {cfg.imperator_baseline_trained_obs} obs",
                "evidence": baselines.get("by_domain", {}),
            }
        )
    return spots


def _rollup_health(snapshot) -> str | None:
    if not snapshot:
        return None
    bad = {"degraded", "stale", "warming_up", "unknown"}
    overalls = [
        f.get("overall")
        for f in snapshot.get("faculties", {}).values()
        if f.get("required")
    ]
    if any(o == "dead" for o in overalls):
        return "down"
    if any(o in bad for o in overalls):
        return "degraded"
    return "healthy"


def _latest_value(history):
    return history[0].get("value") if history else None


def gather(pm, stream_state: dict, now: float, cfg) -> dict:
    """Assemble the full input dict for compute_auspices + compute_self_model."""
    report = resolve_latest_reflection(pm)
    analyses = (report or {}).get("analyses", {})
    pd = analyses.get("precision", {}).get("per_domain", {})
    ratios: list[float] = []
    for d in pd.values():
        if isinstance(d, dict) and d.get("precision_ratio") is not None:
            ratios.append(d["precision_ratio"])
    precision = (sum(ratios) / len(ratios)) if ratios else None
    util = analyses.get("utility", {})
    utility = util.get("utility_score")
    utility_no_data = (not util) or "No advice events" in str(util.get("reason", ""))

    rates = windowed_rates(pm, now, cfg.imperator_rate_window_s)
    baselines = pm.scan_baseline_maturity(
        trained_obs=cfg.imperator_baseline_trained_obs
    )
    coverage_no_data = baselines["total"] == 0
    coverage_depth = (
        None if coverage_no_data else baselines["trained"] / baselines["total"]
    )
    health = pm.load_health_snapshot()
    last_advice = pm.load_last_advice()
    advice_rate = pm.load_advice_rate() if hasattr(pm, "load_advice_rate") else None
    dismissal = advice_rate.get("rate_ewma") if isinstance(advice_rate, dict) else None

    intens_hist = pm.get_history("activity_intensity", limit=1)

    rollup = _rollup_health(health)
    return {
        "session_id": _current_sid(pm),
        "activity": pm.load_focused_app(
            now=now, max_age_s=getattr(cfg, "focused_app_max_age_s", 300.0)
        ),
        "intensity_ewma": _latest_value(intens_hist),
        "anomaly_load": stream_state.get("anomaly_load"),
        "escalation_tier": stream_state.get("escalation_tier"),
        "has_active_correlation": stream_state.get("has_active_correlation"),
        "active_correlations": stream_state.get("active_correlations"),
        "last_advice": last_advice,
        "reception": resolve_reception(pm, last_advice),
        "pipeline_health_rollup": rollup,
        "precision": precision,
        "utility": utility,
        "utility_no_data": utility_no_data,
        "mrt": analyses.get("gate", {}).get("mrt"),
        "suppression_rate": rates["suppression_rate"],
        "dismissal_rate": dismissal,
        "advice_volume": rates["advice_volume"],
        "pipeline_health_full": health,
        "health_score": 1.0 if rollup == "healthy" else 0.5,
        "coverage": {"coverage_depth": coverage_depth, **baselines},
        "coverage_no_data": coverage_no_data,
        "latest_decision": resolve_latest_decision(pm),
        "blind_spots": _build_blind_spots(pm, baselines, cfg),
        "recent_self_tuning": (report or {}).get("adjustments"),
        # Epoch of the reflection actually folded into this self-model, so the
        # Imperator-II freshness gate can content-check (not wall-clock) that the
        # triggering reflection has been incorporated. 0.0 when none exists.
        "reflection_ts": _to_epoch((report or {}).get("timestamp")),
    }
