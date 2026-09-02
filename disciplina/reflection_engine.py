"""Session reflection engine — self-adjusts Augur parameters after each session.

Triggers on augur.responsum.complete (end of feedback collection) or
augur.disciplina.trigger (manual). Runs four analyses:

1. Precision  — Were anomaly detections accurate? Adjusts sigma threshold.
2. Utility    — Was the advice useful? May mutate LLM prompt via Ollama.
3. Counterfactual — Would +-10% threshold variants have been better?
4. Correlation tuning — Per-rule EWMA confidence with hysteresis to
   tune the cross-domain escalation matrix.

Publishes a reflection report to augur.disciplina.complete and persists
it via PersistenceManager.save_reflection.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any  # noqa: F401 — used in PEP-563 deferred annotations

import httpx
import nats
import redis

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tabula.config import AugurConfig
from tabula.connections import connect_redis
from tabula.heartbeat import start_heartbeat
from tabula.persistence import PersistenceManager
from nexus.correlator import DEFAULT_ESCALATION_MATRIX
from nexus import matrix_ops
from consilium.prompt_safety import (
    violates_forbidden_patterns as _violates_forbidden_patterns,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("reflection_engine")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SUBJECT_FEEDBACK_COMPLETE = "augur.responsum.complete"
SUBJECT_REFLECT_TRIGGER = "augur.disciplina.trigger"
SUBJECT_REFLECT_COMPLETE = "augur.disciplina.complete"

# Fallback domain used only when a session's feedback contains no usable
# standalone advice events. Per-session domain is now derived from
# feedback.advice_events instead of being hardcoded (ARCH-06 fix). Previous
# behaviour silently applied chess-domain threshold and prompt tuning to
# typing-only sessions.
DEFAULT_DOMAIN = "chess"

# ---------------------------------------------------------------------------
# Gate offline-pass tuning constants (spec §9)
# ---------------------------------------------------------------------------
# Self-tolerance membership requires BOTH chronic presence (the channel has
# been advised on repeatedly across the de-duped feedback) AND explicit
# dismissal corroboration — never behavioral evidence alone.  Exempt-shaped
# channels (correlated HIGH) are excluded outright.
GATE_CHRONIC_MIN_PRESENCE = 5  # min advice events on a channel to be "chronic"
GATE_DISMISSAL_MIN = 3  # min explicit "n" dismissals to corroborate
# How far each session may move the per-channel habituation floor / the global
# advice-rate operating point — small, conservative steps so a single noisy
# session never swings the gate (mirrors the EWMA alphas in config).
GATE_FLOOR_STEP = 0.05
GATE_FLOOR_MAX = 0.6
GATE_ADVICE_RATE_ALPHA = 0.1


def _attribution_weights(event: dict) -> dict[str, float]:
    """Return {domain: weight} per advice event.

    Standalone (non-correlated) advice: full weight 1.0 to event['domain'].
    Correlated advice with involved_domains: each involved domain gets 1/N.
    Old records without 'involved_domains' (or empty): falls back to {primary: 1.0}.
    """
    primary = event.get("domain", "unknown")
    if not event.get("correlation_found"):
        return {primary: 1.0}
    domains = event.get("involved_domains") or []
    if not domains:
        return {primary: 1.0}
    weight = 1.0 / len(domains)
    return {d: weight for d in domains}


def _derive_domain(feedback: dict) -> str:
    """Pick a domain for utility/counterfactual single-domain scope.

    Priority order:
    1. Most-common domain among standalone (non-correlated) advice events.
       Standalone advice is what utility analysis & prompt mutation operate on.
    2. Most-common primary domain across ALL advice events (used when the
       session was entirely correlated; post-Task-8 correlated records carry
       the real primary domain so this is meaningful).
    3. DEFAULT_DOMAIN as last resort.

    Praesagium forewarnings are excluded from candidate counting at both
    levels: they are a detector-loop domain, not a real advice domain, and
    analyze_utility already filters them out of its own scoring. Without this
    exclusion here too, a session where praesagium events are the numeric
    majority could still derive domain="praesagium" while a minority real
    domain's poor utility trips needs_prompt_mutation -- sending the earned
    mutate_prompt call (a live LLM call) to praesagium instead of the real
    domain, plus a dead prompt write nothing loads. If every event is
    praesagium, fall through to the existing DEFAULT_DOMAIN behaviour.
    """
    advice_events = feedback.get("advice_events", [])
    standalone_domains = [
        ev.get("domain")
        for ev in advice_events
        if ev.get("domain")
        and ev.get("domain") != "praesagium"
        and not ev.get("correlation_found")
    ]
    if standalone_domains:
        return Counter(standalone_domains).most_common(1)[0][0]
    # Round-3 fallback: post-Task-8, correlated records have a real domain too.
    all_domains = [
        ev.get("domain")
        for ev in advice_events
        if ev.get("domain") and ev.get("domain") != "praesagium"
    ]
    if all_domains:
        return Counter(all_domains).most_common(1)[0][0]
    return DEFAULT_DOMAIN


# ---------------------------------------------------------------------------
# Precision analysis
# ---------------------------------------------------------------------------


def analyze_precision(
    feedback: dict,
    current_thresholds_per_domain: dict[str, dict],
    config: AugurConfig,
) -> dict:
    """Per-domain detection-precision analysis with multi-domain attribution.

    Standalone advice contributes 1.0 weight to its primary domain;
    correlated advice contributes 1/N to each involved domain. A domain
    needs >= 2.0 weighted-total signal to receive a sigma adjustment.

    Returns:
      {
        "analysis": "precision",
        "per_domain": {<domain>: {total_anomalies, useful, precision_ratio,
                                   action, sigma_before, sigma_after, reason}, ...},
        "domains_evaluated": [<domain>, ...],
      }
    """
    from collections import defaultdict

    weighted_totals: dict[str, float] = defaultdict(float)
    weighted_useful: dict[str, float] = defaultdict(float)

    # Praesagium forewarnings have no sensor and no Vigil threshold to tune
    # (detector-loop containment, spec 2026-07-09 §4.7) — excluded here so
    # they never accumulate weighted signal toward a sigma adjustment.
    for ev in feedback.get("advice_events", []):
        if ev.get("domain") == "praesagium":
            continue
        weights = _attribution_weights(ev)
        useful = (
            ev.get("explicit_rating") == "y" or ev.get("behavioral_score", 0) >= 0.7
        )
        for domain, w in weights.items():
            weighted_totals[domain] += w
            if useful:
                weighted_useful[domain] += w

    per_domain: dict[str, dict] = {}
    for domain in sorted(weighted_totals.keys()):
        total = weighted_totals[domain]
        useful = weighted_useful[domain]
        thresholds = current_thresholds_per_domain.get(domain, {"sigma_threshold": 2.0})
        sigma_before = thresholds.get("sigma_threshold", 2.0)
        sigma_after = sigma_before
        action = "none"

        if total < 2.0:
            reason = f"Insufficient signal for {domain} ({total:.1f} weighted events)"
        else:
            precision = useful / total
            if precision < 0.3:
                sigma_after = min(
                    sigma_before + config.sigma_adjust_step, config.sigma_max
                )
                action = "raise_sigma"
                reason = (
                    f"{domain} precision {precision:.0%} ({useful:.1f}/{total:.1f}); "
                    f"raising sigma {sigma_before:.1f} -> {sigma_after:.1f}"
                )
            elif precision > 0.8:
                sigma_after = max(
                    sigma_before - config.sigma_adjust_step, config.sigma_min
                )
                action = "lower_sigma"
                reason = (
                    f"{domain} precision {precision:.0%} ({useful:.1f}/{total:.1f}); "
                    f"lowering sigma {sigma_before:.1f} -> {sigma_after:.1f}"
                )
            else:
                reason = f"{domain} precision {precision:.0%} acceptable"

        per_domain[domain] = {
            "total_anomalies": round(total, 3),
            "useful": round(useful, 3),
            "precision_ratio": round(useful / total, 3) if total > 0 else 0.0,
            "action": action,
            "sigma_before": sigma_before,
            "sigma_after": sigma_after,
            "reason": reason,
        }

    return {
        "analysis": "precision",
        "per_domain": per_domain,
        "domains_evaluated": list(per_domain.keys()),
    }


# ---------------------------------------------------------------------------
# Utility analysis
# ---------------------------------------------------------------------------


def analyze_utility(feedback: dict, config: AugurConfig) -> dict:
    """Evaluate advice utility from explicit + behavioral signals.

    Weighted score: 60% explicit, 40% behavioral.
    If utility < config.utility_mutation_threshold, flags for prompt mutation.

    Excludes correlated advice events (correlation_found=True) because
    prompt mutation only affects the single-domain DOMAIN_HANDLERS path;
    the correlation path uses build_correlation_prompt which is
    self-contained and not managed by PersistenceManager.save_prompt.

    Also excludes praesagium-domain advice (detector-loop containment, spec
    2026-07-09 §4.7) — a low-utility run of forewarnings must never trigger
    an Ollama call to mutate a prompt nothing loads.
    """
    # Filter out correlated advice before computing the score that drives
    # prompt mutation. The correlation path is tuned by the matrix tuning
    # analysis in a later step of run_reflection, not by prompt mutation.
    all_events = feedback.get("advice_events", [])
    advice_events = [
        e
        for e in all_events
        if not e.get("correlation_found") and e.get("domain") != "praesagium"
    ]

    total = len(advice_events)
    if total == 0:
        return {
            "analysis": "utility",
            "utility_score": 1.0,
            "explicit_component": 1.0,
            "behavioral_component": 1.0,
            "needs_prompt_mutation": False,
            "reason": "No advice events to evaluate",
        }

    # Explicit component: y=1.0, n=0.0, no_response=0.5
    explicit_scores: list[float] = []
    for ev in advice_events:
        rating = ev.get("explicit_rating", "no_response")
        if rating == "y":
            explicit_scores.append(1.0)
        elif rating == "n":
            explicit_scores.append(0.0)
        else:
            explicit_scores.append(0.5)

    explicit_avg = (
        sum(explicit_scores) / len(explicit_scores) if explicit_scores else 0.5
    )

    # Behavioral component. Under the surprise-reduction metric (spec §7) a
    # finalized 0.0 is a VALID strong-negative outcome, not "missing" — filter
    # on behavioral_finalized + not unmeasurable, never on `> 0`.
    behavioral_scores = [
        ev.get("behavioral_score", 0.5)
        for ev in advice_events
        if ev.get("behavioral_finalized") and not ev.get("unmeasurable")
    ]
    behavioral_avg = (
        sum(behavioral_scores) / len(behavioral_scores) if behavioral_scores else 0.5
    )

    # Weighted combination
    utility = 0.6 * explicit_avg + 0.4 * behavioral_avg
    needs_mutation = utility < config.utility_mutation_threshold and total >= 2

    reason = f"Utility {utility:.2f} (explicit={explicit_avg:.2f}, behavioral={behavioral_avg:.2f})"
    if needs_mutation:
        reason += " — below threshold, prompt mutation recommended"

    return {
        "analysis": "utility",
        "utility_score": round(utility, 3),
        "explicit_component": round(explicit_avg, 3),
        "behavioral_component": round(behavioral_avg, 3),
        "needs_prompt_mutation": needs_mutation,
        "reason": reason,
    }


# ---------------------------------------------------------------------------
# Counterfactual analysis
# ---------------------------------------------------------------------------


def analyze_counterfactual(
    pm: PersistenceManager,
    domain: str,
    current_thresholds: dict,
) -> dict:
    """Replay recent events against +-10% sigma variants.

    Counts how many events each variant would have flagged.
    """
    history = pm.get_history(domain, limit=100)
    if not history:
        return {
            "analysis": "counterfactual",
            "events_replayed": 0,
            "variants": {},
            "recommendation": "No history to replay",
        }

    sigma = current_thresholds.get("sigma_threshold", 2.0)
    variants = {
        "current": sigma,
        "minus_10pct": round(sigma * 0.9, 2),
        "plus_10pct": round(sigma * 1.1, 2),
    }

    # Compute EWMA stats from history to estimate deviations
    # History is newest-first from Redis; reverse for chronological
    events = list(reversed(history))
    alpha = current_thresholds.get("ewma_alpha", 0.3)

    # Per-entity baselines
    entity_baselines: dict[str, dict[str, float]] = {}
    entity_deviations: dict[str, list[float]] = {}

    for ev in events:
        entity = ev.get("entity", "unknown")
        value = ev.get("value", 0.0)

        if entity not in entity_baselines:
            entity_baselines[entity] = {"mean": value, "var": 0.0, "n": 0}
            entity_deviations[entity] = []

        bl = entity_baselines[entity]
        bl["n"] += 1

        if bl["n"] == 1:
            bl["mean"] = value
            bl["var"] = 0.0
            entity_deviations[entity].append(0.0)
        else:
            std = math.sqrt(max(bl["var"], 0.0))
            dev = abs(value - bl["mean"]) / std if std > 0.01 else 0.0
            entity_deviations[entity].append(dev)
            diff = value - bl["mean"]
            bl["mean"] += alpha * diff
            bl["var"] = (1 - alpha) * (bl["var"] + alpha * diff * diff)

    # Flatten all deviations
    all_devs = [d for devs in entity_deviations.values() for d in devs]

    counts = {}
    for name, threshold in variants.items():
        flagged = sum(1 for d in all_devs if d >= threshold)
        counts[name] = {
            "sigma": threshold,
            "would_flag": flagged,
            "flag_rate": round(flagged / len(all_devs), 3) if all_devs else 0.0,
        }

    # Recommendation
    current_flags = counts["current"]["would_flag"]
    lower_flags = counts["minus_10pct"]["would_flag"]
    higher_flags = counts["plus_10pct"]["would_flag"]

    if lower_flags > current_flags * 1.5:
        recommendation = (
            f"Lower threshold ({variants['minus_10pct']}) would catch "
            f"{lower_flags - current_flags} more events"
        )
    elif higher_flags < current_flags * 0.7:
        recommendation = (
            f"Higher threshold ({variants['plus_10pct']}) would reduce "
            f"flags by {current_flags - higher_flags}"
        )
    else:
        recommendation = "Current threshold appears well-calibrated"

    return {
        "analysis": "counterfactual",
        "events_replayed": len(all_devs),
        "variants": counts,
        "recommendation": recommendation,
    }


# ---------------------------------------------------------------------------
# Correlation matrix tuning (cross-domain escalation rule tuning)
# ---------------------------------------------------------------------------


def analyze_correlation_tuning(
    feedback: dict,
    current_matrix: dict,
    current_confidence_state: dict,
    config: AugurConfig,
) -> dict[str, Any]:
    """Tune the cross-domain escalation matrix via per-rule EWMA confidence.

    For each correlated advice event with a valid rule_key, attribute the
    feedback utility to that rule. Update a per-rule confidence state via
    EWMA across sessions. Use two hysteresis thresholds to decide whether
    to keep a rule at its current target, disable it (target=LOW), or
    re-enable it (restore to snapshotted target).

    Pure function: no Redis or NATS I/O. See design doc at
    docs/superpowers/specs/2026-04-09-reflection-matrix-tuning-design.md
    for the full algorithm specification.
    """
    if not config.correlation_tuning_enabled:
        return {
            "analysis": "correlation_tuning",
            "disabled": True,
            "reason": "Tuning disabled via AUGUR_CORRELATION_TUNING_ENABLED=false",
        }

    # Filter to correlated events with a valid rule_key
    all_events = feedback.get("advice_events", [])
    events_with_rule = [
        e
        for e in all_events
        if e.get("correlation_found") is True and e.get("rule_key") is not None
    ]

    if not events_with_rule:
        return {
            "analysis": "correlation_tuning",
            "rules_evaluated": 0,
            "per_rule": {},
            "new_confidence_state": dict(current_confidence_state),
            "new_matrix": None,
            "reason": "No correlated advice events with rule attribution",
        }

    # Group by rule_key
    events_per_rule: dict[str, list[dict]] = {}
    for ev in events_with_rule:
        rk = ev["rule_key"]
        events_per_rule.setdefault(rk, []).append(ev)

    alpha = config.correlation_tuning_alpha
    enable_t = config.correlation_tuning_enable_threshold
    disable_t = config.correlation_tuning_disable_threshold

    # Start with a shallow copy of the existing state so unchanged rules pass through.
    updated_state: dict = {k: dict(v) for k, v in current_confidence_state.items()}
    new_matrix_rules = dict(current_matrix.get("rules", {}))
    per_rule_result: dict = {}
    matrix_changed = False

    explicit_map = {"y": 1.0, "n": 0.0, "no_response": 0.5}

    for rule_key, events in events_per_rule.items():
        # Per-rule session utility (60/40 explicit/behavioral)
        explicit_scores = [
            explicit_map.get(ev.get("explicit_rating", "no_response"), 0.5)
            for ev in events
        ]
        explicit_avg = sum(explicit_scores) / len(explicit_scores)

        # A finalized 0.0 is a valid negative outcome under the surprise-reduction
        # metric (spec §7) — filter on behavioral_finalized, not `> 0`.
        behavioral_scores = [
            ev.get("behavioral_score", 0.0)
            for ev in events
            if ev.get("behavioral_finalized") and not ev.get("unmeasurable")
        ]
        behavioral_avg = (
            sum(behavioral_scores) / len(behavioral_scores)
            if behavioral_scores
            else 0.5
        )

        session_utility = 0.6 * explicit_avg + 0.4 * behavioral_avg

        # Load previous state or initialize to 1.0 (presumed useful)
        prev_state = updated_state.get(
            rule_key, {"confidence": 1.0, "restore_target": None}
        )
        prev_conf = prev_state["confidence"]
        prev_restore = prev_state.get("restore_target")

        # EWMA update, rounded to 3 decimals for deterministic comparison
        new_conf = round((1 - alpha) * prev_conf + alpha * session_utility, 3)

        # Current target in the matrix (may differ from default due to past tuning
        # or manual MCP edits). "LOW" fallback for rules not in the matrix.
        current_target = new_matrix_rules.get(rule_key, "LOW")

        # Derive new target via hysteresis + restore_target snapshot
        if new_conf >= enable_t:
            if current_target == "LOW":
                # Recovering from disabled — restore to snapshot (or current if no snapshot)
                new_target = (
                    prev_restore if prev_restore is not None else current_target
                )
            else:
                # Already enabled — track the live matrix value so manual edits propagate
                new_target = current_target
            # Refresh the snapshot to track the current live target
            new_restore = new_target
        elif new_conf < disable_t:
            # Disable
            new_target = "LOW"
            if prev_restore is not None:
                new_restore = prev_restore
            else:
                # First-ever disable — capture current_target (unless it's already LOW)
                new_restore = current_target if current_target != "LOW" else None
        else:
            # Hysteresis band — freeze target and restore_target
            new_target = current_target
            new_restore = prev_restore

        # Classify action based on target change
        if current_target == "LOW" and new_target != "LOW":
            action = "re-enabled"
        elif current_target != "LOW" and new_target == "LOW":
            action = "disabled"
        else:
            action = "tracked"

        # Track matrix mutation
        if new_target != current_target:
            new_matrix_rules[rule_key] = new_target
            matrix_changed = True

        # Update state
        updated_state[rule_key] = {
            "confidence": new_conf,
            "restore_target": new_restore,
        }

        per_rule_result[rule_key] = {
            "session_utility": round(session_utility, 3),
            "event_count": len(events),
            "confidence_before": prev_conf,
            "confidence_after": new_conf,
            "target_before": current_target,
            "target_after": new_target,
            "restore_target_before": prev_restore,
            "restore_target_after": new_restore,
            "action": action,
        }

    # Build the new_matrix dict if anything changed
    if matrix_changed:
        new_matrix: dict | None = {
            "version": current_matrix.get("version", "1.0"),
            "rules": new_matrix_rules,
        }
    else:
        new_matrix = None

    # Build a summary reason string
    reason_parts = [
        f"{rk} conf {r['confidence_before']}->{r['confidence_after']}, "
        f"{r['target_before']}->{r['target_after']} ({r['action']})"
        for rk, r in per_rule_result.items()
    ]
    reason = "; ".join(reason_parts) if reason_parts else "No rules updated"

    return {
        "analysis": "correlation_tuning",
        "rules_evaluated": len(events_per_rule),
        "per_rule": per_rule_result,
        "new_confidence_state": updated_state,
        "new_matrix": new_matrix,
        "reason": reason,
    }


# ---------------------------------------------------------------------------
# Correlation window tuning (per-rule observed-lag EWMA)
# ---------------------------------------------------------------------------


def analyze_correlation_window_tuning(
    feedback: dict,
    current_matrix: dict,
    current_window_state: dict,
    config: AugurConfig,
) -> dict[str, Any]:
    """Update per-rule observed-lag EWMA, derive new rule_windows.

    EWMA tracks observed *lag*; window is derived from EWMA at update time
    via lag_multiplier and clamp to [min_s, max_s]. Hysteresis on the
    derived window prevents flapping.

    Pairwise rule_keys only this phase — N-way (3+) keys are skipped.

    Pure function: no Redis or NATS I/O.
    """
    from collections import defaultdict

    if not config.correlation_tuning_enabled:
        return {
            "analysis": "correlation_window_tuning",
            "disabled": True,
            "reason": "Tuning disabled via AUGUR_CORRELATION_TUNING_ENABLED=false",
            "rules_evaluated": 0,
            "per_rule": {},
            "new_window_state": dict(current_window_state),
            "new_rule_windows": None,
        }

    events_with_lag = [
        e
        for e in feedback.get("advice_events", [])
        if e.get("correlation_found")
        and isinstance(e.get("rule_key"), str)
        and e["rule_key"].count("+") == 1  # pairwise only
        and isinstance(e.get("correlation_span_s"), (int, float))
    ]
    if not events_with_lag:
        return {
            "analysis": "correlation_window_tuning",
            "rules_evaluated": 0,
            "per_rule": {},
            "new_window_state": dict(current_window_state),
            "new_rule_windows": None,
            "reason": "No pairwise correlated advice events with lag data",
        }

    spans_per_rule: dict[str, list[float]] = defaultdict(list)
    for ev in events_with_lag:
        spans_per_rule[ev["rule_key"]].append(ev["correlation_span_s"])

    current_windows = current_matrix.get("rule_windows", {})
    new_windows = dict(current_windows)
    new_state = {k: dict(v) for k, v in current_window_state.items()}
    per_rule_result: dict = {}
    windows_changed = False

    alpha = config.correlation_window_tuning_alpha

    for rule_key, spans in spans_per_rule.items():
        session_mean = sum(spans) / len(spans)

        prev_state = new_state.get(rule_key, {"ewma_lag": session_mean})
        prev_ewma_lag = prev_state["ewma_lag"]
        new_ewma_lag = round((1 - alpha) * prev_ewma_lag + alpha * session_mean, 3)

        target_window = max(
            config.correlation_window_min_s,
            min(
                new_ewma_lag * config.correlation_window_lag_multiplier,
                config.correlation_window_max_s,
            ),
        )
        target_window = round(target_window, 1)

        current_window = current_windows.get(rule_key, config.correlation_window_s)
        delta_pct = (
            abs(target_window - current_window) / current_window
            if current_window > 0
            else 1.0
        )

        if delta_pct >= config.correlation_window_tuning_hysteresis_pct:
            new_windows[rule_key] = target_window
            windows_changed = True
            action = "tuned"
            window_after = target_window
        else:
            action = "held"
            window_after = current_window

        new_state[rule_key] = {"ewma_lag": new_ewma_lag}

        per_rule_result[rule_key] = {
            "session_mean_span": round(session_mean, 3),
            "event_count": len(spans),
            "ewma_lag_before": prev_ewma_lag,
            "ewma_lag_after": new_ewma_lag,
            "window_before": current_window,
            "window_after": window_after,
            "delta_pct": round(delta_pct, 3),
            "action": action,
        }

    return {
        "analysis": "correlation_window_tuning",
        "rules_evaluated": len(spans_per_rule),
        "per_rule": per_rule_result,
        "new_rule_windows": new_windows if windows_changed else None,
        "new_window_state": new_state,
        "reason": "; ".join(
            f"{k}: {v['action']} {v['window_before']}→{v['window_after']}"
            for k, v in per_rule_result.items()
        )
        or "No rules updated",
    }


# ---------------------------------------------------------------------------
# Prompt mutation via Ollama
# ---------------------------------------------------------------------------


def maybe_rollback_prompt(
    pm: PersistenceManager, domain: str, config: AugurConfig, *, ctx=None
) -> bool:
    """Roll back if the current prompt's REALIZED score regressed past the margin
    vs its predecessor's realized score (spec 1E §9). Both scores must exist
    (a freshly-seeded prompt with no predecessor never rolls back). Returns True
    iff a rollback was performed."""
    cur, prev = pm.get_prompt_score_pair(domain)
    if cur is None or prev is None:
        return False
    if (prev - cur) > config.prompt_rollback_margin:
        return pm.rollback_prompt(domain, ctx=ctx)
    return False


async def mutate_prompt(
    pm: PersistenceManager,
    domain: str,
    utility_result: dict,
    http_client: httpx.AsyncClient,
    config: AugurConfig,
    *,
    ctx=None,
) -> dict | None:
    """Ask Ollama to suggest a better system prompt based on feedback."""
    current_prompt = pm.load_prompt(domain)
    seeded = current_prompt is None
    if seeded:
        # No managed prompt yet — create a seed.
        current_prompt = (
            "You are a chess analyst reviewing a game in progress. "
            "Provide concise, actionable advice about timing anomalies."
        )
        # Persist the seed (with the current realized utility) BEFORE mutating so
        # the first mutation archives it into history → rollback has a target
        # (spec 1E, MEDIUM-2).
        pm.save_prompt(
            domain, current_prompt, score=utility_result["utility_score"], ctx=ctx
        )

    prompt = f"""You are an AI prompt engineer. A chess advisor system has been receiving
low utility scores from users (score: {utility_result["utility_score"]:.2f}/1.0).

The current system prompt for the advisor is:
---
{current_prompt}
---

Explicit feedback score: {utility_result["explicit_component"]:.2f}/1.0
Behavioral feedback score: {utility_result["behavioral_component"]:.2f}/1.0

Suggest an improved version of the system prompt that would produce more useful,
actionable chess advice. Focus on making advice more specific and practical.

Return ONLY the new prompt text, nothing else."""

    try:
        payload = {
            "model": config.ollama_model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.8, "num_predict": 512},
        }
        t0 = time.monotonic()
        resp = await http_client.post(
            f"{config.ollama_url}/api/generate",
            json=payload,
            timeout=config.ollama_timeout,
        )
        latency_ms = (time.monotonic() - t0) * 1000
        resp.raise_for_status()
        new_prompt = resp.json().get("response", "").strip()

        if not new_prompt or len(new_prompt) < 20:
            log.warning("Ollama returned unusable prompt mutation")
            return None

        # 1E content guard: reject a mutation that reintroduces a forbidden
        # valence/meta pattern (e.g. "take a break") — keep the current prompt.
        if _violates_forbidden_patterns(new_prompt, config):
            log.warning("Rejected prompt mutation for '%s' (forbidden pattern)", domain)
            return {"mutated": False, "rejected": "forbidden_pattern"}

        # Save with the current utility score
        pm.save_prompt(
            domain, new_prompt, score=utility_result["utility_score"], ctx=ctx
        )
        log.info(
            "Mutated prompt for '%s' (%.0fms, %d chars)",
            domain,
            latency_ms,
            len(new_prompt),
        )
        return {
            "mutated": True,
            "new_prompt_length": len(new_prompt),
            "latency_ms": round(latency_ms, 1),
        }

    except (httpx.HTTPError, httpx.ConnectError, httpx.TimeoutException) as exc:
        log.error("Prompt mutation failed: %s", exc)
        return {"mutated": False, "error": str(exc)}


# ---------------------------------------------------------------------------
# Gate offline pass (spec §9 — the SIXTH reflection analysis)
# ---------------------------------------------------------------------------


def reconstruct_state_key(row: dict) -> str:
    """Deterministically rebuild a gate ``state_key`` from a feedback row.

    A ``gate_decision_event`` already carries an authoritative ``state_key``
    (use it verbatim).  Otherwise (an ``advice_event``) rebuild it the same way
    the gate's ``build_signature`` does: a correlated event keys on its
    sorted ``involved_domains`` (``correlation:{a,b,...}``); a single event keys
    on ``single:{domain}:{entity}``.  Severity is intentionally omitted — it is
    omitted from the gate ``state_key`` too (advisor_gate.py).
    """
    sk = row.get("state_key")
    if sk:
        return sk
    if row.get("correlation_found") and row.get("involved_domains"):
        joined = ",".join(sorted(row["involved_domains"]))
        return f"correlation:{joined}"
    return f"single:{row.get('domain', 'unknown')}:{row.get('entity', 'unknown')}"


def _is_exempt_shaped(row: dict) -> bool:
    """True if the row's channel is exempt-shaped (correlated HIGH).

    Exempt signatures always FIRE (audit-only), so they must never be added to
    the self-tolerance set (spec §5 Arm 1, §9).
    """
    return bool(row.get("correlation_found")) and row.get("severity") == "high"


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    """Pearson correlation of two equal-length lists, or None if undefined.

    Undefined (returns None) when there are fewer than 2 points or either
    series has zero variance.
    """
    n = len(xs)
    if n < 2:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx <= 0.0 or syy <= 0.0:
        return None
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    return round(sxy / math.sqrt(sxx * syy), 4)


def _dedupe_feedback_by_session(rows: list[dict]) -> list[dict]:
    """De-dupe ``get_all_feedback`` rows by ``session_id`` (spec §9).

    The feedback index LPUSHes the same ``session_id`` on every intermediate
    save (persistence.py), so the cross-session list can contain duplicates.
    The index is newest-first, so the FIRST occurrence of each id is the most
    recent (final) saved state — keep it, drop the rest.
    """
    seen: set[str] = set()
    out: list[dict] = []
    for row in rows:
        sid = row.get("session_id")
        if sid in seen:
            continue
        seen.add(sid)
        out.append(row)
    return out


def _behavioral_audit(advice_rows: list[dict], config: AugurConfig) -> dict[str, Any]:
    """Reliability audit of ``behavioral_score`` vs explicit rating (spec §10).

    Computed ONLY over rows with a genuine explicit ``y``/``n`` AND
    ``behavioral_finalized`` (a no_response or an unfinalized 0.0 would
    attenuate the correlation).  Requires ``gate_behavioral_min_samples``
    genuine responses before the weight may be adjusted, and reports the
    genuine-response rate so a low rate flags weak evidence.
    """
    total = len(advice_rows)
    genuine = [
        r
        for r in advice_rows
        if r.get("explicit_rating") in ("y", "n") and r.get("behavioral_finalized")
    ]
    explicit_vals = [1.0 if r["explicit_rating"] == "y" else 0.0 for r in genuine]
    behavioral_vals = [float(r.get("behavioral_score", 0.0) or 0.0) for r in genuine]
    n = len(genuine)
    sufficient = n >= config.gate_behavioral_min_samples
    correlation = _pearson(behavioral_vals, explicit_vals) if sufficient else None
    return {
        "genuine_samples": n,
        "genuine_response_rate": round(total and n / total or 0.0, 4),
        "sufficient": sufficient,
        "correlation": correlation,
    }


def _audit_slice(rows: list[dict], config: AugurConfig) -> dict[str, Any]:
    """Reliability stats over one slice (spec §7). Genuine y/n + finalized +
    measurable + current metric version only — matching the IPW filter so the
    validation Pearson is over one homogeneous metric."""
    total = len(rows)
    genuine = [
        r
        for r in rows
        if r.get("explicit_rating") in ("y", "n")
        and r.get("behavioral_finalized")
        and not r.get("unmeasurable")
        and r.get("outcome_metric_version") == 2
    ]
    excluded_old_version = sum(
        1
        for r in rows
        if r.get("explicit_rating") in ("y", "n")
        and r.get("behavioral_finalized")
        and not r.get("unmeasurable")
        and r.get("outcome_metric_version") != 2
    )
    explicit_vals = [1.0 if r["explicit_rating"] == "y" else 0.0 for r in genuine]
    behavioral_vals = [float(r.get("behavioral_score", 0.0) or 0.0) for r in genuine]
    n = len(genuine)
    sufficient = n >= config.gate_behavioral_min_samples
    return {
        "genuine_samples": n,
        "genuine_response_rate": round(total and n / total or 0.0, 4),
        "excluded_old_version": excluded_old_version,
        "sufficient": sufficient,
        "correlation": _pearson(behavioral_vals, explicit_vals) if sufficient else None,
    }


def _behavioral_audit_per_arm(rows: list[dict], config: AugurConfig) -> dict[str, Any]:
    """Per-arm + per-domain reliability audit (spec §7). Rows carry ``_arm``
    ('fired'|'withheld') and ``domain``. Reports the overall slice plus per-arm
    and per-domain breakdowns so we can see whether the σ-space metric tracks
    felt usefulness symmetrically across arms and for non-chess domains."""
    by_arm: dict[str, list[dict]] = {}
    by_domain: dict[str, list[dict]] = {}
    for r in rows:
        by_arm.setdefault(r.get("_arm", "fired"), []).append(r)
        by_domain.setdefault(r.get("domain", "unknown"), []).append(r)
    return {
        "overall": _audit_slice(rows, config),
        "per_arm": {a: _audit_slice(rs, config) for a, rs in by_arm.items()},
        "per_domain": {d: _audit_slice(rs, config) for d, rs in by_domain.items()},
    }


def _mrt_ipw_readout(
    emissions: list[dict],
    silences: list[dict],
    fired_records: list[dict],
    withheld_records: list[dict],
) -> dict[str, Any]:
    """MRT/IPW excursion readout from persisted records ALONE (spec §9).

    Treatment (fired) arm = probe emissions that are ``mrt_eligible``, joined by
    ``decision_id`` to a fired ``advice_event`` carrying the behavioral outcome,
    inverse-probability-weighted by ``p_fire``.  Control (withheld) arm =
    ``mrt_eligible`` silences joined by ``decision_id`` to a
    ``gate_decision_event`` carrying the behavioral outcome, weighted by
    ``p_withhold``.  An ``mrt_eligible`` silence with no matching
    ``gate_decision_event`` is MRT-unobservable: excluded from the estimand and
    counted in ``unobservable_rate``.  Deterministic (non-``mrt_eligible``)
    silences are excluded entirely (not unobservable).  Honest framing: a
    low-power, directional readout — never a significance claim.
    """
    fired_by_id = {
        r["decision_id"]: r for r in fired_records if r.get("decision_id") is not None
    }
    withheld_by_id = {
        r["decision_id"]: r
        for r in withheld_records
        if r.get("decision_id") is not None
    }

    # Treatment arm: mrt_eligible probe emissions joined to fired feedback.
    fired_num = 0.0
    fired_den = 0.0
    fired_n = 0
    for em in emissions:
        if not em.get("mrt_eligible"):
            continue
        did = em.get("decision_id")
        fb = fired_by_id.get(did)
        if fb is None or not fb.get("behavioral_finalized"):
            continue
        # Exclude unmeasurable rows (forced-0.5, not a measurement) and old
        # chess-formula rows (no outcome_metric_version → incompatible) so the
        # estimand is over one homogeneous, measured outcome (spec §7).
        if fb.get("unmeasurable") or fb.get("outcome_metric_version") != 2:
            continue
        p = em.get("p_fire") or fb.get("p_fire")
        if not p or p <= 0.0:
            continue
        w = 1.0 / p
        fired_num += w * float(fb.get("behavioral_score", 0.0) or 0.0)
        fired_den += w
        fired_n += 1

    # Control arm: mrt_eligible silences joined to withheld feedback.
    withheld_num = 0.0
    withheld_den = 0.0
    withheld_n = 0
    eligible_silences = 0
    unobservable = 0
    for si in silences:
        if not si.get("mrt_eligible"):
            continue  # deterministic silence — outside the estimand entirely
        eligible_silences += 1
        did = si.get("decision_id")
        fb = withheld_by_id.get(did)
        if fb is None:
            # mrt_eligible withheld with no matching gate_decision_event tracker.
            unobservable += 1
            continue
        if not fb.get("behavioral_finalized"):
            continue
        # Same homogeneity exclusions as the fired arm, plus rated control rows
        # (non-null withheld_rating_p) which received a post-window interruption
        # and are stratified out of the primary behavioral estimand (spec §5.3).
        if fb.get("unmeasurable") or fb.get("outcome_metric_version") != 2:
            continue
        if fb.get("withheld_rating_p") is not None:
            continue
        p = si.get("p_withhold") or fb.get("p_withhold")
        if not p or p <= 0.0:
            continue
        w = 1.0 / p
        withheld_num += w * float(fb.get("behavioral_score", 0.0) or 0.0)
        withheld_den += w
        withheld_n += 1

    fired_mean = round(fired_num / fired_den, 4) if fired_den > 0 else None
    withheld_mean = round(withheld_num / withheld_den, 4) if withheld_den > 0 else None
    excursion = (
        round(fired_mean - withheld_mean, 4)
        if fired_mean is not None and withheld_mean is not None
        else None
    )
    return {
        "fired_n": fired_n,
        "withheld_n": withheld_n,
        "fired_mean_ipw": fired_mean,
        "withheld_mean_ipw": withheld_mean,
        "excursion_estimate": excursion,
        "unobservable_rate": (
            round(unobservable / eligible_silences, 4) if eligible_silences else 0.0
        ),
        "directional": True,  # honest: low-power, never a significance claim
    }


def analyze_gate(
    session_id: str,
    pm: PersistenceManager,
    config: AugurConfig,
) -> dict[str, Any]:
    """The SIXTH reflection analysis — offline gate tuning + MRT readout (§9).

    Reads a DE-DUPED ``get_all_feedback`` cross-session plus the persisted gate
    ``emissions``/``silences``, then:

    1. tunes ``self_tolerance`` membership (chronic presence AND explicit
       dismissal; exempt-shaped excluded), the per-channel habituation floor,
       per-class credibility, and the global advice-rate operating point —
       persisted atomically via ``save_gate_tuning_state``;
    2. runs the ``behavioral_score`` reliability audit (genuine y/n only);
    3. reports the MRT/IPW excursion estimate from persisted records alone.

    Idempotent via the ``gate`` ``pass_name`` marker (independent of the
    ``correlation`` marker).
    """
    if pm.is_tuning_applied(session_id, pass_name="gate"):
        log.info("Skipping gate offline pass — already applied for %s", session_id)
        return {"analysis": "gate", "skipped": True, "reason": "already_applied"}

    all_feedback = _dedupe_feedback_by_session(pm.get_all_feedback(limit=50))

    # Flatten advice + gate-decision rows across all de-duped sessions.
    advice_rows: list[dict] = []
    gate_rows: list[dict] = []
    for fb in all_feedback:
        advice_rows.extend(fb.get("advice_events", []))
        gate_rows.extend(fb.get("gate_decision_events", []))

    # ── Self-tolerance: chronic presence AND explicit dismissal ──────────────
    presence: Counter[str] = Counter()
    dismissals: Counter[str] = Counter()
    exempt_keys: set[str] = set()
    for row in advice_rows:
        sk = reconstruct_state_key(row)
        if _is_exempt_shaped(row):
            exempt_keys.add(sk)
            continue
        presence[sk] += 1
        if row.get("explicit_rating") == "n":
            dismissals[sk] += 1

    existing_tolerance = pm.load_self_tolerance()
    tolerance_add = [
        sk
        for sk in presence
        if sk not in exempt_keys
        and sk not in existing_tolerance
        and presence[sk] >= GATE_CHRONIC_MIN_PRESENCE
        and dismissals[sk] >= GATE_DISMISSAL_MIN
    ]

    # ── Habituation floor: raise the floor for chronically-dismissed channels ─
    # A dismissed channel should habituate faster, so raise its floor (which
    # caps responsiveness) one conservative step toward GATE_FLOOR_MAX.
    # Decay clocks the gate reads back as floats (gate.py `_arm_habituation` /
    # `_arm_signaller_credibility`) — must be wall-clock, never the session id.
    now_ts = time.time()

    floors: dict[str, dict] = {}
    for sk in tolerance_add:
        prev = float((pm.load_habituation_floor(sk) or {}).get("floor", 0.0) or 0.0)
        new_floor = round(min(prev + GATE_FLOOR_STEP, GATE_FLOOR_MAX), 4)
        floors[sk] = {"floor": new_floor, "last_ts": now_ts}

    # ── Class credibility: EWMA from genuine explicit ratings per class ──────
    class_ratings: dict[str, list[float]] = {}
    for row in advice_rows:
        if row.get("explicit_rating") not in ("y", "n"):
            continue
        if row.get("correlation_found"):
            cls = row.get("escalation_rule") or reconstruct_state_key(row)
        else:
            cls = f"{row.get('domain', 'unknown')}:{row.get('severity', 'unknown')}"
        class_ratings.setdefault(cls, []).append(
            1.0 if row["explicit_rating"] == "y" else 0.0
        )
    credibility: dict[str, dict] = {}
    alpha = config.gate_credibility_alpha
    for cls, vals in class_ratings.items():
        observed = sum(vals) / len(vals)
        entry = pm.load_credibility(cls) or {}
        prev = float(entry.get("cred", config.gate_cred_mid))
        n = int(entry.get("n", 0)) + len(vals)
        new_cred = round(prev + alpha * (observed - prev), 4)
        credibility[cls] = {"cred": new_cred, "n": n, "last_fb_ts": now_ts}

    # ── Dismissal rate: EWMA of how often delivered advice is rated "n" ──────
    # This writes `dismissal_ewma`, NOT `rate_ewma`. The two are different
    # quantities that shared one field: `rate_ewma` is the online gate's
    # unit-impulse EWMA of delivery *volume* (limen/gate.py), while this is the
    # fraction of delivered advice the user *rejected*. Reconciling them onto
    # one field meant whichever writer ran last defined the meaning — and since
    # reflection only runs after feedback, a system that had simply delivered a
    # lot read as a system that was being dismissed. `last_ts` is owned by the
    # online writer; the offline pass leaves it untouched.
    advice_rate_update: dict | None = None
    genuine_delivered = [
        r for r in advice_rows if r.get("explicit_rating") in ("y", "n")
    ]
    if genuine_delivered:
        dismissed = sum(1 for r in genuine_delivered if r["explicit_rating"] == "n")
        observed_rate = dismissed / len(genuine_delivered)
        prev_state = pm.load_advice_rate() or {}
        prev_rate = float(prev_state.get("dismissal_ewma", observed_rate))
        new_rate = round(
            prev_rate + GATE_ADVICE_RATE_ALPHA * (observed_rate - prev_rate), 4
        )
        advice_rate_update = {**prev_state, "dismissal_ewma": new_rate}

    # ── Behavioral audit + MRT/IPW readout ───────────────────────────────────
    audit = _behavioral_audit(advice_rows, config)
    # Per-arm + per-domain reliability audit (spec §7 — the validation
    # deliverable). Tag each row with its arm, then split fired vs withheld and
    # by domain so we can see whether the σ-metric tracks felt usefulness.
    for r in advice_rows:
        r["_arm"] = "fired"
    for r in gate_rows:
        r["_arm"] = "withheld"
    reliability_audit = _behavioral_audit_per_arm(advice_rows + gate_rows, config)
    # CL11: the readout tunes the gate, so it reads the LEARNING view of the
    # logs — a synthetic driver's emissions/silences are excluded under ENFORCE
    # (the online arms still read them unfiltered).
    mrt = _mrt_ipw_readout(
        pm.load_emissions(limit=100, learnable_only=True),
        pm.load_silence_records(limit=100, learnable_only=True),
        advice_rows,
        gate_rows,
    )

    # ── Persist atomically + mark idempotent ─────────────────────────────────
    gate_ctx = pm.resolve_learn_context(session_id)
    if floors or credibility or tolerance_add or advice_rate_update is not None:
        try:
            pm.save_gate_tuning_state(
                floors=floors or None,
                credibility=credibility or None,
                tolerance_add=tolerance_add or None,
                advice_rate=advice_rate_update,
                ctx=gate_ctx,
            )
        except redis.RedisError as exc:
            log.error("Gate tuning-state save failed: %s", exc)
            return {
                "analysis": "gate",
                "error": "save_failed",
                "reason": str(exc),
            }

    pm.mark_tuning_applied(session_id, pass_name="gate", ctx=gate_ctx)

    return {
        "analysis": "gate",
        "sessions_analyzed": len(all_feedback),
        "tolerance_added": tolerance_add,
        "floors_tuned": sorted(floors.keys()),
        "credibility_classes_tuned": sorted(credibility.keys()),
        "advice_rate": advice_rate_update,
        "behavioral_audit": audit,
        "reliability_audit": reliability_audit,
        "mrt": mrt,
    }


def run_memory_sweep(session_id: str, pm, config) -> dict:
    """Memoria 7th reflection pass (spec 2026-06-10 §6) — commit-last.

    Ingests advised-correlation patterns from this session's feedback, plans
    decay/promote/prune via the pure memoria package, and commits atomically.
    The durable augur:memoria:processed_sessions set is the authoritative
    idempotency gate AND the active-session clock (SCARD).
    """
    from memoria.fsrs import make_memory_id, normalize_severity
    from memoria.tiers import plan_sweep

    if not config.memory_store_enabled:
        return {"analysis": "memory", "skipped": True, "reason": "disabled"}
    if pm.is_session_processed(session_id):
        return {"analysis": "memory", "skipped": True, "reason": "already_processed"}

    active_session = pm.active_session_count() + 1

    feedback = pm.get_feedback(session_id) or {}
    observed: dict[str, dict] = {}
    for ev in feedback.get("advice_events", []):
        if not ev.get("correlation_found") or ev.get("rule_key") is None:
            continue  # advised correlations only (skip standalone passthrough)
        pattern = {
            "kind": "episodic",
            "domains": sorted(d.lower() for d in ev.get("involved_domains", [])),
            "rule_key": ev["rule_key"],
            "severity": normalize_severity(ev.get("severity")),
        }
        mid = make_memory_id(pattern)
        observed[mid] = {**pattern, "memory_id": mid}

    plan = plan_sweep(
        pm.load_all_memory_states(),
        list(observed.values()),
        active_session,
        session_id,
        config,
    )
    mem_ctx = pm.resolve_learn_context(session_id)
    committed = pm.apply_memory_sweep(session_id, plan, ctx=mem_ctx)
    if not committed:
        return {
            "analysis": "memory",
            "skipped": True,
            "reason": "race_already_processed",
        }
    pm.mark_tuning_applied(session_id, pass_name="memory", ctx=mem_ctx)
    return {"analysis": "memory", "active_session": active_session, **plan.counts()}


# ---------------------------------------------------------------------------
# Core reflection
# ---------------------------------------------------------------------------


async def run_reflection(
    session_id: str,
    feedback: dict,
    pm: PersistenceManager,
    redis_client: redis.Redis,
    http_client: httpx.AsyncClient,
    nc: nats.aio.client.Client,
    config: AugurConfig,
) -> dict[str, Any]:
    """Run the full reflection — every analysis pass reported under
    ``analyses`` plus the Memoria and Conscientia sweeps — and build the
    reflection report."""
    domain = _derive_domain(feedback)
    # Resolve this reflection's provenance ONCE and thread it to every learned
    # write below (spec §4.3a); a non-learnable session's tuning is withheld
    # under ENFORCE while the passes still run and report.
    learn_ctx = pm.resolve_learn_context(session_id)
    log.info(
        "Starting reflection for session %s (derived domain=%s)",
        session_id,
        domain,
    )

    # Build per-domain thresholds by walking attribution weights across all events
    domains_with_advice: set[str] = set()
    for ev in feedback.get("advice_events", []):
        domains_with_advice.update(_attribution_weights(ev).keys())
    if not domains_with_advice:
        domains_with_advice = {DEFAULT_DOMAIN}
    default_thresholds = {
        "sigma_threshold": 2.0,
        "ewma_alpha": 0.3,
        "hst_threshold": 0.7,
    }
    thresholds_per_domain: dict[str, dict] = {
        d: {**default_thresholds, **(pm.load_thresholds(d) or {})}
        for d in sorted(domains_with_advice)
    }

    # 1. Precision analysis (per-domain)
    precision = analyze_precision(feedback, thresholds_per_domain, config)

    sigma_values_after: dict[str, float] = {}
    any_sigma_adjusted = False
    for dom, result in precision["per_domain"].items():
        if result["action"] in ("raise_sigma", "lower_sigma"):
            thresholds_per_domain[dom]["sigma_threshold"] = result["sigma_after"]
            pm.save_thresholds(dom, thresholds_per_domain[dom], ctx=learn_ctx)
            any_sigma_adjusted = True
        sigma_values_after[dom] = thresholds_per_domain[dom]["sigma_threshold"]

    # Single-domain thresholds for analyze_counterfactual (and log summary)
    current_thresholds = thresholds_per_domain.get(domain, dict(default_thresholds))

    log.info(
        "Precision: %d domain(s) evaluated: %s",
        len(precision["per_domain"]),
        ", ".join(f"{d}={r['action']}" for d, r in precision["per_domain"].items())
        or "no signal",
    )

    # 2. Utility analysis
    utility = analyze_utility(feedback, config)
    log.info("Utility: %s", utility["reason"])

    # 1E: stamp the live prompt's REALIZED score (this session's utility), then
    # roll back a regression OR mutate a low-utility prompt (mutually exclusive),
    # behind a per-session marker so a re-run neither double-rolls nor re-mutates.
    mutation_result = None
    if not pm.is_tuning_applied(session_id, pass_name="prompt"):
        pm.update_current_prompt_score(domain, utility["utility_score"], ctx=learn_ctx)
        if maybe_rollback_prompt(pm, domain, config, ctx=learn_ctx):
            log.info("Auto-rolled-back regressed prompt for '%s'", domain)
        elif utility["needs_prompt_mutation"]:
            log.info(
                "Utility below %.1f — attempting prompt mutation",
                config.utility_mutation_threshold,
            )
            mutation_result = await mutate_prompt(
                pm, domain, utility, http_client, config, ctx=learn_ctx
            )
            if mutation_result and mutation_result.get("mutated"):
                log.info("Prompt mutation successful")
            else:
                log.warning("Prompt mutation skipped or failed")
        pm.mark_tuning_applied(session_id, pass_name="prompt", ctx=learn_ctx)

    # 3. Counterfactual analysis
    counterfactual = analyze_counterfactual(pm, domain, current_thresholds)
    log.info("Counterfactual: %s", counterfactual["recommendation"])

    # Step 4 + 5 — unified marker covers both correlation tuning + window tuning.
    # Single matrix load, two analyses, single merged matrix save.
    matrix_tuning: dict[str, Any]
    window_tuning: dict[str, Any]
    if pm.is_tuning_applied(session_id, pass_name="correlation"):
        log.info(
            "Skipping correlation + window tuning — already applied for session %s",
            session_id,
        )
        matrix_tuning = {
            "analysis": "correlation_tuning",
            "skipped": True,
            "reason": "already_applied_for_session",
        }
        window_tuning = {
            "analysis": "correlation_window_tuning",
            "skipped": True,
            "reason": "already_applied_for_session",
        }
    else:
        current_matrix = pm.load_escalation_matrix() or DEFAULT_ESCALATION_MATRIX
        current_confidence = pm.load_rule_confidence() or {}
        current_window_state = pm.load_rule_window_state() or {}

        matrix_tuning = analyze_correlation_tuning(
            feedback, current_matrix, current_confidence, config
        )
        log.info("Correlation tuning: %s", matrix_tuning.get("reason", "no reason"))

        window_tuning = analyze_correlation_window_tuning(
            feedback, current_matrix, current_window_state, config
        )
        log.info(
            "Correlation window tuning: %s", window_tuning.get("reason", "no reason")
        )

        matrix_changed = (
            matrix_tuning.get("new_matrix") is not None
            or window_tuning.get("new_rule_windows") is not None
        )
        matrix_save_ok = True
        if matrix_changed:
            cur_rules = dict(current_matrix.get("rules", {}))
            new_rules = (
                matrix_tuning["new_matrix"]["rules"]
                if matrix_tuning.get("new_matrix") is not None
                else cur_rules
            )
            cur_windows = dict(current_matrix.get("rule_windows", {}))
            new_windows = (
                window_tuning["new_rule_windows"]
                if window_tuning.get("new_rule_windows") is not None
                else cur_windows
            )
            # Patch ONLY changed keys through the shared CAS helper. The matrix now
            # has multiple writers (Disciplina + Imperator II); a whole-matrix
            # overwrite would clobber a concurrent II patch to OTHER rules/windows.
            changed_rules = {
                k: v for k, v in new_rules.items() if cur_rules.get(k) != v
            }
            changed_windows = {
                k: v for k, v in new_windows.items() if cur_windows.get(k) != v
            }
            res = matrix_ops.apply_matrix_update(
                pm,
                rules=changed_rules or None,
                rule_windows=changed_windows or None,
                mode="patch",
                ctx=learn_ctx,
            )
            if "error" in res:
                matrix_save_ok = False
                log.error("Merged matrix CAS update failed: %s", res["error"])
            else:
                log.info(
                    "Merged matrix updated: matrix=%s, windows=%s",
                    "yes" if matrix_tuning.get("new_matrix") else "unchanged",
                    "yes" if window_tuning.get("new_rule_windows") else "unchanged",
                )

        state_save_ok = True
        if matrix_save_ok:
            confidence_to_save = (
                matrix_tuning["new_confidence_state"]
                if matrix_tuning.get("rules_evaluated", 0) > 0
                else None
            )
            window_state_to_save = (
                window_tuning["new_window_state"]
                if window_tuning.get("rules_evaluated", 0) > 0
                else None
            )
            if confidence_to_save is not None or window_state_to_save is not None:
                try:
                    pm.save_tuning_state(
                        confidence=confidence_to_save,
                        window_state=window_state_to_save,
                        ctx=learn_ctx,
                    )
                except redis.RedisError as exc:
                    state_save_ok = False
                    log.error("Atomic tuning-state save failed: %s", exc)
        else:
            state_save_ok = False
            log.warning(
                "Skipping atomic tuning-state save because matrix save failed; "
                "next reflection trigger will redo the EWMA update against the "
                "original (unchanged) state."
            )

        if (
            (
                matrix_tuning.get("rules_evaluated", 0) > 0
                or window_tuning.get("rules_evaluated", 0) > 0
            )
            and matrix_save_ok
            and state_save_ok
        ):
            pm.mark_tuning_applied(session_id, pass_name="correlation", ctx=learn_ctx)
        elif not (matrix_save_ok and state_save_ok):
            log.warning(
                "Skipping mark_tuning_applied because at least one tuning write failed; "
                "next reflection trigger will retry."
            )

    # 6. Gate offline pass (spec §9) — own idempotency marker (pass_name="gate"),
    # independent of the correlation/window marker above.
    try:
        gate = analyze_gate(session_id, pm, config)
        log.info(
            "Gate offline pass: %s",
            gate.get("reason")
            or f"{len(gate.get('tolerance_added', []))} tolerance add(s), "
            f"mrt fired={gate.get('mrt', {}).get('fired_n')} "
            f"withheld={gate.get('mrt', {}).get('withheld_n')}",
        )
    except redis.RedisError as exc:
        gate = {"analysis": "gate", "error": "exception", "reason": str(exc)}
        log.error("Gate offline pass failed: %s", exc)

    # 7. Memory spine pass (Lane 2, spec 2026-06-10) — own idempotency via the
    # durable processed_sessions set; independent of the other markers.
    try:
        memory = run_memory_sweep(session_id, pm, config)
        log.info(
            "Memory sweep: %s",
            memory.get("reason")
            or f"created={memory.get('created')} reviewed={memory.get('reviewed')} "
            f"promoted={memory.get('promoted')} archived={memory.get('archived')}",
        )
    except redis.RedisError as exc:
        memory = {"analysis": "memory", "error": "exception", "reason": str(exc)}
        log.error("Memory sweep failed: %s", exc)

    # 8. Conscientia review pass (Task 11) — offline audit of any GATED
    # proposals Imperator II has raised since the last reflection. The
    # auditor self-gates on config.conscientia_enabled (returns the
    # zero-shape result when disabled), so it is called unconditionally here
    # — same try/except discipline as the Memory spine pass above.
    try:
        from conscientia.auditor import run_conscientia_review

        conscientia = await run_conscientia_review(pm, nc, config)
        log.info(
            "Conscientia review: %d proposal(s) reviewed, recommendations=%s",
            conscientia.get("reviewed", 0),
            conscientia.get("recommendations"),
        )
    except Exception as exc:
        conscientia = {"error": str(exc)}
        log.warning("Conscientia review failed (non-fatal): %s", exc)

    # 9. Praesagium mining pass — cross-session A→B pattern mining (spec 2026-07-09).
    try:
        from praesagium.miner import run_praesagium_mining

        praesagium = run_praesagium_mining(session_id, pm, config)
        log.info(
            "Praesagium mining: %s",
            praesagium.get("reason")
            or f"active={praesagium.get('active', 0)} retired={praesagium.get('retired', 0)}",
        )
    except Exception as exc:
        praesagium = {"error": str(exc)}
        log.warning("Praesagium mining failed (non-fatal): %s", exc)

    # Build report
    report = {
        "session_id": session_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "analyses": {
            "precision": precision,
            "utility": utility,
            "counterfactual": counterfactual,
            "correlation_tuning": matrix_tuning,
            "correlation_window_tuning": window_tuning,
            "gate": gate,
            "memory": memory,
        },
        "conscientia": conscientia,
        "praesagium": praesagium,
        "adjustments": {
            "sigma_adjusted": any_sigma_adjusted,
            "sigma_values": sigma_values_after,
            "prompt_mutated": (
                mutation_result is not None and mutation_result.get("mutated", False)
            ),
            "matrix_mutated": matrix_tuning.get("new_matrix") is not None,
            "windows_tuned": window_tuning.get("new_rule_windows") is not None,
        },
    }

    if mutation_result:
        report["adjustments"]["mutation_details"] = mutation_result

    # Persist via PersistenceManager (ARCH-03: was a bare redis_client.set
    # call that bypassed the persistence abstraction).
    try:
        pm.save_reflection(session_id, report)
        log.info("Saved reflection report for session %s", session_id)
    except redis.RedisError as exc:
        log.error("Failed to persist reflection: %s", exc)

    # Publish completion event
    try:
        await nc.publish(
            SUBJECT_REFLECT_COMPLETE,
            json.dumps(report).encode(),
        )
        log.info("Published reflection to %s", SUBJECT_REFLECT_COMPLETE)
    except Exception as exc:
        log.error("Failed to publish reflection: %s", exc)

    return report


# ---------------------------------------------------------------------------
# Core loop
# ---------------------------------------------------------------------------


async def run() -> None:
    config = AugurConfig.from_env()

    redis_client = connect_redis(config)
    pm = PersistenceManager(redis_client)

    nc = await nats.connect(
        config.nats_url, connect_timeout=config.nats_connect_timeout
    )
    hb_task = (
        start_heartbeat(nc, "disciplina", config.praefectus_heartbeat_interval_s)
        if config.praefectus_enabled
        else None
    )
    log.info("NATS connected (%s)", config.nats_url)

    http_client = httpx.AsyncClient()

    # Check Ollama reachability (non-fatal)
    try:
        probe = await http_client.get(f"{config.ollama_url}/api/tags", timeout=5)
        probe.raise_for_status()
        log.info("Ollama reachable at %s", config.ollama_url)
    except (httpx.HTTPError, httpx.ConnectError) as exc:
        log.warning(
            "Ollama not reachable at startup (%s) — prompt mutation will be skipped",
            exc,
        )

    reflection_lock = asyncio.Lock()

    async def on_feedback_complete(msg: nats.aio.client.Msg) -> None:
        try:
            data = json.loads(msg.data.decode())
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            log.warning("Bad feedback complete payload: %s", exc)
            return

        session_id = data.get("session_id", "unknown")

        if reflection_lock.locked():
            log.warning(
                "Reflection already in progress, skipping session %s", session_id
            )
            return

        async with reflection_lock:
            # Load full feedback from Redis
            feedback = pm.get_feedback(session_id)
            if feedback is None:
                log.warning("No feedback found for session %s", session_id)
                return

            await run_reflection(
                session_id, feedback, pm, redis_client, http_client, nc, config
            )

    async def on_trigger(msg: nats.aio.client.Msg) -> None:
        """Manual trigger — accepts {session_id} or runs on latest session."""
        try:
            data = json.loads(msg.data.decode())
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            log.debug("Trigger payload unparseable, using empty: %s", exc)
            data = {}

        session_id = data.get("session_id")

        if session_id is None:
            # Try the most recent session from feedback index
            all_fb = pm.get_all_feedback(limit=1)
            if all_fb:
                session_id = all_fb[0].get("session_id", "unknown")
            else:
                log.warning("No sessions found for manual trigger")
                return

        if reflection_lock.locked():
            log.warning(
                "Reflection already in progress, skipping trigger for %s", session_id
            )
            return

        async with reflection_lock:
            feedback = pm.get_feedback(session_id)
            if feedback is None:
                log.warning("No feedback found for session %s", session_id)
                return

            await run_reflection(
                session_id, feedback, pm, redis_client, http_client, nc, config
            )

    # LEAK-06: save subscription handles so we can unsubscribe on shutdown
    # rather than relying on nc.close() to tear them down abruptly.
    sub_feedback = await nc.subscribe(
        SUBJECT_FEEDBACK_COMPLETE, cb=on_feedback_complete
    )
    sub_trigger = await nc.subscribe(SUBJECT_REFLECT_TRIGGER, cb=on_trigger)

    log.info(
        "Subscribed to: %s, %s", SUBJECT_FEEDBACK_COMPLETE, SUBJECT_REFLECT_TRIGGER
    )
    log.info("Waiting for session feedback or manual trigger...")

    try:
        while True:
            await asyncio.sleep(1)
    except asyncio.CancelledError:
        pass
    finally:
        if hb_task is not None:
            hb_task.cancel()
            try:
                await hb_task
            except asyncio.CancelledError:
                pass
        # LEAK-06: close NATS first (stop delivering messages to callbacks)
        # before closing the HTTP client so an in-flight reflection cannot
        # see a mid-shutdown "client is closed" error from Ollama.
        try:
            await sub_feedback.unsubscribe()
            await sub_trigger.unsubscribe()
        except Exception as exc:
            log.debug("Unsubscribe failed during shutdown: %s", exc)
        await nc.close()
        await http_client.aclose()
        log.info("Shut down cleanly")


def main() -> None:
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        log.info("Interrupted")


if __name__ == "__main__":
    main()
