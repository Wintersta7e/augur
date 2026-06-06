"""Session reflection engine — self-adjusts Augur parameters after each session.

Triggers on augur.feedback.complete (end of feedback collection) or
augur.reflect.trigger (manual). Runs four analyses:

1. Precision  — Were anomaly detections accurate? Adjusts sigma threshold.
2. Utility    — Was the advice useful? May mutate LLM prompt via Ollama.
3. Counterfactual — Would +-10% threshold variants have been better?
4. Correlation tuning — Per-rule EWMA confidence with hysteresis to
   tune the cross-domain escalation matrix.

Publishes a reflection report to augur.reflect.complete and persists
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
from blackboard.config import AugurConfig
from blackboard.connections import connect_redis
from blackboard.persistence import PersistenceManager
from reasoning.correlator import DEFAULT_ESCALATION_MATRIX

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
SUBJECT_FEEDBACK_COMPLETE = "augur.feedback.complete"
SUBJECT_REFLECT_TRIGGER = "augur.reflect.trigger"
SUBJECT_REFLECT_COMPLETE = "augur.reflect.complete"

# Fallback domain used only when a session's feedback contains no usable
# standalone advice events. Per-session domain is now derived from
# feedback.advice_events instead of being hardcoded (ARCH-06 fix). Previous
# behaviour silently applied chess-domain threshold and prompt tuning to
# typing-only sessions.
DEFAULT_DOMAIN = "chess"


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
    """
    advice_events = feedback.get("advice_events", [])
    standalone_domains = [
        ev.get("domain")
        for ev in advice_events
        if ev.get("domain") and not ev.get("correlation_found")
    ]
    if standalone_domains:
        return Counter(standalone_domains).most_common(1)[0][0]
    # Round-3 fallback: post-Task-8, correlated records have a real domain too.
    all_domains = [ev.get("domain") for ev in advice_events if ev.get("domain")]
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

    for ev in feedback.get("advice_events", []):
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
    """
    # Filter out correlated advice before computing the score that drives
    # prompt mutation. The correlation path is tuned by the matrix tuning
    # analysis in a later step of run_reflection, not by prompt mutation.
    all_events = feedback.get("advice_events", [])
    advice_events = [e for e in all_events if not e.get("correlation_found")]

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

    # Behavioral component
    behavioral_scores = [
        ev.get("behavioral_score", 0.5)
        for ev in advice_events
        if ev.get("behavioral_score", 0) > 0
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

        behavioral_scores = [
            ev.get("behavioral_score", 0.0)
            for ev in events
            if ev.get("behavioral_score", 0.0) > 0
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


async def mutate_prompt(
    pm: PersistenceManager,
    domain: str,
    utility_result: dict,
    http_client: httpx.AsyncClient,
    config: AugurConfig,
) -> dict | None:
    """Ask Ollama to suggest a better system prompt based on feedback."""
    current_prompt = pm.load_prompt(domain)
    if current_prompt is None:
        # No managed prompt yet — create a seed
        current_prompt = (
            "You are a chess analyst reviewing a game in progress. "
            "Provide concise, actionable advice about timing anomalies."
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

        # Save with the current utility score
        pm.save_prompt(domain, new_prompt, score=utility_result["utility_score"])
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
    """Execute all four analyses and build the reflection report."""
    domain = _derive_domain(feedback)
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
            pm.save_thresholds(dom, thresholds_per_domain[dom])
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

    # Prompt mutation if utility is low
    mutation_result = None
    if utility["needs_prompt_mutation"]:
        log.info(
            "Utility below %.1f — attempting prompt mutation",
            config.utility_mutation_threshold,
        )
        mutation_result = await mutate_prompt(pm, domain, utility, http_client, config)
        if mutation_result and mutation_result.get("mutated"):
            log.info("Prompt mutation successful")
        else:
            log.warning("Prompt mutation skipped or failed")

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
            merged: dict[str, Any] = {
                "version": current_matrix.get("version", "1.0"),
                "rules": (
                    matrix_tuning["new_matrix"]["rules"]
                    if matrix_tuning.get("new_matrix") is not None
                    else dict(current_matrix.get("rules", {}))
                ),
                "rule_windows": (
                    window_tuning["new_rule_windows"]
                    if window_tuning.get("new_rule_windows") is not None
                    else dict(current_matrix.get("rule_windows", {}))
                ),
            }
            try:
                pm.save_escalation_matrix(merged)
                log.info(
                    "Merged matrix updated: matrix=%s, windows=%s",
                    "yes" if matrix_tuning.get("new_matrix") else "unchanged",
                    "yes" if window_tuning.get("new_rule_windows") else "unchanged",
                )
            except redis.RedisError as exc:
                matrix_save_ok = False
                log.error("Merged matrix save failed: %s", exc)

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
            pm.mark_tuning_applied(session_id, pass_name="correlation")
        elif not (matrix_save_ok and state_save_ok):
            log.warning(
                "Skipping mark_tuning_applied because at least one tuning write failed; "
                "next reflection trigger will retry."
            )

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
        },
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
