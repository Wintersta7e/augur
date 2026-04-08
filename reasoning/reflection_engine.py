"""Session reflection engine — self-adjusts Augur parameters after each session.

Triggers on augur.feedback.complete (end of feedback collection) or
augur.reflect.trigger (manual). Runs three analyses:

1. Precision  — Were anomaly detections accurate? Adjusts sigma threshold.
2. Utility    — Was the advice useful? May mutate LLM prompt via Ollama.
3. Counterfactual — Would +-10% threshold variants have been better?

Publishes a reflection report to augur.reflect.complete and persists
it in Redis at augur:reflect:<session_id>.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx
import nats
import redis

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from blackboard.config import AugurConfig
from blackboard.persistence import PersistenceManager

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

# Default domain for chess
DEFAULT_DOMAIN = "chess"

# ---------------------------------------------------------------------------
# Precision analysis
# ---------------------------------------------------------------------------


def analyze_precision(
    feedback: dict,
    current_thresholds: dict,
    config: AugurConfig,
) -> dict:
    """Evaluate detection precision from session feedback.

    Looks at how many anomalies were fired vs how many received positive
    explicit feedback or high behavioral scores.
    """
    advice_events = feedback.get("advice_events", [])
    summary = feedback.get("session_summary", {})

    total = summary.get("total_advice", len(advice_events))
    if total == 0:
        return {
            "analysis": "precision",
            "total_anomalies": 0,
            "escalated": 0,
            "precision_ratio": 1.0,
            "action": "none",
            "reason": "No anomalies this session",
            "sigma_before": current_thresholds.get("sigma_threshold", 2.0),
            "sigma_after": current_thresholds.get("sigma_threshold", 2.0),
        }

    # Count "useful" detections: explicit positive OR high behavioral score
    useful = 0
    for ev in advice_events:
        if ev.get("explicit_rating") == "y":
            useful += 1
        elif ev.get("behavioral_score", 0) >= 0.7:
            useful += 1

    precision = useful / total if total > 0 else 0.0
    sigma_before = current_thresholds.get("sigma_threshold", 2.0)
    sigma_after = sigma_before
    action = "none"
    reason = f"Precision {precision:.0%} is acceptable"

    if precision < 0.3 and total >= 2:
        # Too many false positives — raise threshold
        sigma_after = min(sigma_before + config.sigma_adjust_step, config.sigma_max)
        action = "raise_sigma"
        reason = (
            f"Low precision ({precision:.0%}): {useful}/{total} useful. "
            f"Raising sigma {sigma_before:.1f} -> {sigma_after:.1f}"
        )
    elif precision > 0.8 and total >= 2:
        # Very high precision — could lower threshold to catch more
        sigma_after = max(sigma_before - config.sigma_adjust_step, config.sigma_min)
        action = "lower_sigma"
        reason = (
            f"High precision ({precision:.0%}): {useful}/{total} useful. "
            f"Lowering sigma {sigma_before:.1f} -> {sigma_after:.1f}"
        )

    return {
        "analysis": "precision",
        "total_anomalies": total,
        "escalated": useful,
        "precision_ratio": round(precision, 3),
        "action": action,
        "reason": reason,
        "sigma_before": sigma_before,
        "sigma_after": sigma_after,
    }


# ---------------------------------------------------------------------------
# Utility analysis
# ---------------------------------------------------------------------------


def analyze_utility(feedback: dict, config: AugurConfig) -> dict:
    """Evaluate advice utility from explicit + behavioral signals.

    Weighted score: 60% explicit, 40% behavioral.
    If utility < config.utility_mutation_threshold, flags for prompt mutation.
    """
    advice_events = feedback.get("advice_events", [])
    summary = feedback.get("session_summary", {})

    total = summary.get("total_advice", len(advice_events))
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
) -> dict:
    """Execute all three analyses and build the reflection report."""
    domain = DEFAULT_DOMAIN
    log.info("Starting reflection for session %s", session_id)

    # Load current thresholds
    stored_thresholds = pm.load_thresholds(domain)
    current_thresholds = {
        "sigma_threshold": 2.0,
        "ewma_alpha": 0.3,
        "hst_threshold": 0.7,
        **(stored_thresholds or {}),
    }

    # 1. Precision analysis
    precision = analyze_precision(feedback, current_thresholds, config)
    log.info("Precision: %s", precision["reason"])

    # Apply sigma adjustment if needed
    if precision["action"] in ("raise_sigma", "lower_sigma"):
        current_thresholds["sigma_threshold"] = precision["sigma_after"]
        pm.save_thresholds(domain, current_thresholds)
        log.info(
            "Saved updated sigma threshold: %.1f -> %.1f",
            precision["sigma_before"],
            precision["sigma_after"],
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

    # Build report
    report = {
        "session_id": session_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "analyses": {
            "precision": precision,
            "utility": utility,
            "counterfactual": counterfactual,
        },
        "adjustments": {
            "sigma_adjusted": precision["action"] != "none",
            "sigma_value": precision["sigma_after"],
            "prompt_mutated": mutation_result is not None
            and mutation_result.get("mutated", False),
        },
    }

    if mutation_result:
        report["adjustments"]["mutation_details"] = mutation_result

    # Persist to Redis
    reflect_key = f"augur:reflect:{session_id}"
    try:
        redis_client.set(reflect_key, json.dumps(report))
        log.info("Saved reflection to %s", reflect_key)
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

    redis_client = redis.Redis(
        host=config.redis_host,
        port=config.redis_port,
        socket_connect_timeout=config.redis_connect_timeout,
    )
    redis_client.ping()
    log.info("Redis connected (%s)", config.redis_url)

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
        except (json.JSONDecodeError, UnicodeDecodeError):
            log.warning("Bad feedback complete payload")
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
        except (json.JSONDecodeError, UnicodeDecodeError):
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

    await nc.subscribe(SUBJECT_FEEDBACK_COMPLETE, cb=on_feedback_complete)
    await nc.subscribe(SUBJECT_REFLECT_TRIGGER, cb=on_trigger)

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
        await http_client.aclose()
        await nc.close()
        log.info("Shut down cleanly")


def main() -> None:
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        log.info("Interrupted")


if __name__ == "__main__":
    main()
