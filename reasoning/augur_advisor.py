"""Multi-domain LLM advisor triggered by anomaly detection.

Subscribes to NATS 'augur.detection.anomaly', routes to domain-specific
prompt builders, queries Ollama for advice, and publishes results.
Only activates for medium/high severity anomalies.

Supports:
  - chess: board context from Redis, strategic advice
  - typing: cognitive load analysis, productivity suggestions
  - (new domains: add a prompt builder + register in DOMAIN_HANDLERS)
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
import time
from collections.abc import Callable  # noqa: F401 — PEP-563 deferred annotations
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
import nats
import redis

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from blackboard.config import AugurConfig
from blackboard.connections import connect_redis
from blackboard.persistence import PersistenceManager
from blackboard.session import get_active_session
from reasoning.advisor_gate import (
    Gate,
    GateDecision,
    Signature,  # noqa: F401 — PEP-563 deferred annotation (annotations only)
    build_signature,
)
from reasoning.advisor_gate_scheduler import MustFireScheduler
from reasoning.app_descriptor import (
    ACTIVITY_DOMAINS,
    ClassifierLane,
    classifier_model_available,
    descriptor_suffix,
    resolve_app_descriptor,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("augur_advisor")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SUBSCRIBE_SUBJECT = "augur.correlation.detected"
PUBLISH_SUBJECT = "augur.reasoning.advice"

# Gate visibility subjects (spec §8). Distinct subjects so the MRT control arm
# (PendingGateDecision, subscribed only to SUBJECT_SUPPRESSED) never tracks an
# infrastructure non-delivery.
SUBJECT_SUPPRESSED = "augur.advisor.suppressed"
SUBJECT_DELIVERY_FAILURE = "augur.advisor.delivery_failure"

REDIS_KEY_LAST_MOVE = "augur:chess:last_move"
REDIS_KEY_HISTORY = "augur:chess:move_history"
REDIS_KEY_ADVICE = "augur:reasoning:last_advice"

SEVERITY_GATE = {"medium", "high"}

# ---------------------------------------------------------------------------
# Default prompts per domain (used when PersistenceManager has no stored prompt)
# ---------------------------------------------------------------------------

DEFAULT_PROMPTS = {
    "chess": (
        "You are a chess analyst reviewing a game in progress. "
        "A timing anomaly has been detected. Provide concise, actionable "
        "chess advice based on the position and timing pattern."
    ),
    "typing": (
        "You are a cognitive load analyst monitoring a user's typing patterns. "
        "A significant pause or rhythm anomaly has been detected. Analyze what "
        "this might indicate about their mental state and provide a helpful suggestion."
    ),
}


def resolve_advisor_path(payload: dict) -> str:
    """Return 'correlation' if payload.correlation_found else 'single'."""
    return "correlation" if payload.get("correlation_found") else "single"


# ---------------------------------------------------------------------------
# Redis helpers
# ---------------------------------------------------------------------------


def read_board_context(r: redis.Redis) -> tuple[dict | None, list[dict]]:
    """Return (last_move, move_history) from Redis."""
    last_raw = r.get(REDIS_KEY_LAST_MOVE)
    last_move = json.loads(last_raw) if last_raw else None

    history_raw = r.lrange(REDIS_KEY_HISTORY, 0, -1)
    history = [json.loads(entry) for entry in history_raw]
    # lrange returns newest-first; reverse for chronological order
    history.reverse()
    return last_move, history


# ---------------------------------------------------------------------------
# Chess prompt builder
# ---------------------------------------------------------------------------


def build_chess_prompt(
    anomaly: dict,
    r: redis.Redis,
    system_prompt: str,
) -> str:
    """Build an LLM prompt with chess-specific game context."""
    player = anomaly.get("player", anomaly.get("entity", "?"))
    think_time = anomaly.get("think_time", anomaly.get("value", 0))
    baseline_mean = anomaly.get("baseline_mean", "?")
    deviation = anomaly.get("deviation_score", "?")
    severity = anomaly.get("severity", "?")

    # Gather board context
    last_move, history = read_board_context(r)

    history_lines: list[str] = []
    for m in history:
        p = m.get("player", "?")
        san = m.get("move_san", "?")
        t = m.get("think_time_seconds", "?")
        num = m.get("move_number", "?")
        prefix = f"{num}." if p == "white" else f"{num}..."
        history_lines.append(f"  {prefix} {san} ({p}, {t}s)")

    history_block = (
        "\n".join(history_lines) if history_lines else "  (no history available)"
    )

    current_move = "unknown"
    if last_move:
        current_move = (
            f"{last_move.get('move_san', '?')} (UCI: {last_move.get('move_uci', '?')})"
        )

    return f"""{system_prompt}

## Situation
- **Player struggling:** {player}
- **Current move:** {current_move}
- **Think time:** {think_time}s (baseline average: {baseline_mean}s)
- **Deviation:** {deviation} standard deviations from normal
- **Severity:** {severity}

## Recent move history (with think times)
{history_block}

## Your task
1. Based on the move sequence, what is the likely board position and game phase (opening/middlegame/endgame)?
2. Why might {player} be taking significantly {"longer" if think_time > float(str(baseline_mean)) else "shorter"} than usual on this move?
3. What strategic challenges might {player} be facing at this point?
4. Provide one concrete piece of advice for {player}.

Keep your response concise (3-5 sentences). Focus on actionable chess insight, not generic advice."""


# ---------------------------------------------------------------------------
# Typing prompt builder
# ---------------------------------------------------------------------------


def build_typing_prompt(
    anomaly: dict,
    _r: redis.Redis,
    system_prompt: str,
) -> str:
    """Build an LLM prompt focused on cognitive load from typing patterns."""
    value = anomaly.get("value", 0)
    unit = anomaly.get("unit", "seconds")
    event_type = anomaly.get("event_type", "pause")
    baseline_mean = anomaly.get("baseline_mean", "?")
    deviation = anomaly.get("deviation_score", "?")
    severity = anomaly.get("severity", "?")
    ctx = anomaly.get("context", {})

    avg_wpm = ctx.get("avg_wpm", "?")
    keypress_count = ctx.get("keypress_count", "?")
    pause_position = ctx.get("pause_position", "?")

    event_desc = "a long pause" if event_type == "pause" else "an unusual rhythm change"

    return f"""{system_prompt}

## Situation
- **Event:** {event_desc}
- **Value:** {value} {unit} (baseline: {baseline_mean})
- **Deviation:** {deviation} standard deviations from normal
- **Severity:** {severity}
- **Typing speed:** {avg_wpm} WPM
- **Total keypresses this session:** {keypress_count}
- **Keypresses since last pause:** {pause_position}

## Your task
1. Interpret the typing pattern. Consider all plausible explanations:
   - A normal variation that simply differs from the per-user baseline (most common).
   - Deep focused thought, composing, or reading before typing.
   - A genuine pause for an unrelated reason (interruption, stretch).
   - Being stuck, distracted, or fatigued.
   Default to "normal variation" unless other context clearly points elsewhere.
   Baselines form in only a few observations, so early sessions over-report.
2. Only if a brief, gentle intervention is clearly warranted, suggest one.
   Otherwise say none is needed.

Keep your response concise (2-3 sentences). Be supportive, not intrusive."""


# ---------------------------------------------------------------------------
# Activity focus and intensity prompt builders
# ---------------------------------------------------------------------------


def build_activity_focus_prompt(
    anomaly: dict,
    _r: "redis.Redis",
    system_prompt: str,
) -> str:
    """Build an LLM prompt for an activity-focus dwell anomaly."""
    ctx = anomaly.get("context", {}) or {}
    entity = anomaly.get("entity")
    new_app = ctx.get("new_app")
    active_dwell = ctx.get("active_dwell_s")
    baseline_mean = anomaly.get("baseline_mean")
    deviation = anomaly.get("deviation_score")
    severity = anomaly.get("severity")

    missing = [
        k
        for k, v in (
            ("entity", entity),
            ("context.new_app", new_app),
            ("context.active_dwell_s", active_dwell),
            ("baseline_mean", baseline_mean),
            ("deviation_score", deviation),
            ("severity", severity),
        )
        if v is None
    ]
    if missing:
        raise ValueError(
            f"build_activity_focus_prompt: missing required fields: {missing}"
        )

    idle_dwell = ctx.get("idle_dwell_s", 0.0)
    total_dwell = ctx.get("total_dwell_s", active_dwell)

    return f"""{system_prompt}

## Situation
- **Event:** unusual dwell in {entity}{descriptor_suffix(ctx)} before switching to {new_app}
- **Active dwell:** {active_dwell}s   (idle: {idle_dwell}s, total: {total_dwell}s)
- **Baseline (log1p s):** {baseline_mean}
- **Deviation:** {deviation} standard deviations
- **Severity:** {severity}

## Your task
1. Interpret the dwell pattern. Consider all plausible explanations:
   - A normal task duration that simply differs from the per-app baseline (most common reason).
   - A varied task — e.g., long-form reading, watching, drafting — that genuinely takes longer.
   - Deep focused work.
   - Being stuck on a problem.
   - Distraction or context switching.
   Default to "normal varied task duration" unless other context clearly points elsewhere. Baselines form in only a few observations, so early sessions over-report.
2. Only if a brief, gentle intervention is clearly warranted, suggest one. Otherwise say none is needed.

Keep your response concise (2-3 sentences). Be supportive, not intrusive."""


def build_activity_intensity_prompt(
    anomaly: dict,
    _r: "redis.Redis",
    system_prompt: str,
) -> str:
    """Build an LLM prompt for an activity-intensity spike or drop."""
    ctx = anomaly.get("context", {}) or {}
    entity = anomaly.get("entity", "?")
    value = anomaly.get("value", "?")
    keystrokes = ctx.get("keystroke_count", "?")
    mouse = ctx.get("mouse_event_count", "?")
    idle = ctx.get("idle_seconds", "?")
    window = ctx.get("window_duration_s", "?")
    baseline_mean = anomaly.get("baseline_mean", "?")
    deviation = anomaly.get("deviation_score", "?")
    severity = anomaly.get("severity", "?")

    return f"""{system_prompt}

## Situation
- **Event:** unusual interaction intensity in {entity}{descriptor_suffix(ctx)}
- **Rate:** {value} interactions/min   (baseline: {baseline_mean})
- **Breakdown:** {keystrokes} keystrokes + {mouse} clicks/scrolls over {window}s ({idle}s idle)
- **Deviation:** {deviation} standard deviations
- **Severity:** {severity}

## Your task
1. Interpret the pattern. Weigh the keystrokes and clicks breakdown:
   - Click-heavy with few keystrokes → likely normal browsing, scrolling, gaming, or media use of a click-driven app.
   - Typing-heavy → focused writing, coding, or messaging.
   - Both elevated → engaged work.
   Default to "normal heavy use of this app" unless the breakdown or context clearly points to fatigue, automation, or distraction. Note: deviation alone is not enough — every new app's baseline forms in only a few observations, so early sessions over-report.
2. Only if a brief, gentle intervention is clearly warranted, suggest one. Otherwise say none is needed.

Keep your response concise (2-3 sentences). Be supportive, not intrusive."""


# ---------------------------------------------------------------------------
# Generic prompt builder (fallback for unknown domains)
# ---------------------------------------------------------------------------


def build_generic_prompt(
    anomaly: dict,
    _r: redis.Redis,
    system_prompt: str,
) -> str:
    """Fallback prompt for domains without a specialized builder."""
    domain = anomaly.get("domain", "unknown")
    entity = anomaly.get("entity", "?")
    value = anomaly.get("value", 0)
    unit = anomaly.get("unit", "")
    baseline_mean = anomaly.get("baseline_mean", "?")
    deviation = anomaly.get("deviation_score", "?")
    severity = anomaly.get("severity", "?")
    ctx = anomaly.get("context", {})

    return f"""{system_prompt}

## Anomaly detected in domain: {domain}
- **Entity:** {entity}
- **Value:** {value} {unit} (baseline: {baseline_mean})
- **Deviation:** {deviation} standard deviations
- **Severity:** {severity}
- **Context:** {json.dumps(ctx, indent=2)}

## Your task
1. Interpret the anomaly. The most common cause is normal variation from a
   baseline formed in only a few observations (early sessions over-report).
   Weigh benign explanations before concerning ones.
2. Only if a brief, gentle intervention is clearly warranted, suggest one.
   Otherwise say none is needed.

Keep your response concise (2-3 sentences). Be supportive, not intrusive."""


# ---------------------------------------------------------------------------
# Lightweight per-domain one-liner formatter (used by correlation prompts)
# ---------------------------------------------------------------------------


def _describe_chess(anomaly: dict) -> str:
    move = anomaly.get("context", {}).get("move_san") or anomaly.get("move", "?")
    think = anomaly.get("value", anomaly.get("think_time", 0))
    baseline = anomaly.get("baseline_mean", "?")
    deviation = anomaly.get("deviation_score", "?")
    return (
        f"CHESS (timing): {anomaly.get('entity', '?')} paused {think}s on "
        f"move {move}. Baseline: {baseline}s. Deviation: {deviation}σ."
    )


def _describe_typing(anomaly: dict) -> str:
    pause = anomaly.get("value", 0)
    unit = anomaly.get("unit", "seconds")
    ctx = anomaly.get("context", {})
    avg_wpm = ctx.get("avg_wpm", "?")
    baseline = anomaly.get("baseline_mean", "?")
    return (
        f"TYPING (rhythm): Pause duration {pause}{unit[:1]}. "
        f"Average speed {avg_wpm} wpm. Baseline pause: {baseline}s."
    )


def _describe_activity_focus(anomaly: dict) -> str:
    ctx = anomaly.get("context", {}) or {}
    entity = anomaly.get("entity", "?")
    new_app = ctx.get("new_app", "?")
    active = ctx.get("active_dwell_s", "?")
    baseline = anomaly.get("baseline_mean", "?")
    deviation = anomaly.get("deviation_score", "?")
    return (
        f"ACTIVITY_FOCUS: {entity}{descriptor_suffix(ctx)} dwell {active}s (then switched to {new_app}). "
        f"Baseline (log1p): {baseline}. Deviation: {deviation}σ."
    )


def _describe_activity_intensity(anomaly: dict) -> str:
    ctx = anomaly.get("context", {}) or {}
    entity = anomaly.get("entity", "?")
    value = anomaly.get("value", "?")
    keystrokes = ctx.get("keystroke_count", "?")
    baseline = anomaly.get("baseline_mean", "?")
    deviation = anomaly.get("deviation_score", "?")
    return (
        f"ACTIVITY_INTENSITY: {entity}{descriptor_suffix(ctx)} {value} ipm "
        f"(keystrokes={keystrokes}). Baseline: {baseline}. Deviation: {deviation}σ."
    )


DOMAIN_DESCRIBERS: dict[str, Callable[[dict], str]] = {
    "chess": _describe_chess,
    "typing": _describe_typing,
    "activity_focus": _describe_activity_focus,
    "activity_intensity": _describe_activity_intensity,
}


def describe_signal(domain: str, anomaly: dict) -> str:
    """Return a one-line human-readable summary of a single-domain anomaly.

    Used inside build_correlation_prompt to embed each domain's contribution
    without rebuilding a full domain-specific prompt. Does not share code
    with the full prompt builders — different purpose, different format.
    """
    describer = DOMAIN_DESCRIBERS.get(domain)
    if describer:
        return describer(anomaly)

    # Generic fallback
    value = anomaly.get("value", "?")
    unit = anomaly.get("unit", "")
    deviation = anomaly.get("deviation_score", "?")
    return (
        f"{domain.upper()}: {anomaly.get('event_type', 'event')} "
        f"value={value}{unit}  deviation={deviation}\u03c3"
    )


def enrich_payload_descriptors(
    pm: PersistenceManager, lane: ClassifierLane, path: str, payload: dict
) -> None:
    """Enrich the primary anomaly (always) and, for a correlation payload, each
    correlated event, with app descriptors before prompt-building."""
    enrich_activity_descriptor(pm, lane, payload["primary_anomaly"])
    if path == "correlation":
        for correlated_ev in payload.get("correlated_events", []):
            enrich_activity_descriptor(pm, lane, correlated_ev)


def enrich_activity_descriptor(
    pm: PersistenceManager, lane: ClassifierLane, anomaly: dict
) -> None:
    """Resolve an activity anomaly's descriptor into context['app_descriptor'].

    OS identity is cached + used immediately; a cache miss enqueues a background
    classification. No-op for non-activity domains. Mutates ``anomaly`` in place.
    """
    if anomaly.get("domain") not in ACTIVITY_DOMAINS:
        return
    ctx = anomaly.setdefault("context", {})
    entity = anomaly.get("entity", "")
    descriptor, needs_classification = resolve_app_descriptor(pm, entity, ctx)
    if descriptor:
        ctx["app_descriptor"] = descriptor
    elif needs_classification:
        lane.enqueue(entity)


def build_correlation_prompt(payload: dict) -> str:
    """Build a cross-domain correlation prompt from a correlation payload.

    Unlike the single-domain prompt builders, this does not take a redis
    client or system_prompt — it uses its own purpose-built template focused
    on RELATIONAL reasoning (not two prompts concatenated).
    """
    primary = payload["primary_anomaly"]
    correlated = payload["correlated_events"]

    lines = [describe_signal(primary["domain"], primary)]
    for ev in correlated:
        lines.append(describe_signal(ev["domain"], ev))
    signals_block = "\n".join(lines)

    lag = payload.get("temporal_lag_seconds", "?")
    combined = payload.get("combined_severity", "?")
    rule = payload.get("escalation_rule", "")
    escalated_from = ""
    if rule:
        # rule looks like "LOW+LOW→MEDIUM"
        left = rule.split("\u2192")[0]
        escalated_from = f" (escalated from {left})"

    return f"""Two or more simultaneous anomalies detected across different behavioral domains:

{signals_block}

These signals occurred within {lag} seconds of each other.
Combined severity: {combined}{escalated_from}.

Reason about what the COMBINATION of these signals suggests about the operator's
current state. What is the most likely underlying cause? What single piece of
advice would address the root cause rather than either symptom individually?

Keep your response concise (3-5 sentences). Focus on the relationship between
the domains, not each signal in isolation."""


# ---------------------------------------------------------------------------
# Advice event builder
# ---------------------------------------------------------------------------


def _build_advice_event(
    payload: dict,
    advice_text: str,
    model_used: str,
    decision: GateDecision | None = None,
) -> dict:
    """Build the advice event dict published on augur.reasoning.advice.

    Derives domain/entity/value/severity from the payload so the result is
    fully self-contained. The caller merges in ``latency_ms`` (only available
    in the async context) before publishing.

    When a ``GateDecision`` is supplied, threads ``decision_id``/``mrt_eligible``
    /``p_fire``/``probe`` into the payload so feedback can join the fired arm by
    exact key, inverse-probability-weight it, and distinguish a bet-hedge
    probe-fire (spec §9). When omitted, these fields carry safe defaults so
    legacy callers keep working.
    """
    primary = payload.get("primary_anomaly", {})
    primary_domain = primary.get("domain", "unknown")

    path = resolve_advisor_path(payload)
    if path == "correlation":
        domain = "multi"
        entity = (
            "+".join(e.get("domain", "?") for e in payload.get("correlated_events", []))
            or "?"
        )
        value = payload.get("temporal_lag_seconds", 0) or 0
    else:
        domain = primary_domain
        entity = primary.get("entity", primary.get("player", "?"))
        value = primary.get("value", primary.get("think_time", 0))

    severity = str(
        payload.get("combined_severity", primary.get("severity", "low"))
    ).lower()

    return {
        "domain": domain,
        "entity": entity,
        "advice": advice_text,
        "value": value,
        "severity": severity,
        "model": model_used,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        # Correlation metadata
        "correlation_found": bool(payload.get("correlation_found")),
        "correlated_domains": [
            e.get("domain") for e in payload.get("correlated_events", [])
        ],
        "rule_key": payload.get("rule_key"),
        "escalation_rule": payload.get("escalation_rule"),
        # Compat aliases for console_display and feedback_collector
        "player": primary.get("entity", entity),
        "move": primary.get("move", primary.get("context", {}).get("label", "?")),
        "think_time": primary.get("value", value),
        # Decision-time frozen baseline for the outcome metric (spec 1A/§4.3):
        # the detector emits these pre-update on every anomaly; thread them so
        # feedback scores surprise-reduction against the decision-time baseline.
        "baseline_mean": primary.get("baseline_mean"),
        "baseline_std": primary.get("baseline_std"),
        "deviation_score": primary.get("deviation_score"),
        "baseline_observation_count": primary.get("baseline_observation_count"),
        # NEW Phase 3 polish fields
        "involved_domains": payload.get("involved_domains") or [primary_domain],
        "correlation_span_s": payload.get("correlation_span_s"),
        "rule_window_s": payload.get("rule_window_s"),
        "temporal_lag_seconds": payload.get("temporal_lag_seconds"),
        # Gate MRT linkage (spec §9): decision_id joins emission/silence/feedback;
        # mrt_eligible/p_fire make the fired arm inverse-probability-weightable;
        # probe flags a bet-hedge probe-fire so analyze_gate (§9/§11.1) can tell
        # the probe-fired arm from a withheld mrt_eligible silence (joined by
        # decision_id) — without it PendingAdvice.probe is silently always False.
        "decision_id": decision.id if decision is not None else None,
        "mrt_eligible": decision.mrt_eligible if decision is not None else False,
        "p_fire": decision.p_fire if decision is not None else None,
        "probe": decision.probe if decision is not None else False,
    }


# ---------------------------------------------------------------------------
# Domain handler registry
# ---------------------------------------------------------------------------

# Maps domain name -> prompt builder function.
# The prompt builder signature is (anomaly, redis_client, system_prompt) -> str.
# Using collections.abc.Callable (not the lowercase built-in `callable`, which
# is a function and not a valid generic alias) so mypy/pyright can check that
# new handlers match the expected shape.
DOMAIN_HANDLERS: dict[str, Callable[[dict, redis.Redis, str], str]] = {
    "chess": build_chess_prompt,
    "typing": build_typing_prompt,
    "activity_focus": build_activity_focus_prompt,
    "activity_intensity": build_activity_intensity_prompt,
}

# ---------------------------------------------------------------------------
# Ollama client
# ---------------------------------------------------------------------------


async def query_ollama(
    prompt: str,
    client: httpx.AsyncClient,
    config: AugurConfig,
) -> tuple[str, float]:
    """Send prompt to Ollama, return (response_text, latency_ms)."""
    payload = {
        "model": config.ollama_model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.7,
            "num_predict": 512,
        },
    }

    t0 = time.monotonic()
    resp = await client.post(
        f"{config.ollama_url}/api/generate",
        json=payload,
        timeout=config.ollama_timeout,
    )
    latency_ms = (time.monotonic() - t0) * 1000
    resp.raise_for_status()

    body = resp.json()
    text = body.get("response", "").strip()
    if not text:
        raise ValueError("Empty response from Ollama")

    return text, latency_ms


# ---------------------------------------------------------------------------
# Gate visibility publishers (spec §8)
# ---------------------------------------------------------------------------


def _primary_field(payload: dict, key: str, default: Any = None) -> Any:
    """Read a field off the primary anomaly (spec §8 suppressed/failure payloads)."""
    return (payload.get("primary_anomaly") or {}).get(key, default)


def _resolve_session_id(redis_client: redis.Redis | None) -> str | None:
    """Best-effort current session_id for the §8 suppressed payload.

    Never raises: a Redis read failure / no active session degrades to ``None``
    (the suppressed event is still published; only the session linkage is lost).
    """
    if redis_client is None:
        return None
    try:
        return get_active_session(redis_client)
    except Exception:  # pragma: no cover - defensive (publish must not crash)
        return None


async def publish_suppressed_event(
    nc: nats.aio.client.Client,
    signature: Signature,
    decision: GateDecision,
    payload: dict,
    redis_client: redis.Redis | None,
) -> None:
    """Publish the full §8 suppressed payload on ``augur.advisor.suppressed``.

    Carries everything ``PendingGateDecision`` + the console need so feedback
    never reconstructs from Redis: decision/state/primary domain+entity/value/
    baseline/severity/session/arm/reason/mrt_eligible/p_withhold and the
    ORIGINATING anomaly's timestamp (so console dedup matches).
    """
    event = {
        "decision_id": decision.id,
        "state_key": signature.state_key,
        "domain": signature.domain,
        "entity": signature.entity,
        "value": signature.value,
        "baseline_mean": _primary_field(payload, "baseline_mean"),
        "baseline_std": _primary_field(payload, "baseline_std"),
        "deviation_score": _primary_field(payload, "deviation_score"),
        "baseline_observation_count": _primary_field(
            payload, "baseline_observation_count"
        ),
        "severity": signature.severity,
        "session_id": _resolve_session_id(redis_client),
        "arm": decision.deciding_arm,
        "reason": decision.reason,
        "mrt_eligible": decision.mrt_eligible,
        "p_withhold": decision.p_withhold,
        "timestamp": _primary_field(payload, "timestamp"),
    }
    await nc.publish(SUBJECT_SUPPRESSED, json.dumps(event).encode())


async def publish_delivery_failure_event(
    nc: nats.aio.client.Client,
    signature: Signature,
    decision: GateDecision,
    payload: dict,
) -> None:
    """Publish an infra non-delivery on ``augur.advisor.delivery_failure`` (§8).

    A distinct subject from ``.suppressed`` so ``PendingGateDecision`` (which
    subscribes only to ``.suppressed``) never tracks an infra drop.
    """
    event = {
        "decision_id": decision.id,
        "state_key": signature.state_key,
        "domain": signature.domain,
        "entity": signature.entity,
        "reason": decision.reason,
        "timestamp": _primary_field(payload, "timestamp"),
    }
    await nc.publish(SUBJECT_DELIVERY_FAILURE, json.dumps(event).encode())


# ---------------------------------------------------------------------------
# Per-message control flow (spec §3) — extracted from run() so it is directly
# unit-testable with the LLM (query_ollama) + NATS (nc.publish) mocked.
# ---------------------------------------------------------------------------


def _safe(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Call ``fn`` swallowing any Exception at ERROR (spec §3 ``_safe``).

    A state-write bug must never terminate the NATS callback or drop a must-fire.
    """
    try:
        return fn(*args, **kwargs)
    except Exception:
        log.error("gate state write failed: %s", getattr(fn, "__name__", fn))
        return None


async def _build_prompt_and_deliver(
    *,
    payload: dict,
    signature: Signature,
    decision: GateDecision,
    pm: PersistenceManager,
    nc: nats.aio.client.Client,
    http_client: httpx.AsyncClient,
    redis_client: redis.Redis | None,
    config: AugurConfig,
    now: float,
    query_ollama_fn: Callable,
    gate: Gate,
    tier: int,
    audit_only: bool,
) -> None:
    """Build prompt → query LLM → publish advice → record_delivery_success.

    A prompt-build / Ollama / publish failure writes ONLY a ``delivery_failure``
    (no phantom delivery; ``record_delivery_success`` runs solely after a
    successful publish).  ``record_delivery_success`` is wrapped in ``_safe`` so
    a state-write bug degrades safe (consecutive_suppressions simply isn't reset).
    """
    path = resolve_advisor_path(payload)
    domain = signature.domain
    # ── prompt build ──
    try:
        if path == "correlation":
            prompt = build_correlation_prompt(payload)
        else:
            primary = payload["primary_anomaly"]
            stored_prompt = pm.load_prompt(domain)
            system_prompt = stored_prompt or DEFAULT_PROMPTS.get(
                domain,
                f"You are an analyst monitoring '{domain}' data. An anomaly was detected.",
            )
            builder = DOMAIN_HANDLERS.get(domain, build_generic_prompt)
            prompt = builder(primary, redis_client, system_prompt)
    except Exception as exc:
        log.error("Prompt build failed for '%s': %s", domain, exc)
        _safe(
            pm.save_delivery_failure, signature, "prompt_build_failed", now, decision.id
        )
        await publish_delivery_failure_event(
            nc, signature, decision.as_fire("prompt_build_failed"), payload
        )
        return

    # ── LLM ──
    try:
        advice, latency_ms = await query_ollama_fn(prompt, http_client, config)
    except httpx.ConnectError:
        log.error("Ollama unreachable at %s", config.ollama_url)
        _safe(
            pm.save_delivery_failure, signature, "ollama_unreachable", now, decision.id
        )
        await publish_delivery_failure_event(
            nc, signature, decision.as_fire("ollama_unreachable"), payload
        )
        return
    except httpx.TimeoutException:
        log.error("Ollama timed out after %ds", config.ollama_timeout)
        _safe(pm.save_delivery_failure, signature, "ollama_timeout", now, decision.id)
        await publish_delivery_failure_event(
            nc, signature, decision.as_fire("ollama_timeout"), payload
        )
        return
    except Exception as exc:
        log.error("Ollama returned unusable response: %s", exc)
        _safe(pm.save_delivery_failure, signature, "ollama_error", now, decision.id)
        await publish_delivery_failure_event(
            nc, signature, decision.as_fire("ollama_error"), payload
        )
        return

    # ── publish advice ──
    advice_payload = _build_advice_event(
        payload, advice_text=advice, model_used=config.ollama_model, decision=decision
    )
    advice_payload["latency_ms"] = round(latency_ms, 1)
    advice_payload["tier"] = tier
    try:
        await nc.publish(PUBLISH_SUBJECT, json.dumps(advice_payload).encode())
        log.info("Published advice to %s", PUBLISH_SUBJECT)
    except Exception as exc:
        log.error("NATS publish failed: %s", exc)
        _safe(pm.save_delivery_failure, signature, "publish_failed", now, decision.id)
        return

    # ── record AFTER a successful publish (no phantom) ──
    _safe(pm.save_last_advice, advice_payload)
    _safe(
        gate.record_delivery_success,
        signature,
        pm,
        now,
        decision=decision,
        tier=tier,
        audit_only=audit_only,
    )


async def _publish_tier1_note(
    *,
    payload: dict,
    signature: Signature,
    decision: GateDecision,
    nc: nats.aio.client.Client,
    config: AugurConfig,
) -> None:
    """Publish a templated Tier-1 note on ``augur.reasoning.advice`` (tier=1)."""
    note = _build_advice_event(
        payload,
        advice_text=(
            f"(Tier-1 note) A {signature.severity} {signature.domain} signal was "
            f"observed on {signature.entity}; no full analysis was warranted."
        ),
        model_used="tier1-template",
        decision=decision,
    )
    note["tier"] = 1
    await nc.publish(PUBLISH_SUBJECT, json.dumps(note).encode())


async def process_message(
    *,
    payload: dict,
    gate: Gate,
    scheduler: MustFireScheduler,
    pm: PersistenceManager,
    nc: nats.aio.client.Client,
    http_client: httpx.AsyncClient,
    redis_client: redis.Redis | None,
    classifier_lane: ClassifierLane,
    config: AugurConfig,
    now: float,
    query_ollama: Callable = query_ollama,
) -> None:
    """Gate-driven per-message control flow (spec §3).

    Severity-gate → enrich descriptors → gate.evaluate (fail-open on any
    Exception) → suppress / downgrade / fire, with the must-fire scheduler
    serializing exempt / fail_open / anti_starvation_release ahead of ordinary
    fires (which never await).  The §3 pseudocode is implemented verbatim.
    """
    # Gate on combined_severity (uppercase from correlator) — compare lowercase.
    severity = str(payload.get("combined_severity", "low")).lower()
    if severity not in SEVERITY_GATE:
        primary = payload.get("primary_anomaly", {})
        log.info(
            "Ignoring %s severity event for %s/%s",
            severity,
            primary.get("domain", "?"),
            primary.get("entity", "?"),
        )
        return

    path = resolve_advisor_path(payload)

    # Populate the app-descriptor map BEFORE the gate so it fills even when advice
    # is suppressed/skipped (resolve_app_descriptor guards RedisError; enqueue is
    # sync-guarded, so this cannot raise out of the handler).
    enrich_payload_descriptors(pm, classifier_lane, path, payload)

    signature = build_signature(payload)

    # ── gate.evaluate (READ-ONLY) — any Exception ⇒ fail open to FIRE (inv. C) ──
    try:
        decision = gate.evaluate(signature, pm, config, now=now)
    except Exception as exc:
        log.error("Advisor gate failed open: %s", exc)
        decision = GateDecision.fire("gate_error_fail_open")

    # ── suppress (authoritative record_suppression or FIRE) ──
    if decision.action == "suppress":
        try:
            ok = gate.record_suppression(decision, signature, pm, now)
        except Exception as exc:
            log.error("record_suppression failed open: %s", exc)
            ok = False
        if not ok:
            decision = decision.as_fire("gate_error_fail_open")
            # falls through to the must_fire block below
        else:
            try:
                await publish_suppressed_event(
                    nc, signature, decision, payload, redis_client
                )
            except Exception as exc:
                log.error("suppressed publish failed: %s", exc)
                _safe(
                    pm.save_delivery_failure,
                    signature,
                    "suppressed_publish_failed",
                    now,
                    decision.id,
                )
            return

    # ── downgrade → Tier-1 note ──
    if decision.action == "downgrade":
        try:
            await _publish_tier1_note(
                payload=payload,
                signature=signature,
                decision=decision,
                nc=nc,
                config=config,
            )
        except Exception as exc:
            log.error("tier-1 note publish failed: %s", exc)
            _safe(
                pm.save_delivery_failure,
                signature,
                "tier1_publish_failed",
                now,
                decision.id,
            )
            return
        _safe(
            gate.record_delivery_success,
            signature,
            pm,
            now,
            decision=decision,
            tier=1,
        )
        return

    # ── fire (normal, exempt, probe, anti_starvation_release, cap_fail_open) ──
    must_fire = signature.exempt or decision.reason in (
        "anti_starvation_release",
        "gate_error_fail_open",
    )

    async def _deliver() -> None:
        await _build_prompt_and_deliver(
            payload=payload,
            signature=signature,
            decision=decision,
            pm=pm,
            nc=nc,
            http_client=http_client,
            redis_client=redis_client,
            config=config,
            now=now,
            query_ollama_fn=query_ollama,
            gate=gate,
            tier=decision.tier or 2,
            audit_only=signature.exempt,
        )

    if must_fire:
        if decision.reason == "anti_starvation_release" and scheduler.release_in_flight(
            signature.state_key
        ):
            return  # coalesce: one in-flight anti-starvation release per channel
        priority = (
            "exempt"
            if signature.exempt
            else (
                "fail_open"
                if decision.reason == "gate_error_fail_open"
                else "anti_starvation"
            )
        )
        try:
            with scheduler.track_release(signature.state_key):
                async with scheduler.acquire(priority):
                    if (
                        decision.reason == "anti_starvation_release"
                        and not gate.still_starved(signature, pm, now)
                    ):
                        return  # concurrent delivery already served this channel
                    await _deliver()
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # inv. C: a scheduler bug must never drop a must-fire
            log.error("must-fire scheduler failed: %s", exc)
            await scheduler.emergency_deliver(_deliver)
        return

    # ── ordinary fire — nonblocking; fails if lock held OR a must-fire queued ──
    if not scheduler.try_acquire_ordinary():
        if decision.reason == "cap_fail_open":
            _safe(
                pm.save_delivery_failure,
                signature,
                "cap_fail_open_busy",
                now,
                decision.id,
            )
            await publish_delivery_failure_event(nc, signature, decision, payload)
            return
        try:
            tracked = gate.record_busy_skip(signature, pm, now)
        except Exception as exc:  # inv. C: a record_busy_skip bug must not silence
            log.error("record_busy_skip failed open: %s", exc)
            decision = decision.as_fire("gate_error_fail_open")
            try:
                async with scheduler.acquire("fail_open"):
                    await _deliver()
            except asyncio.CancelledError:
                raise
            except Exception as exc2:
                log.error("must-fire scheduler failed: %s", exc2)
                await scheduler.emergency_deliver(_deliver)
            return
        if not tracked:
            _safe(
                pm.save_delivery_failure,
                signature,
                "advisor_busy_untrackable",
                now,
                decision.id,
            )
        await publish_delivery_failure_event(nc, signature, decision, payload)
        return

    try:
        await _deliver()
    finally:
        scheduler.release_ordinary()


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
    classifier_lane = ClassifierLane(pm, http_client, config)

    # Check Ollama reachability at startup (non-fatal)
    try:
        probe = await http_client.get(f"{config.ollama_url}/api/tags", timeout=5)
        probe.raise_for_status()
        models = [m["name"] for m in probe.json().get("models", [])]
        log.info("Ollama reachable. Available models: %s", models or "(none)")
        if not any(config.ollama_model in m for m in models):
            log.warning(
                "Model '%s' not found locally. "
                "It will be pulled on first request (may be slow).",
                config.ollama_model,
            )
        if classifier_lane.enabled and not classifier_model_available(
            config.ollama_classifier_model, models
        ):
            log.warning(
                "Classifier model '%s' not available; disabling LLM descriptor "
                "fallback for this session (OS metadata still applies).",
                config.ollama_classifier_model,
            )
            classifier_lane.enabled = False
        elif classifier_lane.enabled:
            log.info(
                "App-descriptor classifier lane enabled (model=%s). Requires "
                "OLLAMA_MAX_LOADED_MODELS>=2 and OLLAMA_NUM_PARALLEL>=2 on the "
                "Ollama host for true parallelism.",
                config.ollama_classifier_model,
            )
    except (httpx.HTTPError, httpx.ConnectError) as exc:
        log.warning(
            "Ollama not reachable at startup (%s). Will retry when anomalies arrive.",
            exc,
        )

    # Serialize LLM deliveries via the must-fire scheduler over reasoning_lock
    # (spec §4): exempt / fail_open / anti_starvation_release acquire through it;
    # ordinary fires use the nonblocking try_acquire_ordinary path.
    reasoning_lock = asyncio.Lock()
    scheduler = MustFireScheduler(
        reasoning_lock,
        max_release_wait_s=config.gate_max_release_wait_s,
        max_release_overtake=config.gate_max_release_overtake,
    )
    # still_starved (called under the lock) needs the starvation bounds → inject
    # the live config so the gate uses the operator's tuning.
    gate = Gate(config=config)

    async def on_message(msg: nats.aio.client.Msg) -> None:
        try:
            payload = json.loads(msg.data.decode())
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            log.warning("Bad correlation payload: %s", exc)
            return

        await process_message(
            payload=payload,
            gate=gate,
            scheduler=scheduler,
            pm=pm,
            nc=nc,
            http_client=http_client,
            redis_client=redis_client,
            classifier_lane=classifier_lane,
            config=config,
            now=datetime.now(timezone.utc).timestamp(),
        )

    sub = await nc.subscribe(SUBSCRIBE_SUBJECT, cb=on_message)
    log.info("Subscribed to %s", SUBSCRIBE_SUBJECT)
    log.info("Severity gate: only processing %s", SEVERITY_GATE)
    log.info("Supported domains: %s (+ generic fallback)", list(DOMAIN_HANDLERS.keys()))
    log.info(
        "Ollama model: %s (timeout: %ds)", config.ollama_model, config.ollama_timeout
    )
    log.info("Waiting for anomalies...")

    try:
        while True:
            await asyncio.sleep(1)
    except asyncio.CancelledError:
        pass
    finally:
        await sub.unsubscribe()
        await classifier_lane.shutdown()
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
