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
log = logging.getLogger("augur_advisor")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SUBSCRIBE_SUBJECT = "augur.correlation.detected"
PUBLISH_SUBJECT = "augur.reasoning.advice"

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


def connect_redis(config: AugurConfig) -> redis.Redis:
    client = redis.Redis(
        host=config.redis_host,
        port=config.redis_port,
        socket_connect_timeout=config.redis_connect_timeout,
    )
    client.ping()
    log.info("Redis connected")
    return client


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
    entity = anomaly.get("entity", "user")
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
1. What might this typing pattern indicate about the user's cognitive state?
2. Are they likely stuck on a problem, distracted, or fatigued?
3. Provide one concrete, helpful suggestion (e.g., take a break, switch tasks, review what they just wrote).

Keep your response concise (2-4 sentences). Be supportive, not intrusive. Focus on cognitive well-being and productivity."""


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

Analyze this anomaly and provide a concise assessment (2-4 sentences)."""


# ---------------------------------------------------------------------------
# Lightweight per-domain one-liner formatter (used by correlation prompts)
# ---------------------------------------------------------------------------


def describe_signal(domain: str, anomaly: dict) -> str:
    """Return a one-line human-readable summary of a single-domain anomaly.

    Used inside build_correlation_prompt to embed each domain's contribution
    without rebuilding a full domain-specific prompt. Does not share code
    with the full prompt builders — different purpose, different format.
    """
    if domain == "chess":
        move = anomaly.get("context", {}).get("move_san") or anomaly.get("move", "?")
        think = anomaly.get("value", anomaly.get("think_time", 0))
        baseline = anomaly.get("baseline_mean", "?")
        deviation = anomaly.get("deviation_score", "?")
        return (
            f"CHESS (timing): {anomaly.get('entity', '?')} paused {think}s on "
            f"move {move}. Baseline: {baseline}s. Deviation: {deviation}\u03c3."
        )

    if domain == "typing":
        pause = anomaly.get("value", 0)
        unit = anomaly.get("unit", "seconds")
        ctx = anomaly.get("context", {})
        avg_wpm = ctx.get("avg_wpm", "?")
        baseline = anomaly.get("baseline_mean", "?")
        return (
            f"TYPING (rhythm): Pause duration {pause}{unit[:1]}. "
            f"Average speed {avg_wpm} wpm. Baseline pause: {baseline}s."
        )

    # Generic fallback
    value = anomaly.get("value", "?")
    unit = anomaly.get("unit", "")
    deviation = anomaly.get("deviation_score", "?")
    return (
        f"{domain.upper()}: {anomaly.get('event_type', 'event')} "
        f"value={value}{unit}  deviation={deviation}\u03c3"
    )


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
    except (httpx.HTTPError, httpx.ConnectError) as exc:
        log.warning(
            "Ollama not reachable at startup (%s). Will retry when anomalies arrive.",
            exc,
        )

    # Track in-flight requests to avoid piling up during slow LLM responses
    reasoning_lock = asyncio.Lock()

    async def on_message(msg: nats.aio.client.Msg) -> None:
        try:
            payload = json.loads(msg.data.decode())
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            log.warning("Bad correlation payload: %s", exc)
            return

        # Gate on combined_severity (uppercase from correlator) — compare lowercase
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

        if path == "correlation":
            domain = "multi"
            entity = (
                "+".join(
                    e.get("domain", "?") for e in payload.get("correlated_events", [])
                )
                or "?"
            )
            value = payload.get("temporal_lag_seconds", 0) or 0
        else:
            primary = payload["primary_anomaly"]
            domain = primary.get("domain", "unknown")
            entity = primary.get("entity", primary.get("player", "?"))
            value = primary.get("value", primary.get("think_time", 0))

        log.info(
            "Event received [%s] path=%s domain=%s entity=%s — querying LLM",
            severity.upper(),
            path,
            domain,
            entity,
        )

        if reasoning_lock.locked():
            log.warning("LLM reasoning already in progress, skipping")
            return

        async with reasoning_lock:
            if path == "correlation":
                prompt = build_correlation_prompt(payload)
                system_prompt = None  # correlation prompt is self-contained
            else:
                primary = payload["primary_anomaly"]
                stored_prompt = pm.load_prompt(domain)
                if stored_prompt:
                    system_prompt = stored_prompt
                    log.info("Using stored prompt for domain '%s'", domain)
                else:
                    system_prompt = DEFAULT_PROMPTS.get(
                        domain,
                        f"You are an analyst monitoring '{domain}' data. An anomaly was detected.",
                    )
                builder = DOMAIN_HANDLERS.get(domain, build_generic_prompt)
                try:
                    prompt = builder(primary, redis_client, system_prompt)
                except Exception as exc:
                    log.error("Prompt build failed for '%s': %s", domain, exc)
                    return

            log.debug("Prompt:\n%s", prompt)

            try:
                advice, latency_ms = await query_ollama(prompt, http_client, config)
            except httpx.ConnectError:
                log.error("Ollama unreachable at %s", config.ollama_url)
                return
            except httpx.TimeoutException:
                log.error("Ollama timed out after %ds", config.ollama_timeout)
                return
            except httpx.HTTPStatusError as exc:
                log.error("Ollama HTTP error: %s", exc.response.status_code)
                return
            except ValueError as exc:
                log.error("Ollama returned unusable response: %s", exc)
                return

            log.info(
                "LLM responded in %.0fms (%d chars)",
                latency_ms,
                len(advice),
            )
            log.info("Advice for %s/%s:\n%s", domain, entity, advice)

            # Build advice payload — include correlation fields for downstream
            primary_compat = payload.get("primary_anomaly", {})
            advice_payload = {
                "domain": domain,
                "entity": entity,
                "advice": advice,
                "value": value,
                "severity": severity,
                "model": config.ollama_model,
                "latency_ms": round(latency_ms, 1),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                # Correlation metadata
                "correlation_found": bool(payload.get("correlation_found")),
                "correlated_domains": [
                    e.get("domain") for e in payload.get("correlated_events", [])
                ],
                "rule_key": payload.get("rule_key"),
                "escalation_rule": payload.get("escalation_rule"),
                # Compat aliases for console_display and feedback_collector
                "player": primary_compat.get("entity", entity),
                "move": primary_compat.get(
                    "move", primary_compat.get("context", {}).get("label", "?")
                ),
                "think_time": primary_compat.get("value", value),
            }

            try:
                await nc.publish(
                    PUBLISH_SUBJECT,
                    json.dumps(advice_payload).encode(),
                )
                log.info("Published advice to %s", PUBLISH_SUBJECT)
            except Exception as exc:
                log.error("NATS publish failed: %s", exc)

            try:
                redis_client.set(REDIS_KEY_ADVICE, json.dumps(advice_payload))
                log.info("Wrote advice to Redis key %s", REDIS_KEY_ADVICE)
            except redis.RedisError as exc:
                log.error("Redis write failed: %s", exc)

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
