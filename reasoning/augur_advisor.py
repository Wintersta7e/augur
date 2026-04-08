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
SUBSCRIBE_SUBJECT = "augur.detection.anomaly"
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
# Domain handler registry
# ---------------------------------------------------------------------------

# Maps domain name -> prompt builder function
DOMAIN_HANDLERS: dict[str, callable] = {
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

    async def on_anomaly(msg: nats.aio.client.Msg) -> None:
        try:
            anomaly = json.loads(msg.data.decode())
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            log.warning("Bad anomaly payload: %s", exc)
            return

        severity = anomaly.get("severity", "low")
        domain = anomaly.get("domain", "unknown")
        entity = anomaly.get("entity", anomaly.get("player", "?"))
        value = anomaly.get("value", anomaly.get("think_time", 0))

        if severity not in SEVERITY_GATE:
            log.info(
                "Ignoring %s severity anomaly for %s/%s (%.2f)",
                severity,
                domain,
                entity,
                value,
            )
            return

        log.info(
            "Anomaly received [%s] %s/%s value=%.2f — querying LLM",
            severity.upper(),
            domain,
            entity,
            value,
        )

        if reasoning_lock.locked():
            log.warning("LLM reasoning already in progress, skipping this anomaly")
            return

        async with reasoning_lock:
            # Load domain-specific prompt from persistence, or use default
            stored_prompt = pm.load_prompt(domain)
            if stored_prompt:
                system_prompt = stored_prompt
                log.info("Using stored prompt for domain '%s'", domain)
            else:
                system_prompt = DEFAULT_PROMPTS.get(
                    domain,
                    f"You are an analyst monitoring '{domain}' data. An anomaly was detected.",
                )
                log.info("Using default prompt for domain '%s'", domain)

            # Select prompt builder
            builder = DOMAIN_HANDLERS.get(domain, build_generic_prompt)
            try:
                prompt = builder(anomaly, redis_client, system_prompt)
            except Exception as exc:
                log.error("Prompt build failed for domain '%s': %s", domain, exc)
                return

            log.debug("Prompt:\n%s", prompt)

            # Query Ollama
            try:
                advice, latency_ms = await query_ollama(prompt, http_client, config)
            except httpx.ConnectError:
                log.error(
                    "Ollama unreachable at %s — is it running?",
                    config.ollama_url,
                )
                return
            except httpx.TimeoutException:
                log.error(
                    "Ollama timed out after %ds",
                    config.ollama_timeout,
                )
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

            # Build output payload — include domain + compat fields
            advice_payload = {
                "domain": domain,
                "entity": entity,
                "advice": advice,
                "value": value,
                "severity": severity,
                "model": config.ollama_model,
                "latency_ms": round(latency_ms, 1),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                # Compat aliases for console_display and downstream
                "player": entity,
                "move": anomaly.get(
                    "move", anomaly.get("context", {}).get("label", "?")
                ),
                "think_time": value,
            }

            # Publish to NATS
            try:
                await nc.publish(
                    PUBLISH_SUBJECT,
                    json.dumps(advice_payload).encode(),
                )
                log.info("Published advice to %s", PUBLISH_SUBJECT)
            except Exception as exc:
                log.error("NATS publish failed: %s", exc)

            # Write to Redis
            try:
                redis_client.set(REDIS_KEY_ADVICE, json.dumps(advice_payload))
                log.info("Wrote advice to Redis key %s", REDIS_KEY_ADVICE)
            except redis.RedisError as exc:
                log.error("Redis write failed: %s", exc)

    sub = await nc.subscribe(SUBSCRIBE_SUBJECT, cb=on_anomaly)
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
