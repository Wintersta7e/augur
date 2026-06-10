"""LLM-powered chess advisor triggered by anomaly detection.

Subscribes to NATS 'augur.detection.anomaly', enriches with board context
from Redis, queries Ollama for strategic advice, and publishes results.
Only activates for medium/high severity anomalies.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from datetime import datetime, timezone

import httpx
import nats
import redis

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("chess_advisor")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
NATS_URL = "nats://localhost:4222"
SUBSCRIBE_SUBJECT = "augur.detection.anomaly"
PUBLISH_SUBJECT = "augur.reasoning.advice"

REDIS_KEY_LAST_MOVE = "augur:sensus:chess:last_move"
REDIS_KEY_HISTORY = "augur:sensus:chess:move_history"
REDIS_KEY_ADVICE = "augur:reasoning:last_advice"

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://host.docker.internal:11434")
OLLAMA_MODEL = "qwen2.5:32b"
OLLAMA_TIMEOUT_S = 120

SEVERITY_GATE = {"medium", "high"}

# ---------------------------------------------------------------------------
# Redis helpers
# ---------------------------------------------------------------------------


def connect_redis() -> redis.Redis:
    client = redis.Redis(host="localhost", port=6379, socket_connect_timeout=5)
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
# Prompt construction
# ---------------------------------------------------------------------------


def build_prompt(anomaly: dict, last_move: dict | None, history: list[dict]) -> str:
    """Build an LLM prompt with full game context."""
    player = anomaly["player"]
    think_time = anomaly["think_time"]
    baseline_mean = anomaly.get("baseline_mean", "?")
    deviation = anomaly.get("deviation_score", "?")
    severity = anomaly["severity"]

    # Format move history as a numbered list
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

    return f"""You are a chess analyst reviewing a game in progress. A timing anomaly has been detected.

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
# Ollama client
# ---------------------------------------------------------------------------


async def query_ollama(prompt: str, client: httpx.AsyncClient) -> tuple[str, float]:
    """Send prompt to Ollama, return (response_text, latency_ms).

    Raises on timeout or connection error.
    """
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.7,
            "num_predict": 512,
        },
    }

    t0 = time.monotonic()
    resp = await client.post(
        f"{OLLAMA_URL}/api/generate",
        json=payload,
        timeout=OLLAMA_TIMEOUT_S,
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
    redis_client = connect_redis()

    nc = await nats.connect(NATS_URL, connect_timeout=5)
    log.info("NATS connected (%s)", NATS_URL)

    http_client = httpx.AsyncClient()

    # Check Ollama reachability at startup (non-fatal)
    try:
        probe = await http_client.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        probe.raise_for_status()
        models = [m["name"] for m in probe.json().get("models", [])]
        log.info("Ollama reachable. Available models: %s", models or "(none)")
        if not any(OLLAMA_MODEL in m for m in models):
            log.warning(
                "Model '%s' not found locally. "
                "It will be pulled on first request (may be slow).",
                OLLAMA_MODEL,
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
        player = anomaly.get("player", "?")
        move = anomaly.get("move", "?")
        think_time = anomaly.get("think_time", 0)

        if severity not in SEVERITY_GATE:
            log.info(
                "Ignoring %s severity anomaly for %s (%s, %.2fs)",
                severity,
                player,
                move,
                think_time,
            )
            return

        log.info(
            "Anomaly received [%s] %s played %s in %.2fs — querying LLM",
            severity.upper(),
            player,
            move,
            think_time,
        )

        if reasoning_lock.locked():
            log.warning("LLM reasoning already in progress, skipping this anomaly")
            return

        async with reasoning_lock:
            # Gather board context from Redis
            try:
                last_move, history = read_board_context(redis_client)
            except redis.RedisError as exc:
                log.error("Failed to read board context from Redis: %s", exc)
                return

            prompt = build_prompt(anomaly, last_move, history)
            log.debug("Prompt:\n%s", prompt)

            # Query Ollama
            try:
                advice, latency_ms = await query_ollama(prompt, http_client)
            except httpx.ConnectError:
                log.error(
                    "Ollama unreachable at %s — is it running?",
                    OLLAMA_URL,
                )
                return
            except httpx.TimeoutException:
                log.error(
                    "Ollama timed out after %ds",
                    OLLAMA_TIMEOUT_S,
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
            log.info("Advice for %s:\n%s", player, advice)

            # Build output payload
            advice_payload = {
                "player": player,
                "move": move,
                "advice": advice,
                "think_time": think_time,
                "severity": severity,
                "model": OLLAMA_MODEL,
                "latency_ms": round(latency_ms, 1),
                "timestamp": datetime.now(timezone.utc).isoformat(),
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
    log.info("Ollama model: %s (timeout: %ds)", OLLAMA_MODEL, OLLAMA_TIMEOUT_S)
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
