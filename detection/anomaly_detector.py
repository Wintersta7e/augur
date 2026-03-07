"""Anomaly detector for chess move timing data.

Subscribes to NATS 'augur.perception.chess', scores each move using
River's HalfSpaceTrees and an EWMA statistical baseline, then publishes
anomalies to 'augur.detection.anomaly' and Redis.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone

import nats
import redis
from river.anomaly import HalfSpaceTrees

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("anomaly_detector")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
NATS_URL = "nats://localhost:4222"
SUBSCRIBE_SUBJECT = "augur.perception.chess"
PUBLISH_SUBJECT = "augur.detection.anomaly"
REDIS_KEY_ANOMALY = "augur:detection:last_anomaly"

MIN_MOVES_BEFORE_SCORING = 3

# EWMA smoothing factor (higher = more weight on recent observations)
EWMA_ALPHA = 0.3

# Standard-deviation multiplier for statistical anomaly
SIGMA_THRESHOLD = 2.0

# HalfSpaceTrees anomaly score threshold (scores are 0-1)
HST_THRESHOLD = 0.7

# Severity boundaries (based on how many sigmas the think time deviates)
SEVERITY_MEDIUM_SIGMA = 2.5
SEVERITY_HIGH_SIGMA = 4.0

# ---------------------------------------------------------------------------
# Per-player tracker
# ---------------------------------------------------------------------------

@dataclass
class PlayerBaseline:
    """Tracks EWMA mean and variance for one player's think times."""

    ewma_mean: float = 0.0
    ewma_var: float = 0.0
    move_count: int = 0
    hst: HalfSpaceTrees = field(default_factory=lambda: HalfSpaceTrees(
        n_trees=15,
        height=8,
        window_size=50,
        seed=42,
    ))

    @property
    def ewma_std(self) -> float:
        return math.sqrt(max(self.ewma_var, 0.0))

    def update(self, think_time: float) -> None:
        """Update EWMA stats and train the HST model."""
        self.move_count += 1
        if self.move_count == 1:
            self.ewma_mean = think_time
            self.ewma_var = 0.0
        else:
            diff = think_time - self.ewma_mean
            self.ewma_mean += EWMA_ALPHA * diff
            # Welford-style EWMA variance
            self.ewma_var = (1 - EWMA_ALPHA) * (self.ewma_var + EWMA_ALPHA * diff * diff)

        self.hst.learn_one({"think_time": think_time})

    def score(self, think_time: float) -> tuple[float, float]:
        """Return (deviation_sigmas, hst_anomaly_score).

        deviation_sigmas: how many standard deviations from EWMA mean.
        hst_anomaly_score: HalfSpaceTrees anomaly score in [0, 1].
        """
        std = self.ewma_std
        if std < 0.01:
            # Avoid division by near-zero; treat as no deviation data yet
            deviation = 0.0
        else:
            deviation = abs(think_time - self.ewma_mean) / std

        hst_score: float = self.hst.score_one({"think_time": think_time})
        return deviation, hst_score


def classify_severity(deviation: float, hst_score: float) -> str:
    """Map deviation magnitude to a severity label."""
    if deviation >= SEVERITY_HIGH_SIGMA or hst_score >= 0.9:
        return "high"
    if deviation >= SEVERITY_MEDIUM_SIGMA or hst_score >= 0.8:
        return "medium"
    return "low"

# ---------------------------------------------------------------------------
# Redis
# ---------------------------------------------------------------------------

def connect_redis() -> redis.Redis:
    client = redis.Redis(host="localhost", port=6379, socket_connect_timeout=5)
    client.ping()
    log.info("Redis connected")
    return client

# ---------------------------------------------------------------------------
# Core loop
# ---------------------------------------------------------------------------

async def run() -> None:
    # -- Infrastructure connections ------------------------------------------
    redis_client = connect_redis()

    nc = await nats.connect(NATS_URL, connect_timeout=5)
    log.info("NATS connected (%s)", NATS_URL)

    baselines: dict[str, PlayerBaseline] = {
        "white": PlayerBaseline(),
        "black": PlayerBaseline(),
    }

    # -- Message handler -----------------------------------------------------
    async def on_move(msg: nats.aio.client.Msg) -> None:
        try:
            data = json.loads(msg.data.decode())
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            log.warning("Bad message payload: %s", exc)
            return

        player: str = data.get("player", "")
        think_time: float = data.get("think_time_seconds", 0.0)
        move_san: str = data.get("move_san", "?")
        move_number: int = data.get("move_number", 0)

        if player not in baselines:
            log.warning("Unknown player value: %r", player)
            return

        bl = baselines[player]

        # Score BEFORE updating the baseline so the observation is novel
        deviation, hst_score = bl.score(think_time)
        is_trained = bl.move_count >= MIN_MOVES_BEFORE_SCORING

        # Now update the baseline with this observation
        bl.update(think_time)

        # Log every move's scoring
        log.info(
            "[%s] move=%d %s  think=%.2fs  ewma_mean=%.2fs  ewma_std=%.2fs  "
            "dev=%.2f\u03c3  hst=%.3f  trained=%s",
            player, move_number, move_san, think_time,
            bl.ewma_mean, bl.ewma_std,
            deviation, hst_score, is_trained,
        )

        if not is_trained:
            log.info(
                "  \u2514\u2500 Building baseline (%d/%d moves)",
                bl.move_count, MIN_MOVES_BEFORE_SCORING,
            )
            return

        # Check anomaly thresholds
        is_anomaly = deviation >= SIGMA_THRESHOLD or hst_score >= HST_THRESHOLD

        if not is_anomaly:
            return

        severity = classify_severity(deviation, hst_score)
        anomaly_payload = {
            "player": player,
            "move": move_san,
            "move_number": move_number,
            "think_time": round(think_time, 3),
            "baseline_mean": round(bl.ewma_mean, 3),
            "baseline_std": round(bl.ewma_std, 3),
            "deviation_score": round(deviation, 3),
            "anomaly_score": round(hst_score, 3),
            "severity": severity,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        log.warning(
            "  \u26a0 ANOMALY [%s] %s: think=%.2fs  dev=%.1f\u03c3  hst=%.3f",
            severity.upper(), player, think_time, deviation, hst_score,
        )

        # Publish to NATS
        try:
            await nc.publish(
                PUBLISH_SUBJECT,
                json.dumps(anomaly_payload).encode(),
            )
            log.info("  Published anomaly to %s", PUBLISH_SUBJECT)
        except Exception as exc:
            log.error("  NATS publish failed: %s", exc)

        # Write to Redis
        try:
            redis_client.set(REDIS_KEY_ANOMALY, json.dumps(anomaly_payload))
            log.info("  Wrote anomaly to Redis key %s", REDIS_KEY_ANOMALY)
        except redis.RedisError as exc:
            log.error("  Redis write failed: %s", exc)

    # -- Subscribe and wait --------------------------------------------------
    sub = await nc.subscribe(SUBSCRIBE_SUBJECT, cb=on_move)
    log.info("Subscribed to %s  (min %d moves before scoring)", SUBSCRIBE_SUBJECT, MIN_MOVES_BEFORE_SCORING)
    log.info("Anomaly thresholds: sigma >= %.1f  OR  HST >= %.2f", SIGMA_THRESHOLD, HST_THRESHOLD)
    log.info("Severity: low < %.1f\u03c3, medium < %.1f\u03c3, high >= %.1f\u03c3",
             SEVERITY_MEDIUM_SIGMA, SEVERITY_HIGH_SIGMA, SEVERITY_HIGH_SIGMA)
    log.info("Waiting for moves...")

    try:
        while True:
            await asyncio.sleep(1)
    except asyncio.CancelledError:
        pass
    finally:
        await sub.unsubscribe()
        await nc.close()
        log.info("Shut down cleanly")


def main() -> None:
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        log.info("Interrupted")


if __name__ == "__main__":
    main()
