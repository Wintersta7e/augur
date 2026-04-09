"""Domain-agnostic anomaly detector.

Subscribes to NATS 'augur.perception.>' (wildcard), parses incoming
messages as PerceptionEvent, scores each observation using per-(domain,
entity) EWMA baselines and River HalfSpaceTrees, then publishes anomalies
to 'augur.detection.anomaly' and Redis.

Baselines are persisted to Redis and survive restarts. Thresholds are
loaded from PersistenceManager per domain, falling back to defaults.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import nats
import redis
from river.anomaly import HalfSpaceTrees

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from blackboard.config import AugurConfig
from blackboard.connections import connect_redis
from blackboard.contracts import PerceptionEvent
from blackboard.persistence import PersistenceManager

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
SUBSCRIBE_SUBJECT = "augur.perception.>"
PUBLISH_SUBJECT = "augur.detection.anomaly"
REDIS_KEY_ANOMALY = "augur:detection:last_anomaly"

# Default thresholds (overridden per-domain via PersistenceManager)
DEFAULT_THRESHOLDS = {
    "min_observations": 3,
    "ewma_alpha": 0.3,
    "sigma_threshold": 2.0,
    "hst_threshold": 0.7,
    "severity_medium_sigma": 2.5,
    "severity_high_sigma": 4.0,
}

# LEAK-10: cap on the in-memory baselines dict. The detector creates one
# EntityBaseline per unique (domain, entity) pair seen on the wildcard
# perception subject. Each EntityBaseline owns a River HalfSpaceTrees
# model with 15 trees and a 50-sample window, so the per-entity memory
# footprint is non-trivial. Without a cap, arbitrary entity names (e.g.,
# from inject_sequence or malicious publishers) could balloon the dict.
# When the cap is reached, new (domain, entity) pairs are logged and
# dropped rather than creating a new baseline.
MAX_BASELINE_ENTITIES = 10_000

# ---------------------------------------------------------------------------
# Per-entity baseline
# ---------------------------------------------------------------------------


@dataclass
class EntityBaseline:
    """Tracks EWMA mean/variance and HST model for one (domain, entity)."""

    ewma_mean: float = 0.0
    ewma_var: float = 0.0
    observation_count: int = 0
    hst: HalfSpaceTrees = field(
        default_factory=lambda: HalfSpaceTrees(
            n_trees=15,
            height=8,
            window_size=50,
            seed=42,
        )
    )

    @property
    def ewma_std(self) -> float:
        return math.sqrt(max(self.ewma_var, 0.0))

    def update(self, value: float, alpha: float) -> None:
        self.observation_count += 1
        if self.observation_count == 1:
            self.ewma_mean = value
            self.ewma_var = 0.0
        else:
            diff = value - self.ewma_mean
            self.ewma_mean += alpha * diff
            self.ewma_var = (1 - alpha) * (self.ewma_var + alpha * diff * diff)
        self.hst.learn_one({"value": value})

    def score(self, value: float) -> tuple[float, float]:
        std = self.ewma_std
        if std < 0.01:
            deviation = 0.0
        else:
            deviation = abs(value - self.ewma_mean) / std
        hst_score: float = self.hst.score_one({"value": value})
        return deviation, hst_score

    def to_state_dict(self) -> dict:
        return {
            "ewma_mean": self.ewma_mean,
            "ewma_var": self.ewma_var,
            "observation_count": self.observation_count,
        }

    @classmethod
    def from_state_dict(cls, state: dict) -> EntityBaseline:
        bl = cls()
        bl.ewma_mean = state.get("ewma_mean", 0.0)
        bl.ewma_var = state.get("ewma_var", 0.0)
        bl.observation_count = state.get("observation_count", 0)
        return bl


def classify_severity(
    deviation: float,
    hst_score: float,
    medium_sigma: float,
    high_sigma: float,
) -> str:
    if deviation >= high_sigma or hst_score >= 0.9:
        return "high"
    if deviation >= medium_sigma or hst_score >= 0.8:
        return "medium"
    return "low"


# ---------------------------------------------------------------------------
# Threshold loading
# ---------------------------------------------------------------------------


def load_domain_thresholds(pm: PersistenceManager, domain: str) -> dict:
    stored = pm.load_thresholds(domain)
    if stored is not None:
        merged = {**DEFAULT_THRESHOLDS, **stored}
        log.info("Loaded thresholds for domain '%s': %s", domain, merged)
        return merged
    return dict(DEFAULT_THRESHOLDS)


# ---------------------------------------------------------------------------
# Baseline loading from persistence
# ---------------------------------------------------------------------------


def load_persisted_baselines(
    pm: PersistenceManager,
    r: redis.Redis,
) -> dict[tuple[str, str], EntityBaseline]:
    """Scan Redis for existing augur:profile:* keys and restore baselines."""
    baselines: dict[tuple[str, str], EntityBaseline] = {}
    cursor = 0
    while True:
        cursor, keys = r.scan(cursor, match="augur:profile:*", count=100)
        for key in keys:
            key_str = key.decode() if isinstance(key, bytes) else key
            parts = key_str.split(":")
            if len(parts) >= 4:
                domain = parts[2]
                entity = parts[3]
                state = pm.load_baseline(domain, entity)
                if state is not None:
                    bl = EntityBaseline.from_state_dict(state)
                    baselines[(domain, entity)] = bl
                    log.info(
                        "Restored baseline (%s, %s): %d observations, mean=%.2f",
                        domain,
                        entity,
                        bl.observation_count,
                        bl.ewma_mean,
                    )
        if cursor == 0:
            break
    return baselines


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

    # Restore persisted baselines
    baselines: dict[tuple[str, str], EntityBaseline] = load_persisted_baselines(
        pm,
        redis_client,
    )
    if baselines:
        log.info("Restored %d baselines from persistence", len(baselines))

    # Cache of per-domain thresholds (loaded on first encounter)
    thresholds_cache: dict[str, dict] = {}

    def get_thresholds(domain: str) -> dict:
        if domain not in thresholds_cache:
            thresholds_cache[domain] = load_domain_thresholds(pm, domain)
        return thresholds_cache[domain]

    # -- Message handler -----------------------------------------------------
    async def on_event(msg: nats.aio.client.Msg) -> None:
        try:
            event = PerceptionEvent.from_json(msg.data)
        except (ValueError, TypeError, UnicodeDecodeError) as exc:
            log.warning("Bad perception event: %s", exc)
            return

        domain = event.domain
        entity = event.entity
        value = event.value
        key = (domain, entity)
        th = get_thresholds(domain)

        # Get or create baseline. LEAK-10: refuse new entries once the
        # cap is reached so the dict cannot grow unbounded under arbitrary
        # entity names.
        if key not in baselines:
            if len(baselines) >= MAX_BASELINE_ENTITIES:
                log.warning(
                    "Baseline cap reached (%d entities); dropping new %s/%s",
                    MAX_BASELINE_ENTITIES,
                    domain,
                    entity,
                )
                return
            baselines[key] = EntityBaseline()

        bl = baselines[key]

        # Score BEFORE updating
        deviation, hst_score = bl.score(value)
        is_trained = bl.observation_count >= th["min_observations"]

        # Update baseline
        bl.update(value, th["ewma_alpha"])

        # Persist baseline
        try:
            pm.save_baseline(domain, entity, bl.to_state_dict())
        except redis.RedisError as exc:
            log.error("Failed to persist baseline (%s, %s): %s", domain, entity, exc)

        # Persist event to history
        try:
            pm.append_event(event)
        except redis.RedisError as exc:
            log.error("Failed to persist event: %s", exc)

        # Build label for logging
        ctx = event.context
        label = ctx.get("move_san", ctx.get("label", f"{value}{event.unit}"))

        log.info(
            "[%s/%s] %s  value=%.2f%s  ewma=%.2f  std=%.2f  "
            "dev=%.2f\u03c3  hst=%.3f  trained=%s",
            domain,
            entity,
            label,
            value,
            event.unit,
            bl.ewma_mean,
            bl.ewma_std,
            deviation,
            hst_score,
            is_trained,
        )

        if not is_trained:
            log.info(
                "  \u2514\u2500 Building baseline (%d/%d observations)",
                bl.observation_count,
                th["min_observations"],
            )
            return

        is_anomaly = (
            deviation >= th["sigma_threshold"] or hst_score >= th["hst_threshold"]
        )
        if not is_anomaly:
            return

        severity = classify_severity(
            deviation,
            hst_score,
            th["severity_medium_sigma"],
            th["severity_high_sigma"],
        )

        # Anomaly payload includes full event context plus compat aliases
        anomaly_payload = {
            "domain": domain,
            "stream_id": event.stream_id,
            "entity": entity,
            "event_type": event.event_type,
            "value": round(value, 3),
            "unit": event.unit,
            "context": ctx,
            "session_id": event.session_id,
            "baseline_mean": round(bl.ewma_mean, 3),
            "baseline_std": round(bl.ewma_std, 3),
            "deviation_score": round(deviation, 3),
            "anomaly_score": round(hst_score, 3),
            "severity": severity,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            # Compat aliases for downstream consumers not yet updated
            "player": entity,
            "move": ctx.get("move_san", label),
            "move_number": ctx.get("move_number", 0),
            "think_time": round(value, 3),
        }

        log.warning(
            "  \u26a0 ANOMALY [%s] %s/%s: value=%.2f  dev=%.1f\u03c3  hst=%.3f",
            severity.upper(),
            domain,
            entity,
            value,
            deviation,
            hst_score,
        )

        try:
            await nc.publish(PUBLISH_SUBJECT, json.dumps(anomaly_payload).encode())
            log.info("  Published anomaly to %s", PUBLISH_SUBJECT)
        except Exception as exc:
            log.error("  NATS publish failed: %s", exc)

        try:
            redis_client.set(REDIS_KEY_ANOMALY, json.dumps(anomaly_payload))
            log.info("  Wrote anomaly to Redis key %s", REDIS_KEY_ANOMALY)
        except redis.RedisError as exc:
            log.error("  Redis write failed: %s", exc)

    # -- Subscribe and wait --------------------------------------------------
    sub = await nc.subscribe(SUBSCRIBE_SUBJECT, cb=on_event)
    log.info("Subscribed to %s (wildcard — all perception domains)", SUBSCRIBE_SUBJECT)
    log.info("Default thresholds: %s", DEFAULT_THRESHOLDS)
    log.info("Waiting for perception events...")

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
