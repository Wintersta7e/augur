"""Domain-agnostic anomaly detector.

Subscribes to NATS 'augur.sensus.>' (wildcard), parses incoming
messages as PerceptionEvent, scores each observation using per-(domain,
entity) EWMA baselines and River HalfSpaceTrees, then publishes anomalies
to 'augur.vigil.anomaly' and Redis.

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
from river import drift as river_drift
from river.anomaly import HalfSpaceTrees

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tabula.config import AugurConfig
from tabula.connections import connect_redis
from tabula.contracts import PerceptionEvent
from tabula.heartbeat import start_heartbeat
from tabula.persistence import PersistenceManager

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
SUBSCRIBE_SUBJECT = "augur.sensus.>"
PUBLISH_SUBJECT = "augur.vigil.anomaly"
REDIS_KEY_ANOMALY = "augur:vigil:last_anomaly"

# Default thresholds (overridden per-domain via PersistenceManager)
DEFAULT_THRESHOLDS = {
    "min_observations": 15,
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

# 1C drift reset: cap the restart variance at this multiple of |Δmean| so a
# reset never carries the old (inflated) std forward (spec §6).
DRIFT_RESTART_STD_CAP_FACTOR = 4.0

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

    # 1C drift detector (in-memory; NOT persisted — a process restart is a
    # natural detector reset). Plain class attrs (no annotation) so the
    # @dataclass decorator does not treat them as fields; always reassigned
    # per-instance in enable_drift / _maybe_drift_reset.
    _drift = None
    _drift_cfg = None
    _cooldown_left = 0
    _just_reset = False
    drift_resets = 0

    @property
    def ewma_std(self) -> float:
        return math.sqrt(max(self.ewma_var, 0.0))

    def enable_drift(
        self,
        detector: str,
        *,
        min_observations: int,
        cooldown_obs: int,
        restart_std_factor: float,
    ) -> None:
        self._drift = (
            river_drift.ADWIN() if detector == "adwin" else river_drift.PageHinkley()
        )
        self._drift_cfg = (min_observations, cooldown_obs, restart_std_factor)
        self._cooldown_left = 0
        self.drift_resets = 0

    def update(self, value: float, alpha: float) -> None:
        self._just_reset = False
        mean_before = self.ewma_mean
        obs_before = self.observation_count  # pre-increment (drift trained-check)
        self.observation_count += 1
        if self.observation_count == 1:
            self.ewma_mean = value
            self.ewma_var = 0.0
        else:
            diff = value - self.ewma_mean
            self.ewma_mean += alpha * diff
            self.ewma_var = (1 - alpha) * (self.ewma_var + alpha * diff * diff)
        self.hst.learn_one({"value": value})
        self._maybe_drift_reset(value, mean_before, obs_before)

    def _maybe_drift_reset(
        self, value: float, mean_before: float, obs_before: int
    ) -> None:
        if self._drift is None or self._drift_cfg is None:
            return
        min_obs, cooldown_obs, restart_factor = self._drift_cfg
        if self._cooldown_left > 0:
            self._cooldown_left -= 1
        if obs_before < min_obs:
            return  # still warming up
        # Feed the RAW value. The detector is per-(domain, entity), so there is
        # no cross-entity scale domination to normalize away — and a z-score
        # self-normalizes as the EWMA tracks the shift, collapsing a sustained
        # level change into a single spike ADWIN can't detect. Raw values let
        # ADWIN see the sustained 10→30-style shift it is designed to catch.
        self._drift.update(value)
        if getattr(self._drift, "drift_detected", False) and self._cooldown_left == 0:
            delta = value - mean_before
            # Restart variance bounded and INDEPENDENT of the (possibly inflated)
            # std_before — never carry the old inflation forward (spec §6).
            restart_std = restart_factor * abs(delta)
            lo, hi = abs(delta) * 0.25, abs(delta) * DRIFT_RESTART_STD_CAP_FACTOR
            restart_std = min(max(restart_std, lo), hi)
            self.ewma_mean = value
            self.ewma_var = max(restart_std, 0.01) ** 2
            self.observation_count = max(1, min_obs // 2)
            self._cooldown_left = cooldown_obs
            self.drift_resets += 1
            self._just_reset = True
            # Fresh detector window (HST intentionally untouched — it self-recovers).
            self._drift = (
                river_drift.ADWIN()
                if isinstance(self._drift, river_drift.ADWIN)
                else river_drift.PageHinkley()
            )

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


def build_anomaly_payload(
    event: PerceptionEvent,
    *,
    deviation: float,
    hst_score: float,
    severity: str,
    mean_before: float,
    std_before: float,
    obs_before: int,
    drift_reset: bool,
    timestamp: str,
) -> dict:
    """Assemble the ``augur.vigil.anomaly`` payload from the DECISION-TIME
    (pre-update) baseline snapshot.

    Kept a pure function (taking the frozen snapshot explicitly, never the live
    baseline) so the spec §4.3 invariant — ``baseline_mean``/``baseline_std``
    are the PRE-update values, consistent with ``deviation_score`` — is
    structurally guaranteed and unit-testable. A regression to post-update
    values is impossible here because this function has no access to the
    updated baseline.
    """
    ctx = event.context
    label = ctx.get("move_san", ctx.get("label", f"{event.value}{event.unit}"))
    return {
        "domain": event.domain,
        "stream_id": event.stream_id,
        "entity": event.entity,
        "event_type": event.event_type,
        "value": round(event.value, 3),
        "unit": event.unit,
        "context": ctx,
        "session_id": event.session_id,
        "baseline_mean": round(mean_before, 3),
        "baseline_std": round(std_before, 3),
        "baseline_observation_count": obs_before,
        "deviation_score": round(deviation, 3),
        "drift_reset": drift_reset,
        "anomaly_score": round(hst_score, 3),
        "severity": severity,
        "timestamp": timestamp,
        # Compat aliases for downstream consumers not yet updated
        "player": event.entity,
        "move": ctx.get("move_san", label),
        "move_number": ctx.get("move_number", 0),
        "think_time": round(event.value, 3),
    }


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
    """Scan Redis for existing augur:vigil:profile:* keys and restore baselines."""
    baselines: dict[tuple[str, str], EntityBaseline] = {}
    cursor = 0
    while True:
        cursor, keys = r.scan(cursor, match="augur:vigil:profile:*", count=100)
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
    hb_task = (
        start_heartbeat(nc, "vigil", config.praefectus_heartbeat_interval_s)
        if config.praefectus_enabled
        else None
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

        # Enable the drift detector once per baseline (covers freshly-created and
        # Redis-restored baselines; detector state is in-memory, spec §6).
        if config.drift_detector_enabled and bl._drift is None:
            bl.enable_drift(
                config.drift_detector,
                min_observations=config.min_observations,
                cooldown_obs=config.drift_reset_cooldown_obs,
                restart_std_factor=config.drift_restart_std_factor,
            )

        # Score BEFORE updating
        deviation, hst_score = bl.score(value)
        is_trained = bl.observation_count >= th["min_observations"]

        # Freeze the DECISION-TIME (pre-update) baseline so the emitted
        # baseline_mean/std are consistent with deviation_score (also pre-update)
        # and the downstream outcome metric (spec 2026-06-09 §4.3). Persistence
        # still saves the UPDATED baseline below.
        mean_before = bl.ewma_mean
        std_before = bl.ewma_std
        obs_before = bl.observation_count

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

        # Anomaly payload from the pre-update snapshot (spec §4.3); the pure
        # builder guarantees baseline_mean/std are decision-time values.
        anomaly_payload = build_anomaly_payload(
            event,
            deviation=deviation,
            hst_score=hst_score,
            severity=severity,
            mean_before=mean_before,
            std_before=std_before,
            obs_before=obs_before,
            drift_reset=bool(bl._just_reset),
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

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
            # R2-ARCH-02: routed through PersistenceManager rather than
            # a bare redis_client.set so the write path matches the
            # read path (pm.load_last_anomaly).
            pm.save_last_anomaly(anomaly_payload)
            log.info("  Wrote anomaly to Redis via PersistenceManager")
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
        if hb_task is not None:
            hb_task.cancel()
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
