"""Praefectus monitor — the async supervision faculty. Subscribes augur.>,
derives graded health each tick, snapshots to Redis, and publishes debounced
augur.praefectus.health transitions. See the spec §5.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time

import nats

from praefectus import health as H
from praefectus.health import HEALTH_SUBJECT
from tabula.config import AugurConfig
from tabula.connections import connect_redis
from tabula.heartbeat import start_heartbeat
from tabula.persistence import PersistenceManager

log = logging.getLogger("praefectus.monitor")


def _transition_for(reason: str) -> str:
    """never_started/lost are deaths; everything else (consilium_stall, delivery_failures)
    is a degradation. Recovery is handled separately on the cleared path."""
    return "dead" if reason in ("never_started", "lost") else "degraded"


def _health_payload(
    states: dict, fac: str, reason: str, transition: str, now: float
) -> bytes:
    st = states[fac]
    return json.dumps(
        {
            "faculty": fac,
            "reason": reason,
            "transition": transition,  # "dead" | "degraded" | "recovered"
            "liveness": st.liveness_state,
            "activity": st.activity_state,
            "overall": st.overall_state,
            "ts": now,
        }
    ).encode()


async def tick(
    nc, pm: PersistenceManager, states, window, now, started_at, config
) -> None:
    """One evaluation cycle: compute health, publish entered/cleared transitions,
    snapshot. Extracted from run() so it is unit-testable without the bus loop."""
    report = H.evaluate(states, window, now, started_at, config)
    for fac, reason in report.entered:
        transition = _transition_for(reason)
        await nc.publish(
            HEALTH_SUBJECT, _health_payload(states, fac, reason, transition, now)
        )
        log.warning("praefectus: %s entered %s (%s)", fac, reason, transition)
    for fac, reason in report.cleared:
        await nc.publish(
            HEALTH_SUBJECT, _health_payload(states, fac, reason, "recovered", now)
        )
        log.info("praefectus: %s recovered from %s", fac, reason)
    pm.save_health_snapshot(H.summarize(report))


def record_message(
    states, window, subject: str, raw: bytes, now: float, config
) -> None:
    """Dispatch one bus message into the registry/window: a heartbeat stamps
    liveness, a faculty work-subject records activity. Pure (mutates states/window,
    no I/O). Extracted from run()'s on_msg so the malformed-payload robustness is
    directly unit-testable; well-formed {faculty, ts} payloads behave unchanged."""
    kind, _ = H.classify_event(subject)
    if kind == "heartbeat":
        try:
            data = json.loads(raw)
            if not isinstance(data, dict):
                return
            ts = float(data.get("ts", now))
        except (ValueError, TypeError):
            return
        H.record_heartbeat(states, data.get("faculty"), ts)
    elif kind == "activity":
        try:
            data = json.loads(raw)
        except (ValueError, TypeError):
            data = {}
        H.record_activity(states, window, subject, data, now, config)


async def run() -> None:
    config = AugurConfig.from_env()
    if not config.praefectus_enabled:
        log.info("praefectus disabled (AUGUR_PRAEFECTUS_ENABLED=false); exiting")
        return
    started_at = time.time()
    redis_client = connect_redis(config)
    pm = PersistenceManager(redis_client)
    nc = await nats.connect(
        config.nats_url, connect_timeout=config.nats_connect_timeout
    )
    log.info("praefectus: NATS connected (%s)", config.nats_url)

    states = H.initial_states(started_at)
    window = H.ActivityWindow()
    hb_task = start_heartbeat(nc, "praefectus", config.praefectus_heartbeat_interval_s)

    async def on_msg(msg) -> None:
        record_message(states, window, msg.subject, msg.data, time.time(), config)

    sub = await nc.subscribe("augur.>", cb=on_msg)
    log.info("praefectus: subscribed augur.> (tick=%.1fs)", config.praefectus_tick_s)
    try:
        while True:
            await asyncio.sleep(config.praefectus_tick_s)
            await tick(nc, pm, states, window, time.time(), started_at, config)
    finally:
        hb_task.cancel()
        try:
            await hb_task
        except asyncio.CancelledError:
            pass
        try:
            await sub.unsubscribe()
        except Exception as exc:  # noqa: BLE001
            log.debug("praefectus: unsubscribe failed during shutdown: %s", exc)
        await nc.close()
        try:
            redis_client.close()
        except Exception:  # noqa: BLE001
            pass


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
