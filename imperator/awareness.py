"""Imperator I runner: subscribe augur.>, decay + recompute the two read-models on a
tick, persist + publish-on-material-change, heartbeat. Read-only on every other faculty.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import sys
import time

# Project-root path shim — this module is launched by file path from run_augur.sh.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import nats  # noqa: E402

from tabula.config import AugurConfig  # noqa: E402
from tabula.connections import connect_redis  # noqa: E402
from tabula.heartbeat import start_heartbeat  # noqa: E402
from tabula.persistence import PersistenceManager  # noqa: E402
from imperator import auspices as auspices_mod  # noqa: E402
from imperator import self_model as self_model_mod  # noqa: E402
from imperator import sources  # noqa: E402

log = logging.getLogger("imperator.awareness")

# Only signals with NO standing Redis home are held in the live stream; everything
# else (advice, suppression, feedback, health) is read from Redis in sources.gather.
_CONSUMED = frozenset({"augur.nexus.detected", "augur.vigil.anomaly"})
_SALIENCE_BUCKET = 0.05


def consumed(subject: str) -> bool:
    return subject in _CONSUMED


def apply_event(state: dict, subject: str, data: dict, now: float) -> None:
    """Update in-memory stream state from one consumed event (no await)."""
    if subject == "augur.nexus.detected":
        state["escalation_tier"] = data.get("combined_severity")
        state["escalation_tier_ts"] = now
        if data.get("correlation_found"):
            state["has_active_correlation"] = True
            state["active_correlations"] = {
                "involved_domains": data.get("involved_domains"),
                "escalation_rule": data.get("escalation_rule"),
            }
            state["active_correlations_ts"] = now
            state["correlation_span_s"] = float(data.get("correlation_span_s") or 60.0)
    elif subject == "augur.vigil.anomaly":
        sev = {"low": 1, "medium": 2, "high": 3}.get(
            str(data.get("severity", "")).lower(), 0
        )
        state["anomaly_load"] = 0.7 * state.get("anomaly_load", 0.0) + 0.3 * sev
        state["anomaly_load_ts"] = now


def decay_stream(state: dict, now: float, cfg) -> None:
    """Time-decay anomaly_load and expire stale correlation / escalation tier each tick."""
    tau = cfg.imperator_salience_window_s
    last = state.get("anomaly_load_ts")
    if last is not None and state.get("anomaly_load"):
        state["anomaly_load"] *= math.exp(-max(0.0, now - last) / tau)
        state["anomaly_load_ts"] = now
    ct = state.get("active_correlations_ts")
    if ct is not None and now - ct > state.get("correlation_span_s", 60.0) + 30.0:
        state["has_active_correlation"] = False
        state["active_correlations"] = None
        state["active_correlations_ts"] = None
    et = state.get("escalation_tier_ts")
    if et is not None and now - et > tau:
        state["escalation_tier"] = "quiescent"
        state["escalation_tier_ts"] = now


def _comparable(snap: dict) -> dict:
    """Project to comparable content: drop timestamps, round floats, bucket headline scores."""
    out = {}
    for k, cell in snap.items():
        if k in ("generated_at", "as_of"):
            continue
        if isinstance(cell, dict) and "value" in cell:
            v = cell["value"]
            if isinstance(v, float):
                v = (
                    round(v / _SALIENCE_BUCKET) * _SALIENCE_BUCKET
                    if k in ("salience", "competence")
                    else round(v, 2)
                )
            out[k] = v
        else:
            out[k] = cell
    return out


def materially_changed(prev: dict, new: dict) -> bool:
    if not prev:
        return True
    return _comparable(prev) != _comparable(new)


async def tick(nc, pm, state, now, cfg, last):
    decay_stream(state, now, cfg)
    inputs = sources.gather(pm, dict(state), now, cfg)
    ausp = auspices_mod.compute_auspices(inputs, now)
    selfm = self_model_mod.compute_self_model(inputs, now)
    pm.save_auspices(ausp)
    pm.save_self_model(selfm)
    if materially_changed(last.get("auspices", {}), ausp):
        await nc.publish("augur.imperator.auspices", json.dumps(ausp).encode())
        last["auspices"] = ausp
    if materially_changed(last.get("self_model", {}), selfm):
        await nc.publish("augur.imperator.self_model", json.dumps(selfm).encode())
        last["self_model"] = selfm


async def run() -> None:
    config = AugurConfig.from_env()
    if not config.imperator_enabled:
        log.info("Imperator disabled (imperator_enabled=False); exiting.")
        return
    redis_client = connect_redis(config)
    pm = PersistenceManager(redis_client)
    nc = await nats.connect(
        config.nats_url, connect_timeout=config.nats_connect_timeout
    )
    # Heartbeat at Praefectus's interval (shared knob): Praefectus's stale/dead
    # thresholds are tuned to this cadence; a separate knob would desync supervision.
    hb_task = start_heartbeat(nc, "imperator", config.praefectus_heartbeat_interval_s)
    state: dict = {}
    last: dict = {}

    async def on_msg(msg):
        try:
            if not consumed(msg.subject):
                return
            apply_event(state, msg.subject, json.loads(msg.data.decode()), time.time())
        except Exception:
            log.debug(
                "imperator on_msg ignored a bad event on %s", msg.subject, exc_info=True
            )

    sub = await nc.subscribe("augur.>", cb=on_msg)
    try:
        while True:
            await asyncio.sleep(config.imperator_tick_s)
            try:
                await tick(nc, pm, state, time.time(), config, last)
            except Exception:
                log.warning("imperator tick failed; continuing", exc_info=True)
    finally:
        hb_task.cancel()
        try:
            await hb_task
        except asyncio.CancelledError:
            pass
        try:
            await sub.unsubscribe()
        except Exception:
            log.debug("unsubscribe failed", exc_info=True)
        await nc.close()
        try:
            redis_client.close()
        except Exception:
            log.debug("redis close failed", exc_info=True)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
