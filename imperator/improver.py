"""Imperator II runner: on augur.disciplina.complete (after a freshness gate), reason
over the self-model, log proposals, and (watch-first) apply the safe-clean class."""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import os
import sys
import time
from datetime import datetime

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import httpx  # noqa: E402
import nats  # noqa: E402

from tabula.config import AugurConfig  # noqa: E402
from tabula.connections import connect_redis  # noqa: E402
from tabula.heartbeat import start_heartbeat  # noqa: E402
from tabula.persistence import PersistenceManager  # noqa: E402
from imperator import proposals as P, reasoner, apply as A  # noqa: E402

log = logging.getLogger("imperator.improver")
_CONSUMED = frozenset({"augur.disciplina.complete", "augur.imperator.ii.trigger"})


def consumed(subject: str) -> bool:
    return subject in _CONSUMED


def parse_reflection_epoch(payload: dict) -> float:
    ts = payload.get("timestamp")
    if not ts:
        return 0.0
    try:
        return datetime.fromisoformat(str(ts)).timestamp()
    except (ValueError, TypeError):
        return 0.0


async def _emit(publish, subject, data):
    res = publish(subject, data)
    if inspect.isawaitable(res):
        await res


async def run_cycle(pm, cfg, *, now, session_id, generate_fn, client, publish):
    sm = pm.load_self_model()
    if sm is None or sm.get("schema_version") != 1:
        return
    recent = (sm.get("recent_self_tuning") or {}).get("value") or {}
    cands = [
        P.normalize_klass(p)
        for p in await generate_fn(sm, client=client, config=cfg, now=now)
    ]
    applied = {
        p["dedupe_key"] for p in cands if pm.is_proposal_applied(p["dedupe_key"])
    }
    for p in P.rank(cands):
        P.gate(p, cfg=cfg, recent_self_tuning=recent, applied_keys=applied)
        if p["status"] != "skipped":
            A.apply_proposal(pm, p, cfg=cfg, session_id=session_id)
        pm.save_proposal(p)
        if p["status"] == "applied":
            applied.add(p["dedupe_key"])
        await _emit(publish, "augur.imperator.proposal", json.dumps(p).encode())


async def _await_fresh(pm, epoch, timeout_s, tick_s=0.5):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        sm = pm.load_self_model()
        if sm and sm.get("generated_at", 0.0) >= epoch:
            return True
        await asyncio.sleep(tick_s)
    return False


async def run() -> None:
    config = AugurConfig.from_env()
    if not config.imperator_ii_enabled:
        log.info("Imperator II disabled; exiting.")
        return
    redis_client = connect_redis(config)
    pm = PersistenceManager(redis_client)
    nc = await nats.connect(
        config.nats_url, connect_timeout=config.nats_connect_timeout
    )
    http = httpx.AsyncClient()
    try:
        await http.get(f"{config.ollama_url}/api/tags", timeout=5.0)
    except Exception:
        log.warning("Ollama not reachable at startup; continuing")
    hb = start_heartbeat(nc, "imperator_ii", config.praefectus_heartbeat_interval_s)
    lock = asyncio.Lock()
    last_run = [0.0]

    async def on_msg(msg):
        if not consumed(msg.subject):
            return
        async with lock:
            now = time.time()
            if now - last_run[0] < config.imperator_ii_min_interval_s:
                return
            try:
                payload = json.loads(msg.data.decode()) if msg.data else {}
                epoch = parse_reflection_epoch(payload)
                if epoch and not await _await_fresh(
                    pm, epoch, config.imperator_ii_freshness_timeout_s
                ):
                    log.info("self-model did not fold reflection in time; skipping")
                    return
                last_run[0] = time.time()

                async def gen(sm, *, client, config, now):
                    return await reasoner.generate_proposals(
                        sm, client=client, config=config, now=now
                    )

                try:
                    await run_cycle(
                        pm,
                        config,
                        now=now,
                        session_id=payload.get("session_id"),
                        generate_fn=gen,
                        client=http,
                        publish=nc.publish,
                    )
                except reasoner.ReasonerError as exc:
                    await nc.publish(
                        "augur.imperator.failure",
                        json.dumps({"reason": exc.reason, "ts": now}).encode(),
                    )
            except Exception:
                log.warning("imperator II cycle failed; continuing", exc_info=True)

    sub = await nc.subscribe("augur.>", cb=on_msg)
    try:
        while True:
            await asyncio.sleep(3600)
    finally:
        hb.cancel()
        try:
            await hb
        except asyncio.CancelledError:
            pass
        try:
            await sub.unsubscribe()
        except Exception:
            log.debug("unsub failed", exc_info=True)
        await http.aclose()
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
