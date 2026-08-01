#!/usr/bin/env python3
"""Stress / scale / soak driver for a running Augur deploy stack.

Three phases:
  1. Burst (throughput + scale): inject a large number of events as fast as
     possible across many distinct entities -> measures events/sec and forces
     Vigil to allocate many baseline models (toward MAX_BASELINE_ENTITIES).
  2. Soak (endurance): sustained moderate-rate injection over a small HOT entity
     pool (so baselines form and the advice path stays exercised) for --soak-s.

Reports throughput + observed anomaly/advice/suppressed counts. Pair with
`docker stats` snapshots (see the stress wrapper) to catch memory leaks, and a
post-run baseline-entity count to confirm scale.

Usage:
  .venv/bin/python scripts/stress_soak.py [--entities 3000] [--burst 12000]
                                          [--hot 20] [--soak-s 180] [--rate 200]
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import nats
import redis as redis_lib

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tabula.contracts import PerceptionEvent  # noqa: E402
from tabula.persistence import PersistenceManager  # noqa: E402

NATS_URL = "nats://127.0.0.1:4222"
REDIS_URL = "redis://127.0.0.1:6379"
_POOL = [3.2, 3.8, 3.5, 4.0, 3.1, 3.9, 3.3, 3.7, 3.4, 3.6]


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--entities", type=int, default=3000, help="distinct entities for the burst"
    )
    ap.add_argument("--burst", type=int, default=12000, help="total burst events")
    ap.add_argument(
        "--hot", type=int, default=20, help="hot entity pool size for the soak"
    )
    ap.add_argument("--soak-s", type=float, default=180.0)
    ap.add_argument("--rate", type=float, default=200.0, help="soak events/sec")
    args = ap.parse_args()

    sid = f"stress-{uuid.uuid4().hex[:8]}"
    # Runs against the live stack: record this session as synthetic so its
    # injected perception is explicitly non-learnable, not merely unprovenanced.
    r = redis_lib.Redis.from_url(
        REDIS_URL, decode_responses=True, socket_connect_timeout=5
    )
    PersistenceManager(r).save_session_meta(
        sid, origin="synthetic", created_by="stress_soak"
    )
    nc = await nats.connect(NATS_URL, connect_timeout=5)
    counts = {"anomaly": 0, "advice": 0, "suppressed": 0}

    async def on(msg):
        if msg.subject == "augur.vigil.anomaly":
            counts["anomaly"] += 1
        elif msg.subject == "augur.consilium.advice":
            counts["advice"] += 1
        elif msg.subject == "augur.limen.suppressed":
            counts["suppressed"] += 1

    await nc.subscribe("augur.vigil.anomaly", cb=on)
    await nc.subscribe("augur.consilium.advice", cb=on)
    await nc.subscribe("augur.limen.suppressed", cb=on)
    await asyncio.sleep(0.5)

    async def emit(entity: str, value: float):
        ev = PerceptionEvent(
            domain="typing",
            stream_id="typing_stream",
            entity=entity,
            event_type="pause",
            value=float(value),
            unit="seconds",
            context={"avg_wpm": 45},
            timestamp=now_iso(),
            session_id=sid,
        )
        await nc.publish("augur.sensus.typing", ev.to_bytes())

    # ---- Phase 1: burst (throughput + scale) ----
    print(
        f"[burst] {args.burst} events across {args.entities} entities ...", flush=True
    )
    t0 = time.monotonic()
    for i in range(args.burst):
        await emit(f"e{i % args.entities}", _POOL[i % 10] + (i % 3) * 0.01)
        if i % 500 == 0:
            await nc.flush()
    await nc.flush()
    dt = time.monotonic() - t0
    print(
        f"[burst] sent {args.burst} in {dt:.1f}s = {args.burst / dt:.0f} ev/s",
        flush=True,
    )
    await asyncio.sleep(3.0)

    # ---- Phase 2: soak (endurance over a hot pool) ----
    print(
        f"[soak] {args.soak_s:.0f}s at ~{args.rate:.0f} ev/s over {args.hot} hot entities",
        flush=True,
    )
    interval = 1.0 / args.rate
    soak0 = time.monotonic()
    sent = 0
    while time.monotonic() - soak0 < args.soak_s:
        ent = f"hot{sent % args.hot}"
        # periodic spike keeps the anomaly/advice path warm under load
        # per-entity-varied baseline (avoid the zero-variance trap: an entity's
        # own occurrence index walks the pool) with a periodic spike.
        occ = sent // args.hot
        val = 18.0 if (sent % 1009 == 0) else (_POOL[occ % 10] + (occ % 7) * 0.03)
        await emit(ent, val)
        sent += 1
        if sent % 200 == 0:
            await nc.flush()
        await asyncio.sleep(interval)
    await nc.flush()
    print(f"[soak] sent {sent} events over {args.soak_s:.0f}s", flush=True)
    await asyncio.sleep(5.0)

    print(
        f"\n[result] observed: anomalies={counts['anomaly']} advice={counts['advice']} "
        f"suppressed={counts['suppressed']}",
        flush=True,
    )
    print(f"[result] total events sent ~{args.burst + sent}", flush=True)
    await nc.drain()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
