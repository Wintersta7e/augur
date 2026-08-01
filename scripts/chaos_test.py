#!/usr/bin/env python3
"""Chaos / fail-open test for a running Augur deploy stack.

Scenarios (PASS/FAIL each):
  1. Malformed events -> faculties drop them gracefully (no crash); a valid
     baseline+spike still produces an anomaly afterwards.
  2. Stop a faculty (docker stop, which `restart: unless-stopped` will NOT
     auto-restart) -> no cascade crash (other faculties stay alive + anomalies
     keep flowing), and Praefectus detects the victim is not-alive.
  3. Start it again -> Praefectus marks it alive (recovery).
  4. Watch-first invariant: Imperator II proposals stay logged, 0 applied,
     through the chaos.

Drives Docker via powershell.exe (Windows-side). Default victim: consilium.

Usage: .venv/bin/python scripts/chaos_test.py [--victim consilium]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
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


async def ps(cmd: str) -> tuple[int, str]:
    proc = await asyncio.create_subprocess_exec(
        "powershell.exe",
        "-NoProfile",
        "-Command",
        cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    out, _ = await proc.communicate()
    return proc.returncode or 0, out.decode("utf-8", "replace").strip()


def health(r) -> dict:
    raw = r.get("augur:praefectus:health")
    if not raw:
        return {}
    try:
        return {
            f: v.get("overall") for f, v in json.loads(raw).get("faculties", {}).items()
        }
    except Exception:
        return {}


async def poll(r, victim, want_alive: bool, timeout: float) -> bool:
    """Poll Praefectus health until victim's alive-state matches want_alive."""
    import time

    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        await asyncio.sleep(3.0)
        st = health(r).get(victim)
        if want_alive and st == "alive":
            return True
        if not want_alive and st not in ("alive", None):
            return True
    return (
        (health(r).get(victim) == "alive")
        if want_alive
        else (health(r).get(victim) not in ("alive", None))
    )


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--victim", default="consilium")
    args = ap.parse_args()
    victim = args.victim
    sid = f"chaos-{uuid.uuid4().hex[:8]}"
    user = f"user_{uuid.uuid4().hex[:6]}"

    r = redis_lib.Redis.from_url(
        REDIS_URL, decode_responses=True, socket_connect_timeout=5
    )
    # Runs against the live stack: record this session as synthetic so its
    # injected perception is explicitly non-learnable, not merely unprovenanced.
    PersistenceManager(r).save_session_meta(
        sid, origin="synthetic", created_by="chaos_test"
    )
    nc = await nats.connect(NATS_URL, connect_timeout=5)
    obs = {"anomaly": 0, "advice": 0}

    async def on(msg):
        if msg.subject == "augur.vigil.anomaly":
            obs["anomaly"] += 1
        elif msg.subject == "augur.consilium.advice":
            obs["advice"] += 1

    await nc.subscribe("augur.vigil.anomaly", cb=on)
    await nc.subscribe("augur.consilium.advice", cb=on)
    await asyncio.sleep(0.5)

    async def emit(entity, value):
        ev = PerceptionEvent(
            domain="typing",
            stream_id="typing_stream",
            entity=entity,
            event_type="pause",
            value=float(value),
            unit="seconds",
            context={"avg_wpm": 40},
            timestamp=now_iso(),
            session_id=sid,
        )
        await nc.publish("augur.sensus.typing", ev.to_bytes())

    async def baseline_then_spike(spike: float) -> bool:
        a0 = obs["anomaly"]
        for i in range(20):
            await emit(user, _POOL[i % 10])
            await asyncio.sleep(0.03)
        await asyncio.sleep(1.0)
        await emit(user, spike)
        await asyncio.sleep(3.0)
        return obs["anomaly"] > a0

    results = []

    # 1. malformed events, then a valid spike must still fire
    print("[1] malformed events ...", flush=True)
    await nc.publish("augur.sensus.typing", b"{not json")
    await nc.publish("augur.sensus.typing", json.dumps({"bogus": 1}).encode())
    await nc.publish("augur.sensus.typing", b"\xff\xfe\x00garbage")
    await nc.flush()
    await asyncio.sleep(2.0)
    ok1 = await baseline_then_spike(19.0)
    results.append(
        (
            "Malformed events ignored; valid flow intact",
            ok1,
            f"anomalies={obs['anomaly']}",
        )
    )

    # 2. stop the victim -> no cascade + Praefectus detects
    print(f"[2] docker stop {victim} ...", flush=True)
    await ps(f"docker stop augur-{victim}-1")
    detected = await poll(r, victim, want_alive=False, timeout=70)
    h_mid = health(r)
    no_cascade = all(
        v == "alive" for k, v in h_mid.items() if k != victim and v is not None
    )
    # upstream still flowing while victim down (anomalies are upstream of consilium)
    a0 = obs["anomaly"]
    for i in range(20):
        await emit(user, _POOL[i % 10])
        await asyncio.sleep(0.03)
    await asyncio.sleep(1.0)
    await emit(user, 21.0)
    await asyncio.sleep(3.0)
    upstream_ok = obs["anomaly"] > a0
    results.append(
        (
            f"Praefectus detected {victim} down",
            detected,
            f"health[{victim}]={h_mid.get(victim)}",
        )
    )
    results.append(("No cascade crash (others alive)", no_cascade, str(h_mid)))
    results.append(
        (
            "Upstream pipeline flows while victim down",
            upstream_ok,
            f"anomalies={obs['anomaly']}",
        )
    )

    # 3. start the victim -> recovery
    print(f"[3] docker start {victim} ...", flush=True)
    await ps(f"docker start augur-{victim}-1")
    recovered = await poll(r, victim, want_alive=True, timeout=70)
    results.append(
        (
            f"{victim} recovered (Praefectus alive)",
            recovered,
            f"health[{victim}]={health(r).get(victim)}",
        )
    )

    # 4. watch-first invariant
    props = []
    try:
        if r.type("augur:imperator:proposals") == "list":
            props = [
                json.loads(p) for p in r.lrange("augur:imperator:proposals", 0, 30) if p
            ]
    except Exception:
        pass
    applied = [p for p in props if p.get("status") == "applied"]
    results.append(
        (
            "Watch-first holds (0 applied)",
            len(applied) == 0,
            f"{len(props)} stored, {len(applied)} applied",
        )
    )

    print("\n" + "=" * 64 + "\nCHAOS / FAIL-OPEN REPORT\n" + "=" * 64)
    allok = True
    for name, ok, detail in results:
        if not ok:
            allok = False
        print(f"  [{'PASS' if ok else 'FAIL'}] {name:42s} {detail}")
    print(f"  final health: {health(r)}")
    print("=" * 64)
    print("OVERALL:", "PASS" if allok else "FAIL")

    await nc.drain()
    r.close()
    return 0 if allok else 2


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
