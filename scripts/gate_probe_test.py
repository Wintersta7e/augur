#!/usr/bin/env python3
"""Live Limen gate-arm probes against a running deploy stack.

Trips the severity-independent gate arms deterministically and scores the
observed augur.limen.suppressed / augur.consilium.advice traffic. Per spec
§5, a non-exempt HIGH bypasses the learned suppressors but REMAINS subject
to refractory_burden, cost_tier, and anti_starvation — so the probes run on
HIGH spikes with baseline re-anchoring (interleaved normal values keep the
EWMA anchored so every spike stays anomalous; a naive repeated spike is
absorbed by the adaptive baseline and stops reaching the gate at all).

  P1 refractory      — a fresh HIGH on a just-delivered channel inside the
                       absolute refractory window (45s) is suppressed
                       (arm=refractory_burden).
  P2 starvation bound— sustained anchored HIGH spikes: refractory suppresses
                       repeats, the consecutive-suppression streak must be
                       broken by a FIRE before it exceeds 8 (invariant D /
                       anti_starvation_release).
  P3 exempt          — HIGH + correlated (typing+activity) always fires
                       (invariant B): advice on the typing channel, zero
                       suppressions on it.
  P4 records         — every suppression is persisted (invariant A) and
                       carries the MRT/IPW fields (decision_id, p_withhold,
                       arm).
  P5 medium band     — informational: compute a spike from the LIVE baseline
                       stats (mean + 3.0*std, inside the 2.5–4.0σ medium
                       band) on fresh entities; report whether MEDIUM is
                       reachable (→ reservoir engagement) or HST dominance
                       forces HIGH/LOW (documented environment behavior).
                       Both outcomes pass; the row records which.

Advice/suppression events are matched by entity (advice carries the primary
entity in its "player" compat alias; suppressed events carry "entity").

Run on a quiet stack (no other drivers) — the gate's pressure/refractory
arms are stateful and cross-traffic skews attribution. A fresh augur-state
flush beforehand gives the cleanest read.

Usage: .venv/bin/python scripts/gate_probe_test.py [--llm-wait 150]
"""

from __future__ import annotations

import argparse
import asyncio
import json
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
_APOOL = [35.2, 41.8, 38.5, 44.0, 36.1, 42.3, 39.0, 43.1, 37.4, 40.6]


def iso() -> str:
    return datetime.now(timezone.utc).isoformat()


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm-wait", type=float, default=150.0)
    args = ap.parse_args()

    rid = uuid.uuid4().hex[:8]
    r = redis_lib.Redis.from_url(
        REDIS_URL, decode_responses=True, socket_connect_timeout=5
    )
    pm = PersistenceManager(r)
    nc = await nats.connect(NATS_URL, connect_timeout=5)

    anomalies: list[dict] = []
    stream: list[tuple[str, str, dict]] = []  # (kind, entity, payload) ordered

    async def on_msg(msg):
        try:
            data = json.loads(msg.data.decode())
        except Exception:
            return
        if msg.subject == "augur.vigil.anomaly":
            anomalies.append(data)
        elif msg.subject == "augur.consilium.advice":
            stream.append(("F", data.get("player"), data))
            print(
                f"  <- advice player={data.get('player')} "
                f"tier={data.get('tier')} sev={data.get('severity')}",
                flush=True,
            )
        elif msg.subject == "augur.limen.suppressed":
            stream.append(("S", data.get("entity"), data))
            print(
                f"  <- suppressed entity={data.get('entity')} "
                f"arm={data.get('arm')} reason={data.get('reason')}",
                flush=True,
            )

    for subj in (
        "augur.vigil.anomaly",
        "augur.consilium.advice",
        "augur.limen.suppressed",
    ):
        await nc.subscribe(subj, cb=on_msg)
    await asyncio.sleep(0.5)

    def fires(entity):
        return [p for k, e, p in stream if k == "F" and e == entity]

    def sups(entity):
        return [p for k, e, p in stream if k == "S" and e == entity]

    def seq(entity):
        return [k for k, e, _ in stream if e == entity]

    async def emit(
        domain, entity, value, sid, context=None, event_type="pause", unit="seconds"
    ):
        ev = PerceptionEvent(
            domain=domain,
            stream_id=domain,
            entity=entity,
            event_type=event_type,
            value=float(value),
            unit=unit,
            context=context or {"avg_wpm": 40},
            timestamp=iso(),
            session_id=sid,
        )
        await nc.publish(f"augur.sensus.{domain}", ev.to_bytes())

    async def baseline(entity, sid, domain="typing", pool=_POOL, n=20, **kw):
        for i in range(n):
            await emit(domain, entity, pool[i % len(pool)], sid, **kw)
            await asyncio.sleep(0.03)
        await asyncio.sleep(1.2)

    async def anchor(entity, sid, n=5):
        """Re-anchor the EWMA with normal values so the next spike stays HIGH."""
        for i in range(n):
            await emit("typing", entity, _POOL[i % 10], sid)
            await asyncio.sleep(0.05)

    async def wait_for(pred, timeout, tick=0.5):
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout:
            if pred():
                return True
            await asyncio.sleep(tick)
        return pred()

    rows: list[tuple[str, bool, str]] = []

    def row(name, ok, detail):
        rows.append((name, ok, detail))
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}", flush=True)

    # ---------------- P1: absolute refractory right after a delivery
    print("\n=== P1: absolute refractory (45s) after a HIGH delivery ===", flush=True)
    e1, sid1 = f"refr_{rid}", f"probe1-{rid}"
    await baseline(e1, sid1)
    await emit("typing", e1, 19.0, sid1)
    delivered = await wait_for(lambda: len(fires(e1)) >= 1, args.llm_wait)
    got_refr = False
    if delivered:
        await anchor(e1, sid1)
        n_sup = len(sups(e1))
        await emit("typing", e1, 19.4, sid1)
        got_refr = await wait_for(
            lambda: any(p.get("arm") == "refractory_burden" for p in sups(e1)[n_sup:]),
            20,
        )
    row(
        "P1 in-refractory HIGH suppressed (arm=refractory_burden)",
        delivered and got_refr,
        f"delivered={delivered} suppressions="
        f"{[(p.get('arm'), p.get('reason')) for p in sups(e1)]}",
    )

    # ---------------- P2: sustained spikes -> streak bounded by release
    print(
        "\n=== P2: anchored HIGH repetition -> refractory streak, "
        "starvation release ===",
        flush=True,
    )
    e2, sid2 = f"strv_{rid}", f"probe2-{rid}"
    await baseline(e2, sid2)
    for i in range(12):
        await anchor(e2, sid2, n=4)
        await emit("typing", e2, 18.5 + 0.3 * i, sid2)
        await asyncio.sleep(11.0)
    await asyncio.sleep(20.0)
    s2, f2, q2 = sups(e2), fires(e2), seq(e2)
    max_streak = cur = 0
    for k in q2:
        cur = cur + 1 if k == "S" else 0
        max_streak = max(max_streak, cur)
    arms2 = {p.get("arm") for p in s2}
    row(
        "P2 refractory suppresses anchored repeats",
        len(s2) >= 3 and "refractory_burden" in arms2,
        f"suppressed={len(s2)} fired={len(f2)} arms={sorted(a for a in arms2 if a)}",
    )
    row(
        "P2 invariant D: consecutive-suppression streak <= 8 with later fire",
        0 < max_streak <= 8 and len(f2) >= 1,
        f"max_streak={max_streak} sequence={''.join(q2)}",
    )

    # ---------------- P3: HIGH + correlated is exempt (invariant B)
    print("\n=== P3: high+correlated always fires (invariant B) ===", flush=True)
    e3t, e3a, sid3 = f"exm_{rid}", f"exa_{rid}", f"probe3-{rid}"
    await baseline(e3t, sid3)
    await baseline(
        e3a,
        sid3,
        domain="activity_intensity",
        pool=_APOOL,
        context={"focused_app": e3a},
        event_type="intensity_sample",
        unit="ipm",
    )
    await emit(
        "activity_intensity",
        e3a,
        400.0,
        sid3,
        context={"focused_app": e3a},
        event_type="intensity_sample",
        unit="ipm",
    )
    await asyncio.sleep(1.0)
    await emit("typing", e3t, 19.0, sid3)
    got_adv3 = await wait_for(lambda: len(fires(e3t)) >= 1, args.llm_wait)
    adv3 = fires(e3t)
    row(
        "P3 exempt fires (advice, zero suppression on typing)",
        got_adv3 and not sups(e3t),
        f"advice={len(adv3)} suppressed={len(sups(e3t))} "
        f"sev={adv3[0].get('severity') if adv3 else '?'} "
        f"correlated={adv3[0].get('correlation_found') if adv3 else '?'}",
    )

    # ---------------- P4: invariant A + MRT fields on silence records
    print("\n=== P4: silence records persisted with MRT fields ===", flush=True)
    recs = pm.load_silence_records(limit=100)
    ours = [x for x in recs if rid in str(x.get("entity", ""))]
    total_sup = len([1 for k, _, _ in stream if k == "S"])
    with_fields = [
        x for x in ours if x.get("decision_id") and x.get("arm") and "p_withhold" in x
    ]
    row(
        "P4 every suppression persisted (invariant A)",
        len(ours) >= total_sup > 0,
        f"records_for_run={len(ours)} nats_suppressed={total_sup}",
    )
    row(
        "P4 records carry MRT/IPW fields",
        bool(ours) and len(with_fields) == len(ours),
        f"{len(with_fields)}/{len(ours)} have decision_id+arm+p_withhold",
    )

    # ---------------- P5: medium-band reachability (informational)
    print("\n=== P5: computed medium-band spike (informational) ===", flush=True)
    observed = []
    for attempt in range(3):
        e5, sid5 = f"med{attempt}_{rid}", f"probe5-{attempt}-{rid}"
        await baseline(e5, sid5, n=30)
        stats = pm.load_baseline("typing", e5) or {}
        mean = float(stats.get("ewma_mean", 3.55))
        std = float(stats.get("ewma_var", 0.09)) ** 0.5 or 0.3
        target = mean + 3.0 * std
        n0 = len(anomalies)
        await emit("typing", e5, target, sid5)
        seen = await wait_for(lambda: len(anomalies) > n0, 8)
        sev = anomalies[-1].get("severity") if seen else "none"
        observed.append((round(target, 2), sev))
        print(
            f"  attempt {attempt}: spike={target:.2f} "
            f"(mean={mean:.2f} std={std:.2f}) -> severity={sev}",
            flush=True,
        )
        if sev == "medium":
            got_res = await wait_for(
                lambda: any(
                    p.get("arm") == "coincidence_evidence_reservoir" for p in sups(e5)
                ),
                15,
            )
            row(
                "P5 medium reachable -> reservoir suppresses first single",
                got_res,
                f"observed={observed} reservoir={got_res}",
            )
            break
    else:
        row(
            "P5 medium band documented (HST/EWMA dominance, no medium in 3 "
            "computed attempts)",
            True,
            f"observed={observed} — learned-arm live coverage comes from the "
            f"soak's organic traffic instead",
        )

    # ---------------- Report
    print("\n" + "=" * 72)
    print("GATE PROBE REPORT")
    print("=" * 72)
    passed = sum(1 for _, ok, _ in rows if ok)
    for name, ok, detail in rows:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}\n          {detail}")
    print("=" * 72)
    print(f"OVERALL: {passed}/{len(rows)} PASS")

    await nc.drain()
    r.close()
    return 0 if passed == len(rows) else 2


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
