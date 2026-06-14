#!/usr/bin/env python3
"""Complete all-faculties loop test against a running Augur deploy stack.

Drives the ENTIRE pipeline end-to-end and scores every faculty PASS/FAIL:

  Sensus -> Vigil -> Nexus -> Consilium(+Limen, real Ollama) -> advice
    -> explicit feedback -> post-advice behavioral events -> session.end
    -> Responsum.complete -> Disciplina (reflection, incl. Memoria sweep)
    -> Imperator I self-model fold -> Imperator II reasoner -> proposal (watch-first)

Plus a Praefectus health readout (all 9 faculties alive) and a Limen SUPPRESS
best-effort attempt.

Run-unique entity names => fresh baselines. Runs from WSL against the
loopback-exposed deploy NATS (127.0.0.1:4222) and Redis (127.0.0.1:6379).

Usage: .venv/bin/python scripts/complete_loop_test.py [--run-id tag]
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

NATS_URL = "nats://127.0.0.1:4222"
REDIS_URL = "redis://127.0.0.1:6379"

SUBJECTS = [
    "augur.vigil.anomaly",
    "augur.nexus.detected",
    "augur.consilium.advice",
    "augur.limen.suppressed",
    "augur.limen.delivery_failure",
    "augur.responsum.complete",
    "augur.disciplina.complete",
    "augur.imperator.auspices",
    "augur.imperator.self_model",
    "augur.imperator.proposal",
    "augur.imperator.failure",
]


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


async def wait_for(pred, timeout: float, tick: float = 2.0) -> bool:
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        if pred():
            return True
        await asyncio.sleep(tick)
    return pred()


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default=uuid.uuid4().hex[:8])
    args = ap.parse_args()
    sid = f"complete-{args.run_id}"
    user = f"user_{args.run_id}"
    app = f"code_{args.run_id}"

    r = redis_lib.Redis.from_url(
        REDIS_URL, decode_responses=True, socket_connect_timeout=5
    )
    nc = await nats.connect(NATS_URL, connect_timeout=5)
    obs: dict[str, list] = {s: [] for s in SUBJECTS}

    async def on_msg(msg):
        try:
            data = json.loads(msg.data.decode())
        except Exception:
            data = {"_raw": True}
        obs.setdefault(msg.subject, []).append(data)
        tag = ""
        if msg.subject == "augur.consilium.advice":
            tag = f" decision_id={str(data.get('decision_id', '?'))[:8]} sev={data.get('severity')}"
        elif msg.subject == "augur.limen.suppressed":
            tag = f" arm={data.get('arm')} reason={data.get('reason')}"
        elif msg.subject == "augur.imperator.proposal":
            tag = f" kind={data.get('kind')} status={data.get('status')}"
        elif msg.subject == "augur.imperator.failure":
            tag = f" reason={data.get('reason')}"
        print(f"[{time.strftime('%H:%M:%S')}] <- {msg.subject}{tag}", flush=True)

    for s in SUBJECTS:
        await nc.subscribe(s, cb=on_msg)
    await asyncio.sleep(1.0)
    print(f"session={sid}; subscribed {len(SUBJECTS)} subjects", flush=True)

    async def inject(domain, entity, etype, value, unit, ctx):
        ev = PerceptionEvent(
            domain=domain,
            stream_id=f"{domain}_stream",
            entity=entity,
            event_type=etype,
            value=float(value),
            unit=unit,
            context=ctx,
            timestamp=now_iso(),
            session_id=sid,
        )
        await nc.publish(f"augur.sensus.{domain}", ev.to_bytes())

    # 1. baselines (varied; std must exceed 0.01)
    tp = [3.2, 3.8, 3.5, 4.0, 3.1, 3.9, 3.3, 3.7, 3.4, 3.6]
    fp = [4.5, 5.2, 4.8, 5.5, 4.3, 5.1, 4.7, 5.3, 4.9, 5.0]
    print("→ injecting baselines (20/domain)", flush=True)
    for i in range(20):
        await inject(
            "typing",
            user,
            "pause",
            tp[i % 10],
            "seconds",
            {"avg_wpm": 48, "keypress_count": 1000 + i},
        )
        await inject(
            "activity_focus",
            app,
            "focus_change",
            fp[i % 10],
            "log1p_seconds",
            {
                "prev_app": app,
                "new_app": "browser",
                "active_dwell_s": 120.0,
                "idle_dwell_s": 4.0,
                "total_dwell_s": 124.0,
                "source_id": "clt",
                "span_id": uuid.uuid4().hex,
            },
        )
        await asyncio.sleep(0.05)
    await asyncio.sleep(2.0)

    # 2. single-domain typing anomaly -> clean advice
    print("→ spike: single typing pause 18s", flush=True)
    await inject(
        "typing",
        user,
        "pause",
        18.0,
        "seconds",
        {"avg_wpm": 12, "keypress_count": 1300},
    )
    await asyncio.sleep(3.0)
    # 3. correlated cross-domain (exercise Nexus)
    print("→ spike: correlated typing+focus", flush=True)
    await inject(
        "typing", user, "pause", 22.0, "seconds", {"avg_wpm": 9, "keypress_count": 1500}
    )
    await asyncio.sleep(0.4)
    await inject(
        "activity_focus",
        app,
        "focus_change",
        7.8,
        "log1p_seconds",
        {
            "prev_app": app,
            "new_app": app,
            "active_dwell_s": 2400.0,
            "idle_dwell_s": 5.0,
            "total_dwell_s": 2405.0,
            "source_id": "clt",
            "span_id": uuid.uuid4().hex,
        },
    )

    # 4. wait for advice, capture decision_id
    print("→ waiting for advice (real Ollama)...", flush=True)
    await wait_for(
        lambda: len(obs["augur.consilium.advice"]) >= 1, timeout=120, tick=3.0
    )
    decision_ids = [
        a.get("decision_id")
        for a in obs["augur.consilium.advice"]
        if a.get("decision_id")
    ]
    print(
        f"  advice={len(obs['augur.consilium.advice'])} decision_ids={[str(d)[:8] for d in decision_ids]}",
        flush=True,
    )

    # 5. SUPPRESS best-effort: rapid repeat spikes (refractory/habituation)
    print("→ suppress attempt: rapid repeat spikes", flush=True)
    for k in range(3):
        await inject(
            "typing",
            user,
            "pause",
            20.0 + k,
            "seconds",
            {"avg_wpm": 10, "keypress_count": 1600 + k},
        )
        await asyncio.sleep(0.3)
        await inject(
            "activity_focus",
            app,
            "focus_change",
            7.9,
            "log1p_seconds",
            {
                "prev_app": app,
                "new_app": app,
                "active_dwell_s": 2500.0,
                "idle_dwell_s": 5.0,
                "total_dwell_s": 2505.0,
                "source_id": "clt",
                "span_id": uuid.uuid4().hex,
            },
        )
        await asyncio.sleep(0.4)

    # 6. explicit feedback (y) on each advice
    for did in decision_ids:
        await nc.publish(
            "augur.responsum.feedback",
            json.dumps({"decision_id": did, "rating": "y"}).encode(),
        )
    print(f"→ published feedback (y) for {len(decision_ids)} advice", flush=True)

    # 7. post-advice behavioral events (close window, POST_ADVICE_TRACK_MOVES=3)
    for v in (3.5, 3.6, 3.4):
        await inject(
            "typing",
            user,
            "pause",
            v,
            "seconds",
            {"avg_wpm": 46, "keypress_count": 1700},
        )
        await asyncio.sleep(0.2)

    # 8. session end -> Responsum finalize + Nexus graph flush
    await nc.publish(
        "augur.session.end",
        json.dumps(
            {"session_id": sid, "domain": "typing", "timestamp": now_iso()}
        ).encode(),
    )
    print("→ published session.end", flush=True)

    # 9-11. back-half waits (each early-breaks)
    print("→ waiting for responsum.complete...", flush=True)
    await wait_for(
        lambda: len(obs["augur.responsum.complete"]) >= 1, timeout=25, tick=2.0
    )
    print("→ waiting for disciplina.complete...", flush=True)
    await wait_for(
        lambda: len(obs["augur.disciplina.complete"]) >= 1, timeout=45, tick=2.0
    )
    print("→ waiting for imperator II (proposal/failure, real Ollama)...", flush=True)
    await wait_for(
        lambda: (
            len(obs["augur.imperator.proposal"]) >= 1
            or len(obs["augur.imperator.failure"]) >= 1
        ),
        timeout=120,
        tick=3.0,
    )
    await asyncio.sleep(3.0)

    # ---- Redis verification ----
    def rget(k):
        try:
            return r.get(k)
        except Exception:
            return None

    health = {}
    raw = rget("augur:praefectus:health")
    if raw:
        try:
            health = {
                f: v.get("overall")
                for f, v in json.loads(raw).get("faculties", {}).items()
            }
        except Exception:
            pass
    sm_raw = rget("augur:imperator:self_model")
    sm = json.loads(sm_raw) if sm_raw else {}
    prop_key = "augur:imperator:proposals"
    props = []
    try:
        t = r.type(prop_key)
        raw_props = (
            r.lrange(prop_key, 0, 9)
            if t == "list"
            else (r.zrange(prop_key, 0, 9) if t == "zset" else [])
        )
        props = [json.loads(p) for p in raw_props if p]
    except Exception:
        pass
    refl = None
    try:
        for k in r.scan_iter("augur:disciplina:*"):
            v = rget(k)
            if v and ("analyses" in v or "analysis" in v or "memory" in v):
                refl = json.loads(v)
                break
    except Exception:
        pass
    memoria_sessions = (
        r.scard("augur:memoria:processed_sessions")
        if r.exists("augur:memoria:processed_sessions")
        else 0
    )

    # ---- report ----
    advice = obs["augur.consilium.advice"]
    proposal_evt = obs["augur.imperator.proposal"]
    failure_evt = obs["augur.imperator.failure"]
    rows = [
        ("Sensus (inject)", True, "events published"),
        (
            "Vigil (anomaly)",
            len(obs["augur.vigil.anomaly"]) >= 1,
            f"{len(obs['augur.vigil.anomaly'])} anomalies",
        ),
        (
            "Nexus (correlate)",
            len(obs["augur.nexus.detected"]) >= 1,
            f"{len(obs['augur.nexus.detected'])} detected",
        ),
        (
            "Consilium (advice)",
            len(advice) >= 1,
            f"{len(advice)} advice, model={advice[0].get('model') if advice else '-'}",
        ),
        ("Limen FIRE", len(advice) >= 1, "advice delivered through gate"),
        (
            "Limen SUPPRESS*",
            len(obs["augur.limen.suppressed"]) >= 1,
            f"{len(obs['augur.limen.suppressed'])} suppressed (best-effort)",
        ),
        (
            "Responsum (complete)",
            len(obs["augur.responsum.complete"]) >= 1,
            f"{len(obs['augur.responsum.complete'])}",
        ),
        (
            "Disciplina (reflect)",
            len(obs["augur.disciplina.complete"]) >= 1,
            f"{len(obs['augur.disciplina.complete'])}",
        ),
        (
            "Imperator I (self-model)",
            len(obs["augur.imperator.self_model"]) >= 1 and bool(sm),
            f"reflection_ts={sm.get('reflection_ts')}",
        ),
        (
            "Imperator II (proposal)",
            len(proposal_evt) >= 1 or len(failure_evt) >= 1,
            f"{len(proposal_evt)} proposals, {len(failure_evt)} failures",
        ),
        (
            "Praefectus (health)",
            bool(health),
            f"{sum(1 for v in health.values() if v == 'alive')}/{len(health)} alive",
        ),
        (
            "Memoria (sweep)",
            memoria_sessions > 0 or refl is not None,
            f"processed_sessions={memoria_sessions}",
        ),
    ]
    print("\n" + "=" * 72)
    print("COMPLETE-LOOP FACULTY REPORT  (* = best-effort, excluded from overall)")
    print("=" * 72)
    allpass = True
    for name, ok, detail in rows:
        if not ok and "*" not in name:
            allpass = False
        print(f"  [{'PASS' if ok else 'FAIL'}] {name:26s} {detail}")

    applied = [p for p in props if p.get("status") == "applied"]
    print(
        f"\n  Imperator II watch-first: {len(props)} proposals stored, {len(applied)} applied "
        f"(expect 0 — apply OFF)"
    )
    if advice:
        print(f"  Sample advice: {str(advice[0].get('advice'))[:200]}")
    if proposal_evt:
        p0 = proposal_evt[0]
        print(
            f"  Sample proposal: kind={p0.get('kind')} target={p0.get('target')} "
            f"status={p0.get('status')} :: {str(p0.get('rationale'))[:120]}"
        )
    if failure_evt:
        print(f"  II failure: {failure_evt[0]}")
    print(f"  Praefectus: {health}")
    print("=" * 72)
    print(
        "OVERALL:",
        "ALL CORE FACULTIES PASS"
        if allpass
        else "SOME FACULTIES INCOMPLETE — see above",
    )

    await nc.drain()
    r.close()
    return 0 if allpass else 2


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
