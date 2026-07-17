#!/usr/bin/env python3
"""Drive a running Augur pipeline with synthetic perception events and observe
the full chain end-to-end (anomaly -> correlation -> advice -> gate -> imperator).

Used for live shakeout against the containerized deploy stack. Injects to NATS
``augur.sensus.<domain>``, subscribes to every downstream subject, prints events
as they arrive, and dumps key Redis state + a summary at the end.

Run-unique entity names guarantee a fresh baseline per run (no flush needed).

Usage:
  .venv/bin/python scripts/inject_and_observe.py [--llm-wait 120] [--baseline 20] [--run-id tag]
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

OBSERVE_SUBJECTS = [
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


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--llm-wait", type=float, default=120.0, help="max seconds to wait for advice"
    )
    ap.add_argument(
        "--baseline", type=int, default=20, help="baseline observations per domain"
    )
    ap.add_argument("--run-id", default=uuid.uuid4().hex[:8])
    args = ap.parse_args()

    session_id = f"shakeout-{args.run_id}"
    user = f"user_{args.run_id}"  # run-unique -> fresh baseline, no prior state
    app = f"code_{args.run_id}"

    r = redis_lib.Redis.from_url(
        REDIS_URL, decode_responses=True, socket_connect_timeout=5
    )
    # Runs against the live stack: record this session as synthetic so its
    # injected perception is explicitly non-learnable, not merely unprovenanced.
    PersistenceManager(r).save_session_meta(
        session_id, origin="synthetic", created_by="inject_and_observe"
    )
    nc = await nats.connect(NATS_URL, connect_timeout=5)
    observed: dict[str, list] = {s: [] for s in OBSERVE_SUBJECTS}

    async def on_msg(msg):
        try:
            data = json.loads(msg.data.decode())
        except Exception:
            data = {"_raw": msg.data[:200].decode("utf-8", "replace")}
        observed.setdefault(msg.subject, []).append(data)
        sev = data.get("severity", "")
        dom = data.get("domain", data.get("involved_domains", ""))
        extra = ""
        if msg.subject == "augur.consilium.advice":
            extra = f" :: {str(data.get('advice', ''))[:160]!r}"
        elif msg.subject == "augur.limen.suppressed":
            extra = f" :: reason={data.get('reason') or data.get('deciding_arm')}"
        print(f"[{_now()}] <- {msg.subject}  {dom} {sev}{extra}", flush=True)

    for s in OBSERVE_SUBJECTS:
        await nc.subscribe(s, cb=on_msg)
    print(
        f"subscribed to {len(OBSERVE_SUBJECTS)} subjects; session={session_id}",
        flush=True,
    )
    await asyncio.sleep(1.0)

    async def inject(domain, entity, event_type, value, unit, context):
        ev = PerceptionEvent(
            domain=domain,
            stream_id=f"{domain}_stream",
            entity=entity,
            event_type=event_type,
            value=float(value),
            unit=unit,
            context=context,
            timestamp=_now(),
            session_id=session_id,
        )
        await nc.publish(f"augur.sensus.{domain}", ev.to_bytes())

    # ---- varied baselines (std must exceed 0.01 or deviation is forced to 0) ----
    typing_pool = [3.2, 3.8, 3.5, 4.0, 3.1, 3.9, 3.3, 3.7, 3.4, 3.6]
    focus_pool = [4.5, 5.2, 4.8, 5.5, 4.3, 5.1, 4.7, 5.3, 4.9, 5.0]
    print(f"injecting {args.baseline} varied baseline obs/domain ...", flush=True)
    for i in range(args.baseline):
        await inject(
            "typing",
            user,
            "pause",
            typing_pool[i % len(typing_pool)],
            "seconds",
            {"avg_wpm": 48, "keypress_count": 1000 + i},
        )
        await inject(
            "activity_focus",
            app,
            "focus_change",
            focus_pool[i % len(focus_pool)],
            "log1p_seconds",
            {
                "prev_app": app,
                "new_app": "browser",
                "active_dwell_s": 120.0,
                "idle_dwell_s": 4.0,
                "total_dwell_s": 124.0,
                "source_id": "shakeout",
                "span_id": uuid.uuid4().hex,
            },
        )
        await asyncio.sleep(0.05)
    await asyncio.sleep(2.0)

    # ---- single-domain anomaly: a long typing pause ----
    print("spike #1: single-domain typing pause (18s)", flush=True)
    await inject(
        "typing",
        user,
        "pause",
        18.0,
        "seconds",
        {"avg_wpm": 12, "keypress_count": 1300},
    )
    await asyncio.sleep(4.0)

    # ---- correlated cross-domain spike inside the 30s window ----
    print("spike #2: correlated typing-pause + stuck-focus within window", flush=True)
    await inject(
        "typing", user, "pause", 22.0, "seconds", {"avg_wpm": 9, "keypress_count": 1500}
    )
    await asyncio.sleep(0.5)
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
            "source_id": "shakeout",
            "span_id": uuid.uuid4().hex,
        },
    )

    print(
        f"waiting up to {args.llm_wait}s for advice (real Ollama, cold ~68s)...",
        flush=True,
    )
    waited = 0.0
    while waited < args.llm_wait:
        await asyncio.sleep(3.0)
        waited += 3.0
        if observed["augur.consilium.advice"] or observed["augur.limen.suppressed"]:
            await asyncio.sleep(4.0)  # grace for trailing events
            break

    print("\n===== OBSERVED EVENT COUNTS =====", flush=True)
    for s in OBSERVE_SUBJECTS:
        print(f"  {s}: {len(observed[s])}")
    for a in observed["augur.consilium.advice"]:
        print("\n--- ADVICE ---")
        print(
            f"  domain={a.get('domain')} severity={a.get('severity')} "
            f"tier={a.get('tier')} model={a.get('model')}"
        )
        print(f"  advice: {a.get('advice')}")
    for s in observed["augur.limen.suppressed"]:
        print(
            f"\n--- SUPPRESSED --- reason={s.get('reason')} arm={s.get('deciding_arm')}"
        )

    print("\n===== REDIS STATE =====", flush=True)
    for key in ("augur:imperator:auspices", "augur:imperator:self_model"):
        print(f"  {key}: {'present' if r.get(key) else 'ABSENT'}")
    print(
        f"  augur:imperator:proposals: {len(r.lrange('augur:imperator:proposals', 0, 9))} recent"
    )

    await nc.drain()
    r.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
