#!/usr/bin/env python3
"""Live teaching-session driver against a running Augur deploy stack.

Exercises the Imperator III dialogue engine (handle_turn) through scripted
teaching arcs, verifying persisted state (PersistenceManager reads) and the
NATS dialogue events after every step:

  1. Fact -> taught: teach a semantic fact, confirm, verify it lands in
     the taught-facts store (the same store the dialogue context feeds
     into every subsequent LLM turn).
  2. Directive -> dropped: propose a context directive, then send a
     non-affirmative turn -- the pending must be actively DROPPED (with
     the "(Dropped the pending proposal.)" notice), never silently kept.
  3. Directive -> fire: propose again, confirm with "yes", verify the
     directive is persisted and augur.imperator.dialogue.applied fires.
  4. Fire -> undo: "undo that" reverses the applied directive; verify the
     directive is removed and the undo audit record + NATS event exist.

Run-unique session IDs => isolated conversation contexts. Runs from WSL
against the loopback-exposed deploy NATS (127.0.0.1:4222) and Redis
(127.0.0.1:6379). The default query_fn is real Ollama (qwen2.5:32b via
AUGUR_OLLAMA_URL); --stub-llm swaps in a canned-JSON query_fn through
handle_turn's existing dependency-injection seam (no engine changes) so
the full arc flow runs against live Redis+NATS without Ollama.

Usage: .venv/bin/python scripts/teaching_session.py [--run-id tag] [--stub-llm] [--dry-run]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
import uuid
from pathlib import Path

import httpx
import nats
import redis as redis_lib

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from imperator.dialogue.engine import handle_turn, query_dialogue_ollama  # noqa: E402
from tabula.config import AugurConfig  # noqa: E402
from tabula.persistence import PersistenceManager  # noqa: E402

NATS_URL = "nats://127.0.0.1:4222"
REDIS_URL = "redis://127.0.0.1:6379"

# The engine publishes exactly these two subjects (engine._publish call sites).
SUBJ_APPLIED = "augur.imperator.dialogue.applied"
SUBJ_TRIGGER = "augur.imperator.ii.trigger"
SUBJECTS = [SUBJ_APPLIED, SUBJ_TRIGGER]

DROP_NOTICE = "Dropped the pending proposal"


async def stub_query(prompt: str, system: str, client, cfg) -> str:
    """Canned-JSON query_fn matching engine.QueryFn -- deterministic intents
    keyed off the user text, so every arc runs without Ollama."""
    text = prompt.lower()
    if "remember" in text:
        obj = {
            "reply": "Noted - I'll remember that.",
            "intent": {
                "kind": "teach_semantic_fact",
                "target": "typing",
                "action": {
                    "domains": ["typing"],
                    "rule_key": "deep_work_morning",
                    "severity": "LOW",
                },
                "rationale": "deep work happens in the mornings",
            },
        }
    elif "stay quiet" in text:
        obj = {
            "reply": "Understood.",
            "intent": {
                "kind": "teach_context_directive",
                "target": "typing",
                "action": {
                    "predicate": {"domain": "typing", "context": "focus_block"},
                    "action": "suppress",
                    "scope": "all",
                },
                "rationale": "quiet during focus blocks",
            },
        }
    elif "undo" in text:
        obj = {"reply": "Let me reverse that.", "intent": {"kind": "undo"}}
    else:
        obj = {"reply": "Acknowledged."}
    return json.dumps(obj)


async def wait_for(pred, timeout: float, tick: float = 0.5) -> bool:
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        if pred():
            return True
        await asyncio.sleep(tick)
    return pred()


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default=uuid.uuid4().hex[:8])
    ap.add_argument(
        "--stub-llm",
        action="store_true",
        help="inject a canned-JSON query_fn through handle_turn's DI seam "
        "(full arc flow against live Redis+NATS, no Ollama needed)",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="parse args and check config only; do not connect or run turns",
    )
    args = ap.parse_args()

    cfg = AugurConfig.from_env()
    session_id = f"teaching-{args.run_id}"
    query_fn = stub_query if args.stub_llm else query_dialogue_ollama

    if args.dry_run:
        print(f"Dry-run mode: --run-id={args.run_id} session_id={session_id}")
        print(
            f"Config: ollama_url={cfg.ollama_url} redis_url={cfg.redis_url} "
            f"query_fn={'stub' if args.stub_llm else 'ollama'}"
        )
        return 0

    r = redis_lib.Redis.from_url(
        REDIS_URL, decode_responses=True, socket_connect_timeout=5
    )
    nc = await nats.connect(NATS_URL, connect_timeout=5)
    pm = PersistenceManager(r)
    http_client = httpx.AsyncClient(timeout=cfg.ollama_timeout)

    obs: dict[str, list] = {s: [] for s in SUBJECTS}

    async def on_msg(msg):
        try:
            data = json.loads(msg.data.decode())
        except Exception:
            data = {"_raw": True}
        obs.setdefault(msg.subject, []).append(data)
        tag = ""
        if msg.subject == SUBJ_APPLIED:
            tag = (
                f" kind={data.get('kind')} status={data.get('status')}"
                f" undo={data.get('undo', False)}"
            )
        elif msg.subject == SUBJ_TRIGGER:
            tag = f" reason={data.get('reason')}"
        print(f"[{time.strftime('%H:%M:%S')}] <- {msg.subject}{tag}", flush=True)

    for s in SUBJECTS:
        await nc.subscribe(s, cb=on_msg)
    await asyncio.sleep(0.5)
    print(
        f"session={session_id}; query_fn={'stub' if args.stub_llm else 'ollama'}; "
        f"subscribed {len(SUBJECTS)} subjects",
        flush=True,
    )

    pm.clear_dialogue_pending(session_id)
    applied_evts = obs[SUBJ_APPLIED]

    async def turn(text: str):
        t = await handle_turn(
            session_id,
            text,
            pm=pm,
            nc=nc,
            http_client=http_client,
            cfg=cfg,
            query_fn=query_fn,
        )
        err = f" error={t.error}" if t.error else ""
        print(f"  you: {text[:60]}\n  augur: {t.reply[:120]}{err}", flush=True)
        return t

    # ---- Arc 1: Fact -> taught (persisted into the dialogue context store)
    print("-> Arc 1: teach a semantic fact + confirm", flush=True)
    facts_before = len(pm.load_taught_facts())
    t1a = await turn(
        "Remember this fact about the typing domain: deep work happens in the mornings."
    )
    arc1_pending = t1a.pending is not None
    t1b = await turn("yes") if arc1_pending else None
    arc1_applied = bool(t1b and t1b.applied and t1b.applied.get("status") == "applied")
    await wait_for(lambda: len(applied_evts) >= 1, timeout=10)
    facts_after = len(pm.load_taught_facts())
    arc1 = arc1_applied and facts_after > facts_before and len(applied_evts) >= 1

    # ---- Arc 2: Directive -> actively dropped by the next non-affirmative turn
    print("-> Arc 2: propose a directive, then drop it", flush=True)
    t2a = await turn("Please stay quiet about typing during focus blocks.")
    arc2_pending = (
        t2a.pending is not None and pm.load_dialogue_pending(session_id) is not None
    )
    t2b = await turn("Actually, tell me what you noticed today instead.")
    arc2_dropped = DROP_NOTICE in t2b.reply
    arc2_cleared = pm.load_dialogue_pending(session_id) is None
    arc2 = arc2_pending and arc2_dropped and arc2_cleared

    # ---- Arc 3: Directive -> fire (confirm + persisted + NATS event)
    print("-> Arc 3: propose the directive again + confirm", flush=True)
    dirs_before = len(pm.load_dialogue_directives())
    t3a = await turn(
        "I do want that directive after all: stay quiet about typing "
        "during focus blocks."
    )
    t3b = await turn("yes") if t3a.pending is not None else None
    arc3_applied = bool(t3b and t3b.applied and t3b.applied.get("status") == "applied")
    await wait_for(lambda: len(applied_evts) >= 2, timeout=10)
    dirs_after = len(pm.load_dialogue_directives())
    arc3 = arc3_applied and dirs_after > dirs_before and len(applied_evts) >= 2

    # ---- Arc 4: Fire -> undo (reverse the applied directive)
    print("-> Arc 4: undo the applied directive", flush=True)
    t4 = await turn("Please undo that.")
    arc4_applied = bool(t4.applied and t4.applied.get("status") == "applied")
    await wait_for(lambda: any(e.get("undo") for e in applied_evts), timeout=10)
    dirs_final = len(pm.load_dialogue_directives())
    undo_evts = [e for e in applied_evts if e.get("undo")]
    arc4 = arc4_applied and dirs_final == dirs_before and len(undo_evts) >= 1

    # ---- Persisted-state readback
    audit = pm.load_dialogue_audit(limit=10)
    turns = pm.load_dialogue_log(limit=20, session_id=session_id)
    undo_audit = [a for a in audit if a.get("undo")]
    trigger_evts = obs[SUBJ_TRIGGER]

    # ---- Report
    rows = [
        (
            "Fact -> taught",
            arc1,
            f"taught_facts {facts_before}->{facts_after}, applied evt seen",
        ),
        (
            "Directive -> dropped",
            arc2,
            f"pending={arc2_pending} drop_notice={arc2_dropped} cleared={arc2_cleared}",
        ),
        (
            "Directive -> fire",
            arc3,
            f"directives {dirs_before}->{dirs_after}, applied evts={len(applied_evts)}",
        ),
        (
            "Fire -> undo",
            arc4,
            f"directives back to {dirs_final}, undo evts={len(undo_evts)}",
        ),
        (
            "Audit trail",
            len(audit) >= 3 and len(undo_audit) >= 1,
            f"{len(audit)} records, {len(undo_audit)} undo",
        ),
        (
            "Dialogue log",
            len(turns) >= 7,
            f"{len(turns)} turns persisted for this session",
        ),
        (
            "II triggers (NATS)",
            len(trigger_evts) >= 3,
            f"{len(trigger_evts)} augur.imperator.ii.trigger events",
        ),
    ]
    print("\n" + "=" * 72)
    print("TEACHING-SESSION DIALOGUE REPORT")
    print("=" * 72)
    allpass = True
    for name, ok, detail in rows:
        if not ok:
            allpass = False
        print(f"  [{'PASS' if ok else 'FAIL'}] {name:26s} {detail}")

    if audit:
        print(f"\n  Audit history (newest first, {len(audit)} entries):")
        for rec in audit[:4]:
            print(
                f"    - kind={rec.get('kind')} status={rec.get('status')} "
                f"undo={rec.get('undo', False)}"
            )
    if turns:
        print(f"\n  Dialogue turns (newest first, {len(turns)} entries):")
        for t in turns[:4]:
            print(
                f"    - you: {(t.get('user_text') or '')[:44]!r}"
                f" -> augur: {(t.get('reply') or '')[:44]!r}"
            )
    print("=" * 72)
    print(
        "OVERALL:", "ALL ARCS PASS" if allpass else "SOME ARCS INCOMPLETE - see above"
    )

    await nc.drain()
    await http_client.aclose()
    r.close()
    return 0 if allpass else 2


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
