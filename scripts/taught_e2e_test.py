#!/usr/bin/env python3
"""End-to-end taught-knowledge effects against a running deploy stack.

Where teaching_session.py scores the dialogue arcs themselves, this driver
scores what taught knowledge DOES downstream — the fact/directive stores are
read by real containerized faculties (Consilium prompt injection, the Limen
Stage-0.5 taught-directive pre-check) while events flow:

  A. Taught fact lands where Consilium's advice path reads it
     (load_taught_facts_for_domains + the exact format_taught_facts block),
     and a live typing anomaly still produces advice. Also reports whether
     the taught rationale text survives into the injected block or only the
     rule_key slug does.
  B. Server-authoritative predicate: the stubbed LLM intent deliberately
     claims a WRONG app in predicate.match/target; the stored directive must
     carry the live focused app instead (router.route override).
  C. Active suppress-directive: a HIGH single-channel typing spike while the
     focused app matches must be suppressed with reason taught_directive:<id>
     (and land in the silence records — invariant A), with no advice emitted.
  D. Undo restores: after undo, a fresh spike fires advice again.
  E. Scope isolation: a directive scoped to another domain must NOT suppress
     typing; advice fires.
  F. Staleness refusal: with only stale (>focused_app_max_age_s) activity
     events, teaching a context directive is refused and nothing is stored.
  G. Downgrade directive: action=downgrade delivers a Tier-1 note on the
     advice subject instead of full advice (and instead of silence).

Advice/suppression events are matched by entity (the advice payload's
"player" compat alias carries the primary entity; suppressed events carry
"entity") — the session_id on those payloads comes from the active-session
registry, which headless drivers don't populate.

Intent classification is stubbed per-step through handle_turn's query_fn DI
seam (deterministic); everything downstream — router, apply, persistence,
the containerized gate + Consilium LLM — is real. Run AFTER any other
dialogue/teaching driver has finished to avoid directive cross-talk.

Usage: .venv/bin/python scripts/taught_e2e_test.py [--llm-wait 150]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
import nats
import redis as redis_lib

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from consilium.advisor import format_taught_facts  # noqa: E402
from imperator.dialogue.engine import handle_turn  # noqa: E402
from tabula.config import AugurConfig  # noqa: E402
from tabula.contracts import PerceptionEvent  # noqa: E402
from tabula.persistence import PersistenceManager  # noqa: E402

NATS_URL = "nats://127.0.0.1:4222"
REDIS_URL = "redis://127.0.0.1:6379"
_POOL = [3.2, 3.8, 3.5, 4.0, 3.1, 3.9, 3.3, 3.7, 3.4, 3.6]
_IPOOL = [35.2, 41.8, 38.5, 44.0, 36.1, 42.3]


def iso(offset_s: float = 0.0) -> str:
    return (datetime.now(timezone.utc) + timedelta(seconds=offset_s)).isoformat()


def make_stub(payloads: list[dict]):
    """query_fn returning canned JSON objects in sequence (per turn)."""
    queue = list(payloads)

    async def stub(prompt: str, system: str, client, cfg) -> str:
        obj = queue.pop(0) if queue else {"reply": "Acknowledged."}
        return json.dumps(obj)

    return stub


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--llm-wait",
        type=float,
        default=150.0,
        help="max seconds to wait for containerized advice",
    )
    args = ap.parse_args()

    rid = uuid.uuid4().hex[:8]
    app = f"e2eapp_{rid}"
    cfg = AugurConfig.from_env()
    r = redis_lib.Redis.from_url(
        REDIS_URL, decode_responses=True, socket_connect_timeout=5
    )
    pm = PersistenceManager(r)
    nc = await nats.connect(NATS_URL, connect_timeout=5)
    http_client = httpx.AsyncClient(timeout=cfg.ollama_timeout)

    advice: list[dict] = []
    suppressed: list[dict] = []

    async def on_msg(msg):
        try:
            data = json.loads(msg.data.decode())
        except Exception:
            return
        if msg.subject == "augur.consilium.advice":
            advice.append(data)
            print(
                f"  <- advice player={data.get('player')} "
                f"tier={data.get('tier')} sev={data.get('severity')}",
                flush=True,
            )
        elif msg.subject == "augur.limen.suppressed":
            suppressed.append(data)
            print(
                f"  <- suppressed entity={data.get('entity')} "
                f"arm={data.get('arm')} reason={data.get('reason')}",
                flush=True,
            )

    await nc.subscribe("augur.consilium.advice", cb=on_msg)
    await nc.subscribe("augur.limen.suppressed", cb=on_msg)
    await asyncio.sleep(0.5)

    def adv_for(entity: str) -> list[dict]:
        return [e for e in advice if e.get("player") == entity]

    def sup_for(entity: str) -> list[dict]:
        return [e for e in suppressed if e.get("entity") == entity]

    async def emit(
        domain: str,
        entity: str,
        value: float,
        sid: str,
        context: dict | None = None,
        ts: str | None = None,
        event_type: str = "pause",
        unit: str = "seconds",
    ):
        ev = PerceptionEvent(
            domain=domain,
            stream_id=domain,
            entity=entity,
            event_type=event_type,
            value=float(value),
            unit=unit,
            context=context or {},
            timestamp=ts or iso(),
            session_id=sid,
        )
        await nc.publish(f"augur.sensus.{domain}", ev.to_bytes())

    async def typing_spike(entity: str, sid: str, spike: float = 19.0):
        for i in range(20):
            await emit("typing", entity, _POOL[i % 10], sid, context={"avg_wpm": 40})
            await asyncio.sleep(0.03)
        await asyncio.sleep(1.0)
        await emit("typing", entity, spike, sid, context={"avg_wpm": 40})

    async def publish_fresh_app(name: str, sid: str):
        for i, v in enumerate(_IPOOL):
            await emit(
                "activity_intensity",
                name,
                v,
                sid,
                context={
                    "focused_app": name,
                    "title": "e2e",
                    "source_id": "e2e",
                    "span_id": f"s{i}",
                    "keystroke_count": int(v),
                    "mouse_event_count": 3,
                    "idle_seconds": 0.0,
                    "window_duration_s": 30.0,
                },
                event_type="intensity_sample",
                unit="ipm",
            )
            await asyncio.sleep(0.05)
        # Quiesce past the correlator window (30/60s) so these activity
        # events can never correlate with the scenario's typing spike —
        # high+correlated is exempt and would bypass the gate entirely.
        print("  (quiescing 65s past the correlation window...)", flush=True)
        await asyncio.sleep(65.0)

    async def wait_for(pred, timeout: float, tick: float = 1.0) -> bool:
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout:
            if pred():
                return True
            await asyncio.sleep(tick)
        return pred()

    async def turn(session: str, text: str, stub):
        t = await handle_turn(
            session, text, pm=pm, nc=nc, http_client=http_client, cfg=cfg, query_fn=stub
        )
        err = f" error={t.error}" if t.error else ""
        print(f"  you: {text[:58]}\n  augur: {t.reply[:100]}{err}", flush=True)
        return t

    def directive_intent(action: str, scope, match: str, target: str) -> dict:
        return {
            "reply": "Understood.",
            "intent": {
                "kind": "teach_context_directive",
                "target": target,
                "action": {
                    "predicate": {"context": "focused_app", "match": match},
                    "action": action,
                    "scope": scope,
                },
                "rationale": "e2e taught directive",
            },
        }

    undo_stub_payloads = [
        {"reply": "Let me reverse that.", "intent": {"kind": "undo"}},
        {"reply": "Confirmed."},
    ]

    async def undo(session: str) -> bool:
        stub = make_stub(list(undo_stub_payloads))
        t1 = await turn(session, "Please undo that.", stub)
        if t1.pending is None:
            return False
        t2 = await turn(session, "yes", stub)
        return bool(t2.applied and t2.applied.get("status") == "applied")

    def directives() -> list[dict]:
        return [d for d in pm.load_dialogue_directives() if isinstance(d, dict)]

    rows: list[tuple[str, bool, str]] = []

    def row(name, ok, detail):
        rows.append((name, ok, detail))
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}", flush=True)

    dirs_baseline = len(directives())

    # ---------- F. Staleness refusal (run first: forces stale-latest state)
    print("\n=== F: staleness refusal (stale focus events only) ===", flush=True)
    sid_f = f"e2e-f-{rid}"
    stale_ts = iso(-(getattr(cfg, "focused_app_max_age_s", 300.0) + 100.0))
    await emit(
        "activity_focus",
        f"stale_{rid}",
        1.0,
        sid_f,
        context={"new_app": f"stale_{rid}", "prev_app": "x", "active_dwell_s": 5.0},
        ts=stale_ts,
        event_type="app_focus",
        unit="event",
    )
    await emit(
        "activity_intensity",
        f"stale_{rid}",
        30.0,
        sid_f,
        context={"focused_app": f"stale_{rid}"},
        ts=stale_ts,
        event_type="intensity_sample",
        unit="ipm",
    )
    await asyncio.sleep(2.0)
    focused_now = pm.load_focused_app(
        now=time.time(), max_age_s=cfg.focused_app_max_age_s
    )
    tf = await turn(
        f"dlg-f-{rid}",
        "From now on stay quiet while I'm in that app.",
        make_stub([directive_intent("suppress", "all", "whatever", "whatever")]),
    )
    f_refused = tf.pending is None
    f_unchanged = len(directives()) == dirs_baseline
    row(
        "F stale focus -> teach refused",
        focused_now is None and f_refused and f_unchanged,
        f"focused={focused_now} pending={tf.pending is not None} "
        f"directives_unchanged={f_unchanged} reply={tf.reply[:60]!r}",
    )

    # ---------- A. Taught fact -> advice plumbing
    print("\n=== A: taught fact -> Consilium injection block ===", flush=True)
    sid_a = f"e2e-a-{rid}"
    dlg_a = f"dlg-a-{rid}"
    ent_a = f"user_a_{rid}"
    fact_intent = {
        "reply": "Noted.",
        "intent": {
            "kind": "teach_semantic_fact",
            "target": "typing",
            "action": {
                "domains": ["typing"],
                "rule_key": f"morning_deep_work_{rid}",
                "severity": "LOW",
            },
            "rationale": "deep work happens in the mornings; long pauses "
            "then are usually thought, not distraction",
        },
    }
    stub_a = make_stub([fact_intent, {"reply": "Confirmed."}])
    ta1 = await turn(
        dlg_a,
        "Remember: my deep work happens in the mornings, "
        "long pauses then are thought not distraction.",
        stub_a,
    )
    ta2 = await turn(dlg_a, "yes", stub_a) if ta1.pending else None
    taught = bool(ta2 and ta2.applied and ta2.applied.get("status") == "applied")

    mine = [
        f
        for f in pm.load_taught_facts_for_domains(["typing"])
        if f.get("pattern", {}).get("rule_key") == f"morning_deep_work_{rid}"
    ]
    block = format_taught_facts(pm.load_taught_facts_for_domains(["typing"]))
    rationale_in_block = "deep work happens in the mornings" in block
    row(
        "A fact taught + readable via advisor load path",
        taught and len(mine) == 1 and f"morning_deep_work_{rid}" in block,
        f"applied={taught} in_store={len(mine)} block_has_rule_key="
        f"{f'morning_deep_work_{rid}' in block}",
    )
    row(
        "A(info) rationale text survives into advice block",
        rationale_in_block,
        f"record_rationale={mine[0].get('rationale') if mine else None!r} "
        f"block={block[:120]!r}",
    )

    await typing_spike(ent_a, sid_a)
    a_advice = await wait_for(lambda: len(adv_for(ent_a)) >= 1, args.llm_wait)
    row(
        "A typing anomaly still produces advice (fact active)",
        a_advice,
        f"advice_for_entity={len(adv_for(ent_a))}",
    )

    # ---------- B. Server-authoritative predicate
    print("\n=== B: server-authoritative directive predicate ===", flush=True)
    sid_b = f"e2e-b-{rid}"
    dlg_b = f"dlg-b-{rid}"
    await publish_fresh_app(app, sid_b)
    live = pm.load_focused_app(now=time.time(), max_age_s=cfg.focused_app_max_age_s)
    stub_b = make_stub(
        [
            directive_intent("suppress", "all", "WRONG_APP", "WRONG_APP"),
            {"reply": "Confirmed."},
        ]
    )
    tb1 = await turn(dlg_b, "Stay quiet while I'm in this app.", stub_b)
    tb2 = await turn(dlg_b, "yes", stub_b) if tb1.pending else None
    applied_b = bool(tb2 and tb2.applied and tb2.applied.get("status") == "applied")
    stored = [d for d in directives() if d.get("predicate", {}).get("match") == app]
    wrong = [
        d for d in directives() if d.get("predicate", {}).get("match") == "WRONG_APP"
    ]
    row(
        "B predicate overridden to live focused app",
        live == app and applied_b and len(stored) == 1 and not wrong,
        f"live={live} applied={applied_b} stored_match_app={len(stored)} "
        f"stored_wrong={len(wrong)}",
    )

    # ---------- C. Suppression while focused app matches
    print("\n=== C: HIGH spike suppressed by taught directive ===", flush=True)
    sid_c = f"e2e-c-{rid}"
    ent_c = f"user_c_{rid}"
    await publish_fresh_app(app, sid_c)  # re-stamp freshness
    await typing_spike(ent_c, sid_c)
    c_sup = await wait_for(
        lambda: any(
            str(e.get("reason", "")).startswith("taught_directive:")
            for e in sup_for(ent_c)
        ),
        45,
    )
    await asyncio.sleep(5.0)
    c_no_advice = not adv_for(ent_c)
    silence_recs = pm.load_silence_records(limit=20)
    c_silence = any(
        str(rec.get("reason", "")).startswith("taught_directive:")
        for rec in silence_recs
    )
    row(
        "C directive suppresses matching-context anomaly",
        c_sup and c_no_advice,
        f"suppressed={len(sup_for(ent_c))} no_advice={c_no_advice} "
        f"reasons={[e.get('reason') for e in sup_for(ent_c)][-3:]}",
    )
    row(
        "C invariant A: suppression in silence records",
        c_silence,
        f"{len(silence_recs)} records, taught_directive present={c_silence}",
    )

    # ---------- D. Undo -> fires again
    print("\n=== D: undo directive -> advice flows again ===", flush=True)
    sid_d = f"e2e-d-{rid}"
    ent_d = f"user_d_{rid}"
    undone = await undo(dlg_b)
    d_removed = not any(
        d.get("predicate", {}).get("match") == app for d in directives()
    )
    await publish_fresh_app(app, sid_d)
    await typing_spike(ent_d, sid_d)
    d_advice = await wait_for(lambda: len(adv_for(ent_d)) >= 1, args.llm_wait)
    row(
        "D undo removes directive; spike fires advice again",
        undone and d_removed and d_advice,
        f"undone={undone} removed={d_removed} advice={d_advice}",
    )

    # ---------- E. Scope isolation (directive scoped to another domain)
    print("\n=== E: chess-scoped directive must not gate typing ===", flush=True)
    sid_e = f"e2e-e-{rid}"
    dlg_e = f"dlg-e-{rid}"
    ent_e = f"user_e_{rid}"
    await publish_fresh_app(app, sid_e)
    stub_e = make_stub(
        [directive_intent("suppress", ["chess"], app, app), {"reply": "Confirmed."}]
    )
    te1 = await turn(dlg_e, "While I'm in this app, stay quiet about chess.", stub_e)
    te2 = await turn(dlg_e, "yes", stub_e) if te1.pending else None
    applied_e = bool(te2 and te2.applied and te2.applied.get("status") == "applied")
    await typing_spike(ent_e, sid_e)
    e_advice = await wait_for(lambda: len(adv_for(ent_e)) >= 1, args.llm_wait)
    e_not_supp = not any(
        str(ev.get("reason", "")).startswith("taught_directive:")
        for ev in sup_for(ent_e)
    )
    row(
        "E out-of-scope directive does not suppress typing",
        applied_e and e_advice and e_not_supp,
        f"applied={applied_e} advice={e_advice} not_suppressed={e_not_supp}",
    )
    await undo(dlg_e)

    # ---------- G. Downgrade directive -> Tier-1 note
    print("\n=== G: downgrade directive -> Tier-1 note ===", flush=True)
    sid_g = f"e2e-g-{rid}"
    dlg_g = f"dlg-g-{rid}"
    ent_g = f"user_g_{rid}"
    await publish_fresh_app(app, sid_g)
    stub_g = make_stub(
        [directive_intent("downgrade", "all", app, app), {"reply": "Confirmed."}]
    )
    tg1 = await turn(dlg_g, "Keep it to a short note while I'm in this app.", stub_g)
    tg2 = await turn(dlg_g, "yes", stub_g) if tg1.pending else None
    applied_g = bool(tg2 and tg2.applied and tg2.applied.get("status") == "applied")
    await typing_spike(ent_g, sid_g)
    g_note = await wait_for(
        lambda: any(e.get("tier") == 1 for e in adv_for(ent_g)), args.llm_wait
    )
    g_evt = next(iter(adv_for(ent_g)), {})
    row(
        "G downgrade directive delivers Tier-1 note",
        applied_g and g_note,
        f"applied={applied_g} tier={g_evt.get('tier')} keys={sorted(g_evt)[:8]}",
    )
    await undo(dlg_g)

    # ---------- Report
    print("\n" + "=" * 72)
    print("TAUGHT-KNOWLEDGE E2E REPORT")
    print("=" * 72)
    passed = sum(1 for _, ok, _ in rows if ok)
    for name, ok, detail in rows:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}\n          {detail}")
    print("=" * 72)
    print(f"OVERALL: {passed}/{len(rows)} PASS")

    await nc.drain()
    await http_client.aclose()
    r.close()
    return 0 if passed == len(rows) else 2


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
