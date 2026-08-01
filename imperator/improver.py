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
from tabula.provenance import ProvenanceMode, get_provenance_mode  # noqa: E402
from imperator import proposals as P, reasoner, apply as A  # noqa: E402

log = logging.getLogger("imperator.improver")
_REFLECTION_SUBJECT = "augur.disciplina.complete"
_CONSUMED = frozenset({_REFLECTION_SUBJECT, "augur.imperator.ii.trigger"})


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
    # Within-cycle de-dup: the reasoner can emit two items with the same
    # (kind, target) — same dedupe_key — in one batch. Only the first (highest
    # ranked, since we iterate P.rank order) is acted on; later duplicates are
    # dropped silently rather than double-logged / double-emitted. The applied-TTL
    # marker only blocks ACROSS cycles, so without this guard the duplicates slip
    # through within the same cycle.
    seen: set[str] = set()
    for p in P.rank(cands):
        if p["dedupe_key"] in seen:
            continue
        seen.add(p["dedupe_key"])
        P.gate(p, cfg=cfg, recent_self_tuning=recent, applied_keys=applied)
        if p["status"] != "skipped":
            A.apply_proposal(pm, p, cfg=cfg, session_id=session_id)
        pm.save_proposal(p, ctx=pm.resolve_learn_context(session_id))
        if p["status"] == "applied":
            applied.add(p["dedupe_key"])
        await _emit(publish, "augur.imperator.proposal", json.dumps(p).encode())


async def _await_fresh(pm, epoch, timeout_s, tick_s=0.5):
    """Wait until the self-model has FOLDED a reflection at least as new as the
    triggering one. Gates on the folded reflection's timestamp (a content check),
    not wall-clock generated_at — which could advance on an Imperator-I tick that
    did NOT fold the triggering reflection (feedback lag / session aged out)."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        sm = pm.load_self_model()
        if sm and sm.get("reflection_ts", 0.0) >= epoch:
            return True
        await asyncio.sleep(tick_s)
    return False


async def _cycle(pm, config, http, *, payload, now, publish) -> None:
    """Freshness-gate + reason + log/apply for one trigger.

    Runs OFF the NATS dispatch path (spawned as a task) so the freshness wait
    (~15s) never blocks the subscription handler — head-of-line blocking there
    would drop events that arrive during the wait. The caller holds the cycle
    lock for this coroutine's whole lifetime, so at most one cycle is in flight.
    Any failure is routed to augur.imperator.failure (fail-open)."""
    epoch = parse_reflection_epoch(payload)
    if epoch and not await _await_fresh(
        pm, epoch, config.imperator_ii_freshness_timeout_s
    ):
        log.info("self-model did not fold reflection in time; skipping")
        return

    async def gen(sm, *, client, config, now):
        return await reasoner.generate_proposals(
            sm, client=client, config=config, now=now
        )

    await _run_and_route(
        lambda: run_cycle(
            pm,
            config,
            now=now,
            session_id=payload.get("session_id"),
            generate_fn=gen,
            client=http,
            publish=publish,
        ),
        publish=publish,
        now=now,
    )


def make_on_msg(pm, config, http, *, lock, last_run, spawn, publish):
    """Build the subscription callback.

    The callback does only cheap, synchronous work — subject filter, rate-limit,
    in-flight check — then hands the cycle off to *spawn* and returns promptly so
    nats-py keeps draining the subscription (the ~15s freshness wait runs in the
    spawned task, never on the dispatch path). At most one cycle is in flight:
    a trigger that arrives while one is running is dropped.

    The in-flight decision is made SYNCHRONOUSLY via lock.locked() inside the
    handler. Because nats-py dispatches a subscription's callbacks sequentially,
    the handler runs to completion (and the spawned task acquires *lock*) before
    the next callback starts — so a second trigger reliably observes the lock as
    held and is dropped rather than queued behind the running cycle.

    *lock* serializes cycles AND backs the in-flight check; *last_run* is a
    1-element list holding the wall-clock start of the last accepted cycle
    (rate-limit anchor, single read); *spawn* schedules the cycle coroutine as an
    independent task (asyncio.create_task in production)."""

    async def on_msg(msg):
        if not consumed(msg.subject):
            return
        now = time.time()
        if now - last_run[0] < config.imperator_ii_min_interval_s:
            log.info(
                "imperator II trigger dropped: rate-limited "
                "(%.1fs since last accepted cycle, min %.1fs)",
                now - last_run[0],
                config.imperator_ii_min_interval_s,
            )
            return
        if lock.locked():
            # A cycle is already in flight; drop this trigger rather than queue
            # it (the next disciplina.complete will re-trigger if still warranted).
            log.info("imperator II trigger dropped: cycle already in flight")
            return
        try:
            payload = json.loads(msg.data.decode()) if msg.data else {}
        except (json.JSONDecodeError, UnicodeDecodeError):
            log.warning("imperator II trigger payload undecodable; skipping")
            return
        # §4.3e second layer: filtering the read-model keeps a non-learnable
        # reflection out of the self-model, but the trigger would still spend a
        # cycle reasoning about it. Only the reflection subject is gated — the
        # dialogue trigger is a direct user action carrying no session.
        if (
            msg.subject == _REFLECTION_SUBJECT
            and get_provenance_mode() is ProvenanceMode.ENFORCE
            and not pm.is_learnable_session(payload.get("session_id"))
        ):
            log.info(
                "imperator II trigger dropped: session %s is not learnable",
                payload.get("session_id"),
            )
            return
        # Take the lock synchronously HERE (not inside the task): this closes the
        # window where two back-to-back callbacks both pass lock.locked()==False
        # before either task gets scheduled, which would queue the second cycle
        # instead of dropping it. Acquisition is immediate because we only reach
        # this line when the lock is free.
        await lock.acquire()
        # Anchor the rate-limit at acceptance (single clock read), AFTER the lock
        # is held — so bursts during the freshness wait are still throttled.
        last_run[0] = now

        async def _guarded():
            try:
                await _cycle(
                    pm, config, http, payload=payload, now=now, publish=publish
                )
            finally:
                lock.release()

        spawn(_guarded())

    return on_msg


async def _run_and_route(coro_fn, *, publish, now) -> None:
    """Run a cycle coroutine; route ANY failure to augur.imperator.failure.

    Reasoner failures carry their classified reason; apply/persistence/other
    failures carry reason='cycle_error'. Both go to the distinct Imperator
    channel so an operator watching (apply is watch-first) sees an apply break,
    and Praefectus never miscounts it as a Consilium terminal. The runner never
    crashes on a cycle failure (fail-open)."""
    try:
        await coro_fn()
    except reasoner.ReasonerError as exc:
        await _emit(
            publish,
            "augur.imperator.failure",
            json.dumps({"reason": exc.reason, "ts": now}).encode(),
        )
    except Exception as exc:
        log.warning("imperator II cycle failed; continuing", exc_info=True)
        try:
            await _emit(
                publish,
                "augur.imperator.failure",
                json.dumps(
                    {"reason": "cycle_error", "detail": str(exc), "ts": now}
                ).encode(),
            )
        except Exception:
            log.debug("imperator II failure publish failed", exc_info=True)


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
    tasks: set[asyncio.Task] = set()

    def spawn(coro):
        # Run the cycle off the dispatch path so the handler returns promptly and
        # nats-py keeps draining the subscription. Keep a strong ref so the task
        # isn't GC'd mid-flight, and drop it from the set when it finishes.
        task = asyncio.create_task(coro)
        tasks.add(task)
        task.add_done_callback(tasks.discard)

    on_msg = make_on_msg(
        pm, config, http, lock=lock, last_run=last_run, spawn=spawn, publish=nc.publish
    )

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
        for task in list(tasks):
            task.cancel()
        for task in list(tasks):
            try:
                await task
            except (asyncio.CancelledError, Exception):
                log.debug("imperator II cycle task teardown error", exc_info=True)
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
