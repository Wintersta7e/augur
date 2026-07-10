import asyncio
import json
import logging

import fakeredis
from tabula.persistence import PersistenceManager
from imperator import improver


def test_parse_reflection_epoch():
    assert (
        improver.parse_reflection_epoch({"timestamp": "2026-06-14T00:00:00+00:00"}) > 0
    )
    assert improver.parse_reflection_epoch({}) == 0.0


def test_run_cycle_logs_and_respects_watch_first():
    pm = PersistenceManager(fakeredis.FakeStrictRedis(decode_responses=False))
    pm.save_escalation_matrix({"version": "v1", "rules": {"LOW+LOW": "LOW"}})
    pm.save_self_model(
        {
            "schema_version": 1,
            "generated_at": 200.0,
            "session_id": "s1",
            "blind_spots": {
                "value": [
                    {
                        "kind": "low_confidence_rule",
                        "detail": "x",
                        "evidence": "LOW+LOW",
                    }
                ],
                "fresh": True,
            },
            "recent_self_tuning": {"value": {}, "fresh": True},
        }
    )

    async def fake_gen(sm, *, client, config, now):
        from imperator import proposals as P

        return [
            P.make_proposal(
                kind="escalation_rule",
                target="LOW+LOW",
                action={"target": "MEDIUM"},
                rationale="r",
                now=now,
            )
        ]

    published = []
    asyncio.run(
        improver.run_cycle(
            pm,
            _Cfg(),
            now=300.0,
            session_id="s1",
            generate_fn=fake_gen,
            client=None,
            publish=lambda subj, data: published.append(subj),
        )
    )
    props = pm.load_proposals()
    assert len(props) == 1 and props[0]["status"] == "logged"
    assert pm.load_escalation_matrix()["rules"]["LOW+LOW"] == "LOW"
    assert "augur.imperator.proposal" in published


def test_run_and_route_publishes_reasoner_reason():
    from imperator import reasoner

    published = []

    async def boom():
        raise reasoner.ReasonerError("ollama_timeout", "slow")

    asyncio.run(
        improver._run_and_route(
            boom, publish=lambda s, d: published.append((s, json.loads(d))), now=5.0
        )
    )
    assert len(published) == 1
    subj, body = published[0]
    assert subj == "augur.imperator.failure" and body["reason"] == "ollama_timeout"


def test_run_and_route_publishes_cycle_error_for_other_failures():
    # An apply/persistence failure (not a reasoner failure) must still surface on
    # the distinct failure channel, tagged cycle_error.
    published = []

    async def boom():
        raise RuntimeError("redis exploded during apply")

    asyncio.run(
        improver._run_and_route(
            boom, publish=lambda s, d: published.append((s, json.loads(d))), now=7.0
        )
    )
    assert len(published) == 1
    subj, body = published[0]
    assert subj == "augur.imperator.failure" and body["reason"] == "cycle_error"


def test_run_and_route_silent_on_success():
    published = []

    async def ok():
        return None

    asyncio.run(
        improver._run_and_route(ok, publish=lambda s, d: published.append(s), now=1.0)
    )
    assert published == []


def test_await_fresh_gates_on_reflection_ts_not_generated_at():
    pm = PersistenceManager(fakeredis.FakeStrictRedis(decode_responses=False))
    # Generated AFTER the trigger (high generated_at) but folding an OLDER
    # reflection (reflection_ts < epoch) must NOT count as fresh — the old
    # wall-clock gate would have wrongly passed here.
    pm.save_self_model(
        {"schema_version": 1, "generated_at": 10_000.0, "reflection_ts": 100.0}
    )
    assert (
        asyncio.run(improver._await_fresh(pm, 500.0, timeout_s=0.05, tick_s=0.02))
        is False
    )
    # A folded reflection at least as new as the trigger IS fresh.
    pm.save_self_model(
        {"schema_version": 1, "generated_at": 10_001.0, "reflection_ts": 500.0}
    )
    assert (
        asyncio.run(improver._await_fresh(pm, 500.0, timeout_s=0.05, tick_s=0.02))
        is True
    )


def test_run_cycle_applies_when_armed_and_is_idempotent():
    from imperator import proposals as P

    pm = PersistenceManager(fakeredis.FakeStrictRedis(decode_responses=False))
    pm.save_escalation_matrix({"version": "v1", "rules": {"LOW+LOW": "LOW"}})
    pm.save_self_model(
        {
            "schema_version": 1,
            "generated_at": 200.0,
            "reflection_ts": 200.0,
            "session_id": "s1",
            "blind_spots": {
                "value": [
                    {
                        "kind": "low_confidence_rule",
                        "detail": "x",
                        "evidence": "LOW+LOW",
                    }
                ],
                "fresh": True,
            },
            "recent_self_tuning": {"value": {}, "fresh": True},
        }
    )

    async def fake_gen(sm, *, client, config, now):
        return [
            P.make_proposal(
                kind="escalation_rule",
                target="LOW+LOW",
                action={"target": "MEDIUM"},
                rationale="r",
                now=now,
            )
        ]

    cfg = _Cfg()
    cfg.imperator_ii_apply_enabled = True  # arm the apply path end-to-end

    asyncio.run(
        improver.run_cycle(
            pm,
            cfg,
            now=300.0,
            session_id="s1",
            generate_fn=fake_gen,
            client=None,
            publish=lambda s, d: None,
        )
    )
    # Armed: the matrix is actually patched, idempotency is marked, record applied.
    assert pm.load_escalation_matrix()["rules"]["LOW+LOW"] == "MEDIUM"
    assert pm.is_proposal_applied(P.dedupe_key("escalation_rule", "LOW+LOW")) is True
    assert [p["status"] for p in pm.load_proposals()].count("applied") == 1

    # Second cycle, same proposal -> skipped across cycles, matrix unchanged.
    asyncio.run(
        improver.run_cycle(
            pm,
            cfg,
            now=400.0,
            session_id="s2",
            generate_fn=fake_gen,
            client=None,
            publish=lambda s, d: None,
        )
    )
    assert pm.load_escalation_matrix()["rules"]["LOW+LOW"] == "MEDIUM"
    assert [p["status"] for p in pm.load_proposals()].count("skipped") == 1


def test_run_cycle_dedupes_duplicate_target_within_cycle():
    # BUG C: two proposals with the same (kind, target) in ONE reasoner batch must
    # be deduped within the cycle — only the first (highest-ranked) is logged and
    # emitted; the second is dropped. The applied-TTL marker only blocks across
    # cycles, so without an in-cycle guard both slip through.
    from imperator import proposals as P

    pm = PersistenceManager(fakeredis.FakeStrictRedis(decode_responses=False))
    pm.save_escalation_matrix({"version": "v1", "rules": {"LOW+LOW": "LOW"}})
    pm.save_self_model(
        {
            "schema_version": 1,
            "generated_at": 200.0,
            "reflection_ts": 200.0,
            "session_id": "s1",
            "recent_self_tuning": {"value": {}, "fresh": True},
        }
    )

    async def fake_gen(sm, *, client, config, now):
        return [
            P.make_proposal(
                kind="escalation_rule",
                target="LOW+LOW",
                action={"target": "MEDIUM"},
                rationale="first",
                rank=1,
                now=now,
            ),
            P.make_proposal(
                kind="escalation_rule",
                target="LOW+LOW",
                action={"target": "HIGH"},
                rationale="dup",
                rank=2,
                now=now,
            ),
        ]

    published = []
    asyncio.run(
        improver.run_cycle(
            pm,
            _Cfg(),
            now=300.0,
            session_id="s1",
            generate_fn=fake_gen,
            client=None,
            publish=lambda subj, data: published.append((subj, json.loads(data))),
        )
    )
    props = pm.load_proposals()
    assert len(props) == 1
    assert props[0]["rationale"] == "first"  # the higher-ranked one survives
    assert [s for s, _ in published].count("augur.imperator.proposal") == 1


# ---------------------------------------------------------------------------
# on_msg / run() orchestration (BUG A head-of-line + BUG B lock serialization)
# ---------------------------------------------------------------------------


class _Msg:
    def __init__(self, subject, payload):
        self.subject = subject
        self.data = json.dumps(payload).encode()


def _seed_fresh(pm):
    pm.save_escalation_matrix({"version": "v1", "rules": {"LOW+LOW": "LOW"}})
    pm.save_self_model(
        {
            "schema_version": 1,
            "generated_at": 200.0,
            # reflection_ts newer than any past-dated trigger -> freshness gate passes
            "reflection_ts": 4_102_444_800.0,  # year 2100
            "session_id": "s1",
            "blind_spots": {
                "value": [
                    {
                        "kind": "low_confidence_rule",
                        "detail": "x",
                        "evidence": "LOW+LOW",
                    }
                ],
                "fresh": True,
            },
            "recent_self_tuning": {"value": {}, "fresh": True},
        }
    )


def _harness(pm, cfg, *, monkeypatch, last_run_at=0.0):
    """Build a make_on_msg wired to a synchronous spawn that records the cycle
    tasks but does NOT run them, so tests can drive scheduling explicitly."""
    lock = asyncio.Lock()
    last_run = [last_run_at]
    spawned = []

    def spawn(coro):
        spawned.append(coro)

    # generate_proposals is stubbed via the reasoner so no Ollama/network is hit.
    async def fake_gen(self_model, *, client, config, now):
        from imperator import proposals as P

        return [
            P.make_proposal(
                kind="escalation_rule",
                target="LOW+LOW",
                action={"target": "MEDIUM"},
                rationale="r",
                now=now,
            )
        ]

    monkeypatch.setattr(improver.reasoner, "generate_proposals", fake_gen)
    on_msg = improver.make_on_msg(
        pm,
        cfg,
        None,
        lock=lock,
        last_run=last_run,
        spawn=spawn,
        publish=lambda s, d: None,
    )
    return on_msg, lock, last_run, spawned


def test_on_msg_ignores_unconsumed_subject(monkeypatch):
    pm = PersistenceManager(fakeredis.FakeStrictRedis(decode_responses=False))
    _seed_fresh(pm)
    on_msg, _lock, _last, spawned = _harness(pm, _Cfg(), monkeypatch=monkeypatch)
    asyncio.run(on_msg(_Msg("augur.consilium.advice", {"session_id": "s1"})))
    assert spawned == []  # not a consumed subject -> no cycle


def test_on_msg_rate_limited(monkeypatch, caplog):
    pm = PersistenceManager(fakeredis.FakeStrictRedis(decode_responses=False))
    _seed_fresh(pm)
    cfg = _Cfg()
    cfg.imperator_ii_min_interval_s = 10_000_000.0  # effectively always rate-limited
    on_msg, _lock, _last, spawned = _harness(
        pm, cfg, monkeypatch=monkeypatch, last_run_at=__import__("time").time()
    )
    with caplog.at_level(logging.INFO, logger="imperator.improver"):
        asyncio.run(on_msg(_Msg("augur.disciplina.complete", {"session_id": "s1"})))
    assert spawned == []
    # The drop must be visible at INFO (live runners log at INFO) — a silent
    # drop cost a root-cause during the Task 15 live verification.
    assert any(
        "rate-limited" in r.getMessage() for r in caplog.records
    )  # within the min-interval -> dropped before spawning


def test_on_msg_spawns_cycle_and_does_not_block_on_freshness(monkeypatch):
    # BUG A: the handler must return PROMPTLY — the freshness wait happens inside
    # the spawned task, not on the dispatch path. We prove this by making the
    # cycle's freshness wait never resolve; on_msg must still complete and the
    # rate-limit anchor must be set the instant the trigger is accepted.
    pm = PersistenceManager(fakeredis.FakeStrictRedis(decode_responses=False))
    pm.save_self_model({"schema_version": 1, "reflection_ts": 0.0})  # never fresh

    blocked = asyncio.Event()

    async def never_fresh(_pm, _epoch, _timeout, tick_s=0.5):
        blocked.set()
        await asyncio.Event().wait()  # block forever
        return False

    monkeypatch.setattr(improver, "_await_fresh", never_fresh)
    cfg = _Cfg()
    cfg.imperator_ii_min_interval_s = 0.0
    cfg.imperator_ii_freshness_timeout_s = 60.0

    async def scenario():
        lock = asyncio.Lock()
        last_run = [0.0]
        tasks = []

        def spawn(coro):
            tasks.append(asyncio.ensure_future(coro))

        on_msg = improver.make_on_msg(
            pm,
            cfg,
            None,
            lock=lock,
            last_run=last_run,
            spawn=spawn,
            publish=lambda s, d: None,
        )
        # A future-dated trigger so epoch is truthy and the freshness gate runs.
        await on_msg(
            _Msg(
                "augur.disciplina.complete", {"timestamp": "2030-01-01T00:00:00+00:00"}
            )
        )
        # Handler returned without waiting for freshness:
        assert last_run[0] > 0.0  # rate-limit anchored at acceptance
        assert lock.locked()  # cycle task holds the lock (in flight)
        # Let the spawned task run up to its (forever) freshness wait.
        await asyncio.sleep(0)
        assert blocked.is_set()  # the wait is happening in the TASK, not the handler
        for t in tasks:
            t.cancel()
        return True

    assert asyncio.run(scenario()) is True


def test_on_msg_drops_trigger_while_cycle_in_flight(monkeypatch, caplog):
    # BUG B: at most one cycle in flight. While the first cycle holds the lock, a
    # second consumed trigger is dropped (not queued) — proven by only one spawn.
    pm = PersistenceManager(fakeredis.FakeStrictRedis(decode_responses=False))
    pm.save_self_model({"schema_version": 1, "reflection_ts": 0.0})

    async def never_fresh(_pm, _epoch, _timeout, tick_s=0.5):
        await asyncio.Event().wait()
        return False

    monkeypatch.setattr(improver, "_await_fresh", never_fresh)
    cfg = _Cfg()
    cfg.imperator_ii_min_interval_s = 0.0

    async def scenario():
        lock = asyncio.Lock()
        last_run = [0.0]
        tasks = []

        def spawn(coro):
            tasks.append(asyncio.ensure_future(coro))

        on_msg = improver.make_on_msg(
            pm,
            cfg,
            None,
            lock=lock,
            last_run=last_run,
            spawn=spawn,
            publish=lambda s, d: None,
        )
        m = _Msg(
            "augur.disciplina.complete", {"timestamp": "2030-01-01T00:00:00+00:00"}
        )
        await on_msg(m)  # accepted, lock now held by task
        with caplog.at_level(logging.INFO, logger="imperator.improver"):
            await on_msg(m)  # second trigger: lock held -> dropped
        assert len(tasks) == 1
        # The drop must be visible at INFO, same as the rate-limit drop.
        assert any("in flight" in r.getMessage() for r in caplog.records)
        for t in tasks:
            t.cancel()
        return True

    assert asyncio.run(scenario()) is True


def test_on_msg_freshness_skip_emits_no_proposal(monkeypatch):
    # The freshness-skip path: epoch is truthy but the model never folds it in
    # time -> the cycle returns without logging/emitting any proposal, and no
    # failure is published (a skip is not a failure).
    pm = PersistenceManager(fakeredis.FakeStrictRedis(decode_responses=False))
    pm.save_self_model({"schema_version": 1, "reflection_ts": 0.0})

    async def stale(_pm, _epoch, _timeout, tick_s=0.5):
        return False  # never fresh

    monkeypatch.setattr(improver, "_await_fresh", stale)
    cfg = _Cfg()
    cfg.imperator_ii_min_interval_s = 0.0
    published = []

    async def scenario():
        await improver._cycle(
            pm,
            cfg,
            None,
            payload={"timestamp": "2030-01-01T00:00:00+00:00", "session_id": "s1"},
            now=1.0,
            publish=lambda s, d: published.append((s, json.loads(d))),
        )

    asyncio.run(scenario())
    assert published == []  # neither a proposal nor a failure
    assert pm.load_proposals() == []


def test_on_msg_lock_released_after_cycle_completes(monkeypatch):
    # After a normal cycle the lock is released, so a later (non-rate-limited)
    # trigger is accepted again.
    pm = PersistenceManager(fakeredis.FakeStrictRedis(decode_responses=False))
    _seed_fresh(pm)
    cfg = _Cfg()
    cfg.imperator_ii_min_interval_s = 0.0

    async def scenario():
        lock = asyncio.Lock()
        last_run = [0.0]
        tasks = []

        def spawn(coro):
            tasks.append(asyncio.ensure_future(coro))

        async def fake_gen(self_model, *, client, config, now):
            from imperator import proposals as P

            return [
                P.make_proposal(
                    kind="escalation_rule",
                    target="LOW+LOW",
                    action={"target": "MEDIUM"},
                    rationale="r",
                    now=now,
                )
            ]

        monkeypatch.setattr(improver.reasoner, "generate_proposals", fake_gen)
        on_msg = improver.make_on_msg(
            pm,
            cfg,
            None,
            lock=lock,
            last_run=last_run,
            spawn=spawn,
            publish=lambda s, d: None,
        )
        await on_msg(
            _Msg(
                "augur.disciplina.complete",
                {"timestamp": "2020-01-01T00:00:00+00:00", "session_id": "s1"},
            )
        )
        await asyncio.gather(*tasks)
        assert not lock.locked()  # released after the cycle finished
        assert len(pm.load_proposals()) == 1
        return True

    assert asyncio.run(scenario()) is True


class _Cfg:
    imperator_ii_apply_enabled = False
    imperator_ii_max_proposals_per_cycle = 5
    imperator_ii_min_interval_s = 60.0
    imperator_ii_freshness_timeout_s = 15.0
    imperator_ii_dedupe_staleness_s = 86400.0
    min_prompt_len = 20
    prompt_forbidden_patterns = ()
    ollama_url = "x"
    ollama_model = "m"
    ollama_timeout = 1.0
    imperator_ii_num_predict = 64
