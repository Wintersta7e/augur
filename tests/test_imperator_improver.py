import asyncio
import json

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


class _Cfg:
    imperator_ii_apply_enabled = False
    imperator_ii_max_proposals_per_cycle = 5
    imperator_ii_dedupe_staleness_s = 86400.0
    min_prompt_len = 20
    prompt_forbidden_patterns = ()
    ollama_url = "x"
    ollama_model = "m"
    ollama_timeout = 1.0
    imperator_ii_num_predict = 64
