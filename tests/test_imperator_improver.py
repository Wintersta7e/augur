import asyncio
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
