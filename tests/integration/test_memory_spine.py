"""Memoria end-to-end over real Redis: recurrence reviews, decay archives.

run_memory_sweep is synchronous; the test is async only because the shared
`redis_client` fixture (tests/integration/conftest.py) is async (it cleans
augur:* before each test and yields a sync redis client).
"""

import pytest

from tabula.config import AugurConfig
from tabula.persistence import PersistenceManager
from disciplina.reflection_engine import run_memory_sweep
from tests.integration.conftest import learnable_session


def _fb(advice_events):
    return {"advice_events": advice_events}


def _ev(domains, rule_key="LOW+LOW", severity="medium"):
    return {
        "correlation_found": True,
        "rule_key": rule_key,
        "involved_domains": domains,
        "severity": severity,
    }


@pytest.mark.asyncio
async def test_recurrence_then_decay(redis_client):
    pm = PersistenceManager(redis_client)
    cfg = AugurConfig()

    # session 1: a chess+typing correlation → create
    pm.save_feedback(learnable_session("mem-s1"), _fb([_ev(["chess", "typing"])]))
    assert run_memory_sweep("mem-s1", pm, cfg)["created"] == 1

    # session 2: same pattern recurs → review (S grows 1.0 → 1.5)
    pm.save_feedback(learnable_session("mem-s2"), _fb([_ev(["chess", "typing"])]))
    assert run_memory_sweep("mem-s2", pm, cfg)["reviewed"] == 1
    state = pm.load_all_memory_states()[0]
    assert state["S"] == 1.5 and len(state["source_sessions"]) == 2
    mid = state["memory_id"]

    # many empty sessions advance the active-session clock; the non-recurring
    # memory decays past the prune floor and is archived (not deleted).
    for i in range(3, 55):
        pm.save_feedback(learnable_session(f"mem-s{i}"), _fb([]))
        run_memory_sweep(f"mem-s{i}", pm, cfg)

    assert pm.load_memory_state(mid) is None  # evicted from active tier
    assert pm.load_archived_memory(mid) is not None  # but recoverable in Cold archive
