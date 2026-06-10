"""The 7th reflection pass: ingest advised correlations → plan → atomic apply."""

import fakeredis
import pytest

from tabula.config import AugurConfig
from tabula.persistence import PersistenceManager
from disciplina.reflection_engine import run_memory_sweep

CFG = AugurConfig()


@pytest.fixture
def pm():
    return PersistenceManager(fakeredis.FakeStrictRedis(decode_responses=True))


def _feedback(advice_events):
    return {"session_id": "s1", "advice_events": advice_events}


def _ev(domains, rule_key="LOW+LOW", severity="medium", correlation_found=True):
    return {
        "correlation_found": correlation_found,
        "rule_key": rule_key,
        "involved_domains": domains,
        "severity": severity,
    }


def test_creates_episodic_from_advised_correlations(pm):
    pm.save_feedback("s1", _feedback([_ev(["chess", "typing"])]))
    out = run_memory_sweep("s1", pm, CFG)
    assert out["created"] == 1
    assert pm.active_session_count() == 1
    assert len(pm.load_all_memory_states()) == 1
    assert pm.is_tuning_applied("s1", pass_name="memory") is True  # marker after commit


def test_skips_passthrough_and_disabled(pm):
    pm.save_feedback(
        "s1", _feedback([_ev(["chess"], rule_key=None, correlation_found=False)])
    )
    out = run_memory_sweep("s1", pm, CFG)
    assert out["created"] == 0
    pm.save_feedback("s2", _feedback([_ev(["chess", "typing"])]))
    out2 = run_memory_sweep("s2", pm, AugurConfig(memory_store_enabled=False))
    assert out2.get("skipped") is True
    assert pm.active_session_count() == 1  # only s1 counted


def test_idempotent_double_run(pm):
    pm.save_feedback("s1", _feedback([_ev(["chess", "typing"])]))
    run_memory_sweep("s1", pm, CFG)
    out = run_memory_sweep("s1", pm, CFG)  # re-fire
    assert out.get("skipped") is True
    assert pm.active_session_count() == 1


def test_recurrence_reviews_grows_s(pm):
    pm.save_feedback("s1", _feedback([_ev(["chess", "typing"])]))
    run_memory_sweep("s1", pm, CFG)
    pm.save_feedback("s2", _feedback([_ev(["chess", "typing"])]))
    out = run_memory_sweep("s2", pm, CFG)
    assert out["reviewed"] == 1
    state = pm.load_all_memory_states()[0]
    assert state["S"] == 1.5
