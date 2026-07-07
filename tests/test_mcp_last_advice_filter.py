"""get_last_advice domain filter — correlated (domain="multi") matching."""

import fakeredis
import pytest

from tabula.persistence import PersistenceManager
import augur_mcp.augur_server as server


@pytest.fixture()
def fake_r(monkeypatch):
    r = fakeredis.FakeStrictRedis(decode_responses=False)
    monkeypatch.setattr(server, "_new_redis", lambda: r)
    return r


def test_single_domain_filter_matches_exact(fake_r):
    PersistenceManager(fake_r).save_last_advice(
        {"domain": "typing", "advice": "a", "decision_id": "d1"}
    )
    assert server.get_last_advice(domain="typing")["decision_id"] == "d1"
    assert "error" in server.get_last_advice(domain="chess")


def test_multi_domain_record_matches_involved_domains(fake_r):
    # Correlated advice is stored with domain="multi"; a domain filter must
    # still return it for any involved domain.
    PersistenceManager(fake_r).save_last_advice(
        {
            "domain": "multi",
            "involved_domains": ["typing", "activity_focus"],
            "advice": "a",
            "decision_id": "d2",
        }
    )
    assert server.get_last_advice(domain="typing")["decision_id"] == "d2"
    assert server.get_last_advice(domain="activity_focus")["decision_id"] == "d2"
    assert "error" in server.get_last_advice(domain="chess")


def test_multi_without_involved_domains_stays_unmatched(fake_r):
    PersistenceManager(fake_r).save_last_advice(
        {"domain": "multi", "advice": "a", "decision_id": "d3"}
    )
    assert "error" in server.get_last_advice(domain="typing")
    # unfiltered read still returns it
    assert server.get_last_advice()["decision_id"] == "d3"
