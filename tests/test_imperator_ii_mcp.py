import json
import fakeredis
import augur_mcp.augur_server as srv


def test_get_proposals(monkeypatch):
    fake = fakeredis.FakeStrictRedis(decode_responses=False)
    fake.lpush(
        "augur:imperator:proposals",
        json.dumps({"proposal_id": "p1", "status": "logged"}),
    )
    monkeypatch.setattr(srv, "_new_redis", lambda: fake)
    assert srv.get_proposals()["proposals"][0]["proposal_id"] == "p1"


def test_imperator_ii_component():
    assert srv.COMPONENT_COMMANDS["imperator_ii"][-1] == "imperator.improver"
