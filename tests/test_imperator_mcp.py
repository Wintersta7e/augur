import json

import fakeredis

import augur_mcp.augur_server as srv


def test_get_auspices_warming_up(monkeypatch):
    fake = fakeredis.FakeStrictRedis(decode_responses=False)
    monkeypatch.setattr(srv, "_new_redis", lambda: fake)
    assert srv.get_auspices() == {"status": "warming_up"}


def test_get_auspices_returns_snapshot(monkeypatch):
    fake = fakeredis.FakeStrictRedis(decode_responses=False)
    fake.set(
        "augur:imperator:auspices",
        json.dumps({"schema_version": 1, "salience": {"value": 0.4}}),
    )
    monkeypatch.setattr(srv, "_new_redis", lambda: fake)
    assert srv.get_auspices()["salience"]["value"] == 0.4


def test_imperator_registered_as_component():
    assert "imperator" in srv.COMPONENT_COMMANDS
    assert srv.COMPONENT_COMMANDS["imperator"][-1] == "imperator.awareness"
