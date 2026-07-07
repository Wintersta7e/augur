from contextlib import contextmanager
from unittest.mock import patch, AsyncMock

import fakeredis

import augur_mcp.augur_server as srv
from tabula.persistence import PersistenceManager


def test_dialogue_turn_tool_returns_reply():
    from imperator.dialogue.engine import DialogueTurn

    async def fake_turn(session_id, text, **kw):
        return DialogueTurn(reply="ack")

    # Mock NATS and HTTP client
    mock_nc = AsyncMock()
    mock_nc.drain = AsyncMock()

    async def mock_connect(*args, **kwargs):
        return mock_nc

    mock_http_client = AsyncMock()
    mock_http_client.aclose = AsyncMock()

    mock_pm = AsyncMock()

    with patch.object(srv, "_dialogue_handle_turn", fake_turn):
        with patch.object(srv, "_persistence_ctx") as mock_pm_ctx:
            mock_pm_ctx.return_value.__enter__.return_value = mock_pm
            with patch("augur_mcp.augur_server.nats_client.connect", mock_connect):
                with patch("httpx.AsyncClient", return_value=mock_http_client):
                    result = srv.dialogue_turn(session_id="s1", message="hi")
    assert result["reply"] == "ack"


def test_dialogue_turn_tool_wraps_errors():
    async def boom(session_id, text, **kw):
        raise RuntimeError("nope")

    # Mock NATS and HTTP client
    mock_nc = AsyncMock()
    mock_nc.drain = AsyncMock()

    async def mock_connect(*args, **kwargs):
        return mock_nc

    mock_http_client = AsyncMock()
    mock_http_client.aclose = AsyncMock()

    mock_pm = AsyncMock()

    with patch.object(srv, "_dialogue_handle_turn", boom):
        with patch.object(srv, "_persistence_ctx") as mock_pm_ctx:
            mock_pm_ctx.return_value.__enter__.return_value = mock_pm
            with patch("augur_mcp.augur_server.nats_client.connect", mock_connect):
                with patch("httpx.AsyncClient", return_value=mock_http_client):
                    result = srv.dialogue_turn(session_id="s1", message="hi")
    assert "error" in result


def test_dialogue_history_scopes_to_session():
    pm = PersistenceManager(fakeredis.FakeStrictRedis(decode_responses=False))
    pm.save_dialogue_turn({"ts": 1.0, "session_id": "A", "user_text": "a1"})
    pm.save_dialogue_turn({"ts": 2.0, "session_id": "B", "user_text": "b1"})
    pm.save_dialogue_turn({"ts": 3.0, "session_id": "A", "user_text": "a2"})

    @contextmanager
    def fake_ctx():
        yield pm

    with patch.object(srv, "_persistence_ctx", fake_ctx):
        result = srv.dialogue_history(session_id="A")
    assert [t["user_text"] for t in result["turns"]] == ["a2", "a1"]
    assert all(t["session_id"] == "A" for t in result["turns"])


def test_dialogue_history_clamps_limit():
    """F10 regression: dialogue_history clamps limit like get_proposals -- a
    negative limit must not slice from the end and return an oversized wrong
    subset."""
    pm = PersistenceManager(fakeredis.FakeStrictRedis(decode_responses=False))
    for i in range(6):
        pm.save_dialogue_turn({"ts": float(i), "session_id": "A", "user_text": f"a{i}"})

    @contextmanager
    def fake_ctx():
        yield pm

    with patch.object(srv, "_persistence_ctx", fake_ctx):
        neg = srv.dialogue_history(session_id="A", limit=-2)
        big = srv.dialogue_history(session_id="A", limit=9999)
    assert len(neg["turns"]) == 1  # clamped to 1, not "all-but-2"
    assert len(big["turns"]) == 6  # clamped to 200, so all 6 stored
