"""Tests for the submit_feedback MCP tool (matrix-tuning blocker B).

The interactive stdin prompt has no TTY in a container, so deployed advice
always logs no_response and Disciplina's matrix self-tuning can never cross the
disable threshold. submit_feedback is the headless explicit-feedback path: it
publishes a y/n/no_response rating to augur.responsum.feedback, where the
collector matches it to the in-flight advice by decision_id.
"""

from __future__ import annotations

import asyncio
import json
from contextlib import contextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import augur_mcp.augur_server as srv


def _pm_ctx(pm):
    """Stand in for the server's PersistenceManager context manager."""

    @contextmanager
    def _ctx():
        yield pm

    return _ctx


def test_submit_feedback_publishes_to_responsum_feedback() -> None:
    nc = AsyncMock()
    with patch.object(srv.nats_client, "connect", AsyncMock(return_value=nc)):
        result = asyncio.run(srv.submit_feedback(decision_id="dec-1", rating="y"))
    assert result["status"] == "submitted"
    assert result["decision_id"] == "dec-1"

    published_subject = nc.publish.call_args[0][0]
    published_payload = json.loads(nc.publish.call_args[0][1].decode())
    assert published_subject == "augur.responsum.feedback"
    assert published_payload == {"decision_id": "dec-1", "rating": "y"}


def test_submit_feedback_accepts_n_and_no_response() -> None:
    nc = AsyncMock()
    with patch.object(srv.nats_client, "connect", AsyncMock(return_value=nc)):
        for rating in ("n", "no_response"):
            result = asyncio.run(
                srv.submit_feedback(decision_id="dec-1", rating=rating)
            )
            assert result["status"] == "submitted"
            payload = json.loads(nc.publish.call_args[0][1].decode())
            assert payload["rating"] == rating


def test_submit_feedback_rejects_invalid_rating() -> None:
    # A bad rating must return an error WITHOUT touching NATS.
    connect = AsyncMock()
    with patch.object(srv.nats_client, "connect", connect):
        result = asyncio.run(srv.submit_feedback(decision_id="dec-1", rating="maybe"))
    assert "error" in result
    connect.assert_not_called()


def test_submit_feedback_closes_connection() -> None:
    nc = AsyncMock()
    with patch.object(srv.nats_client, "connect", AsyncMock(return_value=nc)):
        asyncio.run(srv.submit_feedback(decision_id="dec-1", rating="y"))
    nc.close.assert_awaited()


def test_rating_the_latest_advice_needs_no_decision_id() -> None:
    """Rating must be possible from what the user actually sees.

    Requiring the decision_id meant an explicit rating could only be given by
    someone who had already gone looking for it in a log — which is why no
    rating was ever recorded, and why precision, utility, credibility and the
    dismissal rate all stayed null.
    """
    nc = AsyncMock()
    pm = MagicMock()
    pm.load_last_advice.return_value = {"decision_id": "dec-latest", "advice": "..."}
    with (
        patch.object(srv.nats_client, "connect", AsyncMock(return_value=nc)),
        patch.object(srv, "_persistence_ctx", _pm_ctx(pm)),
    ):
        result = asyncio.run(srv.submit_feedback(rating="y"))
    assert result["status"] == "submitted"
    assert result["decision_id"] == "dec-latest"
    assert json.loads(nc.publish.call_args[0][1].decode()) == {
        "decision_id": "dec-latest",
        "rating": "y",
    }


def test_explicit_decision_id_still_wins_over_the_latest() -> None:
    nc = AsyncMock()
    pm = MagicMock()
    pm.load_last_advice.return_value = {"decision_id": "dec-latest"}
    with (
        patch.object(srv.nats_client, "connect", AsyncMock(return_value=nc)),
        patch.object(srv, "_persistence_ctx", _pm_ctx(pm)),
    ):
        result = asyncio.run(srv.submit_feedback(rating="n", decision_id="dec-chosen"))
    assert result["decision_id"] == "dec-chosen"


def test_no_decision_id_anywhere_is_an_error_not_a_silent_publish() -> None:
    nc = AsyncMock()
    pm = MagicMock()
    pm.load_last_advice.return_value = None
    with (
        patch.object(srv.nats_client, "connect", AsyncMock(return_value=nc)),
        patch.object(srv, "_persistence_ctx", _pm_ctx(pm)),
    ):
        result = asyncio.run(srv.submit_feedback(rating="y"))
    assert "error" in result
    nc.publish.assert_not_called()
