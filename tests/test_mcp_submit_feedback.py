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
from unittest.mock import AsyncMock, patch

import augur_mcp.augur_server as srv


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
