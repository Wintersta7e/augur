"""Unit tests for the get_active_session validity helper."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock


from blackboard.session import get_active_session


def _make_record(
    session_id: str = "sess-abc",
    status: str = "active",
    age_h: float = 0.0,
) -> str:
    started = datetime.now(timezone.utc) - timedelta(hours=age_h)
    return json.dumps(
        {
            "session_id": session_id,
            "started_at": started.isoformat(),
            "status": status,
        }
    )


def test_returns_session_id_when_active_and_fresh():
    r = MagicMock()
    r.get.return_value = _make_record(status="active", age_h=1.0)
    assert get_active_session(r, max_age_h=12.0) == "sess-abc"


def test_returns_none_when_key_missing():
    r = MagicMock()
    r.get.return_value = None
    assert get_active_session(r, max_age_h=12.0) is None


def test_returns_none_when_status_is_ended():
    r = MagicMock()
    r.get.return_value = _make_record(status="ended", age_h=1.0)
    assert get_active_session(r, max_age_h=12.0) is None


def test_returns_none_when_session_older_than_max_age():
    r = MagicMock()
    r.get.return_value = _make_record(status="active", age_h=24.0)
    assert get_active_session(r, max_age_h=12.0) is None


def test_returns_session_id_when_started_at_missing():
    """Older session records may not have started_at — accept them as active."""
    r = MagicMock()
    r.get.return_value = json.dumps({"session_id": "sess-old", "status": "active"})
    assert get_active_session(r, max_age_h=12.0) == "sess-old"


def test_returns_none_when_record_malformed():
    r = MagicMock()
    r.get.return_value = b"not-json"
    assert get_active_session(r, max_age_h=12.0) is None


def test_returns_none_when_record_missing_session_id():
    r = MagicMock()
    r.get.return_value = json.dumps({"status": "active"})
    assert get_active_session(r, max_age_h=12.0) is None


def test_returns_none_when_started_at_is_naive_datetime():
    """A naive (no-tz) started_at would TypeError the subtraction — must be caught."""
    r = MagicMock()
    r.get.return_value = json.dumps(
        {
            "session_id": "sess-naive",
            "started_at": "2026-05-16T10:30:00",  # no tz suffix
            "status": "active",
        }
    )
    assert get_active_session(r, max_age_h=12.0) is None


def test_returns_none_when_started_at_is_not_a_string():
    """A non-string started_at would AttributeError fromisoformat — must be caught."""
    r = MagicMock()
    r.get.return_value = json.dumps(
        {
            "session_id": "sess-bad",
            "started_at": 12345,
            "status": "active",
        }
    )
    assert get_active_session(r, max_age_h=12.0) is None


def test_returns_none_when_redis_raises():
    """Redis connection errors should be treated as 'no valid session'."""
    r = MagicMock()
    r.get.side_effect = ConnectionError("Redis unreachable")
    assert get_active_session(r, max_age_h=12.0) is None
