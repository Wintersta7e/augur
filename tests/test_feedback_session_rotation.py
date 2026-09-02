"""Session rotation in the feedback collector (matrix-tuning blocker A).

The collector memoized ONE session_id for the whole process: on_session_end
never reset current_session_id nor cleared advice_events, so every
augur.responsum.complete carried the same id. Disciplina's correlation-tuning
pass dedups on that id (augur:tuning_applied:correlation:{session_id}), so it
ran ONCE per process and the matrix EWMA was capped at a single step — never
crossing the disable threshold.

These tests drive advice A -> session.end -> advice B -> session.end and assert
the two complete publishes carry DISTINCT session_ids and that each session's
feedback in Redis holds only its own advice (the batch cleared between
sessions). Also covers on_feedback (the headless explicit-feedback path).
"""

from __future__ import annotations

import asyncio
import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import fakeredis
import pytest

import responsum.feedback_collector as fc
from tabula.persistence import PersistenceManager


async def _drive(
    monkeypatch: pytest.MonkeyPatch,
    feed: Any,
    stdin_reader: Any = None,
    tty: bool = True,
) -> dict[str, Any]:
    """Start run(), capture subscriber closures, run a feed coroutine.

    Uses a real fakeredis-backed PersistenceManager so pm.get_feedback(sid)
    reflects exactly what the collector persisted. The fake NATS records every
    publish so the test can read back the augur.responsum.complete payloads.
    ``feed(captured, ctx)`` is awaited with the captured callbacks + a ctx dict
    carrying the live pm / fake redis / recorded publishes.
    """
    captured: dict[str, Any] = {}
    subjects = {
        fc.SUBJECT_ADVICE: "on_advice",
        fc.SUBJECT_PERCEPTION: "on_perception",
        fc.SUBJECT_SESSION_END: "on_session_end",
        fc.SUBJECT_SUPPRESSED: "on_suppressed",
        fc.SUBJECT_FEEDBACK: "on_feedback",
    }

    async def fake_subscribe(subject: str, cb: Any) -> MagicMock:
        if subject in subjects:
            captured[subjects[subject]] = cb
        sub = MagicMock()
        sub.unsubscribe = AsyncMock()
        return sub

    published: list[dict[str, Any]] = []

    async def fake_publish(subject: str, data: bytes) -> None:
        published.append({"subject": subject, "data": json.loads(data.decode())})

    fake_nc = MagicMock()
    fake_nc.subscribe = fake_subscribe
    fake_nc.close = AsyncMock()
    fake_nc.publish = fake_publish

    async def fake_connect(*args: Any, **kwargs: Any) -> MagicMock:
        return fake_nc

    monkeypatch.setattr(fc.nats, "connect", fake_connect)

    fake_redis = fakeredis.FakeStrictRedis(decode_responses=True)
    pm = PersistenceManager(fake_redis)
    monkeypatch.setattr(fc, "PersistenceManager", lambda _r: pm)
    monkeypatch.setattr(fc, "connect_redis", lambda _c: fake_redis)

    async def fake_read(_timeout: float) -> str:
        return "s"

    monkeypatch.setattr(fc, "read_stdin_with_timeout", stdin_reader or fake_read)
    # These drive the interactive path, so they must present as a terminal;
    # without a TTY the collector skips the prompt entirely by design.
    monkeypatch.setattr(fc.sys.stdin, "isatty", lambda: tty)

    task = asyncio.create_task(fc.run())
    for _ in range(50):
        await asyncio.sleep(0)
        if len(captured) == len(subjects):
            break

    ctx = {"pm": pm, "redis": fake_redis, "published": published, "captured": captured}
    await feed(captured, ctx)

    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    return ctx


def _set_session(redis: Any, sid: str) -> None:
    redis.set("augur:session:current", json.dumps({"session_id": sid}))


def _advice_msg(player: str, decision_id: str) -> MagicMock:
    msg = MagicMock()
    msg.data = json.dumps(
        {
            "domain": "chess",
            "player": player,
            "severity": "medium",
            "think_time": 9.0,
            "decision_id": decision_id,
        }
    ).encode()
    return msg


def _feedback_msg(**fields: Any) -> MagicMock:
    msg = MagicMock()
    msg.data = json.dumps(fields).encode()
    return msg


def _completes(published: list[dict]) -> list[dict]:
    return [p for p in published if p["subject"] == fc.SUBJECT_FEEDBACK_COMPLETE]


# ── session rotation ─────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_two_sessions_publish_distinct_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def feed(cap: dict, ctx: dict) -> None:
        _set_session(ctx["redis"], "sess-A")
        await cap["on_advice"](_advice_msg("white", "dec-A"))
        await cap["on_session_end"](MagicMock())
        # New session begins: a fresh id must be derived because the collector
        # rotated current_session_id to None on the first session.end.
        _set_session(ctx["redis"], "sess-B")
        await cap["on_advice"](_advice_msg("black", "dec-B"))
        await cap["on_session_end"](MagicMock())

    ctx = await _drive(monkeypatch, feed)
    completes = _completes(ctx["published"])
    assert len(completes) == 2
    ids = [c["data"]["session_id"] for c in completes]
    assert ids == ["sess-A", "sess-B"]
    assert ids[0] != ids[1]


@pytest.mark.asyncio
async def test_each_session_feedback_holds_only_its_own_advice(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def feed(cap: dict, ctx: dict) -> None:
        _set_session(ctx["redis"], "sess-A")
        await cap["on_advice"](_advice_msg("white", "dec-A"))
        await cap["on_session_end"](MagicMock())
        _set_session(ctx["redis"], "sess-B")
        await cap["on_advice"](_advice_msg("black", "dec-B"))
        await cap["on_session_end"](MagicMock())

    ctx = await _drive(monkeypatch, feed)
    pm = ctx["pm"]

    fb_a = pm.get_feedback("sess-A")
    fb_b = pm.get_feedback("sess-B")
    assert fb_a is not None and fb_b is not None

    a_ids = [e["decision_id"] for e in fb_a["advice_events"]]
    b_ids = [e["decision_id"] for e in fb_b["advice_events"]]
    # The batch cleared between sessions: A's feedback holds only A's advice.
    assert a_ids == ["dec-A"]
    assert b_ids == ["dec-B"]


@pytest.mark.asyncio
async def test_empty_session_end_still_rotates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # An empty session.end (no advice, no gate decisions) must rotate too, so a
    # later real session derives a fresh id rather than inheriting the stale one.
    async def feed(cap: dict, ctx: dict) -> None:
        _set_session(ctx["redis"], "sess-empty")
        await cap["on_session_end"](MagicMock())  # empty -> early return + rotate
        _set_session(ctx["redis"], "sess-real")
        await cap["on_advice"](_advice_msg("white", "dec-real"))
        await cap["on_session_end"](MagicMock())

    ctx = await _drive(monkeypatch, feed)
    completes = _completes(ctx["published"])
    # Only the real session publishes a complete; it carries the FRESH id.
    assert len(completes) == 1
    assert completes[0]["data"]["session_id"] == "sess-real"


# ── on_feedback (headless explicit-feedback path) ────────────────────────────


@pytest.mark.asyncio
async def test_on_feedback_sets_rating_by_decision_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def feed(cap: dict, ctx: dict) -> None:
        _set_session(ctx["redis"], "sess-1")
        await cap["on_advice"](_advice_msg("white", "dec-1"))
        await cap["on_feedback"](_feedback_msg(decision_id="dec-1", rating="n"))
        await cap["on_session_end"](MagicMock())

    ctx = await _drive(monkeypatch, feed)
    fb = ctx["pm"].get_feedback("sess-1")
    assert fb["advice_events"][0]["explicit_rating"] == "n"


@pytest.mark.asyncio
async def test_on_feedback_ignores_unknown_decision_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def feed(cap: dict, ctx: dict) -> None:
        _set_session(ctx["redis"], "sess-1")
        await cap["on_advice"](_advice_msg("white", "dec-1"))
        await cap["on_feedback"](_feedback_msg(decision_id="nope", rating="y"))
        await cap["on_session_end"](MagicMock())

    ctx = await _drive(monkeypatch, feed)
    fb = ctx["pm"].get_feedback("sess-1")
    # Unmatched feedback is a no-op: the advice keeps its default rating.
    assert fb["advice_events"][0]["explicit_rating"] == "no_response"


@pytest.mark.asyncio
async def test_on_feedback_rejects_invalid_rating(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def feed(cap: dict, ctx: dict) -> None:
        _set_session(ctx["redis"], "sess-1")
        await cap["on_advice"](_advice_msg("white", "dec-1"))
        await cap["on_feedback"](_feedback_msg(decision_id="dec-1", rating="maybe"))
        await cap["on_session_end"](MagicMock())

    ctx = await _drive(monkeypatch, feed)
    fb = ctx["pm"].get_feedback("sess-1")
    # Invalid rating rejected before mutation: rating unchanged.
    assert fb["advice_events"][0]["explicit_rating"] == "no_response"


@pytest.mark.asyncio
async def test_on_feedback_matches_by_advice_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # on_advice generates the advice_id; capture it via the saved record, then
    # confirm on_feedback can target it by advice_id when decision_id is absent.
    async def feed(cap: dict, ctx: dict) -> None:
        _set_session(ctx["redis"], "sess-1")
        await cap["on_advice"](_advice_msg("white", "dec-1"))
        # Read back the assigned advice_id from the intermediate save.
        fb = ctx["pm"].get_feedback("sess-1")
        advice_id = fb["advice_events"][0]["advice_id"]
        await cap["on_feedback"](_feedback_msg(advice_id=advice_id, rating="y"))
        await cap["on_session_end"](MagicMock())

    ctx = await _drive(monkeypatch, feed)
    fb = ctx["pm"].get_feedback("sess-1")
    assert fb["advice_events"][0]["explicit_rating"] == "y"


@pytest.mark.asyncio
async def test_stdin_timeout_does_not_clobber_out_of_band_rating(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Race regression: on_advice appends the PendingAdvice, then awaits the
    # stdin prompt for up to EXPLICIT_TIMEOUT_S. With an open-but-idle stdin
    # (TTY allocated, unattended) the read times out (returns None). If an
    # out-of-band rating arrives via on_feedback DURING that window, the
    # else-branch must NOT overwrite it back to 'no_response' — that would
    # silently lose the operator's headless feedback, defeating the whole
    # submit_feedback feature. Here the patched stdin reader fires on_feedback
    # mid-await (rating 'y'), then returns None to hit the timeout/else-branch.
    holder: dict[str, Any] = {}

    async def stdin_injects_feedback(_timeout: float) -> None:
        # Simulate a headless rating landing in the stdin-await window.
        await holder["on_feedback"](_feedback_msg(decision_id="dec-1", rating="y"))
        return None  # idle stdin times out -> on_advice falls to else-branch

    async def feed(cap: dict, ctx: dict) -> None:
        holder["on_feedback"] = cap["on_feedback"]
        _set_session(ctx["redis"], "sess-1")
        await cap["on_advice"](_advice_msg("white", "dec-1"))
        await cap["on_session_end"](MagicMock())

    ctx = await _drive(monkeypatch, feed, stdin_reader=stdin_injects_feedback)
    fb = ctx["pm"].get_feedback("sess-1")
    # The out-of-band 'y' must survive the stdin timeout, not be reset.
    assert fb["advice_events"][0]["explicit_rating"] == "y"


@pytest.mark.asyncio
async def test_headless_advice_skips_the_prompt_but_still_takes_a_rating(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Deployed, this process has no terminal.

    The prompt could then only print into the void while stalling the advice
    handler for up to EXPLICIT_TIMEOUT_S per advice, so it is skipped. The
    out-of-band path (the submit_feedback tool) must still record a rating —
    it is the only way explicit feedback reaches the system in a container,
    and precision, utility and credibility are computed from it alone.
    """
    holder: dict[str, Any] = {}

    async def _unreachable_stdin(_timeout: float) -> None:
        raise AssertionError("stdin must not be read without a TTY")

    async def feed(cap: dict, ctx: dict) -> None:
        holder["on_feedback"] = cap["on_feedback"]
        _set_session(ctx["redis"], "sess-headless")
        await cap["on_advice"](_advice_msg("white", "dec-h1"))
        # Arrives well after any stdin window would have closed.
        await holder["on_feedback"](_feedback_msg(decision_id="dec-h1", rating="n"))
        await cap["on_session_end"](MagicMock())

    ctx = await _drive(monkeypatch, feed, stdin_reader=_unreachable_stdin, tty=False)
    fb = ctx["pm"].get_feedback("sess-headless")
    assert fb["advice_events"][0]["explicit_rating"] == "n"
