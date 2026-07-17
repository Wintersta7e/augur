"""LearnContext value type + the fail-closed provenance resolver (spec 4.3c).

`LearnContext` is the event's provenance resolved ONCE at ingestion and threaded
through the learning paths — session id, learnable bool, and origin kept together
so a write can log and assert its own provenance instead of re-deriving it (or
forgetting to). `resolve_learn_context` is the single reader; `is_learnable_session`
delegates to it so the two can never disagree. Inert for now: no production caller.
"""

from __future__ import annotations

import dataclasses
import json

import fakeredis
import pytest

from tabula.persistence import PersistenceManager
from tabula.provenance import LearnContext
from tabula.session import REDIS_KEY_META, build_session_meta


def _pm() -> PersistenceManager:
    return PersistenceManager(fakeredis.FakeStrictRedis(decode_responses=True))


def _write(pm: PersistenceManager, sid: str, origin: str) -> None:
    pm._r.set(
        REDIS_KEY_META.format(sid=sid),
        json.dumps(
            build_session_meta(sid, origin=origin, created_by="x", started_at="t")
        ),
    )


class TestLearnContextValue:
    def test_dry_run_is_the_negation_of_learnable(self) -> None:
        assert LearnContext("s", True, "real").dry_run is False
        assert LearnContext("s", False, "synthetic").dry_run is True

    def test_is_frozen(self) -> None:
        ctx = LearnContext("s", True, "real")
        with pytest.raises(dataclasses.FrozenInstanceError):
            ctx.learnable = False  # type: ignore[misc]

    def test_unknown_is_the_failclosed_default(self) -> None:
        ctx = LearnContext.unknown("s")
        assert ctx.learnable is False
        assert ctx.origin == "unknown"
        assert ctx.session_id == "s"
        assert ctx.dry_run is True

    def test_unknown_defaults_session_to_none(self) -> None:
        assert LearnContext.unknown().session_id is None


class TestResolveLearnContext:
    def test_real_session(self) -> None:
        pm = _pm()
        _write(pm, "s1", "real")
        ctx = pm.resolve_learn_context("s1")
        assert ctx == LearnContext("s1", True, "real")

    def test_synthetic_session(self) -> None:
        pm = _pm()
        _write(pm, "s1", "synthetic")
        ctx = pm.resolve_learn_context("s1")
        assert ctx.learnable is False
        assert ctx.origin == "synthetic"

    def test_unattributed_session(self) -> None:
        pm = _pm()
        _write(pm, "s1", "unattributed")
        ctx = pm.resolve_learn_context("s1")
        assert ctx.learnable is False
        assert ctx.origin == "unattributed"

    def test_missing_session_fails_closed(self) -> None:
        ctx = _pm().resolve_learn_context("never-written")
        assert ctx.learnable is False
        assert ctx.origin == "unknown"

    def test_none_fails_closed(self) -> None:
        ctx = _pm().resolve_learn_context(None)
        assert ctx.learnable is False
        assert ctx.origin == "unknown"
        assert ctx.session_id is None

    def test_corrupt_json_fails_closed(self) -> None:
        pm = _pm()
        pm._r.set(REDIS_KEY_META.format(sid="s1"), "{not json")
        assert pm.resolve_learn_context("s1") == LearnContext("s1", False, "unknown")

    def test_non_dict_fails_closed(self) -> None:
        pm = _pm()
        pm._r.set(REDIS_KEY_META.format(sid="s1"), json.dumps([1, 2, 3]))
        assert pm.resolve_learn_context("s1").learnable is False

    def test_record_without_origin_reports_unknown_origin(self) -> None:
        pm = _pm()
        pm._r.set(
            REDIS_KEY_META.format(sid="s1"),
            json.dumps({"session_id": "s1", "learnable": True}),
        )
        ctx = pm.resolve_learn_context("s1")
        assert ctx.origin == "unknown"

    def test_redis_error_fails_closed_never_raises(self) -> None:
        class Boom:
            def get(self, *_a, **_k):
                raise RuntimeError("redis down")

        ctx = PersistenceManager(Boom()).resolve_learn_context("s1")
        assert ctx == LearnContext("s1", False, "unknown")


class TestResolverAndPredicateAgree:
    """is_learnable_session must equal resolve_learn_context(...).learnable — always.

    Two independent readers of the same field would be a seam where a bug can
    hide (cf. the credibility-arm timestamp bug). Pin their agreement.
    """

    @pytest.mark.parametrize(
        "setup",
        [
            lambda pm: _write(pm, "s", "real"),
            lambda pm: _write(pm, "s", "synthetic"),
            lambda pm: _write(pm, "s", "unattributed"),
            lambda pm: None,  # unknown / missing
            lambda pm: pm._r.set(REDIS_KEY_META.format(sid="s"), "{bad"),
        ],
    )
    def test_agreement(self, setup) -> None:
        pm = _pm()
        setup(pm)
        assert pm.is_learnable_session("s") == pm.resolve_learn_context("s").learnable
