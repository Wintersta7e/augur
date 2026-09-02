"""Tests for PerceptionEvent — the universal perception envelope.

Every component in the pipeline depends on this contract. If serialization
is lossy or from_json rejects valid payloads, the entire pipeline breaks.
"""

from __future__ import annotations

import json

import pytest

from tabula.contracts import PerceptionEvent, is_sentinel_entity


@pytest.fixture
def sample_event() -> PerceptionEvent:
    return PerceptionEvent(
        domain="chess",
        stream_id="chess_timing",
        entity="white",
        event_type="move",
        value=12.5,
        unit="seconds",
        context={"move_san": "Nf3", "move_number": 7},
        timestamp="2025-01-01T00:00:00Z",
        session_id="abc-123",
    )


class TestRoundTrip:
    def test_json_round_trip(self, sample_event: PerceptionEvent) -> None:
        json_str = sample_event.to_json()
        restored = PerceptionEvent.from_json(json_str)
        assert restored.domain == sample_event.domain
        assert restored.value == sample_event.value
        assert restored.context == sample_event.context
        assert restored.session_id == sample_event.session_id

    def test_bytes_round_trip(self, sample_event: PerceptionEvent) -> None:
        raw = sample_event.to_bytes()
        assert isinstance(raw, bytes)
        restored = PerceptionEvent.from_json(raw)
        assert restored.domain == sample_event.domain
        assert restored.value == sample_event.value

    def test_context_preserved_exactly(self, sample_event: PerceptionEvent) -> None:
        json_str = sample_event.to_json()
        restored = PerceptionEvent.from_json(json_str)
        assert restored.context["move_san"] == "Nf3"
        assert restored.context["move_number"] == 7


class TestFromJson:
    def test_rejects_invalid_json(self) -> None:
        # SEC-03: from_json now wraps json.JSONDecodeError in a ValueError
        # with a clear diagnostic message so the ingestion boundary surfaces
        # schema problems with actionable context rather than leaking the
        # underlying parser's error.
        with pytest.raises(ValueError, match="invalid JSON"):
            PerceptionEvent.from_json("not json")

    def test_rejects_missing_fields(self) -> None:
        # SEC-03: strict schema validation reports the specific missing
        # fields rather than crashing on dataclass __init__.
        with pytest.raises(ValueError, match="missing required fields"):
            PerceptionEvent.from_json('{"domain": "chess"}')

    def test_rejects_extra_fields(self) -> None:
        # SEC-03: extra fields indicate a schema mismatch or a spoofed
        # payload and are now rejected at the ingestion boundary with a
        # clear diagnostic (previously crashed with a bare TypeError from
        # dataclass __init__).
        data = {
            "domain": "chess",
            "stream_id": "chess_timing",
            "entity": "white",
            "event_type": "move",
            "value": 5.0,
            "unit": "seconds",
            "context": {},
            "timestamp": "2025-01-01T00:00:00Z",
            "session_id": "abc",
            "extra_field": "should_not_crash",
        }
        with pytest.raises(ValueError, match="unexpected fields"):
            PerceptionEvent.from_json(json.dumps(data))

    def test_rejects_non_dict_json(self) -> None:
        # SEC-03: a JSON array or scalar at the top level is not a valid
        # PerceptionEvent and should be rejected explicitly.
        with pytest.raises(ValueError, match="expected a JSON object"):
            PerceptionEvent.from_json('["not", "an", "event"]')

    def test_rejects_non_dict_context(self) -> None:
        # ARCH-08: context must be a dict. Nulls or arrays are rejected
        # at construction time so consumers can rely on .get() semantics.
        data = {
            "domain": "chess",
            "stream_id": "chess_timing",
            "entity": "white",
            "event_type": "move",
            "value": 5.0,
            "unit": "seconds",
            "context": None,
            "timestamp": "2025-01-01T00:00:00Z",
            "session_id": "abc",
        }
        with pytest.raises(TypeError, match="context must be a dict"):
            PerceptionEvent.from_json(json.dumps(data))

    def test_rejects_non_numeric_value(self) -> None:
        # ARCH-08: value must be numeric. A string like "high" is not a
        # valid primary signal and should be rejected.
        data = {
            "domain": "chess",
            "stream_id": "chess_timing",
            "entity": "white",
            "event_type": "move",
            "value": "high",
            "unit": "seconds",
            "context": {},
            "timestamp": "2025-01-01T00:00:00Z",
            "session_id": "abc",
        }
        with pytest.raises(TypeError, match="value must be numeric"):
            PerceptionEvent.from_json(json.dumps(data))

    def test_coerces_numeric_string_value(self) -> None:
        # Robustness: perception sources that serialize numbers as strings
        # (e.g., JSON from some shells) should still parse successfully.
        data = {
            "domain": "chess",
            "stream_id": "chess_timing",
            "entity": "white",
            "event_type": "move",
            "value": "12.5",
            "unit": "seconds",
            "context": {},
            "timestamp": "2025-01-01T00:00:00Z",
            "session_id": "abc",
        }
        event = PerceptionEvent.from_json(json.dumps(data))
        assert event.value == 12.5
        assert isinstance(event.value, float)


class TestValueTypes:
    def test_float_value_preserved(self) -> None:
        event = PerceptionEvent(
            domain="typing",
            stream_id="rhythm",
            entity="user",
            event_type="sample",
            value=0.123456789,
            unit="seconds",
            context={},
            timestamp="2025-01-01T00:00:00Z",
            session_id="x",
        )
        restored = PerceptionEvent.from_json(event.to_json())
        assert restored.value == pytest.approx(0.123456789)

    def test_empty_context(self) -> None:
        event = PerceptionEvent(
            domain="test",
            stream_id="test",
            entity="test",
            event_type="test",
            value=1.0,
            unit="s",
            context={},
            timestamp="2025-01-01T00:00:00Z",
            session_id="x",
        )
        restored = PerceptionEvent.from_json(event.to_json())
        assert restored.context == {}


class TestSentinelEntities:
    """Daemon placeholders the activity Sensus emits when it cannot name an app.

    `<no_foreground>` carries the residue of `total_dwell - idle_dwell` over a
    span shorter than the poll interval — float noise around 70ms whose stdev
    still clears the zero-variance floor, so it scores like real data. Vigil,
    Praesagium and Consilium all need the same answer, which is why the
    predicate lives in the shared base rather than in a faculty.
    """

    def test_recognizes_the_daemon_placeholders(self) -> None:
        for s in ("<no_foreground>", "<unknown>", "<denied>", "<gone>"):
            assert is_sentinel_entity(s), s

    def test_a_real_app_is_not_a_sentinel(self) -> None:
        for s in ("firefox", "text editor", "a<b>", "<not closed", "x<y>z"):
            assert not is_sentinel_entity(s), s

    def test_empty_and_none_are_not_sentinels(self) -> None:
        assert not is_sentinel_entity("")
        assert not is_sentinel_entity(None)
