"""Praesagium's episode substrate: canonical keys + compact episode entries.

Spec: docs/superpowers/specs/2026-07-09-praesagium-design.md Sec 3.1-3.2.
Pure functions only -- no Redis/NATS fixtures needed.
"""

from datetime import datetime, timezone

from praesagium.episodes import build_episode, canonical_key, parse_epoch

# ---------------------------------------------------------------------------
# canonical_key
# ---------------------------------------------------------------------------


def test_canonical_key_joins_domain_and_entity():
    assert canonical_key({"domain": "typing", "entity": "user"}) == "typing:user"


def test_canonical_key_none_for_missing_entity_value():
    for bad_entity in (None, "?", ""):
        assert canonical_key({"domain": "typing", "entity": bad_entity}) is None


def test_canonical_key_none_for_missing_entity_key():
    assert canonical_key({"domain": "typing"}) is None


def test_canonical_key_none_for_sentinel_entity():
    """A daemon placeholder must never become a mineable stream.

    `<no_foreground>` precedes nearly every app switch, so as an antecedent it
    has high support and genuine lift for "some app gains focus". It would
    clear the Wilson lower bound and the session-conditional null and be
    promoted as a real pattern — the promotion math rejects coincidence, not a
    placeholder that is definitionally present.
    """
    for sentinel in ("<no_foreground>", "<unknown>", "<denied>", "<gone>"):
        assert (
            canonical_key({"domain": "activity_focus", "entity": sentinel}) is None
        ), sentinel


def test_canonical_key_keeps_an_app_that_merely_contains_angle_brackets():
    assert (
        canonical_key({"domain": "activity_focus", "entity": "a<b>"})
        == "activity_focus:a<b>"
    )


def test_canonical_key_none_for_missing_domain():
    assert canonical_key({"entity": "user"}) is None


def test_canonical_key_none_for_falsy_domain():
    for bad_domain in (None, ""):
        assert canonical_key({"domain": bad_domain, "entity": "user"}) is None


def test_canonical_key_none_for_empty_dict():
    assert canonical_key({}) is None


# ---------------------------------------------------------------------------
# parse_epoch
# ---------------------------------------------------------------------------


def test_parse_epoch_parses_iso_with_offset():
    iso = "2026-07-09T12:00:00+00:00"
    expected = datetime.fromisoformat(iso).timestamp()
    assert parse_epoch(iso) == expected


def test_parse_epoch_matches_datetime_now_isoformat_roundtrip():
    now = datetime.now(timezone.utc)
    iso = now.isoformat()
    assert parse_epoch(iso) == now.timestamp()


def test_parse_epoch_none_for_none():
    assert parse_epoch(None) is None


def test_parse_epoch_none_for_garbage_string():
    assert parse_epoch("not-a-timestamp") is None


def test_parse_epoch_none_for_empty_string():
    assert parse_epoch("") is None


def test_parse_epoch_none_for_non_string_never_raises():
    for garbage in (12345, 12.3, ["2026-07-09T12:00:00+00:00"], {"a": 1}, object()):
        assert parse_epoch(garbage) is None  # type: ignore[arg-type]


def test_parse_epoch_none_for_naive_timestamp():
    # vigil always emits timezone-aware UTC; a naive string is malformed by
    # contract -- not silently interpreted in the host's local timezone.
    assert parse_epoch("2026-07-09T12:00:00") is None


def test_parse_epoch_parses_aware_equivalent_of_naive_case():
    assert parse_epoch("2026-07-09T12:00:00+00:00") is not None


def test_parse_epoch_hardcoded_literal_utc_midnight():
    # Independently computed: datetime(2026, 1, 1, tzinfo=timezone.utc).timestamp()
    assert parse_epoch("2026-01-01T00:00:00+00:00") == 1767225600.0


def test_parse_epoch_hardcoded_literal_non_utc_offset():
    # Independently computed: datetime.fromisoformat("2026-03-15T09:30:00+02:00").timestamp()
    assert parse_epoch("2026-03-15T09:30:00+02:00") == 1773559800.0


# ---------------------------------------------------------------------------
# build_episode
# ---------------------------------------------------------------------------


def _anomaly(**overrides):
    base = {
        "domain": "typing",
        "entity": "user",
        "severity": "medium",
        "value": 1.2,
        "timestamp": "2026-07-09T12:00:00+00:00",
        "session_id": "abc",
        "context": {},
    }
    base.update(overrides)
    return base


def test_build_episode_compact_shape():
    ep = build_episode(_anomaly())
    assert ep == {
        "k": "typing:user",
        "s": "medium",
        "t": parse_epoch("2026-07-09T12:00:00+00:00"),
    }
    assert set(ep.keys()) == {"k", "s", "t"}


def test_build_episode_lowercases_severity():
    ep = build_episode(_anomaly(severity="HIGH"))
    assert ep["s"] == "high"


def test_build_episode_defaults_missing_severity_to_low():
    anomaly = _anomaly()
    del anomaly["severity"]
    ep = build_episode(anomaly)
    assert ep["s"] == "low"


def test_build_episode_defaults_empty_severity_to_low():
    ep = build_episode(_anomaly(severity=""))
    assert ep["s"] == "low"


def test_build_episode_defaults_none_severity_to_low():
    ep = build_episode(_anomaly(severity=None))
    assert ep["s"] == "low"


def test_build_episode_coerces_non_string_severity():
    ep = build_episode(_anomaly(severity=3))
    assert ep["s"] == "3"


def test_build_episode_none_when_unkeyable():
    assert build_episode(_anomaly(entity=None)) is None
    assert build_episode(_anomaly(entity="?")) is None
    assert build_episode(_anomaly(entity="")) is None
    assert build_episode(_anomaly(domain=None)) is None
    assert build_episode(_anomaly(domain="")) is None


def test_build_episode_none_when_timestamp_unparseable():
    assert build_episode(_anomaly(timestamp="garbage")) is None


def test_build_episode_none_when_timestamp_missing():
    anomaly = _anomaly()
    del anomaly["timestamp"]
    assert build_episode(anomaly) is None


def test_build_episode_none_when_timestamp_none():
    assert build_episode(_anomaly(timestamp=None)) is None
