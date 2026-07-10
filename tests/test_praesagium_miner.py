"""Praesagium miner: run_praesagium_mining orchestration -- gating (disabled /
already-processed / rate-limit), the happy-path sweep (corpus -> mine -> fold ->
merge -> expiry -> save -> mark), the expiry sweep, and the trim-loss WARN.

Spec: docs/superpowers/specs/2026-07-09-praesagium-design.md Sec 4.1, 4.6-4.7.
fakeredis-backed PersistenceManager; no real Redis/NATS/Ollama.
"""

from __future__ import annotations

import json
import logging
import time

import fakeredis
import pytest

from praesagium.miner import run_praesagium_mining
from praesagium.patterns import pattern_id
from tabula.config import AugurConfig
from tabula.persistence import PersistenceManager

_LOG_KEY = "augur:praesagium:predictions:log"


def _pm() -> PersistenceManager:
    return PersistenceManager(fakeredis.FakeStrictRedis(decode_responses=False))


def _cfg(**over) -> AugurConfig:
    return AugurConfig(**over)


class _ExplodingPM:
    """Any attribute access is a failure -- proves the disabled path touches
    the persistence manager zero times (invariant PR3, zero reads/writes)."""

    def __getattr__(self, name: str):
        raise AssertionError(f"disabled miner must not touch pm (accessed {name!r})")


def _strong_session_episodes() -> list[dict]:
    """A@0,300,600,900 (low) each catching a medium B at lag 100 -> n=k=4/session.
    Across 3 sessions: support 3, conf 1.0, lift ~2.5, IQR 0 -> promotes."""
    eps: list[dict] = []
    for a in (0, 300, 600, 900):
        eps.append({"k": "s:A", "s": "low", "t": float(a)})
    for b in (100, 400, 700, 1000):
        eps.append({"k": "s:B", "s": "medium", "t": float(b)})
    return eps


def _seed_strong_corpus(pm: PersistenceManager, sessions: int = 3) -> None:
    for s in range(sessions):
        for entry in _strong_session_episodes():
            pm.append_praesagium_episode(f"sess{s}", entry)


# -- gating ------------------------------------------------------------------


def test_disabled_returns_skipped_and_touches_nothing():
    result = run_praesagium_mining("s1", _ExplodingPM(), _cfg(praesagium_enabled=False))
    assert result == {"skipped": True, "reason": "disabled"}


def test_already_processed_short_circuits():
    pm = _pm()
    pm.mark_tuning_applied("s1", pass_name="praesagium")
    result = run_praesagium_mining("s1", pm, _cfg())
    assert result == {"skipped": True, "reason": "already_processed"}
    assert pm.load_praesagium_patterns() is None  # no blob written


def test_recently_mined_short_circuits_without_marking():
    pm = _pm()
    pm.save_praesagium_patterns(
        {
            "version": 1,
            "mined_at": time.time(),
            "hit_rate_watermark": 0.0,
            "patterns": {},
        }
    )
    result = run_praesagium_mining(
        "s1", pm, _cfg(praesagium_mine_min_interval_s=1800.0)
    )
    assert result == {"skipped": True, "reason": "recently_mined"}
    # skipped runs neither mark nor fold -- must NOT set the idempotency marker.
    assert pm.is_tuning_applied("s1", pass_name="praesagium") is False


def test_stale_blob_does_not_rate_limit():
    pm = _pm()
    pm.save_praesagium_patterns(
        {
            "version": 1,
            "mined_at": time.time() - 100_000,
            "hit_rate_watermark": 0.0,
            "patterns": {},
        }
    )
    _seed_strong_corpus(pm)
    result = run_praesagium_mining("s1", pm, _cfg())
    assert "skipped" not in result  # old mined_at -> proceeds


# -- happy path --------------------------------------------------------------


def test_happy_path_mines_saves_marks_and_reports():
    pm = _pm()
    _seed_strong_corpus(pm, sessions=3)
    result = run_praesagium_mining("s1", pm, _cfg())

    # report shape (Sec 4.7) -- all eight keys present.
    assert set(result) == {
        "active",
        "provisional",
        "retired",
        "promoted",
        "reactivated",
        "corpus_sessions",
        "resolutions_folded",
        "expired_open",
    }
    assert result["corpus_sessions"] == 3
    assert result["provisional"] == 1  # first mine -> probation, not active
    assert result["active"] == 0
    assert result["resolutions_folded"] == 0
    assert result["expired_open"] == 0

    blob = pm.load_praesagium_patterns()
    assert blob is not None
    pid = pattern_id("s:A", "s:B")
    assert pid in blob["patterns"]
    assert blob["patterns"][pid]["status"] == "provisional"
    assert pm.is_tuning_applied("s1", pass_name="praesagium") is True


def test_resolutions_folded_end_to_end():
    # Seed the resolved log directly (newest-first LPUSH); the miner must
    # reverse to time order, fold both into the mined pattern, and report the
    # count. prev=None -> watermark 0.0, so both (ts>0) fold.
    pm = _pm()
    _seed_strong_corpus(pm, sessions=3)
    pid = pattern_id("s:A", "s:B")
    pm._r.lpush(
        _LOG_KEY,
        json.dumps({"pattern_id": pid, "outcome": "fulfilled", "resolved_ts": 10.0}),
    )
    pm._r.lpush(
        _LOG_KEY,
        json.dumps({"pattern_id": pid, "outcome": "fulfilled", "resolved_ts": 20.0}),
    )
    result = run_praesagium_mining("s1", pm, _cfg())
    assert result["resolutions_folded"] == 2
    blob = pm.load_praesagium_patterns()
    assert blob["patterns"][pid]["hit_rate"] == 1.0
    assert blob["patterns"][pid]["resolutions"] == 2
    assert blob["hit_rate_watermark"] == 20.0


# -- missing hit_rate_watermark_ids is NOT the same as empty ----------------


def test_resolutions_folded_missing_ids_field_excludes_watermark_ts():
    # A blob written before the exact-fold rule existed (watermark set, ids
    # field never written) migrating through the miner for the first time:
    # the resolution that ESTABLISHED
    # the watermark was already folded by the old strict-`>` code, so the
    # report's resolutions_folded recount must agree with merge_blob's fold
    # and exclude it -- only the genuinely-newer resolution counts.
    pm = _pm()
    _seed_strong_corpus(pm, sessions=3)
    pid = pattern_id("s:A", "s:B")
    T = 10.0
    pm.save_praesagium_patterns(
        {
            "version": 1,
            "mined_at": time.time() - 100_000,
            "hit_rate_watermark": T,
            # NO "hit_rate_watermark_ids" key -- simulates a blob written
            # before the exact-fold rule existed.
            "patterns": {
                pid: {
                    "pattern_id": pid,
                    "antecedent": "s:A",
                    "consequent": "s:B",
                    "window_s": 125.0,
                    "support_sessions": 3,
                    "n": 3,
                    "k": 3,
                    "conf": 1.0,
                    "conf_lower": 0.44,
                    "lift": 2.0,
                    "lag_median_s": 60.0,
                    "lag_p90_s": 100.0,
                    "status": "active",
                    "hit_rate": None,
                    "resolutions": 0,
                    "created_at": 1.0,
                    "mined_at": 1.0,
                    "retired_at": None,
                    "retired_reason": None,
                    "repass_streak": 1,
                }
            },
        }
    )
    pm._r.lpush(
        _LOG_KEY,
        json.dumps(
            {
                "prediction_id": "legacy1",
                "pattern_id": pid,
                "outcome": "fulfilled",
                "resolved_ts": T,  # the watermark-setter -- already folded
            }
        ),
    )
    pm._r.lpush(
        _LOG_KEY,
        json.dumps(
            {
                "prediction_id": "fresh1",
                "pattern_id": pid,
                "outcome": "fulfilled",
                "resolved_ts": T + 10.0,  # genuinely new
            }
        ),
    )
    result = run_praesagium_mining("s1", pm, _cfg())
    assert result["resolutions_folded"] == 1  # legacy1 excluded, fresh1 counted
    blob = pm.load_praesagium_patterns()
    assert blob["patterns"][pid]["resolutions"] == 1


# -- present-but-non-list ids behaves like present-empty-list ---------------


def _seeded_pattern_blob(pid: str, T: float, watermark_ids) -> dict:
    return {
        "version": 1,
        "mined_at": time.time() - 100_000,
        "hit_rate_watermark": T,
        "hit_rate_watermark_ids": watermark_ids,
        "patterns": {
            pid: {
                "pattern_id": pid,
                "antecedent": "s:A",
                "consequent": "s:B",
                "window_s": 125.0,
                "support_sessions": 3,
                "n": 3,
                "k": 3,
                "conf": 1.0,
                "conf_lower": 0.44,
                "lift": 2.0,
                "lag_median_s": 60.0,
                "lag_p90_s": 100.0,
                "status": "active",
                "hit_rate": None,
                "resolutions": 0,
                "created_at": 1.0,
                "mined_at": 1.0,
                "retired_at": None,
                "retired_reason": None,
                "repass_streak": 1,
            }
        },
    }


@pytest.mark.parametrize(
    "garbage", ["not-a-list", 42, {"a": 1}], ids=["str", "int", "dict"]
)
def test_resolutions_folded_non_list_ids_field_still_folds_watermark_ts(garbage):
    # load_praesagium_patterns does no schema validation, so a corrupt/hand-
    # edited blob can carry hit_rate_watermark_ids as a present but non-list
    # garbage value. Both the merge_blob fold and this recount treat it as
    # "known, but nothing folded there yet" (watermark_ids_known=True, empty
    # set) -- functionally IDENTICAL to a present-but-genuinely-empty list.
    # Contrast test_resolutions_folded_missing_ids_field_excludes_watermark_ts
    # above (missing field entirely -> the same-ts resolution is excluded,
    # not folded).
    pm = _pm()
    _seed_strong_corpus(pm, sessions=3)
    pid = pattern_id("s:A", "s:B")
    T = 10.0
    pm.save_praesagium_patterns(_seeded_pattern_blob(pid, T, garbage))
    pm._r.lpush(
        _LOG_KEY,
        json.dumps(
            {
                "prediction_id": "q9",
                "pattern_id": pid,
                "outcome": "fulfilled",
                "resolved_ts": T,  # exactly at the watermark
            }
        ),
    )
    result = run_praesagium_mining("s1", pm, _cfg())
    assert result["resolutions_folded"] == 1  # folds, like present-empty-list
    blob = pm.load_praesagium_patterns()
    assert blob["patterns"][pid]["resolutions"] == 1


# -- present-but-non-empty ids dedups by id, not by exclusion ---------------


def test_resolutions_folded_present_ids_dedups_watermarked_id_only():
    # Contrast with test_resolutions_folded_missing_ids_field_excludes_
    # watermark_ts above: a blob written after the exact-fold rule existed,
    # WITH a non-empty hit_rate_watermark_ids, already carries an id folded
    # at the watermark.
    # The recount must dedup by id (drop the already-watermarked one, keep
    # the fresh same-ts one) rather than excluding the whole ts wholesale.
    pm = _pm()
    _seed_strong_corpus(pm, sessions=3)
    pid = pattern_id("s:A", "s:B")
    T = 10.0
    pm.save_praesagium_patterns(
        {
            "version": 1,
            "mined_at": time.time() - 100_000,
            "hit_rate_watermark": T,
            "hit_rate_watermark_ids": ["already1"],  # present AND non-empty
            "patterns": {
                pid: {
                    "pattern_id": pid,
                    "antecedent": "s:A",
                    "consequent": "s:B",
                    "window_s": 125.0,
                    "support_sessions": 3,
                    "n": 3,
                    "k": 3,
                    "conf": 1.0,
                    "conf_lower": 0.44,
                    "lift": 2.0,
                    "lag_median_s": 60.0,
                    "lag_p90_s": 100.0,
                    "status": "active",
                    "hit_rate": None,
                    "resolutions": 0,
                    "created_at": 1.0,
                    "mined_at": 1.0,
                    "retired_at": None,
                    "retired_reason": None,
                    "repass_streak": 1,
                }
            },
        }
    )
    pm._r.lpush(
        _LOG_KEY,
        json.dumps(
            {
                "prediction_id": "already1",
                "pattern_id": pid,
                "outcome": "fulfilled",
                "resolved_ts": T,  # already watermarked -- must NOT count
            }
        ),
    )
    pm._r.lpush(
        _LOG_KEY,
        json.dumps(
            {
                "prediction_id": "fresh2",
                "pattern_id": pid,
                "outcome": "fulfilled",
                "resolved_ts": T,  # fresh id at the SAME ts -- must count
            }
        ),
    )
    result = run_praesagium_mining("s1", pm, _cfg())
    assert result["resolutions_folded"] == 1  # already1 excluded, fresh2 counted
    blob = pm.load_praesagium_patterns()
    assert blob["patterns"][pid]["resolutions"] == 1


def test_short_sessions_skipped_from_corpus():
    pm = _pm()
    # one 2-episode session and one 1-episode session; only the former counts.
    pm.append_praesagium_episode("big", {"k": "a:A", "s": "low", "t": 0.0})
    pm.append_praesagium_episode("big", {"k": "a:B", "s": "medium", "t": 100.0})
    pm.append_praesagium_episode("tiny", {"k": "a:A", "s": "low", "t": 0.0})
    result = run_praesagium_mining("s1", pm, _cfg())
    assert result["corpus_sessions"] == 1


# -- expiry sweep (Sec 4.6-7) ------------------------------------------------


def test_expiry_sweep_resolves_past_deadline_predictions():
    pm = _pm()
    now = time.time()
    pm.save_praesagium_open_prediction(
        {
            "prediction_id": "p1",
            "pattern_id": "deadbeef0001",
            "created_ts": now - 1200,
            "deadline_ts": now - 1000,  # long past
            "session_id": "sX",
        }
    )
    result = run_praesagium_mining("s1", pm, _cfg())

    assert result["expired_open"] == 1
    resolved = pm.load_praesagium_resolved()
    assert len(resolved) == 1
    assert resolved[0]["prediction_id"] == "p1"
    assert resolved[0]["outcome"] == "expired"
    assert pm.load_praesagium_open_predictions() == []  # open hash drained


def test_expiry_sweep_leaves_future_deadlines_open():
    pm = _pm()
    now = time.time()
    pm.save_praesagium_open_prediction(
        {"prediction_id": "p2", "pattern_id": "d2", "deadline_ts": now + 10_000}
    )
    result = run_praesagium_mining("s1", pm, _cfg())
    assert result["expired_open"] == 0
    assert len(pm.load_praesagium_open_predictions()) == 1


# -- corrupt open records don't abort the expiry sweep -----------------------


def test_expiry_sweep_corrupt_open_records_do_not_raise_stale_one_expires():
    # A non-dict JSON value, a dict missing "prediction_id", and a dict with a
    # non-numeric "deadline_ts" sit alongside one legitimately stale open
    # prediction. The sweep must not raise, must skip the corrupt records
    # (uncounted, untouched), and must still expire the stale one.
    pm = _pm()
    key = "augur:praesagium:predictions:open"
    now = time.time()
    pm._r.hset(key, "bad-nondict", json.dumps("garbage"))
    pm._r.hset(key, "bad-missing-pid", json.dumps({"deadline_ts": now - 10_000}))
    pm._r.hset(
        key,
        "bad-deadline",
        json.dumps({"prediction_id": "bad-deadline", "deadline_ts": "not-a-number"}),
    )
    pm.save_praesagium_open_prediction(
        {
            "prediction_id": "stale1",
            "pattern_id": "deadbeef0002",
            "created_ts": now - 1200,
            "deadline_ts": now - 1000,  # long past
            "session_id": "sX",
        }
    )

    result = run_praesagium_mining("s1", pm, _cfg())  # must not raise

    assert result["expired_open"] == 1
    resolved = pm.load_praesagium_resolved()
    assert len(resolved) == 1
    assert resolved[0]["prediction_id"] == "stale1"
    assert resolved[0]["outcome"] == "expired"
    # The 3 corrupt entries remain in the open hash; the stale one was drained.
    assert pm._r.hlen(key) == 3


# -- trim-loss WARN (Sec 4.6-3) ----------------------------------------------


def test_warns_when_trim_ate_unfolded_resolutions(caplog):
    pm = _pm()
    pm.save_praesagium_patterns(
        {
            "version": 1,
            "mined_at": time.time() - 100_000,
            "hit_rate_watermark": 50.0,
            "patterns": {},
        }
    )
    # oldest retained resolved_ts (100) postdates the watermark (50) -> the
    # resolutions in (50, 100) were LTRIM'd away before they could fold.
    pm._r.lpush(
        _LOG_KEY,
        json.dumps({"pattern_id": "z", "outcome": "expired", "resolved_ts": 100.0}),
    )
    pm._r.lpush(
        _LOG_KEY,
        json.dumps({"pattern_id": "z", "outcome": "fulfilled", "resolved_ts": 200.0}),
    )
    with caplog.at_level(logging.WARNING, logger="praesagium.miner"):
        run_praesagium_mining("s1", pm, _cfg())
    assert any("trim loss" in r.message for r in caplog.records)


def test_no_trim_warn_when_oldest_within_watermark(caplog):
    pm = _pm()
    pm.save_praesagium_patterns(
        {
            "version": 1,
            "mined_at": time.time() - 100_000,
            "hit_rate_watermark": 500.0,
            "patterns": {},
        }
    )
    pm._r.lpush(
        _LOG_KEY,
        json.dumps({"pattern_id": "z", "outcome": "expired", "resolved_ts": 100.0}),
    )
    with caplog.at_level(logging.WARNING, logger="praesagium.miner"):
        run_praesagium_mining("s1", pm, _cfg())
    assert not any("trim loss" in r.message for r in caplog.records)
