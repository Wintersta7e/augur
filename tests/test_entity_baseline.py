"""Tests for EntityBaseline — EWMA math, scoring, and serialization.

These guard the statistical foundation that every anomaly detection decision
depends on. If the EWMA update or scoring is wrong, Augur's baselines drift
and the entire detection layer produces garbage.
"""

from __future__ import annotations

import json

import fakeredis
import pytest

from tabula.config import AugurConfig
from tabula.persistence import PersistenceManager, baseline_key, parse_baseline_key
from vigil.anomaly_detector import (
    EntityBaseline,
    evict_idle_baselines,
    load_persisted_baselines,
)


class TestEWMAUpdate:
    """Verify EWMA mean and variance tracking."""

    def test_first_observation_sets_mean_exactly(self) -> None:
        bl = EntityBaseline()
        bl.update(10.0, alpha=0.3)
        assert bl.ewma_mean == 10.0
        assert bl.ewma_var == 0.0
        assert bl.observation_count == 1

    def test_second_observation_applies_alpha(self) -> None:
        bl = EntityBaseline()
        bl.update(10.0, alpha=0.3)
        bl.update(20.0, alpha=0.3)
        # mean = 10 + 0.3 * (20 - 10) = 13.0
        assert bl.ewma_mean == pytest.approx(13.0)
        assert bl.observation_count == 2

    def test_variance_grows_with_spread(self) -> None:
        bl = EntityBaseline()
        bl.update(10.0, alpha=0.3)
        bl.update(20.0, alpha=0.3)
        # var = (1 - 0.3) * (0.0 + 0.3 * 10^2) = 0.7 * 30 = 21.0
        assert bl.ewma_var == pytest.approx(21.0)

    def test_stable_values_converge_to_low_variance(self) -> None:
        bl = EntityBaseline()
        for _ in range(50):
            bl.update(5.0, alpha=0.3)
        assert bl.ewma_mean == pytest.approx(5.0, abs=0.01)
        assert bl.ewma_var < 0.01

    def test_alpha_one_tracks_last_value(self) -> None:
        bl = EntityBaseline()
        bl.update(10.0, alpha=1.0)
        bl.update(99.0, alpha=1.0)
        assert bl.ewma_mean == pytest.approx(99.0)

    def test_alpha_zero_never_moves_after_second(self) -> None:
        bl = EntityBaseline()
        bl.update(10.0, alpha=0.0)
        bl.update(99.0, alpha=0.0)
        # mean stays at 10 after second observation: 10 + 0*(99-10) = 10
        assert bl.ewma_mean == pytest.approx(10.0)


class TestEWMAStd:
    """Verify standard deviation derivation."""

    def test_zero_variance_gives_zero_std(self) -> None:
        bl = EntityBaseline()
        bl.update(5.0, alpha=0.3)
        assert bl.ewma_std == 0.0

    def test_negative_variance_clamped_to_zero(self) -> None:
        bl = EntityBaseline()
        bl.ewma_var = -0.001  # should not happen, but guard against it
        assert bl.ewma_std == 0.0

    def test_positive_variance_gives_sqrt(self) -> None:
        bl = EntityBaseline()
        bl.ewma_var = 9.0
        assert bl.ewma_std == pytest.approx(3.0)


class TestScoring:
    """Verify deviation and HST scoring."""

    def test_deviation_zero_when_std_near_zero(self) -> None:
        bl = EntityBaseline()
        bl.update(5.0, alpha=0.3)  # single obs, var=0, std<0.01
        deviation, _ = bl.score(100.0)
        assert deviation == 0.0  # can't compute sigma with no spread

    def test_deviation_scales_with_distance(self) -> None:
        bl = EntityBaseline()
        bl.ewma_mean = 10.0
        bl.ewma_var = 4.0  # std = 2.0
        deviation, _ = bl.score(16.0)
        # |16 - 10| / 2 = 3.0 sigma
        assert deviation == pytest.approx(3.0)

    def test_deviation_symmetric(self) -> None:
        bl = EntityBaseline()
        bl.ewma_mean = 10.0
        bl.ewma_var = 4.0
        dev_high, _ = bl.score(16.0)
        dev_low, _ = bl.score(4.0)
        assert dev_high == pytest.approx(dev_low)

    def test_hst_score_returns_numeric(self) -> None:
        bl = EntityBaseline()
        # Train HST with some data so it has something to score against
        for v in [5.0, 5.1, 4.9, 5.2, 4.8]:
            bl.update(v, alpha=0.3)
        _, hst_score = bl.score(5.0)
        assert isinstance(hst_score, (int, float))  # River may return int 0
        assert 0.0 <= hst_score <= 1.0


class TestSerialization:
    """Verify state dict round-trip preserves baseline state."""

    def test_round_trip_preserves_state(self) -> None:
        bl = EntityBaseline()
        for v in [3.0, 7.0, 5.0, 12.0]:
            bl.update(v, alpha=0.3)

        state = bl.to_state_dict()
        restored = EntityBaseline.from_state_dict(state)

        assert restored.ewma_mean == pytest.approx(bl.ewma_mean)
        assert restored.ewma_var == pytest.approx(bl.ewma_var)
        assert restored.observation_count == bl.observation_count

    def test_from_empty_dict_gives_defaults(self) -> None:
        restored = EntityBaseline.from_state_dict({})
        assert restored.ewma_mean == 0.0
        assert restored.ewma_var == 0.0
        assert restored.observation_count == 0

    def test_hst_not_serialized(self) -> None:
        """HST model is not persisted — only EWMA state is.
        A restored baseline starts with a fresh HST. This is by design:
        HST rebuilds quickly from incoming data."""
        bl = EntityBaseline()
        for v in [1.0, 2.0, 3.0]:
            bl.update(v, alpha=0.3)
        state = bl.to_state_dict()
        assert "hst" not in state


class TestIdleEviction:
    """R4: bounded per-entity memory — idle in-memory models get reclaimed.

    Vigil allocates a River model per distinct (domain, entity) on first
    sighting. Without reclamation a churning entity namespace pins ~1.7 MB
    per model up to MAX_BASELINE_ENTITIES (~17 GB at the cap). Idle-TTL
    eviction drops models not seen within ``baseline_entity_idle_evict_s``
    while leaving active ones — and the low-cardinality steady state —
    untouched.
    """

    def _mk(self, n: int) -> dict[tuple[str, str], EntityBaseline]:
        return {("d", f"e{i}"): EntityBaseline() for i in range(n)}

    def test_idle_evicted_active_persists(self) -> None:
        baselines = self._mk(2)
        idle_key, active_key = ("d", "e0"), ("d", "e1")
        # active_key stamped recently; idle_key stamped long ago (idle=3700 > ttl).
        last_seen = {idle_key: 0.0, active_key: 3650.0}
        evicted = evict_idle_baselines(
            baselines, last_seen, now=3700.0, idle_ttl_s=3600.0
        )
        assert evicted == [idle_key]
        assert idle_key not in baselines
        assert idle_key not in last_seen  # bookkeeping map pruned in lockstep
        assert active_key in baselines  # active model preserved
        assert last_seen[active_key] == 3650.0

    def test_exactly_at_ttl_boundary_not_evicted(self) -> None:
        """Idle == TTL is retained; eviction is strictly-greater-than."""
        baselines = self._mk(1)
        key = ("d", "e0")
        last_seen = {key: 0.0}
        evicted = evict_idle_baselines(
            baselines, last_seen, now=3600.0, idle_ttl_s=3600.0
        )
        assert evicted == []
        assert key in baselines

    def test_low_cardinality_steady_state_untouched(self) -> None:
        """A handful of entities all seen recently → nobody is evicted
        (the normal case must behave identically to before)."""
        baselines = self._mk(3)
        last_seen = {k: 3699.0 for k in baselines}
        before = dict(baselines)
        evicted = evict_idle_baselines(
            baselines, last_seen, now=3700.0, idle_ttl_s=3600.0
        )
        assert evicted == []
        assert baselines == before

    def test_missing_last_seen_is_evicted(self) -> None:
        """A baseline with no last_seen stamp (e.g. Redis-restored before its
        first event) is treated as maximally idle and reclaimed, never leaked."""
        baselines = self._mk(1)
        key = ("d", "e0")
        evicted = evict_idle_baselines(baselines, {}, now=10_000.0, idle_ttl_s=3600.0)
        assert evicted == [key]
        assert key not in baselines

    def test_disabled_when_ttl_non_positive(self) -> None:
        """idle_ttl_s <= 0 disables eviction entirely (opt-out)."""
        baselines = self._mk(2)
        last_seen = {k: 0.0 for k in baselines}
        evicted = evict_idle_baselines(baselines, last_seen, now=1e9, idle_ttl_s=0.0)
        assert evicted == []
        assert len(baselines) == 2


class TestIdleEvictConfig:
    """The idle-evict TTL knob exists, has a generous default, and validates."""

    def test_default_is_generous(self) -> None:
        cfg = AugurConfig()
        assert cfg.baseline_entity_idle_evict_s == 3600.0

    def test_zero_is_allowed_as_disable_sentinel(self) -> None:
        # 0.0 must construct (it is the documented "never evict" opt-out).
        AugurConfig(baseline_entity_idle_evict_s=0.0)

    def test_negative_rejected(self) -> None:
        with pytest.raises(ValueError, match="baseline_entity_idle_evict_s"):
            AugurConfig(baseline_entity_idle_evict_s=-1.0)


class TestSeriesScopedBaselines:
    """A baseline is valid over ONE measurement series, not one entity.

    The typing sensor publishes two streams for the same ``user`` entity on
    different scales — ``sample`` in ms (~150-300) and ``pause`` in seconds
    (~7). Keying baselines on (domain, entity) alone folded both into one EWMA,
    so every switch between the streams read as a multi-sigma anomaly and the
    stored mean oscillated between the two scales.
    """

    def test_key_and_parse_round_trip(self) -> None:
        key = baseline_key("typing", "sample", "user")
        assert key == "augur:vigil:profile:typing:sample:user"
        assert parse_baseline_key(key) == ("typing", "sample", "user")

    def test_entity_containing_a_colon_survives_the_round_trip(self) -> None:
        key = baseline_key("activity_focus", "focus_change", "host:app")
        assert parse_baseline_key(key) == ("activity_focus", "focus_change", "host:app")

    def test_legacy_pre_series_key_is_not_parseable(self) -> None:
        assert parse_baseline_key("augur:vigil:profile:typing:user") is None

    def test_foreign_key_is_not_parseable(self) -> None:
        assert parse_baseline_key("augur:limen:advice_rate") is None

    def test_two_event_types_on_one_entity_get_separate_baselines(self) -> None:
        pm = PersistenceManager(fakeredis.FakeStrictRedis())
        pm.save_baseline("typing", "sample", "user", {"ewma_mean": 170.0}, ctx=None)
        pm.save_baseline("typing", "pause", "user", {"ewma_mean": 7.0}, ctx=None)
        assert pm.load_baseline("typing", "sample", "user") == {"ewma_mean": 170.0}
        assert pm.load_baseline("typing", "pause", "user") == {"ewma_mean": 7.0}


class TestUnitGuard:
    """A series has exactly one unit; a change is a sensor bug, not data."""

    def test_untrained_baseline_accepts_any_unit(self) -> None:
        assert EntityBaseline().accepts_unit("ms")

    def test_trained_baseline_rejects_a_different_unit(self) -> None:
        bl = EntityBaseline()
        bl.unit = "ms"
        assert bl.accepts_unit("ms")
        assert not bl.accepts_unit("seconds")

    def test_unit_survives_serialization(self) -> None:
        bl = EntityBaseline()
        bl.unit = "seconds"
        bl.update(7.0, alpha=0.3)
        assert EntityBaseline.from_state_dict(bl.to_state_dict()).unit == "seconds"

    def test_missing_unit_in_legacy_state_defaults_to_permissive(self) -> None:
        bl = EntityBaseline.from_state_dict({"ewma_mean": 1.0, "observation_count": 3})
        assert bl.unit == ""
        assert bl.accepts_unit("anything")


class TestBaselineRestore:
    """Restart must not silently discard every durable baseline.

    ``load_persisted_baselines`` split the Redis key positionally and read the
    segments one index short, so ``domain`` came back as the literal string
    ``"profile"``, every lookup missed, and the detector restarted from zero
    baselines on every process start — then overwrote each durable profile with
    a fresh one-observation EWMA.
    """

    def _redis_with(self, *entries: tuple[str, str, str, dict]):
        r = fakeredis.FakeStrictRedis()
        for domain, event_type, entity, state in entries:
            r.set(baseline_key(domain, event_type, entity), json.dumps(state))
        return r

    def test_restores_a_persisted_baseline(self) -> None:
        r = self._redis_with(
            ("typing", "sample", "user", {"ewma_mean": 170.0, "observation_count": 173})
        )
        out = load_persisted_baselines(PersistenceManager(r), r)
        assert ("typing", "sample", "user") in out
        assert out[("typing", "sample", "user")].observation_count == 173

    def test_restores_an_entity_whose_name_contains_a_space(self) -> None:
        r = self._redis_with(
            (
                "activity_focus",
                "focus_change",
                "text editor",
                {"ewma_mean": 5.0, "observation_count": 20},
            )
        )
        out = load_persisted_baselines(PersistenceManager(r), r)
        assert out[("activity_focus", "focus_change", "text editor")].ewma_mean == 5.0

    def test_skips_legacy_pre_series_keys(self) -> None:
        r = fakeredis.FakeStrictRedis()
        r.set("augur:vigil:profile:typing:user", json.dumps({"observation_count": 999}))
        assert load_persisted_baselines(PersistenceManager(r), r) == {}


class TestRehydrateOnCacheMiss:
    """Idle eviction must not destroy the durable profile it promises to keep.

    ``evict_idle_baselines`` drops the in-memory model and leaves Redis alone,
    but the detector used to build a fresh ``EntityBaseline`` on the cache miss
    and persist it unconditionally — so the first re-sighted event overwrote a
    trained profile with a one-observation EWMA. Any series whose inter-event
    gap exceeded ``baseline_entity_idle_evict_s`` could therefore never
    accumulate observations, however long the process ran. The activity domains
    are exactly that shape.
    """

    def test_evicted_series_rehydrates_instead_of_resetting(self) -> None:
        pm = PersistenceManager(fakeredis.FakeStrictRedis())
        key = ("activity_focus", "focus_change", "browser")
        trained = EntityBaseline()
        for v in (
            2.9,
            3.1,
            2.7,
            3.4,
            2.8,
            3.0,
            3.3,
            2.6,
            3.2,
            2.95,
            3.05,
            2.85,
            3.15,
            2.75,
            3.25,
            2.9,
            3.0,
            3.1,
        ):
            trained.update(v, 0.3)
        trained.unit = "log1p_seconds"
        pm.save_baseline(*key, trained.to_state_dict(), ctx=None)

        baselines = {key: trained}
        assert evict_idle_baselines(
            baselines, {key: 0.0}, now=7200.0, idle_ttl_s=3600.0
        ) == [key]
        assert key not in baselines

        # What on_event does on a cache miss: rehydrate, not reset.
        restored = pm.load_baseline(*key)
        assert restored is not None
        revived = EntityBaseline.from_state_dict(restored)
        assert revived.observation_count == 18
        assert revived.unit == "log1p_seconds"

        # And the next event must extend that history, not replace it.
        revived.update(3.0, 0.3)
        pm.save_baseline(*key, revived.to_state_dict(), ctx=None)
        assert pm.load_baseline(*key)["observation_count"] == 19

    def test_a_genuinely_new_series_still_starts_empty(self) -> None:
        pm = PersistenceManager(fakeredis.FakeStrictRedis())
        assert pm.load_baseline("activity_focus", "focus_change", "unseen") is None
        assert EntityBaseline().observation_count == 0
