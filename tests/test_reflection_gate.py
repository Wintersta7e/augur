"""Unit tests for analyze_gate — the SIXTH reflection pass (spec §9 + §10).

The offline gate-tuning + MRT/IPW readout. All tests use a real
PersistenceManager over fakeredis so the de-dupe, idempotency marker,
and IPW-from-persisted-records-alone behaviours are exercised end to end.
Deterministic fabricated records + explicit expected values throughout.
"""

from __future__ import annotations

import time

import fakeredis
import pytest

from tabula.config import AugurConfig
from tabula.persistence import PersistenceManager
from limen.gate import Gate, build_signature
from disciplina.reflection_engine import (
    GATE_CHRONIC_MIN_PRESENCE,
    GATE_DISMISSAL_MIN,
    analyze_gate,
    reconstruct_state_key,
)
from tests.conftest import SINGLE_MEDIUM_TYPING

CONFIG = AugurConfig()


def _pm() -> PersistenceManager:
    return PersistenceManager(fakeredis.FakeStrictRedis(decode_responses=True))


def _advice(
    *,
    domain: str = "chess",
    entity: str = "board",
    severity: str = "medium",
    explicit: str = "no_response",
    behavioral: float = 0.5,
    behavioral_finalized: bool = True,
    unmeasurable: bool = False,
    outcome_metric_version: int = 2,
    correlation_found: bool = False,
    involved_domains: list[str] | None = None,
    decision_id: str | None = None,
    probe: bool = False,
    mrt_eligible: bool = False,
    p_fire: float | None = None,
) -> dict:
    return {
        "advice_id": "adv",
        "domain": domain,
        "entity": entity,
        "severity": severity,
        "explicit_rating": explicit,
        "behavioral_score": behavioral,
        "behavioral_finalized": behavioral_finalized,
        "unmeasurable": unmeasurable,
        "outcome_metric_version": outcome_metric_version,
        "baseline_mean_at_time": 5.0,
        "timestamp": "2026-06-06T12:00:00+00:00",
        "correlation_found": correlation_found,
        "involved_domains": involved_domains or [],
        "decision_id": decision_id,
        "probe": probe,
        "mrt_eligible": mrt_eligible,
        "p_fire": p_fire,
    }


def _gate_decision(
    *,
    decision_id: str,
    state_key: str = "single:chess:board",
    domain: str = "chess",
    entity: str = "board",
    severity: str = "medium",
    behavioral: float = 0.5,
    behavioral_finalized: bool = True,
    unmeasurable: bool = False,
    outcome_metric_version: int = 2,
    withheld_rating_p: float | None = None,
    explicit: str = "no_response",
    mrt_eligible: bool = True,
    p_withhold: float | None = 0.9,
    reason: str = "low_credibility_class",
) -> dict:
    return {
        "decision_id": decision_id,
        "state_key": state_key,
        "domain": domain,
        "entity": entity,
        "severity": severity,
        "mrt_eligible": mrt_eligible,
        "p_withhold": p_withhold,
        "baseline_mean": 5.0,
        "behavioral_score": behavioral,
        "behavioral_finalized": behavioral_finalized,
        "unmeasurable": unmeasurable,
        "outcome_metric_version": outcome_metric_version,
        "withheld_rating_p": withheld_rating_p,
        "explicit_rating": explicit,
        "reason": reason,
        "timestamp": "2026-06-06T12:00:00+00:00",
    }


def _feedback(
    session_id: str,
    *,
    advice_events: list[dict] | None = None,
    gate_decision_events: list[dict] | None = None,
) -> dict:
    return {
        "session_id": session_id,
        "advice_events": advice_events or [],
        "gate_decision_events": gate_decision_events or [],
        "session_summary": {"total_advice": len(advice_events or [])},
    }


# ── state_key reconstruction ─────────────────────────────────────────────────


class TestReconstructStateKey:
    def test_single_row(self) -> None:
        row = {"domain": "typing", "entity": "user", "correlation_found": False}
        assert reconstruct_state_key(row) == "single:typing:user"

    def test_correlation_row_sorts_involved_domains(self) -> None:
        row = {
            "correlation_found": True,
            "involved_domains": ["typing", "chess"],
            "domain": "chess",
            "entity": "board",
        }
        assert reconstruct_state_key(row) == "correlation:chess,typing"

    def test_gate_decision_row_uses_persisted_state_key(self) -> None:
        # A gate_decision_event already carries an authoritative state_key.
        row = {"state_key": "single:chess:board"}
        assert reconstruct_state_key(row) == "single:chess:board"


# ── de-dupe by session_id ────────────────────────────────────────────────────


class TestDeDupe:
    def test_dedupes_repeated_session_id_before_aggregating(self) -> None:
        pm = _pm()
        # The feedback index LPUSHes the same session_id on every intermediate
        # save (persistence.py). Simulate by saving the same session twice.
        fb = _feedback(
            "sess-A",
            advice_events=[
                _advice(explicit="n", domain="chess", entity="board"),
            ],
        )
        pm.save_feedback("sess-A", fb)
        pm.save_feedback("sess-A", fb)  # second save → duplicate index entry
        # Sanity: the raw index really does contain the duplicate.
        raw = pm.get_all_feedback(limit=50)
        assert sum(1 for r in raw if r["session_id"] == "sess-A") == 2

        result = analyze_gate("sess-A", pm, CONFIG)
        # Exactly one session counted, not two.
        assert result["sessions_analyzed"] == 1


# ── idempotency (gate marker independent of correlation marker) ──────────────


class TestIdempotency:
    def test_double_fire_does_not_double_apply(self) -> None:
        pm = _pm()
        # Chronic + dismissed channel → should be added to self_tolerance.
        events = [
            _advice(explicit="n", domain="chess", entity="board")
            for _ in range(GATE_CHRONIC_MIN_PRESENCE)
        ]
        for i in range(GATE_DISMISSAL_MIN):
            events[i]["explicit_rating"] = "n"
        pm.save_feedback("sess-1", _feedback("sess-1", advice_events=events))

        first = analyze_gate("sess-1", pm, CONFIG)
        assert first.get("skipped") is not True
        assert pm.is_tuning_applied("sess-1", pass_name="gate") is True

        second = analyze_gate("sess-1", pm, CONFIG)
        assert second["skipped"] is True

    def test_a_new_tuning_scope_reopens_the_pass_within_one_session(self) -> None:
        """Periodic in-session reflection must be able to tune more than once.

        The marker is keyed by scope, not by session. With a session-keyed
        marker the first cadence cycle consumed the session's only marker and
        every later cycle short-circuited, so a long sitting still tuned once —
        the exact cadence problem the periodic pass exists to fix.
        """
        pm = _pm()
        events = [
            _advice(explicit="n", domain="chess", entity="board")
            for _ in range(GATE_CHRONIC_MIN_PRESENCE)
        ]
        for i in range(GATE_DISMISSAL_MIN):
            events[i]["explicit_rating"] = "n"
        pm.save_feedback("sess-1", _feedback("sess-1", advice_events=events))

        first = analyze_gate("sess-1", pm, CONFIG, tuning_scope="sess-1#c1")
        assert first.get("skipped") is not True
        # Same scope again → still idempotent.
        assert analyze_gate("sess-1", pm, CONFIG, tuning_scope="sess-1#c1")["skipped"]
        # Next cycle → runs again.
        second = analyze_gate("sess-1", pm, CONFIG, tuning_scope="sess-1#c2")
        assert second.get("skipped") is not True

    def test_scope_defaults_to_the_session(self) -> None:
        """Omitting the scope must reproduce the pre-cadence behaviour exactly."""
        pm = _pm()
        pm.mark_tuning_applied("sess-D", pass_name="gate")
        assert analyze_gate("sess-D", pm, CONFIG)["skipped"] is True

    def test_a_scoped_run_does_not_consume_the_session_marker(self) -> None:
        """A cadence cycle must not block the end-of-session reflection."""
        pm = _pm()
        events = [
            _advice(explicit="n", domain="chess", entity="board")
            for _ in range(GATE_CHRONIC_MIN_PRESENCE)
        ]
        for i in range(GATE_DISMISSAL_MIN):
            events[i]["explicit_rating"] = "n"
        pm.save_feedback("sess-1", _feedback("sess-1", advice_events=events))

        analyze_gate("sess-1", pm, CONFIG, tuning_scope="sess-1#c1")
        assert pm.is_tuning_applied("sess-1", pass_name="gate") is False
        assert analyze_gate("sess-1", pm, CONFIG).get("skipped") is not True

    def test_gate_marker_independent_of_correlation_marker(self) -> None:
        pm = _pm()
        pm.mark_tuning_applied("sess-X", pass_name="correlation")
        pm.save_feedback("sess-X", _feedback("sess-X", advice_events=[_advice()]))
        result = analyze_gate("sess-X", pm, CONFIG)
        # The correlation marker must NOT cause the gate pass to skip.
        assert result.get("skipped") is not True


# ── self_tolerance: chronic AND explicit dismissal, never behavioral alone ───


class TestSelfTolerance:
    def test_chronic_plus_dismissal_adds_to_tolerance(self) -> None:
        pm = _pm()
        events = [
            _advice(explicit="n", domain="chess", entity="board")
            for _ in range(max(GATE_CHRONIC_MIN_PRESENCE, GATE_DISMISSAL_MIN))
        ]
        pm.save_feedback("sess-1", _feedback("sess-1", advice_events=events))
        analyze_gate("sess-1", pm, CONFIG)
        assert "single:chess:board" in pm.load_self_tolerance()

    def test_behavioral_alone_never_adds_to_tolerance(self) -> None:
        pm = _pm()
        # Chronic presence, low behavioral score, but NO explicit dismissal.
        events = [
            _advice(explicit="no_response", behavioral=0.05, domain="chess", entity="x")
            for _ in range(max(GATE_CHRONIC_MIN_PRESENCE, GATE_DISMISSAL_MIN) + 2)
        ]
        pm.save_feedback("sess-1", _feedback("sess-1", advice_events=events))
        analyze_gate("sess-1", pm, CONFIG)
        assert "single:chess:x" not in pm.load_self_tolerance()

    def test_dismissal_without_chronic_does_not_add(self) -> None:
        pm = _pm()
        # A single dismissed event (not chronic) must not be tolerated.
        events = [_advice(explicit="n", domain="chess", entity="rare")]
        pm.save_feedback("sess-1", _feedback("sess-1", advice_events=events))
        analyze_gate("sess-1", pm, CONFIG)
        assert "single:chess:rare" not in pm.load_self_tolerance()

    def test_exempt_shaped_excluded(self) -> None:
        pm = _pm()
        # correlation_found AND severity==high → exempt-shaped, must be excluded
        # even when chronic + dismissed.
        events = [
            _advice(
                explicit="n",
                severity="high",
                correlation_found=True,
                involved_domains=["chess", "typing"],
            )
            for _ in range(max(GATE_CHRONIC_MIN_PRESENCE, GATE_DISMISSAL_MIN) + 1)
        ]
        pm.save_feedback("sess-1", _feedback("sess-1", advice_events=events))
        analyze_gate("sess-1", pm, CONFIG)
        assert pm.load_self_tolerance() == set()


# ── behavioral audit: genuine y/n only, min-samples, behavioral_finalized ────


class TestBehavioralAudit:
    def test_audit_excludes_no_response(self) -> None:
        pm = _pm()
        # 3 genuine y/n + many no_response. Min-samples (default 5) NOT met by
        # genuine responses alone → audit must report insufficient.
        events = [
            _advice(explicit="y", behavioral=0.9),
            _advice(explicit="n", behavioral=0.1),
            _advice(explicit="y", behavioral=0.8),
        ] + [_advice(explicit="no_response", behavioral=0.5) for _ in range(20)]
        pm.save_feedback("sess-1", _feedback("sess-1", advice_events=events))
        result = analyze_gate("sess-1", pm, CONFIG)
        audit = result["behavioral_audit"]
        # Only the 3 genuine responses are counted, not the 20 no_response rows.
        assert audit["genuine_samples"] == 3
        assert audit["sufficient"] is False

    def test_audit_honors_min_samples_and_reports_correlation(self) -> None:
        pm = _pm()
        # 6 genuine y/n rows where behavioral tracks explicit → high positive corr.
        events = [
            _advice(explicit="y", behavioral=0.9, behavioral_finalized=True),
            _advice(explicit="y", behavioral=0.85, behavioral_finalized=True),
            _advice(explicit="y", behavioral=0.95, behavioral_finalized=True),
            _advice(explicit="n", behavioral=0.1, behavioral_finalized=True),
            _advice(explicit="n", behavioral=0.05, behavioral_finalized=True),
            _advice(explicit="n", behavioral=0.15, behavioral_finalized=True),
        ]
        pm.save_feedback("sess-1", _feedback("sess-1", advice_events=events))
        result = analyze_gate("sess-1", pm, CONFIG)
        audit = result["behavioral_audit"]
        assert audit["genuine_samples"] == 6
        assert audit["sufficient"] is True
        assert audit["correlation"] > 0.9

    def test_audit_branches_on_behavioral_finalized(self) -> None:
        pm = _pm()
        # Unfinalized rows (default 0.0 behavioral) must be excluded so an
        # unfinalized 0.0 is not mistaken for a genuine low score.
        events = [
            _advice(explicit="y", behavioral=0.0, behavioral_finalized=False)
            for _ in range(10)
        ]
        pm.save_feedback("sess-1", _feedback("sess-1", advice_events=events))
        result = analyze_gate("sess-1", pm, CONFIG)
        audit = result["behavioral_audit"]
        assert audit["genuine_samples"] == 0
        assert audit["sufficient"] is False

    def test_audit_reports_genuine_response_rate(self) -> None:
        pm = _pm()
        events = [_advice(explicit="y", behavioral=0.9) for _ in range(2)] + [
            _advice(explicit="no_response", behavioral=0.5) for _ in range(8)
        ]
        pm.save_feedback("sess-1", _feedback("sess-1", advice_events=events))
        result = analyze_gate("sess-1", pm, CONFIG)
        audit = result["behavioral_audit"]
        assert audit["genuine_response_rate"] == pytest.approx(0.2)


# ── advice-rate operating point (the field the online gate actually reads) ────


class TestAdviceRateTuning:
    def test_tunes_dismissal_ewma_not_the_online_volume_field(self) -> None:
        # The offline pass measures how often advice was REJECTED. That is a
        # different quantity from the online gate's delivery-VOLUME impulse
        # EWMA (`rate_ewma`, limen/gate.py), and they must not share a field:
        # when they did, a system that had merely delivered a lot of advice
        # reported itself as one whose advice was dismissed.
        pm = _pm()
        events = [
            _advice(explicit="n", behavioral=0.1),
            _advice(explicit="n", behavioral=0.1),
            _advice(explicit="y", behavioral=0.9),
            _advice(explicit="y", behavioral=0.9),
        ]
        pm.save_feedback("sess-1", _feedback("sess-1", advice_events=events))
        result = analyze_gate("sess-1", pm, CONFIG)

        record = pm.load_advice_rate()
        assert record["dismissal_ewma"] == pytest.approx(0.5)
        # The dead field must NOT be injected.
        assert "operating_point" not in record
        # The report mirrors the persisted record (same field).
        assert "dismissal_ewma" in result["advice_rate"]
        assert "operating_point" not in result["advice_rate"]

    def test_offline_pass_never_touches_the_online_volume_field(self) -> None:
        # Regression pin for the field collision: reflection must leave
        # `rate_ewma` exactly as the online writer left it.
        pm = _pm()
        pm.save_advice_rate({"rate_ewma": 0.77, "last_ts": 1000.0})
        events = [_advice(explicit="n", behavioral=0.1) for _ in range(4)]
        pm.save_feedback("sess-1", _feedback("sess-1", advice_events=events))
        analyze_gate("sess-1", pm, CONFIG)

        record = pm.load_advice_rate()
        assert record["rate_ewma"] == pytest.approx(0.77)
        assert record["dismissal_ewma"] == pytest.approx(1.0)

    def test_reconciles_with_existing_dismissal_ewma(self) -> None:
        # When a prior dismissal rate exists, the offline EWMA must blend from
        # THAT value (numeric last_ts preserved as a float), not fall back to
        # the observed rate as if no prior existed.
        pm = _pm()
        pm.save_advice_rate({"dismissal_ewma": 0.2, "last_ts": 1000.0})
        # 100% dismissal → observed_rate 1.0; EWMA nudges 0.2 toward 1.0.
        events = [_advice(explicit="n", behavioral=0.1) for _ in range(4)]
        pm.save_feedback("sess-1", _feedback("sess-1", advice_events=events))
        analyze_gate("sess-1", pm, CONFIG)

        record = pm.load_advice_rate()
        # EWMA(0.2, observed=1.0, alpha=0.1) = 0.2 + 0.1*(1.0-0.2) = 0.28.
        assert record["dismissal_ewma"] == pytest.approx(0.28)
        # last_ts must stay a float (the online writer overwrites it numerically);
        # the offline pass must never clobber it with a session-id string.
        assert isinstance(record["last_ts"], float)


# ── MRT / IPW from persisted records alone ───────────────────────────────────


class TestMRT_IPW:
    def test_ipw_computed_from_persisted_records_alone(self) -> None:
        pm = _pm()
        # Fired (probe) arm: emission with mrt_eligible + p_fire, joined to an
        # advice_event (the behavioral outcome) by decision_id.
        pm.save_emission(
            {
                "ts": 1.0,
                "decision_id": "fire-1",
                "state_key": "single:chess:board",
                "severity": "medium",
                "tier": 2,
                "probe": True,
                "audit_only": False,
                "withheld_reason": None,
                "mrt_eligible": True,
                "p_fire": 0.1,
            }
        )
        # Withheld arm: silence with mrt_eligible + p_withhold, joined to a
        # gate_decision_event by decision_id.
        pm.save_silence_record(
            {
                "ts": 2.0,
                "decision_id": "wh-1",
                "state_key": "single:chess:board",
                "domain": "chess",
                "entity": "board",
                "severity": "medium",
                "arm": "bet_hedge",
                "reason": "low_credibility_class",
                "metrics": {},
                "mrt_eligible": True,
                "p_withhold": 0.9,
            }
        )
        pm.save_feedback(
            "sess-1",
            _feedback(
                "sess-1",
                advice_events=[
                    _advice(
                        decision_id="fire-1",
                        probe=True,
                        mrt_eligible=True,
                        p_fire=0.1,
                        behavioral=0.8,
                        behavioral_finalized=True,
                    )
                ],
                gate_decision_events=[
                    _gate_decision(
                        decision_id="wh-1",
                        behavioral=0.2,
                        behavioral_finalized=True,
                        p_withhold=0.9,
                    )
                ],
            ),
        )
        result = analyze_gate("sess-1", pm, CONFIG)
        mrt = result["mrt"]
        # IPW estimates: fired mean = 0.8/0.1 weighted = 0.8; withheld = 0.2.
        # The excursion estimate is fired_mean - withheld_mean (directional).
        assert mrt["fired_n"] == 1
        assert mrt["withheld_n"] == 1
        assert mrt["excursion_estimate"] == pytest.approx(0.6, abs=1e-9)
        assert mrt["directional"] is True

    def test_excludes_mrt_unobservable_silence_and_reports_rate(self) -> None:
        pm = _pm()
        # Two mrt_eligible silences; only one has a matching gate_decision_event.
        for did in ("wh-obs", "wh-missing"):
            pm.save_silence_record(
                {
                    "ts": 1.0,
                    "decision_id": did,
                    "state_key": "single:chess:board",
                    "domain": "chess",
                    "entity": "board",
                    "severity": "medium",
                    "arm": "bet_hedge",
                    "reason": "r",
                    "metrics": {},
                    "mrt_eligible": True,
                    "p_withhold": 0.9,
                }
            )
        pm.save_feedback(
            "sess-1",
            _feedback(
                "sess-1",
                gate_decision_events=[
                    _gate_decision(decision_id="wh-obs", p_withhold=0.9)
                ],
            ),
        )
        result = analyze_gate("sess-1", pm, CONFIG)
        mrt = result["mrt"]
        # Only the observable withheld decision is in the estimand.
        assert mrt["withheld_n"] == 1
        # One of two mrt_eligible silences was unobservable.
        assert mrt["unobservable_rate"] == pytest.approx(0.5)

    def test_deterministic_silence_excluded_from_estimand(self) -> None:
        pm = _pm()
        # A non-mrt_eligible (deterministic Arms 1-6) silence must not count as
        # an MRT-unobservable row even if it has no gate_decision_event.
        pm.save_silence_record(
            {
                "ts": 1.0,
                "decision_id": "det-1",
                "state_key": "single:chess:board",
                "domain": "chess",
                "entity": "board",
                "severity": "medium",
                "arm": "habituation",
                "reason": "habituated",
                "metrics": {},
                "mrt_eligible": False,
                "p_withhold": None,
            }
        )
        pm.save_feedback("sess-1", _feedback("sess-1"))
        result = analyze_gate("sess-1", pm, CONFIG)
        mrt = result["mrt"]
        assert mrt["withheld_n"] == 0
        # No mrt_eligible silences at all → unobservable_rate is 0.
        assert mrt["unobservable_rate"] == 0.0

    def test_no_mrt_data_is_low_power(self) -> None:
        pm = _pm()
        pm.save_feedback("sess-1", _feedback("sess-1", advice_events=[_advice()]))
        result = analyze_gate("sess-1", pm, CONFIG)
        mrt = result["mrt"]
        assert mrt["fired_n"] == 0
        assert mrt["withheld_n"] == 0
        assert mrt["excursion_estimate"] is None
        assert mrt["directional"] is True


# ── wiring: analyze_gate is the SIXTH analysis in run_reflection ─────────────


class TestRunReflectionWiring:
    @pytest.mark.asyncio
    async def test_gate_is_sixth_analysis(self) -> None:
        from unittest.mock import AsyncMock

        from disciplina.reflection_engine import run_reflection

        pm = _pm()
        feedback = _feedback(
            "sess-wire",
            advice_events=[_advice(explicit="y", behavioral=0.9)],
        )
        pm.save_feedback("sess-wire", feedback)

        report = await run_reflection(
            "sess-wire",
            feedback,
            pm,
            fakeredis.FakeStrictRedis(decode_responses=True),
            AsyncMock(),
            AsyncMock(),
            CONFIG,
        )
        assert "gate" in report["analyses"]
        assert report["analyses"]["gate"]["analysis"] == "gate"
        # Seven analyses total now (added the memory pass).
        assert len(report["analyses"]) == 7

    @pytest.mark.asyncio
    async def test_gate_pass_independent_of_correlation_marker(self) -> None:
        from unittest.mock import AsyncMock

        from disciplina.reflection_engine import run_reflection

        pm = _pm()
        # Correlation already applied → correlation pass skips, gate must NOT.
        pm.mark_tuning_applied("sess-wire2", pass_name="correlation")
        feedback = _feedback(
            "sess-wire2",
            advice_events=[_advice(explicit="y", behavioral=0.9)],
        )
        pm.save_feedback("sess-wire2", feedback)

        report = await run_reflection(
            "sess-wire2",
            feedback,
            pm,
            fakeredis.FakeStrictRedis(decode_responses=True),
            AsyncMock(),
            AsyncMock(),
            CONFIG,
        )
        assert report["analyses"]["correlation_tuning"].get("skipped") is True
        assert report["analyses"]["gate"].get("skipped") is not True


# ── timestamp fields must hold timestamps (round-trip into the gate) ─────────


class TestGateTuningTimestamps:
    """The offline pass writes decay clocks the gate later parses as floats.

    Regression: the pass stored ``session_id`` in ``last_fb_ts``/``last_ts``,
    so the gate's ``float()`` raised ValueError and Arm 6 failed open — the
    credibility arm silently stopped suppressing instead of erroring loudly.
    """

    def test_credibility_last_fb_ts_is_a_timestamp(self) -> None:
        pm = _pm()
        events = [
            _advice(domain="typing", entity="user", severity="medium", explicit="y")
        ]
        pm.save_feedback("sess-1", _feedback("sess-1", advice_events=events))

        before = time.time()
        analyze_gate("sess-1", pm, CONFIG)
        after = time.time()

        entry = pm.load_credibility("typing:medium")
        assert entry, "expected a credibility entry for typing:medium"
        last_fb_ts = entry.get("last_fb_ts")
        assert isinstance(last_fb_ts, (int, float)) and not isinstance(
            last_fb_ts, bool
        ), f"last_fb_ts must be numeric, got {last_fb_ts!r}"
        assert before <= float(last_fb_ts) <= after

    def test_gate_reads_the_credibility_the_pass_wrote(self) -> None:
        """Round-trip: whatever the pass persists, Arm 6 must consume.

        Every other Arm-6 test hand-seeds a numeric ``last_fb_ts``; none reads
        back what the offline pass actually wrote, which is how the ValueError
        survived. ``gate_reservoir_enabled=False`` isolates Arm 6 (an unseeded
        channel would otherwise be held by Arm 5 first).
        """
        pm = _pm()
        events = [
            _advice(domain="typing", entity="user", severity="medium", explicit="n")
        ]
        pm.save_feedback("sess-1", _feedback("sess-1", advice_events=events))
        analyze_gate("sess-1", pm, CONFIG)

        cfg = AugurConfig(gate_reservoir_enabled=False)
        gate = Gate()
        # Must not raise: a ValueError here escapes to the advisor, which fails
        # open to FIRE (inv. C) — suppression silently stops instead of erroring.
        gate.evaluate(build_signature(SINGLE_MEDIUM_TYPING), pm, cfg, now=time.time())
