"""1B calibration-era withheld-rating selection + prompt helper (spec §5.3)."""

import pytest

from tabula.config import AugurConfig
from perception import feedback_collector as fc
from perception.feedback_collector import (
    PendingAdvice,
    PendingGateDecision,
    _should_select_withheld_rating,
    maybe_prompt_withheld_rating,
)


def _cfg(**kw):
    return AugurConfig(**kw)


def _gd(**kw):
    base = dict(
        decision_id="d1",
        state_key="single:typing:keyboard",
        domain="typing",
        entity="keyboard",
        severity="medium",
        baseline_mean=0.9,
        timestamp="t",
        mrt_eligible=True,
        p_withhold=0.9,
        reason="habituation",
        session_id="s1",
    )
    base.update(kw)
    return PendingGateDecision(**base)


# -- selection (deterministic, no I/O) ----------------------------------------


def test_no_select_when_master_off():
    assert not _should_select_withheld_rating(
        _cfg(gate_mrt_withheld_rating=False),
        mrt_eligible=True,
        decision_id="d1",
        sessions_so_far=0,
    )


def test_no_select_when_not_mrt_eligible():
    assert not _should_select_withheld_rating(
        _cfg(gate_mrt_withheld_rating=True, gate_mrt_withheld_rating_rate=0.5),
        mrt_eligible=False,
        decision_id="d1",
        sessions_so_far=0,
    )


def test_sunset_after_max_sessions():
    assert not _should_select_withheld_rating(
        _cfg(
            gate_mrt_withheld_rating=True,
            gate_mrt_withheld_rating_rate=0.5,
            gate_mrt_withheld_rating_max_sessions=15,
        ),
        mrt_eligible=True,
        decision_id="d1",
        sessions_so_far=15,
    )


def test_rate_zero_never_selects():
    assert not _should_select_withheld_rating(
        _cfg(gate_mrt_withheld_rating=True, gate_mrt_withheld_rating_rate=0.0),
        mrt_eligible=True,
        decision_id="d1",
        sessions_so_far=0,
    )


def test_selection_boundary_for_low_fraction_id():
    # 'zzz' hashes to frac=0.0935 (precomputed sha256[:8]/0xFFFFFFFF — asserted as
    # a literal, NOT recomputed, so a wrong slice/divisor would shift the boundary
    # and fail here). rate just above the frac → selected; just below → not.
    above = _cfg(gate_mrt_withheld_rating=True, gate_mrt_withheld_rating_rate=0.12)
    below = _cfg(gate_mrt_withheld_rating=True, gate_mrt_withheld_rating_rate=0.05)
    assert (
        _should_select_withheld_rating(
            above, mrt_eligible=True, decision_id="zzz", sessions_so_far=0
        )
        is True
    )
    assert (
        _should_select_withheld_rating(
            below, mrt_eligible=True, decision_id="zzz", sessions_so_far=0
        )
        is False
    )


def test_high_fraction_ids_never_selected_within_rate_cap():
    # These ids hash to frac > 0.5 (d1=0.5442, dX=0.6874, d2=0.9044), and the rate
    # is capped at 0.5, so they can never be selected — the cap is a hard ceiling.
    cfg = _cfg(gate_mrt_withheld_rating=True, gate_mrt_withheld_rating_rate=0.5)
    for did in ["d1", "dX", "d2"]:
        assert (
            _should_select_withheld_rating(
                cfg, mrt_eligible=True, decision_id=did, sessions_so_far=0
            )
            is False
        )


def test_deterministic_by_decision_id():
    cfg = _cfg(gate_mrt_withheld_rating=True, gate_mrt_withheld_rating_rate=0.5)
    a = _should_select_withheld_rating(
        cfg, mrt_eligible=True, decision_id="dX", sessions_so_far=0
    )
    b = _should_select_withheld_rating(
        cfg, mrt_eligible=True, decision_id="dX", sessions_so_far=0
    )
    assert a == b


# -- prompt helper (stub stdin) -----------------------------------------------


class _FakePM:
    def __init__(self):
        self.marked = []

    def mark_mrt_rating_session(self, sid):
        self.marked.append(sid)


@pytest.mark.asyncio
async def test_helper_noops_when_not_selected():
    pm = _FakePM()
    p = _gd()  # selected_for_rating defaults False
    p.finalized = True
    assert await maybe_prompt_withheld_rating(p, _cfg(), pm) is False
    assert p.withheld_rating_p is None and pm.marked == []


@pytest.mark.asyncio
async def test_helper_noops_when_not_finalized():
    pm = _FakePM()
    p = _gd()
    p.selected_for_rating = True  # but window not finalized
    assert await maybe_prompt_withheld_rating(p, _cfg(), pm) is False
    assert p.withheld_rating_p is None


@pytest.mark.asyncio
async def test_helper_noops_for_fired_arm():
    pm = _FakePM()
    p = PendingAdvice(
        advice_id="a1",
        domain="typing",
        entity="keyboard",
        severity="medium",
        baseline_mean=0.9,
        timestamp="t",
    )
    p.finalized = True
    # PendingAdvice has no selected_for_rating attr → getattr guard returns False
    assert await maybe_prompt_withheld_rating(p, _cfg(), pm) is False


@pytest.mark.asyncio
async def test_helper_prompts_and_marks_when_selected(monkeypatch):
    pm = _FakePM()
    cfg = _cfg(gate_mrt_withheld_rating_rate=0.3)
    p = _gd()
    p.selected_for_rating = True
    p.finalized = True

    async def _fake_stdin(_timeout):
        return "y"

    monkeypatch.setattr(fc, "read_stdin_with_timeout", _fake_stdin)
    result = await maybe_prompt_withheld_rating(p, cfg, pm)
    assert result is True
    assert p.explicit_rating == "y"
    assert p.withheld_rating_p == 0.3
    assert pm.marked == ["s1"]


@pytest.mark.asyncio
async def test_helper_idempotent_already_prompted(monkeypatch):
    pm = _FakePM()
    p = _gd()
    p.selected_for_rating = True
    p.finalized = True
    p.withheld_rating_p = 0.3  # already prompted

    async def _fake_stdin(_timeout):  # pragma: no cover - must not be called
        raise AssertionError("should not prompt again")

    monkeypatch.setattr(fc, "read_stdin_with_timeout", _fake_stdin)
    assert await maybe_prompt_withheld_rating(p, _cfg(), pm) is False
