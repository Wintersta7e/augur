"""Prompt forbidden-pattern guard + realized-score-pair rollback (spec 1E)."""

from tabula.config import AugurConfig
from disciplina.reflection_engine import (
    _violates_forbidden_patterns,
    maybe_rollback_prompt,
)


def test_forbidden_pattern_detected():
    cfg = AugurConfig()
    assert _violates_forbidden_patterns("Always tell them to take a break now.", cfg)
    assert not _violates_forbidden_patterns(
        "Interpret the timing; default to normal variation.", cfg
    )


class _FakePM:
    def __init__(self, cur, prev):
        self._cur, self._prev = cur, prev
        self.rolled_back = False

    def get_prompt_score_pair(self, domain):
        return self._cur, self._prev

    def rollback_prompt(self, domain):
        self.rolled_back = True
        return True


def test_rollback_when_realized_score_regresses_past_margin():
    cfg = AugurConfig(prompt_rollback_margin=0.1)
    pm = _FakePM(cur=0.4, prev=0.7)  # drop 0.3 > 0.1
    assert maybe_rollback_prompt(pm, "typing", cfg) is True
    assert pm.rolled_back


def test_no_rollback_within_margin():
    cfg = AugurConfig(prompt_rollback_margin=0.1)
    pm = _FakePM(cur=0.65, prev=0.7)  # drop 0.05 <= 0.1
    assert maybe_rollback_prompt(pm, "typing", cfg) is False
    assert not pm.rolled_back


def test_no_rollback_when_pair_incomplete():
    cfg = AugurConfig()
    assert maybe_rollback_prompt(_FakePM(0.4, None), "typing", cfg) is False
    assert maybe_rollback_prompt(_FakePM(None, 0.7), "typing", cfg) is False
