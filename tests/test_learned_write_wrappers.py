"""learned_write / non_learning_write — report-only groundwork (spec §7 CL10, step 5.3).

These decorators are the machinery the enforcement flip will stand on. They are
inert in the tree today: applied to no production writer yet, and the global mode
defaults to OFF. This pins their contract in isolation so the later per-writer
migration builds on tested ground:

- ``learned_write`` structurally REQUIRES a ``LearnContext`` in hand (you cannot
  forget provenance);
- OFF is a pure passthrough;
- REPORT logs what a non-learnable session WOULD have withheld, then writes anyway
  (measure blast radius against real traffic before enforcing);
- ENFORCE actually withholds a non-learnable write;
- ``non_learning_write`` records deliberate non-learning intent (+ reason) and is a
  runtime passthrough, so the CL10 discovery pass can tell it from an unguarded write.
"""

from __future__ import annotations

import logging

import pytest

from tabula.provenance import (
    LearnContext,
    ProvenanceMode,
    get_provenance_mode,
    learned_write,
    non_learning_write,
    set_provenance_mode,
)

REAL = LearnContext("s-real", True, "real")
SYNTH = LearnContext("s-synth", False, "synthetic")


@pytest.fixture(autouse=True)
def _restore_mode():
    prev = get_provenance_mode()
    yield
    set_provenance_mode(prev)


class _Writer:
    def __init__(self) -> None:
        self.writes: list = []

    @learned_write
    def save(self, ctx: LearnContext, value):
        self.writes.append(value)
        return "wrote"

    @non_learning_write(reason="audit log — never read back into a decision")
    def audit(self, value):
        self.writes.append(value)
        return "audited"


# -- OFF: pure passthrough ---------------------------------------------------


def test_off_writes_for_both_learnable_and_not() -> None:
    set_provenance_mode(ProvenanceMode.OFF)
    w = _Writer()
    assert w.save(REAL, "a") == "wrote"
    assert w.save(SYNTH, "b") == "wrote"
    assert w.writes == ["a", "b"]


def test_off_does_not_log(caplog) -> None:
    set_provenance_mode(ProvenanceMode.OFF)
    with caplog.at_level(logging.WARNING, logger="provenance"):
        _Writer().save(SYNTH, "x")
    assert not caplog.records


# -- REPORT: log the non-learnable, but still write --------------------------


def test_report_learnable_writes_without_logging(caplog) -> None:
    set_provenance_mode(ProvenanceMode.REPORT)
    w = _Writer()
    with caplog.at_level(logging.WARNING, logger="provenance"):
        assert w.save(REAL, "a") == "wrote"
    assert w.writes == ["a"]
    assert not caplog.records


def test_report_nonlearnable_writes_but_logs(caplog) -> None:
    set_provenance_mode(ProvenanceMode.REPORT)
    w = _Writer()
    with caplog.at_level(logging.WARNING, logger="provenance"):
        assert w.save(SYNTH, "b") == "wrote"  # still writes — report only
    assert w.writes == ["b"]
    assert len(caplog.records) == 1
    msg = caplog.records[0].getMessage()
    assert "s-synth" in msg and "synthetic" in msg
    assert "save" in msg


# -- ENFORCE: withhold the non-learnable write -------------------------------


def test_enforce_withholds_nonlearnable(caplog) -> None:
    set_provenance_mode(ProvenanceMode.ENFORCE)
    w = _Writer()
    with caplog.at_level(logging.WARNING, logger="provenance"):
        assert w.save(SYNTH, "b") is None  # withheld
    assert w.writes == []
    assert len(caplog.records) == 1


def test_enforce_allows_learnable() -> None:
    set_provenance_mode(ProvenanceMode.ENFORCE)
    w = _Writer()
    assert w.save(REAL, "a") == "wrote"
    assert w.writes == ["a"]


# -- Structural: provenance cannot be forgotten ------------------------------


def test_learned_write_requires_a_context() -> None:
    w = _Writer()
    with pytest.raises(TypeError):
        w.save("not-a-context", "x")  # no LearnContext in hand


def test_context_passed_as_keyword_is_found() -> None:
    set_provenance_mode(ProvenanceMode.ENFORCE)
    w = _Writer()
    assert w.save(ctx=SYNTH, value="b") is None
    assert w.writes == []


# -- CL10 discovery markers --------------------------------------------------


def test_learned_write_marks_the_function() -> None:
    assert getattr(_Writer.save, "__learned_write__", False) is True


def test_non_learning_write_marks_intent_and_reason() -> None:
    assert getattr(_Writer.audit, "__non_learning_write__", False) is True
    assert "audit log" in getattr(_Writer.audit, "__non_learning_reason__", "")


def test_non_learning_write_is_passthrough_in_every_mode() -> None:
    for mode in ProvenanceMode:
        set_provenance_mode(mode)
        w = _Writer()
        assert w.audit("x") == "audited"
        assert w.writes == ["x"]
