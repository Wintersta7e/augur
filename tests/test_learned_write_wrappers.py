"""learned_write / non_learning_write — report-only groundwork (spec §7 CL10, step 5.3).

These decorators are the machinery the enforcement flip stands on. Contract:

- ``learned_write`` finds a ``LearnContext`` among the call's arguments
  (conventionally a keyword-only ``ctx``). ``ctx`` is *optional in the signature*
  — a migration convenience so the ~300 existing storage tests keep running in OFF
  without edits — but it is **mandatory under ENFORCE**.
- **OFF** is a pure passthrough (``ctx`` not inspected).
- **REPORT** logs what a non-learnable session would withhold (and logs an
  un-migrated caller that passes no ctx), then writes anyway.
- **ENFORCE** withholds a non-learnable write and **raises** on a missing ctx, so
  provenance cannot be silently forgotten where it is enforced.
- ``non_learning_write`` records deliberate non-learning intent + reason; runtime
  passthrough.
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
    def save(self, value, *, ctx: LearnContext | None = None):
        self.writes.append(value)
        return "wrote"

    @non_learning_write(reason="audit log — never read back into a decision")
    def audit(self, value):
        self.writes.append(value)
        return "audited"


# -- OFF: pure passthrough, ctx not even required ----------------------------


def test_off_writes_with_or_without_ctx(caplog) -> None:
    set_provenance_mode(ProvenanceMode.OFF)
    w = _Writer()
    with caplog.at_level(logging.WARNING, logger="provenance"):
        assert w.save("a") == "wrote"  # no ctx — fine in OFF
        assert w.save("b", ctx=SYNTH) == "wrote"  # non-learnable — still writes
    assert w.writes == ["a", "b"]
    assert not caplog.records


# -- REPORT: log, but still write --------------------------------------------


def test_report_learnable_writes_silently(caplog) -> None:
    set_provenance_mode(ProvenanceMode.REPORT)
    w = _Writer()
    with caplog.at_level(logging.WARNING, logger="provenance"):
        assert w.save("a", ctx=REAL) == "wrote"
    assert w.writes == ["a"]
    assert not caplog.records


def test_report_nonlearnable_writes_but_logs(caplog) -> None:
    set_provenance_mode(ProvenanceMode.REPORT)
    w = _Writer()
    with caplog.at_level(logging.WARNING, logger="provenance"):
        assert w.save("b", ctx=SYNTH) == "wrote"  # still writes — report only
    assert w.writes == ["b"]
    assert len(caplog.records) == 1
    msg = caplog.records[0].getMessage()
    assert "s-synth" in msg and "synthetic" in msg and "save" in msg


def test_report_missing_ctx_writes_but_logs_unmigrated(caplog) -> None:
    set_provenance_mode(ProvenanceMode.REPORT)
    w = _Writer()
    with caplog.at_level(logging.WARNING, logger="provenance"):
        assert w.save("c") == "wrote"  # un-migrated caller — surfaced, not blocked
    assert w.writes == ["c"]
    assert len(caplog.records) == 1
    assert "un-migrated" in caplog.records[0].getMessage()


# -- ENFORCE: withhold the non-learnable, refuse a missing ctx ---------------


def test_enforce_withholds_nonlearnable(caplog) -> None:
    set_provenance_mode(ProvenanceMode.ENFORCE)
    w = _Writer()
    with caplog.at_level(logging.WARNING, logger="provenance"):
        assert w.save("b", ctx=SYNTH) is None  # withheld
    assert w.writes == []
    assert len(caplog.records) == 1


def test_enforce_allows_learnable() -> None:
    set_provenance_mode(ProvenanceMode.ENFORCE)
    w = _Writer()
    assert w.save("a", ctx=REAL) == "wrote"
    assert w.writes == ["a"]


def test_enforce_requires_a_context() -> None:
    set_provenance_mode(ProvenanceMode.ENFORCE)
    w = _Writer()
    with pytest.raises(TypeError):
        w.save("x")  # no ctx under ENFORCE — provenance cannot be forgotten
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
