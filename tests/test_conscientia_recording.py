"""record_violation_best_effort — the shared best-effort violation write.

One helper, one contract: persist the record if possible, and NEVER let a
bookkeeping failure escape to change a screen's already-decided outcome.
"""

import logging

import fakeredis

from conscientia import screens
from conscientia.recording import record_violation_best_effort
from tabula.persistence import PersistenceManager


def _record(surface="teach"):
    return screens.make_violation(
        surface, "refused", "detail", "pietas", session_id="s1"
    )


def test_persists_record_via_persistence_manager():
    pm = PersistenceManager(fakeredis.FakeStrictRedis(decode_responses=False))
    rec = _record()
    record_violation_best_effort(pm, rec)
    stored = pm.load_conscientia_violations()
    assert len(stored) == 1
    assert stored[0]["surface"] == "teach"
    assert stored[0]["session_id"] == "s1"


def test_save_failure_is_swallowed_and_logged(caplog):
    class _BrokenPM:
        def save_conscientia_violation(self, record):
            raise RuntimeError("redis down")

    with caplog.at_level(logging.WARNING, logger="conscientia.recording"):
        record_violation_best_effort(_BrokenPM(), _record())
    assert any("violation record failed" in r.getMessage() for r in caplog.records)
