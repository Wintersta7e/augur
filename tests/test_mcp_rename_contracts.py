"""Guard augur_mcp's renamed contracts (subjects/keys/component commands).

The docker deploy smoke excludes augur_mcp, so these pin its rename surface:
faculty-named component commands, augur.sensus.* inject subjects, the
augur:nexus:window direct read, the augur.disciplina.trigger publish, and the
augur:limen:* gate-silences key. A half-renamed contract here fails the unit
suite rather than silently shipping.
"""

from __future__ import annotations

import asyncio
import inspect
from unittest.mock import AsyncMock, patch

import fakeredis

import augur_mcp.augur_server as srv
from tabula.persistence import PersistenceManager


def test_component_commands_keys_and_modules_renamed():
    # Keys are the faculty-named start_pipeline API; values point at renamed modules.
    cmds = srv.COMPONENT_COMMANDS
    assert set(cmds.keys()) == {
        "vigil",
        "nexus",
        "consilium",
        "responsum",
        "disciplina",
        "vox",
        "praefectus",
        "imperator",
    }
    assert {c[-1] for c in cmds.values()} == {
        "vigil.anomaly_detector",
        "nexus.correlator",
        "consilium.advisor",
        "responsum.feedback_collector",
        "disciplina.reflection_engine",
        "vox.console_display",
        "praefectus.monitor",
        "imperator.awareness",
    }


def test_inject_event_publishes_sensus_subject():
    # PRIMARY guard (a behavioral call, not a source string): inject_event must
    # publish on augur.sensus.{domain}. Mock the NATS client so no broker is needed.
    nc = AsyncMock()
    with patch.object(srv.nats_client, "connect", AsyncMock(return_value=nc)):
        result = asyncio.run(
            srv.inject_event(
                domain="chess",
                entity="white",
                event_type="move",
                value=1.0,
                unit="seconds",
            )
        )
    assert result["status"] == "published"
    assert result["subject"] == "augur.sensus.chess"
    published_subject = nc.publish.call_args[0][0]
    assert published_subject == "augur.sensus.chess"
    assert published_subject.startswith("augur.sensus.")


def test_inject_subjects_are_sensus_not_perception():
    # inject_event and inject_sequence both build f"augur.sensus.{domain}"
    # literally in augur_server — source-guard both ends of the rename. (A comment
    # could mask a single reverted line, which is why the call test above is primary.)
    src = inspect.getsource(srv)
    assert "augur.sensus." in src
    assert "augur.perception." not in src


def test_correlation_window_direct_read_uses_nexus_key():
    # dump_correlation_window reads augur:nexus:window DIRECTLY in augur_server
    # (not via PersistenceManager) — so the literal lives in this source.
    src = inspect.getsource(srv)
    assert "augur:nexus:window" in src
    assert "augur:correlation:window" not in src


def test_trigger_reflection_uses_disciplina_subject():
    # trigger_reflection publishes the manual reflection trigger on the renamed subject.
    src = inspect.getsource(srv)
    assert "augur.disciplina.trigger" in src
    assert "augur.reflect.trigger" not in src


def test_persistence_gate_silences_key_is_limen():
    # get_gate_silences delegates the key to PersistenceManager — so the key family
    # lives there, not in augur_server source. Assert the real family via fakeredis
    # (complements the renamed test_persistence_gate).
    pm = PersistenceManager(fakeredis.FakeStrictRedis(decode_responses=True))
    pm.save_silence_record(
        {
            "ts": "2026-01-01T00:00:00Z",
            "decision_id": "d1",
            "state_key": "single:typing:user",
            "domain": "typing",
            "entity": "user",
            "severity": "medium",
            "arm": "habituation",
            "reason": "habituated",
            "metrics": {},
            "mrt_eligible": False,
            "p_withhold": None,
        }
    )
    r = pm._r
    assert r.keys("augur:limen:*")
    assert not r.keys("augur:gate:*")
