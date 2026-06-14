"""Every managed faculty must start a heartbeat under its expected id."""

import inspect

ASYNC = {
    "vigil.anomaly_detector": "vigil",
    "nexus.correlator": "nexus",
    "consilium.advisor": "consilium",
    "responsum.feedback_collector": "responsum",
    "disciplina.reflection_engine": "disciplina",
    "vox.console_display": "vox",
    "sensus.typing_monitor": "sensus.typing",
    "sensus.activity_monitor": "sensus.activity",
    "imperator.awareness": "imperator",
    "imperator.improver": "imperator_ii",
}


def _src(modpath):
    mod = __import__(modpath, fromlist=["*"])
    return inspect.getsource(mod)


def test_async_faculties_start_heartbeat():
    for modpath, fac_id in ASYNC.items():
        src = _src(modpath)
        assert "start_heartbeat(" in src, f"{modpath} missing start_heartbeat"
        assert f'"{fac_id}"' in src, f"{modpath} missing heartbeat id {fac_id!r}"
        assert ".cancel()" in src, (
            f"{modpath} missing heartbeat task cancel on shutdown"
        )


def test_chess_starts_sync_heartbeat():
    src = _src("sensus.chess_board")
    assert 'start_heartbeat("sensus.chess"' in src
    assert "stop_heartbeat()" in src
