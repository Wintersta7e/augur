"""Every synthetic driver that injects perception must mint synthetic provenance.

The ``scripts/`` drivers run against the *live* deploy stack (Redis db 0). Any
driver that publishes ``PerceptionEvent``s into the learning pipeline therefore
creates a session whose id flows into vigil/nexus/praesagium. Once provenance
enforcement flips (Plan 2b/2c), an unprovenanced driver session would fail
closed and be dropped — but relying on that implicit fallback is fragile. Each
such driver must instead *explicitly* record its session as ``origin="synthetic"``
(⇒ ``learnable=False``) via ``PersistenceManager.save_session_meta`` so its
non-learnability is a stated fact, not an accident of a missing record.

This is a static wiring pin (cf. ``test_cell_guard_wiring``): it discovers the
perception-injecting drivers by AST and fails if any of them — including a
future one — omits the synthetic mint. Dialogue-only drivers (which mint *real*
provenance on purpose) are not perception injectors and are not flagged.
"""

from __future__ import annotations

import ast
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"


def _constructs_perception_event(tree: ast.AST) -> bool:
    """True if the module ever calls ``PerceptionEvent(...)``."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id == "PerceptionEvent":
                return True
            if isinstance(func, ast.Attribute) and func.attr == "PerceptionEvent":
                return True
    return False


def _mints_synthetic_session(tree: ast.AST) -> bool:
    """True if the module calls ``*.save_session_meta(..., origin="synthetic")``."""
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (isinstance(func, ast.Attribute) and func.attr == "save_session_meta"):
            continue
        for kw in node.keywords:
            if (
                kw.arg == "origin"
                and isinstance(kw.value, ast.Constant)
                and kw.value.value == "synthetic"
            ):
                return True
    return False


def _perception_drivers() -> list[Path]:
    drivers = []
    for path in sorted(SCRIPTS_DIR.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        if _constructs_perception_event(tree):
            drivers.append(path)
    return drivers


def test_perception_drivers_are_discovered() -> None:
    # Guards the guard: if discovery silently finds nothing (e.g. a moved dir),
    # the coverage assertion below would vacuously pass.
    names = {p.name for p in _perception_drivers()}
    assert names == {
        "chaos_test.py",
        "complete_loop_test.py",
        "gate_probe_test.py",
        "inject_and_observe.py",
        "stress_soak.py",
        "taught_e2e_test.py",
    }, f"perception-driver set changed: {sorted(names)}"


def test_every_perception_driver_mints_synthetic_provenance() -> None:
    offenders = []
    for path in _perception_drivers():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        if not _mints_synthetic_session(tree):
            offenders.append(path.name)
    assert not offenders, (
        "these drivers inject PerceptionEvents into the live learning pipeline "
        'but never record their session as origin="synthetic" '
        f"(→ would be silently non-learnable, not explicitly): {offenders}"
    )
