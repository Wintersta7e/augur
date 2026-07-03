"""Guards the numeric bound constants that imperator/apply.py, imperator/
dialogue/intents.py, and imperator/dialogue/router.py each keep their own
copy of, per their own docstrings' "mirrored"/"layer divergence" comments.

The duplication is deliberate defense-in-depth (see imperator/dialogue/
intents.py: validate_intent has no cfg parameter by design, so it validates
against compiled-default bounds rather than the env-overridable cfg.sigma_min/
sigma_max apply.py reads) -- not something to single-source away. This test
exists only to catch one copy drifting from the others silently.

Floor bounds mirror disciplina/reflection_engine.py's GATE_FLOOR_MAX, which
this test reads via source text (never imported: that module pulls in
httpx/nats and mutates sys.path at import time -- the same reason apply.py's
own _FLOOR_MIN/_FLOOR_MAX comment gives for not importing it directly).
"""

from __future__ import annotations

import re
from pathlib import Path

from imperator import apply as A
from imperator.dialogue import intents as I
from imperator.dialogue import router as R
from tabula.config import AugurConfig

_REFLECTION_ENGINE_SRC = (
    Path(__file__).resolve().parent.parent / "disciplina" / "reflection_engine.py"
)


def _disciplina_gate_floor_max() -> float:
    src = _REFLECTION_ENGINE_SRC.read_text()
    m = re.search(r"^GATE_FLOOR_MAX\s*=\s*([0-9.]+)", src, re.MULTILINE)
    assert m, "GATE_FLOOR_MAX not found in disciplina/reflection_engine.py"
    return float(m.group(1))


def test_floor_bounds_agree_across_copies_and_disciplina_ceiling():
    assert A._FLOOR_MIN == I._FLOOR_MIN == R._FLOOR_MIN == 0.0
    disciplina_max = _disciplina_gate_floor_max()
    assert A._FLOOR_MAX == I._FLOOR_MAX == R._FLOOR_MAX == disciplina_max


def test_sigma_bounds_agree_across_copies_and_config_defaults():
    cfg = AugurConfig()
    assert I._SIGMA_MIN == R._SIGMA_MIN == cfg.sigma_min
    assert I._SIGMA_MAX == R._SIGMA_MAX == cfg.sigma_max
