"""Conscientia — the value-core / alignment-guardian faculty (a library,
not a process; Memoria precedent). See docs/superpowers/specs/
2026-07-07-conscientia-design.md."""

from conscientia.charter import (  # noqa: F401
    CHARTER_VERSION,
    PRINCIPLES,
    PROTECTED_SURFACES,
    render_charter,
)
from conscientia.recording import record_violation_best_effort  # noqa: F401
from conscientia.review import review_gated  # noqa: F401
from conscientia.screens import (  # noqa: F401
    CORRECTIVE_SUFFIX,
    Verdict,
    make_violation,
    screen_advice_text,
    screen_proposal,
    screen_taught_content,
)
