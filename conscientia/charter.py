"""The value core as data-in-code. Never machine-writable: no persistence
path exists for any of this, and PROTECTED_SURFACES covers conscientia/
itself (invariant C1). Edited only by a human in source control."""

from __future__ import annotations

from dataclasses import asdict, dataclass

CHARTER_VERSION = "1"


@dataclass(frozen=True)
class Principle:
    pid: str
    title: str
    text: str


PRINCIPLES: tuple[Principle, ...] = (
    Principle(
        "pietas",
        "Devotion to the user",
        "The user is the center of Augur's existence; every action is judged "
        "by whether it serves them.",
    ),
    Principle(
        "restraint",
        "Observe, never coerce",
        "Augur informs, and asserts when the user's interest demands; it never "
        "pressures, guilt-trips, or manufactures urgency. Forbidden-valence "
        "phrasings are refused even at the cost of silence.",
    ),
    Principle(
        "reversibility",
        "Every change can be undone",
        "No self-modification without a recorded rollback anchor; what cannot "
        "be reversed is not applied.",
    ),
    Principle(
        "transparency",
        "No silent power",
        "Every blocked output, refused teaching, and self-modification verdict "
        "is logged and observable; Conscientia never censors silently.",
    ),
    Principle(
        "containment",
        "The guardian is not self-modifiable",
        "No proposal may alter Conscientia, the Limen invariants, or the apply "
        "machinery that enforces them.",
    ),
)

# Path/key prefixes no proposal may target (case-insensitive prefix match on
# the proposal's target string). Gated review -> reject; safe screen -> refuse.
PROTECTED_SURFACES: tuple[str, ...] = (
    "conscientia/",
    "limen/",
    "imperator/apply.py",
    "imperator/proposals.py",
    "nexus/matrix_ops.py",
    "augur:conscientia:",
)


def output_patterns(cfg) -> tuple[str, ...]:
    """Valence patterns for outgoing advice text: the shared prompt-mutation
    list plus output-surface extras (single source of truth, spec D8).
    getattr-tolerant so a partially-shaped cfg double degrades to fewer
    patterns instead of raising (the screens treat missing flags as ON)."""
    return tuple(getattr(cfg, "prompt_forbidden_patterns", ())) + tuple(
        getattr(cfg, "conscientia_output_extra_patterns", ())
    )


def teach_patterns(cfg) -> tuple[str, ...]:
    """Valence patterns for user-taught content (rationale / rule_key).
    getattr-tolerant so a partially-shaped cfg double degrades to fewer
    patterns instead of raising (the screens treat missing flags as ON)."""
    return tuple(getattr(cfg, "prompt_forbidden_patterns", ())) + tuple(
        getattr(cfg, "conscientia_teach_extra_patterns", ())
    )


def render_charter() -> dict:
    """The charter as plain data for MCP/introspection. No Redis involved."""
    return {
        "version": CHARTER_VERSION,
        "principles": [asdict(p) for p in PRINCIPLES],
        "protected_surfaces": list(PROTECTED_SURFACES),
    }
