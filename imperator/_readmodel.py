"""Shared pure helpers for the Imperator read-models (auspices + self-model).

No I/O. ``field`` is the uniform read-model cell; ``clamp`` bounds a fused
[0, 1] score. Centralised so the two sibling read-models share one definition
of a "cell".
"""

from __future__ import annotations


def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def field(value, fresh: bool, now: float) -> dict:
    """A read-model cell: a value plus its freshness and as-of timestamp."""
    return {"value": value, "fresh": fresh, "as_of": now}
