"""Advisor receptivity/burden gate — decisions and signature.

This module holds the gate's value objects:

- ``GateDecision`` — the frozen result of a gate evaluation, carrying the
  ``decision_id`` (a ``uuid4`` minted in ``evaluate``) that threads through
  every emission/silence/advice/feedback record for exact MRT/audit linkage.
- ``Signature`` — the deterministic, severity-normalized description of an
  incoming anomaly payload used by the arm pipeline.
- ``build_signature`` — pure constructor for ``Signature`` (spec §5).

The arm pipeline + ``Gate`` class land in later tasks; this task is the value
objects only.  All gate decision logic is intentionally side-effect-free here
(no clock, no Redis, no rng) — those are injected by the ``Gate`` later.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field, replace
from typing import Any, Literal

# Internal module constant guarding divide-by-zero in the novelty arm (spec §5).
_EPS = 1e-6


@dataclass(frozen=True)
class GateDecision:
    """A frozen gate outcome (spec §4).

    ``action`` is the terminal disposition; ``reason``/``deciding_arm`` explain
    it; ``metrics`` carries arm-specific numbers for logging.  ``id`` is the
    ``decision_id`` — minted once via ``uuid4().hex`` and PRESERVED across any
    fail-open/cap conversion so a suppress→fire keeps the same linkage key.
    The MRT fields (``mrt_eligible``/``p_fire``/``p_withhold``) record the
    known randomization probabilities so both arms are inverse-probability
    weightable even under a dynamic ``ε``.
    """

    action: Literal["fire", "suppress", "downgrade"]
    reason: str
    deciding_arm: str = "none"
    metrics: dict[str, Any] = field(default_factory=dict)
    tier: int | None = None
    id: str = ""
    probe: bool = False
    withheld_reason: str | None = None
    mrt_eligible: bool = False
    p_fire: float | None = None
    p_withhold: float | None = None

    @classmethod
    def fire(
        cls,
        reason: str,
        *,
        deciding_arm: str = "none",
        metrics: dict[str, Any] | None = None,
        tier: int | None = None,
        id: str | None = None,
        probe: bool = False,
        withheld_reason: str | None = None,
        mrt_eligible: bool = False,
        p_fire: float | None = None,
        p_withhold: float | None = None,
    ) -> GateDecision:
        return cls(
            action="fire",
            reason=reason,
            deciding_arm=deciding_arm,
            metrics=metrics or {},
            tier=tier,
            id=id if id is not None else uuid.uuid4().hex,
            probe=probe,
            withheld_reason=withheld_reason,
            mrt_eligible=mrt_eligible,
            p_fire=p_fire,
            p_withhold=p_withhold,
        )

    @classmethod
    def suppress(
        cls,
        reason: str,
        *,
        deciding_arm: str = "none",
        metrics: dict[str, Any] | None = None,
        id: str | None = None,
        mrt_eligible: bool = False,
        p_withhold: float | None = None,
    ) -> GateDecision:
        return cls(
            action="suppress",
            reason=reason,
            deciding_arm=deciding_arm,
            metrics=metrics or {},
            id=id if id is not None else uuid.uuid4().hex,
            mrt_eligible=mrt_eligible,
            p_withhold=p_withhold,
        )

    @classmethod
    def downgrade(
        cls,
        reason: str,
        *,
        deciding_arm: str = "none",
        metrics: dict[str, Any] | None = None,
        tier: int | None = 1,
        id: str | None = None,
    ) -> GateDecision:
        return cls(
            action="downgrade",
            reason=reason,
            deciding_arm=deciding_arm,
            metrics=metrics or {},
            tier=tier,
            id=id if id is not None else uuid.uuid4().hex,
        )

    def as_fire(self, reason: str) -> GateDecision:
        """Convert this decision to a FIRE, PRESERVING ``id`` (spec §3/§6/§9).

        Used for suppress→fail-open and busy→fail-open / cap-fail-open
        conversions so the same ``decision_id`` threads through emission,
        silence, advice, and feedback.  ``deciding_arm``/``metrics`` are carried
        for audit; suppress-only MRT/probe fields are dropped.
        """
        return replace(
            self,
            action="fire",
            reason=reason,
            tier=None,
            probe=False,
            withheld_reason=None,
            mrt_eligible=False,
            p_fire=None,
            p_withhold=None,
        )


@dataclass(frozen=True)
class Signature:
    """Deterministic, severity-normalized description of an anomaly (spec §5)."""

    severity: str  # normalized lowercase ("medium"/"high")
    severity_score: float  # medium → 1.0, high → 2.0
    path: str  # "single" or "correlation"
    correlation_found: bool
    exempt: bool  # correlation_found and severity == "high"
    domain: str
    entity: str | None
    value: float
    state_key: str
    involved_domains: tuple[str, ...]
    ungateable: bool  # single event with missing/"?"/empty entity


def _norm(severity: Any) -> str:
    """Normalize a severity value to lowercase (the single normalization helper)."""
    return str(severity or "").lower()


def build_signature(payload: dict[str, Any]) -> Signature:
    """Build the deterministic gate :class:`Signature` for an anomaly payload.

    Severity is normalized to lowercase via :func:`_norm`; ``severity_score``
    is ``medium → 1.0`` / ``high → 2.0``.  ``state_key`` is entity-grouped and
    severity-omitted: single → ``single:{domain}:{entity}``; correlation →
    ``correlation:{'+'.join(sorted(involved_domains))}`` (off ``involved_domains``,
    sorted, stable).  ``exempt`` is ``correlation_found and severity == "high"``.
    A single event with a missing/``"?"``/empty entity is ``ungateable``.
    """
    severity = _norm(payload.get("combined_severity"))
    severity_score = 2.0 if severity == "high" else 1.0
    correlation_found = bool(payload.get("correlation_found"))
    exempt = correlation_found and severity == "high"

    primary = payload.get("primary_anomaly") or {}
    domain = primary.get("domain", "")
    value = float(primary.get("value", 0.0) or 0.0)
    entity = primary.get("entity")

    if correlation_found:
        path = "correlation"
        involved = tuple(sorted(payload.get("involved_domains") or []))
        state_key = f"correlation:{'+'.join(involved)}"
        # Correlation keys off involved_domains, not the primary entity, so a
        # missing entity does not make a correlation ungateable.
        ungateable = False
    else:
        path = "single"
        involved = ()
        ungateable = entity in (None, "?", "")
        state_key = f"single:{domain}:{entity}"

    return Signature(
        severity=severity,
        severity_score=severity_score,
        path=path,
        correlation_found=correlation_found,
        exempt=exempt,
        domain=domain,
        entity=entity,
        value=value,
        state_key=state_key,
        involved_domains=involved,
        ungateable=ungateable,
    )
