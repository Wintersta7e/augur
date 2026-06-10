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

import logging
import math
import random
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Any, Literal

log = logging.getLogger(__name__)

# Internal module constant guarding divide-by-zero in the novelty arm (spec §5).
_EPS = 1e-6

# The Redis hash whose cap (MAX_GATE_STATE_KEYS) decides trackability: a new
# state_key that cannot be created here is untrackable for anti-starvation, so a
# would-suppress on it fails open to FIRE (cap_fail_open, invariant D / spec §6).
_CHANNEL_STATS_KEY = "augur:gate:channel_stats"

# Arms a non-exempt HIGH skips — the learned/recurrence suppressors (spec §5
# "Non-exempt severity==\"high\" bypass").  A standalone high punches through
# central_tolerance/novelty/habituation/reservoir/credibility by construction;
# it remains subject to refractory_burden, cost_tier, and anti_starvation.
_HIGH_BYPASSED_ARMS = frozenset(
    {
        "central_tolerance",
        "novelty_prediction_error",
        "habituation",
        "coincidence_evidence_reservoir",
        "signaller_credibility",
    }
)

# An arm is a pure callable over the loaded snapshot returning a SUPPRESS/
# DOWNGRADE/FIRE GateDecision, or None to pass. Real arms (Phase 4+) are private
# methods bound into this list; the list is injectable so the skeleton + each arm
# is testable in isolation.
Arm = Callable[
    ["Gate", "Signature", dict[str, Any], Any, float, random.Random],
    "GateDecision | None",
]


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
    escalation_rule: str | None  # correlation rule label; None for single events


def _norm(severity: Any) -> str:
    """Normalize a severity value to lowercase (the single normalization helper)."""
    return str(severity or "").lower()


def _arm_name(arm: Arm) -> str:
    """Derive an arm's pipeline name from its ``_arm_<name>`` method name.

    Used by ``evaluate`` to apply the HIGH bypass (which skips named learned/
    recurrence arms) without coupling the loop to the concrete method objects.
    """
    name = getattr(arm, "__name__", "")
    return name[len("_arm_") :] if name.startswith("_arm_") else name


def _credibility_class(sig: Signature) -> str:
    """The per-signal-class credibility key (spec §5 Arm 6).

    Correlated events key off the ``escalation_rule`` (the rule that fired);
    when a correlation has no rule label, fall back to its (severity-omitted)
    ``state_key`` so the key is always a stable, deterministic string.  Single
    events key off the ``(domain, severity)`` pair as ``f"{domain}:{severity}"``.
    """
    if sig.correlation_found:
        return sig.escalation_rule or sig.state_key
    return f"{sig.domain}:{sig.severity}"


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
    escalation_rule = payload.get("escalation_rule")

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
        escalation_rule=escalation_rule,
    )


class Gate:
    """The advisor receptivity/burden gate (spec §3/§4).

    ``evaluate`` is **strictly read-only**: it reads gate state via the guarded
    ``load_gate_*`` helpers, runs the (injected) arm pipeline against that one
    consistent snapshot, and returns a :class:`GateDecision` — it performs no
    Redis write and no ``await``.  All mutation lives in the ``record_*`` methods
    (added in later tasks).

    The arm pipeline is **injectable** (``Gate(arms=[...])`` defaulting to an
    empty list) so the skeleton + each arm are unit-testable in isolation; real
    arms (Phase 4+) are bound in here as private methods.  Each arm is a pure
    callable ``arm(self, sig, state, config, now, rng) -> GateDecision | None``
    (``None`` = pass); the first non-``None`` SUPPRESS/DOWNGRADE/FIRE wins.
    """

    def __init__(self, arms: list[Arm] | None = None, config: Any = None) -> None:
        self.arms: list[Arm] = arms if arms is not None else self._default_arms()
        # ``still_starved`` (called under the lock with the spec-fixed signature
        # ``(signature, pm, now)`` — no config arg) needs the starvation bounds,
        # so the Gate holds a config.  Defaults to AugurConfig() for the unit
        # tests / standalone construction; the advisor injects its live config.
        if config is None:
            from tabula.config import AugurConfig

            config = AugurConfig()
        self._config = config

    def _default_arms(self) -> list[Arm]:
        """The arm pipeline in spec §5 order (suppressors, Phase 1).

        Order is load-bearing: the first arm to suppress wins, so the documented
        precedence (central_tolerance → refractory → novelty → habituation →
        reservoir → credibility) is encoded here.
        """
        return [
            Gate._arm_central_tolerance,
            Gate._arm_refractory_burden,
            Gate._arm_novelty_prediction_error,
            Gate._arm_habituation,
            Gate._arm_coincidence_evidence_reservoir,
            Gate._arm_signaller_credibility,
        ]

    def evaluate(
        self,
        signature: Signature,
        pm: Any,
        config: Any,
        *,
        now: float,
        rng: random.Random = random.Random(),
    ) -> GateDecision:
        """Return a :class:`GateDecision` for ``signature`` (read-only).

        Stage 0 / fast exits (no gate-state reads):

        * master-disable (``not config.gate_enabled``) → ``FIRE("gate_disabled")``;
        * danger exemption (``signature.exempt``) →
          ``FIRE("exempt_high_correlated")`` performing **no** state read (§2 B).

        Otherwise gate state is loaded up front (read-only) and the **two-phase**
        pipeline runs (spec §5):

        * **Phase 1** — suppressors (Arms 1–6); the first to suppress wins.  With
          no suppressor (or none configured) the event passes → a fire-survivor.
        * **Phase 2** — modifiers on the Phase-1 outcome: Arm 7 ``cost_tier_router``
          may downgrade/silence a *fire*-survivor.  (Arms 9/8 join here in later
          tasks.)

        Any SUPPRESS (Phase-1 or a Phase-2 silent downgrade) whose ``state_key``
        is **new and untrackable** (``not pm.can_track_gate_state(...)``) fails
        open to ``FIRE("cap_fail_open")`` preserving the decision id (invariant
        D / §6).
        """
        if not config.gate_enabled:
            return GateDecision.fire("gate_disabled", deciding_arm="master_disable")

        if signature.exempt:
            # §2(B): for an exempt signature NO gate state is ever read.
            return GateDecision.fire(
                "exempt_high_correlated", deciding_arm="danger_exemption"
            )

        if signature.ungateable:
            # §5 missing-entity rule: a single event with a missing/"?"/empty
            # entity gets ONLY the GLOBAL refractory checks — no per-channel arms
            # (no "?" channel state created), no cap-fail-open (no per-channel
            # key).  Suppress only if a global refractory sub-check fires.
            return self._evaluate_ungateable(signature, pm, config, now)

        state = self._load_state(pm, signature)

        # ── Phase 1: suppressors (first to suppress wins) ──
        decision = self._run_suppressors(signature, state, config, now, rng)
        if decision is None:
            decision = GateDecision.fire("passed_all_arms", deciding_arm="none")

        # ── Phase 2: modifiers on the Phase-1 outcome (spec §5) ──
        decision = self._finalize(decision, signature, state, config, now, rng)

        # Cap fail-open: a SUPPRESS for a new, untrackable state_key cannot be
        # anti-starved → fail open to FIRE rather than silence it indefinitely
        # (invariant D, spec §6).  Applies to a Phase-1 suppress AND a Phase-2
        # silent downgrade.
        if decision.action == "suppress" and not pm.can_track_gate_state(
            _CHANNEL_STATS_KEY, signature.state_key
        ):
            return decision.as_fire("cap_fail_open")

        return decision

    def _evaluate_ungateable(
        self,
        signature: Signature,
        pm: Any,
        config: Any,
        now: float,
    ) -> GateDecision:
        """Decide an ungateable signature with ONLY the global arms (spec §5).

        A single event with a missing/``"?"``/empty entity has no stable
        per-channel ``state_key`` to track, so the per-channel arms
        (central_tolerance/novelty/habituation/reservoir/credibility/cost_tier/
        anti_starvation) are skipped — they would otherwise mute every
        entity-less event behind a bogus ``single:{domain}:?`` channel.  Only the
        global ``refractory_burden`` sub-checks (absolute / relative-bar /
        pressure — NOT the per-``state_key`` duplicate) apply.  Reads only the
        global emission log + advice-rate; no per-channel state is loaded.  No
        cap-fail-open (there is no per-channel key to refuse).  Suppress if a
        global sub-check fires, else FIRE.
        """
        if not config.gate_refractory_enabled:
            return GateDecision.fire("passed_all_arms", deciding_arm="none")
        state = {
            "emissions": pm.load_emissions(),
            "advice_rate": pm.load_advice_rate(),
        }
        decision = self._arm_refractory_burden(
            signature, state, config, now, random.Random(), global_only=True
        )
        if decision is not None:
            return decision
        return GateDecision.fire("passed_all_arms", deciding_arm="none")

    def _run_suppressors(
        self,
        signature: Signature,
        state: dict[str, Any],
        config: Any,
        now: float,
        rng: random.Random,
    ) -> GateDecision | None:
        """Run the Phase-1 suppressor pipeline; return the first SUPPRESS or None.

        A non-exempt standalone HIGH skips the learned/recurrence suppressors
        (spec §5 HIGH bypass): it punches through central_tolerance/novelty/
        habituation/reservoir/credibility by construction.
        """
        bypass_learned = signature.severity == "high"
        for arm in self.arms:
            if bypass_learned and _arm_name(arm) in _HIGH_BYPASSED_ARMS:
                continue
            decision = arm(self, signature, state, config, now, rng)
            if decision is not None:
                return decision
        return None

    def _finalize(
        self,
        decision: GateDecision,
        signature: Signature,
        state: dict[str, Any],
        config: Any,
        now: float,
        rng: random.Random,
    ) -> GateDecision:
        """Apply the Phase-2 modifiers to the Phase-1 outcome (spec §5).

        Order: Arm 7 (cost_tier — downgrades a *fire*-survivor) → Arm 9
        (anti-starvation, added in Task 5.3 — deterministically un-suppresses a
        starved channel) → Arm 8 (bet-hedge — stochastically un-suppresses a
        behavioral-driven suppress, only if Arm 9 did not fire).
        """
        if decision.action == "fire":
            decision = self._arm_cost_tier_router(
                decision, signature, state, config, now, rng
            )
        # Arm 9 (anti-starvation) runs BEFORE Arm 8 so a starved channel is
        # released deterministically and short-circuits the stochastic bet-hedge.
        decision = self._arm_anti_starvation_release(
            decision, signature, state, config, now, rng
        )
        decision = self._arm_bet_hedge_override(
            decision, signature, state, config, now, rng
        )
        return decision

    def _load_state(self, pm: Any, signature: Signature) -> dict[str, Any]:
        """Load the gate-state snapshot for ``signature`` (read-only, guarded).

        Each ``load_gate_*`` helper guards its own decode errors and returns a
        safe "unseen" default, so a corrupt/partial read degrades to a clean
        snapshot (invariant C).  Arms (Phase 4+) read from this dict only.
        """
        return {
            "habituation": pm.load_habituation(signature.state_key),
            "habituation_floor": pm.load_habituation_floor(signature.state_key),
            "reservoir": pm.load_reservoir(signature.state_key),
            "cost_tier_memory": pm.load_cost_tier_memory(signature.state_key),
            "channel_stats": pm.load_channel_stats(signature.state_key),
            "self_tolerant": pm.is_self_tolerant(signature.state_key),
            "advice_rate": pm.load_advice_rate(),
            "emissions": pm.load_emissions(),
            "observed": pm.load_observed(signature.state_key),
            "credibility": pm.load_credibility(_credibility_class(signature)),
        }

    # ── Arms (spec §5) ── pure: read the snapshot only, no clock/Redis/rng side
    # effects.  ``now``/``rng`` are injected for the arms that need them.
    def _arm_central_tolerance(
        self,
        sig: Signature,
        state: dict[str, Any],
        config: Any,
        now: float,  # noqa: ARG002 — uniform arm signature
        rng: random.Random,  # noqa: ARG002 — uniform arm signature
    ) -> GateDecision | None:
        """Arm 1 — immune central tolerance (spec §5).

        A non-high event whose ``state_key`` is in the offline-learned
        ``self_tolerance`` set is a chronic, explicitly-dismissed channel →
        ``SUPPRESS("central_tolerance_learned_self")``.  (High never reaches
        this arm — it is skipped by the HIGH bypass in ``evaluate``.)
        """
        if not config.gate_central_tolerance_enabled:
            return None
        if sig.severity == "high":
            return None
        if not state.get("self_tolerant"):
            return None
        return GateDecision.suppress(
            "central_tolerance_learned_self",
            deciding_arm="central_tolerance",
            metrics={"state_key": sig.state_key},
        )

    def _arm_refractory_burden(
        self,
        sig: Signature,
        state: dict[str, Any],
        config: Any,
        now: float,
        rng: random.Random,  # noqa: ARG002 — uniform arm signature; deterministic arm
        *,
        global_only: bool = False,
    ) -> GateDecision | None:
        """Arm 2 — refractory + Treg resolution + sentinel duplicate (spec §5).

        A **global** arm (applies even to an ungateable event and to a HIGH —
        the HIGH bypass skips only the learned/recurrence arms, not this one).
        Reads the emission log IGNORING ``probe``/``audit_only`` entries (those
        are not deliveries that should refract the channel) plus the advice-rate
        EWMA.  Four sub-reasons checked in order:

        When ``global_only=True`` (the ungateable path, spec §5) the per-
        ``state_key`` **duplicate** sub-check is skipped — it is a per-channel
        check and an ungateable signature has no stable channel; only the truly
        global absolute / relative-bar / pressure sub-checks apply.

        1. **Absolute** — ``now - last_global_emit < ABSOLUTE_REFRACTORY_S`` →
           ``SUPPRESS("absolute_refractory", {remaining_s})``.
        2. **Relative** — within ``RELATIVE_REFRACTORY_S`` the bar is raised to
           ``high`` post-fire and decays linearly to ``medium``; an event whose
           ``severity_score`` is below the bar →
           ``SUPPRESS("relative_refractory_raised_bar", {bar, elapsed_s})``.
        3. **Pressure** — ``min(advice_rate, PRESSURE_CAP) * PRESSURE_WEIGHT >
           severity_score`` →
           ``SUPPRESS("active_resolution_recent_advice_pressure", {advice_rate})``.
        4. **Duplicate** — a per-``state_key`` real emission within
           ``RELATIVE_REFRACTORY_S`` →
           ``SUPPRESS("already_covered_recent_equivalent", {dt})``.
        """
        if not config.gate_refractory_enabled:
            return None

        emissions = state.get("emissions") or []
        # Real emissions only: probe (bet-hedge) + audit_only (exempt) fires are
        # gating-invisible — they must not refract the channel (spec §5/§6).
        real = [e for e in emissions if not e.get("probe") and not e.get("audit_only")]

        # ── Absolute + relative both key off the most recent global emission ──
        last_global_ts = max((float(e.get("ts", 0.0)) for e in real), default=None)
        if last_global_ts is not None:
            dt_global = now - last_global_ts
            if 0.0 <= dt_global < config.gate_absolute_refractory_s:
                return GateDecision.suppress(
                    "absolute_refractory",
                    deciding_arm="refractory_burden",
                    metrics={
                        "remaining_s": config.gate_absolute_refractory_s - dt_global
                    },
                )
            # Relative: within the window the bar decays linearly from high (2.0)
            # to medium (1.0).  An event below the current bar is held.
            if 0.0 <= dt_global < config.gate_relative_refractory_s:
                frac = dt_global / config.gate_relative_refractory_s
                bar_score = 2.0 - (2.0 - 1.0) * frac
                if sig.severity_score < bar_score:
                    return GateDecision.suppress(
                        "relative_refractory_raised_bar",
                        deciding_arm="refractory_burden",
                        metrics={"bar": bar_score, "elapsed_s": dt_global},
                    )

        # ── Pressure: active-resolution back-pressure from the advice-rate EWMA ─
        rate = float((state.get("advice_rate") or {}).get("rate_ewma", 0.0) or 0.0)
        effective_rate = min(rate, config.gate_pressure_cap)
        if effective_rate * config.gate_pressure_weight > sig.severity_score:
            return GateDecision.suppress(
                "active_resolution_recent_advice_pressure",
                deciding_arm="refractory_burden",
                metrics={"advice_rate": effective_rate},
            )

        # ── Duplicate: a real same-state_key emission within the relative window ─
        # Per-channel sub-check — skipped on the ungateable (global-only) path.
        if global_only:
            return None
        for e in real:
            if e.get("state_key") != sig.state_key:
                continue
            dt = now - float(e.get("ts", 0.0))
            if 0.0 <= dt < config.gate_relative_refractory_s:
                return GateDecision.suppress(
                    "already_covered_recent_equivalent",
                    deciding_arm="refractory_burden",
                    metrics={"dt": dt},
                )

        return None

    def _arm_novelty_prediction_error(
        self,
        sig: Signature,
        state: dict[str, Any],
        config: Any,
        now: float,  # noqa: ARG002 — uniform arm signature; deterministic arm
        rng: random.Random,  # noqa: ARG002 — uniform arm signature; deterministic arm
    ) -> GateDecision | None:
        """Arm 3 — RPE / predictive coding (spec §5).

        Gates on **value surprise** (distinct from habituation's advice
        frequency).  From the per-``state_key`` observed-value window maintain an
        EWMA ``predicted_value``; the just-noticeable difference is
        ``relative_change = |value - predicted_value| / max(|predicted_value|,
        _EPS)``.  A **familiar** channel (``match_count >= NOVELTY_FAMILIAR_MIN``)
        whose event lands below the Weber JND (``relative_change <
        WEBER_FRACTION``) is fully predicted → ``SUPPRESS`` with the predicted
        value + relative change.  An unseen/first-time / not-yet-familiar
        ``state_key`` → PASS.  (High never reaches this arm — the HIGH bypass in
        ``evaluate`` skips it.)
        """
        if not config.gate_novelty_enabled:
            return None

        observed = state.get("observed") or []
        match_count = len(observed)
        if match_count < config.gate_novelty_familiar_min:
            return None  # unseen / not yet familiar enough to explain away

        # load_observed returns newest-first; fold oldest→newest into an EWMA so
        # the prediction is weighted toward the most recent observations.
        alpha = config.gate_habituation_alpha
        values = [float(r.get("value", 0.0) or 0.0) for r in reversed(observed)]
        predicted = values[0]
        for v in values[1:]:
            predicted = alpha * v + (1.0 - alpha) * predicted

        relative_change = abs(sig.value - predicted) / max(abs(predicted), _EPS)
        if relative_change < config.gate_weber_fraction:
            return GateDecision.suppress(
                "fully_predicted_explained_away",
                deciding_arm="novelty_prediction_error",
                metrics={
                    "predicted_value": predicted,
                    "relative_change": relative_change,
                },
            )
        return None

    def _arm_habituation(
        self,
        sig: Signature,
        state: dict[str, Any],
        config: Any,
        now: float,
        rng: random.Random,  # noqa: ARG002 — uniform arm signature; deterministic arm
    ) -> GateDecision | None:
        """Arm 4 — Aplysia depression + immune anergy (spec §5).

        Gates on **advice frequency** (distinct from novelty's value surprise).
        Each non-probe delivery on a ``state_key`` advances its habituation
        ``h ∈ [0, 1]`` (offline in ``record_delivery_success``); here we read it,
        leak it toward 0 since the last event, and suppress a routine event whose
        leaky response ``R`` has fallen below ``R_THRESHOLD``:

        * ``dt = now - last_event_ts``;
        * ``h_eff = min(h * exp(-dt / TAU_S), 1 - max(FLOOR_MIN, floor))`` — the
          floor-guard caps habituation so an offline-lowered ``floor`` keeps a
          channel responsive;
        * ``R = severity_score * (1 - h_eff)``;
        * ``R < R_THRESHOLD`` → ``SUPPRESS("habituated", {count, interval_s,
          h_eff, dt})``.

        An unseen channel (no habituation state) has ``h = 0`` → ``h_eff = 0`` →
        ``R = severity_score`` ≥ threshold → PASS.  (High never reaches this arm
        — the HIGH bypass in ``evaluate`` skips it.)  Per-channel: a fresh key
        per domain/entity.
        """
        if not config.gate_habituation_enabled:
            return None

        hab = state.get("habituation") or {}
        h = float(hab.get("h", 0.0) or 0.0)
        if h <= 0.0:
            return None  # unseen / fully recovered channel — nothing to habituate

        last_event_ts = float(hab.get("last_event_ts", now) or now)
        dt = now - last_event_ts

        floor = float((state.get("habituation_floor") or {}).get("floor", 0.0) or 0.0)
        cap = 1.0 - max(config.gate_habituation_floor_min, floor)

        decayed = h * math.exp(-dt / config.gate_habituation_tau_s)
        h_eff = min(decayed, cap)

        r = sig.severity_score * (1.0 - h_eff)
        if r < config.gate_habituation_r_threshold:
            return GateDecision.suppress(
                "habituated",
                deciding_arm="habituation",
                metrics={
                    "count": hab.get("count", 0),
                    "interval_s": dt,
                    "h_eff": h_eff,
                    "dt": dt,
                },
            )
        return None

    def _arm_coincidence_evidence_reservoir(
        self,
        sig: Signature,
        state: dict[str, Any],
        config: Any,
        now: float,
        rng: random.Random,  # noqa: ARG002 — uniform arm signature; deterministic arm
    ) -> GateDecision | None:
        """Arm 5 — quorum sensing + immune two-signal (spec §5).

        Meters **only** ``single`` + ``medium`` events.  A second signal makes
        the evidence sufficient on its own, so it short-circuits to PASS:
        ``severity == "high"`` OR ``correlation_found`` (a standalone high also
        never reaches this arm via the HIGH bypass in ``evaluate``).

        Otherwise a per-``state_key`` decaying event count accumulates (+1 per
        qualifying event, leaking by ``exp(-dt / RESERVOIR_LEAK_TAU_S)``); the
        event being evaluated reads the current count + its own prospective +1
        (``effective``).  A **Schmitt-trigger** with hysteresis avoids flapping:

        * a suppressing channel commits (PASS) only when
          ``effective >= RESERVOIR_ON_COUNT``;
        * a committed channel (``suppressing == False``) re-suppresses only when
          its leaked count falls **below** ``RESERVOIR_OFF_COUNT``.

        Below ON (and not held committed by hysteresis) →
        ``SUPPRESS("single_channel_insufficient", {count, on, off})``.  The count
        itself advances in ``record_suppression``/``record_delivery_success``;
        this arm only reads the snapshot.
        """
        if not config.gate_reservoir_enabled:
            return None

        # Two-signal short-circuit: a second independent signal (high or a
        # correlation) is sufficient evidence on its own → never metered.
        if sig.severity == "high" or sig.correlation_found:
            return None

        # Only single+medium is metered (high handled above; correlation keys
        # off a different state_key and is short-circuited above).
        if sig.path != "single":
            return None

        res = state.get("reservoir") or {}
        count = float(res.get("count", 0.0) or 0.0)
        last_ts = float(res.get("last_ts", now) or now)
        # A brand-new / never-committed channel starts latched suppressing.
        suppressing = bool(res.get("suppressing", True))

        dt = max(0.0, now - last_ts)
        leaked = count * math.exp(-dt / config.gate_reservoir_leak_tau_s)
        effective = leaked + 1.0  # the event reads its own prospective +1

        on = config.gate_reservoir_on_count
        off = config.gate_reservoir_off_count

        if suppressing:
            # Latched suppressing: commit (PASS) only on reaching ON.
            committed = effective >= on
        else:
            # Latched committed: re-suppress only once leaked count drops below OFF.
            committed = leaked >= off

        if committed:
            return None

        return GateDecision.suppress(
            "single_channel_insufficient",
            deciding_arm="coincidence_evidence_reservoir",
            metrics={"count": effective, "on": on, "off": off},
        )

    def _arm_signaller_credibility(
        self,
        sig: Signature,
        state: dict[str, Any],
        config: Any,
        now: float,
        rng: random.Random,
    ) -> GateDecision | None:
        """Arm 6 — cry-wolf + Friston precision (spec §5 Arm 6 + §10).

        Per signal-class (``escalation_rule`` for correlated, else
        ``f"{domain}:{severity}"``) an offline EWMA ``credibility ∈ [0, 1]``
        derived from reliability-weighted feedback.  An unseen class sits at the
        prior ``CRED_MID`` (so ``P(suppress)`` is 0 — a new class never
        suppresses).  Two read-time transforms before the suppression draw:

        * **decay-toward-prior** — a class with no recent feedback is pulled back
          toward the prior, ``cred_eff = prior + (cred - prior) *
          exp(-DECAY_ALPHA * (now - last_fb_ts))``; this self-heals a
          frozen-suppressed class (its credibility relaxes to neutral over time).
        * **reliability-weighted fusion (§10)** — a stored ``behavioral_score``
          is fused in **explicit-dominant**
          (``(EXPLICIT_WEIGHT*cred_eff + BEHAVIORAL_WEIGHT*behavioral_score) /
          (EXPLICIT_WEIGHT + BEHAVIORAL_WEIGHT)``) **only** when it is
          ``behavioral_finalized``, has ``>= BEHAVIORAL_MIN_SAMPLES`` genuine
          responses, and clears the deadband (``|behavioral_score - 0.5| >
          BEHAVIORAL_DEADBAND``); otherwise the behavioral signal is quarantined.

        ``P(suppress) = clamp((CRED_MID - credibility) / CRED_MID, 0,
        CRED_MAX_P)``; a positive seeded-``rng`` draw (``rng.random() < p``) →
        ``SUPPRESS("low_credibility_class", {credibility, p})``.  A suppression
        that the behavioral fusion drove is **bet-hedge-eligible** — it carries
        ``mrt_eligible=True`` + ``p_withhold=p`` so Arm 8 can later flip it as the
        measured MRT intervention (spec §5 Arm 8 / §9).  (High never reaches this
        arm — the HIGH bypass in ``evaluate`` skips it, so a standalone high with
        low credibility still fires.)
        """
        if not config.gate_credibility_enabled:
            return None

        prior = config.gate_cred_mid
        entry = state.get("credibility") or {}
        if not entry:
            return None  # unseen class → prior → P(suppress)=0 → never suppress

        # A stored cred of 0.0 is the worst-credibility class (P(suppress) at the
        # CRED_MAX_P ceiling), a real value — not "missing" — so do NOT collapse
        # it to the prior with `or prior` (that would silently yield P=0).
        cred_raw = entry.get("cred")
        cred = float(cred_raw if cred_raw is not None else prior)

        # Decay toward the prior with elapsed time since the last feedback —
        # a stale (no recent feedback) class relaxes to neutral and self-heals.
        # (A stored last_fb_ts of 0.0 is a real timestamp, not "missing", so do
        # NOT collapse it with `or now`.)
        last_fb_ts_raw = entry.get("last_fb_ts")
        last_fb_ts = float(last_fb_ts_raw if last_fb_ts_raw is not None else now)
        dt = max(0.0, now - last_fb_ts)
        cred_eff = prior + (cred - prior) * math.exp(
            -config.gate_credibility_decay_alpha * dt
        )

        # Reliability-weighted fusion (§10): explicit dominant; behavioral only
        # when finalized, above the deadband, and with enough genuine samples.
        behavioral_driven = False
        credibility = cred_eff
        if entry.get("behavioral_finalized"):
            # A stored behavioral_score of 0.0 is a real (maximally-negative)
            # score, not "missing" — fall back to 0.5 only when truly absent.
            bs_raw = entry.get("behavioral_score")
            behavioral_score = float(bs_raw if bs_raw is not None else 0.5)
            behavioral_samples = int(entry.get("behavioral_samples", 0) or 0)
            if (
                behavioral_samples >= config.gate_behavioral_min_samples
                and abs(behavioral_score - 0.5) > config.gate_behavioral_deadband
            ):
                ew = config.gate_explicit_weight
                bw = config.gate_behavioral_weight
                credibility = (ew * cred_eff + bw * behavioral_score) / (ew + bw)
                behavioral_driven = True

        mid = config.gate_cred_mid
        p = max(0.0, min((mid - credibility) / mid, config.gate_cred_max_p))
        if p <= 0.0 or rng.random() >= p:
            return None

        if behavioral_driven:
            # Behavioral-driven suppression → MRT/bet-hedge-eligible (spec §5/§9).
            return GateDecision.suppress(
                "low_credibility_class",
                deciding_arm="signaller_credibility",
                metrics={"credibility": credibility, "p": p},
                mrt_eligible=True,
                p_withhold=p,
            )
        return GateDecision.suppress(
            "low_credibility_class",
            deciding_arm="signaller_credibility",
            metrics={"credibility": credibility, "p": p},
        )

    # ── Phase-2 modifiers (spec §5) ── operate on a fire-survivor, not the
    # first-suppressor loop.  Each preserves the decision's id (the linkage key).
    def _arm_cost_tier_router(
        self,
        decision: GateDecision,
        sig: Signature,
        state: dict[str, Any],
        config: Any,
        now: float,  # noqa: ARG002 — uniform Phase-2 modifier signature
        rng: random.Random,  # noqa: ARG002 — uniform Phase-2 modifier signature; deterministic
    ) -> GateDecision:
        """Arm 7 — innate vs adaptive cost-tier routing (spec §5 Arm 7).

        A **Phase-2 modifier** on a fire-survivor: it never suppresses by itself
        except in ``silent`` mode.  A fire-survivor routes to **Tier-2** (the
        full 32B) when it is

        * ``high`` (a strong stimulus always earns the full model), or
        * ``correlation_found`` (cross-domain evidence), or
        * a **persistent single** — its per-``state_key`` ``cost_tier_memory``
          ``count >= COST_TIER_PERSISTENCE_COUNT`` (a recurring single channel
          has earned escalation), or
        * **Tier-2-earned** (``earned_tier2`` learned offline from feedback).

        Otherwise it is a **Tier-1** candidate (a one-off single+medium):

        * ``TIER1_MODE == "note"`` → ``DOWNGRADE(tier=1, "cost_tier_downgrade")``
          — a templated note still published on ``augur.consilium.advice`` with
          ``tier=1`` so feedback/reflection observe it;
        * ``TIER1_MODE == "silent"`` → ``SUPPRESS("cost_tier_downgrade_silent")``
          — a gate non-delivery (logged per invariant A).

        The decision ``id`` is preserved across the conversion (spec §3/§6/§9).
        """
        if not config.gate_cost_tier_enabled:
            return decision

        mem = state.get("cost_tier_memory") or {}
        count = int(mem.get("count", 0) or 0)
        earned_tier2 = bool(mem.get("earned_tier2", False))

        tier2 = (
            sig.severity == "high"
            or sig.correlation_found
            or count >= config.gate_cost_tier_persistence_count
            or earned_tier2
        )
        if tier2:
            return replace(decision, tier=2)

        # Tier-1 candidate: route per TIER1_MODE, preserving the decision id.
        metrics = {"count": count, "earned_tier2": earned_tier2}
        if config.gate_tier1_mode == "silent":
            return GateDecision.suppress(
                "cost_tier_downgrade_silent",
                deciding_arm="cost_tier_router",
                metrics=metrics,
                id=decision.id,
            )
        return GateDecision.downgrade(
            "cost_tier_downgrade",
            deciding_arm="cost_tier_router",
            metrics=metrics,
            tier=1,
            id=decision.id,
        )

    def _arm_anti_starvation_release(
        self,
        decision: GateDecision,
        sig: Signature,  # noqa: ARG002 — uniform Phase-2 modifier signature
        state: dict[str, Any],
        config: Any,
        now: float,
        rng: random.Random,  # noqa: ARG002 — uniform Phase-2 modifier signature; deterministic
    ) -> GateDecision:
        """Arm 9 — anti-starvation safety override (spec §5 Arm 9, invariant D).

        A **Phase-2 modifier** evaluated **before** bet-hedge (Arm 8): it acts
        **only** on a still-``suppress`` decision whose channel is *starved*.  A
        channel is starved when its ``channel_stats`` shows either bound passed:

        * ``consecutive_suppressions >= MAX_CONSECUTIVE_SUPPRESSIONS`` (a hot
          channel saturated with suppressions), or
        * ``now - suppression_streak_started_ts > MAX_CHANNEL_SILENCE_S`` (a
          sparse stream that never reaches the count bound is still released by
          the time bound).

        In either case the suppression is **deterministically** un-suppressed to
        ``FIRE("anti_starvation_release")`` — never a probe, consuming no ``rng``
        draw — **short-circuiting Arm 8** (so a starved behavioral-driven
        suppress is released, not bet-hedged).  The decision ``id`` (linkage key)
        is preserved across the conversion (spec §3/§6/§9).

        No *trackable* channel can be silenced indefinitely (invariant D); a
        channel with no ``channel_stats`` (untrackable / brand-new at cap) is
        outside D's scope and handled by the cap-fail-open path in ``evaluate``.
        """
        if not config.gate_anti_starvation_enabled:
            return decision
        # Only a still-standing suppress is releasable; a fire-survivor or a
        # downgrade is untouched.
        if decision.action != "suppress":
            return decision

        stats = state.get("channel_stats") or {}
        consecutive = int(stats.get("consecutive_suppressions", 0) or 0)
        streak_started = stats.get("suppression_streak_started_ts")

        by_count = consecutive >= config.gate_max_consecutive_suppressions
        silence_s = now - float(streak_started) if streak_started is not None else 0.0
        by_time = (
            streak_started is not None and silence_s > config.gate_max_channel_silence_s
        )
        if not (by_count or by_time):
            return decision

        return GateDecision.fire(
            "anti_starvation_release",
            deciding_arm="anti_starvation_release",
            metrics={"consecutive": consecutive, "silence_s": silence_s},
            id=decision.id,
        )

    def _arm_bet_hedge_override(
        self,
        decision: GateDecision,
        sig: Signature,
        state: dict[str, Any],  # noqa: ARG002 — uniform Phase-2 modifier signature
        config: Any,
        now: float,  # noqa: ARG002 — uniform Phase-2 modifier signature
        rng: random.Random,
    ) -> GateDecision:
        """Arm 8 — bet-hedging = MRT randomization (spec §5 Arm 8 + §9).

        A **Phase-2 modifier** evaluated **after** anti-starvation (Arm 9): it
        acts **only** on a still-``suppress`` decision that is behavioral-driven
        — the eligible band (the credibility arm's behavioral fusion, single +
        medium → ``mrt_eligible``).  With known probability ``ε``
        (``BET_HEDGE_EPSILON``) it flips the withheld suppression to a
        ``FIRE(probe=True, withheld_reason=<original reason>)`` at Tier-2 — the
        genuine action-randomization that makes the MRT identifiable.

        Whenever Arm 8 **considers** an eligible decision it stamps the **known**
        randomization probabilities (``p_fire = ε``, ``p_withhold = 1 - ε``,
        ``mrt_eligible = True``) on the resulting decision — **regardless of the
        flip outcome** — so both the probe-fired and the withheld arm are
        inverse-probability-weightable offline even under a dynamic ``ε`` (§4).
        The decision ``id`` (linkage key) is preserved across the flip.

        Never touches an exempt fire (it never reaches a suppress) or an
        ``anti_starvation_release`` (which is a deterministic FIRE produced by
        Arm 9 before this arm — caught by the ``action != "suppress"`` guard).
        """
        if not config.gate_bet_hedge_enabled:
            return decision
        # Only a still-standing, behavioral-driven (mrt_eligible) suppress in the
        # eligible band is randomized.  A fire-survivor, an anti-starvation FIRE,
        # or a non-behavioral (explicit-only) suppress are all left untouched.
        if decision.action != "suppress" or not decision.mrt_eligible:
            return decision
        if sig.path != "single" or sig.severity != "medium":
            return decision

        epsilon = config.gate_bet_hedge_epsilon
        p_fire = epsilon
        p_withhold = 1.0 - epsilon

        if rng.random() < epsilon:
            # Flip: deliver the measured Tier-2 intervention as a probe, carrying
            # the original suppress reason as withheld_reason for the MRT join.
            return GateDecision.fire(
                "bet_hedge_probe",
                deciding_arm="bet_hedge_override",
                metrics={**decision.metrics, "epsilon": epsilon},
                tier=2,
                id=decision.id,
                probe=True,
                withheld_reason=decision.reason,
                mrt_eligible=True,
                p_fire=p_fire,
                p_withhold=p_withhold,
            )

        # No flip: the suppression stands, but stamp the known probabilities so
        # the withheld arm is IPW-able against probe-fired siblings (§4/§9).
        return replace(
            decision, mrt_eligible=True, p_fire=p_fire, p_withhold=p_withhold
        )

    # ── Writers (spec §4/§5/§6) ── run AFTER a decision; evaluate stays
    # read-only.  All Redis I/O goes through the injected PersistenceManager.
    def record_delivery_success(
        self,
        signature: Signature,
        pm: Any,
        now: float,
        *,
        decision: GateDecision,
        tier: int | None,
        audit_only: bool = False,
    ) -> None:
        """Advance gate state after a SUCCESSFUL publish (spec §4).

        Called only once the advice (or Tier-1 note) has actually been published,
        so it can never record a delivery that did not happen.  Three modes:

        * ``audit_only=True`` (exempt) — writes **solely** an audit emission
          entry: no ``h``, no ``channel_stats``, no ``observed``, no advice-rate,
          no gating-visible emission (invariant B — exempt path touches no gate
          state).
        * ``decision.probe=True`` (bet-hedge) — appends an emission flagged
          ``probe=True`` (which gating-visible readers ignore) and **nothing
          else**: it does not advance ``h``/advice-rate, reset starvation, or
          append ``observed`` — so a probe neither perturbs the state the MRT
          compares nor counts as a normal delivery.
        * otherwise (normal delivery) — advances habituation ``h`` (EWMA toward
          1, or dishabituation toward 0 for a HIGH), the advice-rate EWMA, the
          per-channel ``cost_tier_memory``; resets ``channel_stats`` (clears the
          suppression streak, stamps ``last_delivery_ts``); and appends both a
          gating-visible emission (carrying ``mrt_eligible``/``p_fire`` for IPW)
          and an ``observed`` value record (novelty depends on it).
        """
        if audit_only:
            pm.save_emission(
                self._emission_record(signature, decision, now, tier, audit_only=True)
            )
            return

        if decision.probe:
            pm.save_emission(
                self._emission_record(signature, decision, now, tier, audit_only=False)
            )
            return

        if signature.ungateable:
            # §5 missing-entity rule: an ungateable event writes ONLY the global
            # emission (for global-burden / refractory tracking) — no per-channel
            # state (no observed/h/advice-rate/channel_stats/cost_tier on a bogus
            # single:{domain}:? key).
            pm.save_emission(
                self._emission_record(signature, decision, now, tier, audit_only=False)
            )
            return

        # ── Normal delivery: advance all online state ──
        pm.save_emission(
            self._emission_record(signature, decision, now, tier, audit_only=False)
        )
        pm.save_observed(
            {
                "ts": now,
                "state_key": signature.state_key,
                "value": signature.value,
                "severity": signature.severity,
            }
        )
        self._advance_habituation(signature, pm, now)
        self._advance_advice_rate(pm, now)
        self._reset_channel_stats(signature, pm, now)
        self._advance_cost_tier_memory(signature, pm, now, tier)

    def record_suppression(
        self,
        decision: GateDecision,
        signature: Signature,
        pm: Any,
        now: float,
    ) -> bool:
        """Write the authoritative silence record + advance accumulators (spec §4).

        The silence write is **authoritative** (invariant A): it must persist or
        the caller fails open and FIRES.  Returns ``True`` once the silence
        record is committed, ``False`` if that write fails (raises) — in which
        case no accumulators are advanced.  On success it also advances the
        evidence a suppressed event legitimately feeds: the reservoir count, the
        observed-value window, and ``channel_stats`` (so a starved channel is
        eventually released by anti-starvation, invariant D).
        """
        record = {
            "ts": now,
            "decision_id": decision.id,
            "state_key": signature.state_key,
            "domain": signature.domain,
            "entity": signature.entity,
            "severity": signature.severity,
            "arm": decision.deciding_arm,
            "reason": decision.reason,
            "metrics": decision.metrics,
            "mrt_eligible": decision.mrt_eligible,
            "p_withhold": decision.p_withhold,
        }
        try:
            pm.save_silence_record(record)
        except Exception:
            log.error("gate silence write failed for %s", signature.state_key)
            return False

        if signature.ungateable:
            # §5 missing-entity rule: the authoritative silence (invariant A) is
            # written, but NO per-channel accumulators advance — an ungateable
            # event has no stable channel to track (no bogus single:{domain}:?
            # reservoir/observed/channel_stats).
            return True

        self._advance_reservoir(signature, pm, now)
        pm.save_observed(
            {
                "ts": now,
                "state_key": signature.state_key,
                "value": signature.value,
                "severity": signature.severity,
            }
        )
        self._bump_suppression_stats(signature, pm, now)
        return True

    def record_busy_skip(self, signature: Signature, pm: Any, now: float) -> bool:
        """Record an ordinary fire dropped because the lock was held (spec §4).

        Bumps ``channel_stats`` (so a hot trackable channel repeatedly
        busy-skipped is eventually released by anti-starvation, invariant D) and
        writes a best-effort ``delivery_failure``.  Returns ``False`` if the
        channel is **untrackable** — a new ``state_key`` already at
        ``MAX_GATE_STATE_KEYS`` so no ``channel_stats`` can be created (outside
        D's trackable scope); the caller then logs a separate delivery_failure.

        An **ungateable** signature (missing/``"?"``/empty entity, spec §5) has
        no stable per-channel ``state_key`` to track, so it writes ONLY the
        best-effort global ``delivery_failure`` — never ``channel_stats`` (no
        ``can_track_gate_state`` probe, no ``_bump_suppression_stats``) — so the
        busy-skip path can never create the bogus ``single:{domain}:?`` channel
        the missing-entity rule eliminates everywhere else.  Returns ``True``
        (handled) so the caller does not double-log a second delivery_failure.
        """
        if signature.ungateable:
            pm.save_delivery_failure(signature, "advisor_busy_skipped", now, "")
            return True
        if not pm.can_track_gate_state(_CHANNEL_STATS_KEY, signature.state_key):
            return False
        self._bump_suppression_stats(signature, pm, now)
        pm.save_delivery_failure(signature, "advisor_busy_skipped", now, "")
        return True

    def still_starved(self, signature: Signature, pm: Any, now: float) -> bool:
        """Re-check (read-only) whether ``signature``'s channel is still starved.

        Used under the lock before an ``anti_starvation_release`` fires, to skip
        a release a concurrent delivery already served.  **Fail-open guarded**:
        returns the safe default ``True`` ("assume starved → fire") on **any**
        read error, so a corrupt/unavailable read can never drop a release
        (invariant C/D, spec §3/§4).
        """
        try:
            stats = pm.load_channel_stats(signature.state_key) or {}
            consecutive = int(stats.get("consecutive_suppressions", 0) or 0)
            streak_started = stats.get("suppression_streak_started_ts")
            by_count = consecutive >= self._config.gate_max_consecutive_suppressions
            silence_s = (
                now - float(streak_started) if streak_started is not None else 0.0
            )
            by_time = (
                streak_started is not None
                and silence_s > self._config.gate_max_channel_silence_s
            )
            return bool(by_count or by_time)
        except Exception:
            log.error("still_starved read failed for %s; assuming starved", signature)
            return True

    # ── Writer helpers (private) ──
    def _emission_record(
        self,
        signature: Signature,
        decision: GateDecision,
        now: float,
        tier: int | None,
        *,
        audit_only: bool,
    ) -> dict[str, Any]:
        """Build a gate emission record (spec §6 emissions schema).

        ``audit_only`` is True only for an exempt audit emission; a probe or a
        normal delivery carries ``audit_only=False``.  Gating-visible readers
        (refractory/pressure/duplicate/habituation) ignore any row whose
        ``probe`` or ``audit_only`` is True.
        """
        return {
            "ts": now,
            "decision_id": decision.id,
            "state_key": signature.state_key,
            "severity": signature.severity,
            "tier": tier,
            "probe": bool(decision.probe),
            "audit_only": bool(audit_only),
            "withheld_reason": decision.withheld_reason,
            "mrt_eligible": decision.mrt_eligible,
            "p_fire": decision.p_fire,
        }

    def _advance_habituation(self, signature: Signature, pm: Any, now: float) -> None:
        """Advance per-channel habituation ``h`` after a non-probe delivery.

        A normal delivery decays the stored ``h`` to ``now`` then EWMAs it toward
        1 (``alpha`` = ``gate_habituation_alpha``): repeated advice on a channel
        raises ``h`` so the habituation arm eventually holds it.  A **HIGH**
        delivery instead **dishabituates** — it resets ``h`` toward 0 (spec §5
        HIGH bypass: a strong stimulus restores responsiveness on its channel).
        """
        hab = pm.load_habituation(signature.state_key) or {}
        h = float(hab.get("h", 0.0) or 0.0)
        count = int(hab.get("count", 0) or 0)

        if signature.severity == "high":
            # Dishabituation: a strong stimulus restores the channel.
            new_h = 0.0
        else:
            last_ts = float(hab.get("last_event_ts", now) or now)
            dt = max(0.0, now - last_ts)
            decayed = h * math.exp(-dt / self._config.gate_habituation_tau_s)
            alpha = self._config.gate_habituation_alpha
            new_h = alpha * 1.0 + (1.0 - alpha) * decayed

        pm.save_habituation(
            signature.state_key,
            {"h": new_h, "last_event_ts": now, "count": count + 1},
        )

    def _advance_advice_rate(self, pm: Any, now: float) -> None:
        """Advance the global advice-rate EWMA after a non-probe delivery.

        Each delivery is a unit impulse EWMA'd with ``gate_pressure_alpha`` — the
        refractory arm's pressure sub-reason reads ``rate_ewma`` as recent
        advice volume.
        """
        rate = pm.load_advice_rate() or {}
        prev = float(rate.get("rate_ewma", 0.0) or 0.0)
        alpha = self._config.gate_pressure_alpha
        new_rate = alpha * 1.0 + (1.0 - alpha) * prev
        pm.save_advice_rate({"rate_ewma": new_rate, "last_ts": now})

    def _reset_channel_stats(self, signature: Signature, pm: Any, now: float) -> None:
        """Clear the suppression streak on a delivery (spec §6, invariant D).

        A delivery resets ``consecutive_suppressions`` to 0, clears
        ``suppression_streak_started_ts``, and stamps ``last_delivery_ts`` — so
        anti-starvation's streak measurement restarts from this delivery.
        """
        stats = pm.load_channel_stats(signature.state_key) or {}
        stats["seen"] = int(stats.get("seen", 0) or 0) + 1
        stats["consecutive_suppressions"] = 0
        stats["suppression_streak_started_ts"] = None
        stats["last_delivery_ts"] = now
        stats["last_ts"] = now
        pm.save_channel_stats(signature.state_key, stats)

    def _advance_cost_tier_memory(
        self, signature: Signature, pm: Any, now: float, tier: int | None
    ) -> None:
        """Advance per-channel ``cost_tier_memory`` online (spec §5 Arm 7).

        ``count`` tracks recurrence (a persistent single earns Tier-2);
        ``earned_tier2`` latches True once the channel was served at Tier-2 —
        both consumed by Arm 7's routing.
        """
        mem = pm.load_cost_tier_memory(signature.state_key) or {}
        mem["count"] = int(mem.get("count", 0) or 0) + 1
        if tier == 2:
            mem["earned_tier2"] = True
        else:
            mem.setdefault("earned_tier2", False)
        mem["last_ts"] = now
        pm.save_cost_tier_memory(signature.state_key, mem)

    def _advance_reservoir(self, signature: Signature, pm: Any, now: float) -> None:
        """Advance the per-channel evidence reservoir on a suppression (spec §5).

        A suppressed single+medium event still legitimately accumulates evidence
        (+1, leaking by ``exp(-dt/tau)``) so the channel eventually reaches the
        ON count and commits.  The latch stays ``suppressing`` until Arm 5
        commits it.
        """
        res = pm.load_reservoir(signature.state_key) or {}
        count = float(res.get("count", 0.0) or 0.0)
        last_ts = float(res.get("last_ts", now) or now)
        dt = max(0.0, now - last_ts)
        leaked = count * math.exp(-dt / self._config.gate_reservoir_leak_tau_s)
        pm.save_reservoir(
            signature.state_key,
            {
                "count": leaked + 1.0,
                "last_ts": now,
                "suppressing": bool(res.get("suppressing", True)),
            },
        )

    def _bump_suppression_stats(
        self, signature: Signature, pm: Any, now: float
    ) -> None:
        """Bump ``channel_stats`` on a suppression / busy-skip (spec §6, inv. D).

        Increments ``consecutive_suppressions`` and stamps
        ``suppression_streak_started_ts`` **only on the first** suppression of a
        streak (a delivery clears it), so anti-starvation's count + time bounds
        measure an unbroken silence run.
        """
        stats = pm.load_channel_stats(signature.state_key) or {}
        stats["seen"] = int(stats.get("seen", 0) or 0) + 1
        stats["consecutive_suppressions"] = (
            int(stats.get("consecutive_suppressions", 0) or 0) + 1
        )
        if stats.get("suppression_streak_started_ts") is None:
            stats["suppression_streak_started_ts"] = now
        stats["last_ts"] = now
        pm.save_channel_stats(signature.state_key, stats)
