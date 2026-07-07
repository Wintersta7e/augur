"""Pure verdict functions. No Redis, no NATS, no LLM — deterministic string
and shape checks only (the recorded constraint: no second judge LLM). All
screens self-gate on config so call sites invoke them unconditionally;
disabled screens pass everything (kill-switch parity, invariant C5)."""

from __future__ import annotations

import time
from dataclasses import dataclass

from conscientia import charter

# Appended to the advice prompt for the single corrective regeneration
# attempt (spec D3). Part of the mechanism, not policy — deliberately not
# configurable.
CORRECTIVE_SUFFIX = (
    "\n\nIMPORTANT: your previous draft was refused because it contained the "
    "forbidden phrasing {matched}. Restate the observation and advice with a "
    "neutral, non-directive tone: no break/rest suggestions, no fatigue or "
    "distraction attributions, no self-references as an AI."
)


@dataclass(frozen=True)
class Verdict:
    ok: bool
    code: str | None = None
    detail: str | None = None
    principle: str | None = None


def _match(text: str, patterns: tuple[str, ...]) -> str | None:
    low = text.lower()
    for pat in patterns:
        if pat and pat.lower() in low:
            return pat
    return None


def screen_advice_text(text, cfg) -> Verdict:
    """Valence screen for any outgoing user-facing text (advice, notes)."""
    if not getattr(cfg, "conscientia_enabled", True) or not getattr(
        cfg, "conscientia_output_screen_enabled", True
    ):
        return Verdict(True)
    if not isinstance(text, str) or not text:
        return Verdict(True)
    hit = _match(text, charter.output_patterns(cfg))
    if hit is None:
        return Verdict(True)
    return Verdict(False, "forbidden_valence", f"matched {hit!r}", "restraint")


def screen_taught_content(rationale, rule_key, cfg) -> Verdict:
    """Valence screen for user-taught free text (teach-time and inject-time)."""
    if not getattr(cfg, "conscientia_enabled", True) or not getattr(
        cfg, "conscientia_teach_screen_enabled", True
    ):
        return Verdict(True)
    pats = charter.teach_patterns(cfg)
    for field in (rationale, rule_key):
        if isinstance(field, str) and field:
            hit = _match(field, pats)
            if hit is not None:
                return Verdict(
                    False, "forbidden_valence", f"matched {hit!r}", "restraint"
                )
    return Verdict(True)


def screen_proposal(p: dict, cfg) -> Verdict:
    """Deterministic charter check in front of SAFE-class applies (spec S4).

    Defense in depth over the kind/klass machinery: refuses forged klasses,
    protected-surface targets, and charter-violating prompt text. Gated
    kinds are refused here too (they can never be safe), though the apply
    paths already exclude them independently (invariant I1)."""
    if not getattr(cfg, "conscientia_enabled", True) or not getattr(
        cfg, "conscientia_proposal_screen_enabled", True
    ):
        return Verdict(True)
    from imperator import proposals as P  # local import: avoid cycle at module load

    kind = p.get("kind", "")
    # normalize_klass mutates its argument in place (and returns it), so pass
    # a shallow copy — the caller's proposal dict must never be mutated by a
    # pure screen.
    if P.normalize_klass(dict(p)).get("klass") != "safe":
        return Verdict(
            False, "not_safe_kind", f"kind {kind!r} is not a safe class", "containment"
        )
    target = str(p.get("target") or "")
    low = target.lower()
    for prefix in charter.PROTECTED_SURFACES:
        if low.startswith(prefix.lower()):
            return Verdict(
                False,
                "protected_surface",
                f"target {target!r} is under protected surface {prefix!r}",
                "containment",
            )
    if kind == "prompt_strategy":
        text = (p.get("action") or {}).get("text", "")
        hit = (
            _match(text, charter.output_patterns(cfg))
            if isinstance(text, str)
            else None
        )
        if hit is not None:
            return Verdict(False, "forbidden_valence", f"matched {hit!r}", "restraint")
    return Verdict(True)


def make_violation(
    surface: str,
    code: str,
    detail: str,
    principle: str,
    *,
    decision_id: str | None = None,
    state_key: str | None = None,
    domain: str | None = None,
    entity: str | None = None,
    session_id: str | None = None,
    regenerated: bool = False,
    now: float | None = None,
) -> dict:
    """Violation record for augur:conscientia:violations + the NATS event."""
    return {
        "surface": surface,
        "code": code,
        "detail": detail,
        "principle": principle,
        "decision_id": decision_id,
        "state_key": state_key,
        "domain": domain,
        "entity": entity,
        "session_id": session_id,
        "regenerated": regenerated,
        "ts": time.time() if now is None else now,
        "charter_version": charter.CHARTER_VERSION,
    }
