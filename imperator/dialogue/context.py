"""Read-only assembler of Augur's current state for the dialogue engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class DialogueContext:
    salience: float = 0.0
    auspices: dict[str, Any] = field(default_factory=dict)
    self_model: dict[str, Any] = field(default_factory=dict)
    recent_suppressions: list[dict] = field(default_factory=list)
    recent_emissions: list[dict] = field(default_factory=list)
    directives: list[dict] = field(default_factory=list)
    taught_facts: list[dict] = field(default_factory=list)
    recent_turns: list[dict] = field(default_factory=list)
    focused_app: str | None = None


def assemble(pm, now: float, cfg) -> DialogueContext:
    auspices = pm.load_auspices() or {}
    salience = float((auspices.get("salience") or {}).get("value", 0.0) or 0.0)
    return DialogueContext(
        salience=salience,
        auspices=auspices,
        self_model=pm.load_self_model() or {},
        recent_suppressions=pm.load_silence_records(limit=10) or [],
        recent_emissions=pm.load_emissions(limit=10) or [],
        directives=(getattr(pm, "load_dialogue_directives", lambda: [])() or []),
        taught_facts=(getattr(pm, "load_taught_facts", lambda: [])() or []),
        recent_turns=pm.load_dialogue_log(limit=cfg.dialogue_context_max_turns) or [],
        # The live focused app (same source the Limen gate reads) so the LLM can
        # name it when the user asks to "stay quiet in this app": a
        # teach_context_directive's predicate.match is filled from this ground
        # truth, not guessed by the model (spec §7.2). getattr-guarded for stub
        # PMs that predate the field.
        focused_app=(getattr(pm, "load_focused_app", lambda: None)() or None),
    )


def render(ctx: DialogueContext, cfg) -> str:
    sm = ctx.self_model
    lines = [
        f"salience={ctx.salience:.2f}",
        f"precision={(sm.get('precision') or {}).get('value')}",
        f"utility={(sm.get('utility') or {}).get('value')}",
        f"blind_spots={[b.get('kind') for b in (sm.get('blind_spots') or {}).get('value') or []]}",
    ]
    if ctx.focused_app:
        lines.append(f"focused_app={ctx.focused_app}")
    if ctx.recent_suppressions:
        lines.append("recent_suppressions:")
        for s in ctx.recent_suppressions[:5]:
            lines.append(
                f"  - {s.get('state_key')} arm={s.get('arm')} reason={s.get('reason')}"
            )
    if ctx.directives:
        lines.append(
            f"active_directives={[d.get('directive_id') for d in ctx.directives]}"
        )
    if ctx.taught_facts:
        lines.append(f"taught_facts={[f.get('memory_id') for f in ctx.taught_facts]}")
    if ctx.recent_turns:
        lines.append("recent_conversation:")
        for t in ctx.recent_turns[:5]:
            lines.append(f"  - you said: {t.get('user_text', '')[:120]}")
    block = "\n".join(lines)
    char_budget = cfg.dialogue_context_token_budget * 4  # ~4 chars/token
    return block[:char_budget]
