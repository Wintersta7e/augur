"""Imperator III dialogue engine — the hybrid turn loop."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from imperator.dialogue import context as C, persona


@dataclass
class DialogueTurn:
    reply: str
    intent: dict | None = None
    pending: dict | None = None
    applied: dict | None = None
    needs_clarification: bool = False
    error: str | None = None


async def query_dialogue_ollama(prompt: str, system: str, client, cfg) -> str:
    resp = await client.post(
        f"{cfg.ollama_url}/api/generate",
        json={
            "model": cfg.dialogue_model,
            "prompt": prompt,
            "system": system,
            "stream": False,
            "format": "json",
            "options": {
                "temperature": cfg.dialogue_temperature,
                "num_predict": cfg.dialogue_num_predict,
            },
        },
        timeout=cfg.ollama_timeout,
    )
    resp.raise_for_status()
    text = resp.json().get("response", "").strip()
    if not text:
        raise ValueError("Empty response from Ollama")
    return text


def _parse(raw: str) -> dict:
    obj = json.loads(raw)
    if not isinstance(obj, dict) or "reply" not in obj:
        raise ValueError("LLM output missing 'reply'")
    return obj


QueryFn = Callable[[str, str, Any, Any], Awaitable[str]]


async def handle_turn(
    session_id: str,
    user_text: str,
    *,
    pm,
    nc,
    http_client,
    cfg,
    query_fn: QueryFn = query_dialogue_ollama,
) -> DialogueTurn:
    ctx = C.assemble(pm, now=time.time(), cfg=cfg)
    register = persona.register_for_salience(ctx.salience)
    system = persona.build_system_prompt(register, C.render(ctx, cfg), cfg)

    # NOTE: P2-6 inserts the "resolve pending confirmation first" branch here.

    try:
        raw = await query_fn(user_text, system, http_client, cfg)
        obj = _parse(raw)
    except Exception as exc:  # fail-truthful: never guess a mutation
        return DialogueTurn(
            reply="I can't reason about that right now.", error=str(exc)
        )

    if obj.get("needs_clarification"):
        turn = DialogueTurn(
            reply=obj.get("question") or obj["reply"], needs_clarification=True
        )
        pm.save_dialogue_turn(
            {
                "ts": time.time(),
                "session_id": session_id,
                "user_text": user_text,
                "reply": turn.reply,
            }
        )
        return turn

    intent = obj.get("intent")
    if intent:
        # Write-path is wired in P2-6. Until then, acknowledge without mutating.
        return DialogueTurn(reply=obj["reply"], intent=intent, needs_clarification=True)

    turn = DialogueTurn(reply=obj["reply"])
    pm.save_dialogue_turn(
        {
            "ts": time.time(),
            "session_id": session_id,
            "user_text": user_text,
            "reply": turn.reply,
        }
    )
    return turn
