"""LLM reasoning for Imperator II: self-model -> candidate proposals. LLM behind an injected callable."""

from __future__ import annotations

import json
import time
from typing import Any, Callable

import httpx

from imperator import proposals as P

_VALID_KINDS = {
    "escalation_rule",
    "prompt_strategy",
    "sigma",
    "gate_calibration",
    "observe_more",
    "code",
    "structural",
}


class ReasonerError(Exception):
    def __init__(self, reason: str, detail: str = ""):
        super().__init__(f"{reason}: {detail}")
        self.reason = reason


def build_reasoning_prompt(self_model: dict) -> str:
    def v(key):
        cell = self_model.get(key) or {}
        return cell.get("value") if cell.get("fresh") else None

    blind = (self_model.get("blind_spots") or {}).get("value") or []
    lines = [
        "You are Imperator, improving the Augur system for its single user.",
        f"competence={v('competence')} precision={v('precision')} utility={v('utility')}",
        f"recent_self_tuning={(self_model.get('recent_self_tuning') or {}).get('value')}",
        "Addressable weaknesses (blind_spots):",
    ]
    lines += [
        f"  - {b.get('kind')}: {b.get('detail')} (evidence={b.get('evidence')})"
        for b in blind
    ]
    lines += [
        "Output ONLY a STRICT JSON array; each item:",
        '{"kind": escalation_rule|prompt_strategy|sigma|gate_calibration|observe_more|code|structural,',
        ' "target": "<rule_key|domain|channel|path>", "action": {...},',
        ' "rationale": "why this serves the user", "rank": <1=most urgent>}',
        'escalation_rule action={"target":"LOW|MEDIUM|HIGH"}; prompt_strategy action={"domain":..,"text":<full prompt>}.',
        "Use code/structural for source/architecture changes you cannot make yourself; these are LOGGED for human/Conscientia review, never auto-applied.",
    ]
    return "\n".join(str(x) for x in lines)


async def query_imperator_ollama(
    prompt: str, client: httpx.AsyncClient, config
) -> tuple[str, float]:
    t0 = time.monotonic()
    resp = await client.post(
        f"{config.ollama_url}/api/generate",
        json={
            "model": config.ollama_model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.7,
                "num_predict": config.imperator_ii_num_predict,
            },
        },
        timeout=config.ollama_timeout,
    )
    resp.raise_for_status()
    text = resp.json().get("response", "").strip()
    if not text:
        raise ValueError("Empty response from Ollama")
    return text, (time.monotonic() - t0) * 1000.0


def _array_from(val: Any) -> list | None:
    """The proposals array from a decoded value: a bare list, or the first
    array-valued field of an object wrapper like {"proposals":[...]}."""
    if isinstance(val, list):
        return val
    if isinstance(val, dict):
        return next((v for v in val.values() if isinstance(v, list)), None)
    return None


def _extract_json_array(text: str) -> list | None:
    """First top-level proposals array in text, ignoring trailing prose.

    Accepts a bare top-level array OR an object whose array-valued field holds
    the proposals (models often wrap them as {"proposals":[...]}/{"items":[...]}).
    Tries to parse the whole (clean) output first, else raw_decode from the
    first '[' so a valid array followed by prose containing brackets still
    parses (a greedy regex would over-capture to a later ']' and fail).
    """
    try:
        return _array_from(json.loads(text.strip()))
    except json.JSONDecodeError:
        pass
    start = text.find("[")
    if start == -1:
        return None
    try:
        val, _ = json.JSONDecoder().raw_decode(text[start:])
    except json.JSONDecodeError:
        return None
    return val if isinstance(val, list) else None


def _valid_action(kind: str, action: dict) -> bool:
    """Kind-specific action shape check (LLM actions are otherwise free-form)."""
    if kind == "prompt_strategy":
        return isinstance(action.get("text"), str) and bool(action.get("text"))
    if kind == "escalation_rule":
        return isinstance(action.get("target"), str) or "window" in action
    return (
        True  # code/structural/sigma/... carry free-form actions (never auto-applied)
    )


def parse_proposals(text: str, now: float, max_n: int) -> list[dict]:
    items = _extract_json_array(text)
    if items is None:
        return []
    out: list[dict] = []
    for it in items:
        if len(out) >= max_n:  # cap on VALID survivors, not raw candidates
            break
        if not isinstance(it, dict):
            continue
        kind, target, action = it.get("kind"), it.get("target"), it.get("action")
        if (
            kind not in _VALID_KINDS
            or not isinstance(target, str)
            or not target
            or not isinstance(action, dict)
            or not _valid_action(kind, action)
        ):
            continue
        try:
            rank = int(it.get("rank", 100))
        except (TypeError, ValueError):
            rank = 100
        out.append(
            P.make_proposal(
                kind=kind,
                target=target,
                action=action,
                rationale=str(it.get("rationale", "")),
                rank=rank,
                now=now,
            )
        )
    return out


async def generate_proposals(
    self_model: dict,
    *,
    client,
    config,
    now: float,
    query_ollama_fn: Callable[..., Any] = query_imperator_ollama,
) -> list[dict]:
    try:
        prompt = build_reasoning_prompt(self_model)
    except Exception as exc:
        raise ReasonerError("prompt_build_failed", str(exc))
    try:
        text, _ = await query_ollama_fn(prompt, client, config)
    except httpx.ConnectError as exc:
        raise ReasonerError("ollama_unreachable", str(exc))
    except (httpx.TimeoutException, TimeoutError) as exc:
        raise ReasonerError("ollama_timeout", str(exc))
    except Exception as exc:
        raise ReasonerError("ollama_error", str(exc))
    return parse_proposals(text, now, config.imperator_ii_max_proposals_per_cycle)
