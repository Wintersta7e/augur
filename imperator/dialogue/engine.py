"""Imperator III dialogue engine — the hybrid turn loop."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from imperator.dialogue import context as C, intents as I, persona, router as R


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


async def _publish(nc, subject: str, payload: dict) -> None:
    """Fire a dialogue event on NATS. No try/except here: a publish failure
    is an infra failure (same class as Redis down), and per the Task 6/7
    design every other faculty's nc.publish call is unguarded too -- it
    propagates out of handle_turn for the surface (console per-turn
    try/except, MCP @_tool_safe) to catch and report truthfully. Only the
    LLM-call/parse path below fails soft inside the engine."""
    if nc is not None:
        await nc.publish(subject, json.dumps(payload).encode())


async def _handle_undo(
    session_id: str, user_text: str, base_reply: str, *, pm, nc, cfg
) -> DialogueTurn:
    """Reverse the most recently audited dialogue change.

    Goes through router.apply_undo (Task 13) rather than a hand-built
    build_inverse + apply.apply_proposal call: apply_undo pre-checks the
    inverse against the CURRENT apply-layer bounds so a rollback anchor
    recorded before those bounds tightened (e.g. a stale prior_sigma/prior
    floor) is reported as a distinct, truthful "blocked" outcome instead of
    silently failing closed with the same generic status as any other
    rejected proposal. All four apply_undo outcomes (unavailable/blocked/
    logged/applied) get their own reply text below -- never a blanket
    "Reversed." regardless of what actually happened.
    """
    audit = pm.load_dialogue_audit(limit=1)
    if not audit or not audit[0].get("proposal"):
        # Either no audit trail at all, or the most recent entry is itself a
        # prior undo attempt that never produced a proposal (unavailable/
        # never-applied) -- router.build_inverse requires a real proposal
        # dict to invert, so a chained "undo that" on such an entry has
        # nothing to invert either. Same truthful reply for both.
        return DialogueTurn(reply="There's nothing recent to undo.")
    prior = audit[0]
    out = R.apply_undo(prior, pm=pm, cfg=cfg, session_id=session_id)
    proposal = out["proposal"]
    record = {
        "ts": time.time(),
        "session_id": session_id,
        "kind": (proposal or {}).get("kind", prior.get("kind")),
        "target": (proposal or {}).get("target", prior.get("target")),
        "proposal": proposal,
        "status": out["status"],
        "undo": True,
        # Confirmation provenance (constraint: audit must record WHO/WHAT
        # authorized this write) -- undo has no separate confirm step, so
        # the triggering utterance itself is the provenance.
        "confirming_text": user_text,
    }
    pm.append_dialogue_audit(record)
    await _publish(nc, "augur.imperator.dialogue.applied", record)
    await _publish(
        nc, "augur.imperator.ii.trigger", {"reason": "dialogue", "ts": record["ts"]}
    )
    status_reply = {
        "applied": "Reversed.",
        # Surface router's blocked reason once, without stacking negations
        # ("can't undo -- cannot restore"): strip its "cannot restore: "
        # prefix and fold the rest into one clean sentence. removeprefix is
        # a no-op if the reason wording ever changes -- worst case the full
        # reason shows verbatim, never a wrong message.
        "blocked": (
            "I can't restore that: "
            f"{(out['reason'] or '').removeprefix('cannot restore: ')}."
        ),
        "unavailable": "That change can't be undone automatically.",
        "logged": "I tried to reverse that, but it didn't take.",
    }.get(out["status"], "I couldn't undo that.")
    reply = f"{base_reply}\n{status_reply}" if base_reply else status_reply
    return DialogueTurn(reply=reply, applied=record)


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

    notice = ""  # prefixed to the fresh-turn reply when a pending was dropped
    pending = pm.load_dialogue_pending(session_id)
    if pending is not None:
        if pending.get("tier") == "heavy":
            ok = I.matches_heavy_phrase(user_text, pending.get("confirm_phrase") or "")
        else:
            ok = I.is_affirmative(user_text)
        pm.clear_dialogue_pending(session_id)  # cleared before apply: no double-apply
        if ok:
            applied = R.apply_confirmed(pending, pm=pm, cfg=cfg, session_id=session_id)
            record = {
                "ts": time.time(),
                "session_id": session_id,
                "kind": pending["proposal"]["kind"],
                "target": pending["proposal"].get("target"),
                "proposal": applied["proposal"],
                "status": applied["status"],
                # Confirmation provenance: who confirmed (session_id) and
                # what turn confirmed it (the "yes"/heavy-phrase text).
                "confirming_text": user_text,
            }
            pm.append_dialogue_audit(record)
            await _publish(nc, "augur.imperator.dialogue.applied", record)
            await _publish(
                nc,
                "augur.imperator.ii.trigger",
                {"reason": "dialogue", "ts": record["ts"]},
            )
            reply = (
                f"Done — {applied['echo']} Say 'undo that' to reverse it."
                if applied["status"] == "applied"
                else "I couldn't apply that."
            )
            turn = DialogueTurn(reply=reply, applied=record)
            pm.save_dialogue_turn(
                {
                    "ts": record["ts"],
                    "session_id": session_id,
                    "user_text": user_text,
                    "reply": reply,
                    # Truthful: a confirmed apply can still resolve "logged"
                    # (kill switch off, non-safe klass, apply error) -- the
                    # conversation log must agree with the reply and audit,
                    # not claim success unconditionally.
                    "applied": applied["status"] == "applied",
                }
            )
            return turn
        # Tier/phrase mismatch: fall through and treat user_text as a fresh
        # turn -- but say the pending was dropped rather than silently
        # discarding the user's un-confirmed proposal.
        notice = "(Dropped the pending proposal.) "
    elif I.is_affirmative(user_text):
        # Expired/absent pending on a "yes": don't waste (or be misled by) an
        # LLM call on a bare confirmation gesture with nothing to confirm.
        turn = DialogueTurn(reply="There's nothing pending to confirm.")
        pm.save_dialogue_turn(
            {
                "ts": time.time(),
                "session_id": session_id,
                "user_text": user_text,
                "reply": turn.reply,
            }
        )
        return turn

    try:
        raw = await query_fn(user_text, system, http_client, cfg)
        obj = _parse(raw)
    except Exception as exc:  # fail-truthful: never guess a mutation
        return DialogueTurn(
            reply=notice + "I can't reason about that right now.", error=str(exc)
        )

    if obj.get("needs_clarification"):
        turn = DialogueTurn(
            reply=notice + (obj.get("question") or obj["reply"]),
            needs_clarification=True,
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
        try:
            valid = I.validate_intent(intent)
            if valid["kind"] == "undo":
                turn = await _handle_undo(
                    session_id, user_text, obj["reply"], pm=pm, nc=nc, cfg=cfg
                )
            else:
                new_pending = R.route(valid, ctx, pm=pm, cfg=cfg)
                pm.save_dialogue_pending(
                    session_id, new_pending, ttl=cfg.dialogue_pending_ttl_s
                )
                ask = (
                    f" Confirm with '{new_pending['confirm_phrase']}'."
                    if new_pending["tier"] == "heavy"
                    else " Confirm? (yes)"
                )
                reply = f"{obj['reply']}\n{new_pending['echo']}{ask}"
                turn = DialogueTurn(reply=reply, intent=valid, pending=new_pending)
        except ValueError as exc:
            turn = DialogueTurn(
                reply=f"I didn't quite follow — {exc}", needs_clarification=True
            )
        turn.reply = notice + turn.reply
        pm.save_dialogue_turn(
            {
                "ts": time.time(),
                "session_id": session_id,
                "user_text": user_text,
                "reply": turn.reply,
            }
        )
        return turn

    turn = DialogueTurn(reply=notice + obj["reply"])
    pm.save_dialogue_turn(
        {
            "ts": time.time(),
            "session_id": session_id,
            "user_text": user_text,
            "reply": turn.reply,
        }
    )
    return turn
