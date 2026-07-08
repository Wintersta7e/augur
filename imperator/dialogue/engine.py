"""Imperator III dialogue engine — the hybrid turn loop."""

from __future__ import annotations

import copy
import json
import logging
import time
from dataclasses import dataclass, replace
from typing import Any, Awaitable, Callable

from conscientia import screens
from imperator.dialogue import context as C, intents as I, persona, router as R

log = logging.getLogger(__name__)


def _cfg_with_num_predict(cfg: Any, num_predict: int) -> Any:
    """A view of ``cfg`` with ``dialogue_num_predict`` overridden to
    ``num_predict`` -- used only for a single LLM call so persona's
    register-scaled reply budget (``persona.num_predict_for_register``,
    spec §6) reaches ``query_dialogue_ollama``'s options dict without
    changing the ``QueryFn`` signature every test's stub ``query_fn``
    relies on. Production ``cfg`` (``AugurConfig``) is an immutable
    dataclass, so this goes through ``dataclasses.replace`` (which
    re-validates ``__post_init__`` bounds); a plain test-stub cfg (not a
    dataclass) gets a shallow attribute-copy instead.
    """
    try:
        return replace(cfg, dialogue_num_predict=num_predict)
    except TypeError:
        view = copy.copy(cfg)
        view.dialogue_num_predict = num_predict
        return view


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
    if not isinstance(obj, dict) or not isinstance(obj.get("reply"), str):
        # The LLM is called with format:json (valid JSON, but no schema), so a
        # model that teaches silently can emit {"reply": null, ...}. Require a
        # string reply here so handle_turn's fail-truthful branch catches it,
        # rather than a later `notice + reply` raising TypeError (lost turn) or
        # a literal "None" leaking into a confirmation prompt (invariant 7).
        raise ValueError("LLM output missing a string 'reply'")
    return obj


QueryFn = Callable[[str, str, Any, Any], Awaitable[str]]


async def _publish(nc, subject: str, payload: dict) -> None:
    """Fire a dialogue event on NATS. Best-effort: a publish failure is logged
    and swallowed, NEVER propagated. This runs only from _record_and_publish --
    i.e. AFTER the confirmed apply/undo has already committed to Redis and its
    audit record is persisted. Letting a NATS blip propagate here would surface
    a committed, truthful change as "turn failed" (a real success reported as a
    failure -- invariant 7 in reverse). The events are downstream triggers (the
    reasoner re-runs next cycle regardless), so dropping one is harmless."""
    if nc is None:
        return
    try:
        await nc.publish(subject, json.dumps(payload).encode())
    except Exception as exc:
        log.warning("dialogue event publish failed on %s: %s", subject, exc)


async def _record_and_publish(pm, nc, record: dict) -> None:
    """Persist a confirmed-apply/undo audit record and fire the two dialogue
    events every such record emits (mirrors the reasoner trigger every other
    faculty's completion publishes). Shared by _resolve_pending and
    _finish_undo -- both build a differently-shaped record dict, then take
    this identical audit+publish action on it."""
    pm.append_dialogue_audit(record)
    await _publish(nc, "augur.imperator.dialogue.applied", record)
    await _publish(
        nc, "augur.imperator.ii.trigger", {"reason": "dialogue", "ts": record["ts"]}
    )


def _save_turn(
    pm,
    session_id: str,
    user_text: str,
    reply: str,
    *,
    ts: float | None = None,
    applied: bool | None = None,
) -> None:
    """Persist a conversation-log turn. ``ts`` lets a confirmed-apply turn
    share its audit record's timestamp instead of drifting from a second
    time.time() call; every other caller gets a fresh timestamp. ``applied``
    is included only when the caller has a truthful apply outcome to report
    (a confirmed-pending turn) -- omitted elsewhere, matching the original
    per-branch dict literals."""
    record = {
        "ts": ts if ts is not None else time.time(),
        "session_id": session_id,
        "user_text": user_text,
        "reply": reply,
    }
    if applied is not None:
        record["applied"] = applied
    pm.save_dialogue_turn(record)


def _undo_echo(prior: dict) -> str:
    """Describe, for the confirmation prompt, what an undo of the audit-head
    record ``prior`` would reverse."""
    kind = prior.get("kind", "change")
    target = prior.get("target")
    if target:
        return f"I'll undo the {kind} change on {target}."
    return f"I'll undo the last {kind} change."


async def _handle_undo(
    session_id: str, user_text: str, base_reply: str, *, pm, cfg
) -> DialogueTurn:
    """Begin an undo request (spec §9: ``undo`` is a LIGHT-tier intent,
    confirmed by a plain affirmative like every other light intent -- never
    applied on the spot). Reads the most recently audited dialogue change
    GLOBALLY: ``pm.load_dialogue_audit`` is not session-scoped (unlike the
    conversation log), so "undo that" reverses the last confirmed change
    from ANY session, not just this one.

    Nothing-to-undo is reported immediately -- no pending stored, nothing
    audited or published -- since there's nothing for a confirmation to
    gate. Otherwise a light pending is stored carrying the audit-head record
    (``prior``); ``_finish_undo`` runs the real ``router.apply_undo`` (bounds
    pre-check + apply) once the user confirms, producing the same
    four-outcome (applied/blocked/logged/unavailable) truthful reply either
    way.
    """
    audit = pm.load_dialogue_audit(limit=1)
    if not audit or not audit[0].get("proposal") or audit[0].get("status") != "applied":
        # Nothing committed to reverse: no audit trail at all; a prior undo
        # attempt that never produced a proposal; OR a confirmed apply that
        # ended "logged" (e.g. a transient write raised after the rollback
        # anchor was recorded) and so wrote nothing. Undoing a non-"applied"
        # record would invert a rollback anchor for a write that never landed
        # and reply a false "Reversed." (invariant 7). router.build_inverse also
        # requires a real proposal dict to invert. Same truthful reply for all.
        return DialogueTurn(reply="There's nothing recent to undo.")
    prior = audit[0]
    echo = _undo_echo(prior)
    pending = {
        "kind": "undo",
        "tier": "light",
        "echo": echo,
        "confirm_phrase": None,
        "prior": prior,
    }
    pm.save_dialogue_pending(session_id, pending, ttl=cfg.dialogue_pending_ttl_s)
    reply = (
        f"{base_reply}\n{echo} Confirm? (yes)"
        if base_reply
        else f"{echo} Confirm? (yes)"
    )
    return DialogueTurn(reply=reply, pending=pending)


async def _finish_undo(
    prior: dict, session_id: str, user_text: str, *, pm, nc, cfg
) -> DialogueTurn:
    """Apply a confirmed undo (spec §9), run by ``_resolve_pending`` once the
    light pending ``_handle_undo`` created is confirmed.

    Goes through router.apply_undo rather than a hand-built build_inverse +
    apply.apply_proposal call: apply_undo pre-checks the inverse against the
    CURRENT apply-layer bounds so a rollback anchor recorded before those
    bounds tightened (e.g. a stale prior_sigma/prior floor) is reported as a
    distinct, truthful "blocked" outcome instead of silently failing closed
    with the same generic status as any other rejected proposal. All four
    apply_undo outcomes (unavailable/blocked/logged/applied) get their own
    reply text below -- never a blanket "Reversed." regardless of what
    actually happened.
    """
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
        # authorized this write) -- the plain-affirmative turn that confirmed
        # the undo pending is the provenance.
        "confirming_text": user_text,
    }
    await _record_and_publish(pm, nc, record)
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
    turn = DialogueTurn(reply=status_reply, applied=record)
    # Truthful: a confirmed undo can still resolve blocked/logged/unavailable
    # -- the conversation log must agree with the reply and audit, not claim
    # success unconditionally.
    _save_turn(
        pm,
        session_id,
        user_text,
        status_reply,
        ts=record["ts"],
        applied=out["status"] == "applied",
    )
    return turn


async def _resolve_pending(
    pending: dict, session_id: str, user_text: str, *, pm, nc, cfg
) -> DialogueTurn | None:
    """Try to resolve an existing pending proposal against this turn's text.

    Returns the completed DialogueTurn if the tier/phrase check passed and
    the proposal was applied. Returns None if it didn't match -- the pending
    has already been cleared either way (no double-apply), and the caller
    should notice-drop it and fall through to treating user_text as a fresh
    turn.
    """
    if pending.get("tier") == "heavy":
        ok = I.matches_heavy_phrase(user_text, pending.get("confirm_phrase") or "")
    else:
        ok = I.is_affirmative(user_text)
    pm.clear_dialogue_pending(session_id)  # cleared before apply: no double-apply
    if not ok:
        return None
    if pending.get("kind") == "undo":
        return await _finish_undo(
            pending["prior"], session_id, user_text, pm=pm, nc=nc, cfg=cfg
        )
    applied = R.apply_confirmed(pending, pm=pm, cfg=cfg, session_id=session_id)
    record = {
        "ts": time.time(),
        "session_id": session_id,
        "kind": pending["proposal"]["kind"],
        "target": pending["proposal"].get("target"),
        "proposal": applied["proposal"],
        "status": applied["status"],
        # Confirmation provenance: who confirmed (session_id) and what turn
        # confirmed it (the "yes"/heavy-phrase text).
        "confirming_text": user_text,
    }
    await _record_and_publish(pm, nc, record)
    reply = (
        f"Done — {applied['echo']} Say 'undo that' to reverse it."
        if applied["status"] == "applied"
        else "I couldn't apply that."
    )
    turn = DialogueTurn(reply=reply, applied=record)
    # Truthful: a confirmed apply can still resolve "logged" (kill switch off,
    # non-safe klass, apply error) -- the conversation log must agree with
    # the reply and audit, not claim success unconditionally.
    _save_turn(
        pm,
        session_id,
        user_text,
        reply,
        ts=record["ts"],
        applied=applied["status"] == "applied",
    )
    return turn


async def _handle_intent(
    intent: dict, base_reply: str, ctx, session_id: str, user_text: str, *, pm, cfg
) -> DialogueTurn:
    """Validate and route a freshly parsed intent: dispatch to undo, screen
    and route a new pending proposal awaiting confirmation, or fall back to a
    clarification reply on an invalid intent. Never applies anything itself
    -- undo (like every other kind) only ever creates a pending here; the
    actual apply/reversal happens at confirm time in ``_resolve_pending``.

    Teach intents (``teach_semantic_fact``/``teach_context_directive``) run
    through Conscientia's teach-time value screen here rather than inside
    ``router.route()``: route() is a pure translation layer with no Redis
    access (pinned by test_dialogue_invariants.py::
    test_route_is_pure_no_state_writes; its own docstring says pm/cfg are
    accepted only for interface symmetry). This is the earliest point where
    the validated intent fields, ``pm``, and ``session_id`` are all in scope
    together for the best-effort violation write. A screen refusal raises
    ValueError, reusing the same except-ValueError-to-truthful-reply seam as
    every other route() rejection (e.g. teach_context_directive's missing-
    focused-app check).
    """
    try:
        valid = I.validate_intent(intent)
        if valid["kind"] == "undo":
            return await _handle_undo(session_id, user_text, base_reply, pm=pm, cfg=cfg)
        if valid["kind"] in ("teach_semantic_fact", "teach_context_directive"):
            v = screens.screen_taught_content(
                valid.get("rationale"),
                (valid.get("action") or {}).get("rule_key"),
                cfg,
            )
            if not v.ok:
                try:
                    pm.save_conscientia_violation(
                        screens.make_violation(
                            "teach",
                            v.code or "refused",
                            v.detail or "",
                            v.principle or "",
                            session_id=session_id,
                        )
                    )
                except Exception:
                    log.warning(
                        "conscientia violation record failed (non-fatal)",
                        exc_info=True,
                    )
                raise ValueError(
                    f"I won't store that: {v.detail} (principle: {v.principle})."
                )
        new_pending = R.route(valid, ctx, pm=pm, cfg=cfg)
        pm.save_dialogue_pending(
            session_id, new_pending, ttl=cfg.dialogue_pending_ttl_s
        )
        ask = (
            f" Confirm with '{new_pending['confirm_phrase']}'."
            if new_pending["tier"] == "heavy"
            else " Confirm? (yes)"
        )
        reply = f"{base_reply}\n{new_pending['echo']}{ask}"
        return DialogueTurn(reply=reply, intent=valid, pending=new_pending)
    except ValueError as exc:
        return DialogueTurn(
            reply=f"I didn't quite follow — {exc}", needs_clarification=True
        )


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
    # cfg.dialogue_enabled is a master kill switch (mirrors imperator_ii_enabled's
    # runner-level check) -- getattr-defaulted True so callers/tests that stub a
    # minimal cfg without this field keep today's always-on behavior.
    if not getattr(cfg, "dialogue_enabled", True):
        return DialogueTurn(
            reply="Dialogue is currently disabled.", error="dialogue_disabled"
        )
    ctx = C.assemble(pm, now=time.time(), cfg=cfg)
    register = persona.register_for_salience(ctx.salience)
    system = persona.build_system_prompt(register, C.render(ctx, cfg), cfg)
    # Register-scaled reply budget (spec §6): a terse register gets a smaller
    # num_predict than an urgent one. Only the LLM-call cfg is adjusted --
    # every other read of ``cfg`` in this turn keeps its configured value.
    call_cfg = _cfg_with_num_predict(
        cfg, persona.num_predict_for_register(register, cfg)
    )

    notice = ""  # prefixed to the fresh-turn reply when a pending was dropped
    pending = pm.load_dialogue_pending(session_id)
    if pending is not None:
        resolved = await _resolve_pending(
            pending, session_id, user_text, pm=pm, nc=nc, cfg=cfg
        )
        if resolved is not None:
            return resolved
        # Tier/phrase mismatch: fall through and treat user_text as a fresh
        # turn -- but say the pending was dropped rather than silently
        # discarding the user's un-confirmed proposal.
        notice = "(Dropped the pending proposal.) "
    elif I.is_affirmative(user_text):
        # Expired/absent pending on a "yes": don't waste (or be misled by) an
        # LLM call on a bare confirmation gesture with nothing to confirm.
        turn = DialogueTurn(reply="There's nothing pending to confirm.")
        _save_turn(pm, session_id, user_text, turn.reply)
        return turn

    try:
        raw = await query_fn(user_text, system, http_client, call_cfg)
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
        _save_turn(pm, session_id, user_text, turn.reply)
        return turn

    intent = obj.get("intent")
    if intent:
        turn = await _handle_intent(
            intent, obj["reply"], ctx, session_id, user_text, pm=pm, cfg=cfg
        )
        turn.reply = notice + turn.reply
        _save_turn(pm, session_id, user_text, turn.reply)
        return turn

    turn = DialogueTurn(reply=notice + obj["reply"])
    _save_turn(pm, session_id, user_text, turn.reply)
    return turn
