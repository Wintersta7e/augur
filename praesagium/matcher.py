"""Praesagium matcher: the runtime prediction lifecycle AND the runner process
(spec 2026-07-09 sec 5).

Core lifecycle functions (Task 6, sec 5.2/5.3/6.1/6.2) -- arming
(``match_patterns``), resolution (``resolve_open_predictions``), the
deterministic forewarning template (``render_forewarning``/``_humanize``), and
the foreseen payload (``build_foreseen_payload``) -- take a
``PersistenceManager`` and an ``async (subject, data: bytes) -> None`` publish
as parameters; they never touch Redis/NATS directly, so they run unchanged
under the async test harness in ``tests/test_praesagium_matcher.py``.

The runner (Task 7, sec 5 intro) wires those functions to the live process:
``make_on_anomaly`` builds the ``augur.vigil.anomaly`` callback (record
episode -> resolve open predictions -> match armed patterns, strictly
sequential -- no task fan-out, PR2), ``on_session_end`` is a bookkeeping stub
for ``augur.session.end``, ``warm_cooldowns`` rebuilds in-memory cooldown
state from persisted predictions after a restart, and ``run``/``main`` are the
``python -m praesagium.matcher`` entry point (connect, heartbeat, subscribe,
sleep-forever, teardown -- the ``imperator/improver.py`` archetype).

Purity (invariant PR2): no LLM, no httpx/ollama, no mining, no patterns-blob
writes. Forewarnings are a deterministic template (``render_forewarning``);
the foreseen envelope (``build_foreseen_payload``) is nexus-detected-shaped
and never-exempt by construction (invariant PR1a).

Spec: docs/superpowers/specs/2026-07-09-praesagium-design.md sec 5, 5.1, 5.2,
5.3, 6.1, 6.2, 11 (PR2/PR3).
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import nats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tabula.config import AugurConfig
from tabula.connections import connect_redis
from tabula.heartbeat import start_heartbeat
from tabula.persistence import MAX_PRAESAGIUM_RESOLVED, PersistenceManager

from praesagium.episodes import build_episode, canonical_key, parse_epoch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("praesagium.matcher")

# Published by this faculty.
SUBJECT_FORESEEN = "augur.praesagium.foreseen"
SUBJECT_RESOLVED = "augur.praesagium.resolved"

# Consumed by this faculty (the runner, sec 5).
SUBSCRIBE_ANOMALY = "augur.vigil.anomaly"
SUBSCRIBE_SESSION_END = "augur.session.end"

# Consequent severities worth resolving a prediction as fulfilled -- Consilium
# only advises on medium/high, so those are the outcomes a forewarning claims.
_FULFIL_SEVERITIES = ("medium", "high")


# -- text ---------------------------------------------------------------------


def _humanize(key: str) -> str:
    """Canonical key -> readable label: ``"typing:user"`` -> ``"typing (user)"``.

    Splits on the FIRST colon only, so a rare entity that itself contains a
    colon is preserved whole (``"activity:app:vscode"`` -> ``"activity
    (app:vscode)"``). A key with no colon passes through unchanged.
    """
    if ":" not in key:
        return key
    domain, entity = key.split(":", 1)
    return f"{domain} ({entity})"


def render_forewarning(p: dict) -> str:
    """Deterministic, neutral-valence forewarning sentence (spec sec 6.2, verbatim).

    A statement of an observation -- no advice, no rest/break suggestion, no
    fatigue/distraction attribution, no self-reference. The confidence claim is
    honest because the miner's pass-2 recount made ``conf_lower`` a statement
    about exactly the window ``window_s`` this sentence names.
    """
    ante = _humanize(p["antecedent"])
    cons = _humanize(p["consequent"])
    return (
        f"Forewarning: in {p['support_sessions']} recent sessions, {ante} was "
        f"followed by {cons} within ~{int(p['window_s'])}s "
        f"(confidence ≥ {int(p['conf_lower'] * 100)}%). {ante} was just observed."
    )


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


# -- foreseen payload ---------------------------------------------------------


def build_foreseen_payload(
    pattern: dict, prediction: dict, session_id: str | None
) -> dict:
    """The nexus-detected-shaped foreseen envelope (spec sec 6.1, every field).

    Never-exempt by construction: ``correlation_found=False`` +
    ``combined_severity="MEDIUM"`` => ``build_signature(...).exempt is False`` and
    ``path == "single"`` (invariant PR1a). Statistical fields come from the
    pattern artifact; ``prediction_id``/``forewarning_text`` from the prediction
    record; ``session_id`` is threaded through from the triggering event.
    """
    pid = pattern["pattern_id"]
    antecedent = pattern["antecedent"]
    consequent = pattern["consequent"]
    window_s = pattern["window_s"]
    support_sessions = pattern["support_sessions"]
    conf_lower = pattern["conf_lower"]
    iso_now = _iso_now()
    return {
        "primary_anomaly": {
            "domain": "praesagium",
            "entity": pid,
            "severity": "medium",
            "event_type": "forewarning",
            "value": round(conf_lower, 3),
            "unit": "confidence",
            "session_id": session_id,
            "timestamp": iso_now,
            "context": {
                "antecedent": antecedent,
                "consequent": consequent,
                "window_s": window_s,
                "support_sessions": support_sessions,
                "lift": round(pattern["lift"], 2),
                "label": f"{antecedent} → {consequent}",
            },
            "baseline_mean": None,
            "baseline_std": None,
            "deviation_score": None,
            "baseline_observation_count": None,
        },
        "correlated_events": [],
        "correlation_found": False,
        "combined_severity": "MEDIUM",
        "temporal_lag_seconds": None,
        "correlation_span_s": None,
        "severity_escalated": False,
        "escalation_rule": None,
        "escalation_matrix_version": None,
        "rule_key": None,
        "rule_window_s": None,
        "involved_domains": ["praesagium"],
        "timestamp": iso_now,
        "source": "anticipatory",
        "anticipatory": {
            "pattern_id": pid,
            "prediction_id": prediction["prediction_id"],
            "antecedent": antecedent,
            "consequent": consequent,
            "window_s": window_s,
            "conf_lower": conf_lower,
            "support_sessions": support_sessions,
            "forewarning_text": prediction["forewarning_text"],
        },
    }


# -- resolution (self-verification, PR6) --------------------------------------


async def resolve_open_predictions(
    pm: Any,
    publish: Any,
    key: str,
    severity: str,
    ts: float,
    *,
    cfg: Any,
) -> int:
    """Resolve open predictions against one anomaly; return this call's count.

    Called on EVERY anomaly BEFORE ``match_patterns`` (the Task-7 call site
    orders resolve-then-match) so an event can never fulfil the prediction it
    just armed. Two mutually exclusive outcomes per open prediction:

    * **fulfilled** -- ``consequent == key`` and ``severity in {medium, high}``
      and ``ts <= deadline_ts`` (``lag_s = ts - created_ts``).
    * **expired** -- ``deadline_ts < ts - praesagium_expiry_grace_s``.

    Each resolution is ONE call to the atomic ``resolve_praesagium_prediction``
    (append-iff-removed): a replay / duplicate delivery returns False and skips
    all side effects (invariant PR6). On a True resolution the resolved record
    is published on :data:`SUBJECT_RESOLVED`, best-effort (failure swallowed,
    logged at debug -- telemetry must not break the hot path).
    """
    sev = str(severity).lower()
    grace = cfg.praesagium_expiry_grace_s
    resolved_count = 0

    for rec in pm.load_praesagium_open_predictions():
        if not isinstance(rec, dict):
            continue
        pid_rec = rec.get("prediction_id")
        if not pid_rec:
            continue
        deadline = rec.get("deadline_ts")
        if not isinstance(deadline, (int, float)):
            continue
        created = rec.get("created_ts")

        outcome: str | None = None
        lag: float | None = None
        if (
            rec.get("consequent") == key
            and sev in _FULFIL_SEVERITIES
            and ts <= deadline
        ):
            outcome = "fulfilled"
            if isinstance(created, (int, float)):
                lag = ts - created
        elif deadline < ts - grace:
            outcome = "expired"
        if outcome is None:
            continue

        resolved_rec = {
            "prediction_id": pid_rec,
            "pattern_id": rec.get("pattern_id"),
            "outcome": outcome,
            "resolved_ts": ts,
            "created_ts": created,
            "deadline_ts": deadline,
            "lag_s": lag,
            "session_id": rec.get("session_id"),
            "resolved_by": "matcher",
        }
        try:
            claimed = pm.resolve_praesagium_prediction(
                pid_rec, resolved_rec, cap=cfg.praesagium_predictions_cap
            )
        except Exception as exc:  # one bad record must not abort the sweep
            log.warning("Praesagium resolve failed for %s: %s", pid_rec, exc)
            continue
        if not claimed:
            continue  # already resolved elsewhere (replay/duplicate) -- no-op
        resolved_count += 1
        try:
            await publish(SUBJECT_RESOLVED, json.dumps(resolved_rec).encode())
        except Exception as exc:  # telemetry only -- swallow
            log.debug("Praesagium resolved publish failed for %s: %s", pid_rec, exc)

    return resolved_count


# -- arming (match armed patterns, PR2) ---------------------------------------


async def match_patterns(
    pm: Any,
    publish: Any,
    key: str,
    ts: float,
    payload: dict,
    *,
    cfg: Any,
    cooldowns: dict[str, float],
) -> None:
    """Arm predictions for every ACTIVE pattern whose antecedent matches *key*.

    Per pattern, in order: skip if within the in-memory cooldown
    (``ts - cooldowns[pid] < praesagium_pattern_cooldown_s``); skip if an open
    prediction for that pattern already exists (persisted no-open-duplicate
    guard, safe as check-then-set because the handler is strictly sequential);
    otherwise save the open record FIRST (``emit_attempted =
    praesagium_emit_enabled``) and, only if emission is armed, publish the
    foreseen payload. On publish success stamp ``emitted_at``; on publish
    failure WARN and leave ``emitted_at`` None (never claim the user was warned
    when no message went out). ``cooldowns[pid]`` is set at arm time regardless
    of emission. A cap-refused save is skipped with a WARN (defense in depth --
    unreachable by invariant PR4).

    Every call also prunes ``cooldowns`` of any pid no longer present in the
    freshly-loaded blob (retired/evicted patterns) -- otherwise the in-memory
    dict grows without bound over weeks of mine cycles. Runs unconditionally
    on every reach of a valid blob, even when no pattern in it matches *key*.
    """
    blob = pm.load_praesagium_patterns()
    if not isinstance(blob, dict):
        return
    patterns = blob.get("patterns")
    if not isinstance(patterns, dict):
        return

    for pid in list(cooldowns):
        if pid not in patterns:
            del cooldowns[pid]

    session_id = payload.get("session_id")
    open_pattern_ids = {
        r.get("pattern_id")
        for r in pm.load_praesagium_open_predictions()
        if isinstance(r, dict)
    }
    cooldown_s = cfg.praesagium_pattern_cooldown_s
    emit_enabled = bool(cfg.praesagium_emit_enabled)

    for pid, pattern in patterns.items():
        if not isinstance(pattern, dict):
            continue
        if pattern.get("status") != "active":
            continue
        if pattern.get("antecedent") != key:
            continue

        last = cooldowns.get(pid)
        if last is not None and ts - last < cooldown_s:
            continue
        if pid in open_pattern_ids:
            continue

        window_s = pattern["window_s"]
        prediction = {
            "prediction_id": uuid4().hex,
            "pattern_id": pid,
            "antecedent": pattern["antecedent"],
            "consequent": pattern["consequent"],
            "window_s": window_s,
            "created_ts": ts,
            "deadline_ts": ts + window_s,
            "session_id": session_id,
            "conf_lower": pattern["conf_lower"],
            "emit_attempted": emit_enabled,
            "emitted_at": None,
            "forewarning_text": render_forewarning(pattern),
        }
        saved = pm.save_praesagium_open_prediction(
            prediction, cap=cfg.praesagium_open_predictions_cap
        )
        if not saved:
            log.warning(
                "Praesagium open-prediction cap reached; skipping arm for %s", pid
            )
            continue

        # Armed: mark cooldown (regardless of emission) and prevent a same-call
        # re-arm of this pattern.
        cooldowns[pid] = ts
        open_pattern_ids.add(pid)

        if not emit_enabled:
            continue

        foreseen = build_foreseen_payload(pattern, prediction, session_id)
        try:
            await publish(SUBJECT_FORESEEN, json.dumps(foreseen).encode())
        except Exception as exc:  # honest: no emitted_at stamp when no message went
            log.warning("Praesagium foreseen publish failed for %s: %s", pid, exc)
            continue
        pm.update_praesagium_open_prediction(
            prediction["prediction_id"], {"emitted_at": ts}
        )


# -- runner (subscriptions, heartbeat, warm-start) -----------------------------


def make_on_anomaly(
    pm: Any, config: AugurConfig, publish: Any, cooldowns: dict[str, float]
) -> Any:
    """Build the ``augur.vigil.anomaly`` subscription callback (spec sec 5.1).

    Order: decode -> kill-switch check (before any pm call, PR3) -> canonical
    key + timestamp (unkeyable/unparseable -> no writes at all) -> episode
    append (skipped, not fatal, when the payload carries no session_id) ->
    resolve open predictions -> match armed patterns. resolve-then-match so an
    event can never fulfil the prediction it just armed (mirrors the Task-6
    call-site ordering pinned by
    ``test_b_event_fulfils_one_and_arms_another_when_ordered``).

    STRICTLY SEQUENTIAL (PR2): no task fan-out anywhere in this path -- the
    Task-6 review found ``emitted_at`` safety depends on the handler running
    to completion before nats-py dispatches the next message. Wrapped so no
    exception escapes: a broken predictor must never break perception,
    correlation, or advice (fail silent-and-logged, spec sec 11 fail
    directions).
    """

    async def on_anomaly(msg: Any) -> None:
        try:
            payload = json.loads(msg.data.decode())
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            log.warning("Praesagium: malformed anomaly payload: %s", exc)
            return

        if not config.praesagium_enabled:
            return  # PR3: zero writes/publishes before any pm call

        try:
            key = canonical_key(payload)
            ts = parse_epoch(payload.get("timestamp"))
            if key is None or ts is None:
                return  # unkeyable/unparseable -- mirrors Limen's ungateable rule

            session_id = payload.get("session_id")
            if isinstance(session_id, str) and session_id:
                entry = build_episode(payload)
                if entry is not None:
                    try:
                        pm.append_praesagium_episode(
                            session_id,
                            entry,
                            cap=config.praesagium_episode_cap_per_session,
                        )
                    except Exception as exc:
                        log.warning("Praesagium episode append failed: %s", exc)
            else:
                log.debug(
                    "Praesagium: anomaly missing session_id; skipping episode "
                    "(resolve+match still run)"
                )

            await resolve_open_predictions(
                pm, publish, key, payload.get("severity"), ts, cfg=config
            )
            await match_patterns(
                pm, publish, key, ts, payload, cfg=config, cooldowns=cooldowns
            )
        except Exception as exc:
            log.warning(
                "Praesagium on_anomaly failed (non-fatal): %s", exc, exc_info=True
            )

    return on_anomaly


async def on_session_end(msg: Any) -> None:
    """Bookkeeping-only handler for ``augur.session.end`` (spec sec 5).

    Clears per-session in-memory state. v1 has none -- cooldowns are keyed
    per pattern_id, not per session -- so this is a no-op stub beyond a debug
    log, kept for the subscription archetype and future per-session state.
    Malformed payloads are logged at debug and ignored (bookkeeping only;
    never worth a WARN).
    """
    try:
        payload = json.loads(msg.data.decode())
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        log.debug("Praesagium: bad session.end payload: %s", exc)
        return
    log.debug(
        "Praesagium: session %s ended (no per-session state to clear)",
        payload.get("session_id"),
    )


def warm_cooldowns(
    pm: Any, *, resolved_limit: int = MAX_PRAESAGIUM_RESOLVED
) -> dict[str, float]:
    """Rebuild in-memory cooldown state from persisted state after a restart
    (spec sec 5, warm-start).

    Per pattern_id, the cooldown anchor is the max of: any open prediction's
    ``created_ts``, and the newest resolved record's ``created_ts`` (the
    resolved log arrives newest-first, so the first record seen per
    pattern_id during the walk is already the newest). Corrupt/missing
    fields are skipped -- warm-start is best-effort, not a correctness
    requirement (a cold cooldown only risks one extra duplicate arm, which
    the no-open-duplicate guard and the pattern's own window largely absorb).

    ``resolved_limit`` bounds the resolved-log read (defaults to the same cap
    the persistence layer caps the log at -- ``load_praesagium_resolved``'s
    own default of 50 is far shallower and would miss older per-pattern
    resolutions on a warm start).
    """
    cooldowns: dict[str, float] = {}

    for rec in pm.load_praesagium_open_predictions():
        if not isinstance(rec, dict):
            continue
        pid = rec.get("pattern_id")
        created = rec.get("created_ts")
        if not pid or not isinstance(created, (int, float)):
            continue
        if pid not in cooldowns or created > cooldowns[pid]:
            cooldowns[pid] = created

    seen: set[str] = set()
    for rec in pm.load_praesagium_resolved(limit=resolved_limit):
        if not isinstance(rec, dict):
            continue
        pid = rec.get("pattern_id")
        if not pid or pid in seen:
            continue
        seen.add(pid)
        created = rec.get("created_ts")
        if not isinstance(created, (int, float)):
            continue
        if pid not in cooldowns or created > cooldowns[pid]:
            cooldowns[pid] = created

    return cooldowns


async def run() -> None:
    config = AugurConfig.from_env()

    redis_client = connect_redis(config)
    pm = PersistenceManager(redis_client)

    nc = await nats.connect(
        config.nats_url, connect_timeout=config.nats_connect_timeout
    )
    # NOTE: heartbeat starts even when praesagium_enabled is False -- PR3's
    # carve-out (liveness != faculty activity; a disabled-but-required
    # faculty must not read dead to Praefectus).
    hb_task = (
        start_heartbeat(nc, "praesagium", config.praefectus_heartbeat_interval_s)
        if config.praefectus_enabled
        else None
    )

    sub_anomaly = None
    sub_session_end = None
    if config.praesagium_enabled:
        cooldowns = warm_cooldowns(pm, resolved_limit=config.praesagium_predictions_cap)
        on_anomaly = make_on_anomaly(pm, config, nc.publish, cooldowns)
        sub_anomaly = await nc.subscribe(SUBSCRIBE_ANOMALY, cb=on_anomaly)
        sub_session_end = await nc.subscribe(SUBSCRIBE_SESSION_END, cb=on_session_end)
        log.info(
            "Praesagium matcher subscribed (%d pattern(s) warm-started)",
            len(cooldowns),
        )
    else:
        log.info("praesagium disabled; idling with heartbeat")

    try:
        while True:
            await asyncio.sleep(3600)
    finally:
        if hb_task is not None:
            hb_task.cancel()
            try:
                await hb_task
            except asyncio.CancelledError:
                pass
        for sub in (sub_anomaly, sub_session_end):
            if sub is not None:
                try:
                    await sub.unsubscribe()
                except Exception:
                    log.debug("Praesagium unsubscribe failed", exc_info=True)
        await nc.close()
        try:
            redis_client.close()
        except Exception:
            log.debug("Praesagium redis close failed", exc_info=True)


def main() -> None:
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        log.info("Interrupted")


if __name__ == "__main__":
    main()
