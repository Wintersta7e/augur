"""Persistence layer — Redis-backed storage for baselines, history, feedback,
prompt versioning, and threshold configuration."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, cast

import redis

from memoria.fsrs import make_memory_id
from tabula.contracts import PerceptionEvent

log = logging.getLogger("persistence")


def _to_epoch(ts: Any) -> float:
    """Coerce an ISO-8601 string or numeric epoch to a float epoch (0.0 on failure)."""
    if ts is None:
        return 0.0
    if isinstance(ts, (int, float)):
        return float(ts)
    try:
        return datetime.fromisoformat(str(ts)).timestamp()
    except (ValueError, TypeError):
        return 0.0


HISTORY_MAX = 1000
PROMPT_HISTORY_MAX = 100
# Cap on distinct app-descriptor entries in the Redis hash. Bounds memory +
# HGETALL latency (the MCP get_app_descriptors tool reads the whole hash) and
# matches the MAX_BASELINE_ENTITIES discipline. New entities beyond this are
# dropped; existing entries can still be updated.
MAX_APP_DESCRIPTORS = 2000

# ── Advisor gate list-log caps ───────────────────────────────────────────────
# Each is stored as a Redis list (lpush + ltrim).  The numbers balance
# diagnostic visibility against memory: silences/emissions/observed are
# high-frequency, delivery_failures are rare.
MAX_GATE_SILENCES: int = 2000
MAX_GATE_EMISSIONS: int = 2000
MAX_GATE_OBSERVED: int = 2000
MAX_GATE_DELIVERY_FAILURES: int = 500
# Cap on distinct per-channel state-key entries in the gate hash stores
# (habituation, channel_stats, reservoir, credibility, cost_tier_memory).
# Matches MAX_APP_DESCRIPTORS discipline: new keys beyond this are refused
# (fail-open / fire-leaning); existing keys keep updating.
MAX_GATE_STATE_KEYS: int = 2000
MAX_MEMORY_ITEMS: int = 5000  # refuse-at-cap ceiling for Memoria tier sets (Lane 2)
MAX_IMPERATOR_PROPOSALS: int = (
    200  # newest-first capped append-log for Imperator proposals
)
MAX_DIALOGUE_LOG: int = 500  # newest-first capped conversation log
# Cap on taught context directives. The whole hash is HGETALL'd into the
# dialogue context on every turn, so this bounds per-turn load latency the
# same way MAX_APP_DESCRIPTORS bounds get_app_descriptors. Directives are
# hand-taught (one per explicit user instruction), so 200 — matching
# MAX_IMPERATOR_PROPOSALS — is already far beyond realistic use. New ids
# beyond the cap are refused; existing ids keep updating.
MAX_DIALOGUE_DIRECTIVES: int = 200

# Default TTL for per-session Redis keys (feedback, correlation graph,
# reflection report). Prevents indefinite growth beyond the 1000-entry
# index trim boundary. Override in tests by passing ``ttl_s=None`` to the
# save methods if you need a persistent key.
SESSION_KEY_TTL_S = 30 * 24 * 3600  # 30 days

# TTL for the correlation-tuning idempotency marker. Long enough to survive
# manual reflect-trigger replays of a recent session, short enough to prevent
# the key from lingering indefinitely.
TUNING_APPLIED_TTL_S = 7 * 24 * 3600  # 7 days


class PersistenceManager:
    """Unified persistence interface for all Augur subsystems."""

    def __init__(self, r: redis.Redis) -> None:
        self._r = r

    # -- JSON get/set helpers ------------------------------------------------
    # Collapse the uniform "set(key, json.dumps(x))" / "raw = get(key); return
    # json.loads(raw) if raw is not None else default" boilerplate. Only the
    # methods that match that pattern EXACTLY use these; archival writes, TTL
    # variants with extra index bookkeeping, in-place updates, list ops, and
    # corrupt-tolerant loaders keep their bespoke bodies.

    def _set_json(self, key: str, value: object, *, ex: int | None = None) -> None:
        """SET *key* to ``json.dumps(value)``, optionally with a TTL.

        When *ex* is None the ``ex`` kwarg is omitted entirely (not passed as
        ``ex=None``) so the underlying ``redis.set`` call is byte-for-byte the
        same as the prior inline ``self._r.set(key, json.dumps(x))`` — callers
        and tests that inspect the call args see no new keyword.
        """
        if ex is None:
            self._r.set(key, json.dumps(value))
        else:
            self._r.set(key, json.dumps(value), ex=ex)

    def _get_json(self, key: str, default: Any = None) -> Any:
        """GET *key* and JSON-decode it; return *default* when the key is absent.

        Mirrors the inline ``json.loads(raw) if raw is not None else default``
        pattern verbatim — a present-but-malformed value still raises, matching
        the prior behavior of every method that used this shape. Returns ``Any``
        (like ``json.loads``) so callers keep their concrete ``dict | None``
        return annotations without a cast.
        """
        raw = cast(Any, self._r.get(key))
        if raw is None:
            return default
        return json.loads(raw)

    # -- Baseline persistence ------------------------------------------------

    def save_baseline(self, domain: str, entity: str, state_dict: dict) -> None:
        key = f"augur:vigil:profile:{domain}:{entity}"
        self._set_json(key, state_dict)
        log.debug("Saved baseline %s", key)

    def load_baseline(self, domain: str, entity: str) -> dict | None:
        key = f"augur:vigil:profile:{domain}:{entity}"
        return self._get_json(key)

    def scan_baseline_maturity(self, *, trained_obs: int = 15) -> dict:
        """Tally Vigil baselines by training maturity via SCAN (never KEYS).

        Returns {total, trained, untrained, by_domain:{domain:{total,trained}}}.
        Trained = observation_count >= trained_obs. Key: augur:vigil:profile:{domain}:{entity}.
        """
        total = trained = 0
        by_domain: dict[str, dict[str, int]] = {}
        for key in self._r.scan_iter(match="augur:vigil:profile:*", count=500):
            raw = cast(Any, self._r.get(key))
            if raw is None:
                continue
            try:
                obs = int(json.loads(raw).get("observation_count", 0))
            except (json.JSONDecodeError, TypeError, ValueError):
                continue
            key_s = key.decode() if isinstance(key, bytes) else key
            parts = key_s.split(":")  # augur:vigil:profile:{domain}:{entity}
            domain = parts[3] if len(parts) >= 5 else "unknown"
            d = by_domain.setdefault(domain, {"total": 0, "trained": 0})
            total += 1
            d["total"] += 1
            if obs >= trained_obs:
                trained += 1
                d["trained"] += 1
        return {
            "total": total,
            "trained": trained,
            "untrained": total - trained,
            "by_domain": by_domain,
        }

    # -- Event history -------------------------------------------------------

    def append_event(self, event: PerceptionEvent) -> None:
        key = f"augur:vigil:history:{event.domain}"
        self._r.lpush(key, event.to_json())
        self._r.ltrim(key, 0, HISTORY_MAX - 1)

    def get_history(self, domain: str, limit: int = 100) -> list[dict]:
        key = f"augur:vigil:history:{domain}"
        raw_list = cast(list[Any], self._r.lrange(key, 0, limit - 1))
        return [json.loads(entry) for entry in raw_list]

    def load_focused_app(
        self, *, now: float | None = None, max_age_s: float | None = None
    ) -> str | None:
        """Newest focused-app value across the two activity Sensus streams.

        The single source of truth for "what app is the user in right now":
        newest of ``activity_focus`` (``context.new_app``) and
        ``activity_intensity`` (``context.focused_app``), read from the
        ``augur:vigil:history:*`` keys via ``get_history``. Consumed by both
        Imperator's ``activity`` read-model field
        (``imperator/sources.py::gather``) and the Limen taught-directive
        pre-check (``limen/gate.py::Gate.evaluate``).

        When both *now* and *max_age_s* are given, a focus/intensity entry
        older than ``now - max_age_s`` is treated as ABSENT. A stalled activity
        Sensor would otherwise keep reporting the last-seen app forever, so a
        taught "stay quiet in app X" directive could keep matching -- and a new
        teach could be pinned to -- an app the user left long ago. Returning
        None on a stale read makes the Limen pre-check fall through toward FIRE
        (the safe direction) and the dialogue teach path refuse truthfully.
        Omitting the bounds (the default) preserves the raw newest-entry read.
        """
        focus_hist = self.get_history("activity_focus", limit=1)
        intens_hist = self.get_history("activity_intensity", limit=1)
        focus = focus_hist[0] if focus_hist else None
        intens = intens_hist[0] if intens_hist else None
        f_ts = _to_epoch(focus.get("timestamp")) if focus else float("-inf")
        i_ts = _to_epoch(intens.get("timestamp")) if intens else float("-inf")
        if now is not None and max_age_s is not None:
            cutoff = now - max_age_s
            if focus is not None and f_ts < cutoff:
                focus, f_ts = None, float("-inf")
            if intens is not None and i_ts < cutoff:
                intens, i_ts = None, float("-inf")
        if intens and i_ts >= f_ts:
            return (intens.get("context") or {}).get("focused_app") or intens.get(
                "entity"
            )
        if focus:
            return (focus.get("context") or {}).get("new_app") or focus.get("entity")
        return None

    # -- Feedback storage ----------------------------------------------------

    def save_feedback(self, session_id: str, feedback_dict: dict) -> None:
        key = f"augur:responsum:{session_id}"
        feedback_dict.setdefault(
            "timestamp",
            datetime.now(timezone.utc).isoformat(),
        )
        self._r.set(key, json.dumps(feedback_dict), ex=SESSION_KEY_TTL_S)
        # Also maintain an ordered index of session IDs for get_all_feedback
        self._r.lpush("augur:responsum:_index", session_id)
        self._r.ltrim("augur:responsum:_index", 0, 999)
        log.debug("Saved feedback for session %s", session_id)

    def get_feedback(self, session_id: str) -> dict | None:
        key = f"augur:responsum:{session_id}"
        return self._get_json(key)

    def get_all_feedback(self, limit: int = 50) -> list[dict]:
        session_ids = cast(
            list[Any], self._r.lrange("augur:responsum:_index", 0, limit - 1)
        )
        results: list[dict] = []
        for sid in session_ids:
            sid_str = sid.decode() if isinstance(sid, bytes) else sid
            fb = self.get_feedback(sid_str)
            if fb is not None:
                fb["session_id"] = sid_str
                results.append(fb)
        return results

    # -- Prompt versioning ---------------------------------------------------

    def save_prompt(
        self,
        domain: str,
        prompt_text: str,
        score: float | None = None,
    ) -> None:
        current_key = f"augur:consilium:prompts:{domain}:current"
        history_key = f"augur:consilium:prompts:{domain}:history"

        # Archive current version before overwriting
        existing = cast(Any, self._r.get(current_key))
        if existing is not None:
            self._r.lpush(history_key, existing)
            self._r.ltrim(history_key, 0, PROMPT_HISTORY_MAX - 1)

        entry = {
            "prompt": prompt_text,
            "score": score,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self._r.set(current_key, json.dumps(entry))
        log.debug("Saved prompt for domain %s (score=%s)", domain, score)

    def load_prompt(self, domain: str) -> str | None:
        current_key = f"augur:consilium:prompts:{domain}:current"
        raw = cast(Any, self._r.get(current_key))
        if raw is None:
            return None
        return json.loads(raw).get("prompt")

    def get_prompt_history(self, domain: str, limit: int = 10) -> list[dict]:
        history_key = f"augur:consilium:prompts:{domain}:history"
        raw_list = cast(list[Any], self._r.lrange(history_key, 0, limit - 1))
        return [json.loads(entry) for entry in raw_list]

    def rollback_prompt(self, domain: str) -> bool:
        """Restore previous prompt version. Returns True if rollback succeeded."""
        current_key = f"augur:consilium:prompts:{domain}:current"
        history_key = f"augur:consilium:prompts:{domain}:history"

        previous_raw = cast(Any, self._r.lpop(history_key))
        if previous_raw is None:
            log.warning("No previous prompt to rollback to for domain %s", domain)
            return False

        # Archive the current (bad) version into history tail
        current_raw = cast(Any, self._r.get(current_key))
        if current_raw is not None:
            self._r.rpush(history_key, current_raw)
            self._r.ltrim(history_key, 0, PROMPT_HISTORY_MAX - 1)

        self._r.set(current_key, previous_raw)
        log.info("Rolled back prompt for domain %s", domain)
        return True

    def update_current_prompt_score(self, domain: str, realized_score: float) -> None:
        """Overwrite the CURRENT prompt entry's score in place (no archive).

        Lets the reflection cycle stamp the live prompt's REALIZED score (spec
        1E §9) so the rollback check compares current-vs-previous realized scores
        rather than the stale motivating-utility recorded at mutation time.
        """
        current_key = f"augur:consilium:prompts:{domain}:current"
        raw = cast(Any, self._r.get(current_key))
        if raw is None:
            return
        entry = json.loads(raw)
        entry["score"] = realized_score
        self._r.set(current_key, json.dumps(entry))

    def get_prompt_score_pair(self, domain: str) -> tuple[float | None, float | None]:
        """Return (current_score, most_recent_previous_score) for the domain's
        prompt — the realized-score pair the 1E rollback gate compares."""
        current_key = f"augur:consilium:prompts:{domain}:current"
        history_key = f"augur:consilium:prompts:{domain}:history"
        cur_raw = cast(Any, self._r.get(current_key))
        cur = json.loads(cur_raw).get("score") if cur_raw else None
        prev_list = cast(list[Any], self._r.lrange(history_key, 0, 0))
        prev = json.loads(prev_list[0]).get("score") if prev_list else None
        return cur, prev

    # -- MRT withheld-rating calibration tracking (spec 1B) ------------------

    def mark_mrt_rating_session(self, session_id: str) -> None:
        """Record that a session issued >=1 withheld-rating prompt (set: dedups)."""
        self._r.sadd("augur:limen:mrt_rating_sessions", session_id)

    def count_mrt_rating_sessions(self) -> int:
        """Number of distinct sessions that issued a withheld-rating prompt."""
        return int(cast(int, self._r.scard("augur:limen:mrt_rating_sessions")))

    # -- Threshold config ----------------------------------------------------

    def save_thresholds(self, domain: str, thresholds_dict: dict) -> None:
        key = f"augur:vigil:thresholds:{domain}"
        self._set_json(key, thresholds_dict)
        log.debug("Saved thresholds for domain %s", domain)

    def load_thresholds(self, domain: str) -> dict | None:
        key = f"augur:vigil:thresholds:{domain}"
        return self._get_json(key)

    # -- Escalation matrix (cross-domain correlator) -------------------------

    def save_escalation_matrix(self, matrix: dict) -> None:
        """Store the full matrix dict (including 'version' and 'rules' keys) as-is."""
        key = "augur:nexus:matrix"
        self._set_json(key, matrix)
        log.debug("Saved escalation matrix (version=%s)", matrix.get("version"))

    def load_escalation_matrix(self) -> dict | None:
        """Return the full matrix dict or None if not set."""
        key = "augur:nexus:matrix"
        return self._get_json(key)

    def save_health_snapshot(self, snapshot: dict) -> None:
        """Store the Praefectus health snapshot (overwritten each tick; no TTL — live state)."""
        key = "augur:praefectus:health"
        self._set_json(key, snapshot)

    def load_health_snapshot(self) -> dict | None:
        """Return the last Praefectus health snapshot or None if not set."""
        key = "augur:praefectus:health"
        return self._get_json(key)

    # ── App descriptors (autonomous app->identity map) ────────────────────

    def save_app_descriptor(
        self, entity: str, descriptor: str, *, overwrite: bool
    ) -> None:
        """Store an app's descriptor in the augur:consilium:app_descriptors hash.

        ``overwrite=True`` (OS FileDescription — authoritative) uses HSET and
        upgrades any earlier value. ``overwrite=False`` (LLM fallback) uses
        HSETNX so a late classification can never clobber OS truth or an
        earlier guess.
        """
        key = "augur:consilium:app_descriptors"
        if (
            not cast(bool, self._r.hexists(key, entity))
            and cast(int, self._r.hlen(key)) >= MAX_APP_DESCRIPTORS
        ):
            log.warning(
                "app_descriptor hash full (%d); dropping new entity %s",
                MAX_APP_DESCRIPTORS,
                entity,
            )
            return
        if overwrite:
            self._r.hset(key, entity, descriptor)
        else:
            self._r.hsetnx(key, entity, descriptor)

    def load_app_descriptor(self, entity: str) -> str | None:
        """Return one app's descriptor (decoded), or None if absent."""
        raw = cast(Any, self._r.hget("augur:consilium:app_descriptors", entity))
        if raw is None:
            return None
        return raw.decode() if isinstance(raw, bytes) else raw

    def load_app_descriptors(self) -> dict[str, str]:
        """Return the full app->descriptor map with keys/values decoded to str."""
        raw = cast(dict[Any, Any], self._r.hgetall("augur:consilium:app_descriptors"))
        out: dict[str, str] = {}
        for k, v in raw.items():
            key = k.decode() if isinstance(k, bytes) else k
            val = v.decode() if isinstance(v, bytes) else v
            out[key] = val
        return out

    # -- Correlation graph (cross-domain correlator) -------------------------

    def save_correlation_graph(self, session_id: str, graph_data: dict) -> None:
        """Persist a session's correlation DiGraph as node_link_data JSON.

        Also maintains an ordered index list so list_correlation_graphs
        can return session ids without scanning keyspace. Uses a 30-day TTL
        to prevent unbounded Redis growth past the index-trim boundary.
        """
        key = f"augur:nexus:graph:{session_id}"
        self._r.set(key, json.dumps(graph_data), ex=SESSION_KEY_TTL_S)
        self._r.lpush("augur:nexus:graph:_index", session_id)
        self._r.ltrim("augur:nexus:graph:_index", 0, 999)
        log.debug(
            "Saved correlation graph for session %s (%d nodes, %d links)",
            session_id,
            len(graph_data.get("nodes", [])),
            len(graph_data.get("links", [])),
        )

    def load_correlation_graph(self, session_id: str) -> dict | None:
        """Load a persisted correlation graph or return None if absent."""
        key = f"augur:nexus:graph:{session_id}"
        return self._get_json(key)

    def list_correlation_graphs(self, limit: int = 50) -> list[str]:
        """Return recent session ids that have persisted correlation graphs.

        Ordered newest-first (lpush index order).
        """
        raw_ids = cast(
            list[Any], self._r.lrange("augur:nexus:graph:_index", 0, limit - 1)
        )
        return [sid.decode() if isinstance(sid, bytes) else sid for sid in raw_ids]

    # -- Rule confidence state (reflection matrix tuning) --------------------

    def save_rule_confidence(self, confidence_state: dict) -> None:
        """Store per-rule EWMA confidence + restore_target snapshots.

        Schema: {rule_key: {"confidence": float, "restore_target": str | None}}
        """
        key = "augur:nexus:escalation_confidence"
        self._set_json(key, confidence_state)
        log.debug(
            "Saved rule confidence state: %d rules",
            len(confidence_state),
        )

    def load_rule_confidence(self) -> dict | None:
        """Return the per-rule confidence state dict or None if not set."""
        key = "augur:nexus:escalation_confidence"
        return self._get_json(key)

    def save_rule_window_state(self, state: dict) -> None:
        """Persist per-rule observed-lag EWMA state.

        Schema: {rule_key: {"ewma_lag": float}}.
        Mirrors save_rule_confidence; written to a separate Redis key.
        """
        self._set_json("augur:nexus:rule_window_state", state)

    def load_rule_window_state(self) -> dict:
        """Load per-rule observed-lag EWMA state. Returns {} if absent or corrupt."""
        raw = cast(Any, self._r.get("augur:nexus:rule_window_state"))
        if not raw:
            return {}
        try:
            if isinstance(raw, bytes):
                raw = raw.decode()
            data = json.loads(raw)
            return data if isinstance(data, dict) else {}
        except (json.JSONDecodeError, UnicodeDecodeError):
            log.warning("rule_window_state Redis value was corrupt; returning empty")
            return {}

    def save_tuning_state(
        self,
        confidence: dict | None = None,
        window_state: dict | None = None,
    ) -> None:
        """Atomically persist rule_confidence + rule_window_state in one
        Redis MULTI/EXEC pipeline.

        Codex round-3 fix: previously the two state writes were separate
        calls. If the second one failed after the first succeeded, the
        confidence/window EWMAs were partially advanced and the next retry
        would double-apply. With pipeline.execute() (transaction=True is
        redis-py's default), either both SETs are committed atomically or
        neither is.
        """
        if confidence is None and window_state is None:
            return
        pipe = self._r.pipeline()  # transaction=True by default
        if confidence is not None:
            pipe.set("augur:nexus:escalation_confidence", json.dumps(confidence))
        if window_state is not None:
            pipe.set("augur:nexus:rule_window_state", json.dumps(window_state))
        pipe.execute()

    # -- Reflection reports --------------------------------------------------

    def save_reflection(self, session_id: str, report_dict: dict) -> None:
        """Persist a per-session reflection report with a 30-day TTL.

        Previously written directly via redis_client.set(...) from
        reflection_engine.py, bypassing this abstraction. Now routed here
        so the key namespace is discoverable alongside other persistence
        concerns and TTL is applied consistently.
        """
        key = f"augur:disciplina:{session_id}"
        self._set_json(key, report_dict, ex=SESSION_KEY_TTL_S)
        log.debug("Saved reflection report for session %s", session_id)

    def load_reflection(self, session_id: str) -> dict | None:
        """Return a session's reflection report or None if not set."""
        key = f"augur:disciplina:{session_id}"
        return self._get_json(key)

    # -- Last anomaly / last advice (live state, not per-session) -----------

    def save_last_anomaly(self, anomaly_dict: dict) -> None:
        """Persist the most recent anomaly event (no TTL — live state)."""
        self._set_json("augur:vigil:last_anomaly", anomaly_dict)

    def load_last_anomaly(self) -> dict | None:
        """Return the most recent anomaly event or None if not set."""
        return self._get_json("augur:vigil:last_anomaly")

    def save_last_advice(self, advice_dict: dict) -> None:
        """Persist the most recent LLM advice payload (no TTL — live state)."""
        self._set_json("augur:consilium:last_advice", advice_dict)

    def load_last_advice(self) -> dict | None:
        """Return the most recent LLM advice payload or None if not set."""
        return self._get_json("augur:consilium:last_advice")

    def save_auspices(self, snapshot: dict) -> None:
        """Overwrite the live auspices snapshot (no TTL)."""
        self._set_json("augur:imperator:auspices", snapshot)

    def load_auspices(self) -> dict | None:
        """Return the auspices snapshot, or None if absent."""
        return self._get_json("augur:imperator:auspices")

    def save_self_model(self, snapshot: dict) -> None:
        """Overwrite the live self-model snapshot (no TTL)."""
        self._set_json("augur:imperator:self_model", snapshot)

    def load_self_model(self) -> dict | None:
        """Return the self-model snapshot, or None if absent."""
        return self._get_json("augur:imperator:self_model")

    def save_proposal(self, record: dict) -> None:
        """Append a terminal-status proposal record (newest-first, capped)."""
        key = "augur:imperator:proposals"
        self._r.lpush(key, json.dumps(record))
        self._r.ltrim(key, 0, MAX_IMPERATOR_PROPOSALS - 1)

    def load_proposals(self, *, limit: int = 50) -> list[dict]:
        """Up to *limit* recent proposals, newest first. [] on corrupt/absent."""
        raw = cast(list[Any], self._r.lrange("augur:imperator:proposals", 0, limit - 1))
        try:
            return [json.loads(e) for e in raw]
        except (json.JSONDecodeError, TypeError, UnicodeDecodeError):
            log.warning(
                "augur:imperator:proposals contained a corrupt entry; returning []"
            )
            return []

    def mark_proposal_applied(self, dedupe_key: str, *, ttl_s: int) -> None:
        """Durable applied-dedup marker (TTL'd), independent of the capped log."""
        self._r.set(f"augur:imperator:applied:{dedupe_key}", "1", ex=int(ttl_s))

    def is_proposal_applied(self, dedupe_key: str) -> bool:
        """True if a recent (un-expired) apply of this dedupe_key exists."""
        return bool(cast(int, self._r.exists(f"augur:imperator:applied:{dedupe_key}")))

    def save_dialogue_turn(self, turn: dict) -> None:
        """Append a conversation turn (newest-first, capped)."""
        key = "augur:imperator:dialogue:log"
        self._r.lpush(key, json.dumps(turn))
        self._r.ltrim(key, 0, MAX_DIALOGUE_LOG - 1)

    def load_dialogue_log(
        self, *, limit: int = 50, session_id: str | None = None
    ) -> list[dict]:
        """Up to *limit* recent turns, newest first. [] on corrupt/absent.

        With *session_id* set, only turns whose stored ``session_id`` matches
        are returned (still newest-first, still capped at *limit*).

        A non-positive *limit* returns [] -- never the whole list: without a
        session filter the LRANGE stop index would be ``limit - 1 == -1``,
        which Redis reads as "through the last element" (the entire log),
        inverting limit=0 ("no history") into "all history".
        """
        if limit <= 0:
            return []
        span = (MAX_DIALOGUE_LOG if session_id is not None else limit) - 1
        raw = cast(list[Any], self._r.lrange("augur:imperator:dialogue:log", 0, span))
        try:
            turns = [json.loads(e) for e in raw]
        except (json.JSONDecodeError, TypeError, UnicodeDecodeError):
            log.warning("dialogue log contained a corrupt entry; returning []")
            return []
        if session_id is not None:
            turns = [t for t in turns if t.get("session_id") == session_id][:limit]
        return turns

    def save_dialogue_pending(self, session_id: str, pending: dict, ttl: float) -> None:
        """Store the single pending-confirmation intent for a session, with TTL.

        ``ex=max(1, int(ttl))``: a sub-second ``ttl`` (e.g. 0.5) truncates to
        0 under plain ``int()``, and Redis SET rejects ``EX 0`` outright --
        clamp to the minimum valid 1-second expiry instead of crashing.
        """
        self._set_json(
            f"augur:imperator:dialogue:pending:{session_id}",
            pending,
            ex=max(1, int(ttl)),
        )

    def load_dialogue_pending(self, session_id: str) -> dict | None:
        return self._get_json(f"augur:imperator:dialogue:pending:{session_id}")

    def clear_dialogue_pending(self, session_id: str) -> None:
        self._r.delete(f"augur:imperator:dialogue:pending:{session_id}")

    def append_dialogue_audit(self, record: dict) -> None:
        """Append a confirmed-apply/undo audit record (newest-first, capped)."""
        key = "augur:imperator:dialogue:audit"
        self._r.lpush(key, json.dumps(record))
        self._r.ltrim(key, 0, MAX_DIALOGUE_LOG - 1)

    def load_dialogue_audit(self, *, limit: int = 50) -> list[dict]:
        """Up to *limit* recent audit records, newest first. [] on corrupt/absent.

        A non-positive *limit* returns [] -- the LRANGE stop index would be
        ``limit - 1 == -1`` (the whole list) otherwise.
        """
        if limit <= 0:
            return []
        raw = cast(
            list[Any],
            self._r.lrange("augur:imperator:dialogue:audit", 0, limit - 1),
        )
        try:
            return [json.loads(e) for e in raw]
        except (json.JSONDecodeError, TypeError, UnicodeDecodeError):
            log.warning("dialogue audit log contained a corrupt entry; returning []")
            return []

    def add_dialogue_directive(self, directive: dict) -> bool:
        """Store a context directive in the augur:imperator:dialogue:directives hash.

        *directive* must include a non-empty "directive_id" key (ValueError
        otherwise). The entire dict is JSON-encoded and stored under that key;
        re-adding an existing id overwrites it (upsert).

        Refuse-at-cap (via _hash_save): returns True if written, False if a
        NEW id would exceed MAX_DIALOGUE_DIRECTIVES — existing ids keep
        updating even at cap. Callers must check the return value to report
        a truthful failure to the user.
        """
        directive_id = directive.get("directive_id")
        if not directive_id:
            raise ValueError("directive must include a non-empty 'directive_id'")
        return self._hash_save(
            "augur:imperator:dialogue:directives",
            directive_id,
            directive,
            cap=MAX_DIALOGUE_DIRECTIVES,
        )

    def remove_dialogue_directive(self, directive_id: str) -> None:
        """Remove a context directive by directive_id."""
        self._r.hdel("augur:imperator:dialogue:directives", directive_id)

    def load_dialogue_directives(self) -> list[dict]:
        """Return all stored context directives. [] on corrupt/absent.

        Each stored directive is JSON-decoded. Corrupt entries are skipped
        with a warning, and the method returns what it could decode.
        """
        raw = cast(
            dict[Any, Any],
            self._r.hgetall("augur:imperator:dialogue:directives"),
        )
        out = []
        for v in raw.values():
            try:
                out.append(json.loads(v))
            except (json.JSONDecodeError, TypeError, UnicodeDecodeError):
                continue
        return out

    def get_dialogue_directive(self, directive_id: str) -> dict | None:
        """Return a single stored context directive by id, or None if absent
        or corrupt. Used by apply.py to read the PRIOR content of a directive
        BEFORE a create/upsert or remove overwrites/deletes it, so the write
        can record a true restore anchor (apply.py's rollback-anchor
        discipline, spec §8)."""
        raw = cast(
            Any, self._r.hget("augur:imperator:dialogue:directives", directive_id)
        )
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, TypeError, UnicodeDecodeError):
            return None

    # -- Current session --------------------------------------------------

    def load_current_session(self) -> dict | None:
        """Return the current session dict written by SessionManager.

        R2-ARCH-01: previously MCP tools reached through ``pm._r`` to read
        ``augur:session:current`` directly. Expose it as a proper PM
        method so the key namespace is discoverable and the private
        client is not leaking.
        """
        return self._get_json("augur:session:current")

    # -- Correlation tuning idempotency marker ------------------------------

    def mark_tuning_applied(
        self, session_id: str, *, pass_name: str = "correlation"
    ) -> None:
        """Mark a session as having had a named tuning pass applied.

        Key: ``augur:tuning_applied:{pass_name}:{session_id}``

        Different passes (e.g. "correlation", "gate") use independent markers
        so they can be idempotent independently.  Sets a 7-day TTL so manual
        reflect-trigger replays of a recent session do not double-apply.
        """
        key = f"augur:tuning_applied:{pass_name}:{session_id}"
        self._r.set(key, "1", ex=TUNING_APPLIED_TTL_S)

    def is_tuning_applied(
        self, session_id: str, *, pass_name: str = "correlation"
    ) -> bool:
        """Return True if mark_tuning_applied was called for this session+pass."""
        key = f"augur:tuning_applied:{pass_name}:{session_id}"
        return bool(cast(int, self._r.exists(key)))

    # ── Advisor gate append-logs (spec §6) ───────────────────────────────────
    # All four follow the same pattern: lpush + ltrim (mirror save_feedback
    # index at persistence.py:78-79).  Loaders decode each entry individually
    # inside a guarded comprehension so a single corrupt entry returns []
    # rather than raising (mirror load_rule_window_state corrupt-read guard,
    # persistence.py:297).

    def save_silence_record(self, record: dict) -> None:
        """Append a gate suppression record to augur:limen:silences (capped).

        Schema: {ts, decision_id, state_key, domain, entity, severity, arm,
                 reason, metrics, mrt_eligible, p_withhold}
        """
        key = "augur:limen:silences"
        self._r.lpush(key, json.dumps(record))
        self._r.ltrim(key, 0, MAX_GATE_SILENCES - 1)

    def load_silence_records(self, *, limit: int = 100) -> list[dict]:
        """Return up to *limit* recent silence records, newest first.

        Returns [] if the list is absent or any entry is corrupt.
        """
        key = "augur:limen:silences"
        raw_list = cast(list[Any], self._r.lrange(key, 0, limit - 1))
        try:
            return [json.loads(entry) for entry in raw_list]
        except (json.JSONDecodeError, TypeError, UnicodeDecodeError):
            log.warning("augur:limen:silences contained a corrupt entry; returning []")
            return []

    def save_emission(self, record: dict) -> None:
        """Append a gate emission record to augur:limen:emissions (capped).

        Schema: {ts, decision_id, state_key, severity, tier, probe,
                 audit_only, withheld_reason, mrt_eligible, p_fire}
        """
        key = "augur:limen:emissions"
        self._r.lpush(key, json.dumps(record))
        self._r.ltrim(key, 0, MAX_GATE_EMISSIONS - 1)

    def load_emissions(self, *, limit: int = 100) -> list[dict]:
        """Return up to *limit* recent emission records, newest first.

        Returns [] if the list is absent or any entry is corrupt.
        """
        key = "augur:limen:emissions"
        raw_list = cast(list[Any], self._r.lrange(key, 0, limit - 1))
        try:
            return [json.loads(entry) for entry in raw_list]
        except (json.JSONDecodeError, TypeError, UnicodeDecodeError):
            log.warning("augur:limen:emissions contained a corrupt entry; returning []")
            return []

    def save_observed(self, record: dict) -> None:
        """Append a gate observed-value record to augur:limen:observed (capped).

        Schema: {ts, state_key, value, severity}
        Written by both record_suppression and non-probe record_delivery_success.
        """
        key = "augur:limen:observed"
        self._r.lpush(key, json.dumps(record))
        self._r.ltrim(key, 0, MAX_GATE_OBSERVED - 1)

    def load_observed(self, state_key: str, *, limit: int = 100) -> list[dict]:
        """Return up to *limit* observed records for *state_key*, newest first.

        Returns [] if absent or any entry is corrupt.  Filters by state_key
        after reading so the caller gets only the records relevant to one channel.
        """
        key = "augur:limen:observed"
        raw_list = cast(list[Any], self._r.lrange(key, 0, limit - 1))
        try:
            all_records = [json.loads(entry) for entry in raw_list]
        except (json.JSONDecodeError, TypeError, UnicodeDecodeError):
            log.warning("augur:limen:observed contained a corrupt entry; returning []")
            return []
        return [r for r in all_records if r.get("state_key") == state_key]

    def save_delivery_failure(
        self,
        signature: object,
        reason: str,
        now: str,
        decision_id: str,
    ) -> None:
        """Append a delivery-failure record to augur:limen:delivery_failures (capped).

        Schema: {ts, decision_id, state_key, domain, entity, reason}
        Used for infrastructure non-deliveries (busy-skip, Ollama failure, etc.).
        *signature* must expose .state_key, .domain, and .entity attributes.
        """
        record = {
            "ts": now,
            "decision_id": decision_id,
            "state_key": getattr(signature, "state_key", None),
            "domain": getattr(signature, "domain", None),
            "entity": getattr(signature, "entity", None),
            "reason": reason,
        }
        key = "augur:limen:delivery_failures"
        self._r.lpush(key, json.dumps(record))
        self._r.ltrim(key, 0, MAX_GATE_DELIVERY_FAILURES - 1)

    def load_delivery_failures(self, *, limit: int = 100) -> list[dict]:
        """Return up to *limit* recent delivery-failure records, newest first.

        Returns [] if the list is absent or any entry is corrupt.
        """
        key = "augur:limen:delivery_failures"
        raw_list = cast(list[Any], self._r.lrange(key, 0, limit - 1))
        try:
            return [json.loads(entry) for entry in raw_list]
        except (json.JSONDecodeError, TypeError, UnicodeDecodeError):
            log.warning(
                "augur:limen:delivery_failures contained a corrupt entry; returning []"
            )
            return []

    # ── Task 1.2: per-field hash stores (spec §6) ────────────────────────────
    # Each save_* returns bool: True if the write succeeded (existing key or
    # room below the cap), False if refused at cap (new key would exceed
    # MAX_GATE_STATE_KEYS).  Mirrors the save_app_descriptor refuse-at-cap
    # pattern (persistence.py:203-216).
    #
    # Each load_* guards json.loads / UnicodeDecodeError → {} (safe "unseen"
    # default), mirroring load_rule_window_state (persistence.py:312-319).
    #
    # habituation and habituation_floor are SEPARATE Redis keys (spec §6).

    def _hash_save(
        self, redis_key: str, field: str, entry: dict, *, cap: int | None = None
    ) -> bool:
        """Per-field HSET with refuse-at-cap. Returns True if written.

        *cap* defaults to MAX_GATE_STATE_KEYS (the gate hash stores) — resolved
        at call time so tests can monkeypatch the module constant; other
        hash-backed stores (e.g. dialogue directives) pass their own cap.
        """
        if cap is None:
            cap = MAX_GATE_STATE_KEYS
        if (
            not cast(bool, self._r.hexists(redis_key, field))
            and cast(int, self._r.hlen(redis_key)) >= cap
        ):
            log.warning(
                "hash %s full (%d); dropping new field %s",
                redis_key,
                cap,
                field,
            )
            return False
        self._r.hset(redis_key, field, json.dumps(entry))
        return True

    def _hash_load(self, redis_key: str, field: str) -> dict:
        """Per-field HGET with corrupt-read guard. Returns {} on missing/corrupt."""
        raw = cast(Any, self._r.hget(redis_key, field))
        if raw is None:
            return {}
        try:
            if isinstance(raw, bytes):
                raw = raw.decode()
            data = json.loads(raw)
            return data if isinstance(data, dict) else {}
        except (json.JSONDecodeError, TypeError, UnicodeDecodeError):
            log.warning("gate hash %s field %s corrupt; returning {}", redis_key, field)
            return {}

    # -- habituation (online: h, last_event_ts, count) -----------------------

    def save_habituation(self, state_key: str, entry: dict) -> bool:
        """Store per-channel habituation state. Returns False if refused at cap."""
        return self._hash_save("augur:limen:habituation", state_key, entry)

    def load_habituation(self, state_key: str) -> dict:
        """Return habituation entry for state_key, or {} if missing/corrupt."""
        return self._hash_load("augur:limen:habituation", state_key)

    # -- habituation_floor (offline: floor, last_ts) — separate Redis key ----

    def save_habituation_floor(self, state_key: str, entry: dict) -> bool:
        """Store per-channel habituation floor. Returns False if refused at cap."""
        return self._hash_save("augur:limen:habituation_floor", state_key, entry)

    def load_habituation_floor(self, state_key: str) -> dict:
        """Return habituation floor for state_key, or {} if missing/corrupt."""
        return self._hash_load("augur:limen:habituation_floor", state_key)

    # -- credibility (offline + conservative online: cred, n, last_fb_ts) ----

    def save_credibility(self, signal_class: str, entry: dict) -> bool:
        """Store per-class credibility state. Returns False if refused at cap."""
        return self._hash_save("augur:limen:credibility", signal_class, entry)

    def load_credibility(self, signal_class: str) -> dict:
        """Return credibility entry for signal_class, or {} if missing/corrupt."""
        return self._hash_load("augur:limen:credibility", signal_class)

    # -- reservoir (online: count, last_ts) -----------------------------------

    def save_reservoir(self, state_key: str, entry: dict) -> bool:
        """Store per-channel reservoir state. Returns False if refused at cap."""
        return self._hash_save("augur:limen:reservoir", state_key, entry)

    def load_reservoir(self, state_key: str) -> dict:
        """Return reservoir entry for state_key, or {} if missing/corrupt."""
        return self._hash_load("augur:limen:reservoir", state_key)

    # -- cost_tier_memory (online + offline: earned_tier2, helped, count, last_ts)

    def save_cost_tier_memory(self, state_key: str, entry: dict) -> bool:
        """Store per-channel cost-tier memory. Returns False if refused at cap."""
        return self._hash_save("augur:limen:cost_tier_memory", state_key, entry)

    def load_cost_tier_memory(self, state_key: str) -> dict:
        """Return cost-tier memory for state_key, or {} if missing/corrupt."""
        return self._hash_load("augur:limen:cost_tier_memory", state_key)

    # -- channel_stats (online: seen, consecutive_suppressions, ...) ----------

    def save_channel_stats(self, state_key: str, entry: dict) -> bool:
        """Store per-channel tracking stats. Returns False if refused at cap."""
        return self._hash_save("augur:limen:channel_stats", state_key, entry)

    def load_channel_stats(self, state_key: str) -> dict:
        """Return channel stats for state_key, or {} if missing/corrupt."""
        return self._hash_load("augur:limen:channel_stats", state_key)

    def load_all_channel_stats(self) -> dict:
        """HGETALL augur:limen:channel_stats → {state_key: stats_dict}. {} if absent."""
        raw = cast(dict[Any, Any], self._r.hgetall("augur:limen:channel_stats"))
        out: dict[str, dict] = {}
        for fld, val in raw.items():
            try:
                fld_s = fld.decode() if isinstance(fld, bytes) else fld
                out[fld_s] = json.loads(val)
            except (json.JSONDecodeError, TypeError, UnicodeDecodeError):
                continue
        return out

    # -- advice_rate (string key: rate_ewma, last_ts) -------------------------

    def save_advice_rate(self, entry: dict) -> None:
        """Persist the global advice-rate EWMA state."""
        self._set_json("augur:limen:advice_rate", entry)

    def load_advice_rate(self) -> dict:
        """Return advice-rate state, or {} if missing/corrupt."""
        raw = cast(Any, self._r.get("augur:limen:advice_rate"))
        if not raw:
            return {}
        try:
            if isinstance(raw, bytes):
                raw = raw.decode()
            data = json.loads(raw)
            return data if isinstance(data, dict) else {}
        except (json.JSONDecodeError, TypeError, UnicodeDecodeError):
            log.warning("augur:limen:advice_rate corrupt; returning {}")
            return {}

    # -- self_tolerance set (offline: SADD/SREM/SISMEMBER/SMEMBERS) -----------

    def add_self_tolerance(self, state_key: str) -> None:
        """Add state_key to the self-tolerance set."""
        self._r.sadd("augur:limen:self_tolerance", state_key)

    def remove_self_tolerance(self, state_key: str) -> None:
        """Remove state_key from the self-tolerance set."""
        self._r.srem("augur:limen:self_tolerance", state_key)

    def is_self_tolerant(self, state_key: str) -> bool:
        """Return True if state_key is in the self-tolerance set."""
        return bool(
            cast(
                bool,
                self._r.sismember("augur:limen:self_tolerance", state_key),
            )
        )

    def load_self_tolerance(self) -> set[str]:
        """Return the full self-tolerance set (decoded to str)."""
        raw = cast(set[Any], self._r.smembers("augur:limen:self_tolerance"))
        return {m.decode() if isinstance(m, bytes) else m for m in raw}

    # -- can_track_gate_state (read-only cap probe) ---------------------------

    def can_track_gate_state(self, hash_name: str, state_key: str) -> bool:
        """Return True if state_key can be stored in hash_name without exceeding cap.

        True when the field already exists (existing key keeps updating even at
        cap) or when hlen < MAX_GATE_STATE_KEYS (room available).  This is the
        read-only probe used by Gate.evaluate() to detect the cap-fail-open
        condition without writing.
        """
        return bool(
            cast(bool, self._r.hexists(hash_name, state_key))
            or cast(int, self._r.hlen(hash_name)) < MAX_GATE_STATE_KEYS
        )

    # ── Task 1.3: atomic gate tuning save (spec §6) ──────────────────────────

    def save_gate_tuning_state(
        self,
        *,
        floors: dict | None = None,
        credibility: dict | None = None,
        cost_tier: dict | None = None,
        tolerance_add: list[str] | None = None,
        tolerance_remove: list[str] | None = None,
        advice_rate: dict | None = None,
    ) -> None:
        """Atomically persist all gate offline keys in one pipeline (transaction).

        Uses per-field HSET for the dual-written hashes (floors, credibility,
        cost_tier) so online writers on other fields are never clobbered.
        Uses SADD/SREM for the self-tolerance set and SET for advice_rate.

        Mirrors save_tuning_state (persistence.py:326-348) but covers the gate
        offline keys.  pipeline(transaction=True) is redis-py's default.
        """
        pipe = self._r.pipeline()  # transaction=True by default
        if floors:
            for field, entry in floors.items():
                pipe.hset("augur:limen:habituation_floor", field, json.dumps(entry))
        if credibility:
            for field, entry in credibility.items():
                pipe.hset("augur:limen:credibility", field, json.dumps(entry))
        if cost_tier:
            for field, entry in cost_tier.items():
                pipe.hset("augur:limen:cost_tier_memory", field, json.dumps(entry))
        if tolerance_add:
            for state_key in tolerance_add:
                pipe.sadd("augur:limen:self_tolerance", state_key)
        if tolerance_remove:
            for state_key in tolerance_remove:
                pipe.srem("augur:limen:self_tolerance", state_key)
        if advice_rate is not None:
            pipe.set("augur:limen:advice_rate", json.dumps(advice_rate))
        pipe.execute()

    # ── Memoria memory spine (Lane 2, spec 2026-06-10) ───────────────────────
    # Dumb transactional storage only; all decay/promote/prune POLICY lives in
    # the pure memoria/ package. Keys: augur:memoria:{dsr,tier,archive,
    # processed_sessions}. The processed_sessions SET is both the idempotency
    # gate and the active-session decay clock (SCARD).

    def save_memory_state(self, memory_id: str, state: dict) -> None:
        """Insert/update a memory's DSR state + tier index. Keeps the tier
        index single-membership (drops the other tier first) so a same-key
        re-save with a changed tier cannot leave a stale index entry."""
        self._r.set(f"augur:memoria:dsr:{memory_id}", json.dumps(state))
        other = "cold" if state["tier"] == "warm" else "warm"
        self._r.srem(f"augur:memoria:tier:{other}", memory_id)
        self._r.sadd(f"augur:memoria:tier:{state['tier']}", memory_id)

    def load_memory_state(self, memory_id: str) -> dict | None:
        return self._get_json(f"augur:memoria:dsr:{memory_id}")

    def list_memory_ids(self, tier: str) -> list[str]:
        # Decode set members: the live connect_redis client has NO
        # decode_responses=True (tabula/connections.py), so smembers() returns
        # bytes in production; fakeredis(decode_responses=True) returns str.
        return sorted(
            (m.decode() if isinstance(m, bytes) else m)
            for m in cast(set[Any], self._r.smembers(f"augur:memoria:tier:{tier}"))
        )

    def load_all_memory_states(self) -> list[dict]:
        out: list[dict] = []
        for tier in ("warm", "cold"):
            mids = [
                m.decode() if isinstance(m, bytes) else m
                for m in cast(set[Any], self._r.smembers(f"augur:memoria:tier:{tier}"))
            ]
            if not mids:
                continue
            # Batch the per-member reads into one MGET instead of a synchronous
            # GET per id: the taught-facts advice injection (advisor.py) reaches
            # this on the advice hot path, where one round-trip per memory scaled
            # linearly with total memory count. MGET preserves key order, and
            # each value is decoded independently, so the result is identical.
            raws = cast(
                list[Any],
                self._r.mget([f"augur:memoria:dsr:{mid}" for mid in mids]),
            )
            out.extend(json.loads(raw) for raw in raws if raw is not None)
        return out

    def load_archived_memory(self, memory_id: str) -> dict | None:
        return self._get_json(f"augur:memoria:archive:{memory_id}")

    def is_session_processed(self, session_id: str) -> bool:
        return bool(
            cast(
                bool,
                self._r.sismember("augur:memoria:processed_sessions", session_id),
            )
        )

    def active_session_count(self) -> int:
        return int(cast(int, self._r.scard("augur:memoria:processed_sessions")))

    def record_memory_review(
        self, memory_id: str, session_id: str, active_session: int
    ) -> None:
        """C2 hook: single-memory recurrence review (load → review → save)."""
        from tabula.config import AugurConfig
        from memoria.fsrs import review

        st = self.load_memory_state(memory_id)
        if st is None:
            return
        self.save_memory_state(
            memory_id, review(st, active_session, session_id, AugurConfig.from_env())
        )

    def apply_memory_sweep(self, session_id: str, plan) -> bool:
        """Atomically apply a SweepPlan AND record the session, or no-op.

        Mirrors the save_tuning_state MULTI/EXEC discipline, with a WATCH on
        processed_sessions so the session is committed exactly once (commit-
        last: nothing persists until this transaction). Returns False if the
        session was already processed.
        """
        pset = "augur:memoria:processed_sessions"
        with self._r.pipeline() as pipe:
            while True:
                try:
                    pipe.watch(pset)
                    if pipe.sismember(pset, session_id):
                        pipe.unwatch()
                        return False
                    pipe.multi()
                    for st in plan.creates:
                        pipe.set(f"augur:memoria:dsr:{st['memory_id']}", json.dumps(st))
                        pipe.sadd(f"augur:memoria:tier:{st['tier']}", st["memory_id"])
                    for st in plan.reviews:
                        pipe.set(f"augur:memoria:dsr:{st['memory_id']}", json.dumps(st))
                    for st in plan.promotions:
                        pipe.smove(
                            "augur:memoria:tier:warm",
                            "augur:memoria:tier:cold",
                            st["memory_id"],
                        )
                        pipe.set(f"augur:memoria:dsr:{st['memory_id']}", json.dumps(st))
                    for st in plan.demotions:
                        pipe.smove(
                            "augur:memoria:tier:cold",
                            "augur:memoria:tier:warm",
                            st["memory_id"],
                        )
                        pipe.set(f"augur:memoria:dsr:{st['memory_id']}", json.dumps(st))
                    for st in plan.prunes:
                        archived = {**st, "status": "archived"}
                        pipe.set(
                            f"augur:memoria:archive:{st['memory_id']}",
                            json.dumps(archived),
                        )
                        pipe.srem(f"augur:memoria:tier:{st['tier']}", st["memory_id"])
                        pipe.delete(f"augur:memoria:dsr:{st['memory_id']}")
                    pipe.sadd(pset, session_id)
                    pipe.execute()
                    return True
                except redis.WatchError:  # pragma: no cover
                    # Single-writer in practice (Disciplina reflection_lock); a
                    # concurrent commit of THIS session is caught by the
                    # sismember re-check on retry.
                    continue

    def create_user_taught_memory(
        self,
        pattern: dict,
        *,
        source: str,
        protect: bool = True,
        session_id: str | None = None,
        cfg: Any = None,
        rationale: str | None = None,
    ) -> str:
        """Create OR re-teach a user-taught semantic memory with FSRS decay.

        Pattern must include: kind="semantic", domains, rule_key, severity
        (ValueError if kind is not "semantic" -- this API owns semantic facts
        only).  Returns the memory_id (deterministic from pattern).

        Re-teaching an EXISTING taught fact -- the pattern hashes to the same
        deterministic memory_id, per make_memory_id's "recurrence == review"
        contract -- is a real FSRS recurrence review via memoria.fsrs.review,
        the SAME path Disciplina's memory sweep uses (memoria/tiers.py
        plan_sweep / tabula/persistence.py record_memory_review): S
        strengthens and last_review_session advances, never resetting decay
        progress. The stored pattern/taught_by/origin_severity are refreshed
        to the newly-taught values (content is updatable even though only
        pattern's identity fields -- kind/domains/rule_key/severity -- affect
        the memory_id), and status is reactivated to "active" (undoes the
        status-flip archival apply.py's semantic_fact remove handler uses). A
        brand-new pattern is created fresh, exactly as before.

        Args:
            pattern: dict with {kind: "semantic", domains, rule_key, severity}
            source: who taught this memory (e.g., "user")
            protect: if True, set origin_severity="HIGH"; else use pattern.severity
            session_id: dialogue session id, passed to the FSRS review on
                re-teach (idempotent per session, mirroring record_memory_review)
            cfg: AugurConfig for the review's FSRS knobs (memory_s_growth_factor,
                memory_s_max) -- only read on re-teach; defaults to
                AugurConfig.from_env() when not supplied
            rationale: the user's free-text teaching rationale, stored as a
                record-level sibling of pattern (NOT inside pattern, so the
                deterministic memory_id is unaffected). Consilium's
                format_taught_facts prefers this text over the rule_key slug
                when rendering the advice-prompt block. On re-teach a
                non-empty rationale replaces the stored one; None/empty
                PRESERVES the existing rationale (a bare re-teach must not
                erase prior teaching text).

        Returns:
            memory_id (SHA-256 of canonical pattern)
        """
        if pattern.get("kind") != "semantic":
            raise ValueError(
                "taught memory pattern must be kind='semantic', got "
                f"{pattern.get('kind')!r}"
            )
        mid = make_memory_id(pattern)
        existing = self.load_memory_state(mid)
        active = self.active_session_count()
        origin_severity = "HIGH" if protect else pattern.get("severity", "LOW")
        if existing is not None:
            from memoria.fsrs import review

            if cfg is None:
                from tabula.config import AugurConfig

                cfg = AugurConfig.from_env()
            reviewed = review(existing, active, session_id or "", cfg)
            reviewed = {
                **reviewed,
                "pattern": pattern,
                "taught_by": source,
                "origin_severity": origin_severity,
                "status": "active",
                # non-empty replaces; None/empty preserves prior teaching text
                **({"rationale": rationale} if rationale else {}),
            }
            self.save_memory_state(mid, reviewed)
            return mid
        state: dict[str, Any] = {
            "memory_id": mid,
            "pattern": pattern,
            "S": 1.0,
            "D": 5.0,
            "last_review_session": active,
            "tier": "warm",
            "status": "active",
            "origin_severity": origin_severity,
            "memory_kind": "semantic",
            "source_sessions": [],
            "taught_by": source,
            "rationale": rationale,
        }
        self.save_memory_state(mid, state)
        return mid

    def load_taught_facts(self) -> list[dict]:
        """Load all ACTIVE taught semantic memories.

        Filters all memory states for pattern.kind == "semantic" AND
        status == "active" (a missing status key counts as active, for
        taught facts stored before the field existed). Archived facts --
        apply.py's semantic_fact remove handler flips status on the live
        record -- must NOT surface here: the dialogue context feeds this
        list verbatim into what the LLM sees, so a user-confirmed
        "forget X" has to actually drop X from every subsequent turn.
        Returns a list of memory dicts (from both warm and cold tiers).
        """
        return [
            m
            for m in self.load_all_memory_states()
            if (m.get("pattern") or {}).get("kind") == "semantic"
            and m.get("status", "active") == "active"
        ]

    def load_taught_facts_for_domains(self, domains: list[str]) -> list[dict]:
        """Load ACTIVE taught semantic memories matching any domain
        (inherits load_taught_facts' status filter).

        Args:
            domains: list of domain names to filter by (case-insensitive)

        Returns:
            list of memory dicts whose pattern.domains overlap with the input list
        """
        want = {d.lower() for d in domains}
        out = []
        for m in self.load_taught_facts():
            mdoms = {d.lower() for d in (m.get("pattern") or {}).get("domains", [])}
            if mdoms & want:
                out.append(m)
        return out
