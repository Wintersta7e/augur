"""Persistence layer — Redis-backed storage for baselines, history, feedback,
prompt versioning, and threshold configuration."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

import redis

from tabula.contracts import PerceptionEvent

log = logging.getLogger("persistence")

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

    # -- Baseline persistence ------------------------------------------------

    def save_baseline(self, domain: str, entity: str, state_dict: dict) -> None:
        key = f"augur:vigil:profile:{domain}:{entity}"
        self._r.set(key, json.dumps(state_dict))
        log.debug("Saved baseline %s", key)

    def load_baseline(self, domain: str, entity: str) -> dict | None:
        key = f"augur:vigil:profile:{domain}:{entity}"
        raw = self._r.get(key)
        if raw is None:
            return None
        return json.loads(raw)

    def scan_baseline_maturity(self, *, trained_obs: int = 15) -> dict:
        """Tally Vigil baselines by training maturity via SCAN (never KEYS).

        Returns {total, trained, untrained, by_domain:{domain:{total,trained}}}.
        Trained = observation_count >= trained_obs. Key: augur:vigil:profile:{domain}:{entity}.
        """
        total = trained = 0
        by_domain: dict[str, dict[str, int]] = {}
        for key in self._r.scan_iter(match="augur:vigil:profile:*", count=500):
            raw = self._r.get(key)
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
        raw_list = self._r.lrange(key, 0, limit - 1)
        return [json.loads(entry) for entry in raw_list]

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
        raw = self._r.get(key)
        if raw is None:
            return None
        return json.loads(raw)

    def get_all_feedback(self, limit: int = 50) -> list[dict]:
        session_ids = self._r.lrange("augur:responsum:_index", 0, limit - 1)
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
        existing = self._r.get(current_key)
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
        raw = self._r.get(current_key)
        if raw is None:
            return None
        return json.loads(raw).get("prompt")

    def get_prompt_history(self, domain: str, limit: int = 10) -> list[dict]:
        history_key = f"augur:consilium:prompts:{domain}:history"
        raw_list = self._r.lrange(history_key, 0, limit - 1)
        return [json.loads(entry) for entry in raw_list]

    def rollback_prompt(self, domain: str) -> bool:
        """Restore previous prompt version. Returns True if rollback succeeded."""
        current_key = f"augur:consilium:prompts:{domain}:current"
        history_key = f"augur:consilium:prompts:{domain}:history"

        previous_raw = self._r.lpop(history_key)
        if previous_raw is None:
            log.warning("No previous prompt to rollback to for domain %s", domain)
            return False

        # Archive the current (bad) version into history tail
        current_raw = self._r.get(current_key)
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
        raw = self._r.get(current_key)
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
        cur_raw = self._r.get(current_key)
        cur = json.loads(cur_raw).get("score") if cur_raw else None
        prev_list = self._r.lrange(history_key, 0, 0)
        prev = json.loads(prev_list[0]).get("score") if prev_list else None
        return cur, prev

    # -- MRT withheld-rating calibration tracking (spec 1B) ------------------

    def mark_mrt_rating_session(self, session_id: str) -> None:
        """Record that a session issued >=1 withheld-rating prompt (set: dedups)."""
        self._r.sadd("augur:limen:mrt_rating_sessions", session_id)

    def count_mrt_rating_sessions(self) -> int:
        """Number of distinct sessions that issued a withheld-rating prompt."""
        return int(self._r.scard("augur:limen:mrt_rating_sessions"))

    # -- Threshold config ----------------------------------------------------

    def save_thresholds(self, domain: str, thresholds_dict: dict) -> None:
        key = f"augur:vigil:thresholds:{domain}"
        self._r.set(key, json.dumps(thresholds_dict))
        log.debug("Saved thresholds for domain %s", domain)

    def load_thresholds(self, domain: str) -> dict | None:
        key = f"augur:vigil:thresholds:{domain}"
        raw = self._r.get(key)
        if raw is None:
            return None
        return json.loads(raw)

    # -- Escalation matrix (cross-domain correlator) -------------------------

    def save_escalation_matrix(self, matrix: dict) -> None:
        """Store the full matrix dict (including 'version' and 'rules' keys) as-is."""
        key = "augur:nexus:matrix"
        self._r.set(key, json.dumps(matrix))
        log.debug("Saved escalation matrix (version=%s)", matrix.get("version"))

    def load_escalation_matrix(self) -> dict | None:
        """Return the full matrix dict or None if not set."""
        key = "augur:nexus:matrix"
        raw = self._r.get(key)
        if raw is None:
            return None
        return json.loads(raw)

    def save_health_snapshot(self, snapshot: dict) -> None:
        """Store the Praefectus health snapshot (overwritten each tick; no TTL — live state)."""
        key = "augur:praefectus:health"
        self._r.set(key, json.dumps(snapshot))

    def load_health_snapshot(self) -> dict | None:
        """Return the last Praefectus health snapshot or None if not set."""
        key = "augur:praefectus:health"
        raw = self._r.get(key)
        if raw is None:
            return None
        return json.loads(raw)

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
            not self._r.hexists(key, entity)
            and self._r.hlen(key) >= MAX_APP_DESCRIPTORS
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
        raw = self._r.hget("augur:consilium:app_descriptors", entity)
        if raw is None:
            return None
        return raw.decode() if isinstance(raw, bytes) else raw

    def load_app_descriptors(self) -> dict[str, str]:
        """Return the full app->descriptor map with keys/values decoded to str."""
        raw = self._r.hgetall("augur:consilium:app_descriptors")
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
        raw = self._r.get(key)
        if raw is None:
            return None
        return json.loads(raw)

    def list_correlation_graphs(self, limit: int = 50) -> list[str]:
        """Return recent session ids that have persisted correlation graphs.

        Ordered newest-first (lpush index order).
        """
        raw_ids = self._r.lrange("augur:nexus:graph:_index", 0, limit - 1)
        return [sid.decode() if isinstance(sid, bytes) else sid for sid in raw_ids]

    # -- Rule confidence state (reflection matrix tuning) --------------------

    def save_rule_confidence(self, confidence_state: dict) -> None:
        """Store per-rule EWMA confidence + restore_target snapshots.

        Schema: {rule_key: {"confidence": float, "restore_target": str | None}}
        """
        key = "augur:nexus:escalation_confidence"
        self._r.set(key, json.dumps(confidence_state))
        log.debug(
            "Saved rule confidence state: %d rules",
            len(confidence_state),
        )

    def load_rule_confidence(self) -> dict | None:
        """Return the per-rule confidence state dict or None if not set."""
        key = "augur:nexus:escalation_confidence"
        raw = self._r.get(key)
        if raw is None:
            return None
        return json.loads(raw)

    def save_rule_window_state(self, state: dict) -> None:
        """Persist per-rule observed-lag EWMA state.

        Schema: {rule_key: {"ewma_lag": float}}.
        Mirrors save_rule_confidence; written to a separate Redis key.
        """
        self._r.set("augur:nexus:rule_window_state", json.dumps(state))

    def load_rule_window_state(self) -> dict:
        """Load per-rule observed-lag EWMA state. Returns {} if absent or corrupt."""
        raw = self._r.get("augur:nexus:rule_window_state")
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
        self._r.set(key, json.dumps(report_dict), ex=SESSION_KEY_TTL_S)
        log.debug("Saved reflection report for session %s", session_id)

    def load_reflection(self, session_id: str) -> dict | None:
        """Return a session's reflection report or None if not set."""
        key = f"augur:disciplina:{session_id}"
        raw = self._r.get(key)
        if raw is None:
            return None
        return json.loads(raw)

    # -- Last anomaly / last advice (live state, not per-session) -----------

    def save_last_anomaly(self, anomaly_dict: dict) -> None:
        """Persist the most recent anomaly event (no TTL — live state)."""
        self._r.set("augur:vigil:last_anomaly", json.dumps(anomaly_dict))

    def load_last_anomaly(self) -> dict | None:
        """Return the most recent anomaly event or None if not set."""
        raw = self._r.get("augur:vigil:last_anomaly")
        if raw is None:
            return None
        return json.loads(raw)

    def save_last_advice(self, advice_dict: dict) -> None:
        """Persist the most recent LLM advice payload (no TTL — live state)."""
        self._r.set("augur:consilium:last_advice", json.dumps(advice_dict))

    def load_last_advice(self) -> dict | None:
        """Return the most recent LLM advice payload or None if not set."""
        raw = self._r.get("augur:consilium:last_advice")
        if raw is None:
            return None
        return json.loads(raw)

    def save_auspices(self, snapshot: dict) -> None:
        """Overwrite the live auspices snapshot (no TTL)."""
        self._r.set("augur:imperator:auspices", json.dumps(snapshot))

    def load_auspices(self) -> dict | None:
        """Return the auspices snapshot, or None if absent."""
        raw = self._r.get("augur:imperator:auspices")
        return None if raw is None else json.loads(raw)

    def save_self_model(self, snapshot: dict) -> None:
        """Overwrite the live self-model snapshot (no TTL)."""
        self._r.set("augur:imperator:self_model", json.dumps(snapshot))

    def load_self_model(self) -> dict | None:
        """Return the self-model snapshot, or None if absent."""
        raw = self._r.get("augur:imperator:self_model")
        return None if raw is None else json.loads(raw)

    def save_proposal(self, record: dict) -> None:
        """Append a terminal-status proposal record (newest-first, capped)."""
        key = "augur:imperator:proposals"
        self._r.lpush(key, json.dumps(record))
        self._r.ltrim(key, 0, MAX_IMPERATOR_PROPOSALS - 1)

    def load_proposals(self, *, limit: int = 50) -> list[dict]:
        """Up to *limit* recent proposals, newest first. [] on corrupt/absent."""
        raw = self._r.lrange("augur:imperator:proposals", 0, limit - 1)
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
        return bool(self._r.exists(f"augur:imperator:applied:{dedupe_key}"))

    # -- Current session --------------------------------------------------

    def load_current_session(self) -> dict | None:
        """Return the current session dict written by SessionManager.

        R2-ARCH-01: previously MCP tools reached through ``pm._r`` to read
        ``augur:session:current`` directly. Expose it as a proper PM
        method so the key namespace is discoverable and the private
        client is not leaking.
        """
        raw = self._r.get("augur:session:current")
        if raw is None:
            return None
        return json.loads(raw)

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
        return bool(self._r.exists(key))

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
        raw_list = self._r.lrange(key, 0, limit - 1)
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
        raw_list = self._r.lrange(key, 0, limit - 1)
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
        raw_list = self._r.lrange(key, 0, limit - 1)
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
        raw_list = self._r.lrange(key, 0, limit - 1)
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

    def _hash_save(self, redis_key: str, field: str, entry: dict) -> bool:
        """Per-field HSET with refuse-at-cap. Returns True if written."""
        if (
            not self._r.hexists(redis_key, field)
            and self._r.hlen(redis_key) >= MAX_GATE_STATE_KEYS
        ):
            log.warning(
                "gate hash %s full (%d); dropping new field %s",
                redis_key,
                MAX_GATE_STATE_KEYS,
                field,
            )
            return False
        self._r.hset(redis_key, field, json.dumps(entry))
        return True

    def _hash_load(self, redis_key: str, field: str) -> dict:
        """Per-field HGET with corrupt-read guard. Returns {} on missing/corrupt."""
        raw = self._r.hget(redis_key, field)
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
        raw = self._r.hgetall("augur:limen:channel_stats")
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
        self._r.set("augur:limen:advice_rate", json.dumps(entry))

    def load_advice_rate(self) -> dict:
        """Return advice-rate state, or {} if missing/corrupt."""
        raw = self._r.get("augur:limen:advice_rate")
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
        return bool(self._r.sismember("augur:limen:self_tolerance", state_key))

    def load_self_tolerance(self) -> set[str]:
        """Return the full self-tolerance set (decoded to str)."""
        raw = self._r.smembers("augur:limen:self_tolerance")
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
            self._r.hexists(hash_name, state_key)
            or self._r.hlen(hash_name) < MAX_GATE_STATE_KEYS
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
        raw = self._r.get(f"augur:memoria:dsr:{memory_id}")
        return None if raw is None else json.loads(raw)

    def list_memory_ids(self, tier: str) -> list[str]:
        # Decode set members: the live connect_redis client has NO
        # decode_responses=True (tabula/connections.py), so smembers() returns
        # bytes in production; fakeredis(decode_responses=True) returns str.
        return sorted(
            (m.decode() if isinstance(m, bytes) else m)
            for m in self._r.smembers(f"augur:memoria:tier:{tier}")
        )

    def load_all_memory_states(self) -> list[dict]:
        out: list[dict] = []
        for tier in ("warm", "cold"):
            for mid in self._r.smembers(f"augur:memoria:tier:{tier}"):
                mid = mid.decode() if isinstance(mid, bytes) else mid
                raw = self._r.get(f"augur:memoria:dsr:{mid}")
                if raw is not None:
                    out.append(json.loads(raw))
        return out

    def load_archived_memory(self, memory_id: str) -> dict | None:
        raw = self._r.get(f"augur:memoria:archive:{memory_id}")
        return None if raw is None else json.loads(raw)

    def is_session_processed(self, session_id: str) -> bool:
        return bool(self._r.sismember("augur:memoria:processed_sessions", session_id))

    def active_session_count(self) -> int:
        return int(self._r.scard("augur:memoria:processed_sessions"))

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
