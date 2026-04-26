"""Persistence layer — Redis-backed storage for baselines, history, feedback,
prompt versioning, and threshold configuration."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

import redis

from blackboard.contracts import PerceptionEvent

log = logging.getLogger("persistence")

HISTORY_MAX = 1000
PROMPT_HISTORY_MAX = 100

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
        key = f"augur:profile:{domain}:{entity}"
        self._r.set(key, json.dumps(state_dict))
        log.debug("Saved baseline %s", key)

    def load_baseline(self, domain: str, entity: str) -> dict | None:
        key = f"augur:profile:{domain}:{entity}"
        raw = self._r.get(key)
        if raw is None:
            return None
        return json.loads(raw)

    # -- Event history -------------------------------------------------------

    def append_event(self, event: PerceptionEvent) -> None:
        key = f"augur:history:{event.domain}"
        self._r.lpush(key, event.to_json())
        self._r.ltrim(key, 0, HISTORY_MAX - 1)

    def get_history(self, domain: str, limit: int = 100) -> list[dict]:
        key = f"augur:history:{domain}"
        raw_list = self._r.lrange(key, 0, limit - 1)
        return [json.loads(entry) for entry in raw_list]

    # -- Feedback storage ----------------------------------------------------

    def save_feedback(self, session_id: str, feedback_dict: dict) -> None:
        key = f"augur:feedback:{session_id}"
        feedback_dict.setdefault(
            "timestamp",
            datetime.now(timezone.utc).isoformat(),
        )
        self._r.set(key, json.dumps(feedback_dict), ex=SESSION_KEY_TTL_S)
        # Also maintain an ordered index of session IDs for get_all_feedback
        self._r.lpush("augur:feedback:_index", session_id)
        self._r.ltrim("augur:feedback:_index", 0, 999)
        log.debug("Saved feedback for session %s", session_id)

    def get_feedback(self, session_id: str) -> dict | None:
        key = f"augur:feedback:{session_id}"
        raw = self._r.get(key)
        if raw is None:
            return None
        return json.loads(raw)

    def get_all_feedback(self, limit: int = 50) -> list[dict]:
        session_ids = self._r.lrange("augur:feedback:_index", 0, limit - 1)
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
        current_key = f"augur:prompts:{domain}:current"
        history_key = f"augur:prompts:{domain}:history"

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
        current_key = f"augur:prompts:{domain}:current"
        raw = self._r.get(current_key)
        if raw is None:
            return None
        return json.loads(raw).get("prompt")

    def get_prompt_history(self, domain: str, limit: int = 10) -> list[dict]:
        history_key = f"augur:prompts:{domain}:history"
        raw_list = self._r.lrange(history_key, 0, limit - 1)
        return [json.loads(entry) for entry in raw_list]

    def rollback_prompt(self, domain: str) -> bool:
        """Restore previous prompt version. Returns True if rollback succeeded."""
        current_key = f"augur:prompts:{domain}:current"
        history_key = f"augur:prompts:{domain}:history"

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

    # -- Threshold config ----------------------------------------------------

    def save_thresholds(self, domain: str, thresholds_dict: dict) -> None:
        key = f"augur:config:thresholds:{domain}"
        self._r.set(key, json.dumps(thresholds_dict))
        log.debug("Saved thresholds for domain %s", domain)

    def load_thresholds(self, domain: str) -> dict | None:
        key = f"augur:config:thresholds:{domain}"
        raw = self._r.get(key)
        if raw is None:
            return None
        return json.loads(raw)

    # -- Escalation matrix (cross-domain correlator) -------------------------

    def save_escalation_matrix(self, matrix: dict) -> None:
        """Store the full matrix dict (including 'version' and 'rules' keys) as-is."""
        key = "augur:config:escalation_matrix"
        self._r.set(key, json.dumps(matrix))
        log.debug("Saved escalation matrix (version=%s)", matrix.get("version"))

    def load_escalation_matrix(self) -> dict | None:
        """Return the full matrix dict or None if not set."""
        key = "augur:config:escalation_matrix"
        raw = self._r.get(key)
        if raw is None:
            return None
        return json.loads(raw)

    # -- Correlation graph (cross-domain correlator) -------------------------

    def save_correlation_graph(self, session_id: str, graph_data: dict) -> None:
        """Persist a session's correlation DiGraph as node_link_data JSON.

        Also maintains an ordered index list so list_correlation_graphs
        can return session ids without scanning keyspace. Uses a 30-day TTL
        to prevent unbounded Redis growth past the index-trim boundary.
        """
        key = f"augur:correlation:graph:{session_id}"
        self._r.set(key, json.dumps(graph_data), ex=SESSION_KEY_TTL_S)
        self._r.lpush("augur:correlation:graph:_index", session_id)
        self._r.ltrim("augur:correlation:graph:_index", 0, 999)
        log.debug(
            "Saved correlation graph for session %s (%d nodes, %d links)",
            session_id,
            len(graph_data.get("nodes", [])),
            len(graph_data.get("links", [])),
        )

    def load_correlation_graph(self, session_id: str) -> dict | None:
        """Load a persisted correlation graph or return None if absent."""
        key = f"augur:correlation:graph:{session_id}"
        raw = self._r.get(key)
        if raw is None:
            return None
        return json.loads(raw)

    def list_correlation_graphs(self, limit: int = 50) -> list[str]:
        """Return recent session ids that have persisted correlation graphs.

        Ordered newest-first (lpush index order).
        """
        raw_ids = self._r.lrange("augur:correlation:graph:_index", 0, limit - 1)
        return [sid.decode() if isinstance(sid, bytes) else sid for sid in raw_ids]

    # -- Rule confidence state (reflection matrix tuning) --------------------

    def save_rule_confidence(self, confidence_state: dict) -> None:
        """Store per-rule EWMA confidence + restore_target snapshots.

        Schema: {rule_key: {"confidence": float, "restore_target": str | None}}
        """
        key = "augur:config:escalation_confidence"
        self._r.set(key, json.dumps(confidence_state))
        log.debug(
            "Saved rule confidence state: %d rules",
            len(confidence_state),
        )

    def load_rule_confidence(self) -> dict | None:
        """Return the per-rule confidence state dict or None if not set."""
        key = "augur:config:escalation_confidence"
        raw = self._r.get(key)
        if raw is None:
            return None
        return json.loads(raw)

    def save_rule_window_state(self, state: dict) -> None:
        """Persist per-rule observed-lag EWMA state.

        Schema: {rule_key: {"ewma_lag": float}}.
        Mirrors save_rule_confidence; written to a separate Redis key.
        """
        self._r.set("augur:config:rule_window_state", json.dumps(state))

    def load_rule_window_state(self) -> dict:
        """Load per-rule observed-lag EWMA state. Returns {} if absent or corrupt."""
        raw = self._r.get("augur:config:rule_window_state")
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
            pipe.set("augur:config:escalation_confidence", json.dumps(confidence))
        if window_state is not None:
            pipe.set("augur:config:rule_window_state", json.dumps(window_state))
        pipe.execute()

    # -- Reflection reports --------------------------------------------------

    def save_reflection(self, session_id: str, report_dict: dict) -> None:
        """Persist a per-session reflection report with a 30-day TTL.

        Previously written directly via redis_client.set(...) from
        reflection_engine.py, bypassing this abstraction. Now routed here
        so the key namespace is discoverable alongside other persistence
        concerns and TTL is applied consistently.
        """
        key = f"augur:reflect:{session_id}"
        self._r.set(key, json.dumps(report_dict), ex=SESSION_KEY_TTL_S)
        log.debug("Saved reflection report for session %s", session_id)

    def load_reflection(self, session_id: str) -> dict | None:
        """Return a session's reflection report or None if not set."""
        key = f"augur:reflect:{session_id}"
        raw = self._r.get(key)
        if raw is None:
            return None
        return json.loads(raw)

    # -- Last anomaly / last advice (live state, not per-session) -----------

    def save_last_anomaly(self, anomaly_dict: dict) -> None:
        """Persist the most recent anomaly event (no TTL — live state)."""
        self._r.set("augur:detection:last_anomaly", json.dumps(anomaly_dict))

    def load_last_anomaly(self) -> dict | None:
        """Return the most recent anomaly event or None if not set."""
        raw = self._r.get("augur:detection:last_anomaly")
        if raw is None:
            return None
        return json.loads(raw)

    def save_last_advice(self, advice_dict: dict) -> None:
        """Persist the most recent LLM advice payload (no TTL — live state)."""
        self._r.set("augur:reasoning:last_advice", json.dumps(advice_dict))

    def load_last_advice(self) -> dict | None:
        """Return the most recent LLM advice payload or None if not set."""
        raw = self._r.get("augur:reasoning:last_advice")
        if raw is None:
            return None
        return json.loads(raw)

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

    def mark_tuning_applied(self, session_id: str) -> None:
        """Mark a session as having had its correlation tuning pass applied.

        Sets a short-lived (7-day) marker so manual reflect-trigger replays
        of a recent session do not double-apply the EWMA update.
        """
        key = f"augur:correlation:tuning_applied:{session_id}"
        self._r.set(key, "1", ex=TUNING_APPLIED_TTL_S)

    def is_tuning_applied(self, session_id: str) -> bool:
        """Return True if mark_tuning_applied was called for this session."""
        key = f"augur:correlation:tuning_applied:{session_id}"
        return bool(self._r.exists(key))
