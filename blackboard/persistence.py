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
        self._r.set(key, json.dumps(feedback_dict))
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
