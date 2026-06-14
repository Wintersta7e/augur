"""Shared, validated escalation-matrix updates (extracted from the MCP tool).

Single sanctioned write path for augur:nexus:matrix: MCP set_escalation_matrix
(mode='replace') and Imperator II (mode='patch'). Validates the FINAL MERGED matrix
under a WATCH/retry CAS before SET. rule_windows: None=preserve; {} or dict=set/merge.
"""

from __future__ import annotations

import json

import redis

from tabula.config import AugurConfig
from tabula.persistence import PersistenceManager

# ---- MOVED VERBATIM from augur_mcp/augur_server.py ----
_VALID_SEVERITIES = {"LOW", "MEDIUM", "HIGH"}

# SEC-04: caps on matrix size. Without these, a caller could pass
# thousands of rules or very long keys, amplifying memory and Redis-read
# latency (the correlator re-reads the matrix on every anomaly event).
# Bumped from 20 → 40 to accommodate 6 pairwise + 10 3-way defaults
# plus room for future expansion.
MAX_ESCALATION_RULES = 40
MAX_ESCALATION_RULE_KEY_LEN = 32
MAX_ESCALATION_VERSION_LEN = 32
MAX_RULE_WINDOWS = 40

_MATRIX_KEY = "augur:nexus:matrix"


def _validate_escalation_rules(rules: dict[str, str]) -> str | None:
    """Return an error message if rules fail shape validation, else None."""
    if not isinstance(rules, dict):
        return "rules must be a dict"
    if len(rules) > MAX_ESCALATION_RULES:
        return f"too many rules: {len(rules)} (max {MAX_ESCALATION_RULES})"
    for key, value in rules.items():
        if not isinstance(key, str) or "+" not in key:
            return f"invalid rule key (expected 'A+B'): {key!r}"
        if len(key) > MAX_ESCALATION_RULE_KEY_LEN:
            return (
                f"rule key too long: {len(key)} chars "
                f"(max {MAX_ESCALATION_RULE_KEY_LEN})"
            )
        parts = key.split("+")
        if not all(p in _VALID_SEVERITIES for p in parts):
            return (
                f"invalid severity in rule key {key!r}: "
                f"each part must be one of {sorted(_VALID_SEVERITIES)}"
            )
        if not isinstance(value, str) or value not in _VALID_SEVERITIES:
            return (
                f"invalid rule value for {key!r}: "
                f"must be one of {sorted(_VALID_SEVERITIES)}, got {value!r}"
            )
    return None


def _validate_escalation_matrix_rule_windows(
    rule_windows: dict | None,
    config: AugurConfig,
) -> str | None:
    """Validate optional rule_windows dict in the matrix.

    Pairwise-only this phase (one '+'). Values must be numeric within
    [correlation_window_min_s, correlation_window_max_s]. Returns None
    if valid; an error string otherwise.
    """
    if rule_windows is None:
        return None
    if not isinstance(rule_windows, dict):
        return "rule_windows must be a dict"
    if len(rule_windows) > MAX_RULE_WINDOWS:
        return f"too many rule_windows: {len(rule_windows)} (max {MAX_RULE_WINDOWS})"
    for key, value in rule_windows.items():
        if not isinstance(key, str):
            return f"invalid rule_windows key type: {type(key).__name__}"
        if len(key) > MAX_ESCALATION_RULE_KEY_LEN:
            return (
                f"rule_windows key '{key}' exceeds {MAX_ESCALATION_RULE_KEY_LEN} chars"
            )
        if key.count("+") != 1:
            return (
                f"rule_windows key '{key}' must be pairwise (one '+'); "
                f"N-way windows are not yet supported."
            )
        if not isinstance(value, (int, float)):
            return f"rule_windows value for '{key}' must be numeric"
        if not (
            config.correlation_window_min_s
            <= float(value)
            <= config.correlation_window_max_s
        ):
            return (
                f"rule_windows[{key}]={value} outside "
                f"[{config.correlation_window_min_s}, {config.correlation_window_max_s}]"
            )
    return None


def apply_matrix_update(
    pm: PersistenceManager,
    *,
    rules: dict | None = None,
    rule_windows: dict | None = None,
    version: str | None = None,
    mode: str = "replace",
    _retries: int = 5,
) -> dict:
    """Apply a validated patch or replace to augur:nexus:matrix under WATCH/CAS.

    Args:
        pm: PersistenceManager wrapping the Redis client.
        rules: Rules to write. In patch mode, merged with existing; in replace,
            overwrites entirely.
        rule_windows: Windows to write. None=preserve existing; {} or dict=set/merge.
        version: Version string. If None in patch mode, preserves the existing
            version. If None in replace mode, defaults to "1.0".
        mode: "replace" (full overwrite) or "patch" (merge into current).
        _retries: Number of WATCH/CAS retry attempts on contention.

    Returns:
        On success: {"status": "saved", "matrix": {...}, "prior_rules": {key: old|None},
        "prior_rule_windows": {key: old|None}} — prior_* hold the pre-write values, read
        from the SAME committed WATCH snapshot, for exactly the keys in rules/rule_windows
        (so a reversible caller anchors rollback to the real prior, not a separate read).
        On failure: {"error": "..."}.
    """
    if mode not in ("replace", "patch"):
        return {"error": f"invalid mode: {mode}"}
    if version is not None and (
        not isinstance(version, str) or len(version) > MAX_ESCALATION_VERSION_LEN
    ):
        return {"error": "version invalid or too long"}
    config = AugurConfig.from_env()
    client = pm._r
    for _ in range(_retries):
        try:
            with client.pipeline() as pipe:
                pipe.watch(_MATRIX_KEY)
                raw = pipe.get(_MATRIX_KEY)
                current = {} if raw is None else json.loads(raw)
                cur_rules = dict(current.get("rules", {}))
                cur_windows = dict(current.get("rule_windows", {}))
                merged_rules = (
                    (rules or {})
                    if mode == "replace"
                    else {**cur_rules, **(rules or {})}
                )
                if rule_windows is None:
                    merged_windows = current.get("rule_windows")
                elif mode == "patch":
                    merged_windows = {**cur_windows, **rule_windows}
                else:
                    merged_windows = rule_windows
                eff_version = (
                    version
                    if version is not None
                    else (current.get("version", "1.0") if mode == "patch" else "1.0")
                )
                # Validate the EFFECTIVE version too: a preserved (patch-mode) version
                # read from an already-corrupt matrix must not be silently re-committed.
                ver_err = (
                    None
                    if isinstance(eff_version, str)
                    and len(eff_version) <= MAX_ESCALATION_VERSION_LEN
                    else "version invalid or too long"
                )
                err = (
                    _validate_escalation_rules(merged_rules)
                    or _validate_escalation_matrix_rule_windows(
                        merged_windows or None, config
                    )
                    or ver_err
                )
                if err is not None:
                    pipe.unwatch()
                    return {"error": err}
                matrix: dict = {"version": eff_version, "rules": merged_rules}
                if merged_windows is not None:
                    matrix["rule_windows"] = merged_windows
                pipe.multi()
                pipe.set(_MATRIX_KEY, json.dumps(matrix))
                pipe.execute()
                return {
                    "status": "saved",
                    "matrix": matrix,
                    "prior_rules": {k: cur_rules.get(k) for k in (rules or {})},
                    "prior_rule_windows": {
                        k: cur_windows.get(k) for k in (rule_windows or {})
                    },
                }
        except redis.WatchError:
            continue
        except Exception as exc:
            return {"error": str(exc)}
    return {"error": "matrix update failed after retries (contention)"}
