"""Cross-domain correlator.

Subscribes to NATS 'augur.detection.anomaly', maintains a Redis-backed
sliding window + in-memory NetworkX DiGraph of correlations, and emits
enriched events on 'augur.correlation.detected' to the advisor.

Two LOW-severity anomalies from different domains within 30 seconds
escalate to MEDIUM via a Redis-stored matrix, producing reasoning that
neither signal would trigger individually.
"""

from __future__ import annotations

import asyncio  # noqa: F401 — used in run loop (Task 8)
import json  # noqa: F401 — used in window serialization (Task 5)
import logging
import sys
from datetime import datetime, timezone  # noqa: F401 — used in window helpers (Task 5)
from pathlib import Path

import nats  # noqa: F401 — used in NATS subscriber (Task 7)
import networkx as nx  # noqa: F401 — used in correlation graph (Task 6)
import redis  # noqa: F401 — used in Redis window store (Task 5)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from blackboard.config import AugurConfig  # noqa: F401 — used in run loop (Task 8)
from blackboard.persistence import PersistenceManager  # noqa: F401 — used in matrix load (Task 5)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("correlator")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SUBSCRIBE_ANOMALY = "augur.detection.anomaly"
SUBSCRIBE_DEBUG_DUMP = "augur.debug.graph_dump"
PUBLISH_CORRELATION = "augur.correlation.detected"

REDIS_KEY_WINDOW = "augur:correlation:window"
REDIS_KEY_MATRIX = "augur:config:escalation_matrix"

CORRELATION_WINDOW_S = 30  # query window (seconds back from primary)
PRUNE_WINDOW_S = 2 * CORRELATION_WINDOW_S  # derived: 60s buffer for clock skew

SEVERITY_ORDER: dict[str, int] = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}

DEFAULT_ESCALATION_MATRIX: dict = {
    "version": "1.0",
    "rules": {
        "LOW+LOW": "MEDIUM",
        "LOW+MEDIUM": "MEDIUM",
        "LOW+HIGH": "HIGH",
        "MEDIUM+MEDIUM": "HIGH",
        "MEDIUM+HIGH": "HIGH",
        "HIGH+HIGH": "HIGH",
    },
}

SEVERITY_GATE_PASSTHROUGH = {"MEDIUM", "HIGH"}  # forwarded even with no correlation


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def normalize_rule_key(sev1: str, sev2: str) -> str | None:
    """Sort a severity pair by rank (LOW<MEDIUM<HIGH) and return 'A+B'.

    Uppercases inputs first because the detector emits lowercase severity.
    Returns None for unknown severity levels — caller must fall back to
    the higher of the two original severities.
    """
    s1, s2 = sev1.upper(), sev2.upper()
    if s1 not in SEVERITY_ORDER or s2 not in SEVERITY_ORDER:
        return None
    pair = sorted([s1, s2], key=lambda s: SEVERITY_ORDER[s])
    return f"{pair[0]}+{pair[1]}"


def lookup_escalation(
    sev1: str,
    sev2: str,
    matrix: dict,
) -> tuple[str, str | None]:
    """Look up a severity pair in the escalation matrix.

    Returns ``(combined_severity, rule_label)`` where:
    - ``combined_severity`` is the escalated severity (uppercase).
    - ``rule_label`` is ``"LOW+LOW→MEDIUM"`` on matrix hit, ``None`` on miss.

    On matrix miss, falls back to ``max(sev1, sev2)`` by rank order. If
    neither severity is recognized at all, returns the first input
    uppercased with ``rule_label=None`` — the caller should have dropped
    this event before reaching lookup, so this path is a safety net only.
    """
    key = normalize_rule_key(sev1, sev2)
    rules = matrix.get("rules", {})

    if key is not None and key in rules:
        combined = rules[key]
        return combined, f"{key}→{combined}"

    # Fallback: take the higher of the two by rank
    s1, s2 = sev1.upper(), sev2.upper()
    r1 = SEVERITY_ORDER.get(s1, -1)
    r2 = SEVERITY_ORDER.get(s2, -1)
    if r1 < 0 and r2 < 0:
        return s1, None
    combined = s1 if r1 >= r2 else s2
    return combined, None
