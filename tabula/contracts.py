"""Standard perception envelope used across all Augur domains."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from typing import Any, Literal

# Known perception domains. New domains MUST be added here.
# Treat unknown domain values as a programming error caught at the
# ingestion boundary (PerceptionEvent.__post_init__) — kept loose
# for backward compatibility; tighten when all callers updated.
Domain = Literal["chess", "typing", "activity_focus", "activity_intensity"]

# Placeholder entities the activity Sensus emits when it cannot name a real
# app: <no_foreground>, <unknown>, <denied>, <gone>. They are daemon
# bookkeeping, not behaviour — `<no_foreground>` measures the sub-poll-interval
# residue of `total_dwell - idle_dwell`, i.e. float noise around 70ms — so
# nothing should baseline them, gate on them or mine patterns from them.
# Lives here rather than in a faculty because Vigil, Praesagium and Consilium
# all need the same answer.
_SENTINEL_ENTITY_RE = re.compile(r"^<[^>]+>$")


def is_sentinel_entity(entity: str | None) -> bool:
    """True for a daemon placeholder entity like ``<no_foreground>``."""
    return bool(entity) and bool(_SENTINEL_ENTITY_RE.match(entity))


# Fields required to construct a valid PerceptionEvent. Used by
# from_json() to surface schema-level problems at the ingestion boundary
# rather than deep in a consumer.
_REQUIRED_FIELDS: frozenset[str] = frozenset(
    {
        "domain",
        "stream_id",
        "entity",
        "event_type",
        "value",
        "unit",
        "context",
        "timestamp",
        "session_id",
    }
)


@dataclass
class PerceptionEvent:
    """Domain-agnostic envelope for perception data published via NATS."""

    domain: str  # e.g. "chess", "typing"
    stream_id: str  # e.g. "chess_timing"
    entity: str  # e.g. "white", "black", "user"
    event_type: str  # e.g. "move", "keypress"
    value: float  # the primary numeric signal
    unit: str  # e.g. "seconds", "wpm"
    context: dict[str, Any]  # domain-specific extras
    timestamp: str  # ISO format
    session_id: str  # unique per session, same for all events in one run

    def __post_init__(self) -> None:
        """Validate the envelope's shape at construction time.

        ARCH-08: catches a corrupted ``context`` (e.g., null from JSON,
        list, or non-dict) at the ingestion boundary rather than deep in
        a consumer's ``.get()`` call. Also coerces numeric strings like
        ``"12.5"`` into floats for ``value`` because some perception
        sources serialize numbers as strings.
        """
        if not isinstance(self.context, dict):
            raise TypeError(
                f"PerceptionEvent.context must be a dict, "
                f"got {type(self.context).__name__}: {self.context!r}"
            )
        if not isinstance(self.value, (int, float)):
            # Tolerate int inputs; coerce numeric strings for robustness.
            try:
                object.__setattr__(self, "value", float(self.value))
            except (TypeError, ValueError) as exc:
                raise TypeError(
                    f"PerceptionEvent.value must be numeric, "
                    f"got {type(self.value).__name__}: {self.value!r}"
                ) from exc

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    def to_bytes(self) -> bytes:
        return self.to_json().encode()

    @classmethod
    def from_json(cls, raw: str | bytes) -> PerceptionEvent:
        """Parse a PerceptionEvent from a NATS message payload.

        Raises ``ValueError`` on any schema problem: invalid JSON, non-object
        top level, missing required fields, or unexpected fields. This makes
        corrupted/spoofed messages surface at the ingestion boundary with a
        clear diagnostic rather than dropping silently or crashing a
        downstream consumer (SEC-03).
        """
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"PerceptionEvent.from_json: invalid JSON ({exc})"
            ) from exc

        if not isinstance(data, dict):
            raise ValueError(
                f"PerceptionEvent.from_json: expected a JSON object, "
                f"got {type(data).__name__}"
            )

        missing = _REQUIRED_FIELDS - data.keys()
        if missing:
            raise ValueError(
                f"PerceptionEvent.from_json: missing required fields: {sorted(missing)}"
            )

        extra = data.keys() - _REQUIRED_FIELDS
        if extra:
            raise ValueError(
                f"PerceptionEvent.from_json: unexpected fields: {sorted(extra)}"
            )

        return cls(**data)
