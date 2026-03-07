"""Standard perception envelope used across all Augur domains."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field


@dataclass
class PerceptionEvent:
    """Domain-agnostic envelope for perception data published via NATS."""

    domain: str          # e.g. "chess", "typing"
    stream_id: str       # e.g. "chess_timing"
    entity: str          # e.g. "white", "black", "user"
    event_type: str      # e.g. "move", "keypress"
    value: float         # the primary numeric signal
    unit: str            # e.g. "seconds", "wpm"
    context: dict        # domain-specific extras
    timestamp: str       # ISO format
    session_id: str      # unique per session, same for all events in one run

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    def to_bytes(self) -> bytes:
        return self.to_json().encode()

    @classmethod
    def from_json(cls, raw: str | bytes) -> PerceptionEvent:
        data = json.loads(raw)
        return cls(**data)
