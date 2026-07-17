#!/usr/bin/env python3
"""Reset contaminated learned/tuned state to a clean slate.

Synthetic test/script data and thresholds tuned under a poisoned lens have
trained the live system. This tool classifies every ``augur:*`` key and, on
``--confirm``, removes the contaminated ones so the faculties revert to their
config defaults on the next observation. It never touches real perception
baselines or user-taught semantic memories.

Safety:
  * ``--dry-run`` is the DEFAULT — it prints the full plan and changes nothing.
  * ``--confirm`` is required to actually delete.
  * Refuses to run against a Redis database other than db 0 unless ``--force``
    (a reset is a live-cell operation; a test cell should never be reset here).
  * Unrecognized keys are KEPT and flagged, never silently deleted.

Usage:
  .venv/bin/python scripts/reset_learned_state.py                 # dry-run
  .venv/bin/python scripts/reset_learned_state.py --confirm       # execute
  AUGUR_REDIS_URL=redis://127.0.0.1:6379 .venv/bin/python scripts/reset_learned_state.py --confirm
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import redis as redis_lib  # noqa: E402

from tabula.config import AugurConfig  # noqa: E402

# Entities/domains that only ever appear in synthetic injections.
_SYNTHETIC_ENTITY_SUFFIX = "_a2b60936"
_SYNTHETIC_DOMAINS = ("chess",)

# Keys that are harmless to keep: a monotonic counter and a live read-model
# that regenerates on its own heartbeat.
_KEEP_ALWAYS = frozenset({"augur:session:count", "augur:praefectus:health"})

# Contaminated learned-policy / session-scoped / transient prefixes. Deleting
# any of these reverts the reading faculty to its config default (verified: no
# faculty errors on an absent key — each treats it as "no prior state").
_DELETE_PREFIXES: tuple[tuple[str, str], ...] = (
    (
        "augur:vigil:thresholds:",
        "tuned detector threshold -> reverts to config default",
    ),
    ("augur:vigil:history:", "per-stream anomaly history -> starts empty"),
    ("augur:vigil:last_anomaly", "transient last-anomaly marker"),
    ("augur:consilium:", "advice/app-descriptor state -> regenerates"),
    ("augur:nexus:", "escalation matrix / window state -> reverts to config default"),
    ("augur:limen:", "gate adaptive state -> neutral prior (fails open)"),
    ("augur:imperator:", "self-improvement read-model -> regenerates on tick"),
    ("augur:disciplina:", "per-session reflection report"),
    ("augur:responsum:", "per-session feedback record"),
    ("augur:tuning_applied:", "per-session idempotency marker"),
    ("augur:praesagium:", "anticipation patterns/episodes/predictions"),
    ("augur:memoria:tier:", "memory tier index -> rebuilds from surviving memories"),
    (
        "augur:memoria:processed_sessions",
        "processed-session set -> references deleted sessions",
    ),
    ("augur:session:current", "active-session pointer (session is being ended)"),
)


@dataclass(frozen=True)
class Decision:
    key: str
    action: str  # "keep" | "delete"
    reason: str


def _is_synthetic_vigil_entity(key: str) -> bool:
    """True for a vigil profile key whose entity/domain is synthetic-only."""
    parts = key.split(":")
    domain = parts[3] if len(parts) > 3 else ""
    entity = parts[-1]
    return entity.endswith(_SYNTHETIC_ENTITY_SUFFIX) or domain in _SYNTHETIC_DOMAINS


def classify_key(key: str, get_json: Callable[[str], object]) -> Decision:
    """Classify one ``augur:*`` key. Pure; ``get_json`` reads the few keys
    whose *content* decides (memoria). Unknown keys are KEPT and flagged so a
    key class this tool does not recognize is never silently destroyed.
    """
    # Real perception baselines are threshold-independent — keep them; only the
    # synthetic entities go.
    if key.startswith("augur:vigil:profile:"):
        if _is_synthetic_vigil_entity(key):
            return Decision(key, "delete", "synthetic baseline entity")
        return Decision(key, "keep", "real perception baseline (threshold-independent)")

    # Memoria: NEVER delete a user-taught (semantic) memory. Only reflection-
    # derived (episodic) memories from contaminated sessions go.
    if key.startswith("augur:memoria:dsr:"):
        rec = get_json(key)
        kind = rec.get("memory_kind") if isinstance(rec, dict) else None
        if kind == "semantic":
            return Decision(key, "keep", "user-taught semantic memory (PROTECTED)")
        return Decision(key, "delete", f"reflection-derived memory (kind={kind})")

    if key in _KEEP_ALWAYS:
        return Decision(key, "keep", "runtime bookkeeping (monotonic / regenerates)")

    for prefix, reason in _DELETE_PREFIXES:
        if key.startswith(prefix):
            return Decision(key, "delete", reason)

    return Decision(key, "keep", "UNRECOGNIZED — kept for manual review")


def plan_reset(r: redis_lib.Redis) -> list[Decision]:
    """Classify every ``augur:*`` key. Read-only."""

    def get_json(k: str) -> object:
        raw = r.get(k)
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return None

    keys = sorted(str(k) for k in r.scan_iter(match="augur:*", count=500))
    return [classify_key(k, get_json) for k in keys]


def _print_plan(decisions: list[Decision]) -> tuple[int, int, int]:
    deletes = [d for d in decisions if d.action == "delete"]
    keeps = [d for d in decisions if d.action == "keep"]
    unrecognized = [d for d in keeps if d.reason.startswith("UNRECOGNIZED")]

    print(f"\n{'=' * 70}\nRESET PLAN — {len(decisions)} keys\n{'=' * 70}")
    print(f"\nDELETE ({len(deletes)}):")
    for d in deletes:
        print(f"  - {d.key}\n      {d.reason}")
    print(f"\nKEEP ({len(keeps)}):")
    for d in keeps:
        flag = "  <-- REVIEW" if d.reason.startswith("UNRECOGNIZED") else ""
        print(f"  + {d.key}  ({d.reason}){flag}")
    if unrecognized:
        print(
            f"\n!! {len(unrecognized)} UNRECOGNIZED key(s) kept — review before trusting this reset."
        )
    return len(deletes), len(keeps), len(unrecognized)


def main() -> int:
    ap = argparse.ArgumentParser(description="Reset contaminated Augur learned state.")
    ap.add_argument(
        "--confirm", action="store_true", help="actually delete (default is dry-run)"
    )
    ap.add_argument(
        "--force", action="store_true", help="allow running against a non-db-0 database"
    )
    args = ap.parse_args()

    config = AugurConfig.from_env()
    if config.redis_db != 0 and not args.force:
        sys.stderr.write(
            f"refusing to reset db {config.redis_db} (not the live cell). "
            "A reset is a live-cell operation; use --force only if you mean it.\n"
        )
        return 2

    r = redis_lib.Redis.from_url(config.redis_url, decode_responses=True)
    decisions = plan_reset(r)
    n_del, n_keep, n_unk = _print_plan(decisions)

    if not args.confirm:
        print(f"\nDRY-RUN — nothing deleted. {n_del} would be deleted, {n_keep} kept.")
        print("Re-run with --confirm to execute.")
        return 0

    if n_unk:
        sys.stderr.write(
            f"\nrefusing to --confirm with {n_unk} unrecognized key(s) present — "
            "classify them first.\n"
        )
        return 3

    to_delete = [d.key for d in decisions if d.action == "delete"]
    if to_delete:
        r.delete(*to_delete)
    print(
        f"\nDONE — deleted {len(to_delete)} keys, kept {n_keep}. "
        "Faculties revert to config defaults on next observation."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
