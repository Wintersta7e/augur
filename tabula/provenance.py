"""Provenance primitives — carried through the learning paths, gated at the writes.

`LearnContext` is an event's provenance resolved ONCE per event at ingestion
(`PersistenceManager.resolve_learn_context`) and threaded through every learned
write, never re-derived at each call site. A bare ``learnable: bool`` would
discard *which* session and *why*, so a write could neither log nor assert its
own provenance and every site would re-derive it (or forget to). Async paths
(e.g. the app-descriptor classifier lane) capture it at enqueue time — a plain
bool resolved at write time gets that wrong.

`learned_write` / `non_learning_write` are the write-boundary decorators. They
are **report-only groundwork**: applied to no production writer yet, and the
global mode defaults to OFF (a pure passthrough). REPORT logs what a
non-learnable session *would* withhold without changing anything (to measure the
blast radius against real traffic); ENFORCE actually withholds. The flip to
ENFORCE happens only once every Redis-mutating write is marked (CL10) — never
before.

Spec: ``docs/superpowers/specs/2026-07-17-cells-and-session-provenance-design.md``
§4.3c (LearnContext) and §7 CL10 (the markers).
"""

from __future__ import annotations

import functools
import logging
import os
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any

log = logging.getLogger("provenance")


@dataclass(frozen=True)
class LearnContext:
    """An event's provenance: which session, whether it may train, and why."""

    session_id: str | None
    """The EVENT's session — never "whatever session is current"."""

    learnable: bool
    """Resolved once at ingestion. Fail-closed default is ``False``."""

    origin: str
    """``"real"`` | ``"synthetic"`` | ``"unattributed"`` | ``"unknown"``."""

    @property
    def dry_run(self) -> bool:
        """True when a learned write must be withheld (observe/report, not persist)."""
        return not self.learnable

    @classmethod
    def unknown(cls, session_id: str | None = None) -> LearnContext:
        """The fail-closed default: unresolvable provenance ⇒ non-learnable."""
        return cls(session_id=session_id, learnable=False, origin="unknown")

    @classmethod
    def system(cls) -> LearnContext:
        """A system/bootstrap write that must always persist — seeding config
        defaults, a migration, an operator action. Not user-perception learning,
        but never withheld (it does not depend on any session)."""
        return cls(session_id=None, learnable=True, origin="system")


class ProvenanceMode(str, Enum):
    """How the write-boundary decorators behave. Watch-first default is OFF."""

    OFF = "off"
    """Pure passthrough — no logging, no withholding."""

    REPORT = "report"
    """Log what a non-learnable session would withhold; still write (measure only)."""

    ENFORCE = "enforce"
    """Withhold learned writes for non-learnable sessions."""


def _mode_from_env() -> ProvenanceMode:
    """Initial mode from ``AUGUR_PROVENANCE_MODE`` (off|report|enforce), default OFF.

    This is the single flip switch: a deployment sets the env var and every
    faculty process picks it up at import — no per-faculty wiring. An unknown
    value fails safe to OFF (never accidentally enforce or crash a faculty).
    """
    raw = os.environ.get("AUGUR_PROVENANCE_MODE", "off").strip().lower()
    try:
        return ProvenanceMode(raw)
    except ValueError:
        log.warning("invalid AUGUR_PROVENANCE_MODE=%r; defaulting to OFF", raw)
        return ProvenanceMode.OFF


_mode = _mode_from_env()


def get_provenance_mode() -> ProvenanceMode:
    return _mode


def set_provenance_mode(mode: ProvenanceMode) -> ProvenanceMode:
    """Set the global mode; return the previous one (for save/restore in tests)."""
    global _mode
    prev = _mode
    _mode = ProvenanceMode(mode)
    return prev


def _find_context(args: tuple, kwargs: dict) -> LearnContext | None:
    for a in args:
        if isinstance(a, LearnContext):
            return a
    for v in kwargs.values():
        if isinstance(v, LearnContext):
            return v
    return None


def learned_write(func: Callable) -> Callable:
    """Mark a Redis-mutating method as LEARNED and gate it on provenance.

    The wrapped callable takes a ``LearnContext`` (conventionally the keyword-only
    ``ctx``); the decorator finds it among the arguments. Runtime behaviour follows
    the global :class:`ProvenanceMode`:

    - **OFF** — pure passthrough; ``ctx`` is not even inspected. Storage tests and
      any not-yet-migrated caller run here unchanged.
    - **REPORT** — a non-learnable ``ctx`` logs what *would* be withheld, then the
      write proceeds anyway (blast-radius measurement); a *missing* ``ctx`` logs an
      un-migrated caller and still writes. No behaviour change.
    - **ENFORCE** — a missing ``ctx`` **raises** ``TypeError`` (provenance cannot be
      forgotten *where it is enforced* — a silent non-learnable drop of real
      learning is the one failure we refuse); a non-learnable ``ctx`` is withheld
      (``None`` returned); a learnable ``ctx`` writes.

    The optional-``ctx`` signature is a migration convenience: it lets the ~300
    existing storage tests keep running in OFF without edits, while ENFORCE still
    makes provenance mandatory in production. Sets ``__learned_write__`` so the CL10
    discovery pass can find the marker.
    """

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        if _mode is ProvenanceMode.OFF:
            return func(*args, **kwargs)
        ctx = _find_context(args, kwargs)
        if ctx is None:
            if _mode is ProvenanceMode.ENFORCE:
                raise TypeError(
                    f"{func.__qualname__} is a @learned_write and requires a "
                    "LearnContext under ENFORCE (provenance in hand)"
                )
            log.warning(
                "provenance report: %s called without a LearnContext "
                "(un-migrated caller)",
                func.__qualname__,
            )
            return func(*args, **kwargs)
        if ctx.dry_run:
            log.warning(
                "provenance %s: %s learned write %s for session=%s origin=%s",
                _mode.value,
                "withholding" if _mode is ProvenanceMode.ENFORCE else "would withhold",
                func.__qualname__,
                ctx.session_id,
                ctx.origin,
            )
            if _mode is ProvenanceMode.ENFORCE:
                return None
        return func(*args, **kwargs)

    wrapper.__learned_write__ = True  # type: ignore[attr-defined]
    return wrapper


def non_learning_write(reason: str) -> Callable[[Callable], Callable]:
    """Mark a Redis-mutating method as deliberately NON-learning (e.g. an audit log).

    Records intent and ``reason`` so the CL10 discovery pass can tell a considered
    non-learning write from an unguarded one. Pure passthrough at runtime.
    """

    def decorator(func: Callable) -> Callable:
        func.__non_learning_write__ = True  # type: ignore[attr-defined]
        func.__non_learning_reason__ = reason  # type: ignore[attr-defined]
        return func

    return decorator
