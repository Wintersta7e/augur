"""Centralized configuration for Augur.

All connection strings, tunables, and behavioral constants live here.
Use AugurConfig.from_env() to allow AUGUR_* env-var overrides at deploy time.
"""

from __future__ import annotations

import dataclasses
import logging
import os
from typing import Any, Callable
from urllib.parse import urlparse

log = logging.getLogger("augur.config")

# Maps field name -> type coercion callable for from_env().
_TYPE_COERCIONS: dict[str, Callable[[str], Any]] = {}


def _coerce_bool(raw: str) -> bool:
    """Parse an env var string as a bool.

    Truthy: ``true``, ``1``, ``yes``, ``on``, ``y`` (case-insensitive,
    whitespace-stripped). Everything else — including the common footgun
    of ``"false"`` — evaluates to False. This replaces the default
    ``bool(raw)`` coercion which treats any non-empty string as True.
    """
    return raw.strip().lower() in {"true", "1", "yes", "on", "y"}


@dataclasses.dataclass(frozen=True)
class AugurConfig:
    """Immutable configuration snapshot for the Augur system."""

    # ── Transport ──────────────────────────────────────────────────────────
    nats_url: str = "nats://localhost:4222"
    redis_url: str = "redis://localhost:6379"
    nats_connect_timeout: int = 5
    redis_connect_timeout: int = 5

    # ── Ollama ─────────────────────────────────────────────────────────────
    ollama_url: str = "http://host.docker.internal:11434"
    ollama_timeout: int = 120
    ollama_model: str = "qwen2.5:32b"

    # ── Detection ──────────────────────────────────────────────────────────
    default_sigma_threshold: float = 2.0
    ewma_alpha: float = 0.3
    hst_n_trees: int = 10
    hst_height: int = 8
    hst_window_size: int = 50
    min_observations: int = 3
    hst_threshold: float = 0.7
    severity_medium_sigma: float = 2.5
    severity_high_sigma: float = 4.0

    # ── Advisor ────────────────────────────────────────────────────────────
    advisor_lock_timeout: int = 180

    # ── Feedback ───────────────────────────────────────────────────────────
    feedback_explicit_timeout: int = 10
    feedback_behavioral_window: int = 30

    # ── Persistence ────────────────────────────────────────────────────────
    history_max_events: int = 1000

    # ── Reflection / self-improvement ─────────────────────────────────────
    sigma_adjust_step: float = 0.1
    sigma_min: float = 1.5
    sigma_max: float = 5.0
    utility_mutation_threshold: float = 0.4

    # ── Reflection: cross-domain matrix tuning ─────────────────────────────
    correlation_tuning_enabled: bool = True
    correlation_tuning_alpha: float = 0.2
    correlation_tuning_enable_threshold: float = 0.6
    correlation_tuning_disable_threshold: float = 0.3

    # ── Constructors ───────────────────────────────────────────────────────

    @classmethod
    def from_env(cls) -> AugurConfig:
        """Build config from defaults, overridden by AUGUR_* environment variables.

        If an env var's value cannot be coerced to the field's type (e.g.,
        ``AUGUR_OLLAMA_TIMEOUT=xyz``), the failure is logged as a warning
        and the field keeps its default. This prevents a misconfigured env
        var from crash-looping every component at startup.
        """
        defaults = dataclasses.asdict(cls())
        overrides: dict[str, Any] = {}
        for field in dataclasses.fields(cls):
            env_key = f"AUGUR_{field.name.upper()}"
            raw = os.environ.get(env_key)
            if raw is None:
                continue
            coerce = _TYPE_COERCIONS.get(field.name, str)
            try:
                overrides[field.name] = coerce(raw)
            except (ValueError, TypeError) as exc:
                log.warning(
                    "%s=%r: coercion via %s failed (%s); using default %r",
                    env_key,
                    raw,
                    getattr(coerce, "__name__", type(coerce).__name__),
                    exc,
                    defaults[field.name],
                )
        return cls(**{**defaults, **overrides})

    # ── Convenience ────────────────────────────────────────────────────────

    def as_dict(self) -> dict[str, Any]:
        """Return all fields as a plain dict."""
        return dataclasses.asdict(self)

    @property
    def redis_host(self) -> str:
        """Hostname extracted from redis_url."""
        return urlparse(self.redis_url).hostname or "localhost"

    @property
    def redis_port(self) -> int:
        """Port extracted from redis_url, defaulting to 6379."""
        return urlparse(self.redis_url).port or 6379


# Build the type-coercion map now that the class exists.
#
# Under ``from __future__ import annotations`` (PEP 563), field.type is a
# *string*, so isinstance(field.type, type) is always False. Use the concrete
# default value's type as the source of truth instead. bool fields get the
# explicit string parser because bool("false") == True (Python treats any
# non-empty string as truthy).
_TYPE_COERCIONS = {}
for _field in dataclasses.fields(AugurConfig):
    _field_type = type(_field.default)
    if _field_type is bool:
        _TYPE_COERCIONS[_field.name] = _coerce_bool
    else:
        _TYPE_COERCIONS[_field.name] = _field_type
