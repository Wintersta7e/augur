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
    nats_url: str = "nats://127.0.0.1:4222"
    redis_url: str = "redis://127.0.0.1:6379"
    nats_connect_timeout: int = 5
    redis_connect_timeout: int = 5

    # ── Ollama ─────────────────────────────────────────────────────────────
    ollama_url: str = "http://host.docker.internal:11434"
    ollama_timeout: int = 120
    ollama_model: str = "qwen2.5:32b"
    ollama_classifier_model: str = "qwen2.5:1.5b"
    ollama_classifier_enabled: bool = True
    ollama_classifier_timeout: int = 15

    # ── Detection ──────────────────────────────────────────────────────────
    default_sigma_threshold: float = 2.0
    ewma_alpha: float = 0.3
    hst_n_trees: int = 10
    hst_height: int = 8
    hst_window_size: int = 50
    min_observations: int = 15
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

    # ── Correlation: adaptive window ───────────────────────────────────────
    correlation_window_s: float = 30.0
    correlation_window_min_s: float = 5.0
    correlation_window_max_s: float = 120.0
    correlation_window_tuning_alpha: float = 0.2
    correlation_window_lag_multiplier: float = 2.5
    correlation_window_tuning_hysteresis_pct: float = 0.20

    # ── Activity perception (Phase 1) ──────────────────────────────────────
    activity_sampling_s: float = 10.0
    activity_intensity_min_events: int = 1
    activity_intensity_min_window_s: float = 2.0
    activity_title_allowlist: str = ""  # comma-separated; consumers split at use
    activity_source_id: str = "windows-host"

    # ── Session validity ───────────────────────────────────────────────────
    session_max_age_h: float = 12.0

    def __post_init__(self) -> None:
        """Validate bounds on tuning fields. Raises ValueError on out-of-range.

        `frozen=True` only blocks attribute reassignment after __init__; calling
        methods (including __post_init__) is fine.
        """
        if not (5.0 <= self.correlation_window_s <= 120.0):
            raise ValueError(
                f"correlation_window_s={self.correlation_window_s} outside [5, 120]"
            )
        if not (0.0 < self.correlation_window_min_s <= self.correlation_window_s):
            raise ValueError(
                f"correlation_window_min_s={self.correlation_window_min_s} must be in "
                f"(0, correlation_window_s={self.correlation_window_s}]"
            )
        if not (self.correlation_window_s <= self.correlation_window_max_s <= 600.0):
            raise ValueError(
                f"correlation_window_max_s={self.correlation_window_max_s} outside "
                f"[{self.correlation_window_s}, 600]"
            )
        if not (1.0 <= self.correlation_window_lag_multiplier <= 5.0):
            raise ValueError(
                f"correlation_window_lag_multiplier={self.correlation_window_lag_multiplier} outside [1.0, 5.0]"
            )
        if not (0.0 <= self.correlation_window_tuning_hysteresis_pct <= 1.0):
            raise ValueError(
                f"correlation_window_tuning_hysteresis_pct={self.correlation_window_tuning_hysteresis_pct} outside [0.0, 1.0]"
            )
        if not (0.0 <= self.correlation_window_tuning_alpha <= 1.0):
            raise ValueError(
                f"correlation_window_tuning_alpha={self.correlation_window_tuning_alpha} outside [0.0, 1.0]"
            )
        if not (1.0 <= self.activity_sampling_s <= 60.0):
            raise ValueError(
                f"activity_sampling_s={self.activity_sampling_s} outside [1.0, 60.0]"
            )
        if not (1 <= self.activity_intensity_min_events <= 100):
            raise ValueError(
                f"activity_intensity_min_events={self.activity_intensity_min_events} outside [1, 100]"
            )
        if not (0.1 <= self.activity_intensity_min_window_s <= 30.0):
            raise ValueError(
                f"activity_intensity_min_window_s={self.activity_intensity_min_window_s} outside [0.1, 30.0]"
            )
        if not (0.5 <= self.session_max_age_h <= 72.0):
            raise ValueError(
                f"session_max_age_h={self.session_max_age_h} outside [0.5, 72.0]"
            )
        if not self.activity_source_id.strip():
            raise ValueError("activity_source_id must be a non-empty string")

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
