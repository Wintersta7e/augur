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
    # Idle reclamation for per-entity in-memory baselines (each owns a River
    # HST model). Evict a (domain, entity) model not seen for this many seconds
    # so a high-churn entity namespace can't pin memory up to
    # MAX_BASELINE_ENTITIES. 0 disables eviction (never reclaim).
    baseline_entity_idle_evict_s: float = 3600.0

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

    # ── Advisor gate ───────────────────────────────────────────────────────
    gate_enabled: bool = True
    gate_central_tolerance_enabled: bool = True
    gate_refractory_enabled: bool = True
    gate_novelty_enabled: bool = True
    gate_habituation_enabled: bool = True
    gate_reservoir_enabled: bool = True
    gate_credibility_enabled: bool = True
    gate_cost_tier_enabled: bool = True
    gate_bet_hedge_enabled: bool = True
    gate_anti_starvation_enabled: bool = True
    gate_absolute_refractory_s: int = 45
    gate_relative_refractory_s: int = 180
    gate_habituation_tau_s: int = 600
    gate_habituation_alpha: float = 0.3
    gate_habituation_floor_min: float = 0.2
    gate_habituation_r_threshold: float = 0.5
    gate_novelty_familiar_min: int = 3
    gate_weber_fraction: float = 0.15
    gate_reservoir_on_count: int = 3
    gate_reservoir_off_count: int = 1
    gate_reservoir_leak_tau_s: int = 120
    gate_pressure_alpha: float = 0.2
    gate_pressure_weight: float = 1.0
    gate_pressure_cap: float = 3.0
    gate_credibility_alpha: float = 0.1
    gate_credibility_decay_alpha: float = 0.02
    gate_cred_mid: float = 0.5
    gate_cred_max_p: float = 0.8
    gate_behavioral_weight: float = 0.2
    gate_explicit_weight: float = 1.0
    gate_behavioral_deadband: float = 0.15
    gate_behavioral_min_samples: int = 5
    gate_bet_hedge_epsilon: float = 0.1
    gate_cost_tier_persistence_count: int = 3
    gate_max_consecutive_suppressions: int = 8
    gate_max_channel_silence_s: int = 1800
    gate_max_release_wait_s: int = 30
    gate_max_release_overtake: int = 5
    gate_mrt_withheld_rating: bool = False
    gate_tier1_mode: str = "note"

    # ── Lane 1: causal measurement & data-quality (spec 2026-06-09) ──────────
    # 1A — domain-agnostic surprise-reduction outcome metric
    post_decision_window: int = 3
    min_baseline_std: float = 0.01
    outcome_trend_bonus: float = 0.1
    # 1B — calibration-era control-arm explicit rating (withheld arm)
    gate_mrt_withheld_rating_rate: float = 0.12
    gate_mrt_withheld_rating_max_sessions: int = 15
    # 1C — River drift detector → deliberate baseline reset
    drift_detector_enabled: bool = True
    drift_detector: str = "adwin"  # {"adwin", "pagehinkley"}
    drift_reset_cooldown_obs: int = 30
    drift_restart_std_factor: float = 1.0
    # 1E — prompt-mutation safety
    prompt_rollback_margin: float = 0.1
    prompt_forbidden_patterns: tuple[str, ...] = (
        "take a break",
        "you are fatigued",
        "you seem distracted",
        "you appear stuck",
        "as an ai",
    )
    # ── Lane 2 — Memoria memory spine (spec 2026-06-10) ─────────────────────
    memory_store_enabled: bool = True
    memory_prune_r: float = 0.05
    memory_promote_s: int = 14
    memory_s_growth_factor: float = 0.5
    memory_s_min: float = 0.1
    memory_s_max: int = 365
    max_memory_items: int = 5000
    memory_decay_form: str = "exponential"  # {"exponential", "powerlaw"}
    # ── Praefectus: faculty supervision & health (spec 2026-06-10) ───────────
    praefectus_enabled: bool = True
    praefectus_heartbeat_interval_s: float = 10.0
    praefectus_stale_after_s: float = 30.0
    praefectus_dead_after_s: float = 90.0
    praefectus_warmup_s: float = 30.0
    praefectus_tick_s: float = 5.0
    praefectus_stall_window_s: float = 0.0  # 0 = auto → effective_stall_window_s
    praefectus_stall_tolerance: int = 1
    praefectus_stall_min_events: int = 2
    praefectus_delivery_failure_spike: int = 3
    praefectus_reflection_window_s: float = (
        0.0  # 0 = auto → effective_reflection_window_s
    )
    # ── Imperator I: Awareness read-models (spec 2026-06-14) ─────────────────
    imperator_enabled: bool = True
    imperator_tick_s: float = 5.0
    imperator_salience_window_s: float = 300.0
    imperator_rate_window_s: float = 900.0
    imperator_baseline_trained_obs: int = 15

    # ── Imperator II: Self-Improvement (spec 2026-06-14) ─────────────────────
    imperator_ii_enabled: bool = True
    imperator_ii_apply_enabled: bool = False  # watch-first: default OFF
    imperator_ii_num_predict: int = 512
    imperator_ii_max_proposals_per_cycle: int = 5
    imperator_ii_min_interval_s: float = 60.0
    imperator_ii_freshness_timeout_s: float = 15.0
    imperator_ii_dedupe_staleness_s: float = 86400.0
    min_prompt_len: int = 20

    # ── Imperator III: Dialogue (spec 2026-06-20) ────────────────────────────
    dialogue_enabled: bool = True
    dialogue_model: str = "qwen2.5:32b"
    dialogue_num_predict: int = 512
    dialogue_temperature: float = 0.6
    dialogue_context_max_turns: int = 12
    dialogue_context_token_budget: int = 2048
    dialogue_pending_ttl_s: float = 300.0
    dialogue_log_cap: int = 500
    dialogue_confirmed_apply_enabled: bool = True

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
        if not (0 <= self.gate_behavioral_weight <= self.gate_explicit_weight):
            raise ValueError(
                f"gate_behavioral_weight={self.gate_behavioral_weight} must be in "
                f"[0, gate_explicit_weight={self.gate_explicit_weight}]"
            )
        if self.gate_tier1_mode not in {"note", "silent"}:
            raise ValueError(
                f"gate_tier1_mode={self.gate_tier1_mode!r} must be 'note' or 'silent'"
            )
        if self.baseline_entity_idle_evict_s < 0.0:
            raise ValueError(
                f"baseline_entity_idle_evict_s={self.baseline_entity_idle_evict_s} "
                "must be >= 0 (0 disables idle eviction)"
            )
        # ── Lane 1 bounds (spec 2026-06-09) ─────────────────────────────────
        if not (2 <= self.post_decision_window <= 50):
            raise ValueError(
                f"post_decision_window={self.post_decision_window} outside [2, 50]"
            )
        if not (self.min_baseline_std > 0.0):
            raise ValueError(f"min_baseline_std={self.min_baseline_std} must be > 0")
        if not (0.0 <= self.outcome_trend_bonus <= 0.5):
            raise ValueError(
                f"outcome_trend_bonus={self.outcome_trend_bonus} outside [0, 0.5]"
            )
        if not (0.0 <= self.gate_mrt_withheld_rating_rate <= 0.5):
            raise ValueError(
                f"gate_mrt_withheld_rating_rate={self.gate_mrt_withheld_rating_rate} outside [0, 0.5]"
            )
        if not (1 <= self.gate_mrt_withheld_rating_max_sessions <= 1000):
            raise ValueError(
                f"gate_mrt_withheld_rating_max_sessions={self.gate_mrt_withheld_rating_max_sessions} outside [1, 1000]"
            )
        if self.drift_detector not in {"adwin", "pagehinkley"}:
            raise ValueError(
                f"drift_detector={self.drift_detector!r} must be 'adwin' or 'pagehinkley'"
            )
        if not (0 <= self.drift_reset_cooldown_obs <= 10000):
            raise ValueError(
                f"drift_reset_cooldown_obs={self.drift_reset_cooldown_obs} outside [0, 10000]"
            )
        if not (0.25 <= self.drift_restart_std_factor <= 4.0):
            # Bound matches the effective clamp in anomaly_detector._maybe_drift_reset
            # ([|Δ|*0.25, |Δ|*4.0]); values outside it had no effect, so reject them.
            raise ValueError(
                f"drift_restart_std_factor={self.drift_restart_std_factor} outside [0.25, 4.0]"
            )
        if not (0.0 <= self.prompt_rollback_margin <= 1.0):
            raise ValueError(
                f"prompt_rollback_margin={self.prompt_rollback_margin} outside [0, 1]"
            )
        # ── Lane 2 bounds (Memoria spec 2026-06-10) ─────────────────────────
        if not (0.0 <= self.memory_prune_r <= 0.5):
            raise ValueError(f"memory_prune_r={self.memory_prune_r} outside [0, 0.5]")
        if not (2 <= self.memory_promote_s <= 1000):
            raise ValueError(
                f"memory_promote_s={self.memory_promote_s} outside [2, 1000]"
            )
        if not (0.0 <= self.memory_s_growth_factor <= 5.0):
            raise ValueError(
                f"memory_s_growth_factor={self.memory_s_growth_factor} outside [0, 5]"
            )
        if not (0.0 < self.memory_s_min <= 1.0):
            raise ValueError(f"memory_s_min={self.memory_s_min} must be in (0, 1]")
        if not (10 <= self.memory_s_max <= 100_000):
            raise ValueError(f"memory_s_max={self.memory_s_max} outside [10, 100000]")
        if not (100 <= self.max_memory_items <= 100_000):
            raise ValueError(
                f"max_memory_items={self.max_memory_items} outside [100, 100000]"
            )
        if self.memory_decay_form not in {"exponential", "powerlaw"}:
            raise ValueError(
                f"memory_decay_form={self.memory_decay_form!r} must be "
                "'exponential' or 'powerlaw'"
            )
        # ── Praefectus bounds (spec 2026-06-10) ──────────────────────────────
        if not (1.0 <= self.praefectus_heartbeat_interval_s <= 120.0):
            raise ValueError(
                f"praefectus_heartbeat_interval_s={self.praefectus_heartbeat_interval_s} outside [1, 120]"
            )
        if not (
            self.praefectus_heartbeat_interval_s
            <= self.praefectus_stale_after_s
            <= 600.0
        ):
            raise ValueError(
                f"praefectus_stale_after_s={self.praefectus_stale_after_s} must be in "
                f"[heartbeat_interval={self.praefectus_heartbeat_interval_s}, 600]"
            )
        if not (
            self.praefectus_stale_after_s <= self.praefectus_dead_after_s <= 3600.0
        ):
            raise ValueError(
                f"praefectus_dead_after_s={self.praefectus_dead_after_s} must be in "
                f"[stale_after={self.praefectus_stale_after_s}, 3600]"
            )
        if not (0.0 <= self.praefectus_warmup_s <= 600.0):
            raise ValueError(
                f"praefectus_warmup_s={self.praefectus_warmup_s} outside [0, 600]"
            )
        if not (1.0 <= self.praefectus_tick_s <= 60.0):
            raise ValueError(
                f"praefectus_tick_s={self.praefectus_tick_s} outside [1, 60]"
            )
        if self.praefectus_stall_window_s != 0.0 and not (
            self.ollama_timeout + 60 <= self.praefectus_stall_window_s <= 1800.0
        ):
            raise ValueError(
                f"praefectus_stall_window_s={self.praefectus_stall_window_s} must be 0 (auto) "
                f"or in [ollama_timeout+60={self.ollama_timeout + 60}, 1800]"
            )
        if not (0 <= self.praefectus_stall_tolerance <= 100):
            raise ValueError(
                f"praefectus_stall_tolerance={self.praefectus_stall_tolerance} outside [0, 100]"
            )
        if not (1 <= self.praefectus_stall_min_events <= 100):
            raise ValueError(
                f"praefectus_stall_min_events={self.praefectus_stall_min_events} outside [1, 100]"
            )
        if not (1 <= self.praefectus_delivery_failure_spike <= 100):
            raise ValueError(
                f"praefectus_delivery_failure_spike={self.praefectus_delivery_failure_spike} outside [1, 100]"
            )
        if self.praefectus_reflection_window_s != 0.0 and not (
            2 * self.ollama_timeout <= self.praefectus_reflection_window_s <= 3600.0
        ):
            raise ValueError(
                f"praefectus_reflection_window_s={self.praefectus_reflection_window_s} must be 0 (auto) "
                f"or in [2*ollama_timeout={2 * self.ollama_timeout}, 3600]"
            )
        # ── Imperator bounds ──
        if not (0.5 <= self.imperator_tick_s <= 60.0):
            raise ValueError(
                f"imperator_tick_s={self.imperator_tick_s} outside [0.5, 60]"
            )
        if not (10.0 <= self.imperator_salience_window_s <= 3600.0):
            raise ValueError(
                f"imperator_salience_window_s={self.imperator_salience_window_s} outside [10, 3600]"
            )
        if not (10.0 <= self.imperator_rate_window_s <= 7200.0):
            raise ValueError(
                f"imperator_rate_window_s={self.imperator_rate_window_s} outside [10, 7200]"
            )
        if not (1 <= self.imperator_baseline_trained_obs <= 1000):
            raise ValueError(
                f"imperator_baseline_trained_obs={self.imperator_baseline_trained_obs} outside [1, 1000]"
            )
        # ── Imperator II bounds ──
        if not (16 <= self.imperator_ii_num_predict <= 4096):
            raise ValueError(
                f"imperator_ii_num_predict={self.imperator_ii_num_predict} outside [16, 4096]"
            )
        if not (1 <= self.imperator_ii_max_proposals_per_cycle <= 50):
            raise ValueError("imperator_ii_max_proposals_per_cycle outside [1, 50]")
        if not (0.0 <= self.imperator_ii_min_interval_s <= 3600.0):
            raise ValueError("imperator_ii_min_interval_s outside [0, 3600]")
        if not (1.0 <= self.imperator_ii_freshness_timeout_s <= 120.0):
            raise ValueError("imperator_ii_freshness_timeout_s outside [1, 120]")
        if not (1.0 <= self.imperator_ii_dedupe_staleness_s <= 31_536_000.0):
            raise ValueError("imperator_ii_dedupe_staleness_s outside [1, 31536000]")
        if not (1 <= self.min_prompt_len <= 500):
            raise ValueError("min_prompt_len outside [1, 500]")
        # ── Imperator III: Dialogue bounds ────────────────────────────────────
        if not (0.0 <= self.dialogue_temperature <= 2.0):
            raise ValueError("dialogue_temperature must be in [0.0, 2.0]")
        if self.dialogue_num_predict < 16:
            raise ValueError("dialogue_num_predict must be >= 16")
        if self.dialogue_pending_ttl_s <= 0:
            raise ValueError("dialogue_pending_ttl_s must be > 0")
        if self.dialogue_context_max_turns < 0:
            raise ValueError("dialogue_context_max_turns must be >= 0")

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

    @property
    def effective_stall_window_s(self) -> float:
        """Resolved stall window: the field if set (>0), else max(300, 2*ollama_timeout).
        Lazy on purpose — from_env() builds defaults via asdict(cls()) BEFORE applying
        AUGUR_* overrides, so the raw 0.0 sentinel must survive to here and resolve
        against the FINAL ollama_timeout (mirrors the redis_host/redis_port idiom)."""
        if self.praefectus_stall_window_s > 0:
            return self.praefectus_stall_window_s
        return max(300.0, 2.0 * self.ollama_timeout)

    @property
    def effective_reflection_window_s(self) -> float:
        """Resolved reflection-lag horizon: the field if set (>0), else max(300, 2*ollama_timeout)."""
        if self.praefectus_reflection_window_s > 0:
            return self.praefectus_reflection_window_s
        return max(300.0, 2.0 * self.ollama_timeout)


def _coerce_gate_tier1_mode(v: str) -> str:
    """Validate gate_tier1_mode value; raises ValueError for unknown values."""
    if v in {"note", "silent"}:
        return v
    raise ValueError(f"gate_tier1_mode={v!r} must be 'note' or 'silent'")


def _coerce_drift_detector(v: str) -> str:
    """Validate drift_detector value; raises ValueError for unknown values."""
    if v in {"adwin", "pagehinkley"}:
        return v
    raise ValueError(f"drift_detector={v!r} must be 'adwin' or 'pagehinkley'")


def _coerce_memory_decay_form(v: str) -> str:
    """Validate memory_decay_form value; raises ValueError for unknown values."""
    if v in {"exponential", "powerlaw"}:
        return v
    raise ValueError(f"memory_decay_form={v!r} must be 'exponential' or 'powerlaw'")


def _coerce_str_tuple(v: str) -> tuple[str, ...]:
    """Comma-split into a tuple of non-empty stripped strings.

    NOT ``tuple(v)`` (which the auto-build loop would assign, splitting a string
    into characters). Used for AUGUR_PROMPT_FORBIDDEN_PATTERNS.
    """
    parts = tuple(p.strip() for p in v.split(",") if p.strip())
    if not parts:
        raise ValueError("expected a non-empty comma-separated list")
    return parts


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
# gate_tier1_mode needs a validating coercion assigned AFTER the auto-build
# loop, which would otherwise overwrite it with plain str.
_TYPE_COERCIONS["gate_tier1_mode"] = _coerce_gate_tier1_mode
# Same pattern for the Lane-1 string/tuple-enum fields: the auto-build loop maps
# drift_detector→str and prompt_forbidden_patterns→tuple (which char-splits).
_TYPE_COERCIONS["drift_detector"] = _coerce_drift_detector
_TYPE_COERCIONS["prompt_forbidden_patterns"] = _coerce_str_tuple
_TYPE_COERCIONS["memory_decay_form"] = _coerce_memory_decay_form
