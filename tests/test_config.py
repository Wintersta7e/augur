"""Tests for tabula/config.py — AugurConfig centralized configuration."""

from __future__ import annotations

import sys
from dataclasses import FrozenInstanceError
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tabula.config import AugurConfig  # noqa: E402


class TestDefaults:
    def test_nats_url_default(self) -> None:
        cfg = AugurConfig()
        # IPv4 literal: localhost resolves to ::1 first on some Windows asyncio
        # event loops, but Docker bindings are IPv4-only. Pinning the literal
        # avoids a silent connect timeout on the daemon's NATS client.
        assert cfg.nats_url == "nats://127.0.0.1:4222"

    def test_redis_url_default(self) -> None:
        cfg = AugurConfig()
        assert cfg.redis_url == "redis://127.0.0.1:6379"

    def test_min_observations_default(self) -> None:
        cfg = AugurConfig()
        # Raised from 3 → 15: a 3-sample baseline forms so quickly that the 4th
        # observation almost always looks anomalous in apps just opened, which
        # drove low-quality LLM advice during live testing.
        assert cfg.min_observations == 15

    def test_ollama_url_default(self) -> None:
        cfg = AugurConfig()
        assert cfg.ollama_url == "http://host.docker.internal:11434"

    def test_ollama_timeout_default(self) -> None:
        cfg = AugurConfig()
        assert cfg.ollama_timeout == 120

    def test_default_sigma_threshold(self) -> None:
        cfg = AugurConfig()
        assert cfg.default_sigma_threshold == 2.0

    def test_ollama_model_default(self) -> None:
        cfg = AugurConfig()
        assert cfg.ollama_model == "qwen2.5:32b"


class TestFromEnv:
    def test_nats_url_override(self) -> None:
        with patch.dict("os.environ", {"AUGUR_NATS_URL": "nats://remotehost:4222"}):
            cfg = AugurConfig.from_env()
        assert cfg.nats_url == "nats://remotehost:4222"

    def test_ollama_timeout_int_coercion(self) -> None:
        with patch.dict("os.environ", {"AUGUR_OLLAMA_TIMEOUT": "60"}):
            cfg = AugurConfig.from_env()
        assert cfg.ollama_timeout == 60
        assert isinstance(cfg.ollama_timeout, int)

    def test_default_sigma_threshold_float_coercion(self) -> None:
        with patch.dict("os.environ", {"AUGUR_DEFAULT_SIGMA_THRESHOLD": "3.5"}):
            cfg = AugurConfig.from_env()
        assert cfg.default_sigma_threshold == 3.5
        assert isinstance(cfg.default_sigma_threshold, float)

    def test_unset_env_uses_default(self) -> None:
        # Ensure AUGUR_NATS_URL is not in the environment
        env_without = {
            k: v for k, v in __import__("os").environ.items() if k != "AUGUR_NATS_URL"
        }
        with patch.dict("os.environ", env_without, clear=True):
            cfg = AugurConfig.from_env()
        assert cfg.nats_url == "nats://127.0.0.1:4222"


class TestFrozenImmutability:
    def test_assigning_to_nats_url_raises(self) -> None:
        cfg = AugurConfig()
        with pytest.raises(FrozenInstanceError):
            cfg.nats_url = "nats://other:4222"  # type: ignore[misc]

    def test_assigning_to_ollama_model_raises(self) -> None:
        cfg = AugurConfig()
        with pytest.raises(FrozenInstanceError):
            cfg.ollama_model = "llama3"  # type: ignore[misc]


class TestAsDict:
    def test_contains_all_expected_keys(self) -> None:
        cfg = AugurConfig()
        d = cfg.as_dict()
        expected_keys = {
            "nats_url",
            "redis_url",
            "ollama_url",
            "ollama_timeout",
            "nats_connect_timeout",
            "redis_connect_timeout",
            "default_sigma_threshold",
            "ewma_alpha",
            "hst_n_trees",
            "hst_height",
            "hst_window_size",
            "min_observations",
            "hst_threshold",
            "severity_medium_sigma",
            "severity_high_sigma",
            "ollama_model",
            "advisor_lock_timeout",
            "feedback_explicit_timeout",
            "feedback_behavioral_window",
            "history_max_events",
            "sigma_adjust_step",
            "sigma_min",
            "sigma_max",
            "utility_mutation_threshold",
        }
        assert expected_keys.issubset(d.keys())

    def test_values_match_defaults(self) -> None:
        cfg = AugurConfig()
        d = cfg.as_dict()
        assert d["nats_url"] == "nats://127.0.0.1:4222"
        assert d["redis_url"] == "redis://127.0.0.1:6379"
        assert d["ollama_url"] == "http://host.docker.internal:11434"
        assert d["ollama_timeout"] == 120
        assert d["default_sigma_threshold"] == 2.0
        assert d["ollama_model"] == "qwen2.5:32b"


class TestRedisProperties:
    def test_redis_host_default(self) -> None:
        cfg = AugurConfig()
        assert cfg.redis_host == "127.0.0.1"

    def test_redis_port_default(self) -> None:
        cfg = AugurConfig()
        assert cfg.redis_port == 6379

    def test_redis_host_custom(self) -> None:
        with patch.dict("os.environ", {"AUGUR_REDIS_URL": "redis://myhost:7777"}):
            cfg = AugurConfig.from_env()
        assert cfg.redis_host == "myhost"

    def test_redis_port_custom(self) -> None:
        with patch.dict("os.environ", {"AUGUR_REDIS_URL": "redis://myhost:7777"}):
            cfg = AugurConfig.from_env()
        assert cfg.redis_port == 7777


class TestDetectorConfigParity:
    """Guard the duplication between AugurConfig and vigil.anomaly_detector.

    DEFAULT_THRESHOLDS in the detector currently shadows AugurConfig values; if
    they drift, the MCP server reports one number while the detector behaves with
    another. This test fails immediately when somebody changes one without the
    other.
    """

    def test_detector_default_min_observations_matches_config(self) -> None:
        from vigil.anomaly_detector import DEFAULT_THRESHOLDS

        cfg = AugurConfig()
        assert DEFAULT_THRESHOLDS["min_observations"] == cfg.min_observations


class TestCorrelationWindowDefaults:
    def test_correlation_window_s_default(self) -> None:
        cfg = AugurConfig.from_env()
        assert cfg.correlation_window_s == 30.0

    def test_correlation_window_s_env_override(self) -> None:
        with patch.dict("os.environ", {"AUGUR_CORRELATION_WINDOW_S": "45.0"}):
            cfg = AugurConfig.from_env()
        assert cfg.correlation_window_s == 45.0

    def test_correlation_window_min_s_default(self) -> None:
        cfg = AugurConfig.from_env()
        assert cfg.correlation_window_min_s == 5.0

    def test_correlation_window_max_s_default(self) -> None:
        cfg = AugurConfig.from_env()
        assert cfg.correlation_window_max_s == 120.0

    def test_correlation_window_tuning_alpha_default(self) -> None:
        cfg = AugurConfig.from_env()
        assert cfg.correlation_window_tuning_alpha == 0.2

    def test_correlation_window_tuning_fields_overridable(self) -> None:
        with patch.dict(
            "os.environ",
            {
                "AUGUR_CORRELATION_WINDOW_MIN_S": "3.0",
                "AUGUR_CORRELATION_WINDOW_MAX_S": "180.0",
                "AUGUR_CORRELATION_WINDOW_TUNING_ALPHA": "0.3",
                "AUGUR_CORRELATION_WINDOW_LAG_MULTIPLIER": "3.0",
                "AUGUR_CORRELATION_WINDOW_TUNING_HYSTERESIS_PCT": "0.10",
            },
        ):
            cfg = AugurConfig.from_env()
        assert cfg.correlation_window_min_s == 3.0
        assert cfg.correlation_window_max_s == 180.0
        assert cfg.correlation_window_tuning_alpha == 0.3
        assert cfg.correlation_window_lag_multiplier == 3.0
        assert cfg.correlation_window_tuning_hysteresis_pct == 0.10


class TestCorrelationWindowBounds:
    def test_correlation_window_s_below_min_raises(self) -> None:
        with patch.dict("os.environ", {"AUGUR_CORRELATION_WINDOW_S": "1.0"}):
            with pytest.raises(ValueError, match="correlation_window_s"):
                AugurConfig.from_env()

    def test_correlation_window_s_above_max_raises(self) -> None:
        with patch.dict("os.environ", {"AUGUR_CORRELATION_WINDOW_S": "200.0"}):
            with pytest.raises(ValueError, match="correlation_window_s"):
                AugurConfig.from_env()

    def test_correlation_window_max_s_below_window_s_raises(self) -> None:
        # Default correlation_window_s is 30.0; setting max to 10.0 should fail
        with patch.dict("os.environ", {"AUGUR_CORRELATION_WINDOW_MAX_S": "10.0"}):
            with pytest.raises(ValueError, match="correlation_window_max_s"):
                AugurConfig.from_env()

    def test_correlation_window_lag_multiplier_out_of_range_raises(self) -> None:
        with patch.dict(
            "os.environ", {"AUGUR_CORRELATION_WINDOW_LAG_MULTIPLIER": "10.0"}
        ):
            with pytest.raises(ValueError, match="lag_multiplier"):
                AugurConfig.from_env()

    def test_correlation_window_tuning_hysteresis_out_of_range_raises(self) -> None:
        with patch.dict(
            "os.environ", {"AUGUR_CORRELATION_WINDOW_TUNING_HYSTERESIS_PCT": "1.5"}
        ):
            with pytest.raises(ValueError, match="hysteresis"):
                AugurConfig.from_env()

    def test_correlation_window_tuning_alpha_out_of_range_raises(self) -> None:
        with patch.dict("os.environ", {"AUGUR_CORRELATION_WINDOW_TUNING_ALPHA": "1.5"}):
            with pytest.raises(ValueError, match="alpha"):
                AugurConfig.from_env()

    def test_correlation_window_min_s_above_window_s_raises(self) -> None:
        """correlation_window_min_s must not exceed correlation_window_s."""
        with patch.dict("os.environ", {"AUGUR_CORRELATION_WINDOW_MIN_S": "60.0"}):
            # Default correlation_window_s is 30.0, so min=60 violates the invariant.
            with pytest.raises(ValueError, match="correlation_window_min_s"):
                AugurConfig.from_env()

    def test_correlation_window_min_s_zero_or_negative_raises(self) -> None:
        with patch.dict("os.environ", {"AUGUR_CORRELATION_WINDOW_MIN_S": "0"}):
            with pytest.raises(ValueError, match="correlation_window_min_s"):
                AugurConfig.from_env()


class TestRedisDb:
    def test_db_index_parsed_from_url_path(self) -> None:
        assert AugurConfig(redis_url="redis://127.0.0.1:6379/1").redis_db == 1

    def test_defaults_to_zero_when_absent(self) -> None:
        assert AugurConfig(redis_url="redis://127.0.0.1:6379").redis_db == 0

    def test_non_numeric_path_is_zero(self) -> None:
        assert AugurConfig(redis_url="redis://127.0.0.1:6379/notadb").redis_db == 0

    def test_bare_slash_is_zero(self) -> None:
        assert AugurConfig(redis_url="redis://127.0.0.1:6379/").redis_db == 0

    def test_unicode_digit_is_zero(self) -> None:
        assert AugurConfig(redis_url="redis://127.0.0.1:6379/²").redis_db == 0
