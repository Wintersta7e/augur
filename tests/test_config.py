"""Tests for blackboard/config.py — AugurConfig centralized configuration."""

from __future__ import annotations

import sys
from dataclasses import FrozenInstanceError
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from blackboard.config import AugurConfig  # noqa: E402


class TestDefaults:
    def test_nats_url_default(self) -> None:
        cfg = AugurConfig()
        assert cfg.nats_url == "nats://localhost:4222"

    def test_redis_url_default(self) -> None:
        cfg = AugurConfig()
        assert cfg.redis_url == "redis://localhost:6379"

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
        assert cfg.nats_url == "nats://localhost:4222"


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
        assert d["nats_url"] == "nats://localhost:4222"
        assert d["redis_url"] == "redis://localhost:6379"
        assert d["ollama_url"] == "http://host.docker.internal:11434"
        assert d["ollama_timeout"] == 120
        assert d["default_sigma_threshold"] == 2.0
        assert d["ollama_model"] == "qwen2.5:32b"


class TestRedisProperties:
    def test_redis_host_default(self) -> None:
        cfg = AugurConfig()
        assert cfg.redis_host == "localhost"

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
