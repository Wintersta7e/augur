"""Tests for AugurConfig bool env-var coercion.

Python's bool("false") == True, so the pre-existing _TYPE_COERCIONS
map (which used `bool` as the coercion callable for bool fields) would
silently make AUGUR_*_ENABLED=false into True. The _coerce_bool helper
fixes this so string "false" / "0" / etc. produce Python False.

This test file pins the helper as a pure function. Integration with
from_env for an actual bool field is tested in Task 2 once we have a
bool field in AugurConfig.
"""

from __future__ import annotations

import os
import pytest
from unittest.mock import patch

from blackboard.config import AugurConfig, _coerce_bool


class TestCoerceBoolTrueValues:
    @pytest.mark.parametrize(
        "raw",
        ["true", "True", "TRUE", "1", "yes", "YES", "Yes", "on", "ON", "y", "Y"],
    )
    def test_truthy_string_becomes_true(self, raw: str) -> None:
        assert _coerce_bool(raw) is True

    def test_whitespace_around_truthy_is_stripped(self) -> None:
        assert _coerce_bool(" true ") is True
        assert _coerce_bool("\ttrue\n") is True


class TestCoerceBoolFalseValues:
    @pytest.mark.parametrize(
        "raw",
        [
            "false",
            "False",
            "FALSE",
            "0",
            "no",
            "NO",
            "off",
            "OFF",
            "n",
            "N",
            "",
            "anything_else",
            "2",
            "truish",
        ],
    )
    def test_non_truthy_string_becomes_false(self, raw: str) -> None:
        assert _coerce_bool(raw) is False

    def test_whitespace_around_falsy_is_stripped(self) -> None:
        assert _coerce_bool(" false ") is False




class TestCorrelationTuningFieldsFromEnv:
    """Exercise the _coerce_bool fix through a real bool field."""

    def test_default_values_match_spec(self) -> None:
        cfg = AugurConfig()
        assert cfg.correlation_tuning_enabled is True
        assert cfg.correlation_tuning_alpha == 0.2
        assert cfg.correlation_tuning_enable_threshold == 0.6
        assert cfg.correlation_tuning_disable_threshold == 0.3

    def test_from_env_false_disables_tuning(self) -> None:
        with patch.dict(os.environ, {"AUGUR_CORRELATION_TUNING_ENABLED": "false"}):
            cfg = AugurConfig.from_env()
            assert cfg.correlation_tuning_enabled is False

    def test_from_env_true_keeps_enabled(self) -> None:
        with patch.dict(os.environ, {"AUGUR_CORRELATION_TUNING_ENABLED": "true"}):
            cfg = AugurConfig.from_env()
            assert cfg.correlation_tuning_enabled is True

    def test_from_env_zero_disables_tuning(self) -> None:
        with patch.dict(os.environ, {"AUGUR_CORRELATION_TUNING_ENABLED": "0"}):
            cfg = AugurConfig.from_env()
            assert cfg.correlation_tuning_enabled is False

    def test_from_env_unset_uses_default_true(self) -> None:
        # Ensure the env var is not set
        env = {
            k: v
            for k, v in os.environ.items()
            if k != "AUGUR_CORRELATION_TUNING_ENABLED"
        }
        with patch.dict(os.environ, env, clear=True):
            cfg = AugurConfig.from_env()
            assert cfg.correlation_tuning_enabled is True

    def test_from_env_alpha_override(self) -> None:
        with patch.dict(os.environ, {"AUGUR_CORRELATION_TUNING_ALPHA": "0.05"}):
            cfg = AugurConfig.from_env()
            assert cfg.correlation_tuning_alpha == 0.05

    def test_from_env_enable_threshold_override(self) -> None:
        with patch.dict(
            os.environ, {"AUGUR_CORRELATION_TUNING_ENABLE_THRESHOLD": "0.8"}
        ):
            cfg = AugurConfig.from_env()
            assert cfg.correlation_tuning_enable_threshold == 0.8

    def test_from_env_disable_threshold_override(self) -> None:
        with patch.dict(
            os.environ, {"AUGUR_CORRELATION_TUNING_DISABLE_THRESHOLD": "0.2"}
        ):
            cfg = AugurConfig.from_env()
            assert cfg.correlation_tuning_disable_threshold == 0.2
