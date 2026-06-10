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

from tabula.config import AugurConfig, _coerce_bool


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


class TestFromEnvGracefulCoercionFailure:
    """SEC-01 / COV-09: malformed env vars must not crash from_env().

    If coercion fails for a given field, the default should be kept and a
    warning should be emitted. This replaces the previous behaviour where
    AUGUR_OLLAMA_TIMEOUT=xyz (or similar) would raise ValueError at startup,
    crash-looping every component inside a docker-compose deploy.
    """

    def test_malformed_float_keeps_default(self, caplog) -> None:
        caplog.set_level("WARNING", logger="augur.config")
        with patch.dict(
            os.environ,
            {"AUGUR_CORRELATION_TUNING_ALPHA": "not_a_float"},
        ):
            cfg = AugurConfig.from_env()
            assert cfg.correlation_tuning_alpha == 0.2  # default preserved
        assert any(
            "AUGUR_CORRELATION_TUNING_ALPHA" in record.getMessage()
            and "not_a_float" in record.getMessage()
            for record in caplog.records
        )

    def test_malformed_int_keeps_default(self, caplog) -> None:
        caplog.set_level("WARNING", logger="augur.config")
        with patch.dict(os.environ, {"AUGUR_OLLAMA_TIMEOUT": "xyz"}):
            cfg = AugurConfig.from_env()
            assert cfg.ollama_timeout == 120  # default preserved
        assert any(
            "AUGUR_OLLAMA_TIMEOUT" in record.getMessage() for record in caplog.records
        )

    def test_well_formed_env_vars_still_work_alongside_malformed(self, caplog) -> None:
        caplog.set_level("WARNING", logger="augur.config")
        with patch.dict(
            os.environ,
            {
                "AUGUR_CORRELATION_TUNING_ALPHA": "garbage",  # bad
                "AUGUR_CORRELATION_TUNING_ENABLE_THRESHOLD": "0.75",  # good
            },
        ):
            cfg = AugurConfig.from_env()
            # Bad one falls back to default
            assert cfg.correlation_tuning_alpha == 0.2
            # Good one still overrides
            assert cfg.correlation_tuning_enable_threshold == 0.75

    def test_type_coercion_map_built_from_defaults(self) -> None:
        """ARCH-01: with `from __future__ import annotations`, field.type is a
        string at runtime. The coercion map must be built from the default's
        type, not field.type. Verify the map has sensible entries."""
        from tabula.config import _TYPE_COERCIONS, _coerce_bool

        # bool field → _coerce_bool
        assert _TYPE_COERCIONS["correlation_tuning_enabled"] is _coerce_bool
        # float field → float
        assert _TYPE_COERCIONS["correlation_tuning_alpha"] is float
        # int field → int
        assert _TYPE_COERCIONS["ollama_timeout"] is int
        # str field → str
        assert _TYPE_COERCIONS["nats_url"] is str
