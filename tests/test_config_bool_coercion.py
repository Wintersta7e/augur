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

import pytest

from blackboard.config import _coerce_bool


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
