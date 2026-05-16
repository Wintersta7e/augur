"""Unit tests for activity_* config fields and bounds validation."""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from blackboard.config import AugurConfig


def test_defaults_present():
    cfg = AugurConfig()
    assert cfg.activity_sampling_s == 10.0
    assert cfg.activity_intensity_min_events == 1
    assert cfg.activity_intensity_min_window_s == 2.0
    assert cfg.activity_title_allowlist == ""
    assert cfg.session_max_age_h == 12.0
    assert cfg.activity_source_id == "windows-host"


@pytest.mark.parametrize(
    "field,value",
    [
        ("activity_sampling_s", 0.5),
        ("activity_sampling_s", 60.1),
        ("activity_intensity_min_events", -1),
        ("activity_intensity_min_events", 101),
        ("activity_intensity_min_window_s", 0.05),
        ("activity_intensity_min_window_s", 30.1),
        ("session_max_age_h", 0.49),
        ("session_max_age_h", 72.1),
    ],
)
def test_bounds_validation_raises_value_error(field, value):
    with pytest.raises(ValueError, match=field):
        AugurConfig(**{field: value})


def test_source_id_must_be_non_empty():
    with pytest.raises(ValueError, match="activity_source_id"):
        AugurConfig(activity_source_id="")


def test_source_id_rejects_whitespace_only():
    with pytest.raises(ValueError, match="activity_source_id"):
        AugurConfig(activity_source_id="   ")


def test_env_override_sampling():
    with patch.dict(os.environ, {"AUGUR_ACTIVITY_SAMPLING_S": "5.0"}, clear=False):
        cfg = AugurConfig.from_env()
        assert cfg.activity_sampling_s == 5.0


def test_env_override_title_allowlist_is_plain_string():
    """Allowlist is a string; consumers .split(',') at use-site.

    This avoids the tuple-of-chars footgun where tuple('a,b') == ('a',',','b').
    """
    with patch.dict(
        os.environ, {"AUGUR_ACTIVITY_TITLE_ALLOWLIST": "code,terminal"}, clear=False
    ):
        cfg = AugurConfig.from_env()
        assert cfg.activity_title_allowlist == "code,terminal"
        # Verify the consumer pattern works
        apps = [s.strip() for s in cfg.activity_title_allowlist.split(",") if s.strip()]
        assert apps == ["code", "terminal"]


def test_env_override_session_max_age():
    with patch.dict(os.environ, {"AUGUR_SESSION_MAX_AGE_H": "1.0"}, clear=False):
        cfg = AugurConfig.from_env()
        assert cfg.session_max_age_h == 1.0


def test_env_override_source_id():
    with patch.dict(os.environ, {"AUGUR_ACTIVITY_SOURCE_ID": "laptop-2"}, clear=False):
        cfg = AugurConfig.from_env()
        assert cfg.activity_source_id == "laptop-2"
