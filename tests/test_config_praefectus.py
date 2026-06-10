"""Praefectus config fields, derived-window properties, and bounds."""

import os
from unittest import mock

import pytest

from tabula.config import AugurConfig


def test_defaults_present():
    c = AugurConfig()
    assert c.praefectus_enabled is True
    assert c.praefectus_heartbeat_interval_s == 10.0
    assert c.praefectus_stale_after_s == 30.0
    assert c.praefectus_dead_after_s == 90.0
    assert c.praefectus_stall_window_s == 0.0  # sentinel
    assert c.praefectus_reflection_window_s == 0.0  # sentinel


def test_effective_windows_auto_from_default_timeout():
    c = AugurConfig()  # ollama_timeout default 120
    assert c.effective_stall_window_s == 300.0  # max(300, 2*120)
    assert c.effective_reflection_window_s == 300.0


def test_effective_windows_track_overridden_timeout():
    c = AugurConfig(ollama_timeout=240)
    assert c.effective_stall_window_s == 480.0  # max(300, 2*240)
    assert c.effective_reflection_window_s == 480.0


def test_effective_window_regression_via_from_env():
    # The asdict-before-overrides trap: the sentinel must survive from_env so the
    # property computes against the FINAL ollama_timeout, not the default.
    with mock.patch.dict(os.environ, {"AUGUR_OLLAMA_TIMEOUT": "240"}, clear=False):
        c = AugurConfig.from_env()
    assert c.ollama_timeout == 240
    assert c.effective_stall_window_s == 480.0


def test_explicit_window_used_when_set():
    c = AugurConfig(praefectus_stall_window_s=600.0)
    assert c.effective_stall_window_s == 600.0


def test_undersized_manual_window_rejected():
    with pytest.raises(ValueError):
        AugurConfig(praefectus_stall_window_s=100.0)  # < ollama_timeout(120)+60


def test_stale_must_not_exceed_dead():
    with pytest.raises(ValueError):
        AugurConfig(praefectus_stale_after_s=200.0, praefectus_dead_after_s=90.0)
