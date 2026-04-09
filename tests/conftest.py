"""Shared fixtures for Augur test suite."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# Ensure project root is on sys.path so imports work without installation
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def mock_persistence_manager() -> MagicMock:
    """PersistenceManager stub — no Redis required.

    COV-08: explicit return values for every save_*/load_* method exposed
    on PersistenceManager so a test using this fixture gets predictable
    ``None`` / ``[]`` results rather than a truthy MagicMock default.
    The truthy default was a subtle footgun: ``pm.load_rule_confidence()
    or {}`` would have returned the MagicMock instance instead of ``{}``,
    silently bypassing first-observation initialization in
    analyze_correlation_tuning and similar.
    """
    pm = MagicMock()
    # Pre-existing load methods
    pm.load_thresholds.return_value = None
    pm.load_baseline.return_value = None
    pm.get_history.return_value = []
    pm.get_feedback.return_value = None
    pm.get_all_feedback.return_value = []
    pm.load_prompt.return_value = None
    pm.get_prompt_history.return_value = []
    # Phase 3B additions
    pm.load_escalation_matrix.return_value = None
    # Option A1 additions
    pm.load_correlation_graph.return_value = None
    pm.list_correlation_graphs.return_value = []
    # Deep-review extensions
    pm.load_rule_confidence.return_value = None
    pm.load_reflection.return_value = None
    pm.load_last_anomaly.return_value = None
    pm.load_last_advice.return_value = None
    pm.is_tuning_applied.return_value = False
    # Save/mark methods return None explicitly (MagicMock default is a
    # new MagicMock, not None — we want callers to see None so a naive
    # `if pm.save_*(...)` check doesn't behave unexpectedly.)
    pm.save_baseline.return_value = None
    pm.save_feedback.return_value = None
    pm.save_prompt.return_value = None
    pm.save_thresholds.return_value = None
    pm.save_escalation_matrix.return_value = None
    pm.save_correlation_graph.return_value = None
    pm.save_rule_confidence.return_value = None
    pm.save_reflection.return_value = None
    pm.save_last_anomaly.return_value = None
    pm.save_last_advice.return_value = None
    pm.mark_tuning_applied.return_value = None
    return pm
