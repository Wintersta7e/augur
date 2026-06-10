"""Tests for new PersistenceManager rule_window_state methods."""

import json
from unittest.mock import MagicMock

from tabula.persistence import PersistenceManager


def test_save_rule_window_state_writes_redis():
    redis_mock = MagicMock()
    pm = PersistenceManager(redis_mock)
    state = {"LOW+LOW": {"ewma_lag": 8.5}, "MEDIUM+HIGH": {"ewma_lag": 12.4}}
    pm.save_rule_window_state(state)
    redis_mock.set.assert_called_once()
    args, kwargs = redis_mock.set.call_args
    assert args[0] == "augur:config:rule_window_state"
    assert json.loads(args[1]) == state


def test_load_rule_window_state_returns_parsed():
    redis_mock = MagicMock()
    state = {"LOW+LOW": {"ewma_lag": 8.5}}
    redis_mock.get.return_value = json.dumps(state).encode()
    pm = PersistenceManager(redis_mock)
    assert pm.load_rule_window_state() == state


def test_load_rule_window_state_missing_returns_empty():
    redis_mock = MagicMock()
    redis_mock.get.return_value = None
    pm = PersistenceManager(redis_mock)
    assert pm.load_rule_window_state() == {}


def test_load_rule_window_state_corrupt_returns_empty():
    redis_mock = MagicMock()
    redis_mock.get.return_value = b"not json"
    pm = PersistenceManager(redis_mock)
    assert pm.load_rule_window_state() == {}


# Atomic state save (round-3 fix) -------------------------------------------


def test_save_tuning_state_uses_pipeline():
    """Both confidence and window_state should be written via pipeline.execute()
    so they commit atomically."""
    redis_mock = MagicMock()
    pipe_mock = MagicMock()
    redis_mock.pipeline.return_value = pipe_mock

    pm = PersistenceManager(redis_mock)
    pm.save_tuning_state(
        confidence={"LOW+LOW": {"confidence": 0.8, "restore_target": "MEDIUM"}},
        window_state={"LOW+LOW": {"ewma_lag": 8.5}},
    )

    redis_mock.pipeline.assert_called_once()
    # Both SETs should have been queued on the pipeline
    assert pipe_mock.set.call_count == 2
    # And executed atomically
    pipe_mock.execute.assert_called_once()


def test_save_tuning_state_with_only_confidence():
    redis_mock = MagicMock()
    pipe_mock = MagicMock()
    redis_mock.pipeline.return_value = pipe_mock

    pm = PersistenceManager(redis_mock)
    pm.save_tuning_state(
        confidence={"LOW+LOW": {"confidence": 0.5}},
        window_state=None,
    )

    pipe_mock.set.assert_called_once()
    args = pipe_mock.set.call_args.args
    assert args[0] == "augur:config:escalation_confidence"


def test_save_tuning_state_with_neither_is_noop():
    redis_mock = MagicMock()

    pm = PersistenceManager(redis_mock)
    pm.save_tuning_state(confidence=None, window_state=None)

    # No pipeline created at all
    redis_mock.pipeline.assert_not_called()
