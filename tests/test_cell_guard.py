"""The integration suite must never point at the live cell."""

from __future__ import annotations

from tabula.config import AugurConfig
from tests.integration.cell_guard import check_test_cell


class TestCheckTestCell:
    def test_db_zero_is_refused(self) -> None:
        cfg = AugurConfig(
            redis_url="redis://127.0.0.1:6379/0",
            nats_url="nats://127.0.0.1:4223",
        )
        reason = check_test_cell(cfg)
        assert reason is not None
        assert "db 0" in reason

    def test_live_nats_is_refused(self) -> None:
        cfg = AugurConfig(
            redis_url="redis://127.0.0.1:6379/1",
            nats_url="nats://127.0.0.1:4222",
        )
        reason = check_test_cell(cfg)
        assert reason is not None
        assert "4222" in reason

    def test_proper_test_cell_passes(self) -> None:
        cfg = AugurConfig(
            redis_url="redis://127.0.0.1:6379/1",
            nats_url="nats://127.0.0.1:4223",
        )
        assert check_test_cell(cfg) is None

    def test_both_wrong_reports_redis_first(self) -> None:
        cfg = AugurConfig(
            redis_url="redis://127.0.0.1:6379/0",
            nats_url="nats://127.0.0.1:4222",
        )
        reason = check_test_cell(cfg)
        assert reason is not None and "db 0" in reason
