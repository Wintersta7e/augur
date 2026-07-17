"""The integration suite must never point at the live cell."""

from __future__ import annotations

import pytest

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

    @pytest.mark.parametrize(
        "nats_url",
        [
            "nats://localhost",
            "nats://127.0.0.1",
            "nats://127.0.0.1:4222",
            "nats://127.0.0.1:4222/",
            "nats://nats:4222",
            "nats://172.28.1.5:4222",
        ],
    )
    def test_every_live_bus_spelling_is_refused(self, nats_url: str) -> None:
        cfg = AugurConfig(
            redis_url="redis://127.0.0.1:6379/1",
            nats_url=nats_url,
        )
        reason = check_test_cell(cfg)
        assert reason is not None, f"{nats_url} was accepted as a test cell"

    def test_malformed_nats_port_is_refused_not_raised(self) -> None:
        cfg = AugurConfig(
            redis_url="redis://127.0.0.1:6379/1",
            nats_url="nats://127.0.0.1:notaport",
        )
        reason = check_test_cell(cfg)
        assert reason is not None

    def test_test_cell_port_is_still_accepted(self) -> None:
        cfg = AugurConfig(
            redis_url="redis://127.0.0.1:6379/1",
            nats_url="nats://127.0.0.1:4223",
        )
        assert check_test_cell(cfg) is None
