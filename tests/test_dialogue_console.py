import asyncio
from unittest.mock import Mock
from imperator.dialogue import console


def test_console_loop_quits_and_echoes(monkeypatch):
    lines = iter(["hello", "quit"])
    out = []

    async def fake_turn(session_id, text, **kw):
        from imperator.dialogue.engine import DialogueTurn

        return DialogueTurn(reply=f"heard:{text}")

    mock_pm = Mock()
    monkeypatch.setattr(console, "_engine_turn", fake_turn)
    monkeypatch.setattr(console, "_connect", lambda cfg: (mock_pm, None, None))
    asyncio.run(
        console.run_console(input_fn=lambda _: next(lines), output_fn=out.append)
    )
    assert any("heard:hello" in o for o in out)


def test_console_reports_turn_error_and_keeps_looping(monkeypatch):
    lines = iter(["boom", "hello", "quit"])
    out = []

    async def fake_turn(session_id, text, **kw):
        from imperator.dialogue.engine import DialogueTurn

        if text == "boom":
            raise ConnectionError("redis down")
        return DialogueTurn(reply=f"heard:{text}")

    mock_pm = Mock()
    monkeypatch.setattr(console, "_engine_turn", fake_turn)
    monkeypatch.setattr(console, "_connect", lambda cfg: (mock_pm, None, None))
    asyncio.run(
        console.run_console(input_fn=lambda _: next(lines), output_fn=out.append)
    )
    assert any("[error] turn failed: redis down" in o for o in out)
    assert any("heard:hello" in o for o in out)  # loop survived the failure


def test_console_eof_exits_gracefully(monkeypatch):
    def raising_input(_):
        raise EOFError

    out = []
    mock_pm = Mock()
    monkeypatch.setattr(console, "_connect", lambda cfg: (mock_pm, None, None))
    # Must not raise: EOF from input_fn is a graceful exit.
    asyncio.run(console.run_console(input_fn=raising_input, output_fn=out.append))
    assert len(out) >= 1  # greeting printed; exit was clean
