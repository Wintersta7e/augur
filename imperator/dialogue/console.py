"""Interactive console — the user's conversation home with Augur."""

from __future__ import annotations

import asyncio
import uuid

import httpx
import nats

from tabula.config import AugurConfig
from tabula.connections import connect_redis
from tabula.persistence import PersistenceManager
from imperator.dialogue.engine import handle_turn

_QUIT = {"quit", "exit", ":q"}


async def _connect(cfg):
    pm = PersistenceManager(connect_redis(cfg))
    nc = await nats.connect(cfg.nats_url, connect_timeout=cfg.nats_connect_timeout)
    http = httpx.AsyncClient()
    return pm, nc, http


async def _engine_turn(session_id, text, **kw):  # seam for tests
    return await handle_turn(session_id, text, **kw)


async def run_console(*, input_fn=input, output_fn=print) -> None:
    cfg = AugurConfig.from_env()
    pm, nc, http = (
        _connect(cfg)
        if not asyncio.iscoroutinefunction(_connect)
        else await _connect(cfg)
    )
    session_id = f"dialogue-{uuid.uuid4().hex[:8]}"
    output_fn("Augur is listening. Type 'quit' to leave.")
    try:
        while True:
            try:
                line = input_fn("you> ").strip()
            except (EOFError, KeyboardInterrupt):
                output_fn("")  # newline, then leave quietly
                break
            if line.lower() in _QUIT:
                break
            if not line:
                continue
            try:
                turn = await _engine_turn(
                    session_id, line, pm=pm, nc=nc, http_client=http, cfg=cfg
                )
            except Exception as exc:  # surface reports infra failures truthfully
                output_fn(f"[error] turn failed: {exc}")
                continue
            output_fn(f"augur> {turn.reply}")
    finally:
        if http is not None:
            await http.aclose()
        if nc is not None:
            await nc.drain()
