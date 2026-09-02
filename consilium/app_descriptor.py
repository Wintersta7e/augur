"""Autonomous app-descriptor logic shared by the advisor.

OS FileDescription (carried in PerceptionEvent.context["app_identity"]) is the
authoritative source; a dedicated small Ollama model classifies the rest on a
single-flight lane that runs parallel to the 32B advice call.
"""

from __future__ import annotations

import asyncio
import logging
import re

import httpx
import redis

from tabula.config import AugurConfig
from tabula.contracts import is_sentinel_entity
from tabula.persistence import PersistenceManager  # noqa: F401

log = logging.getLogger("augur.app_descriptor")

# Must stay in sync with the activity-domain subset of tabula.contracts.Domain.
ACTIVITY_DOMAINS = frozenset({"activity_focus", "activity_intensity"})
_SAFE_ENTITY_RE = re.compile(r"[\w.\- ]{1,64}", re.ASCII)
_MAX_DESCRIPTOR_LEN = 60


def is_sentinel_app(entity: str) -> bool:
    """True for daemon sentinels like <unknown>/<no_foreground>/<denied>/<gone>.

    Thin alias; the predicate lives in ``tabula.contracts`` because Vigil and
    Praesagium need the same answer and must not import a sibling faculty.
    """
    return is_sentinel_entity(entity)


def descriptor_suffix(ctx: dict) -> str:
    """Return ' (descriptor)' if ctx carries one, else '' (clean omit)."""
    descriptor = ctx.get("app_descriptor")
    return f" ({descriptor})" if descriptor else ""


def resolve_app_descriptor(
    pm: PersistenceManager, entity: str, ctx: dict, *, learn_ctx=None
) -> tuple[str | None, bool]:
    """Resolve an activity entity's descriptor.

    Returns (descriptor_or_None, needs_classification). OS identity wins and is
    cached authoritatively; otherwise a cache hit is returned; otherwise it is a
    miss that should be classified. Sentinels and Redis errors are quiet no-ops.
    """
    if not entity or is_sentinel_app(entity):
        return None, False

    identity = ctx.get("app_identity")
    if identity:
        try:
            pm.save_app_descriptor(entity, identity, overwrite=True, ctx=learn_ctx)
        except redis.RedisError as exc:
            log.warning("app_descriptor: OS save failed for %s: %s", entity, exc)
        return identity, False

    try:
        cached = pm.load_app_descriptor(entity)
    except redis.RedisError as exc:
        log.warning("app_descriptor: load failed for %s: %s", entity, exc)
        return None, False
    if cached:
        return cached, False
    # Don't feed pathological process names to the classifier LLM.
    if not _SAFE_ENTITY_RE.fullmatch(entity):
        return None, False
    return None, True


async def classify_app(
    entity: str, client: httpx.AsyncClient, config: AugurConfig
) -> str | None:
    """Ask the small classifier model what kind of app `entity` is.

    Mirrors query_ollama's /api/generate shape. Returns a short descriptor or
    None (empty / 'unknown' / unusable).
    """
    prompt = (
        "In four words or fewer, what kind of application is the process named "
        f"'{entity}'? Reply with only the short description, or the single word "
        f"'unknown' if you cannot tell."
    )
    payload = {
        "model": config.ollama_classifier_model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.0, "num_predict": 16},
    }
    resp = await client.post(
        f"{config.ollama_url}/api/generate",
        json=payload,
        timeout=config.ollama_classifier_timeout,
    )
    resp.raise_for_status()
    text = resp.json().get("response", "").strip()
    if not text or text.lower() == "unknown" or is_sentinel_app(text):
        return None
    return text[:_MAX_DESCRIPTOR_LEN]


def classifier_model_available(model: str, models: list[str]) -> bool:
    """True if `model` matches any available Ollama model tag."""
    return any(model in m for m in models)


class ClassifierLane:
    """Single-flight, per-entity-coalescing lane for LLM app classification.

    Decoupled from the advisor's advice lock so it can run parallel to the 32B
    advice call. One classification at a time (own lock); duplicate entities are
    coalesced while in flight.
    """

    def __init__(
        self, pm: PersistenceManager, client: httpx.AsyncClient, config: AugurConfig
    ) -> None:
        self._pm = pm
        self._client = client
        self._config = config
        self._lock = asyncio.Lock()
        self._pending: set[str] = set()
        self._tasks: set[asyncio.Task] = set()
        self._closing = False
        # Disabled if turned off, or if it would just contend on the advice model.
        self.enabled = config.ollama_classifier_enabled and (
            config.ollama_classifier_model != config.ollama_model
        )

    def enqueue(self, entity: str, *, learn_ctx=None) -> None:
        if (
            not self.enabled
            or self._closing
            or is_sentinel_app(entity)
            or entity in self._pending
        ):
            return
        self._pending.add(entity)
        # Capture provenance at ENQUEUE time: the session may roll over before the
        # background classification writes (spec §4.3c).
        task = asyncio.create_task(self._run(entity, learn_ctx))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _run(self, entity: str, learn_ctx=None) -> None:
        try:
            async with self._lock:  # single-flight on the classifier model
                descriptor = await classify_app(entity, self._client, self._config)
            if descriptor:
                self._pm.save_app_descriptor(
                    entity, descriptor, overwrite=False, ctx=learn_ctx
                )
        except (httpx.HTTPError, redis.RedisError, ValueError) as exc:
            log.warning("app_descriptor: classification failed for %s: %s", entity, exc)
        finally:
            self._pending.discard(entity)

    async def shutdown(self, timeout: float = 5.0) -> None:
        """Flush in-flight classifications, then cancel any exceeding `timeout`."""
        self._closing = True
        if not self._tasks:
            return
        _done, still_running = await asyncio.wait(list(self._tasks), timeout=timeout)
        for task in still_running:
            task.cancel()
        await asyncio.gather(*still_running, return_exceptions=True)
