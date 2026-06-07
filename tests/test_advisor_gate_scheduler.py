"""Async unit tests for ``MustFireScheduler`` + ``emergency_deliver`` (spec §4).

The scheduler wraps the advisor's ``reasoning_lock`` (a plain ``asyncio.Lock``,
which has no priority API) and serializes must-fire deliveries through three
strict-priority FIFO tiers — ``exempt > fail_open > anti_starvation`` — with
*bounded overtaking* so a sustained higher-priority stream can never starve a
queued ``anti_starvation_release`` (invariant D).  Ordinary fires never enter
the queues: they use the nonblocking ``try_acquire_ordinary()`` (never await,
never jump a must-fire) + idempotent ``release_ordinary()``.

These tests use ``asyncio_mode = auto`` (pytest.ini), so coroutine tests need
no decorator.  Time is driven by an injected monotonic clock so the
bounded-overtaking deadline is deterministic.
"""

from __future__ import annotations

import asyncio

import pytest

from reasoning.advisor_gate_scheduler import MustFireScheduler


class FakeClock:
    """Deterministic monotonic clock injected as the scheduler's ``now``."""

    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


def _scheduler(**kw):
    lock = asyncio.Lock()
    clock = FakeClock()
    sched = MustFireScheduler(
        lock,
        max_release_wait_s=kw.get("max_release_wait_s", 30),
        max_release_overtake=kw.get("max_release_overtake", 5),
        now=clock,
    )
    return sched, lock, clock


# --------------------------------------------------------------------------- #
# Priority ordering: exempt acquires ahead of anti_starvation                  #
# --------------------------------------------------------------------------- #


async def _hold_slot(sched):
    """Hold the scheduler slot (a must-fire) and return a release callable.

    Returns ``(task, release_event)``: set ``release_event`` to let the holder
    exit its ``acquire`` block, which calls ``_dispatch`` so the queue drains.
    Holding via the scheduler (not a raw lock) mirrors production: the scheduler
    owns the lock and every release re-triggers selection.
    """
    release = asyncio.Event()
    started = asyncio.Event()

    async def holder() -> None:
        async with sched.acquire("exempt"):
            started.set()
            await release.wait()

    task = asyncio.create_task(holder())
    await started.wait()  # the holder now owns the lock
    return task, release


async def test_exempt_acquires_ahead_of_anti_starvation():
    """A queued exempt waiter is served before a queued anti_starvation waiter."""
    sched, lock, _clock = _scheduler()
    order: list[str] = []

    holder, release = await _hold_slot(sched)

    async def waiter(priority: str) -> None:
        async with sched.acquire(priority):
            order.append(priority)
            await asyncio.sleep(0)

    # Enqueue anti_starvation FIRST, then exempt — strict priority must reorder.
    anti = asyncio.create_task(waiter("anti_starvation"))
    await asyncio.sleep(0)
    exempt = asyncio.create_task(waiter("exempt"))
    await asyncio.sleep(0)

    # Let the holder release; both queued must-fires run, exempt first.
    release.set()
    await asyncio.gather(holder, anti, exempt)

    assert order == ["exempt", "anti_starvation"]
    assert not lock.locked()


async def test_fail_open_between_exempt_and_anti_starvation():
    """Strict tier order exempt > fail_open > anti_starvation."""
    sched, lock, _clock = _scheduler()
    order: list[str] = []

    holder, release = await _hold_slot(sched)

    async def waiter(priority: str) -> None:
        async with sched.acquire(priority):
            order.append(priority)
            await asyncio.sleep(0)

    tasks = []
    for p in ("anti_starvation", "fail_open", "exempt"):
        tasks.append(asyncio.create_task(waiter(p)))
        await asyncio.sleep(0)

    release.set()
    await asyncio.gather(holder, *tasks)
    assert order == ["exempt", "fail_open", "anti_starvation"]


# --------------------------------------------------------------------------- #
# Nonblocking ordinary path                                                    #
# --------------------------------------------------------------------------- #


async def test_try_acquire_ordinary_false_when_lock_held():
    """Ordinary fire fails (never awaits) when the underlying lock is held."""
    sched, lock, _clock = _scheduler()
    await lock.acquire()
    assert sched.try_acquire_ordinary() is False
    lock.release()


async def test_try_acquire_ordinary_false_when_must_fire_queued():
    """Ordinary fire fails when a must-fire is queued even if it could win the lock."""
    sched, lock, _clock = _scheduler()
    # Hold the slot via the scheduler so a must-fire queues behind it.
    holder, release = await _hold_slot(sched)

    served = asyncio.Event()

    async def waiter() -> None:
        async with sched.acquire("exempt"):
            served.set()
            await asyncio.sleep(0)

    task = asyncio.create_task(waiter())
    await asyncio.sleep(0)  # let it enqueue behind the holder

    # The contract: ordinary fails if the lock is held OR a must-fire is queued.
    # Here the holder holds the lock AND a must-fire is queued.
    assert sched.try_acquire_ordinary() is False

    # Release the holder; the must-fire is now granted/in-flight — ordinary must
    # still not jump it.
    release.set()
    await served.wait()
    assert sched.try_acquire_ordinary() is False
    await asyncio.gather(holder, task)


async def test_try_acquire_ordinary_succeeds_when_free_and_empty():
    """Ordinary succeeds when the lock is free and no must-fire is queued."""
    sched, lock, _clock = _scheduler()
    assert sched.try_acquire_ordinary() is True
    assert lock.locked() is True  # the slot is now held
    sched.release_ordinary()
    assert lock.locked() is False


async def test_try_acquire_ordinary_does_not_await():
    """try_acquire_ordinary returns synchronously (no await point)."""
    sched, _lock, _clock = _scheduler()
    coro_or_bool = sched.try_acquire_ordinary()
    assert isinstance(coro_or_bool, bool)
    sched.release_ordinary()


# --------------------------------------------------------------------------- #
# release_ordinary idempotence                                                 #
# --------------------------------------------------------------------------- #


async def test_release_ordinary_idempotent():
    """A double release is a no-op (idempotent, never raises, no deadlock)."""
    sched, lock, _clock = _scheduler()
    assert sched.try_acquire_ordinary() is True
    sched.release_ordinary()
    sched.release_ordinary()  # idempotent
    assert lock.locked() is False
    # Lock is reusable afterward.
    assert sched.try_acquire_ordinary() is True
    sched.release_ordinary()


async def test_release_ordinary_without_acquire_is_noop():
    """Releasing when nothing is held does not raise or unlock a foreign holder."""
    sched, lock, _clock = _scheduler()
    await lock.acquire()  # held by someone else (a must-fire / external)
    sched.release_ordinary()  # must NOT release a lock it doesn't own
    assert lock.locked() is True
    lock.release()


# --------------------------------------------------------------------------- #
# Starvation-free under continuous higher-priority arrivals (count bound)      #
# --------------------------------------------------------------------------- #


async def test_starvation_free_overtake_bound():
    """A queued anti_starvation is promoted served-next after max_release_overtake
    higher-priority grants, even under a continuous exempt stream (D)."""
    sched, lock, clock = _scheduler(max_release_overtake=3, max_release_wait_s=10_000)
    order: list[str] = []

    holder, release = await _hold_slot(sched)

    async def waiter(priority: str, tag: str) -> None:
        async with sched.acquire(priority):
            order.append(tag)
            await asyncio.sleep(0)

    # Queue the anti_starvation release first.
    anti = asyncio.create_task(waiter("anti_starvation", "ANTI"))
    await asyncio.sleep(0)

    # Now a continuous stream of exempt arrivals.  Release the holder and keep
    # feeding exempts; the anti_starvation MUST be served within the overtake
    # bound rather than indefinitely deferred.
    exempts = []
    for i in range(10):
        exempts.append(asyncio.create_task(waiter("exempt", f"E{i}")))
        await asyncio.sleep(0)

    release.set()
    await asyncio.gather(holder, anti, *exempts)

    anti_idx = order.index("ANTI")
    # ANTI must be served by the time it has been overtaken max_release_overtake
    # times — i.e. within the first (overtake + 1) grants.
    assert anti_idx <= 3, f"anti starved: served at index {anti_idx}: {order}"


async def test_starvation_free_wait_bound():
    """A queued anti_starvation past max_release_wait_s is promoted served-next."""
    sched, lock, clock = _scheduler(max_release_overtake=10_000, max_release_wait_s=30)
    order: list[str] = []

    holder, release = await _hold_slot(sched)

    async def waiter(priority: str, tag: str) -> None:
        async with sched.acquire(priority):
            order.append(tag)
            await asyncio.sleep(0)

    anti = asyncio.create_task(waiter("anti_starvation", "ANTI"))
    await asyncio.sleep(0)

    # The anti_starvation has now waited; advance the clock past the deadline.
    clock.advance(31)

    exempts = []
    for i in range(5):
        exempts.append(asyncio.create_task(waiter("exempt", f"E{i}")))
        await asyncio.sleep(0)

    release.set()
    await asyncio.gather(holder, anti, *exempts)

    # Past the wait bound, ANTI is served-next (ahead of further exempts).
    assert order[0] == "ANTI", order


# --------------------------------------------------------------------------- #
# _release_in_flight token cleared in finally                                  #
# --------------------------------------------------------------------------- #


async def test_release_in_flight_added_and_cleared():
    """A state_key is in _release_in_flight for the in-flight window and cleared
    in finally even when the body raises."""
    sched, _lock, _clock = _scheduler()
    key = "single:typing:user"

    assert sched.release_in_flight(key) is False
    with pytest.raises(ValueError):
        with sched.track_release(key):
            assert sched.release_in_flight(key) is True
            raise ValueError("delivery blew up")
    # Cleared in finally → the channel can be released again later.
    assert sched.release_in_flight(key) is False


async def test_release_in_flight_cleared_on_normal_exit():
    """Token cleared on a normal (non-raising) exit too."""
    sched, _lock, _clock = _scheduler()
    key = "pair:chess:e4"
    with sched.track_release(key):
        assert sched.release_in_flight(key) is True
    assert sched.release_in_flight(key) is False


# --------------------------------------------------------------------------- #
# Non-raising contract                                                          #
# --------------------------------------------------------------------------- #


async def test_methods_non_raising_on_bad_priority():
    """acquire never raises for an unknown priority (degrades, treats as lowest)."""
    sched, lock, _clock = _scheduler()
    async with sched.acquire("nonsense"):
        assert lock.locked() is True
    assert not lock.locked()


# --------------------------------------------------------------------------- #
# emergency_deliver                                                             #
# --------------------------------------------------------------------------- #


async def test_emergency_deliver_uses_lock_when_free():
    """When the lock is free, emergency_deliver serializes through it."""
    sched, lock, _clock = _scheduler()
    delivered = []

    async def deliver() -> None:
        assert lock.locked() is True  # serialized
        delivered.append("ok")

    await sched.emergency_deliver(deliver)
    assert delivered == ["ok"]
    assert not lock.locked()


async def test_emergency_deliver_unserialized_when_lock_held(caplog):
    """When the lock cannot be taken, the must-fire still delivers (unserialized)
    with an ERROR log — a must-fire is never dropped."""
    import logging

    sched, lock, _clock = _scheduler()
    await lock.acquire()  # someone else holds it and won't let go
    delivered = []

    async def deliver() -> None:
        delivered.append("ok")

    with caplog.at_level(logging.ERROR):
        await sched.emergency_deliver(deliver)

    assert delivered == ["ok"]  # delivered anyway
    assert lock.locked() is True  # foreign holder untouched
    assert any("unserialized" in r.message.lower() for r in caplog.records)
    lock.release()


async def test_emergency_deliver_non_raising(caplog):
    """A failing deliver coro is logged, never re-raised (must-fire path)."""
    import logging

    sched, lock, _clock = _scheduler()

    async def deliver() -> None:
        raise RuntimeError("ollama down")

    with caplog.at_level(logging.ERROR):
        await sched.emergency_deliver(deliver)  # must not raise

    assert not lock.locked()  # lock released even though deliver raised


# --------------------------------------------------------------------------- #
# No-deadlock: acquire body raising still releases the lock                    #
# --------------------------------------------------------------------------- #


async def test_acquire_releases_lock_on_body_exception():
    """The async-with releases the lock even if the body raises."""
    sched, lock, _clock = _scheduler()
    with pytest.raises(RuntimeError):
        async with sched.acquire("exempt"):
            assert lock.locked() is True
            raise RuntimeError("boom")
    assert not lock.locked()


async def test_acquire_releases_lock_on_cancel():
    """Cancelling a waiter never leaves the lock held or the queue wedged."""
    sched, lock, _clock = _scheduler()
    await lock.acquire()

    async def waiter() -> None:
        async with sched.acquire("exempt"):
            await asyncio.sleep(10)

    task = asyncio.create_task(waiter())
    await asyncio.sleep(0)
    task.cancel()
    lock.release()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not lock.locked()
    # The queue is not wedged: a fresh must-fire can still acquire.
    async with sched.acquire("exempt"):
        assert lock.locked() is True
    assert not lock.locked()
