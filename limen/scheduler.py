"""Must-fire scheduler for the advisor gate (spec §4).

A plain ``asyncio.Lock`` has no priority API, so this small ``MustFireScheduler``
wraps the advisor's ``reasoning_lock`` and gives must-fire deliveries the
ordering guarantees invariants C and D require:

- **Strict-priority FIFO tiers** — ``exempt > fail_open > anti_starvation``.
  Every must-fire (exempt, ``gate_error_fail_open``, ``anti_starvation_release``)
  acquires through the scheduler, so there is no priority inversion between a
  raw-lock waiter and a scheduler waiter.
- **Bounded overtaking** — a queued ``anti_starvation_release`` that has waited
  past ``max_release_wait_s`` *or* been overtaken by ``max_release_overtake``
  higher-priority grants is promoted *served-next*, ahead of further
  exempt/fail-open arrivals.  This makes the scheduler itself starvation-free so
  strict priority cannot violate invariant D.
- **Nonblocking ordinary path** — ordinary fires never enter the queues; they
  call :meth:`try_acquire_ordinary` (succeeds only when the lock is free *and*
  no must-fire is queued) + idempotent :meth:`release_ordinary`.  An ordinary
  fire can neither block nor jump a queued must-fire.
- **Per-``state_key`` release coalescing** — :meth:`track_release` adds the key
  to ``_release_in_flight`` before enqueueing and removes it in a ``finally`` so
  a delivery that fails or is skipped can never coalesce a channel out of future
  releases.
- **Safety contract (invariant C)** — :meth:`acquire`, :meth:`try_acquire_ordinary`
  and :meth:`release_ordinary` are non-raising (except ``asyncio.CancelledError``);
  ``release_ordinary`` is idempotent and the lock is never left held after the
  ``async with`` exits.  :meth:`emergency_deliver` is the last-resort path when a
  must-fire's acquire raises/cannot grant: it tries the lock directly, else
  delivers unserialized with an ERROR log — a must-fire is never dropped.

The scheduler does **no Redis I/O** and holds no gate state; ``now`` is an
injected monotonic clock so the bounded-overtaking deadline is deterministic
under test.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import Awaitable, Callable, Iterator
from dataclasses import dataclass, field

log = logging.getLogger(__name__)

# Strict tier order: lower rank wins.  An unknown priority degrades to the
# lowest tier (anti_starvation) rather than raising — non-raising contract (C).
_TIER_RANK: dict[str, int] = {"exempt": 0, "fail_open": 1, "anti_starvation": 2}
_LOWEST_RANK = 2


@dataclass
class _Waiter:
    """One queued must-fire awaiting its turn through the scheduler."""

    rank: int
    seq: int
    enqueued_at: float
    granted: asyncio.Event
    is_anti: bool
    overtaken: int = field(default=0)


def _default_now() -> float:
    return asyncio.get_event_loop().time()


class MustFireScheduler:
    """Priority-ordered, starvation-free serializer over an ``asyncio.Lock``."""

    def __init__(
        self,
        lock: asyncio.Lock,
        *,
        max_release_wait_s: float,
        max_release_overtake: int,
        now: Callable[[], float] = _default_now,
    ) -> None:
        self._lock = lock
        self._max_wait_s = max_release_wait_s
        self._max_overtake = max_release_overtake
        self._now = now
        self._waiters: list[_Waiter] = []
        self._seq = 0
        # The single waiter currently authorized to hold/take the lock.  Only one
        # waiter is ever granted at a time, so the underlying lock's own FIFO
        # never reorders our strict-priority selection.
        self._granted_seq: int | None = None
        # Slot held by an ordinary fire via try_acquire_ordinary (idempotent
        # release).  Tracked separately from a must-fire holding the lock.
        self._ordinary_held = False
        self._release_in_flight: set[str] = set()

    # ------------------------------------------------------------------ #
    # Must-fire path                                                      #
    # ------------------------------------------------------------------ #

    @contextlib.asynccontextmanager
    async def acquire(self, priority: str):
        """Acquire the lock for a must-fire at ``priority`` (async context manager).

        Strict-priority FIFO with bounded overtaking.  Non-raising except
        ``asyncio.CancelledError``; the lock is always released on exit.
        """
        rank = _TIER_RANK.get(priority, _LOWEST_RANK)
        self._seq += 1
        waiter = _Waiter(
            rank=rank,
            seq=self._seq,
            enqueued_at=self._now(),
            granted=asyncio.Event(),
            is_anti=(rank == _LOWEST_RANK),
        )
        self._waiters.append(waiter)
        self._dispatch()

        held = False
        try:
            await waiter.granted.wait()
            # We are the single authorized waiter; take the real lock.  Because
            # nobody else is authorized, this acquire cannot be reordered.
            await self._lock.acquire()
            held = True
            yield
        finally:
            # Drop from the queue if still present (cancelled before grant).
            with contextlib.suppress(ValueError):
                self._waiters.remove(waiter)
            if held and self._lock.locked():
                self._lock.release()
            if self._granted_seq == waiter.seq:
                self._granted_seq = None
            self._dispatch()

    def _dispatch(self) -> None:
        """Authorize the best eligible waiter if no waiter is currently granted.

        Selection: a queued anti_starvation past its wait/overtake bound is
        promoted served-next; otherwise the highest tier wins, FIFO within a
        tier.  Granting a higher-priority waiter increments the overtake counter
        of every still-queued anti_starvation (bounded-overtaking accounting).
        Only one waiter is granted at a time; the granted waiter then awaits the
        underlying lock and the next is chosen when its ``acquire`` context exits
        (which calls ``_dispatch`` again).
        """
        if self._granted_seq is not None:
            return
        if not self._waiters:
            return
        # Grant only when the lock is free, so the highest-priority waiter
        # *present at that moment* is selected (strict priority is not violated
        # by greedily granting the first arrival before a higher-priority one
        # appears).  The scheduler owns the lock: every release — ordinary slot,
        # prior must-fire, or emergency — calls ``_dispatch`` again, so a freeing
        # lock always re-triggers selection.  At most one waiter is ever granted,
        # so the lock's own FIFO can never reorder our selection.
        if self._lock.locked():
            return

        chosen = self._select()
        if chosen is None:
            return

        # Overtaking accounting: a higher-priority grant overtakes every queued
        # anti_starvation other than the grantee itself.
        if not chosen.is_anti:
            for w in self._waiters:
                if w.is_anti and w.seq != chosen.seq:
                    w.overtaken += 1

        self._granted_seq = chosen.seq
        chosen.granted.set()

    def _select(self) -> _Waiter | None:
        """Pick the next waiter honoring bounded overtaking, then strict tier."""
        promoted = [w for w in self._waiters if w.is_anti and self._past_bound(w)]
        if promoted:
            # Served-next: oldest promoted anti_starvation first.
            return min(promoted, key=lambda w: w.seq)
        # Strict priority, FIFO within tier.
        return min(self._waiters, key=lambda w: (w.rank, w.seq))

    def _past_bound(self, w: _Waiter) -> bool:
        waited = self._now() - w.enqueued_at
        return waited > self._max_wait_s or w.overtaken >= self._max_overtake

    # ------------------------------------------------------------------ #
    # Ordinary (nonblocking) path                                         #
    # ------------------------------------------------------------------ #

    def _try_lock_nonblocking(self) -> bool:
        """Take the underlying lock synchronously iff it is free; never block.

        ``asyncio.Lock`` has no public non-blocking acquire, so drive its
        ``acquire()`` coroutine directly: when the lock is free it completes on
        the first ``send`` (``StopIteration``).  If it ever *suspends* (the lock
        was contended after our ``locked()`` check), we close the coroutine and
        report failure rather than leak it — a strictly non-blocking attempt.
        """
        coro = self._lock.acquire()
        try:
            coro.send(None)
        except StopIteration:
            return True  # acquired synchronously
        # Suspended → contended → not acquired; clean up the coroutine.
        coro.close()
        return False

    def try_acquire_ordinary(self) -> bool:
        """Try to grab the slot for an ordinary fire WITHOUT awaiting.

        Succeeds only when the underlying lock is free AND no must-fire is
        queued/granted — so an ordinary fire never blocks and never jumps a
        must-fire.  Returns ``True`` on success (the slot is now held; release
        with :meth:`release_ordinary`), ``False`` otherwise.  Never raises.
        """
        try:
            if self._lock.locked() or self._waiters or self._granted_seq is not None:
                return False
            if not self._try_lock_nonblocking():
                return False
            self._ordinary_held = True
            return True
        except Exception:  # pragma: no cover - defensive (non-raising contract)
            log.exception("try_acquire_ordinary failed; treating as busy")
            return False

    def release_ordinary(self) -> None:
        """Release an ordinary-fire slot.  Idempotent; never raises.

        A no-op if this scheduler does not currently hold an ordinary slot (so a
        double release, or a release after a must-fire took the lock, can never
        free a lock it does not own → no deadlock, no foreign unlock).
        """
        try:
            if not self._ordinary_held:
                return
            self._ordinary_held = False
            if self._lock.locked():
                self._lock.release()
            self._dispatch()
        except Exception:  # pragma: no cover - defensive (non-raising contract)
            log.exception("release_ordinary failed")

    # ------------------------------------------------------------------ #
    # Anti-starvation release coalescing                                  #
    # ------------------------------------------------------------------ #

    def release_in_flight(self, state_key: str) -> bool:
        """True if an anti_starvation release for ``state_key`` is in flight."""
        return state_key in self._release_in_flight

    @contextlib.contextmanager
    def track_release(self, state_key: str) -> Iterator[None]:
        """Mark ``state_key`` in-flight for the duration; clear in ``finally``.

        A delivery that fails or is skipped still clears the token, so a channel
        can never be permanently coalesced out of future releases (spec §4).
        """
        self._release_in_flight.add(state_key)
        try:
            yield
        finally:
            self._release_in_flight.discard(state_key)

    # ------------------------------------------------------------------ #
    # Emergency last-resort delivery (invariant C)                        #
    # ------------------------------------------------------------------ #

    async def emergency_deliver(
        self, deliver_coro: Callable[[], Awaitable[None]]
    ) -> None:
        """Last-resort delivery when a must-fire's ``acquire`` raises/cannot grant.

        Tries to take ``reasoning_lock`` directly (bounded, non-blocking attempt);
        failing that, delivers **unserialized** (accepting possible overlap) with
        an ERROR log.  A must-fire is never dropped; never re-raises (except
        ``asyncio.CancelledError``) so a delivery bug degrades to an unserialized
        fire rather than silencing the advisor.
        """
        took_lock = False
        if not self._lock.locked():
            try:
                took_lock = self._try_lock_nonblocking()
            except Exception:  # pragma: no cover - defensive
                took_lock = False
        if not took_lock:
            log.error(
                "must-fire scheduler unavailable; delivering unserialized (lock held)"
            )
        try:
            await deliver_coro()
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("emergency_deliver delivery failed")
        finally:
            if took_lock and self._lock.locked():
                self._lock.release()
                self._dispatch()
