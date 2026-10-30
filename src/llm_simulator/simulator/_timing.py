"""Deadline-based sleeping shared by every timed engine."""

from typing import Awaitable, Callable

# Injected into engines so tests can drive time with a fake clock.
Clock = Callable[[], float]
Sleep = Callable[[float], Awaitable[None]]


async def sleep_until(deadline: float, clock: Clock, sleep: Sleep) -> float:
    """Sleep until an absolute clock time, and return how late the wake-up was.

    A deadline that has already passed (a slow consumer, or request overhead
    longer than the simulated delay) returns at once rather than shifting
    later deadlines, so lateness never accumulates.

    A sleep can also end early: uvloop's timers have millisecond resolution
    and read a cached loop clock, and wake up to about a millisecond before
    the deadline. Sleeping again for whatever remains means a token is never
    released before its scheduled time; the extra sleeps are short and still
    yield to the event loop.
    """
    while (remaining := deadline - clock()) > 0:
        await sleep(remaining)
    return clock() - deadline
