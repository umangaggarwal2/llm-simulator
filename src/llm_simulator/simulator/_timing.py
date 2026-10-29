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
    """
    delay = deadline - clock()
    if delay > 0:
        await sleep(delay)
    return max(0.0, clock() - deadline)
