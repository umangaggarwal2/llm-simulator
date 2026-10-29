"""Bounded stores of recent measurements, for exact percentiles and rates.

Recording is constant time and memory is fixed by the capacity, however long
the server runs. The expensive part, sorting for percentiles, happens only
when a summary is read.
"""

import time
from array import array
from typing import Callable, Dict, List, Optional, Sequence, Tuple

PERCENTILES = (50, 90, 95, 99)


class SampleWindow:
    """The most recent `capacity` values, plus how many were ever added.

    Values live in an array of doubles: 8 bytes each, and copying the window
    for a summary is a single memory copy. The array grows until it is full,
    then the oldest value is overwritten. Order is not kept, since
    percentiles do not depend on it.
    """

    def __init__(self, capacity: int):
        if capacity < 1:
            raise ValueError("capacity must be at least 1")
        self._capacity = capacity
        self._values = array("d")
        self._next = 0
        # Values added since creation or the last clear(), including those
        # since overwritten.
        self.count = 0

    def add(self, value: float) -> None:
        if len(self._values) < self._capacity:
            self._values.append(value)
        else:
            self._values[self._next] = value
            self._next = (self._next + 1) % self._capacity
        self.count += 1

    def clear(self) -> None:
        self._values = array("d")
        self._next = 0
        self.count = 0

    def __len__(self) -> int:
        return len(self._values)

    def snapshot(self) -> array:
        """A copy of the values kept, unaffected by later adds."""
        return self._values[:]


class RateWindow:
    """Totals in one-second buckets over the last `seconds` seconds.

    Each bucket remembers which second it holds, so a stale bucket is simply
    zeroed when its slot is reused: no timer, and memory is fixed.
    """

    def __init__(self, seconds: int, clock: Callable[[], float] = time.monotonic):
        if seconds < 1:
            raise ValueError("seconds must be at least 1")
        self._seconds = seconds
        self._clock = clock
        self.clear()

    def clear(self) -> None:
        self._since = self._clock()
        self._epochs: List[int] = [-1] * self._seconds
        self._totals: List[float] = [0.0] * self._seconds

    def add(self, amount: float = 1.0) -> None:
        second = int(self._clock())
        slot = second % self._seconds
        if self._epochs[slot] == second:
            self._totals[slot] += amount
        else:
            self._epochs[slot] = second
            self._totals[slot] = amount

    def rate(self, now: Optional[float] = None) -> Tuple[float, float]:
        """The per-second rate and the seconds it was measured over.

        The window covers the last `seconds` buckets, the current partial
        one included, or less just after creation or a clear. It is at least
        one second, so a few events right after a reset do not read as a
        burst.
        """
        now = self._clock() if now is None else now
        current = int(now)
        oldest = current - self._seconds + 1
        total = sum(
            amount
            for epoch, amount in zip(self._epochs, self._totals)
            if oldest <= epoch <= current
        )
        covered = max(1.0, now - max(self._since, oldest))
        return total / covered, covered


def percentile(ordered: Sequence[float], q: float) -> float:
    """The q-th percentile of sorted values, interpolating between ranks.

    The same method as numpy's default and
    statistics.quantiles(method="inclusive").
    """
    position = (len(ordered) - 1) * q / 100
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def describe(values: Sequence[float], count: int) -> Dict[str, Optional[float]]:
    """Count, samples kept, mean, min, percentiles and max of a window's values.

    count is how many values were recorded, samples how many the window
    still holds; the statistics cover the samples, and are None without any.
    """
    stats: Dict[str, Optional[float]] = {"count": count, "samples": len(values)}
    if not values:
        stats.update(mean=None, min=None)
        stats.update({f"p{p}": None for p in PERCENTILES})
        stats["max"] = None
        return stats
    ordered = sorted(values)
    stats["mean"] = sum(ordered) / len(ordered)
    stats["min"] = ordered[0]
    stats.update({f"p{p}": percentile(ordered, p) for p in PERCENTILES})
    stats["max"] = ordered[-1]
    return stats
