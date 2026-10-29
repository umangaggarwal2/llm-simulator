import statistics

import pytest

from llm_simulator.metrics.window import RateWindow, SampleWindow, describe


class Clock:
    def __init__(self, now=100.0):
        self.now = now

    def __call__(self):
        return self.now


# --- SampleWindow ---


def test_keeps_every_value_until_full():
    window = SampleWindow(3)
    for value in (1.0, 2.0):
        window.add(value)
    assert sorted(window.snapshot()) == [1.0, 2.0]
    assert (window.count, len(window)) == (2, 2)


def test_overwrites_the_oldest_once_full():
    window = SampleWindow(3)
    for value in range(1, 8):
        window.add(float(value))
    assert sorted(window.snapshot()) == [5.0, 6.0, 7.0]
    assert (window.count, len(window)) == (7, 3)


def test_snapshot_is_a_copy():
    window = SampleWindow(2)
    window.add(1.0)
    copy = window.snapshot()
    window.add(2.0)
    window.add(3.0)
    assert list(copy) == [1.0]


def test_clear():
    window = SampleWindow(2)
    for value in (1.0, 2.0, 3.0):
        window.add(value)
    window.clear()
    assert (window.count, len(window)) == (0, 0)
    window.add(4.0)
    assert list(window.snapshot()) == [4.0]


def test_capacity_must_be_positive():
    with pytest.raises(ValueError):
        SampleWindow(0)


# --- describe ---


def test_describe_matches_statistics_quantiles():
    values = [0.3, 0.01, 0.7, 0.2, 0.05, 1.5, 0.9, 0.4, 0.11, 0.6, 2.0]
    cuts = statistics.quantiles(values, n=100, method="inclusive")
    stats = describe(values, count=20)
    assert stats["count"] == 20
    assert stats["samples"] == len(values)
    assert stats["mean"] == pytest.approx(statistics.fmean(values))
    assert (stats["min"], stats["max"]) == (0.01, 2.0)
    for p in (50, 90, 95, 99):
        assert stats[f"p{p}"] == pytest.approx(cuts[p - 1])


def test_describe_single_value():
    stats = describe([0.25], count=1)
    assert {stats[key] for key in ("mean", "min", "p50", "p99", "max")} == {0.25}


def test_describe_empty_window():
    assert describe([], count=0) == {
        "count": 0,
        "samples": 0,
        "mean": None,
        "min": None,
        "p50": None,
        "p90": None,
        "p95": None,
        "p99": None,
        "max": None,
    }


# --- RateWindow ---


def test_rate_over_a_full_window():
    clock = Clock(100.0)
    window = RateWindow(10, clock)
    for second in range(30):
        clock.now = 100.0 + second
        window.add(5)
    clock.now = 130.0
    # Buckets 121..130 hold 5 each except 130, which is empty, over 10 s.
    rate, covered = window.rate()
    assert covered == pytest.approx(9.0)
    assert rate == pytest.approx(45 / 9.0)


def test_rate_just_after_start_covers_only_the_time_since():
    clock = Clock(100.0)
    window = RateWindow(60, clock)
    clock.now = 103.5
    window.add(70)
    assert window.rate() == (pytest.approx(20.0), pytest.approx(3.5))


def test_rate_is_measured_over_at_least_one_second():
    clock = Clock(100.0)
    window = RateWindow(60, clock)
    clock.now = 100.01
    window.add(1)
    assert window.rate() == (1.0, 1.0)


def test_stale_buckets_are_not_counted():
    clock = Clock(100.0)
    window = RateWindow(5, clock)
    window.add(100)
    clock.now = 110.0
    assert window.rate()[0] == 0.0
    # Second 110 reuses second 100's slot, which starts over rather than
    # adding to the stale total. The window is seconds 106 to now: 4 s.
    window.add(1)
    assert window.rate() == (pytest.approx(1 / 4), pytest.approx(4.0))


def test_rate_at_an_explicit_time():
    clock = Clock(100.0)
    window = RateWindow(60, clock)
    window.add(10)
    assert window.rate(now=102.0) == (5.0, 2.0)


def test_clear_restarts_the_window():
    clock = Clock(100.0)
    window = RateWindow(60, clock)
    window.add(50)
    clock.now = 120.0
    window.clear()
    clock.now = 122.0
    window.add(4)
    assert window.rate() == (2.0, 2.0)


def test_seconds_must_be_positive():
    with pytest.raises(ValueError):
        RateWindow(0)
