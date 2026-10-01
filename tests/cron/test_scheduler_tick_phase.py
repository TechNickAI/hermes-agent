"""Tick cadence must be fixed-RATE, not fixed-DELAY.

Regression for observed production drift: the in-process ticker slept a flat
``interval`` AFTER each tick returned, so the real period was
``interval + tick_work_seconds``. With ~0.11s of dispatch work per tick the
period measured 60.114s, and a 5-minute job's fire time walked forward ~0.6s
per fire until it crossed a minute boundary and skipped a slot entirely
(sawtooth: lag climbed 0.1s -> 59.2s, then wrapped).

The fix anchors each sleep to a monotonic schedule so cumulative drift is zero.
"""

import pytest

from cron.scheduler_provider import _backoff_wait_seconds, _next_wait_seconds


class TestFixedRateWait:
    def test_wait_subtracts_elapsed_tick_work(self):
        """Sleep the REMAINDER of the interval, not the whole interval."""
        # Tick started at t=100.0, work took 0.11s, interval is 60s.
        assert _next_wait_seconds(
            interval=60.0, tick_started_monotonic=100.0, now_monotonic=100.11
        ) == pytest.approx(59.89)

    def test_no_cumulative_drift_over_many_ticks(self):
        """N ticks of real work must not push the phase forward at all."""
        interval, work = 60.0, 0.11
        now = 1000.0
        anchor = now
        for _ in range(100):
            tick_started = now
            now += work  # tick does work
            now += _next_wait_seconds(interval, tick_started, now)  # then sleeps
        # 100 ticks * 60s == 6000s exactly, no accumulated 0.11s-per-tick creep.
        assert now - anchor == pytest.approx(6000.0, abs=1e-6)

    def test_overrunning_tick_does_not_sleep_negative(self):
        """A tick slower than the interval must yield immediately, never a negative sleep."""
        assert _next_wait_seconds(
            interval=60.0, tick_started_monotonic=0.0, now_monotonic=95.0
        ) == 0.0

    def test_overrun_does_not_burn_cpu_catching_up(self):
        """After a long overrun the next wait is bounded by the interval (no catch-up storm)."""
        w = _next_wait_seconds(interval=60.0, tick_started_monotonic=0.0, now_monotonic=610.0)
        assert 0.0 <= w <= 60.0

    def test_emfile_backoff_still_wins_over_fixed_rate(self):
        """Fd-exhaustion backoff must not be undone by drift correction."""
        # Backoff path is separate and unchanged.
        assert _backoff_wait_seconds(60.0, 3) == 240.0
        assert _backoff_wait_seconds(60.0, 0) == 60.0
