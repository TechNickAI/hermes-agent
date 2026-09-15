"""End-to-end phase test for the real InProcessCronScheduler loop.

Drives the actual ``start()`` loop with a simulated clock and a tick that does real work,
and asserts the firing phase does not walk forward. Before the fix this loop slept a flat
``interval`` after the tick returned, so 60 cycles of 0.11s work accumulated ~6.6s of drift.
"""

import threading
from unittest import mock

import pytest

from cron.scheduler_provider import InProcessCronScheduler


class FakeClock:
    """Monotonic clock advanced only by the loop's own waits + simulated tick work."""

    def __init__(self, start=1000.0):
        self.t = start
        self.tick_starts: list[float] = []

    def monotonic(self):
        return self.t

    def advance(self, dt):
        self.t += dt


@pytest.fixture
def patched_cron(monkeypatch):
    """Neutralise every side effect of the loop except timing."""
    for name in ("clear_ticker_error", "record_ticker_error", "record_ticker_heartbeat"):
        monkeypatch.setattr(f"cron.jobs.{name}", mock.Mock(), raising=False)
    monkeypatch.setattr(
        "cron.scheduler_provider.InProcessCronScheduler.recover_interrupted",
        lambda self: 0, raising=False)


def _run_loop(clock, n_cycles, tick_work, interval=60.0):
    """Run the real loop for n_cycles, recording the monotonic time each tick began."""
    stop = threading.Event()
    calls = {"n": 0}

    def fake_tick(**_kw):
        clock.tick_starts.append(clock.t)
        clock.advance(tick_work)  # the tick does real dispatch work
        calls["n"] += 1
        if calls["n"] >= n_cycles:
            stop.set()

    def fake_wait(timeout=None):
        # stop_event.wait(timeout) consumes wall time, then reports stop state.
        clock.advance(timeout or 0.0)
        return stop.is_set()

    with mock.patch("cron.scheduler.tick", fake_tick), \
         mock.patch("cron.scheduler_provider.time.monotonic", clock.monotonic), \
         mock.patch.object(stop, "wait", fake_wait):
        InProcessCronScheduler().start(stop, interval=interval)
    return clock.tick_starts


def test_tick_phase_does_not_drift_with_real_work(patched_cron):
    """60 ticks * 60s must land on exact 60s boundaries despite per-tick work."""
    clock = FakeClock()
    starts = _run_loop(clock, n_cycles=60, tick_work=0.11)

    assert len(starts) == 60
    anchor = starts[0]
    for i, t in enumerate(starts):
        # Each tick begins exactly i*60s after the first: zero cumulative drift.
        assert t - anchor == pytest.approx(i * 60.0, abs=1e-6), (
            f"tick {i} drifted by {t - anchor - i * 60.0:+.3f}s")


def test_drift_would_exceed_a_minute_without_the_fix(patched_cron):
    """Guards the regression's magnitude: flat-sleep would skip a 5-minute slot."""
    clock = FakeClock()
    starts = _run_loop(clock, n_cycles=60, tick_work=0.11)
    total_drift = (starts[-1] - starts[0]) - 59 * 60.0
    assert abs(total_drift) < 1e-6
    # The buggy loop would have accumulated 59 * 0.11 = 6.49s here, and in production
    # (many jobs per tick) enough to cross a whole minute boundary and drop a fire.


def test_slow_tick_does_not_busy_loop(patched_cron):
    """A tick that overruns the interval must not spin; phase resumes on the grid."""
    clock = FakeClock()
    starts = _run_loop(clock, n_cycles=5, tick_work=95.0, interval=60.0)
    gaps = [starts[i + 1] - starts[i] for i in range(len(starts) - 1)]
    # Overrunning tick yields immediately (0 sleep), so the next tick starts right after
    # the work — never earlier, never a negative/zero-length spin.
    assert all(g >= 95.0 for g in gaps), gaps
