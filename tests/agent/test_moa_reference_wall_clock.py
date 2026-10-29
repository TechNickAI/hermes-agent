"""MoA ``reference_timeout`` must bound the WHOLE fan-out by wall clock (WO-MC28).

2026-10-19, cora highstakes: the grok advisor's relay held its stream open with no
first token for 497s while claude-think and gpt-chat had already answered; the turn
never reached the aggregator. ``reference_timeout`` was only forwarded as a
per-request transport timeout, which a held-open stream does not trip, so the
fan-out waited for the slowest advisor's full aux ceiling. With a wall-clock
deadline the aggregator answers from the advisors that returned, and the late one
becomes a ``[failed: …]`` note (degraded, not hung).
"""

from __future__ import annotations

import threading
import time

import pytest

import agent.moa_loop as moa


@pytest.fixture(autouse=True)
def _clean_abandoned_registry():
    with moa._abandoned_lock:
        moa._abandoned_inflight.clear()
    yield
    with moa._abandoned_lock:
        moa._abandoned_inflight.clear()


def _slot(model):
    return {"provider": "custom", "model": model}


@pytest.fixture
def stalled_leg(monkeypatch):
    release = threading.Event()

    def fake_run_reference(slot, ref_messages, **kwargs):
        if slot["model"] == "grok":
            release.wait(30)  # a held-open stream: never returns within the test
            return moa._slot_label(slot), "late answer", moa._RefAccounting(moa.CanonicalUsage())
        return moa._slot_label(slot), f"answer from {slot['model']}", moa._RefAccounting(moa.CanonicalUsage())

    monkeypatch.setattr(moa, "_run_reference", fake_run_reference)
    monkeypatch.setattr(moa, "_REFERENCE_POLL_INTERVAL_S", 0.2)
    yield
    release.set()


def test_reference_timeout_bounds_fanout_and_degrades(stalled_leg):
    slots = [_slot("claude-think"), _slot("grok"), _slot("gpt-chat")]
    started = time.monotonic()
    out = moa._run_references_parallel(slots, [{"role": "user", "content": "q"}], reference_timeout=1.0)
    elapsed = time.monotonic() - started
    assert elapsed < 5.0, f"fan-out hung {elapsed:.1f}s past reference_timeout=1.0"
    texts = {label: text for label, text, _ in out}
    assert texts["custom:claude-think"] == "answer from claude-think"
    assert texts["custom:gpt-chat"] == "answer from gpt-chat"
    assert texts["custom:grok"].startswith("[failed: reference timed out after 1s")
    # The aggregator gets the two survivors plus the loud degraded notice.
    agg_refs, notice, all_failed = moa._guidance_inputs(out, False, "loud")
    assert [label for label, _, _ in agg_refs] == ["custom:claude-think", "custom:gpt-chat"]
    assert notice == "[Reference models unavailable: custom:grok]" and all_failed is False


def test_without_reference_timeout_the_fanout_waits_for_the_slow_leg(stalled_leg, monkeypatch):
    """Counterfactual: no deadline -> the fan-out is still waiting after 3s."""
    slots = [_slot("claude-think"), _slot("grok")]
    box = {}
    worker = threading.Thread(
        target=lambda: box.setdefault("out", moa._run_references_parallel(slots, [{"role": "user", "content": "q"}])),
        daemon=True,
    )
    worker.start()
    worker.join(3.0)
    assert worker.is_alive(), "fan-out returned without a deadline; the stall is not reproduced"


def test_timed_out_advisor_cannot_restamp_the_activity_clock(monkeypatch):
    """Codex P1: an advisor still streaming after the deadline must not keep re-arming the
    turn-liveness watchdog through its progress hook once the fan-out has abandoned it."""
    touches = []
    agent = type("A", (), {"_touch_activity": lambda self, d: touches.append(d)})()
    fence = threading.Event()
    hook = moa._reference_liveness_hook(agent, "custom:grok", fence)
    hook()
    assert len(touches) == 1
    fence.set()
    hook(); hook()
    assert len(touches) == 1, "abandoned advisor chunks still stamped the activity clock"


def test_still_running_abandoned_advisor_is_not_redispatched(stalled_leg, monkeypatch):
    """Codex P1: a call that cannot be killed must not be stacked again by the next fan-out."""
    calls = []
    real = moa._run_reference

    def counting(slot, ref_messages, **kwargs):
        calls.append(slot["model"])
        return real(slot, ref_messages, **kwargs)

    monkeypatch.setattr(moa, "_run_reference", counting)
    slots = [_slot("claude-think"), _slot("grok")]
    msgs = [{"role": "user", "content": "q"}]
    moa._run_references_parallel(slots, msgs, reference_timeout=0.5)
    second = moa._run_references_parallel(slots, msgs, reference_timeout=0.5)
    assert calls.count("grok") == 1, calls
    texts = {label: text for label, text, _ in second}
    assert texts["custom:grok"].startswith("[failed: previous call to this advisor is still running")
    assert texts["custom:claude-think"] == "answer from claude-think"
