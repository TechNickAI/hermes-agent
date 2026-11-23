"""A judge WAIT can never park a goal for hours.

Witness (2026-11-22 07:34 UTC): the agent ended a standing-goal turn saying its next evidence was
a report due the next morning; the judge returned ``wait_for_seconds: 111600`` (31 h) and the loop
parked. Nothing re-judged until the owner asked "so you gave up?" 13 hours later. Timed waits exist
for backoff and live delegations, so the loop clamps them to ``_MAX_TIMED_WAIT_S`` and re-judges.
"""
import time
from unittest.mock import patch

from hermes_cli import goals


def _mgr(tmp_path, monkeypatch, sid):
    from pathlib import Path

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    (tmp_path / ".hermes").mkdir(exist_ok=True)
    goals._DB_CACHE.clear()
    mgr = goals.GoalManager(session_id=sid)
    mgr.set("make Uno exits good")
    return mgr


def test_a_31_hour_judge_wait_is_clamped_to_the_cap(tmp_path, monkeypatch):
    mgr = _mgr(tmp_path, monkeypatch, "clamp-sid")
    with patch(
        "hermes_cli.goals.judge_goal",
        return_value=("wait", "next evidence is tomorrow's report", False, {"seconds": 111600}, False),
    ):
        decision = mgr.evaluate_after_turn("Paused until the live report lands tomorrow morning.")
    assert decision["verdict"] == "wait"
    assert mgr.state is not None
    left = mgr.state.waiting_until - time.time()
    assert 0 < left <= goals._MAX_TIMED_WAIT_S + 1
    assert goals._MAX_TIMED_WAIT_S <= 30 * 60
    goals._DB_CACHE.clear()


def test_a_short_backoff_wait_is_kept_as_asked(tmp_path, monkeypatch):
    mgr = _mgr(tmp_path, monkeypatch, "short-sid")
    with patch(
        "hermes_cli.goals.judge_goal",
        return_value=("wait", "rate limited", False, {"seconds": 120}, False),
    ):
        mgr.evaluate_after_turn("Backing off two minutes after a 429.")
    assert mgr.state is not None
    left = mgr.state.waiting_until - time.time()
    assert 100 < left <= 121
    goals._DB_CACHE.clear()


def test_judge_prompt_says_waiting_for_later_data_is_continue():
    assert "tomorrow's games" in goals.JUDGE_SYSTEM_PROMPT
    assert "capped at 1800 s" in goals.JUDGE_SYSTEM_PROMPT
