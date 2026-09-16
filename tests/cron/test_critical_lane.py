"""Reserved critical-lane worker pool in the cron scheduler.

Why this file exists: ``tick`` dispatched every due job into ONE shared
``ThreadPoolExecutor``. Slots are FIFO with no priority, so long-running LLM
agent jobs (measured on the real fleet: p90 60min, p99 92min, max 189min for a
single job) hold every worker while a 17-second money-path script job waits.
Measured queue wait when saturated: median 461s, p95 1410s, max 1573s; the
5-minute autosell job missed 29 scheduled slots in one day.

Every test here drives the REAL ``cron.scheduler.tick`` dispatch path and
asserts on an observable OUTCOME — did the job actually start, and which
lane's threads ran it. No assertions against source text, and the routing
decision is never re-implemented in the test.

``TestSabotage`` at the bottom inverts the production behaviour at runtime to
prove the starvation assertions above have real power rather than passing
vacuously.
"""

import concurrent.futures
import threading
import time

import pytest


TICK_DISPATCH_SLACK_SECONDS = 5.0
"""Generous upper bound for 'the critical job started promptly'.

The general pool in these tests is held by jobs that never return on their own,
so on the unfixed single-pool scheduler the critical job cannot start within
this window no matter how loaded the box is. A value this loose keeps the test
from flaking on a slow CI machine while still failing unambiguously on the bug.
"""

STARVATION_SETTLE_SECONDS = 0.5
"""Time allowed for a wrongly-queued job to (not) appear before asserting absence."""


def _job(job_id, *, critical=None, **extra):
    """Minimal due-job record in the shape ``tick`` passes to the pools."""
    job = {
        "id": job_id,
        "name": job_id,
        "prompt": "test",
        "schedule": "every 5m",
        "enabled": True,
        "next_run_at": "2020-01-01T00:00:00",
        "deliver": "local",
    }
    if critical is not None:
        job["critical"] = critical
    job.update(extra)
    return job


@pytest.fixture()
def lane_harness(monkeypatch):
    """Run a real ``tick`` over a supplied due set, recording per-job outcomes.

    Patches only the scheduler's I/O collaborators (job store, execution
    ledger, sweeps). Pool construction, lane selection, ``_submit_with_guard``
    and the in-flight dedupe guard all execute for real.
    """
    import cron.scheduler as sched

    class Harness:
        def __init__(self):
            self.started = {}          # job_id -> monotonic instant the worker ran it
            self.thread_names = {}     # job_id -> executor thread name that ran it
            self.release = threading.Event()
            self.blockers = set()      # job ids that occupy their worker until released
            self.dispatch_started = None

        def run_tick(self, due_jobs, sync=False, **tick_kwargs):
            monkeypatch.setattr(sched, "get_due_jobs", lambda: list(due_jobs))
            self.dispatch_started = time.monotonic()
            return sched.tick(verbose=False, sync=sync, **tick_kwargs)

        def wait_for_start(self, job_id, timeout):
            """Seconds from dispatch to start, or None if it never started."""
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                if job_id in self.started:
                    return self.started[job_id] - self.dispatch_started
                time.sleep(0.01)
            return None

    h = Harness()

    def fake_process_due_job(job, adapters, loop, verbose):
        job_id = job["id"]
        h.started[job_id] = time.monotonic()
        h.thread_names[job_id] = threading.current_thread().name
        if job_id in h.blockers:
            # Hold this worker until the test explicitly frees it.
            h.release.wait(timeout=60)
        return True

    monkeypatch.setattr(sched, "_process_due_job", fake_process_due_job)
    monkeypatch.setattr(sched, "advance_next_runs", lambda *_a, **_kw: 0)
    monkeypatch.setattr(
        sched, "create_execution", lambda job_id, **_kw: {"id": f"exec-{job_id}"})
    monkeypatch.setattr(sched, "finish_execution", lambda *_a, **_kw: None)
    monkeypatch.setattr(sched, "_sweep_stale_inflight_for_tick", lambda *_a, **_kw: None)
    monkeypatch.setattr(sched, "_sweep_mcp_orphans", lambda *_a, **_kw: None)
    monkeypatch.setattr(sched, "_maybe_reap_dead_owners", lambda *_a, **_kw: None)
    monkeypatch.setattr(sched, "_maybe_run_worktree_maintenance", lambda *_a, **_kw: None)

    # Fresh pools per test so a previous test's saturation cannot leak in.
    sched._shutdown_parallel_pool()
    sched._running_job_ids.clear()
    sched._running_futures.clear()
    sched._running_since.clear()
    try:
        yield h
    finally:
        h.release.set()
        sched._shutdown_parallel_pool()
        sched._running_job_ids.clear()
        sched._running_futures.clear()
        sched._running_since.clear()


class TestCriticalSelector:
    """``is_critical_job`` reads the flag off the Hermes job record."""

    @pytest.mark.parametrize(
        "record, expected",
        [
            ({"critical": True}, True),
            ({"critical": "true"}, True),
            ({"critical": "TRUE"}, True),
            ({"critical": " yes "}, True),
            ({"critical": 1}, True),
            ({"critical": False}, False),
            ({"critical": "false"}, False),
            ({"critical": "no"}, False),
            ({"critical": 0}, False),
            ({"critical": None}, False),
            ({"critical": "maybe"}, False),
            ({}, False),                        # absent flag: unchanged behaviour
            ({"critical": {"tier": 1}}, False),  # junk never grants reserved capacity
            ({"critical": []}, False),
        ],
    )
    def test_flag_parsing(self, record, expected):
        import cron.scheduler as sched

        assert sched.is_critical_job(record) is expected

    def test_membership_is_record_driven_not_id_driven(self):
        """Two records identical except for the flag must classify oppositely.

        Behavioural proof that no job-id allowlist is baked into the scheduler:
        the same id routes both ways depending only on the field.
        """
        import cron.scheduler as sched

        assert sched.is_critical_job(_job("682408eaa586", critical=True)) is True
        assert sched.is_critical_job(_job("682408eaa586")) is False
        assert sched.is_critical_job(_job("any-other-job", critical=True)) is True

    def test_non_dict_input_is_not_critical(self):
        import cron.scheduler as sched

        assert sched.is_critical_job(None) is False
        assert sched.is_critical_job("critical") is False


class TestCriticalLaneNotStarved:
    """THE REGRESSION TESTS. These fail on the pre-fix single-pool scheduler."""

    def test_critical_job_starts_while_general_pool_is_saturated(
        self, monkeypatch, lane_harness
    ):
        """Saturate every general-pool worker, then dispatch a critical job.

        Pre-fix the critical job queues behind the blockers in the one shared
        executor and never starts (the blockers hold their workers until the
        test releases them), so ``wait_for_start`` returns None. Post-fix it
        runs immediately on the reserved lane.
        """
        monkeypatch.setenv("HERMES_CRON_MAX_PARALLEL", "2")
        monkeypatch.setenv("HERMES_CRON_CRITICAL_LANE_WORKERS", "2")

        blockers = [_job("long-agent-1"), _job("long-agent-2")]
        lane_harness.blockers = {"long-agent-1", "long-agent-2"}
        lane_harness.run_tick(blockers, sync=False)

        # Both general workers must really be occupied before we measure.
        assert lane_harness.wait_for_start("long-agent-1", 5.0) is not None
        assert lane_harness.wait_for_start("long-agent-2", 5.0) is not None

        critical = _job("autosell-watcher", critical=True)
        lane_harness.run_tick([critical], sync=False)

        waited = lane_harness.wait_for_start("autosell-watcher", TICK_DISPATCH_SLACK_SECONDS)
        assert waited is not None, (
            "critical job never started while the general pool was saturated — "
            "it is queued behind long-running jobs (the starvation bug)")
        assert waited < TICK_DISPATCH_SLACK_SECONDS

        # And it genuinely ran on a reserved-lane thread, not a general worker.
        assert lane_harness.thread_names["autosell-watcher"].startswith("cron-critical"), (
            f"critical job ran on {lane_harness.thread_names['autosell-watcher']!r}; "
            "expected a reserved-lane thread")

    def test_critical_job_is_not_delayed_by_a_deep_queue_of_agent_jobs(
        self, monkeypatch, lane_harness
    ):
        """Same starvation shape with an oversubscribed backlog behind it."""
        monkeypatch.setenv("HERMES_CRON_MAX_PARALLEL", "2")
        monkeypatch.setenv("HERMES_CRON_CRITICAL_LANE_WORKERS", "1")

        hogs = [_job(f"agent-{i}") for i in range(8)]
        lane_harness.blockers = {j["id"] for j in hogs}
        lane_harness.run_tick(hogs, sync=False)
        assert lane_harness.wait_for_start("agent-0", 5.0) is not None
        assert lane_harness.wait_for_start("agent-1", 5.0) is not None

        critical = _job("autosell-watchdog", critical=True)
        lane_harness.run_tick([critical], sync=False)
        assert lane_harness.wait_for_start(
            "autosell-watchdog", TICK_DISPATCH_SLACK_SECONDS) is not None, (
            "critical job starved behind 8 queued agent jobs")

    def test_both_tier1_jobs_run_concurrently_under_full_general_saturation(
        self, monkeypatch, lane_harness
    ):
        """The intended tier-1 pair must both get slots at the default width 2."""
        monkeypatch.setenv("HERMES_CRON_MAX_PARALLEL", "2")
        monkeypatch.delenv("HERMES_CRON_CRITICAL_LANE_WORKERS", raising=False)

        hogs = [_job("hog-1"), _job("hog-2")]
        lane_harness.blockers = {"hog-1", "hog-2"}
        lane_harness.run_tick(hogs, sync=False)
        assert lane_harness.wait_for_start("hog-1", 5.0) is not None
        assert lane_harness.wait_for_start("hog-2", 5.0) is not None

        pair = [_job("watcher", critical=True), _job("watchdog", critical=True)]
        assert lane_harness.run_tick(pair, sync=True) == 2
        assert "watcher" in lane_harness.started
        assert "watchdog" in lane_harness.started
        for job_id in ("watcher", "watchdog"):
            assert lane_harness.thread_names[job_id].startswith("cron-critical")


class TestLaneRouting:
    """Each job must actually execute on its own lane's threads."""

    def test_non_critical_job_never_runs_on_a_reserved_thread(
        self, monkeypatch, lane_harness
    ):
        """Reserved capacity must be unreachable for ordinary jobs."""
        monkeypatch.setenv("HERMES_CRON_MAX_PARALLEL", "4")
        monkeypatch.setenv("HERMES_CRON_CRITICAL_LANE_WORKERS", "2")

        jobs = [_job(f"ordinary-{i}") for i in range(6)]
        assert lane_harness.run_tick(jobs, sync=True) == 6

        assert len(lane_harness.thread_names) == 6
        for job_id, thread_name in lane_harness.thread_names.items():
            assert not thread_name.startswith("cron-critical"), (
                f"non-critical job {job_id!r} occupied a RESERVED lane thread "
                f"({thread_name!r}) — reserved capacity is not reserved")

    def test_explicitly_non_critical_flag_stays_in_the_general_pool(
        self, monkeypatch, lane_harness
    ):
        monkeypatch.setenv("HERMES_CRON_CRITICAL_LANE_WORKERS", "2")
        jobs = [_job("flag-false", critical=False), _job("flag-absent")]
        assert lane_harness.run_tick(jobs, sync=True) == 2
        for job_id in ("flag-false", "flag-absent"):
            assert lane_harness.thread_names[job_id].startswith("cron-parallel")

    def test_mixed_due_set_splits_across_lanes_per_job(self, monkeypatch, lane_harness):
        """Routing is per-job, not per-tick."""
        monkeypatch.setenv("HERMES_CRON_MAX_PARALLEL", "4")
        monkeypatch.setenv("HERMES_CRON_CRITICAL_LANE_WORKERS", "2")

        jobs = [
            _job("ordinary-a"),
            _job("crit-a", critical=True),
            _job("ordinary-b"),
            _job("crit-b", critical=True),
        ]
        assert lane_harness.run_tick(jobs, sync=True) == 4

        names = lane_harness.thread_names
        assert names["crit-a"].startswith("cron-critical")
        assert names["crit-b"].startswith("cron-critical")
        assert names["ordinary-a"].startswith("cron-parallel")
        assert names["ordinary-b"].startswith("cron-parallel")

    def test_serial_general_pool_does_not_serialize_the_critical_lane(
        self, monkeypatch, lane_harness
    ):
        """max_parallel_jobs=1 (serial general pool) must not bind critical jobs."""
        monkeypatch.setenv("HERMES_CRON_MAX_PARALLEL", "1")
        monkeypatch.setenv("HERMES_CRON_CRITICAL_LANE_WORKERS", "2")

        lane_harness.blockers = {"serial-hog"}
        lane_harness.run_tick([_job("serial-hog")], sync=False)
        assert lane_harness.wait_for_start("serial-hog", 5.0) is not None

        lane_harness.run_tick([_job("crit-under-serial", critical=True)], sync=False)
        assert lane_harness.wait_for_start(
            "crit-under-serial", TICK_DISPATCH_SLACK_SECONDS) is not None


class TestComposesWithExistingMachinery:
    """The lane split must not weaken the guards already in place."""

    def test_inflight_dedupe_guard_still_applies_to_critical_jobs(
        self, monkeypatch, lane_harness
    ):
        """A critical job already in flight is skipped, same as any other."""
        import cron.scheduler as sched

        monkeypatch.setenv("HERMES_CRON_CRITICAL_LANE_WORKERS", "2")
        sched._running_job_ids.add("crit-dedupe")
        try:
            assert lane_harness.run_tick(
                [_job("crit-dedupe", critical=True)], sync=True) == 0
            assert "crit-dedupe" not in lane_harness.started
        finally:
            sched._running_job_ids.discard("crit-dedupe")

    def test_critical_job_releases_its_inflight_claim_after_running(
        self, monkeypatch, lane_harness
    ):
        """Reserved-lane runs must not leak running-set membership."""
        import cron.scheduler as sched

        monkeypatch.setenv("HERMES_CRON_CRITICAL_LANE_WORKERS", "2")
        assert lane_harness.run_tick([_job("crit-release", critical=True)], sync=True) == 1
        assert "crit-release" not in sched._running_job_ids

    def test_sync_tick_waits_for_critical_lane_futures(self, monkeypatch, lane_harness):
        """sync=True must collect reserved-lane results, not just general ones."""
        monkeypatch.setenv("HERMES_CRON_CRITICAL_LANE_WORKERS", "2")
        jobs = [_job("crit-sync-1", critical=True), _job("crit-sync-2", critical=True)]
        assert lane_harness.run_tick(jobs, sync=True) == 2
        # Already finished by the time tick returned — that is what sync means.
        assert set(lane_harness.started) == {"crit-sync-1", "crit-sync-2"}

    def test_sync_tick_collects_results_from_both_lanes(self, monkeypatch, lane_harness):
        monkeypatch.setenv("HERMES_CRON_CRITICAL_LANE_WORKERS", "2")
        jobs = [_job("mix-plain"), _job("mix-crit", critical=True)]
        assert lane_harness.run_tick(jobs, sync=True) == 2
        assert set(lane_harness.started) == {"mix-plain", "mix-crit"}

    def test_async_tick_returns_before_a_blocking_critical_job_finishes(
        self, monkeypatch, lane_harness
    ):
        """sync=False must not block the ticker thread on the reserved lane."""
        monkeypatch.setenv("HERMES_CRON_CRITICAL_LANE_WORKERS", "2")
        lane_harness.blockers = {"crit-async"}

        t0 = time.monotonic()
        assert lane_harness.run_tick([_job("crit-async", critical=True)], sync=False) == 1
        assert time.monotonic() - t0 < TICK_DISPATCH_SLACK_SECONDS
        assert lane_harness.wait_for_start("crit-async", 5.0) is not None

    def test_mcp_orphan_sweep_still_fires_after_critical_lane_jobs(
        self, monkeypatch, lane_harness
    ):
        """The async done-callback sweep must see reserved-lane futures too."""
        import cron.scheduler as sched

        swept = threading.Event()
        monkeypatch.setattr(sched, "_sweep_mcp_orphans", lambda *_a, **_kw: swept.set())
        monkeypatch.setenv("HERMES_CRON_CRITICAL_LANE_WORKERS", "2")

        assert lane_harness.run_tick([_job("crit-sweep", critical=True)], sync=False) == 1
        assert swept.wait(timeout=5.0), (
            "MCP orphan sweep never ran after a reserved-lane job completed")

    def test_tick_with_no_critical_jobs_never_builds_a_reserved_pool(
        self, monkeypatch, lane_harness
    ):
        """Zero cost for the jobs that do not opt in."""
        import cron.scheduler as sched

        jobs = [_job(f"plain-{i}") for i in range(3)]
        assert lane_harness.run_tick(jobs, sync=True) == 3
        assert sched._critical_pool is None

    def test_tick_with_only_critical_jobs_never_builds_a_general_pool(
        self, monkeypatch, lane_harness
    ):
        import cron.scheduler as sched

        monkeypatch.setenv("HERMES_CRON_CRITICAL_LANE_WORKERS", "2")
        assert lane_harness.run_tick([_job("only-crit", critical=True)], sync=True) == 1
        assert sched._parallel_pool is None


class TestReservedPoolLifecycle:
    """Pool construction, resize and shutdown."""

    def test_pool_is_reused_when_width_is_unchanged(self):
        import cron.scheduler as sched

        sched._shutdown_parallel_pool()
        try:
            assert sched._get_critical_pool(2) is sched._get_critical_pool(2)
        finally:
            sched._shutdown_parallel_pool()

    def test_pool_is_replaced_and_not_leaked_when_width_changes(self):
        """A config change swaps the executor and shuts the old one down."""
        import cron.scheduler as sched

        sched._shutdown_parallel_pool()
        try:
            first = sched._get_critical_pool(2)
            second = sched._get_critical_pool(3)
            assert first is not second
            assert sched._critical_pool_max_workers == 3
            # Observable proof the old executor was shut down: it refuses work.
            with pytest.raises(RuntimeError):
                first.submit(lambda: None)
        finally:
            sched._shutdown_parallel_pool()

    def test_reserved_pool_is_a_separate_executor_from_the_general_pool(self):
        import cron.scheduler as sched

        sched._shutdown_parallel_pool()
        try:
            assert sched._get_critical_pool(2) is not sched._get_parallel_pool(2)
        finally:
            sched._shutdown_parallel_pool()

    def test_shutdown_clears_both_pools(self):
        import cron.scheduler as sched

        sched._get_parallel_pool(2)
        sched._get_critical_pool(2)
        sched._shutdown_parallel_pool()
        assert sched._parallel_pool is None
        assert sched._critical_pool is None
        assert sched._critical_pool_max_workers is None

    def test_shutdown_is_idempotent(self):
        import cron.scheduler as sched

        sched._get_critical_pool(2)
        sched._shutdown_parallel_pool()
        sched._shutdown_parallel_pool()
        assert sched._critical_pool is None

    def test_reserved_pool_width_is_honoured(self, monkeypatch, lane_harness):
        """A 1-wide reserved lane really only runs one critical job at a time."""
        monkeypatch.setenv("HERMES_CRON_CRITICAL_LANE_WORKERS", "1")

        jobs = [_job("crit-x", critical=True), _job("crit-y", critical=True)]
        lane_harness.blockers = {"crit-x", "crit-y"}
        lane_harness.run_tick(jobs, sync=False)

        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline and not lane_harness.started:
            time.sleep(0.01)
        time.sleep(STARVATION_SETTLE_SECONDS)
        assert len(lane_harness.started) == 1, (
            f"reserved lane width=1 ran {len(lane_harness.started)} jobs concurrently")


class TestLaneWidthConfig:
    """``_resolve_critical_lane_workers``: env > config.yaml > safe default."""

    def test_default_when_unset(self, monkeypatch):
        import cron.scheduler as sched

        monkeypatch.delenv("HERMES_CRON_CRITICAL_LANE_WORKERS", raising=False)
        monkeypatch.setattr(sched, "load_config", lambda: {})
        assert sched._resolve_critical_lane_workers() == sched._DEFAULT_CRITICAL_LANE_WORKERS

    def test_env_wins_over_config(self, monkeypatch):
        import cron.scheduler as sched

        monkeypatch.setenv("HERMES_CRON_CRITICAL_LANE_WORKERS", "5")
        monkeypatch.setattr(sched, "load_config", lambda: {"cron": {"critical_lane_workers": 3}})
        assert sched._resolve_critical_lane_workers() == 5

    def test_config_used_when_env_absent(self, monkeypatch):
        import cron.scheduler as sched

        monkeypatch.delenv("HERMES_CRON_CRITICAL_LANE_WORKERS", raising=False)
        monkeypatch.setattr(sched, "load_config", lambda: {"cron": {"critical_lane_workers": 4}})
        assert sched._resolve_critical_lane_workers() == 4

    @pytest.mark.parametrize("bad", ["0", "-3", "abc", "2.5"])
    def test_invalid_env_falls_back_to_default_never_disables_the_lane(
        self, monkeypatch, bad
    ):
        import cron.scheduler as sched

        monkeypatch.setenv("HERMES_CRON_CRITICAL_LANE_WORKERS", bad)
        monkeypatch.setattr(sched, "load_config", lambda: {})
        assert sched._resolve_critical_lane_workers() == sched._DEFAULT_CRITICAL_LANE_WORKERS

    def test_blank_env_falls_through_to_config(self, monkeypatch):
        """An empty env var must not mask a valid config value."""
        import cron.scheduler as sched

        monkeypatch.setenv("HERMES_CRON_CRITICAL_LANE_WORKERS", "   ")
        monkeypatch.setattr(sched, "load_config", lambda: {"cron": {"critical_lane_workers": 6}})
        assert sched._resolve_critical_lane_workers() == 6

    @pytest.mark.parametrize("bad", [0, -1, "nope", [], {}])
    def test_invalid_config_falls_back_to_default(self, monkeypatch, bad):
        import cron.scheduler as sched

        monkeypatch.delenv("HERMES_CRON_CRITICAL_LANE_WORKERS", raising=False)
        monkeypatch.setattr(sched, "load_config", lambda: {"cron": {"critical_lane_workers": bad}})
        assert sched._resolve_critical_lane_workers() == sched._DEFAULT_CRITICAL_LANE_WORKERS

    def test_load_config_failure_falls_back_to_default(self, monkeypatch):
        import cron.scheduler as sched

        monkeypatch.delenv("HERMES_CRON_CRITICAL_LANE_WORKERS", raising=False)

        def boom():
            raise RuntimeError("config unreadable")

        monkeypatch.setattr(sched, "load_config", boom)
        assert sched._resolve_critical_lane_workers() == sched._DEFAULT_CRITICAL_LANE_WORKERS

    def test_config_change_between_ticks_resizes_the_lane_and_still_runs_jobs(
        self, monkeypatch, lane_harness
    ):
        """Observable: width changes across ticks, and the new pool works."""
        import cron.scheduler as sched

        monkeypatch.setenv("HERMES_CRON_CRITICAL_LANE_WORKERS", "1")
        lane_harness.run_tick([_job("crit-resize-1", critical=True)], sync=True)
        assert sched._critical_pool_max_workers == 1
        first_pool = sched._critical_pool

        monkeypatch.setenv("HERMES_CRON_CRITICAL_LANE_WORKERS", "3")
        lane_harness.run_tick([_job("crit-resize-2", critical=True)], sync=True)
        assert sched._critical_pool_max_workers == 3
        assert sched._critical_pool is not first_pool
        assert "crit-resize-2" in lane_harness.started


class TestSabotage:
    """Prove the starvation assertions above have power.

    Each test inverts the production behaviour at runtime and asserts the
    BROKEN outcome is observable. If one of these fails, the corresponding
    positive test is passing vacuously and cannot be trusted.
    """

    def test_routing_critical_jobs_to_the_general_pool_reproduces_starvation(
        self, monkeypatch, lane_harness
    ):
        """SABOTAGE: reserved lane returns the GENERAL pool.

        With both lanes backed by one saturated executor the critical job must
        NOT start — which is exactly what
        ``test_critical_job_starts_while_general_pool_is_saturated`` asserts
        against, so that assertion is load-bearing.
        """
        import cron.scheduler as sched

        monkeypatch.setenv("HERMES_CRON_MAX_PARALLEL", "2")
        shared = sched._get_parallel_pool(2)
        monkeypatch.setattr(sched, "_get_critical_pool", lambda _w: shared)

        lane_harness.blockers = {"hog-1", "hog-2"}
        lane_harness.run_tick([_job("hog-1"), _job("hog-2")], sync=False)
        assert lane_harness.wait_for_start("hog-1", 5.0) is not None
        assert lane_harness.wait_for_start("hog-2", 5.0) is not None

        lane_harness.run_tick([_job("crit", critical=True)], sync=False)
        assert lane_harness.wait_for_start("crit", TICK_DISPATCH_SLACK_SECONDS) is None, (
            "SABOTAGE INEFFECTIVE: a shared executor did not starve the critical "
            "job, so the reserved-lane assertion proves nothing")

    def test_selector_returning_false_reproduces_starvation(
        self, monkeypatch, lane_harness
    ):
        """SABOTAGE: the critical selector never fires.

        A flagged job then takes the general lane and starves, proving the
        selector is what routes jobs to safety.
        """
        import cron.scheduler as sched

        monkeypatch.setenv("HERMES_CRON_MAX_PARALLEL", "2")
        monkeypatch.setattr(sched, "is_critical_job", lambda job: False)

        lane_harness.blockers = {"hog-a", "hog-b"}
        lane_harness.run_tick([_job("hog-a"), _job("hog-b")], sync=False)
        assert lane_harness.wait_for_start("hog-a", 5.0) is not None
        assert lane_harness.wait_for_start("hog-b", 5.0) is not None

        lane_harness.run_tick([_job("crit-sel", critical=True)], sync=False)
        assert lane_harness.wait_for_start(
            "crit-sel", TICK_DISPATCH_SLACK_SECONDS) is None, (
            "SABOTAGE INEFFECTIVE: a dead selector did not starve the flagged job, "
            "so the selector assertions prove nothing")

    def test_selector_returning_true_for_everything_lets_ordinary_jobs_take_reserved_slots(
        self, monkeypatch, lane_harness
    ):
        """SABOTAGE: every job is 'critical'.

        Ordinary jobs then land on reserved threads, which is precisely what
        ``test_non_critical_job_never_runs_on_a_reserved_thread`` forbids.
        """
        import cron.scheduler as sched

        monkeypatch.setattr(sched, "is_critical_job", lambda job: True)
        monkeypatch.setenv("HERMES_CRON_CRITICAL_LANE_WORKERS", "2")

        jobs = [_job("ordinary-1"), _job("ordinary-2")]
        lane_harness.run_tick(jobs, sync=True)

        assert any(
            name.startswith("cron-critical") for name in lane_harness.thread_names.values()
        ), (
            "SABOTAGE INEFFECTIVE: ordinary jobs did not reach reserved threads, so "
            "the 'reserved capacity is reserved' assertion proves nothing")
