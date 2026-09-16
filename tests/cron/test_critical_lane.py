"""Reserved critical-lane worker pool in the cron scheduler.

Why this file exists: ``tick`` dispatched every due job into ONE shared
``ThreadPoolExecutor``.  Slots are FIFO with no priority, so long-running LLM
agent jobs hold every worker while a short deterministic script job waits
behind them for as long as those agent jobs run.

Every test here drives the REAL ``cron.scheduler.tick`` dispatch path (or the
real selector) and asserts on an observable OUTCOME -- did the job actually
start, and on which lane's threads.  No test asserts on source text, and no
test re-implements the routing decision it is checking.

``TestSabotage`` at the bottom is the meta-layer: it re-introduces each bug
this feature fixes and proves the assertions above actually go red.  A
regression test that cannot fail is not a test.
"""

import threading
import time

import pytest


TICK_DISPATCH_SLACK_SECONDS = 5.0
"""Generous upper bound for 'the critical job started promptly'.

The general pool in the regression test is held far longer than this by jobs
that never return on their own, so on the unfixed code the critical job cannot
start within this window no matter how loaded the CI box is.  A value this
loose keeps the test from flaking on a slow machine while still failing
unambiguously on the starvation bug.
"""


def _job(job_id, *, critical=None, agent=False, script="echo hi", **extra):
    """Minimal due-job record in the shape ``tick`` passes to the pools.

    Defaults to a ``no_agent`` SCRIPT job, because that is the only shape the
    reserved lane admits.  ``agent=True`` builds a prompt-driven agent job
    instead (the shape that must always stay in the general pool).
    """
    job = {
        "id": job_id,
        "name": job_id,
        "prompt": "test",
        "schedule": "every 5m",
        "enabled": True,
        "next_run_at": "2020-01-01T00:00:00",
        "deliver": "local",
    }
    if agent:
        job["no_agent"] = False
        job["script"] = None
    else:
        job["no_agent"] = True
        job["script"] = script
    if critical is not None:
        job["critical"] = critical
    job.update(extra)
    return job


@pytest.fixture()
def lane_harness(monkeypatch):
    """Run a real ``tick`` over a supplied due set, recording per-job outcomes.

    Patches only the scheduler's I/O collaborators (job store, execution
    ledger, delivery).  Pool construction, lane selection, ``_submit_with_guard``
    and the in-flight dedupe guard all run for real.
    """
    import cron.scheduler as sched

    class Harness:
        def __init__(self):
            self.started = {}          # job_id -> monotonic instant the job body ran
            self.thread_names = {}     # job_id -> executor thread that ran it
            self.release = threading.Event()
            self.blockers = set()      # job ids that wait on self.release
            self.dispatch_started = None

        def run_tick(self, due_jobs, sync=False, **tick_kwargs):
            _set_due(monkeypatch, due_jobs)
            self.dispatch_started = time.monotonic()
            return sched.tick(verbose=False, sync=sync, **tick_kwargs)

        def wait_for_start(self, job_id, timeout):
            """Seconds from dispatch to job start, or None if it never started."""
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
            # Occupy the worker until the test explicitly frees it.
            h.release.wait(timeout=60)
        return True

    monkeypatch.setattr(sched, "_process_due_job", fake_process_due_job)
    monkeypatch.setattr(sched, "advance_next_runs", lambda *_a, **_kw: 0)
    monkeypatch.setattr(
        sched, "create_execution",
        lambda job_id, **_kw: {"id": f"exec-{job_id}"})
    monkeypatch.setattr(sched, "_sweep_stale_inflight_for_tick", lambda *_a, **_kw: None)
    monkeypatch.setattr(sched, "_sweep_mcp_orphans", lambda *_a, **_kw: None)
    monkeypatch.setattr(sched, "_sweep_mcp_orphans_when_all_done", lambda *_a, **_kw: None)
    monkeypatch.setattr(sched, "_maybe_reap_dead_owners", lambda *_a, **_kw: None)
    monkeypatch.setattr(sched, "_maybe_run_worktree_maintenance", lambda *_a, **_kw: None)
    monkeypatch.setattr(sched, "_should_yield_tick_to_fresh_gateway", lambda *_a, **_kw: None)

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


def _set_due(monkeypatch, jobs):
    import cron.scheduler as sched

    monkeypatch.setattr(sched, "get_due_jobs", lambda: list(jobs))


class TestCriticalSelector:
    """``is_critical_job``: the flag AND a no_agent script job, both required."""

    @pytest.mark.parametrize(
        "flag, expected",
        [
            (True, True),
            ("true", True),
            ("TRUE", True),
            ("yes", True),
            ("on", True),
            (1, True),
            (False, False),
            ("false", False),
            ("no", False),
            (0, False),
            (None, False),
            ("maybe", False),
            ({"tier": 1}, False),   # junk never grants reserved capacity
            ([], False),
        ],
    )
    def test_flag_parsing_on_a_qualifying_script_job(self, flag, expected):
        import cron.scheduler as sched

        assert sched.is_critical_job(_job("j", critical=flag)) is expected

    def test_absent_flag_is_not_critical(self):
        """Every existing job keeps its current general-pool behaviour."""
        import cron.scheduler as sched

        assert sched.is_critical_job(_job("plain-script-job")) is False

    def test_non_dict_is_not_critical(self):
        import cron.scheduler as sched

        for junk in (None, "critical", 1, [], object()):
            assert sched.is_critical_job(junk) is False  # type: ignore[arg-type]

    def test_no_job_ids_are_hardcoded_in_the_selector(self):
        """Membership is operator-controlled via the record, not baked into code.

        Behavioural, not textual: two records identical except for the flag
        must classify oppositely regardless of their ids/names, so an id-based
        allowlist could not produce this result.
        """
        import cron.scheduler as sched

        assert sched.is_critical_job(_job("682408eaa586", critical=True)) is True
        assert sched.is_critical_job(_job("682408eaa586")) is False
        assert sched.is_critical_job(_job("some-other-job", critical=True)) is True


class TestOnlyScriptJobsAreAdmitted:
    """The lane is for SHORT DETERMINISTIC scripts; agent jobs never qualify.

    Admitting an agent job would put unbounded-duration work (a model turn)
    into a reserved slot -- precisely the failure the reservation exists to
    prevent, only now inside the protected lane where nothing else can evict
    it.
    """

    def test_flagged_agent_job_is_not_admitted(self):
        import cron.scheduler as sched

        flagged_agent = _job("llm-job", critical=True, agent=True)
        assert sched.is_critical_job(flagged_agent) is False

    def test_flagged_job_with_script_but_no_no_agent_flag_is_not_admitted(self):
        """A `script` on an AGENT job is prompt context, not the whole job."""
        import cron.scheduler as sched

        job = _job("ctx", critical=True)
        job["no_agent"] = False
        assert sched.is_critical_job(job) is False

    @pytest.mark.parametrize("script", [None, "", "   ", 0, [], {}])
    def test_flagged_no_agent_job_without_a_real_script_is_not_admitted(self, script):
        """no_agent with no script has nothing to run -- it must not hold a slot."""
        import cron.scheduler as sched

        assert sched.is_critical_job(_job("empty", critical=True, script=script)) is False

    def test_flagged_agent_job_actually_runs_on_the_general_pool(
        self, monkeypatch, lane_harness
    ):
        """End-to-end: the exclusion is enforced by dispatch, not just the selector."""
        monkeypatch.setenv("HERMES_CRON_MAX_PARALLEL", "4")
        monkeypatch.setenv("HERMES_CRON_CRITICAL_LANE_WORKERS", "2")

        flagged_agent = _job("llm-job", critical=True, agent=True)
        assert lane_harness.run_tick([flagged_agent], sync=True) == 1

        thread_name = lane_harness.thread_names["llm-job"]
        assert not thread_name.startswith("cron-critical"), (
            f"a flagged AGENT job occupied a reserved slot ({thread_name!r}); "
            "unbounded model turns must never enter the reserved lane"
        )

    def test_agent_job_cannot_evict_a_script_job_from_the_reserved_lane(
        self, monkeypatch, lane_harness
    ):
        """The money test for the exclusion, with a 1-wide lane.

        If flagged agent jobs were admitted, this long-running agent job would
        hold the single reserved slot and the script job the lane exists for
        would starve -- the original bug, relocated inside the lane.
        """
        monkeypatch.setenv("HERMES_CRON_MAX_PARALLEL", "4")
        monkeypatch.setenv("HERMES_CRON_CRITICAL_LANE_WORKERS", "1")

        hog = _job("flagged-llm-hog", critical=True, agent=True)
        lane_harness.blockers = {"flagged-llm-hog"}
        lane_harness.run_tick([hog], sync=False)
        assert lane_harness.wait_for_start("flagged-llm-hog", 5.0) is not None

        script_job = _job("autosell", critical=True)
        lane_harness.run_tick([script_job], sync=False)

        waited = lane_harness.wait_for_start("autosell", TICK_DISPATCH_SLACK_SECONDS)
        assert waited is not None, (
            "the critical SCRIPT job starved because a flagged agent job was "
            "holding the reserved slot"
        )
        assert lane_harness.thread_names["autosell"].startswith("cron-critical")


class TestCriticalLaneNotStarved:
    """THE REGRESSION TEST. Fails on the pre-fix single-pool scheduler."""

    def test_critical_job_starts_while_general_pool_is_saturated(
        self, monkeypatch, lane_harness
    ):
        """Saturate every general-pool worker, then dispatch a critical job.

        Pre-fix, the critical job is queued behind the blockers in the one
        shared executor and never starts (the blockers hold their workers
        until the test releases them), so ``wait_for_start`` returns None.
        Post-fix it runs on the reserved lane immediately.
        """
        monkeypatch.setenv("HERMES_CRON_MAX_PARALLEL", "2")
        monkeypatch.setenv("HERMES_CRON_CRITICAL_LANE_WORKERS", "2")

        blockers = [_job("long-agent-1", agent=True), _job("long-agent-2", agent=True)]
        lane_harness.blockers = {"long-agent-1", "long-agent-2"}
        lane_harness.run_tick(blockers, sync=False)

        # Both general workers must actually be occupied before we measure.
        assert lane_harness.wait_for_start("long-agent-1", 5.0) is not None
        assert lane_harness.wait_for_start("long-agent-2", 5.0) is not None

        # Now the money job fires. It must not wait on those two.
        critical = _job("autosell-watcher", critical=True)
        lane_harness.run_tick([critical], sync=False)

        waited = lane_harness.wait_for_start("autosell-watcher", TICK_DISPATCH_SLACK_SECONDS)
        assert waited is not None, (
            "critical job never started while the general pool was saturated -- "
            "it is queued behind long-running jobs (the starvation bug)"
        )
        assert waited < TICK_DISPATCH_SLACK_SECONDS

        # And it genuinely ran on the reserved lane, not a general worker.
        assert lane_harness.thread_names["autosell-watcher"].startswith("cron-critical"), (
            f"critical job ran on {lane_harness.thread_names['autosell-watcher']!r}; "
            "expected a reserved-lane thread"
        )

    def test_many_saturating_agent_jobs_still_do_not_delay_the_critical_job(
        self, monkeypatch, lane_harness
    ):
        """Same starvation shape with an oversubscribed queue behind it."""
        monkeypatch.setenv("HERMES_CRON_MAX_PARALLEL", "2")
        monkeypatch.setenv("HERMES_CRON_CRITICAL_LANE_WORKERS", "1")

        hogs = [_job(f"agent-{i}", agent=True) for i in range(8)]
        lane_harness.blockers = {j["id"] for j in hogs}
        lane_harness.run_tick(hogs, sync=False)
        assert lane_harness.wait_for_start("agent-0", 5.0) is not None
        assert lane_harness.wait_for_start("agent-1", 5.0) is not None

        critical = _job("autosell-watchdog", critical=True)
        lane_harness.run_tick([critical], sync=False)
        assert lane_harness.wait_for_start("autosell-watchdog", TICK_DISPATCH_SLACK_SECONDS) \
            is not None, "critical job starved behind 8 queued agent jobs"

    def test_critical_job_in_the_same_tick_as_saturating_jobs(
        self, monkeypatch, lane_harness
    ):
        """Starvation within ONE tick, not just across ticks.

        The critical job is dispatched LAST in the due list, behind enough
        blockers to fill the general pool. Single-pool ordering would bury it.
        """
        monkeypatch.setenv("HERMES_CRON_MAX_PARALLEL", "2")
        monkeypatch.setenv("HERMES_CRON_CRITICAL_LANE_WORKERS", "1")

        hogs = [_job(f"hog-{i}", agent=True) for i in range(4)]
        lane_harness.blockers = {j["id"] for j in hogs}
        due = [*hogs, _job("crit-same-tick", critical=True)]
        lane_harness.run_tick(due, sync=False)

        assert lane_harness.wait_for_start("crit-same-tick", TICK_DISPATCH_SLACK_SECONDS) \
            is not None, "critical job starved behind same-tick blockers"
        assert lane_harness.thread_names["crit-same-tick"].startswith("cron-critical")


class TestLaneRouting:
    """Each job must actually execute on its own lane's threads."""

    def test_non_critical_job_never_runs_on_a_reserved_thread(
        self, monkeypatch, lane_harness
    ):
        """The reserved capacity must be unreachable for ordinary jobs."""
        monkeypatch.setenv("HERMES_CRON_MAX_PARALLEL", "4")
        monkeypatch.setenv("HERMES_CRON_CRITICAL_LANE_WORKERS", "2")

        jobs = [_job(f"ordinary-{i}") for i in range(6)]
        assert lane_harness.run_tick(jobs, sync=True) == 6

        assert len(lane_harness.thread_names) == 6
        for job_id, thread_name in lane_harness.thread_names.items():
            assert not thread_name.startswith("cron-critical"), (
                f"non-critical job {job_id!r} occupied a RESERVED lane thread "
                f"({thread_name!r}) -- reserved capacity is not reserved"
            )

    def test_critical_and_ordinary_jobs_in_one_tick_split_across_lanes(
        self, monkeypatch, lane_harness
    ):
        """A mixed due set routes per-job, not per-tick."""
        monkeypatch.setenv("HERMES_CRON_MAX_PARALLEL", "4")
        monkeypatch.setenv("HERMES_CRON_CRITICAL_LANE_WORKERS", "2")

        jobs = [
            _job("ordinary-a"),
            _job("crit-a", critical=True),
            _job("ordinary-b"),
            _job("crit-b", critical=True),
        ]
        assert lane_harness.run_tick(jobs, sync=True) == 4

        for job_id in ("crit-a", "crit-b"):
            assert lane_harness.thread_names[job_id].startswith("cron-critical"), (
                f"{job_id} did not run on the reserved lane")
        for job_id in ("ordinary-a", "ordinary-b"):
            assert not lane_harness.thread_names[job_id].startswith("cron-critical"), (
                f"{job_id} leaked into the reserved lane")

    def test_tick_with_only_ordinary_jobs_never_builds_a_reserved_pool(
        self, monkeypatch, lane_harness
    ):
        """No cost imposed on the overwhelmingly common case."""
        import cron.scheduler as sched

        monkeypatch.setenv("HERMES_CRON_MAX_PARALLEL", "2")
        jobs = [_job("only-ordinary")]
        lane_harness.run_tick(jobs, sync=True)
        assert sched._critical_pool is None, (
            "reserved pool was constructed for a tick containing no critical jobs")

    def test_tick_return_count_includes_both_lanes(self, monkeypatch, lane_harness):
        """The dispatch count must not silently drop reserved-lane jobs."""
        monkeypatch.setenv("HERMES_CRON_MAX_PARALLEL", "4")
        monkeypatch.setenv("HERMES_CRON_CRITICAL_LANE_WORKERS", "2")

        jobs = [_job("o-1"), _job("c-1", critical=True), _job("c-2", critical=True)]
        assert lane_harness.run_tick(jobs, sync=True) == 3


class TestInFlightGuardAppliesToBothLanes:
    """The reserved lane must not bypass the existing dedupe guard."""

    def test_critical_job_already_running_is_not_dispatched_twice(
        self, monkeypatch, lane_harness
    ):
        import cron.scheduler as sched

        monkeypatch.setenv("HERMES_CRON_CRITICAL_LANE_WORKERS", "4")

        job = _job("crit-dupe", critical=True)
        lane_harness.blockers = {"crit-dupe"}
        lane_harness.run_tick([job], sync=False)
        assert lane_harness.wait_for_start("crit-dupe", 5.0) is not None

        first_start = lane_harness.started["crit-dupe"]
        assert "crit-dupe" in sched._running_job_ids

        # Second tick while still in flight: must be skipped, not re-run.
        assert lane_harness.run_tick([job], sync=True) == 0
        assert lane_harness.started["crit-dupe"] == first_start

    def test_critical_job_releases_its_claim_after_completion(
        self, monkeypatch, lane_harness
    ):
        import cron.scheduler as sched

        monkeypatch.setenv("HERMES_CRON_CRITICAL_LANE_WORKERS", "2")

        job = _job("crit-release", critical=True)
        assert lane_harness.run_tick([job], sync=True) == 1
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and "crit-release" in sched._running_job_ids:
            time.sleep(0.01)
        assert "crit-release" not in sched._running_job_ids, (
            "reserved-lane job never released its in-flight claim -- it would "
            "wedge as 'already running' forever")


class TestReservedPoolLifecycle:
    """Pool construction, resize and shutdown."""

    def test_reserved_pool_is_reused_across_ticks(self):
        import cron.scheduler as sched

        sched._shutdown_parallel_pool()
        try:
            assert sched._get_critical_pool(2) is sched._get_critical_pool(2)
        finally:
            sched._shutdown_parallel_pool()

    def test_resize_replaces_and_shuts_down_the_old_pool(self):
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

    def test_reserved_pool_is_separate_from_the_general_pool(self):
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
        assert sched._parallel_pool_max_workers is None
        assert sched._critical_pool is None
        assert sched._critical_pool_max_workers is None

    def test_reserved_pool_width_is_honoured(self, monkeypatch, lane_harness):
        """A 1-wide reserved lane really only runs one critical job at a time."""
        monkeypatch.setenv("HERMES_CRON_CRITICAL_LANE_WORKERS", "1")

        jobs = [_job("crit-x", critical=True), _job("crit-y", critical=True)]
        lane_harness.blockers = {"crit-x", "crit-y"}
        lane_harness.run_tick(jobs, sync=False)

        # Exactly one of the two can be running; the other waits for the slot.
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline and not lane_harness.started:
            time.sleep(0.01)
        time.sleep(0.3)
        assert len(lane_harness.started) == 1, (
            f"reserved lane width=1 ran {len(lane_harness.started)} jobs concurrently")

    def test_reserved_lane_runs_widthwise_in_parallel(self, monkeypatch, lane_harness):
        """A 3-wide lane really runs three critical jobs at once."""
        monkeypatch.setenv("HERMES_CRON_CRITICAL_LANE_WORKERS", "3")

        jobs = [_job(f"crit-p{i}", critical=True) for i in range(3)]
        lane_harness.blockers = {j["id"] for j in jobs}
        lane_harness.run_tick(jobs, sync=False)

        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and len(lane_harness.started) < 3:
            time.sleep(0.01)
        assert len(lane_harness.started) == 3


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

    @pytest.mark.parametrize("bad", ["0", "-3", "abc", "2.5", ""])
    def test_invalid_env_falls_back_to_default_never_disables_the_lane(
        self, monkeypatch, bad
    ):
        import cron.scheduler as sched

        monkeypatch.setenv("HERMES_CRON_CRITICAL_LANE_WORKERS", bad)
        monkeypatch.setattr(sched, "load_config", lambda: {})
        assert sched._resolve_critical_lane_workers() == sched._DEFAULT_CRITICAL_LANE_WORKERS

    @pytest.mark.parametrize("bad", [0, -1, "nope", None])
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

    def test_config_default_is_a_usable_lane_width(self):
        """The shipped config default must itself pass the >= 1 rule."""
        from hermes_cli.config_defaults import DEFAULT_CONFIG

        assert DEFAULT_CONFIG["cron"]["critical_lane_workers"] >= 1

    def test_config_change_between_ticks_resizes_the_lane(self, monkeypatch, lane_harness):
        """Observable: width actually changes across ticks without leaking."""
        import cron.scheduler as sched

        monkeypatch.setenv("HERMES_CRON_CRITICAL_LANE_WORKERS", "1")
        lane_harness.run_tick([_job("crit-resize-1", critical=True)], sync=True)
        assert sched._critical_pool_max_workers == 1

        monkeypatch.setenv("HERMES_CRON_CRITICAL_LANE_WORKERS", "3")
        lane_harness.run_tick([_job("crit-resize-2", critical=True)], sync=True)
        assert sched._critical_pool_max_workers == 3
        assert "crit-resize-2" in lane_harness.started


class TestSabotage:
    """Prove the tests above can actually FAIL.

    Each case re-introduces one specific bug this feature fixes and asserts
    the corresponding check goes red.  A green suite means nothing unless the
    assertions have discriminating power; these tests measure that power
    directly instead of assuming it.
    """

    def test_single_pool_routing_reproduces_the_starvation_bug(
        self, monkeypatch, lane_harness
    ):
        """SABOTAGE: send critical jobs to the general pool (the pre-fix code).

        This is the exact behaviour before the change.  The regression test's
        central assertion -- the critical job starts promptly -- must fail.
        """
        import cron.scheduler as sched

        # The pre-fix world: nothing is ever critical, so everything shares one pool.
        monkeypatch.setattr(sched, "is_critical_job", lambda job: False)
        monkeypatch.setenv("HERMES_CRON_MAX_PARALLEL", "2")

        blockers = [_job("hog-1", agent=True), _job("hog-2", agent=True)]
        lane_harness.blockers = {"hog-1", "hog-2"}
        lane_harness.run_tick(blockers, sync=False)
        assert lane_harness.wait_for_start("hog-1", 5.0) is not None
        assert lane_harness.wait_for_start("hog-2", 5.0) is not None

        critical = _job("autosell-watcher", critical=True)
        lane_harness.run_tick([critical], sync=False)

        waited = lane_harness.wait_for_start("autosell-watcher", TICK_DISPATCH_SLACK_SECONDS)
        assert waited is None, (
            "SABOTAGE INEFFECTIVE: the critical job started even with single-pool "
            "routing, so the regression test could pass on the unfixed code and "
            "proves nothing"
        )

    def test_shared_pool_object_reproduces_the_starvation_bug(
        self, monkeypatch, lane_harness
    ):
        """SABOTAGE: keep two lanes, but make them the SAME executor.

        Catches a 'fix' that routes correctly while the reserved pool is not
        actually reserved (e.g. returning the general pool).
        """
        import cron.scheduler as sched

        monkeypatch.setenv("HERMES_CRON_MAX_PARALLEL", "2")
        shared = sched._get_parallel_pool(2)
        monkeypatch.setattr(sched, "_get_critical_pool", lambda _w: shared)

        blockers = [_job("hog-1", agent=True), _job("hog-2", agent=True)]
        lane_harness.blockers = {"hog-1", "hog-2"}
        lane_harness.run_tick(blockers, sync=False)
        assert lane_harness.wait_for_start("hog-1", 5.0) is not None
        assert lane_harness.wait_for_start("hog-2", 5.0) is not None

        lane_harness.run_tick([_job("crit", critical=True)], sync=False)
        assert lane_harness.wait_for_start("crit", TICK_DISPATCH_SLACK_SECONDS) is None, (
            "SABOTAGE INEFFECTIVE: a shared executor did not starve the critical "
            "job, so the 'reserved' assertion has no power"
        )

    def test_admitting_agent_jobs_reproduces_in_lane_starvation(
        self, monkeypatch, lane_harness
    ):
        """SABOTAGE: drop the no_agent requirement from the selector.

        Proves TestOnlyScriptJobsAreAdmitted is load-bearing: with agent jobs
        admitted, a flagged agent job eats the single reserved slot and the
        script job starves inside the lane.
        """
        import cron.scheduler as sched

        monkeypatch.setattr(
            sched, "is_critical_job", lambda job: sched._critical_flag_is_set(job))
        monkeypatch.setenv("HERMES_CRON_MAX_PARALLEL", "4")
        monkeypatch.setenv("HERMES_CRON_CRITICAL_LANE_WORKERS", "1")

        hog = _job("flagged-llm-hog", critical=True, agent=True)
        lane_harness.blockers = {"flagged-llm-hog"}
        lane_harness.run_tick([hog], sync=False)
        assert lane_harness.wait_for_start("flagged-llm-hog", 5.0) is not None
        assert lane_harness.thread_names["flagged-llm-hog"].startswith("cron-critical"), (
            "SABOTAGE SETUP FAILED: the agent job did not reach the reserved lane")

        lane_harness.run_tick([_job("autosell", critical=True)], sync=False)
        assert lane_harness.wait_for_start("autosell", TICK_DISPATCH_SLACK_SECONDS) is None, (
            "SABOTAGE INEFFECTIVE: the script job ran anyway, so the no_agent "
            "requirement is not what protects the lane"
        )

    def test_blocker_harness_would_notice_a_job_that_never_runs(
        self, monkeypatch, lane_harness
    ):
        """Control: the harness reports None for a job that is never dispatched.

        Without this, a 'waited is None' sabotage assertion could pass simply
        because the harness is broken rather than because starvation occurred.
        """
        monkeypatch.setenv("HERMES_CRON_MAX_PARALLEL", "2")
        lane_harness.run_tick([_job("dispatched")], sync=True)
        assert lane_harness.wait_for_start("dispatched", 5.0) is not None
        assert lane_harness.wait_for_start("never-submitted", 0.3) is None

    def test_reserved_thread_name_assertion_has_power(self, monkeypatch, lane_harness):
        """Control: ordinary jobs really do produce non-'cron-critical' names.

        If every thread name started with the reserved prefix, the routing
        assertions would be vacuous.
        """
        monkeypatch.setenv("HERMES_CRON_MAX_PARALLEL", "2")
        monkeypatch.setenv("HERMES_CRON_CRITICAL_LANE_WORKERS", "1")

        lane_harness.run_tick(
            [_job("ordinary"), _job("critical-one", critical=True)], sync=True)
        assert not lane_harness.thread_names["ordinary"].startswith("cron-critical")
        assert lane_harness.thread_names["critical-one"].startswith("cron-critical")


@pytest.mark.parametrize("mutation", ["single_pool", "shared_executor", "admit_agents"])
def test_regression_assertions_reject_sabotaged_scheduler(monkeypatch, lane_harness, mutation):
    """Run the actual positive regression against a broken production seam."""
    import cron.scheduler as sched

    if mutation == "single_pool":
        monkeypatch.setattr(sched, "is_critical_job", lambda job: False)
    elif mutation == "shared_executor":
        shared = sched._get_parallel_pool(2)
        monkeypatch.setattr(sched, "_get_critical_pool", lambda width: shared)
    else:
        monkeypatch.setattr(sched, "is_critical_job", sched._critical_flag_is_set)

    if mutation == "admit_agents":
        regression = TestOnlyScriptJobsAreAdmitted().test_agent_job_cannot_evict_a_script_job_from_the_reserved_lane
        message = "critical SCRIPT job starved"
    else:
        regression = TestCriticalLaneNotStarved().test_critical_job_starts_while_general_pool_is_saturated
        message = "critical job never started"
    with pytest.raises(AssertionError, match=message):
        regression(monkeypatch, lane_harness)
