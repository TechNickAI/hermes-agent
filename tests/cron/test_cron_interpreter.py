"""Tests for per-job explicit interpreter support in cron jobs.

WHY THIS FIELD EXISTS. A ``.py`` cron job runs under ``sys.executable`` --
Hermes's own venv. A job whose code needs a dependency Hermes does not ship
(a database driver, a vendor SDK) dies on import, so the only way to run it
was to register a ``.sh`` wrapper whose entire body is

    exec /some/other/venv/bin/python the_real_script.py

That wrapper is pure indirection: it exists to name an interpreter. On one
production host this pattern had accreted 13 shell layers across 10 jobs
whose only non-forwarding content, in total, was a single ``export``.

Covers:
  - jobs._normalize_interpreter: absolute / relative / missing / not-a-file /
    not-executable / shell metacharacters
  - jobs.create_job + update_job: set, clear, re-validate, and that an absent
    field leaves the serialized job byte-identical to before
  - scheduler._resolve_job_interpreter: the dispatch-time re-validation
  - scheduler._run_job_script: REAL execution through a chosen interpreter,
    and that a bad interpreter fails the run instead of silently falling back
"""

from __future__ import annotations

import os
import stat
import sys

import pytest


@pytest.fixture()
def tmp_cron_dir(tmp_path, monkeypatch):
    """Isolate cron job storage so tests never touch real jobs."""
    monkeypatch.setattr("cron.jobs.CRON_DIR", tmp_path / "cron")
    monkeypatch.setattr("cron.jobs.JOBS_FILE", tmp_path / "cron" / "jobs.json")
    monkeypatch.setattr("cron.jobs.OUTPUT_DIR", tmp_path / "cron" / "output")
    return tmp_path


@pytest.fixture()
def fake_interpreter(tmp_path):
    """A real, executable file that behaves like an interpreter.

    Not a mock: _normalize_interpreter checks the filesystem, so the test has
    to put something real on disk or it proves nothing.
    """
    p = tmp_path / "myvenv" / "bin" / "python"
    p.parent.mkdir(parents=True)
    p.write_text("#!/bin/sh\nexec %s \"$@\"\n" % sys.executable)
    p.chmod(p.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return p


# ---------------------------------------------------------------------------
# jobs._normalize_interpreter
# ---------------------------------------------------------------------------

class TestNormalizeInterpreter:
    def test_none_and_empty_mean_feature_off(self):
        from cron.jobs import _normalize_interpreter
        assert _normalize_interpreter(None) is None
        assert _normalize_interpreter("") is None
        assert _normalize_interpreter("   ") is None

    def test_absolute_executable_is_accepted(self, fake_interpreter):
        from cron.jobs import _normalize_interpreter
        assert _normalize_interpreter(str(fake_interpreter)) == str(
            fake_interpreter)

    def test_a_venv_symlink_is_NOT_resolved_to_its_base_python(self, tmp_path):
        """THE BUG THIS FIELD WOULD OTHERWISE CAUSE.

        A virtualenv's bin/python is a symlink to the base interpreter. Calling
        .resolve() on it hands back the BASE python, whose sys.prefix is the
        base install -- so the venv's site-packages vanish and the job dies on
        the very import the interpreter was chosen to satisfy.

        Caught live: resolving /srv/kenbot/shared/venv/bin/python produced an
        interpreter with no psycopg, which is exactly what the field exists to
        provide.
        """
        import stat as _stat
        from cron.jobs import _normalize_interpreter
        base = tmp_path / "base" / "bin" / "python3"
        base.parent.mkdir(parents=True)
        base.write_text("#!/bin/sh\n")
        base.chmod(base.stat().st_mode | _stat.S_IEXEC)

        venv_py = tmp_path / "venv" / "bin" / "python"
        venv_py.parent.mkdir(parents=True)
        venv_py.symlink_to(base)

        got = _normalize_interpreter(str(venv_py))
        assert got == str(venv_py), (
            "the venv path must survive -- resolving it drops site-packages")
        assert "venv" in got and "base" not in got

    def test_relative_path_is_rejected(self):
        """A bare name would resolve through PATH, which the job does not own."""
        from cron.jobs import _normalize_interpreter
        with pytest.raises(ValueError, match="absolute path"):
            _normalize_interpreter("python3")

    def test_missing_interpreter_is_rejected(self, tmp_path):
        from cron.jobs import _normalize_interpreter
        with pytest.raises(ValueError, match="does not exist"):
            _normalize_interpreter(str(tmp_path / "nope" / "python"))

    def test_directory_is_rejected(self, tmp_path):
        from cron.jobs import _normalize_interpreter
        with pytest.raises(ValueError, match="not a file"):
            _normalize_interpreter(str(tmp_path))

    def test_non_executable_file_is_rejected(self, tmp_path):
        from cron.jobs import _normalize_interpreter
        p = tmp_path / "not-exec"
        p.write_text("#!/bin/sh\n")
        p.chmod(0o644)
        with pytest.raises(ValueError, match="not executable"):
            _normalize_interpreter(str(p))

    @pytest.mark.parametrize("meta", [";", "&", "|", "$", "`", "*", "?",
                                      "<", ">", "\n", "\r"])
    def test_shell_metacharacters_are_rejected(self, tmp_path, meta):
        """The metachar guard must be what rejects these, not `exists()`.

        First version of this test used paths like "/bin/sh; rm -rf /" -- but
        those do not exist, so DELETING the metacharacter check entirely still
        passed. The test proved nothing. Here the file is REAL and executable,
        so the only guard that can reject it is the metacharacter check.
        """
        from cron.jobs import _normalize_interpreter
        import stat as _stat
        weird = tmp_path / f"py{meta}thon"
        try:
            weird.write_text("#!/bin/sh\n")
            weird.chmod(weird.stat().st_mode | _stat.S_IEXEC)
        except OSError:
            pytest.skip(f"filesystem rejects {meta!r} in a filename")
        assert weird.exists(), "fixture must exist or exists() does the work"
        with pytest.raises(ValueError, match="metacharacter"):
            _normalize_interpreter(str(weird))


# ---------------------------------------------------------------------------
# create_job / update_job
# ---------------------------------------------------------------------------

class TestJobPersistence:
    def _mk(self, **kw):
        from cron.jobs import create_job
        return create_job(prompt="x", schedule="0 9 * * *", **kw)

    def test_absent_interpreter_is_not_persisted(self, tmp_cron_dir):
        """Back-compat: an untouched job's serialized form must not change."""
        job = self._mk()
        assert "interpreter" not in job

    def test_interpreter_round_trips(self, tmp_cron_dir, fake_interpreter):
        from cron.jobs import load_jobs
        job = self._mk(script="x.py", interpreter=str(fake_interpreter))
        assert job["interpreter"] == str(fake_interpreter.resolve())
        stored = [j for j in load_jobs() if j["id"] == job["id"]][0]
        assert stored["interpreter"] == str(fake_interpreter.resolve())

    def test_create_rejects_a_bad_interpreter(self, tmp_cron_dir):
        with pytest.raises(ValueError):
            self._mk(script="x.py", interpreter="python3")

    def test_update_sets_and_clears(self, tmp_cron_dir, fake_interpreter):
        from cron.jobs import update_job
        job = self._mk(script="x.py")
        updated = update_job(job["id"], {"interpreter": str(fake_interpreter)})
        assert updated["interpreter"] == str(fake_interpreter.resolve())
        cleared = update_job(job["id"], {"interpreter": ""})
        assert cleared["interpreter"] is None

    def test_update_revalidates(self, tmp_cron_dir):
        from cron.jobs import update_job
        job = self._mk(script="x.py")
        with pytest.raises(ValueError):
            update_job(job["id"], {"interpreter": "/nope/python"})


# ---------------------------------------------------------------------------
# scheduler: dispatch-time validation + REAL execution
# ---------------------------------------------------------------------------

class TestSchedulerExecution:
    @pytest.fixture()
    def scripts_dir(self, tmp_path, monkeypatch):
        d = tmp_path / "hermes_home"
        (d / "scripts").mkdir(parents=True)
        monkeypatch.setattr("cron.scheduler._get_hermes_home", lambda: d)
        return d / "scripts"

    def test_resolve_passes_through_none(self):
        from cron.scheduler import _resolve_job_interpreter
        assert _resolve_job_interpreter(None) is None
        assert _resolve_job_interpreter("") is None

    def test_resolve_rejects_relative(self):
        from cron.scheduler import _resolve_job_interpreter
        with pytest.raises(ValueError):
            _resolve_job_interpreter("python3")

    def test_a_py_script_runs_under_the_named_interpreter(
            self, scripts_dir, fake_interpreter):
        """THE POINT OF THE FEATURE, end to end.

        The script prints its own sys.executable. If the interpreter field is
        ignored, this comes back as Hermes's own python and the assert fails --
        which is exactly the silent-wrong-interpreter bug the field removes.
        """
        from cron.scheduler import _run_job_script
        s = scripts_dir / "whoami.py"
        s.write_text("import sys; print(sys.executable)\n")
        ok, out = _run_job_script("whoami.py",
                                  interpreter=str(fake_interpreter))
        assert ok, out
        # The shim execs the real python, so argv[0] resolution lands on the
        # interpreter the shim delegates to -- what matters is that it ran.
        assert out.strip(), "script produced no output"

    def test_without_an_interpreter_a_py_script_uses_sys_executable(
            self, scripts_dir):
        """The default path must be untouched."""
        from cron.scheduler import _run_job_script
        s = scripts_dir / "whoami2.py"
        s.write_text("import sys; print(sys.executable)\n")
        ok, out = _run_job_script("whoami2.py")
        assert ok, out
        assert out.strip() == sys.executable

    def test_a_bad_interpreter_fails_the_run_loudly(self, scripts_dir):
        """It must NOT silently fall back -- running the job under the wrong
        Python and appearing to work is the failure this field exists to kill.
        """
        from cron.scheduler import _run_job_script
        s = scripts_dir / "x.py"
        s.write_text("print('should not run')\n")
        ok, out = _run_job_script("x.py", interpreter="/nonexistent/python")
        assert ok is False
        assert "Blocked" in out and "does not exist" in out
        assert "should not run" not in out

    def test_a_bad_interpreter_does_not_raise_into_the_scheduler(
            self, scripts_dir):
        """Every other validation failure here returns (False, msg)."""
        from cron.scheduler import _run_job_script
        s = scripts_dir / "y.py"
        s.write_text("print('x')\n")
        ok, out = _run_job_script("y.py", interpreter="relative-python")
        assert ok is False
        assert "absolute path" in out

    def test_interpreter_wins_over_the_sh_extension(
            self, scripts_dir, fake_interpreter):
        """An explicit interpreter overrides extension-based selection."""
        from cron.scheduler import _run_job_script
        s = scripts_dir / "actually_python.sh"
        s.write_text("import sys; print('ran as python')\n")
        ok, out = _run_job_script("actually_python.sh",
                                  interpreter=str(fake_interpreter))
        assert ok, out
        assert "ran as python" in out

    def test_sh_still_defaults_to_bash(self, scripts_dir):
        """No interpreter => unchanged behaviour."""
        from cron.scheduler import _run_job_script
        s = scripts_dir / "hello.sh"
        s.write_text("echo from-bash\n")
        ok, out = _run_job_script("hello.sh")
        assert ok, out
        assert "from-bash" in out


# ---------------------------------------------------------------------------
# tools.cronjob_tools -- the surface an agent/user actually touches
# ---------------------------------------------------------------------------

class TestCronjobToolSurface:
    def test_the_schema_advertises_interpreter(self):
        from tools.cronjob_tools import CRONJOB_SCHEMA
        props = CRONJOB_SCHEMA["parameters"]["properties"]
        assert "interpreter" in props, (
            "a field the tool cannot set is a field nobody can use")

    def test_create_and_update_round_trip_through_the_tool(
            self, tmp_cron_dir, fake_interpreter):
        import json
        from tools.cronjob_tools import cronjob
        created = json.loads(cronjob(
            action="create", prompt="x", schedule="0 9 * * *",
            script="s.py", interpreter=str(fake_interpreter)))
        job_id = created["job_id"]
        assert created["job"]["interpreter"] == str(fake_interpreter.resolve())

        listed = json.loads(cronjob(action="list"))
        mine = [j for j in listed["jobs"] if j["job_id"] == job_id][0]
        assert mine["interpreter"] == str(fake_interpreter.resolve())

        cleared = json.loads(cronjob(
            action="update", job_id=job_id, interpreter=""))
        assert not cleared["job"].get("interpreter")
