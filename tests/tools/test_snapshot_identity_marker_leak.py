"""Identity markers must never persist into the shared bash session snapshot.

``tools/environments/base.py`` keeps a per-session snapshot file that every
command ``source``s and then re-dumps ``export -p`` into. A ``delegate_task``
child shares the parent's terminal environment (``_resolve_container_task_id``
deliberately collapses child task_ids onto the parent's environment), so any
identity marker that reaches ``export -p`` becomes STICKY for the life of the
shell — and it leaks in BOTH directions:

* parent <- child: ``HERMES_DELEGATED_CHILD_CONTEXT=1`` set for the child
  outlives it, and ``_is_delegated_child_cli_mutation`` /
  ``_assert_not_delegated_child_mutation`` then refuse every later kanban
  mutation from that shell, including from scheduled jobs that are not
  delegated children at all.
* child <- parent: the child re-inherits the parent's ``HERMES_KANBAN_TASK``,
  defeating the process-env scrub in ``_scrub_delegated_child_kanban_env`` /
  ``scrub_kanban_env`` and letting a child mutate the PARENT's card. This is
  the more dangerous direction, so a one-directional test is not enough.

The third test is the two-sided guard: the snapshot EXISTS to carry ordinary
exported variables across commands, so a blanket wipe would "fix" the leak by
silently breaking ``export FOO=bar`` continuity.
"""

import os
import re
import sys

import pytest

from agent.delegation_context import (
    DELEGATED_CHILD_ENV_MARKER,
    KANBAN_ENV_KEYS,
    delegated_child_context,
)
from tools.environments.base import (
    _SNAPSHOT_EXCLUDED_ENV_REGEX,
    _export_dump_excluding_session_vars,
)

_MARKERS = (DELEGATED_CHILD_ENV_MARKER, *KANBAN_ENV_KEYS)


# ---------------------------------------------------------------------------
# Unit: the Python-side contract (regex) and the emitted shell both cover them.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", _MARKERS)
def test_regex_excludes_identity_markers(name):
    rx = re.compile(_SNAPSHOT_EXCLUDED_ENV_REGEX)
    assert rx.search(f'declare -x {name}="value"'), (
        f"{name} must be excluded from the snapshot"
    )


def test_regex_still_admits_ordinary_exports():
    """The exclusion must be targeted, not a blanket HERMES_/every-var wipe.

    The identity names match EXACTLY: the dump unsets them by exact name, so a
    regex matching a longer lookalike would advertise an exclusion that the
    shell never performs.
    """
    rx = re.compile(_SNAPSHOT_EXCLUDED_ENV_REGEX)
    for name in (
        "FOO",
        "PATH",
        "HERMES_HOME",
        "HERMES_KANBAN_DB_PATH",
        "HERMES_KANBAN_TASKS",
        "HERMES_KANBAN_HOME",
    ):
        assert not rx.search(f'declare -x {name}="value"'), (
            f"{name} must still persist in the snapshot"
        )


@pytest.mark.parametrize("name", _MARKERS)
def test_dump_snippet_unsets_identity_markers_without_caller_help(name):
    """The dump excludes them unconditionally — callers pass no extra names.

    A caller that forgot to pass them would silently restore the leak, so the
    exclusion cannot depend on the call site.
    """
    snippet = _export_dump_excluding_session_vars('"$__hermes_snap_tmp"')
    assert re.search(rf"\bunset\b[^;]*\b{re.escape(name)}\b", snippet), (
        f"{name} must be unset before export -p"
    )
    # Unset-by-name, never a line-based filter (bash 3.2 multi-line values).
    assert "grep -vE" not in snippet


# ---------------------------------------------------------------------------
# Integration: the real LocalEnvironment and its real snapshot file.
# ---------------------------------------------------------------------------

def _run(env, cmd):
    result = env.execute(cmd, timeout=30)
    return (result.get("output") or "").strip()


@pytest.fixture
def local_env(tmp_path):
    from tools.environments.local import LocalEnvironment

    env = LocalEnvironment(cwd=str(tmp_path), timeout=30)
    assert env._snapshot_ready, "snapshot must be active or this proves nothing"
    try:
        yield env
    finally:
        env.cleanup()


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX bash snapshot path")
def test_child_marker_does_not_leak_into_later_parent_commands(local_env):
    """Direction 1: parent <- child."""
    assert _run(local_env, f'echo "[${{{DELEGATED_CHILD_ENV_MARKER}:-unset}}]"') == "[unset]"

    with delegated_child_context("child-session"):
        seen_by_child = _run(
            local_env, f'echo "[${{{DELEGATED_CHILD_ENV_MARKER}:-unset}}]"'
        )
    # The child must still RECEIVE the marker via the process env; only its
    # PERSISTENCE is the bug.
    assert seen_by_child == "[1]", f"child lost its own marker: {seen_by_child}"

    after = _run(local_env, f'echo "[${{{DELEGATED_CHILD_ENV_MARKER}:-unset}}]"')
    assert after == "[unset]", f"child marker leaked into the parent shell: {after}"

    with open(local_env._snapshot_path) as fh:
        assert DELEGATED_CHILD_ENV_MARKER not in fh.read()


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX bash snapshot path")
def test_parent_kanban_task_does_not_leak_into_a_child_command(
    local_env, monkeypatch
):
    """Direction 2 (the dangerous one): child <- parent."""
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_PARENT_CARD")

    # Warm the snapshot as the parent, so the parent's value is what export -p
    # would have captured.
    assert _run(local_env, 'echo "[${HERMES_KANBAN_TASK:-unset}]"') == "[t_PARENT_CARD]"

    with delegated_child_context("child-session"):
        seen_by_child = _run(local_env, 'echo "[${HERMES_KANBAN_TASK:-unset}]"')

    assert seen_by_child == "[unset]", (
        f"child regained the parent's card via the snapshot: {seen_by_child}"
    )

    with open(local_env._snapshot_path) as fh:
        assert "HERMES_KANBAN_TASK" not in fh.read()


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX bash snapshot path")
def test_a_real_worker_still_sees_its_own_kanban_env_on_every_command(
    local_env, monkeypatch
):
    """Scrubbing the snapshot must not disarm a legitimate dispatcher worker.

    The values reach each command through the process env, so a genuine worker
    keeps seeing its own card on command 1, 2 and 3.
    """
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_MY_OWN_CARD")
    monkeypatch.setenv("HERMES_KANBAN_WORKSPACE", "/tmp/my-workspace")

    for attempt in range(3):
        assert _run(local_env, 'echo "[${HERMES_KANBAN_TASK:-unset}]"') == (
            "[t_MY_OWN_CARD]"
        ), f"worker lost its own task on command {attempt}"
        assert _run(local_env, 'echo "[${HERMES_KANBAN_WORKSPACE:-unset}]"') == (
            "[/tmp/my-workspace]"
        ), f"worker lost its own workspace on command {attempt}"


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX bash snapshot path")
def test_ordinary_exports_still_persist_across_commands(local_env):
    """Two-sided guard: the snapshot must keep doing its actual job."""
    _run(local_env, "export SNAPSHOT_GUARD_VAR=persisted")
    assert _run(local_env, 'echo "[${SNAPSHOT_GUARD_VAR:-unset}]"') == "[persisted]"
    # And across a further command, so this is persistence and not one-shot env.
    _run(local_env, "true")
    assert _run(local_env, 'echo "[${SNAPSHOT_GUARD_VAR:-unset}]"') == "[persisted]"

    with open(local_env._snapshot_path) as fh:
        assert "SNAPSHOT_GUARD_VAR" in fh.read(), (
            "the snapshot stopped carrying ordinary exports"
        )


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX bash snapshot path")
def test_no_identity_marker_reaches_the_snapshot_file(local_env, monkeypatch):
    """Whole-family sweep, so a new KANBAN_ENV_KEYS entry is covered."""
    for index, name in enumerate(_MARKERS):
        monkeypatch.setenv(name, f"leaked-value-{index}")

    _run(local_env, "true")

    with open(local_env._snapshot_path) as fh:
        snapshot = fh.read()
    for name in _MARKERS:
        assert name not in snapshot, f"{name} persisted into the snapshot"
    assert "leaked-value-" not in snapshot
    assert os.path.exists(local_env._snapshot_path)
