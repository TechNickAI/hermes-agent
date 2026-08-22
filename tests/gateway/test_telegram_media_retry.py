"""Regression tests for transient Telegram media-download retries.

Context (2026-08-12 weekly error review): user media downloads called
``get_file()`` + ``download_as_bytearray()`` with ZERO retries. A single
transient ``httpcore.ReadTimeout`` on the response headers aborted the whole
download and surfaced to the user as "could not be downloaded ... please try
sending it again" — pushing the retry onto the human. Observed across
2026-08-06/07: 5 voice messages and 4 photos each cost a manual resend.

Oracle note: these tests assert the OBSERVABLE outcome the user experiences
(did the bytes arrive? did the human get asked to resend?), and drive the real
``_is_transient_media_error`` classifier with real exception objects — including
the exact ``telegram.error.TimedOut``-wrapping-``httpcore.ReadTimeout`` shape
seen in logs/errors.log. They do not re-implement the classifier's own logic.
"""

import asyncio
import types
from pathlib import Path

import pytest
from unittest import mock

from tests.gateway._plugin_adapter_loader import load_plugin_adapter

# Load via the shared helper, never `sys.path` + `import adapter`: every plugin
# ships its own adapter.py, so a bare import races for sys.modules["adapter"]
# between xdist workers. The conftest guard rejects the anti-pattern.
adapter_mod = load_plugin_adapter("telegram")

_is_transient = adapter_mod._is_transient_media_error


class _FakeFile:
    def __init__(self, data: bytes, file_path: str = "voice/file_0.ogg"):
        self._data = data
        self.file_path = file_path

    async def download_as_bytearray(self):
        return bytearray(self._data)


class _FlakySource:
    """Fails with `exc` for the first `fail_times` calls, then succeeds."""

    def __init__(self, exc: BaseException, fail_times: int, data: bytes = b"OGGDATA"):
        self._exc = exc
        self._fail_times = fail_times
        self._data = data
        self.calls = 0

    async def get_file(self):
        self.calls += 1
        if self.calls <= self._fail_times:
            raise self._exc
        return _FakeFile(self._data)


def _real_readtimeout_chain() -> BaseException:
    """Reproduce the exact exception shape from logs/errors.log.

    PTB collapses httpcore.ReadTimeout into telegram.error.TimedOut("Timed out"),
    so the transient signal lives in the message and/or the __cause__ chain.
    """
    try:
        import httpcore

        inner: BaseException = httpcore.ReadTimeout()
    except Exception:  # pragma: no cover - httpcore always present via httpx
        inner = TimeoutError("timed out")
    try:
        from telegram.error import TimedOut

        outer: BaseException = TimedOut()
    except Exception:  # pragma: no cover
        outer = TimeoutError("Timed out")
    outer.__cause__ = inner
    return outer


def _run(coro):
    return asyncio.get_event_loop_policy().new_event_loop().run_until_complete(coro)


# --- classifier -----------------------------------------------------------


def test_real_logged_readtimeout_is_transient():
    """The exact failure that cost Julianna 9 resends must be retryable."""
    assert _is_transient(_real_readtimeout_chain()) is True


def test_permanent_errors_are_not_retried():
    """Real errors must still reach the user immediately, not be masked."""
    from telegram.error import BadRequest, Forbidden

    assert _is_transient(BadRequest("file is too big")) is False
    assert _is_transient(Forbidden("bot was blocked by the user")) is False


def test_cycle_in_exception_chain_terminates():
    a = RuntimeError("a")
    b = RuntimeError("b")
    a.__cause__ = b
    b.__cause__ = a
    assert _is_transient(a) is False  # must return, not hang


# --- retry behavior -------------------------------------------------------


def test_transient_failure_recovers_without_user_resend():
    """Two timeouts then success => bytes delivered, human never asked."""
    src = _FlakySource(_real_readtimeout_chain(), fail_times=2)
    data, file_obj = _run(
        adapter_mod.TelegramAdapter._download_media_with_retry(
            src, what="voice message", attempts=3, base_delay=0.0
        )
    )
    assert data == b"OGGDATA"
    assert file_obj.file_path == "voice/file_0.ogg"
    assert src.calls == 3


def test_permanent_failure_raises_on_first_attempt():
    """No pointless backoff on an error retrying cannot fix."""
    from telegram.error import BadRequest

    src = _FlakySource(BadRequest("file is too big"), fail_times=99)
    with pytest.raises(BadRequest):
        _run(
            adapter_mod.TelegramAdapter._download_media_with_retry(
                src, what="document", attempts=3, base_delay=0.0
            )
        )
    assert src.calls == 1


def test_default_attempts_retry_without_an_explicit_argument():
    """Production call sites pass no ``attempts``, so the DEFAULT must retry.

    Every other test here pins ``attempts=3`` explicitly, which would keep
    passing if the default silently regressed to 1 — and the default is the
    only value the real download sites ever use.
    """
    src = _FlakySource(_real_readtimeout_chain(), fail_times=2)
    data, _ = _run(
        adapter_mod.TelegramAdapter._download_media_with_retry(
            src, what="voice message", base_delay=0.0
        )
    )
    assert data == b"OGGDATA"
    assert src.calls == 3


def test_exhausted_retries_still_surface_to_user():
    """When it truly is down, the original error propagates (user IS told)."""
    exc = _real_readtimeout_chain()
    src = _FlakySource(exc, fail_times=99)
    with pytest.raises(Exception) as caught:
        _run(
            adapter_mod.TelegramAdapter._download_media_with_retry(
                src, what="photo", attempts=3, base_delay=0.0
            )
        )
    assert caught.value is exc
    assert src.calls == 3


def test_permanent_error_wins_regardless_of_chain_position():
    """A permanent failure nested under a timeout must not be retried.

    PTB wraps freely, so BadRequest("file is too big") can surface as the
    __cause__ of a TimedOut. Classifying on the first transient-looking node
    made the verdict depend on nesting order: this pair was retried three
    times for a file that can never download, while the same pair nested the
    other way raised at once.
    """
    from telegram.error import BadRequest, TimedOut

    # The outer node must be independently transient, or this test passes
    # without ever exercising precedence. PTB stamps "Timed out" by default;
    # the suite's telegram mock does not, so state it explicitly.
    outer = TimedOut("Timed out")
    assert _is_transient(TimedOut("Timed out")) is True, "fixture must be transient alone"
    outer.__cause__ = BadRequest("file is too big")
    assert _is_transient(outer) is False

    inner = BadRequest("file is too big")
    inner.__cause__ = TimedOut("Timed out")
    assert _is_transient(inner) is False


def test_permanent_on_the_context_branch_is_still_found():
    """A node can have BOTH __cause__ and __context__; walk both.

    A single-path walk that follows `__cause__ or __context__` never sees the
    context branch when a cause is set, so a permanent error hiding there was
    classified transient.
    """
    from telegram.error import BadRequest, TimedOut

    exc = TimedOut("Timed out")
    exc.__cause__ = TimedOut("Timed out")
    exc.__context__ = BadRequest("file is too big")
    assert _is_transient(exc) is False


def test_neutral_wrapper_requires_walking_to_the_inner_cause():
    """The classifier must read the CHAIN, not just the outermost exception.

    The other fixtures use telegram.error.TimedOut, which is independently
    transient by both class name and message -- so they would pass even if
    chain traversal were removed. This wrapper is neutral on both counts.
    """
    import httpcore

    class _NeutralWrapper(Exception):
        pass

    exc = _NeutralWrapper("upload pipeline failed")
    exc.__cause__ = httpcore.ReadTimeout()
    assert _is_transient(exc) is True


def test_backoff_is_exponential_and_not_merely_a_retry_count():
    """Pin the schedule, not just the attempt count.

    Every other retry test passes base_delay=0.0, so replacing the sleep with
    asyncio.sleep(0) would leave the suite green while backoff silently
    vanished.
    """
    slept: list = []

    async def _fake_sleep(delay):
        slept.append(delay)

    src = _FlakySource(_real_readtimeout_chain(), fail_times=2)
    with mock.patch.object(adapter_mod.asyncio, "sleep", _fake_sleep):
        data, _ = _run(
            adapter_mod.TelegramAdapter._download_media_with_retry(
                src, what="photo", attempts=3
            )
        )
    assert data == b"OGGDATA"
    assert slept == [1.0, 2.0]


def test_exhausted_photo_retry_reaches_the_user_surface():
    """The end-to-end contract: after retries fail, the human IS told.

    The helper-level test only proves the exception propagates. This drives
    _handle_media_message itself and asserts the two user-visible effects --
    the Telegram reply and the agent-visible note -- so that unwiring
    _surface_media_cache_failure from the photo branch fails the suite.
    """
    exc = _real_readtimeout_chain()
    photo = _FlakySource(exc, fail_times=99)

    replies: list = []
    notes: list = []

    class _Msg:
        caption = None
        sticker = None
        voice = audio = video = document = None
        media_group_id = None

        def __init__(self):
            self.photo = [photo]

        async def reply_text(self, text, **kw):
            replies.append(text)

    msg = _Msg()
    event = types.SimpleNamespace(text="", media_urls=[], media_types=[])

    adapter = object.__new__(adapter_mod.TelegramAdapter)
    adapter._bot = None          # identity learning is a no-op without a bot

    async def _surface(m, ev, kind, e, display_name=None):
        notes.append((kind, e))
        ev.text = (ev.text or "") + f"[{kind} failed]"
        await m.reply_text(f"Couldn't download your {kind}.")

    adapter._surface_media_cache_failure = _surface
    adapter._media_message_type = lambda m: "photo"
    adapter._build_message_event = lambda *a, **k: event
    adapter._clean_bot_trigger_text = lambda t: t
    adapter._apply_telegram_group_observe_attribution = lambda ev: ev
    adapter.handle_message = lambda ev: asyncio.sleep(0)

    update = types.SimpleNamespace(message=msg, update_id=1)
    _run(adapter._handle_media_message(update, None))

    assert photo.calls == 3, "expected the photo branch to retry via the helper"
    assert notes and notes[0][0] == "photo", "user surface was not invoked"
    assert replies, "the human was never told"
    assert "[photo failed]" in event.text, "agent-visible note missing"


def test_all_download_sites_use_the_retry_helper():
    """Guard the bug CLASS: no raw get_file() download outside the helper."""
    source = Path(adapter_mod.__file__).read_text()
    raw = [
        ln.strip()
        for ln in source.splitlines()
        if ".download_as_bytearray()" in ln
    ]
    # Exactly one raw download call may exist: the one inside the helper.
    assert len(raw) == 1, f"unretried media download site(s) reintroduced: {raw}"
