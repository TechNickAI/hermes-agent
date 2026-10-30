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
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
for candidate in (REPO, REPO / "plugins" / "platforms" / "telegram"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

adapter_mod = pytest.importorskip(
    "adapter",
    reason="Telegram adapter module not importable in this environment",
)

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


def test_retry_resets_http_pool_between_attempts():
    """A wedged keep-alive connection must not be reused by every retry (2026-10-14:
    three 60s timeouts in a row on a healthy network lost a voice message)."""
    src = _FlakySource(_real_readtimeout_chain(), fail_times=2)
    resets = []

    async def _reset():
        resets.append(src.calls)

    data, _ = _run(
        adapter_mod.TelegramAdapter._download_media_with_retry(
            src, what="voice message", attempts=3, base_delay=0.0, reset=_reset
        )
    )
    assert data == b"OGGDATA"
    assert resets == [1, 2]  # reset after each failed attempt, before the next


def _av_harness(fetch_result):
    """Run the real _cache_inbound_av with a bot download that always times out."""
    from types import SimpleNamespace

    adapter = object.__new__(adapter_mod.TelegramAdapter)
    surfaced = []
    fetched = []

    async def _fetch(msg, kind, ext):
        fetched.append((kind, ext))
        return fetch_result

    async def _surface(msg, event, label, exc):
        surfaced.append(label)

    async def _no_reset():
        return None

    adapter._telegram_media_size_allowed = lambda source, label: (True, None)
    adapter._fetch_media_via_user_account = _fetch
    adapter._surface_media_cache_failure = _surface
    adapter._drain_general_connections_after_pool_timeout = _no_reset
    adapter._download_media_with_retry = lambda source, **kw: adapter_mod.TelegramAdapter._download_media_with_retry(
        source, what=kw.get("what", "media"), attempts=2, base_delay=0.0)
    event = SimpleNamespace(media_urls=[], media_types=[], text="")
    src = _FlakySource(_real_readtimeout_chain(), fail_times=99)
    _run(adapter._cache_inbound_av(SimpleNamespace(), event, src, "voice message", "voice", ".ogg", "audio/ogg"))
    return event, surfaced, fetched


def test_bot_timeout_recovers_via_user_account_without_resend(tmp_path, monkeypatch):
    """2026-10-14: bot getFile stalled 3x60s on a healthy network and the memo was lost;
    the owner's own account fetched it in <1s. The fallback must deliver the audio and
    must NOT ask the human to resend."""
    event, surfaced, fetched = _av_harness(b"OGGFROMUSER")
    assert fetched == [("voice", ".ogg")]
    assert surfaced == []
    assert len(event.media_urls) == 1 and Path(event.media_urls[0]).read_bytes() == b"OGGFROMUSER"


def test_fallback_failure_still_asks_for_resend():
    event, surfaced, fetched = _av_harness(None)
    assert fetched and surfaced == ["voice message"]
    assert event.media_urls == []


def test_stalled_bot_download_switches_to_user_account_fast():
    """2026-10-14 afternoon: getFile for this bot hung (504 after 120s) on every memo while the
    owner's account was instant. A stalled bot download must hand off after one short try,
    not after 3x60s of retries."""
    import asyncio as _aio
    import time as _time
    from types import SimpleNamespace

    adapter = object.__new__(adapter_mod.TelegramAdapter)
    adapter._BOT_AV_FIRST_TRY_TIMEOUT = 0.2
    fetched, surfaced = [], []

    class _Hang:
        async def get_file(self):
            await _aio.sleep(30)

    async def _fetch(msg, kind, ext):
        fetched.append(kind)
        return b"OGGFAST"

    async def _surface(msg, event, label, exc):
        surfaced.append(label)

    adapter._telegram_media_size_allowed = lambda source, label: (True, None)
    adapter._fetch_media_via_user_account = _fetch
    adapter._surface_media_cache_failure = _surface
    event = SimpleNamespace(media_urls=[], media_types=[], text="")
    t0 = _time.monotonic()
    _run(adapter._cache_inbound_av(SimpleNamespace(), event, _Hang(), "voice message", "voice", ".ogg", "audio/ogg"))
    assert _time.monotonic() - t0 < 5
    assert fetched == ["voice"] and surfaced == []
    assert Path(event.media_urls[0]).read_bytes() == b"OGGFAST"
