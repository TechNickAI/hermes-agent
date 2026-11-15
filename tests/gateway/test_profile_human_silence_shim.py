"""Fleet shim: profile-wide allow_human_silence_markers is the room default for upstream allow_silence."""

import pytest

from gateway.config import GatewayConfig
from gateway.response_filters import silence_allowed
from tests.gateway.test_gateway_silence_tokens import _runner, _source

HEALTHY = {"completed": True}


def _cfg(profile_opt_in, room=None):
    raw = {"allow_human_silence_markers": profile_opt_in,
           "platforms": {"telegram": {"channel_overrides": {}}}}
    if room is not None:
        raw["platforms"]["telegram"]["channel_overrides"]["-1001"] = {"allow_silence": room}
    return GatewayConfig.from_dict(raw)


@pytest.mark.parametrize("profile_opt_in,room,expected", [
    (True, None, True),      # profile opt-in alone silences a healthy human turn
    (False, None, False),    # default stays loud
    (True, False, False),    # explicit room false beats the profile opt-in
    (False, True, True),     # upstream room opt-in still works on its own
])
def test_profile_default_feeds_room_silence(profile_opt_in, room, expected, monkeypatch, tmp_path):
    runner = _runner(monkeypatch, tmp_path)
    runner.config = _cfg(profile_opt_in, room)
    src = _source()
    src.chat_id = "-1001"
    assert runner._silence_allowed_for_source(src, "user", True, HEALTHY) is expected


def test_profile_opt_in_never_hides_failed_turn(monkeypatch, tmp_path):
    runner = _runner(monkeypatch, tmp_path)
    runner.config = _cfg(True)
    src = _source()
    src.chat_id = "-1001"
    assert runner._silence_allowed_for_source(src, "user", True, {"failed": True}) is False


def test_key_round_trips_and_defaults_off():
    assert GatewayConfig.from_dict({}).allow_human_silence_markers is False
    cfg = GatewayConfig.from_dict({"allow_human_silence_markers": True})
    assert cfg.allow_human_silence_markers is True
    assert cfg.to_dict()["allow_human_silence_markers"] is True
    assert silence_allowed("user", True) is False
