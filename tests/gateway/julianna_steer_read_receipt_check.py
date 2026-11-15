"""Steered (mid-turn) Telegram messages get the 👍 read receipt. Stdlib only; run with the gateway venv python.
Red/green: HERMES_AGENT_ROOT may point at a scratch copy holding the unpatched run_busy.py."""
import asyncio, os, sys
from types import SimpleNamespace
from unittest.mock import AsyncMock
root = os.environ.get("HERMES_AGENT_ROOT", os.path.expanduser("~/.hermes/hermes-agent"))
sys.path.insert(0, root)
from gateway.config import Platform, PlatformConfig
from gateway.platforms.event import MessageEvent, MessageType
from gateway.session import SessionSource
import gateway.run_busy as rb
from gateway.run import _AGENT_PENDING_SENTINEL  # noqa: F401  (import used by the method)
from plugins.platforms.telegram.adapter import TelegramAdapter

def make_adapter():
    a = object.__new__(TelegramAdapter)
    a.platform = Platform.TELEGRAM
    a.config = PlatformConfig(enabled=True, token="fake", extra={"reactions": True})
    a._bot = AsyncMock(); a._bot.set_message_reaction = AsyncMock()
    return a

def make_event(text="Builder warranty", mid="604"):
    return MessageEvent(text=text, message_type=MessageType.TEXT, message_id=mid,
        source=SessionSource(platform=Platform.TELEGRAM, chat_id="-1004415515814", chat_type="group",
                             user_id="42", user_name="Julianna Scruggs", thread_id="602"))

from gateway.run import GatewayRunner
class Runner(rb.GatewayBusySessionMixin):
    _BusySteerOutcome = GatewayRunner._BusySteerOutcome
    def __init__(self, adapter, steer_ok=True):
        self._adapter = adapter; self._steer_ok = steer_ok
    def _delivery_adapter_for(self, source): return self._adapter
    def _agent_has_active_subagents(self, a): return False
    async def _session_has_compression_in_flight(self, k): return False
    async def _prepare_busy_steer_text(self, e): return e.text
    def _try_agent_verb(self, agent, verb, text, key, event=None): return self._steer_ok
    def _fold_into_running_turn(self, agent, key, event): return None

def reactions(adapter):
    return [c.kwargs.get("reaction") for c in adapter._bot.set_message_reaction.call_args_list]

async def main():
    fails = 0
    agent = SimpleNamespace(steer=lambda t: True)
    # 1. steered message is marked 👍 (literal production case: msg 604 "Builder warranty", 11-07 14:39)
    a = make_adapter(); out = await Runner(a)._resolve_busy_steer_or_redirect(make_event(), "k", "steer", agent)
    ok = out.steered and reactions(a) == ["\U0001f44d"]
    print("PASS" if ok else "FAIL", "steered message gets thumbs-up", reactions(a)); fails += not ok
    # 2. steer that falls back to queue gets no reaction here (the queued turn's own hooks will mark it)
    a = make_adapter(); out = await Runner(a, steer_ok=False)._resolve_busy_steer_or_redirect(make_event(), "k", "steer", agent)
    ok = (not out.steered) and out.effective_mode == "queue" and reactions(a) == []
    print("PASS" if ok else "FAIL", "queued fallback not double-marked", reactions(a)); fails += not ok
    # 3. reactions disabled -> nothing sent
    a = make_adapter(); a.config = PlatformConfig(enabled=True, token="fake", extra={"reactions": False})
    os.environ.pop("TELEGRAM_REACTIONS", None)
    await Runner(a)._resolve_busy_steer_or_redirect(make_event(), "k", "steer", agent)
    ok = reactions(a) == []
    print("PASS" if ok else "FAIL", "respects reactions off", reactions(a)); fails += not ok
    # 4. API error does not break the steer
    a = make_adapter(); a._bot.set_message_reaction.side_effect = RuntimeError("boom")
    out = await Runner(a)._resolve_busy_steer_or_redirect(make_event(), "k", "steer", agent)
    ok = out.steered
    print("PASS" if ok else "FAIL", "reaction failure is fail-open"); fails += not ok
    print(f"{4-fails}/4"); return fails

rc = asyncio.run(main()); sys.exit(1 if rc else 0)
