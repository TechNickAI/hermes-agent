"""Claude 5.5 rejecting ``thinking: {type: disabled}`` must take the drop-the-disable rung.

Opus/Sonnet 5.5 answer the disable with HTTP 400 '"thinking.type.disabled" is not supported'.
A combo alias (claude-chat, claude-spare-capacity, claude-think) cannot be recognised by model
string, so on the alias path this classifier rung is the only guard. OmniRoute relays the 400 as
an SSE error frame behind HTTP 200, which the status-400 handler never sees.
"""
import json

import pytest

from agent.error_classifier import FailoverReason, classify_api_error
from tests.agent.test_error_classifier import MockAPIError

# Anthropic's live wording (Opus 5.5 via OmniRoute, captured 2026-10-12).
_CLAUDE55_DISABLE_400 = ('"thinking.type.disabled" is not supported for this model. Use '
                         '"thinking.type.adaptive" and "output_config.effort" to control thinking behavior.')

def test_claude55_thinking_disable_400_is_reasoning_mandatory():
    """Opus/Sonnet 5.5 reject ``thinking: {type: disabled}``. A combo alias (claude-chat,
    claude-spare-capacity, claude-think) cannot be recognised by model string, so the alias path
    relies on this rung: drop the disable and retry once, never a format_error abort."""
    body = {"type": "error", "error": {"type": "invalid_request_error", "message": _CLAUDE55_DISABLE_400}}
    for model in ("claude-chat", "claude-spare-capacity", "claude-think"):
        result = classify_api_error(
            MockAPIError(_CLAUDE55_DISABLE_400, status_code=400, body=body), provider="custom", model=model,
        )
        assert result.reason == FailoverReason.reasoning_mandatory, model
        assert result.retryable is True and result.should_fallback is False

def test_claude55_thinking_disable_status_200_relay_frame_is_reasoning_mandatory():
    """SDK-independent shape of the relayed rejection: the Anthropic SDK raises APIStatusError with
    ``status_code == 200`` and the frame dict as body/message. Runs where ``anthropic`` is absent."""
    body = {"error": {"message": "claude/claude-opus-5-5: model — " + _CLAUDE55_DISABLE_400 + " (HTTP 400)",
                      "type": "invalid_request_error", "code": "bad_request"}}
    result = classify_api_error(MockAPIError(str(body), status_code=200, body=body),
                                provider="custom", model="claude-spare-capacity")
    assert result.reason == FailoverReason.reasoning_mandatory
    assert result.retryable is True and result.should_fallback is False

def test_claude55_thinking_disable_relayed_as_sse_error_is_reasoning_mandatory():
    """OmniRoute relays the upstream 400 as an ``event: error`` frame on an HTTP 200 stream, so the
    Anthropic SDK raises APIStatusError with status 200 and the 400 handler never runs. Without the
    exact-wording rule it lands on ``unknown`` and every retry resends the same disable (observed
    live: 6 rejected attempts before giving up). Real SDK, real SSE bytes."""
    import json as _json
    anthropic = pytest.importorskip("anthropic")
    httpx = pytest.importorskip("httpx")
    err = {"error": {"message": "claude/claude-opus-5-5: model — " + _CLAUDE55_DISABLE_400 + " (HTTP 400)",
                     "type": "invalid_request_error", "code": "bad_request"}}
    sse = ('event: ping\ndata: {"type":"ping"}\n\nevent: error\ndata: ' + json.dumps(err) + "\n\n").encode()
    transport = httpx.MockTransport(
        lambda req: httpx.Response(200, headers={"content-type": "text/event-stream"}, content=sse))
    client = anthropic.Anthropic(api_key="x", base_url="http://relay.test", http_client=httpx.Client(transport=transport))
    with pytest.raises(anthropic.APIStatusError) as exc:
        with client.messages.stream(model="claude-spare-capacity", max_tokens=8,
                                    messages=[{"role": "user", "content": "hi"}]) as stream:
            for _ in stream:
                pass
    result = classify_api_error(exc.value, provider="custom", model="claude-spare-capacity")
    assert result.reason == FailoverReason.reasoning_mandatory
    assert result.retryable is True and result.should_fallback is False

