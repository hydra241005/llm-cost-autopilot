from __future__ import annotations

import json

import anthropic
import httpx
import pytest
import respx

from autopilot.domain.enums import FinishReason, Provider, Tier
from autopilot.domain.errors import (
    ProviderBadRequest,
    ProviderRateLimited,
    ProviderServerError,
    ProviderTimeout,
)
from autopilot.infrastructure.providers.anthropic_adapter import AnthropicAdapter
from tests.conftest import make_model

BASE_URL = "https://api.anthropic.test"
MESSAGES_URL = f"{BASE_URL}/v1/messages"


@pytest.fixture
def adapter() -> AnthropicAdapter:
    client = anthropic.AsyncAnthropic(api_key="test-key", base_url=BASE_URL, max_retries=0)
    return AnthropicAdapter(api_key="test-key", client=client)


@pytest.fixture
def model():
    return make_model(
        "anthropic:claude-sonnet-5",
        provider=Provider.ANTHROPIC,
        tier=Tier.MODERATE,
        input_cost="3.00",
        output_cost="15.00",
        supports_temperature=False,
    )


def _message_body(
    *, text: str = "Relativity in brief.", stop_reason: str = "end_turn", blocks=None
) -> dict:
    return {
        "id": "msg_123",
        "type": "message",
        "role": "assistant",
        "model": "claude-sonnet-5",
        "content": blocks if blocks is not None else [{"type": "text", "text": text}],
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {"input_tokens": 120, "output_tokens": 45},
    }


@respx.mock
async def test_successful_completion_is_normalized(adapter, model, sample_request):
    respx.post(MESSAGES_URL).mock(return_value=httpx.Response(200, json=_message_body()))

    result = await adapter.complete(sample_request, model, timeout_s=5.0)

    assert result.text == "Relativity in brief."
    assert result.provider is Provider.ANTHROPIC
    assert result.model_id == "anthropic:claude-sonnet-5"
    assert result.usage.input_tokens == 120
    assert result.usage.output_tokens == 45
    assert result.finish_reason is FinishReason.STOP
    assert result.vendor_request_id == "msg_123"


@respx.mock
async def test_system_prompt_is_sent_out_of_band(adapter, model, sample_request):
    route = respx.post(MESSAGES_URL).mock(
        return_value=httpx.Response(200, json=_message_body())
    )

    await adapter.complete(sample_request, model, timeout_s=5.0)

    body = json.loads(route.calls.last.request.content)
    assert body["system"] == "You are terse."
    assert [m["role"] for m in body["messages"]] == ["user"]


@respx.mock
async def test_temperature_is_omitted_for_models_that_reject_it(adapter, model, sample_request):
    route = respx.post(MESSAGES_URL).mock(
        return_value=httpx.Response(200, json=_message_body())
    )

    await adapter.complete(sample_request, model, timeout_s=5.0)

    assert "temperature" not in route.calls.last.request.content.decode()


@respx.mock
async def test_temperature_is_sent_when_supported(adapter, sample_request):
    model = make_model(
        "anthropic:claude-opus-4-6",
        provider=Provider.ANTHROPIC,
        supports_temperature=True,
    )
    route = respx.post(MESSAGES_URL).mock(
        return_value=httpx.Response(200, json=_message_body())
    )

    await adapter.complete(sample_request, model, timeout_s=5.0)

    assert "temperature" in route.calls.last.request.content.decode()


@respx.mock
async def test_thinking_block_before_text_is_handled(adapter, model, sample_request):
    blocks = [
        {"type": "thinking", "thinking": "internal", "signature": "sig"},
        {"type": "text", "text": "The answer."},
    ]
    respx.post(MESSAGES_URL).mock(
        return_value=httpx.Response(200, json=_message_body(blocks=blocks))
    )

    result = await adapter.complete(sample_request, model, timeout_s=5.0)

    assert result.text == "The answer."


@respx.mock
async def test_max_tokens_stop_reason_maps_to_length(adapter, model, sample_request):
    respx.post(MESSAGES_URL).mock(
        return_value=httpx.Response(200, json=_message_body(stop_reason="max_tokens"))
    )

    result = await adapter.complete(sample_request, model, timeout_s=5.0)

    assert result.finish_reason is FinishReason.LENGTH
    assert result.raw_finish_reason == "max_tokens"


@respx.mock
async def test_refusal_becomes_bad_request(adapter, model, sample_request):
    respx.post(MESSAGES_URL).mock(
        return_value=httpx.Response(200, json=_message_body(stop_reason="refusal"))
    )

    with pytest.raises(ProviderBadRequest, match="refused"):
        await adapter.complete(sample_request, model, timeout_s=5.0)


@respx.mock
async def test_empty_content_becomes_bad_request(adapter, model, sample_request):
    respx.post(MESSAGES_URL).mock(
        return_value=httpx.Response(200, json=_message_body(blocks=[]))
    )

    with pytest.raises(ProviderBadRequest, match="no text"):
        await adapter.complete(sample_request, model, timeout_s=5.0)


@respx.mock
async def test_rate_limit_maps_to_retryable_error_with_retry_after(
    adapter, model, sample_request
):
    respx.post(MESSAGES_URL).mock(
        return_value=httpx.Response(
            429,
            headers={"retry-after": "7"},
            json={"type": "error", "error": {"type": "rate_limit_error", "message": "slow"}},
        )
    )

    with pytest.raises(ProviderRateLimited) as exc:
        await adapter.complete(sample_request, model, timeout_s=5.0)

    assert exc.value.retryable is True
    assert exc.value.retry_after_s == 7.0
    assert exc.value.model_id == "anthropic:claude-sonnet-5"


@respx.mock
async def test_server_error_maps_to_retryable_error(adapter, model, sample_request):
    respx.post(MESSAGES_URL).mock(
        return_value=httpx.Response(
            500, json={"type": "error", "error": {"type": "api_error", "message": "oops"}}
        )
    )

    with pytest.raises(ProviderServerError) as exc:
        await adapter.complete(sample_request, model, timeout_s=5.0)

    assert exc.value.retryable is True


@respx.mock
async def test_overloaded_529_maps_to_retryable_error(adapter, model, sample_request):
    respx.post(MESSAGES_URL).mock(
        return_value=httpx.Response(
            529,
            json={"type": "error", "error": {"type": "overloaded_error", "message": "busy"}},
        )
    )

    with pytest.raises(ProviderServerError):
        await adapter.complete(sample_request, model, timeout_s=5.0)


@respx.mock
async def test_bad_request_is_not_retryable(adapter, model, sample_request):
    respx.post(MESSAGES_URL).mock(
        return_value=httpx.Response(
            400,
            json={
                "type": "error",
                "error": {"type": "invalid_request_error", "message": "bad param"},
            },
        )
    )

    with pytest.raises(ProviderBadRequest) as exc:
        await adapter.complete(sample_request, model, timeout_s=5.0)

    assert exc.value.retryable is False
    assert "bad param" in exc.value.message


@respx.mock
async def test_authentication_error_is_not_retryable(adapter, model, sample_request):
    respx.post(MESSAGES_URL).mock(
        return_value=httpx.Response(
            401,
            json={
                "type": "error",
                "error": {"type": "authentication_error", "message": "bad key"},
            },
        )
    )

    with pytest.raises(ProviderBadRequest) as exc:
        await adapter.complete(sample_request, model, timeout_s=5.0)

    assert exc.value.retryable is False


@respx.mock
async def test_timeout_maps_to_provider_timeout(adapter, model, sample_request):
    respx.post(MESSAGES_URL).mock(side_effect=httpx.ReadTimeout("timed out"))

    with pytest.raises(ProviderTimeout) as exc:
        await adapter.complete(sample_request, model, timeout_s=5.0)

    assert exc.value.retryable is True


@respx.mock
async def test_connection_error_maps_to_server_error(adapter, model, sample_request):
    respx.post(MESSAGES_URL).mock(side_effect=httpx.ConnectError("no route"))

    with pytest.raises(ProviderServerError):
        await adapter.complete(sample_request, model, timeout_s=5.0)


@respx.mock
async def test_max_tokens_is_capped_at_model_output_limit(adapter, sample_request):
    model = make_model(
        "anthropic:claude-haiku-4-5", provider=Provider.ANTHROPIC, max_output_tokens=100
    )
    route = respx.post(MESSAGES_URL).mock(
        return_value=httpx.Response(200, json=_message_body())
    )

    await adapter.complete(sample_request, model, timeout_s=5.0)

    assert json.loads(route.calls.last.request.content)["max_tokens"] == 100
