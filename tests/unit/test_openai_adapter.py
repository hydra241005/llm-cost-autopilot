from __future__ import annotations

import json

import httpx
import openai
import pytest
import respx

from autopilot.domain.enums import FinishReason, Provider, Tier
from autopilot.domain.errors import (
    ProviderBadRequest,
    ProviderRateLimited,
    ProviderServerError,
    ProviderTimeout,
)
from autopilot.infrastructure.providers.openai_adapter import OpenAIAdapter
from tests.conftest import make_model

BASE_URL = "https://api.openai.test/v1"
CHAT_URL = f"{BASE_URL}/chat/completions"


@pytest.fixture
def adapter() -> OpenAIAdapter:
    client = openai.AsyncOpenAI(api_key="test-key", base_url=BASE_URL, max_retries=0)
    return OpenAIAdapter(api_key="test-key", client=client)


@pytest.fixture
def model():
    return make_model("openai:gpt-4o-mini", tier=Tier.MODERATE)


def _completion_body(
    *,
    text: str = "A concise summary.",
    finish_reason: str = "stop",
    usage: dict | None = None,
    choices: list | None = None,
) -> dict:
    return {
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "created": 1_700_000_000,
        "model": "gpt-4o-mini",
        "choices": choices
        if choices is not None
        else [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": finish_reason,
            }
        ],
        "usage": usage or {"prompt_tokens": 90, "completion_tokens": 30, "total_tokens": 120},
    }


@respx.mock
async def test_successful_completion_is_normalized(adapter, model, sample_request):
    respx.post(CHAT_URL).mock(return_value=httpx.Response(200, json=_completion_body()))

    result = await adapter.complete(sample_request, model, timeout_s=5.0)

    assert result.text == "A concise summary."
    assert result.provider is Provider.OPENAI
    assert result.usage.input_tokens == 90
    assert result.usage.output_tokens == 30
    assert result.finish_reason is FinishReason.STOP
    assert result.vendor_request_id == "chatcmpl-123"


@respx.mock
async def test_system_message_stays_in_the_message_list(adapter, model, sample_request):
    route = respx.post(CHAT_URL).mock(
        return_value=httpx.Response(200, json=_completion_body())
    )

    await adapter.complete(sample_request, model, timeout_s=5.0)

    body = json.loads(route.calls.last.request.content)
    assert [m["role"] for m in body["messages"]] == ["system", "user"]
    assert body["temperature"] == 0.2


@respx.mock
async def test_length_finish_reason_is_mapped(adapter, model, sample_request):
    respx.post(CHAT_URL).mock(
        return_value=httpx.Response(200, json=_completion_body(finish_reason="length"))
    )

    result = await adapter.complete(sample_request, model, timeout_s=5.0)

    assert result.finish_reason is FinishReason.LENGTH


@respx.mock
async def test_content_filter_is_mapped(adapter, model, sample_request):
    respx.post(CHAT_URL).mock(
        return_value=httpx.Response(200, json=_completion_body(finish_reason="content_filter"))
    )

    result = await adapter.complete(sample_request, model, timeout_s=5.0)

    assert result.finish_reason is FinishReason.CONTENT_FILTER


@respx.mock
async def test_cached_prompt_tokens_are_recorded(adapter, model, sample_request):
    usage = {
        "prompt_tokens": 100,
        "completion_tokens": 20,
        "total_tokens": 120,
        "prompt_tokens_details": {"cached_tokens": 64},
    }
    respx.post(CHAT_URL).mock(
        return_value=httpx.Response(200, json=_completion_body(usage=usage))
    )

    result = await adapter.complete(sample_request, model, timeout_s=5.0)

    assert result.usage.cached_input_tokens == 64


@respx.mock
async def test_no_choices_becomes_bad_request(adapter, model, sample_request):
    respx.post(CHAT_URL).mock(
        return_value=httpx.Response(200, json=_completion_body(choices=[]))
    )

    with pytest.raises(ProviderBadRequest, match="no choices"):
        await adapter.complete(sample_request, model, timeout_s=5.0)


@respx.mock
async def test_empty_content_becomes_bad_request(adapter, model, sample_request):
    respx.post(CHAT_URL).mock(
        return_value=httpx.Response(200, json=_completion_body(text=""))
    )

    with pytest.raises(ProviderBadRequest, match="empty content"):
        await adapter.complete(sample_request, model, timeout_s=5.0)


@respx.mock
async def test_rate_limit_is_retryable_and_carries_retry_after(adapter, model, sample_request):
    respx.post(CHAT_URL).mock(
        return_value=httpx.Response(
            429,
            headers={"retry-after": "3"},
            json={"error": {"message": "rate limited", "type": "rate_limit_error"}},
        )
    )

    with pytest.raises(ProviderRateLimited) as exc:
        await adapter.complete(sample_request, model, timeout_s=5.0)

    assert exc.value.retryable is True
    assert exc.value.retry_after_s == 3.0


@respx.mock
async def test_server_error_is_retryable(adapter, model, sample_request):
    respx.post(CHAT_URL).mock(
        return_value=httpx.Response(503, json={"error": {"message": "unavailable"}})
    )

    with pytest.raises(ProviderServerError) as exc:
        await adapter.complete(sample_request, model, timeout_s=5.0)

    assert exc.value.retryable is True


@respx.mock
async def test_bad_request_is_not_retryable(adapter, model, sample_request):
    respx.post(CHAT_URL).mock(
        return_value=httpx.Response(400, json={"error": {"message": "unknown parameter"}})
    )

    with pytest.raises(ProviderBadRequest) as exc:
        await adapter.complete(sample_request, model, timeout_s=5.0)

    assert exc.value.retryable is False
    assert "unknown parameter" in exc.value.message


@respx.mock
async def test_timeout_maps_to_provider_timeout(adapter, model, sample_request):
    respx.post(CHAT_URL).mock(side_effect=httpx.ReadTimeout("slow"))

    with pytest.raises(ProviderTimeout):
        await adapter.complete(sample_request, model, timeout_s=5.0)


@respx.mock
async def test_connection_error_maps_to_server_error(adapter, model, sample_request):
    respx.post(CHAT_URL).mock(side_effect=httpx.ConnectError("refused"))

    with pytest.raises(ProviderServerError):
        await adapter.complete(sample_request, model, timeout_s=5.0)


@respx.mock
async def test_malformed_json_body_is_surfaced(adapter, model, sample_request):
    respx.post(CHAT_URL).mock(return_value=httpx.Response(200, content=b"not json"))

    with pytest.raises((ProviderBadRequest, ProviderServerError, openai.APIError)):
        await adapter.complete(sample_request, model, timeout_s=5.0)
