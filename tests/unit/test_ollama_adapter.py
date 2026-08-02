from __future__ import annotations

import json

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
from autopilot.infrastructure.providers.ollama_adapter import OllamaAdapter
from tests.conftest import make_model

BASE_URL = "http://ollama.test:11434"
CHAT_URL = f"{BASE_URL}/api/chat"
TAGS_URL = f"{BASE_URL}/api/tags"


@pytest.fixture
def adapter() -> OllamaAdapter:
    return OllamaAdapter(base_url=BASE_URL)


@pytest.fixture
def model():
    return make_model(
        "ollama:llama3.1-8b",
        provider=Provider.OLLAMA,
        tier=Tier.SIMPLE,
        input_cost="0",
        output_cost="0",
    )


def _chat_body(
    *,
    text: str = "Local answer.",
    done_reason: str = "stop",
    prompt_eval_count: int | None = 80,
    eval_count: int | None = 25,
) -> dict:
    body: dict = {
        "model": "llama3.1:8b",
        "message": {"role": "assistant", "content": text},
        "done": True,
        "done_reason": done_reason,
    }
    if prompt_eval_count is not None:
        body["prompt_eval_count"] = prompt_eval_count
    if eval_count is not None:
        body["eval_count"] = eval_count
    return body


@respx.mock
async def test_successful_completion_is_normalized(adapter, model, sample_request):
    respx.post(CHAT_URL).mock(return_value=httpx.Response(200, json=_chat_body()))

    result = await adapter.complete(sample_request, model, timeout_s=5.0)

    assert result.text == "Local answer."
    assert result.provider is Provider.OLLAMA
    assert result.model_id == "ollama:llama3.1-8b"
    assert result.usage.input_tokens == 80
    assert result.usage.output_tokens == 25
    assert result.finish_reason is FinishReason.STOP


@respx.mock
async def test_streaming_is_disabled_and_options_are_sent(adapter, model, sample_request):
    route = respx.post(CHAT_URL).mock(return_value=httpx.Response(200, json=_chat_body()))

    await adapter.complete(sample_request, model, timeout_s=5.0)

    body = json.loads(route.calls.last.request.content)
    assert body["stream"] is False
    assert body["model"] == "llama3.1-8b"
    assert body["options"]["num_predict"] == min(sample_request.max_tokens, model.max_output_tokens)
    assert [m["role"] for m in body["messages"]] == ["system", "user"]


@respx.mock
async def test_length_done_reason_is_mapped(adapter, model, sample_request):
    respx.post(CHAT_URL).mock(
        return_value=httpx.Response(200, json=_chat_body(done_reason="length"))
    )

    result = await adapter.complete(sample_request, model, timeout_s=5.0)

    assert result.finish_reason is FinishReason.LENGTH


@respx.mock
async def test_missing_eval_count_falls_back_to_an_estimate(adapter, model, sample_request):
    respx.post(CHAT_URL).mock(
        return_value=httpx.Response(200, json=_chat_body(eval_count=None))
    )

    result = await adapter.complete(sample_request, model, timeout_s=5.0)

    assert result.usage.output_tokens > 0


@respx.mock
async def test_empty_content_becomes_bad_request(adapter, model, sample_request):
    respx.post(CHAT_URL).mock(return_value=httpx.Response(200, json=_chat_body(text="")))

    with pytest.raises(ProviderBadRequest, match="no message content"):
        await adapter.complete(sample_request, model, timeout_s=5.0)


@respx.mock
async def test_missing_message_key_becomes_bad_request(adapter, model, sample_request):
    respx.post(CHAT_URL).mock(return_value=httpx.Response(200, json={"done": True}))

    with pytest.raises(ProviderBadRequest):
        await adapter.complete(sample_request, model, timeout_s=5.0)


@respx.mock
async def test_malformed_json_body_maps_to_server_error(adapter, model, sample_request):
    respx.post(CHAT_URL).mock(return_value=httpx.Response(200, content=b"<html>nope"))

    with pytest.raises(ProviderServerError, match="malformed"):
        await adapter.complete(sample_request, model, timeout_s=5.0)


@respx.mock
async def test_model_not_pulled_maps_to_bad_request(adapter, model, sample_request):
    respx.post(CHAT_URL).mock(
        return_value=httpx.Response(404, json={"error": "model 'llama3.1' not found"})
    )

    with pytest.raises(ProviderBadRequest) as exc:
        await adapter.complete(sample_request, model, timeout_s=5.0)

    assert "ollama pull llama3.1-8b" in exc.value.message
    assert exc.value.retryable is False


@respx.mock
async def test_server_error_is_retryable(adapter, model, sample_request):
    respx.post(CHAT_URL).mock(return_value=httpx.Response(500, json={"error": "out of memory"}))

    with pytest.raises(ProviderServerError) as exc:
        await adapter.complete(sample_request, model, timeout_s=5.0)

    assert exc.value.retryable is True


@respx.mock
async def test_rate_limit_carries_retry_after(adapter, model, sample_request):
    respx.post(CHAT_URL).mock(
        return_value=httpx.Response(429, headers={"retry-after": "2"}, json={"error": "busy"})
    )

    with pytest.raises(ProviderRateLimited) as exc:
        await adapter.complete(sample_request, model, timeout_s=5.0)

    assert exc.value.retry_after_s == 2.0


@respx.mock
async def test_timeout_maps_to_provider_timeout(adapter, model, sample_request):
    respx.post(CHAT_URL).mock(side_effect=httpx.ReadTimeout("slow"))

    with pytest.raises(ProviderTimeout) as exc:
        await adapter.complete(sample_request, model, timeout_s=5.0)

    assert exc.value.retryable is True


@respx.mock
async def test_unreachable_server_maps_to_server_error(adapter, model, sample_request):
    respx.post(CHAT_URL).mock(side_effect=httpx.ConnectError("connection refused"))

    with pytest.raises(ProviderServerError, match="Could not reach Ollama"):
        await adapter.complete(sample_request, model, timeout_s=5.0)


@respx.mock
async def test_health_returns_true_when_server_answers():
    adapter = OllamaAdapter(base_url=BASE_URL)
    respx.get(TAGS_URL).mock(return_value=httpx.Response(200, json={"models": []}))

    assert await adapter.health() is True


@respx.mock
async def test_health_returns_false_when_server_is_down():
    adapter = OllamaAdapter(base_url=BASE_URL)
    respx.get(TAGS_URL).mock(side_effect=httpx.ConnectError("refused"))

    assert await adapter.health() is False


async def test_injected_client_is_not_closed_by_the_adapter():
    client = httpx.AsyncClient(base_url=BASE_URL)
    adapter = OllamaAdapter(base_url=BASE_URL, client=client)

    await adapter.aclose()

    assert not client.is_closed
    await client.aclose()


def test_count_tokens_is_a_positive_estimate(adapter):
    assert adapter.count_tokens("a" * 400) > 0
