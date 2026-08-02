"""OpenAI Chat Completions adapter.

Uses the official ``openai`` async SDK. Vendor exceptions are translated into the
domain hierarchy so the retry and breaker layers never see an OpenAI type.
"""

from __future__ import annotations

from typing import Any

import openai

from autopilot.domain.entities import CompletionInput, ModelConfig, ProviderResponse, Usage
from autopilot.domain.enums import FinishReason, Provider
from autopilot.domain.errors import ProviderBadRequest, ProviderServerError, ProviderTimeout
from autopilot.domain.interfaces import LLMProvider
from autopilot.infrastructure.observability.logging import get_logger
from autopilot.infrastructure.providers.base import (
    error_for_status,
    estimate_tokens,
    latency_timer,
    map_finish_reason,
)

_log = get_logger(__name__)

#: OpenAI ``finish_reason`` values mapped onto the domain vocabulary.
_FINISH_REASONS = {
    "stop": FinishReason.STOP,
    "length": FinishReason.LENGTH,
    "content_filter": FinishReason.CONTENT_FILTER,
    "tool_calls": FinishReason.TOOL_USE,
    "function_call": FinishReason.TOOL_USE,
}


class OpenAIAdapter(LLMProvider):
    """Calls GPT models through the OpenAI Chat Completions API."""

    name = Provider.OPENAI

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str | None = None,
        client: openai.AsyncOpenAI | None = None,
    ) -> None:
        """Create the adapter.

        Args:
            api_key: OpenAI API key, sourced from the environment.
            base_url: Override for the API host, used by tests and proxies.
            client: Pre-built SDK client, injected by tests.
        """
        self._client = client or openai.AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            max_retries=0,  # Retry policy is owned by resilience/retry.py.
        )

    async def complete(
        self,
        req: CompletionInput,
        cfg: ModelConfig,
        timeout_s: float,
    ) -> ProviderResponse:
        """Execute one completion call against the Chat Completions API.

        Args:
            req: The provider-agnostic request.
            cfg: Registry entry for the model to call.
            timeout_s: Total wall-clock budget for this attempt.

        Returns:
            The normalized provider response.

        Raises:
            ProviderTimeout: The call exceeded ``timeout_s``.
            ProviderRateLimited: The vendor returned HTTP 429.
            ProviderServerError: The vendor returned 5xx or the connection failed.
            ProviderBadRequest: The vendor rejected the request itself.
        """
        payload = self._build_payload(req, cfg)

        with latency_timer() as elapsed_ms:
            try:
                completion = await self._client.chat.completions.create(
                    timeout=timeout_s, **payload
                )
            except openai.APITimeoutError as exc:
                raise ProviderTimeout(
                    f"OpenAI call exceeded {timeout_s}s.",
                    provider=self.name,
                    model_id=cfg.id,
                ) from exc
            except openai.APIStatusError as exc:
                raise error_for_status(
                    exc.status_code,
                    self._error_message(exc),
                    provider=self.name,
                    model_id=cfg.id,
                    headers=dict(exc.response.headers) if exc.response is not None else None,
                ) from exc
            except openai.APIConnectionError as exc:
                raise ProviderServerError(
                    f"Could not reach OpenAI: {exc}",
                    provider=self.name,
                    model_id=cfg.id,
                ) from exc

        return self._to_response(completion, cfg, elapsed_ms[0])

    def _build_payload(self, req: CompletionInput, cfg: ModelConfig) -> dict[str, Any]:
        """Translate a domain request into Chat Completions parameters."""
        payload: dict[str, Any] = {
            "model": cfg.vendor_model_id,
            "max_tokens": min(req.max_tokens, cfg.max_output_tokens),
            "messages": [{"role": m.role.value, "content": m.content} for m in req.messages],
        }
        if req.stop_sequences:
            payload["stop"] = list(req.stop_sequences)
        if cfg.supports_temperature:
            payload["temperature"] = req.temperature
        return payload

    def _to_response(
        self, completion: Any, cfg: ModelConfig, latency_ms: int
    ) -> ProviderResponse:
        """Normalize an SDK completion into a :class:`ProviderResponse`.

        Raises:
            ProviderBadRequest: The response carried no usable choice or text.
            ProviderServerError: The body was not a parseable completion object.
        """
        # A non-JSON body is handed back by the SDK as a raw string rather than
        # a parsed model, so shape is checked before attribute access.
        if not hasattr(completion, "choices"):
            raise ProviderServerError(
                "OpenAI returned a malformed response body.",
                provider=self.name,
                model_id=cfg.id,
            )
        if not completion.choices:
            raise ProviderBadRequest(
                "OpenAI returned no choices.", provider=self.name, model_id=cfg.id
            )

        choice = completion.choices[0]
        text = choice.message.content or ""
        if not text:
            raise ProviderBadRequest(
                "OpenAI returned empty content.", provider=self.name, model_id=cfg.id
            )

        raw_usage = completion.usage
        usage = Usage(
            input_tokens=raw_usage.prompt_tokens if raw_usage else estimate_tokens(text),
            output_tokens=raw_usage.completion_tokens if raw_usage else estimate_tokens(text),
            cached_input_tokens=self._cached_tokens(raw_usage),
        )
        return ProviderResponse(
            model_id=cfg.id,
            provider=self.name,
            text=text,
            usage=usage,
            finish_reason=map_finish_reason(choice.finish_reason, _FINISH_REASONS),
            raw_finish_reason=choice.finish_reason,
            latency_ms=latency_ms,
            vendor_request_id=getattr(completion, "id", None),
        )

    @staticmethod
    def _cached_tokens(raw_usage: Any) -> int:
        """Return prompt tokens served from OpenAI's prompt cache, if reported."""
        details = getattr(raw_usage, "prompt_tokens_details", None)
        return int(getattr(details, "cached_tokens", 0) or 0)

    @staticmethod
    def _error_message(exc: openai.APIStatusError) -> str:
        """Extract a safe, credential-free message from an SDK error."""
        body = getattr(exc, "body", None)
        if isinstance(body, dict):
            error = body.get("error")
            if isinstance(error, dict) and isinstance(error.get("message"), str):
                return str(error["message"])
        return str(exc)

    def count_tokens(self, text: str) -> int:
        """Return an estimated token count for ``text``."""
        return estimate_tokens(text)

    async def aclose(self) -> None:
        """Close the underlying SDK client."""
        await self._client.close()
