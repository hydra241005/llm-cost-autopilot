"""Anthropic Messages API adapter.

Uses the official ``anthropic`` async SDK rather than raw HTTP so vendor API
drift is absorbed by the SDK rather than by this file.

Three modern-model behaviours are handled deliberately:

* ``content`` is a *list of typed blocks*. A ``thinking`` block can precede the
  text, so blocks are filtered by type rather than indexed.
* Opus 5 and Sonnet 5 reject non-default sampling parameters, so ``temperature``
  is sent only when the registry entry declares support for it.
* ``stop_details`` is populated only for refusals and is guarded accordingly.
"""

from __future__ import annotations

from typing import Any

import anthropic
from anthropic.types import TextBlock

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

#: Anthropic ``stop_reason`` values mapped onto the domain vocabulary.
_FINISH_REASONS = {
    "end_turn": FinishReason.STOP,
    "stop_sequence": FinishReason.STOP,
    "max_tokens": FinishReason.LENGTH,
    "tool_use": FinishReason.TOOL_USE,
    "pause_turn": FinishReason.OTHER,
    "refusal": FinishReason.CONTENT_FILTER,
}


class AnthropicAdapter(LLMProvider):
    """Calls Claude models through the Anthropic Messages API."""

    name = Provider.ANTHROPIC

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str | None = None,
        client: anthropic.AsyncAnthropic | None = None,
    ) -> None:
        """Create the adapter.

        Args:
            api_key: Anthropic API key, sourced from the environment.
            base_url: Override for the API host, used by tests and proxies.
            client: Pre-built SDK client, injected by tests.
        """
        self._client = client or anthropic.AsyncAnthropic(
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
        """Execute one completion call against the Messages API.

        Args:
            req: The provider-agnostic request.
            cfg: Registry entry for the Claude model to call.
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
                message = await self._client.messages.create(timeout=timeout_s, **payload)
            except anthropic.APITimeoutError as exc:
                raise ProviderTimeout(
                    f"Anthropic call exceeded {timeout_s}s.",
                    provider=self.name,
                    model_id=cfg.id,
                ) from exc
            except anthropic.APIStatusError as exc:
                raise error_for_status(
                    exc.status_code,
                    self._error_message(exc),
                    provider=self.name,
                    model_id=cfg.id,
                    headers=dict(exc.response.headers) if exc.response is not None else None,
                ) from exc
            except anthropic.APIConnectionError as exc:
                raise ProviderServerError(
                    f"Could not reach Anthropic: {exc}",
                    provider=self.name,
                    model_id=cfg.id,
                ) from exc

        return self._to_response(message, cfg, elapsed_ms[0])

    def _build_payload(self, req: CompletionInput, cfg: ModelConfig) -> dict[str, Any]:
        """Translate a domain request into Messages API parameters."""
        payload: dict[str, Any] = {
            "model": cfg.vendor_model_id,
            "max_tokens": min(req.max_tokens, cfg.max_output_tokens),
            "messages": [
                {"role": m.role.value, "content": m.content} for m in req.conversation
            ],
        }
        if (system := req.system_prompt) is not None:
            payload["system"] = system
        if req.stop_sequences:
            payload["stop_sequences"] = list(req.stop_sequences)
        # Opus 5 / Sonnet 5 reject non-default sampling parameters with a 400.
        if cfg.supports_temperature:
            payload["temperature"] = req.temperature
        return payload

    def _to_response(
        self, message: anthropic.types.Message, cfg: ModelConfig, latency_ms: int
    ) -> ProviderResponse:
        """Normalize an SDK ``Message`` into a :class:`ProviderResponse`.

        Raises:
            ProviderBadRequest: The model refused the request, or returned no text.
        """
        # content is a list of typed blocks; a thinking block can precede the text.
        text = "".join(
            block.text for block in message.content if isinstance(block, TextBlock)
        )

        if message.stop_reason == "refusal":
            # stop_details is populated only for refusals; guard before reading it.
            detail = getattr(message, "stop_details", None)
            reason = getattr(detail, "type", "refusal") if detail is not None else "refusal"
            raise ProviderBadRequest(
                f"Anthropic refused the request ({reason}).",
                provider=self.name,
                model_id=cfg.id,
            )

        if not text:
            raise ProviderBadRequest(
                "Anthropic returned no text content.",
                provider=self.name,
                model_id=cfg.id,
            )

        usage = Usage(
            input_tokens=message.usage.input_tokens,
            output_tokens=message.usage.output_tokens,
            cached_input_tokens=getattr(message.usage, "cache_read_input_tokens", 0) or 0,
        )
        return ProviderResponse(
            model_id=cfg.id,
            provider=self.name,
            text=text,
            usage=usage,
            finish_reason=map_finish_reason(message.stop_reason, _FINISH_REASONS),
            raw_finish_reason=message.stop_reason,
            latency_ms=latency_ms,
            vendor_request_id=getattr(message, "id", None),
        )

    @staticmethod
    def _error_message(exc: anthropic.APIStatusError) -> str:
        """Extract a safe, credential-free message from an SDK error."""
        body = getattr(exc, "body", None)
        if isinstance(body, dict):
            error = body.get("error")
            if isinstance(error, dict) and isinstance(error.get("message"), str):
                return str(error["message"])
        return str(exc)

    def count_tokens(self, text: str) -> int:
        """Return an estimated token count for ``text``.

        A heuristic is used rather than the vendor's count-tokens endpoint: the
        router's budget is well under ten milliseconds, which rules out a network
        round trip on the request path.
        """
        return estimate_tokens(text)

    async def aclose(self) -> None:
        """Close the underlying SDK client."""
        await self._client.close()
