"""Ollama adapter for local models.

Ollama has no official Python SDK, so this adapter speaks its ``/api/chat``
endpoint over ``httpx`` directly. It is the free tier-1 floor that lets the whole
stack run with no cloud credentials at all.
"""

from __future__ import annotations

from typing import Any

import httpx

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

#: Ollama ``done_reason`` values mapped onto the domain vocabulary.
_FINISH_REASONS = {
    "stop": FinishReason.STOP,
    "length": FinishReason.LENGTH,
    "load": FinishReason.OTHER,
}


class OllamaAdapter(LLMProvider):
    """Calls locally hosted models through the Ollama chat API."""

    name = Provider.OLLAMA

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        *,
        client: httpx.AsyncClient | None = None,
        connect_timeout_s: float = 3.0,
    ) -> None:
        """Create the adapter.

        Args:
            base_url: Root URL of the Ollama server.
            client: Pre-built HTTP client, injected by tests.
            connect_timeout_s: Connection-establishment budget.
        """
        self._base_url = base_url.rstrip("/")
        self._connect_timeout_s = connect_timeout_s
        self._client = client or httpx.AsyncClient(base_url=self._base_url)
        self._owns_client = client is None

    async def complete(
        self,
        req: CompletionInput,
        cfg: ModelConfig,
        timeout_s: float,
    ) -> ProviderResponse:
        """Execute one completion call against the Ollama chat API.

        Args:
            req: The provider-agnostic request.
            cfg: Registry entry for the local model to call.
            timeout_s: Total wall-clock budget for this attempt.

        Returns:
            The normalized provider response.

        Raises:
            ProviderTimeout: The call exceeded ``timeout_s``.
            ProviderServerError: Ollama returned 5xx or was unreachable.
            ProviderBadRequest: Ollama rejected the request or returned no text.
        """
        payload = self._build_payload(req, cfg)
        timeout = httpx.Timeout(timeout_s, connect=self._connect_timeout_s)

        with latency_timer() as elapsed_ms:
            try:
                response = await self._client.post(
                    f"{self._base_url}/api/chat", json=payload, timeout=timeout
                )
            except httpx.TimeoutException as exc:
                raise ProviderTimeout(
                    f"Ollama call exceeded {timeout_s}s.",
                    provider=self.name,
                    model_id=cfg.id,
                ) from exc
            except httpx.HTTPError as exc:
                raise ProviderServerError(
                    f"Could not reach Ollama at {self._base_url}: {exc}",
                    provider=self.name,
                    model_id=cfg.id,
                ) from exc

            if response.status_code >= 400:
                message = self._error_message(response)
                if response.status_code == 404 and "not found" in message.lower():
                    raise ProviderBadRequest(
                        f"Ollama has no model {cfg.vendor_model_id!r} pulled. "
                        f"Run: ollama pull {cfg.vendor_model_id}",
                        provider=self.name,
                        model_id=cfg.id,
                    )
                raise error_for_status(
                    response.status_code,
                    message,
                    provider=self.name,
                    model_id=cfg.id,
                    headers=dict(response.headers),
                )

            try:
                body = response.json()
            except ValueError as exc:
                raise ProviderServerError(
                    "Ollama returned a malformed JSON body.",
                    provider=self.name,
                    model_id=cfg.id,
                ) from exc

        return self._to_response(body, cfg, elapsed_ms[0])

    def _build_payload(self, req: CompletionInput, cfg: ModelConfig) -> dict[str, Any]:
        """Translate a domain request into Ollama chat parameters."""
        options: dict[str, Any] = {"num_predict": min(req.max_tokens, cfg.max_output_tokens)}
        if cfg.supports_temperature:
            options["temperature"] = req.temperature
        if req.stop_sequences:
            options["stop"] = list(req.stop_sequences)
        return {
            "model": cfg.vendor_model_id,
            "messages": [{"role": m.role.value, "content": m.content} for m in req.messages],
            "stream": False,
            "options": options,
        }

    def _to_response(
        self, body: dict[str, Any], cfg: ModelConfig, latency_ms: int
    ) -> ProviderResponse:
        """Normalize an Ollama response body into a :class:`ProviderResponse`.

        Raises:
            ProviderBadRequest: The body carried no assistant text.
        """
        message = body.get("message")
        text = message.get("content", "") if isinstance(message, dict) else ""
        if not text:
            raise ProviderBadRequest(
                "Ollama returned no message content.",
                provider=self.name,
                model_id=cfg.id,
            )

        # Ollama reports counts only after a completed generation; fall back to
        # an estimate so cost and savings math always has a number to work with.
        usage = Usage(
            input_tokens=int(body.get("prompt_eval_count") or 0),
            output_tokens=int(body.get("eval_count") or estimate_tokens(text)),
        )
        raw_reason = body.get("done_reason")
        return ProviderResponse(
            model_id=cfg.id,
            provider=self.name,
            text=text,
            usage=usage,
            finish_reason=map_finish_reason(raw_reason, _FINISH_REASONS),
            raw_finish_reason=raw_reason,
            latency_ms=latency_ms,
        )

    @staticmethod
    def _error_message(response: httpx.Response) -> str:
        """Extract a readable message from an Ollama error response."""
        try:
            body = response.json()
        except ValueError:
            return response.text[:500]
        if isinstance(body, dict) and isinstance(body.get("error"), str):
            return str(body["error"])
        return response.text[:500]

    async def health(self) -> bool:
        """Return whether the Ollama server is reachable.

        Used at startup to decide whether the free local tier is available.
        """
        try:
            response = await self._client.get(
                f"{self._base_url}/api/tags", timeout=self._connect_timeout_s
            )
        except httpx.HTTPError:
            return False
        return response.status_code == 200

    async def installed_models(self) -> frozenset[str] | None:
        """Return the vendor model names currently pulled on the server.

        Both the tagged name Ollama reports (``llama3.1:8b``) and its bare stem
        (``llama3.1``) are included, because the registry may reference either.

        Returns:
            Installed model names, or ``None`` when the server is unreachable.
            The two cases are distinguished so callers can tell "Ollama is not
            running" from "Ollama is running with nothing pulled" — they need
            very different guidance.
        """
        try:
            response = await self._client.get(
                f"{self._base_url}/api/tags", timeout=self._connect_timeout_s
            )
            response.raise_for_status()
            body = response.json()
        except (httpx.HTTPError, ValueError):
            return None

        names: set[str] = set()
        for entry in body.get("models", []) if isinstance(body, dict) else []:
            name = entry.get("name") if isinstance(entry, dict) else None
            if isinstance(name, str) and name:
                names.add(name)
                names.add(name.split(":", 1)[0])
        return frozenset(names)

    async def pull(self, vendor_model_id: str, *, timeout_s: float = 1800.0) -> None:
        """Download ``vendor_model_id`` onto the Ollama server.

        Model downloads are measured in gigabytes, hence the generous default
        budget. This is never called implicitly: auto-pull is opt-in, because
        silently consuming several gigabytes of a developer's disk and bandwidth
        during startup is not a decision this process should make on its own.

        Args:
            vendor_model_id: Name to pull, e.g. ``llama3.1:8b``.
            timeout_s: Total wall-clock budget for the download.

        Raises:
            ProviderServerError: The pull failed or the server was unreachable.
        """
        try:
            response = await self._client.post(
                f"{self._base_url}/api/pull",
                json={"model": vendor_model_id, "stream": False},
                timeout=httpx.Timeout(timeout_s, connect=self._connect_timeout_s),
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise ProviderServerError(
                f"Could not pull {vendor_model_id!r} from Ollama: {exc}",
                provider=self.name,
                model_id=vendor_model_id,
            ) from exc
        _log.info("ollama.model_pulled", model=vendor_model_id)

    def count_tokens(self, text: str) -> int:
        """Return an estimated token count for ``text``."""
        return estimate_tokens(text)

    async def aclose(self) -> None:
        """Close the HTTP client when this adapter owns it."""
        if self._owns_client:
            await self._client.aclose()
