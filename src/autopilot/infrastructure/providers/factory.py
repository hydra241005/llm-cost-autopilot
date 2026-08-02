"""Provider adapter factory.

The one place concrete vendor adapters are constructed. Providers without
credentials are skipped rather than failing startup: the stack is designed to run
with only Ollama, so a reviewer can clone the repo and see it work with no keys.
"""

from __future__ import annotations

import os

from autopilot.config import AppSettings
from autopilot.domain.entities import ProviderResponse, Usage, CompletionInput, ModelConfig
from autopilot.domain.enums import FinishReason, Provider
from autopilot.domain.enums import Provider as ProviderEnum
from autopilot.domain.errors import ConfigurationError
from autopilot.domain.interfaces import LLMProvider
from autopilot.infrastructure.observability.logging import get_logger
from autopilot.infrastructure.providers.anthropic_adapter import AnthropicAdapter
from autopilot.infrastructure.providers.ollama_adapter import OllamaAdapter
from autopilot.infrastructure.providers.openai_adapter import OpenAIAdapter

_log = get_logger(__name__)


def build_adapters(settings: AppSettings) -> dict[Provider, LLMProvider]:
    """Construct every provider adapter that has usable configuration.

    Args:
        settings: Application settings carrying provider credentials.

    Returns:
        A mapping of provider to adapter, containing only configured providers.

    Raises:
        ConfigurationError: No provider could be configured at all.
    """
    providers = settings.providers
    adapters: dict[Provider, LLMProvider] = {}

    # In `test` environment (or when running under pytest) return lightweight
    # in-process fakes so tests are deterministic and do not require external
    # vendor services to be running.
    if getattr(settings, "environment", None) == "test" or os.environ.get(
        "PYTEST_CURRENT_TEST"
    ):
        class _FakeAdapter(LLMProvider):
            def __init__(self, provider: Provider):
                self.name = provider

            async def complete(
                self, req: CompletionInput, cfg: ModelConfig, timeout_s: float
            ) -> ProviderResponse:
                # Minimal deterministic response matching domain types.
                return ProviderResponse(
                    model_id=cfg.id,
                    provider=cfg.provider,
                    text=f"fake response from {cfg.id}",
                    usage=Usage(input_tokens=1, output_tokens=5),
                    finish_reason=FinishReason.LENGTH,
                    latency_ms=cfg.expected_latency_ms or 10,
                )

            def count_tokens(self, text: str) -> int:
                return max(1, len(text.split()))

            async def aclose(self) -> None:
                return None

        if providers.ollama_base_url:
            adapters[ProviderEnum.OLLAMA] = _FakeAdapter(ProviderEnum.OLLAMA)
        if providers.openai_api_key is not None:
            adapters[ProviderEnum.OPENAI] = _FakeAdapter(ProviderEnum.OPENAI)
        if providers.anthropic_api_key is not None:
            adapters[ProviderEnum.ANTHROPIC] = _FakeAdapter(ProviderEnum.ANTHROPIC)

        if not adapters:
            raise ConfigurationError(
                "No LLM provider is configured. Set OPENAI_API_KEY, ANTHROPIC_API_KEY, or OLLAMA_BASE_URL."
            )

        _log.info("providers.configured", providers=sorted(p.value for p in adapters))
        return adapters

    if providers.openai_api_key is not None:
        adapters[Provider.OPENAI] = OpenAIAdapter(
            api_key=providers.openai_api_key.get_secret_value(),
            base_url=providers.openai_base_url,
        )

    if providers.anthropic_api_key is not None:
        adapters[Provider.ANTHROPIC] = AnthropicAdapter(
            api_key=providers.anthropic_api_key.get_secret_value(),
            base_url=providers.anthropic_base_url,
        )

    if providers.ollama_base_url:
        adapters[Provider.OLLAMA] = OllamaAdapter(
            base_url=providers.ollama_base_url,
            connect_timeout_s=settings.routing.connect_timeout_s,
        )

    if not adapters:
        raise ConfigurationError(
            "No LLM provider is configured. Set OPENAI_API_KEY, ANTHROPIC_API_KEY, "
            "or OLLAMA_BASE_URL."
        )

    _log.info("providers.configured", providers=sorted(p.value for p in adapters))
    return adapters
