"""Shared pytest fixtures and provider fakes."""

from __future__ import annotations

from decimal import Decimal

import pytest

from autopilot.application.provider_gateway import ProviderGateway
from autopilot.domain.entities import (
    CompletionInput,
    Message,
    ModelConfig,
    ProviderResponse,
    Usage,
)
from autopilot.domain.enums import FinishReason, Provider, Role, Tier
from autopilot.domain.errors import ProviderError
from autopilot.domain.interfaces import LLMProvider
from autopilot.infrastructure.providers.registry import YamlModelRegistry
from autopilot.infrastructure.resilience.retry import RetryPolicy


def make_model(
    model_id: str = "openai:gpt-4o-mini",
    *,
    provider: Provider = Provider.OPENAI,
    tier: Tier = Tier.SIMPLE,
    input_cost: str = "0.15",
    output_cost: str = "0.60",
    active: bool = True,
    supports_temperature: bool = True,
    max_context_tokens: int = 128_000,
    max_output_tokens: int = 16_384,
) -> ModelConfig:
    return ModelConfig(
        id=model_id,
        provider=provider,
        vendor_model_id=model_id.split(":", 1)[1],
        tier=tier,
        input_cost_per_mtok=Decimal(input_cost),
        output_cost_per_mtok=Decimal(output_cost),
        max_context_tokens=max_context_tokens,
        max_output_tokens=max_output_tokens,
        expected_latency_ms=1000,
        active=active,
        supports_temperature=supports_temperature,
    )


class FakeProvider(LLMProvider):
    """A scripted provider: returns queued responses or raises queued errors."""

    def __init__(
        self,
        name: Provider = Provider.OPENAI,
        *,
        script: list[ProviderResponse | ProviderError] | None = None,
        text: str = "fake response",
    ) -> None:
        self.name = name
        self._script = list(script or [])
        self._text = text
        self.calls: list[tuple[CompletionInput, ModelConfig, float]] = []
        self.closed = False

    async def complete(
        self, req: CompletionInput, cfg: ModelConfig, timeout_s: float
    ) -> ProviderResponse:
        self.calls.append((req, cfg, timeout_s))
        if self._script:
            item = self._script.pop(0)
            if isinstance(item, ProviderError):
                raise item
            return item
        return ProviderResponse(
            model_id=cfg.id,
            provider=self.name,
            text=self._text,
            usage=Usage(input_tokens=1000, output_tokens=500),
            finish_reason=FinishReason.STOP,
            latency_ms=42,
        )

    def count_tokens(self, text: str) -> int:
        return max(1, len(text) // 4)

    async def aclose(self) -> None:
        self.closed = True


@pytest.fixture
def sample_request() -> CompletionInput:
    return CompletionInput(
        messages=(
            Message(role=Role.SYSTEM, content="You are terse."),
            Message(role=Role.USER, content="Summarize the theory of relativity."),
        ),
        max_tokens=256,
        temperature=0.2,
    )


@pytest.fixture
def registry() -> YamlModelRegistry:
    return YamlModelRegistry(
        [
            make_model("ollama:llama3.1-8b", provider=Provider.OLLAMA, tier=Tier.SIMPLE,
                       input_cost="0", output_cost="0"),
            make_model("openai:gpt-4o-mini", tier=Tier.MODERATE),
            make_model("anthropic:claude-sonnet-5", provider=Provider.ANTHROPIC,
                       tier=Tier.MODERATE, input_cost="3.00", output_cost="15.00",
                       supports_temperature=False),
            make_model("openai:gpt-4o", tier=Tier.COMPLEX,
                       input_cost="2.50", output_cost="10.00"),
        ]
    )


@pytest.fixture
def gateway(registry: YamlModelRegistry) -> ProviderGateway:
    return ProviderGateway(
        registry=registry,
        adapters={
            Provider.OPENAI: FakeProvider(Provider.OPENAI),
            Provider.ANTHROPIC: FakeProvider(Provider.ANTHROPIC),
            Provider.OLLAMA: FakeProvider(Provider.OLLAMA),
        },
        retry_policy=RetryPolicy(max_retries=2, base_delay_s=0.0, max_delay_s=0.0),
        baseline_model_id="openai:gpt-4o",
    )
