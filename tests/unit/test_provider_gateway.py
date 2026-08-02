from __future__ import annotations

from decimal import Decimal

import pytest

from autopilot.application.provider_gateway import ProviderGateway
from autopilot.domain.enums import Provider, Tier
from autopilot.domain.errors import (
    CircuitOpenError,
    ConfigurationError,
    ModelNotFoundError,
    NoCapableModelError,
    ProviderBadRequest,
    ProviderServerError,
)
from autopilot.infrastructure.providers.registry import YamlModelRegistry
from autopilot.infrastructure.resilience.breaker import BreakerPolicy
from autopilot.infrastructure.resilience.health import ProviderHealthManager
from autopilot.infrastructure.resilience.retry import RetryPolicy
from tests.conftest import FakeProvider, make_model

FAST_RETRY = RetryPolicy(max_retries=2, base_delay_s=0.0, max_delay_s=0.0)


async def test_completes_through_the_registered_adapter(gateway, sample_request):
    outcome = await gateway.complete("openai:gpt-4o-mini", sample_request)

    assert outcome.response.text == "fake response"
    assert outcome.attempts == 1


async def test_cost_is_priced_from_the_registry_entry(gateway, sample_request):
    outcome = await gateway.complete("openai:gpt-4o-mini", sample_request)

    # 1000 in @ $0.15/MTok + 500 out @ $0.60/MTok
    assert outcome.cost_usd == Decimal("0.000450")


async def test_savings_are_measured_against_the_configured_baseline(gateway, sample_request):
    outcome = await gateway.complete("openai:gpt-4o-mini", sample_request)

    # baseline gpt-4o: 1000 in @ $2.50 + 500 out @ $10.00 = $0.0075
    assert outcome.cost.baseline_usd == Decimal("0.007500")
    assert outcome.saved_usd == Decimal("0.007050")


async def test_the_free_local_model_costs_nothing(gateway, sample_request):
    outcome = await gateway.complete("ollama:llama3.1-8b", sample_request)

    assert outcome.cost_usd == Decimal("0.000000")
    assert outcome.saved_usd == Decimal("0.007500")


async def test_transient_failures_are_retried(registry, sample_request):
    provider = FakeProvider(
        Provider.OPENAI,
        script=[ProviderServerError("flaky", provider="openai", model_id="openai:gpt-4o-mini")],
    )
    gw = ProviderGateway(
        registry=registry,
        adapters={Provider.OPENAI: provider},
        retry_policy=FAST_RETRY,
        baseline_model_id="openai:gpt-4o",
    )

    outcome = await gw.complete("openai:gpt-4o-mini", sample_request)

    assert outcome.attempts == 2
    assert len(provider.calls) == 2


async def test_non_retryable_failures_fail_immediately(registry, sample_request):
    provider = FakeProvider(
        Provider.OPENAI,
        script=[ProviderBadRequest("bad", provider="openai", model_id="openai:gpt-4o-mini")],
    )
    gw = ProviderGateway(
        registry=registry,
        adapters={Provider.OPENAI: provider},
        retry_policy=FAST_RETRY,
    )

    with pytest.raises(ProviderBadRequest):
        await gw.complete("openai:gpt-4o-mini", sample_request)

    assert len(provider.calls) == 1


async def test_unknown_model_raises_model_not_found(gateway, sample_request):
    with pytest.raises(ModelNotFoundError):
        await gateway.complete("openai:does-not-exist", sample_request)


async def test_inactive_model_raises_no_capable_model(sample_request):
    registry = YamlModelRegistry([make_model("openai:gpt-4o-mini", active=False)])
    gw = ProviderGateway(
        registry=registry,
        adapters={Provider.OPENAI: FakeProvider(Provider.OPENAI)},
        retry_policy=FAST_RETRY,
    )

    with pytest.raises(NoCapableModelError, match="deactivated"):
        await gw.complete("openai:gpt-4o-mini", sample_request)


async def test_unconfigured_provider_raises_no_capable_model(registry, sample_request):
    gw = ProviderGateway(
        registry=registry,
        adapters={Provider.OPENAI: FakeProvider(Provider.OPENAI)},
        retry_policy=FAST_RETRY,
    )

    with pytest.raises(NoCapableModelError, match="No adapter configured"):
        await gw.complete("anthropic:claude-sonnet-5", sample_request)


def test_gateway_requires_at_least_one_adapter(registry):
    with pytest.raises(ConfigurationError):
        ProviderGateway(registry=registry, adapters={})


def test_supports_reflects_registration_and_adapter_availability(registry):
    gw = ProviderGateway(
        registry=registry,
        adapters={Provider.OPENAI: FakeProvider(Provider.OPENAI)},
    )

    assert gw.supports("openai:gpt-4o-mini")
    assert not gw.supports("anthropic:claude-sonnet-5")
    assert not gw.supports("openai:does-not-exist")


def test_available_providers_lists_configured_adapters(gateway):
    assert gateway.available_providers == frozenset(
        {Provider.OPENAI, Provider.ANTHROPIC, Provider.OLLAMA}
    )


async def test_tier_timeout_budget_is_passed_to_the_adapter(registry, sample_request):
    provider = FakeProvider(Provider.OPENAI)
    gw = ProviderGateway(
        registry=registry,
        adapters={Provider.OPENAI: provider},
        retry_policy=FAST_RETRY,
        timeouts_by_tier={1: 10.0, 2: 20.0, 3: 45.0},
    )

    await gw.complete("openai:gpt-4o", sample_request)

    assert provider.calls[0][2] == 45.0


async def test_explicit_timeout_overrides_the_tier_budget(registry, sample_request):
    provider = FakeProvider(Provider.OPENAI)
    gw = ProviderGateway(
        registry=registry,
        adapters={Provider.OPENAI: provider},
        retry_policy=FAST_RETRY,
    )

    await gw.complete("openai:gpt-4o-mini", sample_request, timeout_s=1.5)

    assert provider.calls[0][2] == 1.5


async def test_baseline_defaults_to_the_most_expensive_registered_model(sample_request):
    registry = YamlModelRegistry(
        [
            make_model("openai:gpt-4o-mini", tier=Tier.SIMPLE),
            make_model("openai:gpt-4o", tier=Tier.COMPLEX,
                       input_cost="2.50", output_cost="10.00"),
        ]
    )
    gw = ProviderGateway(
        registry=registry,
        adapters={Provider.OPENAI: FakeProvider(Provider.OPENAI)},
        retry_policy=FAST_RETRY,
    )

    outcome = await gw.complete("openai:gpt-4o-mini", sample_request)

    assert outcome.cost.baseline_usd == Decimal("0.007500")


async def test_single_free_model_registry_reports_zero_savings(sample_request):
    registry = YamlModelRegistry(
        [
            make_model("ollama:llama3.1-8b", provider=Provider.OLLAMA,
                       input_cost="0", output_cost="0"),
        ]
    )
    gw = ProviderGateway(
        registry=registry,
        adapters={Provider.OLLAMA: FakeProvider(Provider.OLLAMA)},
        retry_policy=FAST_RETRY,
    )

    outcome = await gw.complete("ollama:llama3.1-8b", sample_request)

    assert outcome.saved_usd == Decimal("0.000000")


async def test_aclose_closes_every_adapter(registry):
    adapters = {
        Provider.OPENAI: FakeProvider(Provider.OPENAI),
        Provider.OLLAMA: FakeProvider(Provider.OLLAMA),
    }
    gw = ProviderGateway(registry=registry, adapters=adapters)

    await gw.aclose()

    assert all(a.closed for a in adapters.values())


async def test_successful_calls_are_recorded_in_provider_health(gateway, sample_request):
    await gateway.complete("openai:gpt-4o-mini", sample_request)

    view = gateway.health.health(Provider.OPENAI)
    assert view.metrics.successes == 1
    assert view.metrics.success_rate == 1.0


async def test_failures_are_recorded_and_can_open_the_circuit(registry, sample_request):
    failures = [
        ProviderServerError("boom", provider="openai", model_id="openai:gpt-4o-mini")
        for _ in range(30)
    ]
    provider = FakeProvider(Provider.OPENAI, script=failures)
    gw = ProviderGateway(
        registry=registry,
        adapters={Provider.OPENAI: provider},
        retry_policy=FAST_RETRY,
        health=ProviderHealthManager(
            [Provider.OPENAI],
            breaker_policy=BreakerPolicy(failure_threshold=2, failure_rate=0.5),
        ),
    )

    # The circuit trips on the second failure, so the third attempt is rejected
    # locally rather than spending another round trip on a known-broken vendor.
    with pytest.raises(CircuitOpenError):
        await gw.complete("openai:gpt-4o-mini", sample_request)

    assert len(provider.calls) == 2
    assert gw.health.is_available(Provider.OPENAI) is False


async def test_an_open_circuit_short_circuits_the_next_call(registry, sample_request):
    provider = FakeProvider(Provider.OPENAI)
    health = ProviderHealthManager(
        [Provider.OPENAI],
        breaker_policy=BreakerPolicy(failure_threshold=1, failure_rate=0.5),
    )
    gw = ProviderGateway(
        registry=registry,
        adapters={Provider.OPENAI: provider},
        retry_policy=FAST_RETRY,
        health=health,
    )
    health.record_failure(
        Provider.OPENAI,
        ProviderServerError("boom", provider="openai", model_id="openai:gpt-4o-mini"),
    )

    with pytest.raises(CircuitOpenError):
        await gw.complete("openai:gpt-4o-mini", sample_request)

    assert provider.calls == []


async def test_a_default_health_manager_covers_every_adapter(gateway):
    assert gateway.health.providers == gateway.available_providers
