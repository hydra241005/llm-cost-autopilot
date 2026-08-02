"""Provider health manager tests."""

from __future__ import annotations

import pytest

from autopilot.domain.enums import BreakerState, Provider
from autopilot.domain.errors import (
    CircuitOpenError,
    ProviderBadRequest,
    ProviderServerError,
    ProviderTimeout,
)
from autopilot.infrastructure.clock import FrozenClock
from autopilot.infrastructure.observability.metrics import ProviderMetricsCollector
from autopilot.infrastructure.resilience.breaker import BreakerPolicy
from autopilot.infrastructure.resilience.health import ProviderHealthManager

OPENAI = Provider.OPENAI
POLICY = BreakerPolicy(failure_threshold=3, failure_rate=0.5, window_s=30.0, cooldown_s=20.0)


def _server_error() -> ProviderServerError:
    return ProviderServerError("boom", provider="openai", model_id="openai:gpt-4o-mini")


@pytest.fixture
def clock() -> FrozenClock:
    return FrozenClock()


@pytest.fixture
def manager(clock: FrozenClock) -> ProviderHealthManager:
    return ProviderHealthManager(
        [OPENAI, Provider.ANTHROPIC],
        breaker_policy=POLICY,
        metrics=ProviderMetricsCollector(window_s=60.0, clock=clock),
        clock=clock,
    )


def test_tracked_providers_are_exposed(manager: ProviderHealthManager) -> None:
    assert manager.providers == frozenset({OPENAI, Provider.ANTHROPIC})


def test_healthy_provider_passes_check(manager: ProviderHealthManager) -> None:
    manager.check(OPENAI)
    assert manager.is_available(OPENAI) is True


def test_unknown_provider_is_treated_as_available(manager: ProviderHealthManager) -> None:
    manager.check(Provider.OLLAMA)
    assert manager.is_available(Provider.OLLAMA) is True


def test_repeated_server_errors_open_the_circuit(manager: ProviderHealthManager) -> None:
    for _ in range(3):
        manager.record_failure(OPENAI, _server_error())

    assert manager.is_available(OPENAI) is False
    with pytest.raises(CircuitOpenError):
        manager.check(OPENAI)


def test_bad_requests_never_trip_the_circuit(manager: ProviderHealthManager) -> None:
    for _ in range(10):
        manager.record_failure(
            OPENAI, ProviderBadRequest("nope", provider="openai", model_id="m")
        )

    assert manager.is_available(OPENAI) is True
    assert manager.health(OPENAI).metrics.bad_requests == 10


def test_rejected_call_is_recorded_but_not_double_counted(
    manager: ProviderHealthManager,
) -> None:
    for _ in range(3):
        manager.record_failure(OPENAI, _server_error())
    with pytest.raises(CircuitOpenError) as excinfo:
        manager.check(OPENAI)
    manager.record_failure(OPENAI, excinfo.value)

    assert manager.health(OPENAI).metrics.circuit_open_rejections == 1


def test_success_after_cooldown_closes_the_circuit(
    manager: ProviderHealthManager, clock: FrozenClock
) -> None:
    for _ in range(3):
        manager.record_failure(OPENAI, _server_error())
    clock.advance(21.0)

    manager.check(OPENAI)
    manager.record_success(OPENAI, 120)
    assert manager.health(OPENAI).state is BreakerState.CLOSED


def test_one_provider_outage_does_not_affect_another(manager: ProviderHealthManager) -> None:
    for _ in range(3):
        manager.record_failure(OPENAI, _server_error())

    assert manager.is_available(OPENAI) is False
    assert manager.is_available(Provider.ANTHROPIC) is True


def test_health_view_reports_state_and_metrics(manager: ProviderHealthManager) -> None:
    manager.record_success(OPENAI, 100)
    manager.record_failure(OPENAI, ProviderTimeout("slow", provider="openai", model_id="m"))

    view = manager.health(OPENAI)
    assert view.provider is OPENAI
    assert view.state is BreakerState.CLOSED
    assert view.healthy is True
    assert view.failure_count == 1
    assert view.metrics.timeouts == 1
    assert view.metrics.p95_latency_ms == 100


def test_health_all_is_ordered_and_complete(manager: ProviderHealthManager) -> None:
    views = manager.health_all()
    assert [v.provider.value for v in views] == ["anthropic", "openai"]


def test_reset_reopens_a_tripped_provider(manager: ProviderHealthManager) -> None:
    for _ in range(3):
        manager.record_failure(OPENAI, _server_error())
    manager.reset(OPENAI)
    assert manager.is_available(OPENAI) is True


def test_reset_without_a_provider_clears_all(manager: ProviderHealthManager) -> None:
    for _ in range(3):
        manager.record_failure(OPENAI, _server_error())
    for _ in range(3):
        manager.record_failure(Provider.ANTHROPIC, _server_error())
    manager.reset()
    assert all(v.healthy for v in manager.health_all())


def test_metrics_collector_is_exposed(manager: ProviderHealthManager) -> None:
    assert isinstance(manager.metrics, ProviderMetricsCollector)
