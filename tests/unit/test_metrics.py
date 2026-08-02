"""Rolling provider-metrics tests."""

from __future__ import annotations

import pytest

from autopilot.domain.enums import Provider
from autopilot.domain.errors import (
    CircuitOpenError,
    ProviderBadRequest,
    ProviderRateLimited,
    ProviderServerError,
    ProviderTimeout,
)
from autopilot.infrastructure.clock import FrozenClock
from autopilot.infrastructure.observability.metrics import (
    ProviderMetricsCollector,
    classify_failure,
)

OPENAI = Provider.OPENAI


@pytest.fixture
def clock() -> FrozenClock:
    return FrozenClock()


@pytest.fixture
def collector(clock: FrozenClock) -> ProviderMetricsCollector:
    return ProviderMetricsCollector(window_s=60.0, clock=clock)


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (ProviderTimeout("t", provider="openai", model_id="m"), "timeout"),
        (ProviderRateLimited("r", provider="openai", model_id="m"), "rate_limited"),
        (ProviderServerError("s", provider="openai", model_id="m"), "server_error"),
        (ProviderBadRequest("b", provider="openai", model_id="m"), "bad_request"),
        (CircuitOpenError("c", provider="openai"), "circuit_open"),
        (ValueError("unexpected"), "bad_request"),
    ],
)
def test_classify_failure(error: Exception, expected: str) -> None:
    assert classify_failure(error) == expected


def test_idle_provider_reports_none_rather_than_a_misleading_hundred_percent(
    collector: ProviderMetricsCollector,
) -> None:
    snap = collector.snapshot(OPENAI)
    assert snap.total_calls == 0
    assert snap.success_rate is None
    assert snap.availability is None
    assert snap.p95_latency_ms is None


def test_counts_are_bucketed_by_failure_kind(collector: ProviderMetricsCollector) -> None:
    collector.record_success(OPENAI, 100)
    collector.record_failure(OPENAI, "timeout")
    collector.record_failure(OPENAI, "rate_limited")
    collector.record_failure(OPENAI, "server_error")
    collector.record_failure(OPENAI, "bad_request")

    snap = collector.snapshot(OPENAI)
    assert (snap.total_calls, snap.successes, snap.failures) == (5, 1, 4)
    assert (snap.timeouts, snap.rate_limited) == (1, 1)
    assert (snap.server_errors, snap.bad_requests) == (1, 1)


def test_circuit_rejections_lower_availability_not_success_rate(
    collector: ProviderMetricsCollector,
) -> None:
    for _ in range(8):
        collector.record_success(OPENAI, 50)
    for _ in range(2):
        collector.record_failure(OPENAI, "circuit_open")

    snap = collector.snapshot(OPENAI)
    assert snap.circuit_open_rejections == 2
    assert snap.success_rate == 1.0
    assert snap.availability == pytest.approx(0.8)


def test_percentiles_use_nearest_rank(collector: ProviderMetricsCollector) -> None:
    for ms in range(1, 101):
        collector.record_success(OPENAI, ms)
    snap = collector.snapshot(OPENAI)
    assert snap.p50_latency_ms == 50
    assert snap.p95_latency_ms == 95


def test_single_sample_percentiles(collector: ProviderMetricsCollector) -> None:
    collector.record_success(OPENAI, 42)
    snap = collector.snapshot(OPENAI)
    assert snap.p50_latency_ms == 42
    assert snap.p95_latency_ms == 42


def test_observations_age_out_of_the_window(
    collector: ProviderMetricsCollector, clock: FrozenClock
) -> None:
    collector.record_success(OPENAI, 10)
    collector.record_failure(OPENAI, "timeout")
    clock.advance(61.0)
    collector.record_success(OPENAI, 20)

    snap = collector.snapshot(OPENAI)
    assert snap.total_calls == 1
    assert snap.p50_latency_ms == 20


def test_providers_are_tracked_independently(collector: ProviderMetricsCollector) -> None:
    collector.record_success(OPENAI, 10)
    collector.record_failure(Provider.ANTHROPIC, "timeout")

    assert collector.snapshot(OPENAI).failures == 0
    assert collector.snapshot(Provider.ANTHROPIC).successes == 0


def test_snapshot_all_covers_every_requested_provider(
    collector: ProviderMetricsCollector,
) -> None:
    snaps = collector.snapshot_all([OPENAI, Provider.OLLAMA])
    assert set(snaps) == {OPENAI, Provider.OLLAMA}


def test_reset_discards_observations(collector: ProviderMetricsCollector) -> None:
    collector.record_success(OPENAI, 10)
    collector.reset()
    assert collector.snapshot(OPENAI).total_calls == 0


def test_window_is_exposed(collector: ProviderMetricsCollector) -> None:
    assert collector.window_s == 60.0
    assert collector.snapshot(OPENAI).window_s == 60.0
