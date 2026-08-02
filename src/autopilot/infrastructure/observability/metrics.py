"""Rolling per-provider call metrics.

Every provider call is recorded here so the health endpoint, the dashboard, and
the routing engine can all read the same numbers. Observations age out of a
rolling window, which keeps memory bounded and makes the figures reflect *current*
health rather than a lifetime average that a long-past outage would poison.

Latency percentiles are computed by nearest-rank over the retained samples. That
is exact for the window rather than approximate, which is affordable because the
window is small and bounded.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

from autopilot.domain.enums import Provider
from autopilot.domain.errors import (
    CircuitOpenError,
    ProviderError,
    ProviderRateLimited,
    ProviderTimeout,
)
from autopilot.domain.interfaces import Clock
from autopilot.infrastructure.clock import SystemClock

#: Why a call failed, tracked separately because each implies a different fix.
FailureKind = Literal["timeout", "rate_limited", "server_error", "bad_request", "circuit_open"]

#: Hard cap on retained samples per provider, so a burst cannot grow memory.
_MAX_SAMPLES = 1_000


def classify_failure(error: Exception) -> FailureKind:
    """Return the metrics bucket for a failed provider call.

    Args:
        error: The exception raised by the provider call.

    Returns:
        The failure category to increment.
    """
    if isinstance(error, ProviderTimeout):
        return "timeout"
    if isinstance(error, ProviderRateLimited):
        return "rate_limited"
    if isinstance(error, CircuitOpenError):
        return "circuit_open"
    if isinstance(error, ProviderError) and error.retryable:
        return "server_error"
    return "bad_request"


@dataclass(frozen=True, slots=True)
class ProviderMetrics:
    """A point-in-time view of one provider's health.

    Attributes:
        provider: The provider these figures describe.
        window_s: Width of the rolling window the figures cover.
        total_calls: Calls observed in the window.
        successes: Calls that returned a response.
        failures: Calls that raised.
        timeouts: Failures caused by exceeding the timeout budget.
        rate_limited: Failures caused by HTTP 429.
        server_errors: Retryable upstream failures (5xx and connection errors).
        bad_requests: Non-retryable failures attributable to the request.
        circuit_open_rejections: Calls short-circuited without reaching the vendor.
        success_rate: Successes over total calls; ``None`` when idle.
        availability: Share of calls not rejected by an open circuit.
        p50_latency_ms: Median latency of successful calls.
        p95_latency_ms: 95th-percentile latency of successful calls.
    """

    provider: Provider
    window_s: float
    total_calls: int
    successes: int
    failures: int
    timeouts: int
    rate_limited: int
    server_errors: int
    bad_requests: int
    circuit_open_rejections: int
    success_rate: float | None
    availability: float | None
    p50_latency_ms: int | None
    p95_latency_ms: int | None


class ProviderMetricsCollector:
    """Accumulates call outcomes per provider over a rolling window.

    Not thread-safe by design: it is mutated only from the asyncio event loop,
    where operations between awaits are already serialized. Adding a lock would
    cost contention for no benefit.
    """

    def __init__(self, window_s: float = 300.0, *, clock: Clock | None = None) -> None:
        """Create the collector.

        Args:
            window_s: How far back the reported figures reach, in seconds.
            clock: Time source, injected by tests.
        """
        self._window_s = window_s
        self._clock = clock or SystemClock()
        self._successes: dict[Provider, deque[tuple[float, int]]] = {}
        self._failures: dict[Provider, deque[tuple[float, FailureKind]]] = {}

    @property
    def window_s(self) -> float:
        """Width of the rolling window in seconds."""
        return self._window_s

    def record_success(self, provider: Provider, latency_ms: int) -> None:
        """Record a successful call and its latency.

        Args:
            provider: The provider that answered.
            latency_ms: Observed round-trip latency in milliseconds.
        """
        series = self._successes.setdefault(provider, deque(maxlen=_MAX_SAMPLES))
        series.append((self._clock.monotonic(), latency_ms))

    def record_failure(self, provider: Provider, kind: FailureKind) -> None:
        """Record a failed call.

        Args:
            provider: The provider that failed.
            kind: Why it failed, from :func:`classify_failure`.
        """
        series = self._failures.setdefault(provider, deque(maxlen=_MAX_SAMPLES))
        series.append((self._clock.monotonic(), kind))

    def snapshot(self, provider: Provider) -> ProviderMetrics:
        """Return current metrics for ``provider``.

        Args:
            provider: The provider to report on.

        Returns:
            Metrics covering the rolling window. A provider with no observed
            calls reports zeroes and ``None`` rates rather than a misleading
            100%.
        """
        cutoff = self._clock.monotonic() - self._window_s
        latencies = sorted(
            ms for at, ms in self._successes.get(provider, ()) if at >= cutoff
        )
        kinds = [kind for at, kind in self._failures.get(provider, ()) if at >= cutoff]

        successes = len(latencies)
        failures = len(kinds)
        total = successes + failures
        rejected = kinds.count("circuit_open")
        attempted = total - rejected

        return ProviderMetrics(
            provider=provider,
            window_s=self._window_s,
            total_calls=total,
            successes=successes,
            failures=failures,
            timeouts=kinds.count("timeout"),
            rate_limited=kinds.count("rate_limited"),
            server_errors=kinds.count("server_error"),
            bad_requests=kinds.count("bad_request"),
            circuit_open_rejections=rejected,
            # Success rate is measured over calls that actually reached the
            # vendor; circuit rejections are an availability problem, not a
            # vendor-quality one, and are reported separately.
            success_rate=(successes / attempted) if attempted else None,
            availability=(attempted / total) if total else None,
            p50_latency_ms=_percentile(latencies, 0.50),
            p95_latency_ms=_percentile(latencies, 0.95),
        )

    def snapshot_all(self, providers: Iterable[Provider]) -> dict[Provider, ProviderMetrics]:
        """Return metrics for every provider in ``providers``."""
        return {p: self.snapshot(p) for p in providers}

    def reset(self) -> None:
        """Discard all observations."""
        self._successes.clear()
        self._failures.clear()


def _percentile(sorted_values: list[int], fraction: float) -> int | None:
    """Return the nearest-rank percentile of a pre-sorted list.

    Args:
        sorted_values: Ascending samples.
        fraction: Percentile expressed as a fraction, e.g. ``0.95``.

    Returns:
        The percentile value, or ``None`` when there are no samples.
    """
    if not sorted_values:
        return None
    rank = max(1, math.ceil(fraction * len(sorted_values)))
    return sorted_values[rank - 1]
