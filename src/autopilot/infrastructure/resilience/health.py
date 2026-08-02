"""Provider health management.

One object owns the answer to "can I call this provider right now, and how has
it been behaving?" It holds a circuit breaker per provider and feeds the metrics
collector, so the gateway, the health endpoint, and (from Phase 2) the routing
engine all read a single consistent view instead of each keeping their own.

Guarding at the *provider* level rather than per model is deliberate: when a
vendor has an outage, every one of its models is affected, and opening one
circuit should protect all of them.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from autopilot.domain.enums import BreakerState, Provider
from autopilot.domain.errors import CircuitOpenError
from autopilot.domain.interfaces import Clock
from autopilot.infrastructure.clock import SystemClock
from autopilot.infrastructure.observability.logging import get_logger
from autopilot.infrastructure.observability.metrics import (
    ProviderMetrics,
    ProviderMetricsCollector,
    classify_failure,
)
from autopilot.infrastructure.resilience.breaker import BreakerPolicy, CircuitBreaker

_log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class ProviderHealth:
    """Combined circuit state and metrics for one provider.

    Attributes:
        provider: The provider described.
        state: Current circuit state.
        healthy: Whether calls are currently being admitted.
        failure_count: Failures inside the breaker's rolling window.
        metrics: Rolling call statistics.
    """

    provider: Provider
    state: BreakerState
    healthy: bool
    failure_count: int
    metrics: ProviderMetrics


class ProviderHealthManager:
    """Tracks circuit state and call metrics for every configured provider."""

    def __init__(
        self,
        providers: Iterable[Provider],
        *,
        breaker_policy: BreakerPolicy | None = None,
        metrics: ProviderMetricsCollector | None = None,
        clock: Clock | None = None,
    ) -> None:
        """Create a health manager.

        Args:
            providers: Providers to track; one breaker is created per provider.
            breaker_policy: Thresholds shared by every breaker.
            metrics: Collector to feed; created if not supplied.
            clock: Time source, injected by tests.
        """
        resolved_clock = clock or SystemClock()
        self._policy = breaker_policy or BreakerPolicy()
        self._metrics = metrics or ProviderMetricsCollector(clock=resolved_clock)
        self._breakers = {
            provider: CircuitBreaker(provider.value, self._policy, clock=resolved_clock)
            for provider in providers
        }

    @property
    def providers(self) -> frozenset[Provider]:
        """Providers this manager tracks."""
        return frozenset(self._breakers)

    @property
    def metrics(self) -> ProviderMetricsCollector:
        """The underlying metrics collector."""
        return self._metrics

    def is_available(self, provider: Provider) -> bool:
        """Return whether ``provider`` is currently accepting calls.

        Unknown providers are treated as available; the gateway rejects them
        earlier for the more precise reason that no adapter is configured.
        """
        breaker = self._breakers.get(provider)
        return True if breaker is None else breaker.state is not BreakerState.OPEN

    def check(self, provider: Provider) -> None:
        """Raise if ``provider``'s circuit is open.

        Called immediately before dispatch so a known-broken provider fails in
        microseconds instead of consuming its full timeout budget.

        Raises:
            CircuitOpenError: The circuit is open, or a trial call is already
                in flight while half-open.
        """
        breaker = self._breakers.get(provider)
        if breaker is None or breaker.allows():
            return
        self._metrics.record_failure(provider, "circuit_open")
        raise CircuitOpenError(
            f"Circuit is open for provider {provider.value!r}; "
            f"retrying in up to {self._policy.cooldown_s:.0f}s.",
            provider=provider.value,
        )

    def record_success(self, provider: Provider, latency_ms: int) -> None:
        """Record a successful call against ``provider``."""
        breaker = self._breakers.get(provider)
        if breaker is not None:
            breaker.record_success()
        self._metrics.record_success(provider, latency_ms)

    def record_failure(self, provider: Provider, error: Exception) -> None:
        """Record a failed call against ``provider``.

        Only failures the provider is responsible for move the circuit. A
        malformed request would fail identically against a perfectly healthy
        vendor, so counting it would open circuits for a caller-side bug.
        """
        kind = classify_failure(error)
        if kind == "circuit_open":
            return  # Already recorded by check(); do not double-count.

        self._metrics.record_failure(provider, kind)
        breaker = self._breakers.get(provider)
        if breaker is not None and kind != "bad_request":
            breaker.record_failure()

    def health(self, provider: Provider) -> ProviderHealth:
        """Return the combined health view for ``provider``."""
        breaker = self._breakers.get(provider)
        state = breaker.state if breaker is not None else BreakerState.CLOSED
        return ProviderHealth(
            provider=provider,
            state=state,
            healthy=state is not BreakerState.OPEN,
            failure_count=breaker.failure_count if breaker is not None else 0,
            metrics=self._metrics.snapshot(provider),
        )

    def health_all(self) -> list[ProviderHealth]:
        """Return health views for every tracked provider, in stable order."""
        return [self.health(p) for p in sorted(self._breakers, key=lambda p: p.value)]

    def reset(self, provider: Provider | None = None) -> None:
        """Close circuits and forget history, for one provider or all of them."""
        targets = [provider] if provider is not None else list(self._breakers)
        for target in targets:
            breaker = self._breakers.get(target)
            if breaker is not None:
                breaker.reset()
