"""Circuit breaker state-machine tests, driven by a frozen clock."""

from __future__ import annotations

import pytest

from autopilot.domain.enums import BreakerState
from autopilot.infrastructure.clock import FrozenClock
from autopilot.infrastructure.resilience.breaker import BreakerPolicy, CircuitBreaker

POLICY = BreakerPolicy(failure_threshold=3, failure_rate=0.5, window_s=30.0, cooldown_s=20.0)


@pytest.fixture
def clock() -> FrozenClock:
    return FrozenClock()


@pytest.fixture
def breaker(clock: FrozenClock) -> CircuitBreaker:
    return CircuitBreaker("openai", POLICY, clock=clock)


def test_starts_closed_and_allows(breaker: CircuitBreaker) -> None:
    assert breaker.state is BreakerState.CLOSED
    assert breaker.allows() is True


def test_opens_when_count_and_rate_are_both_breached(breaker: CircuitBreaker) -> None:
    for _ in range(3):
        breaker.record_failure()
    assert breaker.state is BreakerState.OPEN
    assert breaker.allows() is False


def test_count_alone_does_not_open_when_rate_is_low(breaker: CircuitBreaker) -> None:
    for _ in range(50):
        breaker.record_success()
    for _ in range(3):
        breaker.record_failure()
    assert breaker.state is BreakerState.CLOSED


def test_rate_alone_does_not_open_below_the_count_threshold(breaker: CircuitBreaker) -> None:
    breaker.record_failure()
    breaker.record_failure()
    assert breaker.failure_count == 2
    assert breaker.state is BreakerState.CLOSED


def test_observations_age_out_of_the_window(breaker: CircuitBreaker, clock: FrozenClock) -> None:
    breaker.record_failure()
    breaker.record_failure()
    clock.advance(31.0)
    breaker.record_failure()
    assert breaker.failure_count == 1
    assert breaker.state is BreakerState.CLOSED


def test_half_opens_after_cooldown(breaker: CircuitBreaker, clock: FrozenClock) -> None:
    for _ in range(3):
        breaker.record_failure()
    clock.advance(19.9)
    assert breaker.state is BreakerState.OPEN
    clock.advance(0.2)
    assert breaker.state is BreakerState.HALF_OPEN


def test_half_open_admits_exactly_one_trial(breaker: CircuitBreaker, clock: FrozenClock) -> None:
    for _ in range(3):
        breaker.record_failure()
    clock.advance(21.0)
    assert breaker.allows() is True
    assert breaker.allows() is False


def test_successful_trial_closes_the_circuit(breaker: CircuitBreaker, clock: FrozenClock) -> None:
    for _ in range(3):
        breaker.record_failure()
    clock.advance(21.0)
    assert breaker.allows() is True
    breaker.record_success()
    assert breaker.state is BreakerState.CLOSED
    assert breaker.failure_count == 0


def test_failed_trial_serves_another_cooldown(breaker: CircuitBreaker, clock: FrozenClock) -> None:
    for _ in range(3):
        breaker.record_failure()
    clock.advance(21.0)
    assert breaker.allows() is True
    breaker.record_failure()
    assert breaker.state is BreakerState.OPEN
    clock.advance(19.0)
    assert breaker.state is BreakerState.OPEN
    clock.advance(2.0)
    assert breaker.state is BreakerState.HALF_OPEN


def test_reset_closes_and_clears_history(breaker: CircuitBreaker) -> None:
    for _ in range(3):
        breaker.record_failure()
    breaker.reset()
    assert breaker.state is BreakerState.CLOSED
    assert breaker.failure_count == 0


def test_policy_is_exposed(breaker: CircuitBreaker) -> None:
    assert breaker.policy is POLICY


def test_default_policy_matches_the_blueprint() -> None:
    policy = CircuitBreaker("anthropic").policy
    assert (policy.failure_threshold, policy.failure_rate) == (5, 0.5)
    assert (policy.window_s, policy.cooldown_s) == (30.0, 20.0)
