from __future__ import annotations

import pytest

from autopilot.domain.errors import (
    ProviderBadRequest,
    ProviderRateLimited,
    ProviderServerError,
    ProviderTimeout,
)
from autopilot.infrastructure.resilience.retry import RetryPolicy, call_with_retry


@pytest.fixture
def no_sleep():
    slept: list[float] = []

    async def _sleep(seconds: float) -> None:
        slept.append(seconds)

    return slept, _sleep


async def test_returns_first_successful_result(no_sleep):
    _, sleep = no_sleep
    calls = 0

    async def op() -> str:
        nonlocal calls
        calls += 1
        return "ok"

    assert await call_with_retry(op, RetryPolicy(), sleep=sleep) == "ok"
    assert calls == 1


async def test_retries_transient_failure_then_succeeds(no_sleep):
    slept, sleep = no_sleep
    attempts = 0

    async def op() -> str:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise ProviderServerError("boom", provider="openai", model_id="m")
        return "recovered"

    result = await call_with_retry(op, RetryPolicy(max_retries=2), sleep=sleep)

    assert result == "recovered"
    assert attempts == 2
    assert len(slept) == 1


async def test_bad_request_is_never_retried(no_sleep):
    slept, sleep = no_sleep
    attempts = 0

    async def op() -> str:
        nonlocal attempts
        attempts += 1
        raise ProviderBadRequest("invalid", provider="openai", model_id="m")

    with pytest.raises(ProviderBadRequest):
        await call_with_retry(op, RetryPolicy(max_retries=2), sleep=sleep)

    assert attempts == 1
    assert slept == []


async def test_exhausts_retries_and_reraises_last_error(no_sleep):
    slept, sleep = no_sleep
    attempts = 0

    async def op() -> str:
        nonlocal attempts
        attempts += 1
        raise ProviderTimeout("slow", provider="openai", model_id="m")

    with pytest.raises(ProviderTimeout):
        await call_with_retry(op, RetryPolicy(max_retries=2), sleep=sleep)

    assert attempts == 3  # initial call plus two retries
    assert len(slept) == 2


async def test_retry_after_header_overrides_backoff(no_sleep):
    slept, sleep = no_sleep

    async def op() -> str:
        raise ProviderRateLimited(
            "slow down", provider="openai", model_id="m", retry_after_s=2.5
        )

    with pytest.raises(ProviderRateLimited):
        await call_with_retry(op, RetryPolicy(max_retries=1, max_delay_s=8.0), sleep=sleep)

    assert slept == [2.5]


def test_retry_after_is_capped_at_max_delay():
    policy = RetryPolicy(max_delay_s=5.0)
    assert policy.delay_for(0, retry_after_s=60.0) == 5.0


def test_jittered_delay_stays_within_the_exponential_ceiling():
    policy = RetryPolicy(base_delay_s=0.5, max_delay_s=8.0)
    for attempt in range(4):
        ceiling = min(0.5 * (2**attempt), 8.0)
        assert 0.0 <= policy.delay_for(attempt) <= ceiling


async def test_zero_retries_makes_exactly_one_call(no_sleep):
    _, sleep = no_sleep
    attempts = 0

    async def op() -> str:
        nonlocal attempts
        attempts += 1
        raise ProviderServerError("boom", provider="openai", model_id="m")

    with pytest.raises(ProviderServerError):
        await call_with_retry(op, RetryPolicy(max_retries=0), sleep=sleep)

    assert attempts == 1
