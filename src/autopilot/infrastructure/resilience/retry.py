"""Retry with exponential backoff and full jitter.

Retry eligibility is decided entirely by :attr:`ProviderError.retryable`, which
the adapters set when they map vendor failures. Nothing here inspects an HTTP
status code — that knowledge belongs in the adapter that produced the error.

Full jitter (``uniform(0, base * 2**n)``) rather than fixed backoff: when a
provider returns 429 to many concurrent callers, fixed backoff re-synchronizes
them into a second thundering herd.
"""

from __future__ import annotations

import asyncio
import random
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TypeVar

from autopilot.domain.errors import ProviderError
from autopilot.infrastructure.observability.logging import get_logger

_log = get_logger(__name__)

T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    """Backoff parameters for a retried provider call.

    Attributes:
        max_retries: Attempts after the first. ``2`` means up to three calls.
        base_delay_s: Backoff base; delay for attempt ``n`` is drawn from
            ``[0, base_delay_s * 2**n]``.
        max_delay_s: Ceiling applied before jitter is drawn.
    """

    max_retries: int = 2
    base_delay_s: float = 0.5
    max_delay_s: float = 8.0

    def delay_for(self, attempt: int, *, retry_after_s: float | None = None) -> float:
        """Return how long to wait before ``attempt``.

        A provider-supplied ``Retry-After`` always wins: the vendor knows when its
        own quota resets better than our backoff curve does.

        Args:
            attempt: Zero-based index of the retry about to be made.
            retry_after_s: Cooldown advertised by the provider, if any.

        Returns:
            Delay in seconds.
        """
        if retry_after_s is not None:
            return min(retry_after_s, self.max_delay_s)
        ceiling = min(self.base_delay_s * (2**attempt), self.max_delay_s)
        return random.uniform(0.0, ceiling)


async def call_with_retry(
    operation: Callable[[], Awaitable[T]],
    policy: RetryPolicy,
    *,
    context: dict[str, str] | None = None,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> T:
    """Invoke ``operation``, retrying retryable provider failures.

    Args:
        operation: Zero-argument coroutine factory to invoke. Must be safe to
            call more than once.
        policy: Backoff parameters.
        context: Extra fields for log lines, e.g. model id and provider.
        sleep: Injected for tests that must not actually wait.

    Returns:
        Whatever ``operation`` returns on its first successful attempt.

    Raises:
        ProviderError: The last failure, once retries are exhausted or the error
            is not retryable.
    """
    fields = context or {}
    last_error: ProviderError

    for attempt in range(policy.max_retries + 1):
        try:
            return await operation()
        except ProviderError as exc:
            last_error = exc
            exhausted = attempt >= policy.max_retries
            if not exc.retryable or exhausted:
                _log.warning(
                    "provider.call_failed",
                    attempt=attempt,
                    retryable=exc.retryable,
                    error_code=exc.code,
                    **fields,
                )
                raise
            delay = policy.delay_for(attempt, retry_after_s=exc.retry_after_s)
            _log.info(
                "provider.retry",
                attempt=attempt,
                delay_s=round(delay, 3),
                error_code=exc.code,
                **fields,
            )
            await sleep(delay)

    raise last_error
