"""Shared provider-adapter machinery.

Holds what every adapter needs and no adapter should reimplement: HTTP status →
domain error mapping, ``Retry-After`` parsing, a latency timer, and a heuristic
token estimator for vendors that do not report usage.

Adapters are intentionally *not* forced onto a single HTTP client. The Anthropic
and OpenAI adapters use their official SDKs (which track vendor API drift for us);
Ollama, which has no SDK, uses ``httpx`` directly. The port boundary makes the
choice invisible to callers.
"""

from __future__ import annotations

import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager

from autopilot.domain.enums import FinishReason
from autopilot.domain.errors import (
    ProviderBadRequest,
    ProviderError,
    ProviderRateLimited,
    ProviderServerError,
    ProviderTimeout,
)

#: Rough characters-per-token ratio for English prose. Used only when a vendor
#: reports no usage and no tokenizer is installed; real counts always win.
_CHARS_PER_TOKEN = 4


@contextmanager
def latency_timer() -> Iterator[list[int]]:
    """Measure wall-clock milliseconds for the enclosed block.

    Yields:
        A single-element list that holds the elapsed milliseconds once the block
        exits. A list is used because the value is not known until then.
    """
    holder: list[int] = [0]
    start = time.perf_counter()
    try:
        yield holder
    finally:
        holder[0] = int((time.perf_counter() - start) * 1000)


def estimate_tokens(text: str) -> int:
    """Return a cheap token-count estimate for ``text``.

    Args:
        text: The text to measure.

    Returns:
        Estimated tokens, never less than one for non-empty text.
    """
    if not text:
        return 0
    return max(1, len(text) // _CHARS_PER_TOKEN)


def parse_retry_after(headers: Mapping[str, str] | None) -> float | None:
    """Extract a ``Retry-After`` delay in seconds from response headers.

    Only the delta-seconds form is honoured; the HTTP-date form is rare from LLM
    vendors and falling back to normal backoff is safe.

    Args:
        headers: Response headers, case-insensitively searched.

    Returns:
        The advertised delay, or ``None`` when absent or unparseable.
    """
    if not headers:
        return None
    for key, value in headers.items():
        if key.lower() == "retry-after":
            try:
                return max(0.0, float(value))
            except (TypeError, ValueError):
                return None
    return None


def error_for_status(
    status_code: int,
    message: str,
    *,
    provider: str,
    model_id: str,
    headers: Mapping[str, str] | None = None,
) -> ProviderError:
    """Map an HTTP status code onto the domain error hierarchy.

    The retryable/non-retryable split lives here so every adapter agrees on it:
    429 and 5xx are transient, other 4xx are the caller's fault and will fail
    identically on a retry.

    Args:
        status_code: HTTP status returned by the vendor.
        message: Vendor error message, already stripped of any credentials.
        provider: Vendor name, for log context.
        model_id: Registry id of the model called.
        headers: Response headers, searched for ``Retry-After``.

    Returns:
        The matching :class:`~autopilot.domain.errors.ProviderError` subclass,
        not raised.
    """
    retry_after_s = parse_retry_after(headers)
    if status_code == 408:
        return ProviderTimeout(message, provider=provider, model_id=model_id)
    if status_code == 429:
        return ProviderRateLimited(
            message, provider=provider, model_id=model_id, retry_after_s=retry_after_s
        )
    if status_code >= 500:
        return ProviderServerError(
            message, provider=provider, model_id=model_id, retry_after_s=retry_after_s
        )
    if status_code == 409:
        # Conflict is transient for vendors that use it for concurrent-request limits.
        return ProviderServerError(message, provider=provider, model_id=model_id)
    return ProviderBadRequest(message, provider=provider, model_id=model_id)


def map_finish_reason(raw: str | None, mapping: Mapping[str, FinishReason]) -> FinishReason:
    """Normalize a vendor stop reason onto :class:`FinishReason`.

    Args:
        raw: The vendor's value, which may be absent.
        mapping: Vendor-specific translation table.

    Returns:
        The normalized reason, defaulting to :attr:`FinishReason.OTHER`.
    """
    if raw is None:
        return FinishReason.OTHER
    return mapping.get(raw, FinishReason.OTHER)
