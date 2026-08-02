"""Domain exception hierarchy.

Every adapter translates vendor-specific failures into these types, so retry,
circuit-breaker, and fallback logic stays provider-agnostic. The split that
matters most is :attr:`ProviderError.retryable`: the resilience layer never
inspects status codes itself.
"""

from __future__ import annotations


class AutopilotError(Exception):
    """Base class for every error raised by the autopilot domain."""

    #: Stable machine-readable code surfaced in the API error envelope.
    code: str = "autopilot_error"

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class ConfigurationError(AutopilotError):
    """Raised when static configuration is missing, malformed, or self-inconsistent."""

    code = "configuration_error"


class ModelNotFoundError(AutopilotError):
    """Raised when a model id is referenced but absent from the registry."""

    code = "model_not_found"

    def __init__(self, model_id: str) -> None:
        super().__init__(f"Model {model_id!r} is not present in the registry.")
        self.model_id = model_id


class NoCapableModelError(AutopilotError):
    """Raised when no registered model can serve a request.

    Happens when every candidate for the effective tier is inactive, has an open
    circuit breaker, or has a context window smaller than the prompt.
    """

    code = "provider_unavailable"


class ProviderError(AutopilotError):
    """Base class for failures originating from an upstream provider.

    Attributes:
        provider: Which vendor produced the failure.
        model_id: Registry id of the model that was called.
        retryable: Whether the resilience layer may retry this call. Set by the
            subclass, never by status-code inspection at the call site.
        retry_after_s: Provider-advertised cooldown, when one was sent.
    """

    code = "provider_error"
    retryable: bool = False

    def __init__(
        self,
        message: str,
        *,
        provider: str | None = None,
        model_id: str | None = None,
        retry_after_s: float | None = None,
    ) -> None:
        super().__init__(message)
        self.provider = provider
        self.model_id = model_id
        self.retry_after_s = retry_after_s


class ProviderTimeout(ProviderError):
    """The provider did not respond within the tier's timeout budget."""

    code = "provider_timeout"
    retryable = True


class ProviderRateLimited(ProviderError):
    """The provider rejected the call with a rate-limit response (HTTP 429)."""

    code = "provider_rate_limited"
    retryable = True


class ProviderServerError(ProviderError):
    """The provider returned a 5xx, or the connection failed in transit."""

    code = "provider_server_error"
    retryable = True


class ProviderBadRequest(ProviderError):
    """The provider rejected the request itself (4xx other than 429).

    Never retried: a malformed or unauthorized request will fail identically on
    a second attempt.
    """

    code = "provider_bad_request"
    retryable = False


class CircuitOpenError(ProviderError):
    """The circuit breaker for this model is open, so the call was not attempted."""

    code = "circuit_open"
    retryable = False


class ClassifierError(AutopilotError):
    """Raised when feature extraction or prediction fails irrecoverably."""

    code = "classifier_error"


class JudgeError(AutopilotError):
    """Raised when the verification judge cannot produce a usable verdict."""

    code = "judge_error"


class RateLimitExceededError(AutopilotError):
    """Raised when an API key exhausts its token bucket."""

    code = "rate_limit_exceeded"

    def __init__(self, message: str, *, retry_after_s: float) -> None:
        super().__init__(message)
        self.retry_after_s = retry_after_s
