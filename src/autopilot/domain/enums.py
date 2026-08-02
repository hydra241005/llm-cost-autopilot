"""Enumerations shared across the domain.

These are the closed vocabularies of the system: every routing decision, provider
call, and persisted row references one of these values rather than a bare string.
"""

from __future__ import annotations

from enum import IntEnum, StrEnum


class Tier(IntEnum):
    """Complexity tier of a request, and by extension the class of model it needs.

    Ordered so arithmetic works directly: a low-confidence prediction is bumped
    with ``Tier(min(tier + 1, Tier.COMPLEX))``.
    """

    SIMPLE = 1
    MODERATE = 2
    COMPLEX = 3

    @classmethod
    def bump(cls, tier: Tier, steps: int = 1) -> Tier:
        """Return ``tier`` raised by ``steps``, capped at :attr:`COMPLEX`.

        Args:
            tier: The tier to raise.
            steps: How many tiers to climb. Negative values lower the tier.

        Returns:
            The adjusted tier, clamped to the valid range.
        """
        return cls(min(max(int(tier) + steps, int(cls.SIMPLE)), int(cls.COMPLEX)))


class Provider(StrEnum):
    """An upstream LLM vendor with a concrete adapter implementation."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    OLLAMA = "ollama"


class RequestStatus(StrEnum):
    """Terminal state of a completion request as persisted."""

    SUCCESS = "success"
    FAILED = "failed"
    RATE_LIMITED = "rate_limited"
    TIMEOUT = "timeout"


class FinishReason(StrEnum):
    """Normalized reason a provider stopped generating.

    Vendor-specific values (OpenAI ``length``, Anthropic ``max_tokens``,
    Ollama's ``done_reason``) are mapped onto this common vocabulary by the
    adapters so downstream policy code never branches on a vendor.
    """

    STOP = "stop"
    LENGTH = "length"
    CONTENT_FILTER = "content_filter"
    TOOL_USE = "tool_use"
    OTHER = "other"


class VerdictType(StrEnum):
    """Outcome of an asynchronous quality verification."""

    PASS = "pass"
    FAIL = "fail"
    INCONCLUSIVE = "inconclusive"


class EscalationMode(StrEnum):
    """Which of the two escalation paths produced an event.

    ``SYNCHRONOUS`` guardrails run before the response is delivered and may
    re-route the request once. ``ASYNCHRONOUS`` learning happens after delivery
    and only ever produces training data.
    """

    SYNCHRONOUS = "synchronous"
    ASYNCHRONOUS = "asynchronous"


class TaskType(StrEnum):
    """Client-declared task category, used as a classifier feature and judge rubric key."""

    GENERAL = "general"
    SUMMARIZATION = "summarization"
    EXTRACTION = "extraction"
    TRANSLATION = "translation"
    CLASSIFICATION = "classification"
    CODE = "code"
    REASONING = "reasoning"
    CREATIVE = "creative"


class Role(StrEnum):
    """Chat message role in a completion request."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class BreakerState(StrEnum):
    """Circuit-breaker state for a single model id."""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class ClassifierStatus(StrEnum):
    """Lifecycle state of a versioned classifier artifact."""

    DRAFT = "draft"
    VALIDATED = "validated"
    CANDIDATE = "candidate"
    PRODUCTION = "production"
    ARCHIVED = "archived"
    ACTIVE = "production"
    RETIRED = "archived"


class TrainingJobStatus(StrEnum):
    """Lifecycle state of an asynchronous training run."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class ExampleSource(StrEnum):
    """Where a labelled training example came from.

    Recorded per row because provenance changes how a label should be trusted:
    a hand-written seed is deliberate, whereas a verification failure is
    evidence the router got something wrong in production.
    """

    SEED = "seed"
    VERIFICATION_FAILURE = "verification_failure"
    HUMAN_REVIEW = "human_review"
