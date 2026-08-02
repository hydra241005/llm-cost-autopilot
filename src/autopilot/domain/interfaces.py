"""Domain ports.

Abstract base classes and protocols that infrastructure implements and
application code depends on. Nothing here imports a framework, a driver, or a
vendor SDK — that is the whole point of the file.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from datetime import datetime
from typing import Any, Protocol, runtime_checkable

from autopilot.domain.entities import (
    ClassifierMetadata,
    CompletionInput,
    FeatureVector,
    JudgeVerdict,
    ModelConfig,
    Prediction,
    ProviderResponse,
    RequestRecord,
    VerificationResult,
)
from autopilot.domain.enums import Provider, TaskType, Tier


class LLMProvider(ABC):
    """A callable upstream LLM vendor.

    Implementations translate :class:`CompletionInput` into a vendor request and
    the vendor's reply into a :class:`ProviderResponse`, mapping every vendor
    failure onto the domain error hierarchy.
    """

    #: Vendor this adapter speaks to.
    name: Provider

    @abstractmethod
    async def complete(
        self,
        req: CompletionInput,
        cfg: ModelConfig,
        timeout_s: float,
    ) -> ProviderResponse:
        """Execute one completion call.

        Args:
            req: The provider-agnostic request.
            cfg: Registry entry for the model to call.
            timeout_s: Total wall-clock budget for this attempt.

        Returns:
            The normalized provider response.

        Raises:
            ProviderTimeout: The call exceeded ``timeout_s``.
            ProviderRateLimited: The vendor returned a rate-limit response.
            ProviderServerError: The vendor returned 5xx or the connection failed.
            ProviderBadRequest: The vendor rejected the request itself.
        """

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """Estimate the token count of ``text`` for this vendor's tokenizer."""

    @abstractmethod
    async def aclose(self) -> None:
        """Release network resources held by the adapter."""


class ModelRegistry(ABC):
    """Read model of the callable model catalogue."""

    @abstractmethod
    def get(self, model_id: str) -> ModelConfig:
        """Return the config for ``model_id``.

        Raises:
            ModelNotFoundError: No such model is registered.
        """

    @abstractmethod
    def by_tier(self, tier: Tier) -> Sequence[ModelConfig]:
        """Return active models for ``tier``, cheapest first."""

    @abstractmethod
    def cheapest(self, tier: Tier) -> ModelConfig:
        """Return the cheapest active model for ``tier``.

        Raises:
            NoCapableModelError: The tier has no active models.
        """

    @abstractmethod
    def all(self) -> Sequence[ModelConfig]:
        """Return every registered model, active or not."""


class ComplexityClassifier(ABC):
    """Predicts the complexity tier of a request from its feature vector."""

    #: Artifact version, recorded on every routing decision for reproducibility.
    version: str

    @abstractmethod
    def predict(self, fv: FeatureVector) -> Prediction:
        """Return the predicted tier, confidence, per-class probabilities, and evidence."""

    @property
    def metadata(self) -> ClassifierMetadata | None:
        """Provenance of the loaded artifact, when one was trained.

        Rule-based classifiers have no training run to describe and return ``None``.
        """
        return None


@runtime_checkable
class FeatureExtractor(Protocol):
    """Turns a request into a numeric feature vector in well under a millisecond."""

    #: Schema version, compared against a classifier's before a prediction is trusted.
    schema_version: str

    def extract(self, req: CompletionInput) -> FeatureVector:
        """Return the feature vector for ``req``."""
        ...

    def feature_names(self) -> Sequence[str]:
        """Return every feature name this extractor can emit, in stable order."""
        ...


class ClassifierStore(ABC):
    """Versioned persistence for trained classifier artifacts.

    Kept a port rather than a filesystem call so the artifact can later move to
    object storage without the router noticing.
    """

    @abstractmethod
    def save(self, version: str, artifact: Any, metadata: ClassifierMetadata) -> None:
        """Persist ``artifact`` and its provenance under ``version``."""

    @abstractmethod
    def load(self, version: str) -> tuple[Any, ClassifierMetadata]:
        """Return the artifact and metadata stored under ``version``.

        Raises:
            ClassifierError: No such version exists, or it cannot be deserialized.
        """

    @abstractmethod
    def load_active(self) -> tuple[Any, ClassifierMetadata] | None:
        """Return the currently active artifact, or ``None`` when none is promoted."""

    @abstractmethod
    def list_versions(self) -> Sequence[ClassifierMetadata]:
        """Return metadata for every stored version, newest first."""

    @abstractmethod
    def promote(self, version: str) -> None:
        """Mark ``version`` active and retire whichever version held that status."""


class ResponseCache(ABC):
    """Exact-match cache of completed responses."""

    @abstractmethod
    async def get(self, key: str) -> ProviderResponse | None:
        """Return the cached response for ``key``, or ``None`` on a miss."""

    @abstractmethod
    async def set(self, key: str, value: ProviderResponse, ttl_s: int) -> None:
        """Store ``value`` under ``key`` for ``ttl_s`` seconds."""

    @abstractmethod
    def make_key(self, req: CompletionInput, model_map_version: str) -> str:
        """Derive a stable cache key from the request and the active model map."""


class RequestRepository(ABC):
    """Persistence port for completion request records."""

    @abstractmethod
    async def save(self, record: RequestRecord) -> None:
        """Insert ``record``."""

    @abstractmethod
    async def get(self, request_id: str) -> RequestRecord | None:
        """Return the record for ``request_id``, or ``None``."""

    @abstractmethod
    async def stats_window(self, start: datetime, end: datetime) -> Sequence[RequestRecord]:
        """Return records created within the half-open interval ``[start, end)``."""


class VerificationRepository(ABC):
    """Persistence port for verification results."""

    @abstractmethod
    async def save(self, result: VerificationResult) -> None:
        """Insert ``result``."""

    @abstractmethod
    async def failure_rate(self, model_id: str, since: datetime) -> float:
        """Return the fraction of verifications for ``model_id`` that failed since ``since``."""


class ConfigRepository(ABC):
    """Persistence port for versioned routing configuration."""

    @abstractmethod
    async def active_routing_config(self) -> dict[str, Any]:
        """Return the currently active routing config document."""

    @abstractmethod
    async def save_version(self, config: dict[str, Any], actor: str) -> int:
        """Persist a new config version and return its version number."""


class JobQueue(ABC):
    """Port for deferring work off the request path."""

    @abstractmethod
    async def enqueue_verification(self, payload: dict[str, Any]) -> None:
        """Schedule an asynchronous quality verification."""

    @abstractmethod
    async def enqueue_retrain(self) -> None:
        """Schedule a classifier retraining run."""


class Judge(ABC):
    """LLM-as-judge port used by the verification service."""

    @abstractmethod
    async def evaluate(
        self,
        task_type: TaskType,
        prompt: str,
        candidate: str,
        reference: str,
    ) -> JudgeVerdict:
        """Compare ``candidate`` against ``reference`` and return a structured verdict."""


class RateLimiter(ABC):
    """Per-key request admission control."""

    @abstractmethod
    async def allow(self, key: str, capacity: int) -> tuple[bool, float, int]:
        """Attempt to consume one token from ``key``'s bucket.

        Args:
            key: Bucket identity, normally an API key id.
            capacity: Bucket size in requests per minute.

        Returns:
            A tuple of ``(allowed, retry_after_s, remaining)``.
        """


class Clock(Protocol):
    """Time source, injected so tests can freeze it."""

    def now(self) -> datetime:
        """Return the current UTC time."""
        ...

    def monotonic(self) -> float:
        """Return a monotonic timer value in seconds."""
        ...


class IdGenerator(Protocol):
    """Identifier source, injected so tests can make ids deterministic."""

    def new_id(self) -> str:
        """Return a fresh, time-ordered unique identifier."""
        ...
