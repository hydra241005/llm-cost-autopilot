"""Domain entities and value objects.

Immutable pydantic models with no I/O. These are the vocabulary every layer
speaks: adapters return them, policies consume them, repositories persist them.
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from hashlib import sha256
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from autopilot.domain.enums import (
    BreakerState,
    ClassifierStatus,
    ExampleSource,
    FinishReason,
    Provider,
    RequestStatus,
    Role,
    TaskType,
    Tier,
    TrainingJobStatus,
    VerdictType,
)


class _Frozen(BaseModel):
    """Base for immutable value objects with strict validation."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class Message(_Frozen):
    """A single chat message in a completion request."""

    role: Role
    content: str = Field(min_length=1)


class CompletionInput(_Frozen):
    """A validated, provider-agnostic completion request.

    This is what the orchestrator hands to any :class:`~autopilot.domain.interfaces.LLMProvider`.
    It deliberately excludes vendor-specific knobs; adapters translate it.
    """

    messages: tuple[Message, ...] = Field(min_length=1)
    max_tokens: int = Field(default=1024, ge=1, le=64_000)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    task_type: TaskType = TaskType.GENERAL
    stop_sequences: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _require_non_system_message(self) -> CompletionInput:
        """Reject inputs consisting solely of system messages."""
        if all(m.role is Role.SYSTEM for m in self.messages):
            raise ValueError("messages must contain at least one user or assistant turn")
        return self

    @property
    def system_prompt(self) -> str | None:
        """Concatenated system messages, or ``None`` when no system turn is present.

        Anthropic carries the system prompt out of band, so adapters need it split out.
        """
        parts = [m.content for m in self.messages if m.role is Role.SYSTEM]
        return "\n\n".join(parts) if parts else None

    @property
    def conversation(self) -> tuple[Message, ...]:
        """Messages excluding system turns, in original order."""
        return tuple(m for m in self.messages if m.role is not Role.SYSTEM)


class ModelConfig(_Frozen):
    """Registry entry for one callable model.

    Pricing is data, not code: updating a price is a config change plus an audit
    row, never a code deploy.
    """

    id: str = Field(min_length=1, description="Registry id, e.g. 'anthropic:claude-haiku-4-5'.")
    provider: Provider
    vendor_model_id: str = Field(min_length=1, description="Id sent on the wire to the vendor.")
    tier: Tier
    input_cost_per_mtok: Decimal = Field(ge=0)
    output_cost_per_mtok: Decimal = Field(ge=0)
    max_context_tokens: int = Field(gt=0)
    max_output_tokens: int = Field(gt=0)
    expected_latency_ms: int = Field(gt=0)
    active: bool = True
    supports_temperature: bool = Field(
        default=True,
        description="False for models that reject non-default sampling parameters.",
    )
    notes: str = ""

    @property
    def is_free(self) -> bool:
        """Whether this model costs nothing to call (a local model)."""
        return self.input_cost_per_mtok == 0 and self.output_cost_per_mtok == 0


class Usage(_Frozen):
    """Token accounting for a single provider call."""

    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    cached_input_tokens: int = Field(default=0, ge=0)

    @property
    def total_tokens(self) -> int:
        """Sum of billed input and output tokens."""
        return self.input_tokens + self.output_tokens


class ProviderResponse(_Frozen):
    """Normalized result of a successful provider call."""

    model_id: str
    provider: Provider
    text: str
    usage: Usage
    finish_reason: FinishReason
    latency_ms: int = Field(ge=0)
    vendor_request_id: str | None = None
    raw_finish_reason: str | None = Field(
        default=None, description="Unmapped vendor value, kept for debugging."
    )


class TrainingExample(_Frozen):
    """One labelled prompt used to train or evaluate the complexity classifier.

    The prompt text is carried so features can be re-extracted whenever the
    schema changes; ``prompt_hash`` is what deduplication and the dataset
    fingerprint are computed from.
    """

    prompt: str = Field(min_length=1)
    tier: Tier
    task_type: TaskType = TaskType.GENERAL
    source: ExampleSource = ExampleSource.SEED
    is_holdout: bool = False
    notes: str = ""

    @property
    def prompt_hash(self) -> str:
        """SHA-256 of the normalized prompt, used as the dedupe key."""
        return sha256(" ".join(self.prompt.split()).lower().encode("utf-8")).hexdigest()

    @property
    def dedupe_key(self) -> tuple[str, Tier]:
        """The ``(prompt_hash, tier)`` pair a dataset is deduplicated on.

        The tier is part of the key deliberately: the same prompt relabelled at
        a different tier is a genuine correction, not a duplicate.
        """
        return (self.prompt_hash, self.tier)


class FeatureVector(_Frozen):
    """Extracted, model-ready features for one request.

    Stored on every request row so retraining never needs the raw prompt.
    """

    values: dict[str, float]
    schema_version: str = Field(
        default="unknown",
        description="Feature-schema version, so a model is never fed a vector it was not trained on.",
    )

    def as_dict(self) -> dict[str, float]:
        """Return a mutable copy suitable for a scikit-learn ``DictVectorizer``."""
        return dict(self.values)

    def top(self, n: int = 5) -> tuple[tuple[str, float], ...]:
        """Return the ``n`` largest non-zero features, descending by magnitude.

        A cheap, model-independent explanation of what the extractor actually saw,
        useful when the classifier itself offers no attribution.
        """
        ranked = sorted(
            ((k, v) for k, v in self.values.items() if v != 0.0),
            key=lambda kv: abs(kv[1]),
            reverse=True,
        )
        return tuple(ranked[:n])


class FeatureContribution(_Frozen):
    """One feature's signed influence on a prediction.

    ``weight`` is model-specific (a linear coefficient, a rule's score) and is
    only ever compared against other contributions from the same prediction.
    """

    name: str = Field(min_length=1)
    value: float
    weight: float
    contribution: float = Field(description="``value × weight``; positive pushes toward the tier.")

    @property
    def direction(self) -> str:
        """Whether this feature argued for a higher or lower tier."""
        if self.contribution > 0:
            return "raises"
        return "lowers" if self.contribution < 0 else "neutral"


class Prediction(_Frozen):
    """Classifier output: a tier, a confidence, and the evidence behind them."""

    tier: Tier
    confidence: float = Field(ge=0.0, le=1.0)
    probabilities: dict[Tier, float] = Field(default_factory=dict)
    classifier_version: str
    top_features: tuple[FeatureContribution, ...] = Field(
        default=(),
        description="Highest-magnitude drivers of this prediction, most influential first.",
    )
    rationale: str = Field(
        default="",
        description="One-line human-readable summary of why this tier was predicted.",
    )


class ClassifierMetadata(_Frozen):
    """Provenance of one trained classifier artifact.

    Persisted alongside the artifact so any historical routing decision can be
    traced back to the exact model, dataset, and code that produced it.
    """

    version: str = Field(min_length=1)
    status: ClassifierStatus = ClassifierStatus.CANDIDATE
    algorithm: str = Field(min_length=1)
    trained_at: datetime
    dataset_hash: str = Field(min_length=1, description="SHA-256 over the ordered training rows.")
    dataset_size: int = Field(ge=0)
    holdout_size: int = Field(default=0, ge=0)
    feature_schema: tuple[str, ...] = Field(
        default=(), description="Ordered feature names the artifact expects."
    )
    feature_schema_version: str = "unknown"
    metrics: dict[str, float] = Field(default_factory=dict)
    confusion_matrix: tuple[tuple[int, ...], ...] = ()
    git_commit: str | None = None
    notes: str = ""

    @property
    def cost_weighted_error(self) -> float | None:
        """The promotion gate's headline metric, when it was recorded."""
        return self.metrics.get("cost_weighted_error")


class TrainingJob(_Frozen):
    """Operational state for an asynchronous classifier training run."""

    job_id: str = Field(min_length=1)
    version: str = Field(min_length=1)
    status: TrainingJobStatus = TrainingJobStatus.PENDING
    created_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
    error: str | None = None
    artifact_version: str | None = None


class CandidateEvaluation(_Frozen):
    """Why one candidate model was accepted or skipped.

    The router emits one of these per candidate considered, in order, which is
    what makes a routing decision auditable rather than merely asserted.
    """

    model_id: str
    provider: Provider
    tier: Tier
    eligible: bool
    reason: str = Field(description="Why it was skipped, or why it was selected.")
    estimated_cost_usd: Decimal = Field(default=Decimal("0"), ge=0)
    expected_latency_ms: int = Field(default=0, ge=0)
    provider_state: BreakerState = BreakerState.CLOSED
    selected: bool = False


class RoutingExplanation(_Frozen):
    """The full, human-readable account of one routing decision."""

    predicted_tier: Tier
    effective_tier: Tier
    confidence: float = Field(ge=0.0, le=1.0)
    estimated_cost_usd: Decimal = Field(ge=0)
    baseline_cost_usd: Decimal = Field(ge=0)
    estimated_latency_ms: int = Field(ge=0)
    prompt_tokens: int = Field(ge=0)
    provider_state: BreakerState = BreakerState.CLOSED
    provider_healthy: bool = True
    tier_adjustments: tuple[str, ...] = Field(
        default=(), description="Policy names and reasons that moved the effective tier."
    )
    candidates: tuple[CandidateEvaluation, ...] = ()
    top_features: tuple[FeatureContribution, ...] = ()
    summary: str = Field(default="", description="One-sentence justification of the chosen model.")

    @property
    def estimated_savings_usd(self) -> Decimal:
        """Projected saving versus sending this request to the baseline model."""
        return self.baseline_cost_usd - self.estimated_cost_usd

    @property
    def skipped(self) -> tuple[CandidateEvaluation, ...]:
        """Candidates that were considered and rejected."""
        return tuple(c for c in self.candidates if not c.eligible)


class RoutingDecision(_Frozen):
    """A full, reproducible account of why one model was chosen."""

    request_id: str | None = None
    predicted_tier: Tier
    effective_tier: Tier
    confidence: float = Field(ge=0.0, le=1.0)
    chosen_model_id: str
    candidate_model_ids: tuple[str, ...]
    fallback_depth: int = Field(default=0, ge=0)
    confidence_bumped: bool = False
    classifier_version: str
    routing_config_version: int = Field(default=1, ge=1)
    decision_latency_ms: float = Field(default=0.0, ge=0.0)
    explanation: RoutingExplanation | None = Field(
        default=None,
        description="Populated by the routing engine; absent only on replayed historical rows.",
    )


class CostBreakdown(_Frozen):
    """Actual cost, counterfactual baseline cost, and the delta between them."""

    actual_usd: Decimal = Field(ge=0)
    baseline_usd: Decimal = Field(ge=0)
    baseline_model_id: str

    @property
    def saved_usd(self) -> Decimal:
        """Savings versus the baseline. Negative when routing cost more."""
        return self.baseline_usd - self.actual_usd

    @property
    def savings_ratio(self) -> float:
        """Fraction of baseline spend avoided, or ``0.0`` when the baseline is free."""
        if self.baseline_usd == 0:
            return 0.0
        return float(self.saved_usd / self.baseline_usd)


class RequestRecord(_Frozen):
    """The persisted record of one completion request end to end."""

    request_id: str
    api_key_id: str | None
    created_at: datetime
    status: RequestStatus
    task_type: TaskType
    prompt_hash: str
    prompt_text: str | None = Field(
        default=None,
        description="Populated only when the API key opts into prompt storage.",
    )
    features: FeatureVector | None = None
    decision: RoutingDecision | None = None
    usage: Usage | None = None
    cost: CostBreakdown | None = None
    latency_ms: int = Field(default=0, ge=0)
    cache_hit: bool = False
    error_code: str | None = None


class JudgeVerdict(_Frozen):
    """Structured output of the LLM-as-judge for one verification."""

    verdict: VerdictType
    score: float = Field(ge=0.0, le=5.0)
    agreement: float = Field(
        ge=0.0, le=1.0, description="Agreement across position-swapped pairwise passes."
    )
    rationale: str
    axis_scores: dict[str, float] = Field(default_factory=dict)


class VerificationResult(_Frozen):
    """Outcome of verifying one delivered response against a reference model."""

    request_id: str
    verified_at: datetime
    judge_model_id: str
    reference_model_id: str
    verdict: JudgeVerdict
    verification_cost_usd: Decimal = Field(ge=0)
    sampling_reason: str


class EscalationEvent(_Frozen):
    """A record that a request was escalated, and why."""

    request_id: str
    occurred_at: datetime
    mode: str
    from_tier: Tier
    to_tier: Tier
    trigger: str
    detail: dict[str, Any] = Field(default_factory=dict)


class BreakerSnapshot(_Frozen):
    """Observable circuit-breaker state for one model, surfaced via the API."""

    model_id: str
    state: BreakerState
    failure_count: int = Field(ge=0)
    opened_at: datetime | None = None
