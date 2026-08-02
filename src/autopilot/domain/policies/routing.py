"""Composable routing policies.

The routing engine is deliberately not a tier-to-model lookup. It is a small
pipeline of independent policies, each of which either adjusts the effective
tier or rules a candidate model out, and each of which records *why*. Adding a
rule — a budget cap, a data-residency constraint, a per-key model allowlist —
means writing one class and appending it to a list, not editing the engine.

Two policy kinds:

* :class:`TierPolicy` runs once, before candidates are enumerated, and may raise
  or lower the effective tier.
* :class:`CandidatePolicy` runs per candidate and answers "may this model serve
  this request?" with a reason either way.

Both are pure: they take a context and return a verdict, perform no I/O, and are
trivially unit-testable in isolation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Protocol

from autopilot.domain.entities import CompletionInput, ModelConfig, Prediction
from autopilot.domain.enums import BreakerState, Provider, Tier


class HealthView(Protocol):
    """Read-only view of upstream provider health, as seen by the router."""

    def is_available(self, provider: Provider) -> bool:
        """Return whether ``provider`` is currently accepting calls."""
        ...

    def state(self, provider: Provider) -> BreakerState:
        """Return ``provider``'s current circuit state."""
        ...


class _AlwaysHealthy:
    """Fallback health view used when no manager is injected."""

    def is_available(self, provider: Provider) -> bool:
        """Report every provider as available."""
        return True

    def state(self, provider: Provider) -> BreakerState:
        """Report every circuit as closed."""
        return BreakerState.CLOSED


#: Health view assumed when the engine is constructed without one.
ALWAYS_HEALTHY: HealthView = _AlwaysHealthy()


@dataclass(frozen=True, slots=True)
class RoutingContext:
    """Everything a policy is allowed to see when making its judgement.

    Attributes:
        request: The validated completion request.
        prediction: Classifier output for this request.
        prompt_tokens: Estimated prompt size, used by the context guard.
        confidence_threshold: Below this confidence the tier is bumped.
        health: Live provider health.
    """

    request: CompletionInput
    prediction: Prediction
    prompt_tokens: int
    confidence_threshold: float
    health: HealthView = ALWAYS_HEALTHY


@dataclass(frozen=True, slots=True)
class TierAdjustment:
    """A tier policy's verdict.

    Attributes:
        tier: The tier after this policy ran.
        changed: Whether the policy moved the tier.
        reason: Why, in one line. Empty when nothing changed.
    """

    tier: Tier
    changed: bool
    reason: str = ""


@dataclass(frozen=True, slots=True)
class CandidateVerdict:
    """A candidate policy's verdict on one model.

    Attributes:
        eligible: Whether the model may serve this request.
        reason: Why it was rejected. Empty when eligible.
    """

    eligible: bool
    reason: str = ""


#: Reusable verdict for the common case; avoids allocating on every candidate.
ELIGIBLE = CandidateVerdict(eligible=True)


class TierPolicy(ABC):
    """Adjusts the effective tier before candidates are enumerated."""

    #: Stable identifier recorded in the routing explanation.
    name: str

    @abstractmethod
    def apply(self, tier: Tier, ctx: RoutingContext) -> TierAdjustment:
        """Return the tier after this policy, and why it changed."""


class CandidatePolicy(ABC):
    """Decides whether one candidate model may serve the request."""

    #: Stable identifier recorded in the routing explanation.
    name: str

    @abstractmethod
    def evaluate(self, model: ModelConfig, ctx: RoutingContext) -> CandidateVerdict:
        """Return whether ``model`` is eligible, and why not when it is not."""


class ConfidenceBumpPolicy(TierPolicy):
    """Raises the tier by one when the classifier is unsure.

    Mis-routing up wastes cents; mis-routing down damages the answer a user
    actually receives. The asymmetry is the entire justification for this rule.
    """

    name = "confidence_bump"

    def __init__(self, steps: int = 1) -> None:
        """Create the policy.

        Args:
            steps: How many tiers to climb on low confidence.
        """
        self._steps = steps

    def apply(self, tier: Tier, ctx: RoutingContext) -> TierAdjustment:
        """Bump ``tier`` when confidence falls below the configured threshold."""
        confidence = ctx.prediction.confidence
        if confidence >= ctx.confidence_threshold:
            return TierAdjustment(tier=tier, changed=False)
        bumped = Tier.bump(tier, self._steps)
        if bumped is tier:
            return TierAdjustment(
                tier=tier,
                changed=False,
                reason=f"confidence {confidence:.2f} is low but tier {int(tier)} is already the ceiling",
            )
        return TierAdjustment(
            tier=bumped,
            changed=True,
            reason=(
                f"confidence {confidence:.2f} < {ctx.confidence_threshold:.2f}, "
                f"so tier {int(tier)} was raised to {int(bumped)}"
            ),
        )


class LongPromptPolicy(TierPolicy):
    """Raises the tier for prompts large enough that a small model will struggle.

    Length is not complexity, but past a threshold it correlates strongly enough
    that serving a 30k-token prompt from the cheapest tier reliably disappoints.
    """

    name = "long_prompt"

    def __init__(self, token_threshold: int = 8_000, steps: int = 1) -> None:
        """Create the policy.

        Args:
            token_threshold: Prompt size at or above which the tier is raised.
            steps: How many tiers to climb.
        """
        self._threshold = token_threshold
        self._steps = steps

    def apply(self, tier: Tier, ctx: RoutingContext) -> TierAdjustment:
        """Bump ``tier`` when the prompt is at or above the threshold."""
        if ctx.prompt_tokens < self._threshold:
            return TierAdjustment(tier=tier, changed=False)
        bumped = Tier.bump(tier, self._steps)
        if bumped is tier:
            return TierAdjustment(tier=tier, changed=False)
        return TierAdjustment(
            tier=bumped,
            changed=True,
            reason=(
                f"prompt of {ctx.prompt_tokens} tokens is at or above the "
                f"{self._threshold}-token threshold, so tier {int(tier)} "
                f"was raised to {int(bumped)}"
            ),
        )


class ActiveModelPolicy(CandidatePolicy):
    """Rejects models that have been deactivated in the registry."""

    name = "active_model"

    def evaluate(self, model: ModelConfig, ctx: RoutingContext) -> CandidateVerdict:
        """Reject ``model`` when it is marked inactive."""
        if model.active:
            return ELIGIBLE
        return CandidateVerdict(eligible=False, reason="model is deactivated in the registry")


class ContextWindowPolicy(CandidatePolicy):
    """Rejects models whose context window cannot hold prompt plus completion.

    Checked before dispatch rather than after a vendor 400: the failure is fully
    predictable from data already in hand, so spending a round trip to discover
    it would be pure latency.
    """

    name = "context_window"

    def evaluate(self, model: ModelConfig, ctx: RoutingContext) -> CandidateVerdict:
        """Reject ``model`` when the request does not fit its limits."""
        required = ctx.prompt_tokens + ctx.request.max_tokens
        if required > model.max_context_tokens:
            return CandidateVerdict(
                eligible=False,
                reason=(
                    f"needs {required} tokens of context but the model holds "
                    f"{model.max_context_tokens}"
                ),
            )
        if ctx.request.max_tokens > model.max_output_tokens:
            return CandidateVerdict(
                eligible=False,
                reason=(
                    f"requested {ctx.request.max_tokens} output tokens but the model "
                    f"caps at {model.max_output_tokens}"
                ),
            )
        return ELIGIBLE


class ProviderHealthPolicy(CandidatePolicy):
    """Rejects models whose provider has an open circuit.

    Guarding at the provider level is deliberate: a vendor outage affects every
    one of its models, so one open circuit should divert all of them.
    """

    name = "provider_health"

    def evaluate(self, model: ModelConfig, ctx: RoutingContext) -> CandidateVerdict:
        """Reject ``model`` when its provider is not currently accepting calls."""
        if ctx.health.is_available(model.provider):
            return ELIGIBLE
        return CandidateVerdict(
            eligible=False,
            reason=f"circuit is open for provider {model.provider.value!r}",
        )


#: Default tier pipeline, in the order the engine applies it.
DEFAULT_TIER_POLICIES: tuple[TierPolicy, ...] = (
    ConfidenceBumpPolicy(),
    LongPromptPolicy(),
)

#: Default candidate filters. Ordered cheapest-to-evaluate first so an obviously
#: ineligible model is rejected without consulting live health state.
DEFAULT_CANDIDATE_POLICIES: tuple[CandidatePolicy, ...] = (
    ActiveModelPolicy(),
    ContextWindowPolicy(),
    ProviderHealthPolicy(),
)
