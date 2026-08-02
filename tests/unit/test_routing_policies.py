"""Routing policy tests."""

from __future__ import annotations

import pytest

from autopilot.domain.entities import CompletionInput, Message, Prediction
from autopilot.domain.enums import BreakerState, Provider, Role, Tier
from autopilot.domain.policies.fallback import RoutingConfig, TierRoute
from autopilot.domain.policies.routing import (
    ActiveModelPolicy,
    ConfidenceBumpPolicy,
    ContextWindowPolicy,
    LongPromptPolicy,
    ProviderHealthPolicy,
    RoutingContext,
)
from tests.conftest import make_model


class FakeHealth:
    def __init__(self, open_providers: set[Provider] | None = None) -> None:
        self._open = open_providers or set()

    def is_available(self, provider: Provider) -> bool:
        return provider not in self._open

    def state(self, provider: Provider) -> BreakerState:
        return BreakerState.OPEN if provider in self._open else BreakerState.CLOSED


def context(
    *,
    confidence: float = 0.9,
    tier: Tier = Tier.SIMPLE,
    prompt_tokens: int = 100,
    max_tokens: int = 256,
    threshold: float = 0.6,
    health: FakeHealth | None = None,
) -> RoutingContext:
    return RoutingContext(
        request=CompletionInput(
            messages=(Message(role=Role.USER, content="hi"),), max_tokens=max_tokens
        ),
        prediction=Prediction(tier=tier, confidence=confidence, classifier_version="test"),
        prompt_tokens=prompt_tokens,
        confidence_threshold=threshold,
        health=health or FakeHealth(),
    )


def test_high_confidence_leaves_the_tier_alone():
    adjustment = ConfidenceBumpPolicy().apply(Tier.SIMPLE, context(confidence=0.9))

    assert adjustment.changed is False
    assert adjustment.tier is Tier.SIMPLE


def test_low_confidence_raises_the_tier_and_says_why():
    adjustment = ConfidenceBumpPolicy().apply(Tier.SIMPLE, context(confidence=0.4))

    assert adjustment.changed is True
    assert adjustment.tier is Tier.MODERATE
    assert "0.40" in adjustment.reason


def test_the_bump_cannot_exceed_the_top_tier():
    adjustment = ConfidenceBumpPolicy().apply(Tier.COMPLEX, context(confidence=0.1))

    assert adjustment.changed is False
    assert adjustment.tier is Tier.COMPLEX
    assert "ceiling" in adjustment.reason


def test_the_threshold_boundary_does_not_bump():
    # Exactly at the threshold is confident enough; the doc defines the bump as
    # strictly below tau.
    adjustment = ConfidenceBumpPolicy().apply(Tier.SIMPLE, context(confidence=0.6, threshold=0.6))

    assert adjustment.changed is False


def test_a_long_prompt_raises_the_tier():
    policy = LongPromptPolicy(token_threshold=1_000)

    assert policy.apply(Tier.SIMPLE, context(prompt_tokens=999)).changed is False
    bumped = policy.apply(Tier.SIMPLE, context(prompt_tokens=1_000))
    assert bumped.changed is True
    assert bumped.tier is Tier.MODERATE


def test_a_long_prompt_at_the_top_tier_is_a_no_op():
    policy = LongPromptPolicy(token_threshold=10)

    assert policy.apply(Tier.COMPLEX, context(prompt_tokens=5_000)).changed is False


def test_inactive_models_are_rejected():
    verdict = ActiveModelPolicy().evaluate(make_model(active=False), context())

    assert verdict.eligible is False
    assert "deactivated" in verdict.reason


def test_active_models_pass():
    assert ActiveModelPolicy().evaluate(make_model(), context()).eligible is True


def test_a_prompt_larger_than_the_context_window_is_rejected():
    model = make_model(max_context_tokens=1_000)

    verdict = ContextWindowPolicy().evaluate(model, context(prompt_tokens=900, max_tokens=500))

    assert verdict.eligible is False
    assert "1400" in verdict.reason


def test_an_output_cap_smaller_than_requested_is_rejected():
    model = make_model(max_context_tokens=100_000, max_output_tokens=512)

    verdict = ContextWindowPolicy().evaluate(model, context(max_tokens=4_096))

    assert verdict.eligible is False
    assert "caps at 512" in verdict.reason


def test_a_request_that_fits_exactly_is_accepted():
    model = make_model(max_context_tokens=1_000, max_output_tokens=500)

    verdict = ContextWindowPolicy().evaluate(model, context(prompt_tokens=500, max_tokens=500))

    assert verdict.eligible is True


def test_an_open_circuit_rejects_every_model_from_that_provider():
    health = FakeHealth({Provider.OPENAI})

    rejected = ProviderHealthPolicy().evaluate(
        make_model(provider=Provider.OPENAI), context(health=health)
    )
    accepted = ProviderHealthPolicy().evaluate(
        make_model("anthropic:x", provider=Provider.ANTHROPIC), context(health=health)
    )

    assert rejected.eligible is False
    assert "circuit is open" in rejected.reason
    assert accepted.eligible is True


def routing_config(**kwargs) -> RoutingConfig:
    defaults = dict(
        baseline_model_id="openai:gpt-4o",
        tiers={
            Tier.SIMPLE: TierRoute(primary="ollama:llama3.1-8b", fallbacks=("openai:gpt-4o-mini",)),
            Tier.MODERATE: TierRoute(primary="openai:gpt-4o-mini"),
            Tier.COMPLEX: TierRoute(primary="openai:gpt-4o"),
        },
    )
    return RoutingConfig(**{**defaults, **kwargs})


def test_a_chain_puts_the_primary_first():
    route = TierRoute(primary="a", fallbacks=("b", "c"))

    assert route.chain == ("a", "b", "c")


def test_a_chain_drops_a_fallback_that_repeats_the_primary():
    route = TierRoute(primary="a", fallbacks=("a", "b"))

    assert route.chain == ("a", "b")


def test_the_escalation_chain_continues_into_higher_tiers():
    # Exhausting a tier is no reason to 503 while a pricier model is healthy.
    chain = routing_config().escalation_chain(Tier.SIMPLE)

    assert chain == ("ollama:llama3.1-8b", "openai:gpt-4o-mini", "openai:gpt-4o")


def test_the_escalation_chain_never_descends():
    assert routing_config().escalation_chain(Tier.COMPLEX) == ("openai:gpt-4o",)


def test_a_config_missing_a_tier_is_refused():
    with pytest.raises(ValueError, match="missing tier"):
        RoutingConfig(
            baseline_model_id="openai:gpt-4o",
            tiers={Tier.SIMPLE: TierRoute(primary="a")},
        )


def test_a_non_mapping_document_is_a_configuration_error():
    from autopilot.domain.errors import ConfigurationError

    with pytest.raises(ConfigurationError, match="must be a mapping"):
        RoutingConfig.from_dict(["not", "a", "mapping"])
