"""Routing engine tests."""

from __future__ import annotations

import pytest

from autopilot.application.routing_engine import RoutingEngine
from autopilot.domain.entities import CompletionInput, FeatureVector, Message, Prediction
from autopilot.domain.enums import BreakerState, Provider, Role, Tier
from autopilot.domain.errors import ClassifierError, NoCapableModelError
from autopilot.domain.interfaces import ComplexityClassifier
from autopilot.domain.policies.fallback import RoutingConfig, TierRoute
from autopilot.infrastructure.clock import FrozenClock
from autopilot.infrastructure.ml.features import FEATURE_SCHEMA_VERSION
from autopilot.infrastructure.providers.registry import YamlModelRegistry
from autopilot.infrastructure.routing_config import load_routing_config
from tests.conftest import make_model
from tests.unit.test_routing_policies import FakeHealth

CHEAP = "ollama:llama3.1-8b"
MID = "openai:gpt-4o-mini"
PREMIUM = "anthropic:claude-opus-5"


def registry() -> YamlModelRegistry:
    return YamlModelRegistry(
        [
            make_model(CHEAP, provider=Provider.OLLAMA, tier=Tier.SIMPLE, input_cost="0", output_cost="0"),
            make_model(MID, provider=Provider.OPENAI, tier=Tier.MODERATE),
            make_model(
                PREMIUM,
                provider=Provider.ANTHROPIC,
                tier=Tier.COMPLEX,
                input_cost="15",
                output_cost="75",
            ),
        ]
    )


def config(**kwargs) -> RoutingConfig:
    defaults = dict(
        baseline_model_id=PREMIUM,
        tiers={
            Tier.SIMPLE: TierRoute(primary=CHEAP, fallbacks=(MID,)),
            Tier.MODERATE: TierRoute(primary=MID),
            Tier.COMPLEX: TierRoute(primary=PREMIUM),
        },
    )
    return RoutingConfig(**{**defaults, **kwargs})


class StubClassifier(ComplexityClassifier):
    """Returns a fixed prediction, or raises, without touching sklearn."""

    def __init__(
        self, tier: Tier = Tier.SIMPLE, confidence: float = 0.95, *, error: str | None = None
    ) -> None:
        self.version = "stub-1"
        self._tier = tier
        self._confidence = confidence
        self._error = error

    def predict(self, fv: FeatureVector) -> Prediction:
        if self._error:
            raise ClassifierError(self._error)
        return Prediction(
            tier=self._tier, confidence=self._confidence, classifier_version=self.version
        )


def engine(
    *,
    classifier: ComplexityClassifier | None = None,
    health: FakeHealth | None = None,
    cfg: RoutingConfig | None = None,
    reg: YamlModelRegistry | None = None,
) -> RoutingEngine:
    return RoutingEngine(
        reg or registry(),
        cfg or config(),
        classifier=classifier or StubClassifier(),
        health=health,
        clock=FrozenClock(),
    )


def request(text: str = "List the days of the week.", **kwargs) -> CompletionInput:
    return CompletionInput(messages=(Message(role=Role.USER, content=text),), **kwargs)


def test_a_confident_simple_prediction_picks_the_cheapest_model():
    decision = engine().route(request())

    assert decision.chosen_model_id == CHEAP
    assert decision.effective_tier is Tier.SIMPLE
    assert decision.confidence_bumped is False


def test_low_confidence_escalates_a_tier():
    decision = engine(classifier=StubClassifier(Tier.SIMPLE, 0.3)).route(request())

    assert decision.predicted_tier is Tier.SIMPLE
    assert decision.effective_tier is Tier.MODERATE
    assert decision.confidence_bumped is True
    assert decision.chosen_model_id == MID


def test_the_explanation_carries_everything_needed_to_audit_the_decision():
    explanation = engine().route(request()).explanation

    assert explanation is not None
    assert explanation.predicted_tier is Tier.SIMPLE
    assert 0.0 <= explanation.confidence <= 1.0
    assert explanation.estimated_cost_usd >= 0
    assert explanation.estimated_latency_ms > 0
    assert explanation.prompt_tokens > 0
    assert explanation.provider_healthy is True
    assert explanation.provider_state is BreakerState.CLOSED
    assert explanation.summary
    assert explanation.top_features


def test_the_explanation_prices_the_saving_against_the_baseline():
    explanation = engine().route(request()).explanation

    # The cheap local model is free; the premium baseline is not.
    assert explanation.baseline_cost_usd > explanation.estimated_cost_usd
    assert explanation.estimated_savings_usd == (
        explanation.baseline_cost_usd - explanation.estimated_cost_usd
    )


def test_every_candidate_considered_is_recorded_with_a_reason():
    explanation = engine().route(request()).explanation

    assert len(explanation.candidates) == 3
    assert all(c.reason for c in explanation.candidates)
    assert [c.selected for c in explanation.candidates].count(True) == 1


def test_an_open_circuit_diverts_to_the_next_provider_and_says_so():
    decision = engine(health=FakeHealth({Provider.OLLAMA})).route(request())
    explanation = decision.explanation

    assert decision.chosen_model_id == MID
    skipped = explanation.skipped
    assert [s.model_id for s in skipped] == [CHEAP]
    assert "circuit is open" in skipped[0].reason


def test_a_tier_adjustment_is_named_in_the_explanation():
    explanation = engine(classifier=StubClassifier(Tier.SIMPLE, 0.2)).route(request()).explanation

    assert explanation.tier_adjustments
    assert "confidence_bump" in explanation.tier_adjustments[0]


def test_a_broken_classifier_degrades_to_the_heuristic_instead_of_failing():
    decision = engine(classifier=StubClassifier(error="artifact corrupt")).route(request())

    assert decision.classifier_version.startswith("heuristic")
    assert decision.chosen_model_id


def test_no_capable_model_names_why_each_candidate_was_rejected():
    with pytest.raises(NoCapableModelError) as exc:
        engine(
            health=FakeHealth({Provider.OLLAMA, Provider.OPENAI, Provider.ANTHROPIC})
        ).route(request())

    message = str(exc.value)
    assert CHEAP in message and MID in message and PREMIUM in message
    assert "circuit is open" in message


def test_a_request_too_large_for_the_cheap_model_falls_through():
    small = YamlModelRegistry(
        [
            make_model(
                CHEAP,
                provider=Provider.OLLAMA,
                tier=Tier.SIMPLE,
                max_context_tokens=100,
                max_output_tokens=100,
            ),
            make_model(MID, provider=Provider.OPENAI, tier=Tier.MODERATE),
            make_model(PREMIUM, provider=Provider.ANTHROPIC, tier=Tier.COMPLEX),
        ]
    )

    decision = engine(reg=small).route(request("word " * 500))

    assert decision.chosen_model_id != CHEAP
    assert "the model holds 100" in decision.explanation.skipped[0].reason


def test_an_unknown_model_in_the_config_is_skipped_not_fatal():
    stale = config(
        tiers={
            Tier.SIMPLE: TierRoute(primary="openai:retired-model", fallbacks=(CHEAP,)),
            Tier.MODERATE: TierRoute(primary=MID),
            Tier.COMPLEX: TierRoute(primary=PREMIUM),
        }
    )

    decision = engine(cfg=stale).route(request())

    assert decision.chosen_model_id == CHEAP
    assert all(c.model_id != "openai:retired-model" for c in decision.explanation.candidates)


def test_a_missing_baseline_degrades_the_savings_figure_not_the_decision():
    decision = engine(cfg=config(baseline_model_id="openai:does-not-exist")).route(request())

    assert decision.chosen_model_id == CHEAP
    assert decision.explanation.baseline_cost_usd == 0


def test_the_decision_records_the_versions_that_produced_it():
    decision = engine(cfg=config(version=7)).route(request())

    assert decision.routing_config_version == 7
    assert decision.classifier_version == "stub-1"


def test_the_shipped_routing_config_is_valid_and_routes_every_tier():
    from pathlib import Path

    cfg = load_routing_config(Path("configs/routing.yaml"))

    assert cfg.baseline_model_id
    for tier in Tier:
        assert cfg.chain_for(tier)


def test_hot_swapping_the_classifier_leaves_the_original_engine_untouched():
    original = engine()
    swapped = original.with_classifier(StubClassifier(Tier.COMPLEX, 0.99))

    assert original.route(request()).chosen_model_id == CHEAP
    assert swapped.route(request()).chosen_model_id == PREMIUM
    assert original.classifier is not swapped.classifier


def test_the_extractor_and_classifier_agree_on_the_feature_schema():
    # A guard against silently feeding an fx-2 vector to an fx-1 artifact.
    fv = engine()._extractor.extract(request())

    assert fv.schema_version == FEATURE_SCHEMA_VERSION
