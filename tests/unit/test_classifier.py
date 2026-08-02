"""Complexity classifier tests."""

from __future__ import annotations

import pytest

from autopilot.domain.entities import ClassifierMetadata, CompletionInput, FeatureVector, Message
from autopilot.domain.enums import ClassifierStatus, Role, Tier
from autopilot.domain.errors import ClassifierError
from autopilot.infrastructure.ml.classifier import (
    HEURISTIC_CONFIDENCE,
    HeuristicClassifier,
    SklearnClassifier,
)
from autopilot.infrastructure.ml.features import FEATURE_SCHEMA_VERSION, PromptFeatureExtractor

EXTRACTOR = PromptFeatureExtractor()


def features(text: str) -> FeatureVector:
    return EXTRACTOR.extract(
        CompletionInput(messages=(Message(role=Role.USER, content=text),))
    )


@pytest.fixture
def heuristic() -> HeuristicClassifier:
    return HeuristicClassifier()


def test_a_trivial_prompt_is_tier_one(heuristic):
    assert heuristic.predict(features("List the days of the week.")).tier is Tier.SIMPLE


def test_an_analytic_prompt_climbs_above_tier_one(heuristic):
    prediction = heuristic.predict(
        features("Analyze this architecture and evaluate the trade-offs, step by step.")
    )

    assert prediction.tier > Tier.SIMPLE


def test_a_heavily_analytic_prompt_reaches_tier_three(heuristic):
    prediction = heuristic.predict(
        features(
            "Analyze, compare, and evaluate these designs. Explain why each fails, "
            "step by step, and then derive the optimal trade-off. " + "context " * 400
        )
    )

    assert prediction.tier is Tier.COMPLEX


def test_confidence_is_fixed_and_deliberately_low(heuristic):
    # Below the default 0.6 threshold on purpose: the bump policy then routes
    # conservatively rather than trusting an unlearned rule engine.
    assert heuristic.predict(features("anything")).confidence == HEURISTIC_CONFIDENCE
    assert HEURISTIC_CONFIDENCE < 0.6


def test_probabilities_are_uniform_because_rules_have_no_distribution(heuristic):
    probabilities = heuristic.predict(features("hello")).probabilities

    assert set(probabilities) == set(Tier)
    assert len(set(probabilities.values())) == 1


def test_top_features_explain_the_prediction(heuristic):
    prediction = heuristic.predict(features("Analyze the trade-offs step by step."))

    assert prediction.top_features
    names = [f.name for f in prediction.top_features]
    assert "analytic_verb_count" in names
    assert prediction.rationale


def test_contributions_are_ranked_by_magnitude(heuristic):
    prediction = heuristic.predict(features("Analyze and compare these, step by step."))

    magnitudes = [abs(f.contribution) for f in prediction.top_features]
    assert magnitudes == sorted(magnitudes, reverse=True)


def test_simple_verbs_argue_downward(heuristic):
    prediction = heuristic.predict(features("Extract and list the names."))

    simple = next(f for f in prediction.top_features if f.name == "simple_verb_count")
    assert simple.direction == "lowers"


def test_the_version_is_recorded_on_every_prediction():
    classifier = HeuristicClassifier(version="heuristic-test")

    assert classifier.predict(features("hi")).classifier_version == "heuristic-test"


def test_a_rule_classifier_has_no_training_metadata(heuristic):
    assert heuristic.metadata is None


class FakePipeline:
    """Minimal stand-in for a fitted sklearn pipeline."""

    def __init__(self, probabilities: list[float], *, linear: bool = True) -> None:
        self.classes_ = [1, 2, 3]
        self._probabilities = probabilities
        self.named_steps = {}
        if linear:
            self.named_steps = {
                "vectorizer": _FakeVectorizer(),
                "estimator": _FakeEstimator(self.classes_),
            }

    def predict_proba(self, rows):
        return [self._probabilities for _ in rows]


class _FakeVectorizer:
    def get_feature_names_out(self):
        return ["analytic_verb_count", "log_token_count", "simple_verb_count"]


class _FakeEstimator:
    def __init__(self, classes):
        self.classes_ = classes
        self.coef_ = [
            [-1.0, -0.2, 1.5],
            [0.4, 0.1, -0.3],
            [1.8, 0.6, -1.2],
        ]


def metadata(**kwargs) -> ClassifierMetadata:
    from datetime import UTC, datetime

    defaults = dict(
        version="v1",
        status=ClassifierStatus.ACTIVE,
        algorithm="logistic_regression",
        trained_at=datetime(2026, 1, 1, tzinfo=UTC),
        dataset_hash="abc123",
        dataset_size=200,
        feature_schema_version=FEATURE_SCHEMA_VERSION,
    )
    return ClassifierMetadata(**{**defaults, **kwargs})


def test_the_trained_classifier_picks_the_highest_probability_class():
    classifier = SklearnClassifier(FakePipeline([0.1, 0.2, 0.7]), metadata())

    prediction = classifier.predict(features("Analyze this."))

    assert prediction.tier is Tier.COMPLEX
    assert prediction.confidence == 0.7
    assert prediction.probabilities[Tier.SIMPLE] == 0.1


def test_a_schema_mismatch_is_refused_rather_than_silently_mis_scored():
    classifier = SklearnClassifier(FakePipeline([0.1, 0.2, 0.7]), metadata())
    stale = FeatureVector(values={"analytic_verb_count": 1.0}, schema_version="fx-0")

    with pytest.raises(ClassifierError, match="schema mismatch"):
        classifier.predict(stale)


def test_a_pipeline_failure_becomes_a_domain_error():
    class Broken(FakePipeline):
        def predict_proba(self, rows):
            raise RuntimeError("estimator not fitted")

    classifier = SklearnClassifier(Broken([0.3, 0.3, 0.4]), metadata())

    with pytest.raises(ClassifierError, match="failed to predict"):
        classifier.predict(features("hi"))


def test_linear_coefficients_become_feature_contributions():
    classifier = SklearnClassifier(FakePipeline([0.1, 0.2, 0.7]), metadata())

    prediction = classifier.predict(features("Analyze and compare these designs."))

    assert prediction.top_features
    assert all(f.contribution == f.value * f.weight for f in prediction.top_features)


def test_a_non_linear_pipeline_reports_no_attribution_rather_than_guessing():
    classifier = SklearnClassifier(FakePipeline([0.1, 0.2, 0.7], linear=False), metadata())

    assert classifier.predict(features("Analyze this.")).top_features == ()


def test_metadata_exposes_the_promotion_gate_metric():
    classifier = SklearnClassifier(
        FakePipeline([0.5, 0.3, 0.2]), metadata(metrics={"cost_weighted_error": 1.4})
    )

    assert classifier.metadata.cost_weighted_error == 1.4
    assert metadata().cost_weighted_error is None
