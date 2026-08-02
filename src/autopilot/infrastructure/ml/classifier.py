"""Complexity classifiers.

Two implementations of the same port:

* :class:`HeuristicClassifier` — rule-based, always available, no artifact. It
  serves before a model has been trained and whenever loading one fails. It
  reports a deliberately low fixed confidence, which via the confidence-bump
  policy makes it route conservatively rather than confidently wrong.
* :class:`SklearnClassifier` — a trained linear model. Linear is a design
  choice, not a limitation: coefficients give per-feature attribution for free,
  and an explainable model with a working feedback loop beats an opaque one
  without.

Both return the evidence behind a prediction, not just its verdict.
"""

from __future__ import annotations

from typing import Any

from autopilot.domain.entities import (
    ClassifierMetadata,
    FeatureContribution,
    FeatureVector,
    Prediction,
)
from autopilot.domain.enums import Tier
from autopilot.domain.errors import ClassifierError
from autopilot.domain.interfaces import ComplexityClassifier

#: How many drivers to attach to a prediction. Enough to justify a decision in a
#: dashboard tooltip; few enough that the explanation stays readable.
_TOP_FEATURE_COUNT = 5

#: Confidence reported by the cold-start classifier. Below the default threshold
#: on purpose: an unlearned router should escalate, not assert.
HEURISTIC_CONFIDENCE = 0.5


def _rationale(tier: Tier, drivers: tuple[FeatureContribution, ...]) -> str:
    """Render a one-line justification from a prediction's top drivers."""
    if not drivers:
        return f"tier {int(tier)} with no distinguishing signal in the prompt"
    named = ", ".join(f"{d.name}={d.value:g}" for d in drivers[:3])
    return f"tier {int(tier)} driven by {named}"


class HeuristicClassifier(ComplexityClassifier):
    """Rule-based cold-start classifier.

    Scores a handful of features whose relationship to complexity is not in
    doubt — analytic verbs, reasoning markers, sheer length — and thresholds the
    result. It exists so the router is never unavailable, not so it is accurate.
    """

    #: Weights applied to each scored feature. Signs encode direction only; the
    #: magnitudes were chosen so any single strong signal can reach tier 2 but
    #: tier 3 requires agreement between at least two.
    WEIGHTS: dict[str, float] = {
        "analytic_verb_count": 1.2,
        "reasoning_marker_count": 1.0,
        "log_token_count": 0.35,
        "code_fence_count": 0.5,
        "multi_part_count": 0.4,
        "constraint_word_count": 0.15,
        "has_math_notation": 0.5,
        "simple_verb_count": -0.6,
        "question_count": 0.1,
    }

    #: Score at or above which each tier is assigned, highest first.
    THRESHOLDS: tuple[tuple[float, Tier], ...] = (
        (4.0, Tier.COMPLEX),
        (2.0, Tier.MODERATE),
    )

    def __init__(self, version: str = "heuristic-1") -> None:
        """Create the cold-start classifier.

        Args:
            version: Recorded on every decision so heuristic-served traffic is
                distinguishable from model-served traffic in the request log.
        """
        self.version = version

    def predict(self, fv: FeatureVector) -> Prediction:
        """Return a tier from weighted rules, at fixed low confidence."""
        values = fv.values
        contributions = tuple(
            FeatureContribution(
                name=name,
                value=values.get(name, 0.0),
                weight=weight,
                contribution=values.get(name, 0.0) * weight,
            )
            for name, weight in self.WEIGHTS.items()
            if values.get(name, 0.0) != 0.0
        )
        score = sum(c.contribution for c in contributions)
        tier = next((t for threshold, t in self.THRESHOLDS if score >= threshold), Tier.SIMPLE)
        drivers = tuple(
            sorted(contributions, key=lambda c: abs(c.contribution), reverse=True)
        )[:_TOP_FEATURE_COUNT]
        return Prediction(
            tier=tier,
            confidence=HEURISTIC_CONFIDENCE,
            # A uniform distribution is the honest report from a rule engine: it
            # has an opinion about the tier but none about the alternatives.
            probabilities=dict.fromkeys(Tier, round(1 / len(Tier), 4)),
            classifier_version=self.version,
            top_features=drivers,
            rationale=f"{_rationale(tier, drivers)} (heuristic score {score:.2f})",
        )


class SklearnClassifier(ComplexityClassifier):
    """Trained classifier wrapping a fitted scikit-learn pipeline.

    The pipeline is passed in already fitted; this class owns prediction and
    explanation, never training. That keeps scikit-learn out of the request
    path's import graph beyond what inference needs.
    """

    def __init__(self, pipeline: Any, metadata: ClassifierMetadata) -> None:
        """Wrap a fitted pipeline.

        Args:
            pipeline: A fitted estimator exposing ``predict_proba`` over dicts.
            metadata: Provenance of the artifact, recorded on every decision.
        """
        self._pipeline = pipeline
        self._metadata = metadata
        self.version = metadata.version

    @property
    def metadata(self) -> ClassifierMetadata:
        """Provenance of the loaded artifact."""
        return self._metadata

    def predict(self, fv: FeatureVector) -> Prediction:
        """Return the predicted tier with calibrated probabilities and drivers.

        Raises:
            ClassifierError: The vector's schema does not match the artifact's,
                or the underlying pipeline failed.
        """
        expected = self._metadata.feature_schema_version
        if expected != "unknown" and fv.schema_version != expected:
            raise ClassifierError(
                f"Feature schema mismatch: classifier {self.version!r} expects "
                f"{expected!r} but received {fv.schema_version!r}."
            )
        try:
            probabilities = self._pipeline.predict_proba([fv.as_dict()])[0]
            classes = self._pipeline.classes_
        except Exception as exc:
            raise ClassifierError(f"Classifier {self.version!r} failed to predict: {exc}") from exc

        by_tier = {Tier(int(cls)): float(p) for cls, p in zip(classes, probabilities, strict=True)}
        tier = max(by_tier, key=lambda t: by_tier[t])
        drivers = self._contributions(fv, tier)
        return Prediction(
            tier=tier,
            confidence=round(by_tier[tier], 4),
            probabilities={t: round(p, 4) for t, p in by_tier.items()},
            classifier_version=self.version,
            top_features=drivers,
            rationale=_rationale(tier, drivers),
        )

    def _contributions(self, fv: FeatureVector, tier: Tier) -> tuple[FeatureContribution, ...]:
        """Return the largest per-feature contributions to ``tier``'s score.

        Reads coefficients straight off the fitted linear model. Returns nothing
        rather than guessing when the pipeline is not linear — a tree ensemble
        would need SHAP, which costs far more than the router's latency budget.
        """
        try:
            vectorizer = self._pipeline.named_steps["vectorizer"]
            estimator = self._pipeline.named_steps["estimator"]
            coefficients = estimator.coef_
            names = vectorizer.get_feature_names_out()
            row = list(estimator.classes_).index(int(tier))
        except (AttributeError, KeyError, ValueError):
            return ()

        values = fv.values
        contributions = [
            FeatureContribution(
                name=str(name),
                value=values.get(str(name), 0.0),
                weight=float(weight),
                contribution=values.get(str(name), 0.0) * float(weight),
            )
            for name, weight in zip(names, coefficients[row], strict=True)
            if values.get(str(name), 0.0) != 0.0
        ]
        contributions.sort(key=lambda c: abs(c.contribution), reverse=True)
        return tuple(contributions[:_TOP_FEATURE_COUNT])
