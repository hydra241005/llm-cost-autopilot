"""The routing engine.

Turns a request into a ranked, justified list of models to try. It composes
policies rather than branching on tiers: the tier map supplies *candidates*, the
tier policies decide *which* tier's candidates, and the candidate policies
decide which of them are usable. Adding a routing rule means writing a policy
and appending it to a list.

The engine performs no I/O and calls no model. It answers "what should we call,
and why?" — executing that answer is the gateway's job. That separation is what
makes every decision reproducible from the persisted explanation alone.
"""

from __future__ import annotations

from collections.abc import Sequence
from decimal import Decimal

from autopilot.domain.entities import (
    CandidateEvaluation,
    CompletionInput,
    FeatureContribution,
    FeatureVector,
    ModelConfig,
    Prediction,
    RoutingDecision,
    RoutingExplanation,
)
from autopilot.domain.enums import BreakerState, Tier
from autopilot.domain.errors import ClassifierError, ModelNotFoundError, NoCapableModelError
from autopilot.domain.interfaces import Clock, ComplexityClassifier, FeatureExtractor, ModelRegistry
from autopilot.domain.policies.fallback import RoutingConfig
from autopilot.domain.policies.pricing import estimate_cost
from autopilot.domain.policies.routing import (
    ALWAYS_HEALTHY,
    DEFAULT_CANDIDATE_POLICIES,
    DEFAULT_TIER_POLICIES,
    CandidatePolicy,
    HealthView,
    RoutingContext,
    TierPolicy,
)
from autopilot.infrastructure.clock import SystemClock
from autopilot.infrastructure.ml.classifier import HeuristicClassifier
from autopilot.infrastructure.ml.features import (
    PromptFeatureExtractor,
    estimate_tokens,
    prompt_text,
)
from autopilot.infrastructure.observability.logging import get_logger
from autopilot.infrastructure.observability.operational_store import OperationalStore

_log = get_logger(__name__)

#: Assumed completion length when projecting cost before a call. The request's
#: own ``max_tokens`` is a ceiling, not a forecast; most completions land well
#: short of it, and using the ceiling would overstate every projection.
_OUTPUT_ESTIMATE_RATIO = 0.5


class RoutingEngine:
    """Selects the model chain for a request and explains the selection.

    Every collaborator is injected as a domain port, so the engine is testable
    with no network, no filesystem, and no trained artifact.
    """

    def __init__(
        self,
        registry: ModelRegistry,
        config: RoutingConfig,
        *,
        classifier: ComplexityClassifier | None = None,
        extractor: FeatureExtractor | None = None,
        health: HealthView | None = None,
        tier_policies: Sequence[TierPolicy] | None = None,
        candidate_policies: Sequence[CandidatePolicy] | None = None,
        clock: Clock | None = None,
        operational_store: OperationalStore | None = None,
    ) -> None:
        """Wire the engine.

        Args:
            registry: Catalogue of callable models.
            config: Versioned tier-to-model map with fallback chains.
            classifier: Complexity predictor; defaults to the heuristic cold start.
            extractor: Feature extractor; defaults to the prompt extractor.
            health: Live provider health; defaults to treating all as healthy.
            tier_policies: Ordered policies that adjust the effective tier.
            candidate_policies: Policies that rule candidate models in or out.
            clock: Time source, injected so decision latency is testable.
        """
        self._registry = registry
        self._config = config
        self._classifier = classifier or HeuristicClassifier()
        self._extractor = extractor or PromptFeatureExtractor()
        self._health = health or ALWAYS_HEALTHY
        self._tier_policies = tuple(
            DEFAULT_TIER_POLICIES if tier_policies is None else tier_policies
        )
        self._candidate_policies = tuple(
            DEFAULT_CANDIDATE_POLICIES if candidate_policies is None else candidate_policies
        )
        self._clock = clock or SystemClock()
        self._operational_store = operational_store

    @property
    def config(self) -> RoutingConfig:
        """The active routing configuration."""
        return self._config

    @property
    def classifier(self) -> ComplexityClassifier:
        """The classifier currently serving predictions."""
        return self._classifier

    def with_classifier(self, classifier: ComplexityClassifier) -> RoutingEngine:
        """Return a copy of this engine using ``classifier``.

        Used for hot-swapping a newly promoted artifact: a fresh engine is built
        and published atomically, so in-flight decisions finish on the old
        version rather than observing a half-swapped one.
        """
        return RoutingEngine(
            self._registry,
            self._config,
            classifier=classifier,
            extractor=self._extractor,
            health=self._health,
            tier_policies=self._tier_policies,
            candidate_policies=self._candidate_policies,
            clock=self._clock,
        )

    def route(self, req: CompletionInput) -> RoutingDecision:
        """Return the model chain to try for ``req``, with a full explanation.

        Args:
            req: The validated completion request.

        Returns:
            A decision naming the chosen model, the ordered fallbacks behind it,
            and a :class:`RoutingExplanation` accounting for both.

        Raises:
            NoCapableModelError: Every candidate was ruled out. The raised error
                names the reason each one failed, so a 503 is actionable.
        """
        started = self._clock.monotonic()
        features = self._extractor.extract(req)
        prediction = self._predict(features)
        prompt_tokens = estimate_tokens(prompt_text(req))

        ctx = RoutingContext(
            request=req,
            prediction=prediction,
            prompt_tokens=prompt_tokens,
            confidence_threshold=self._config.confidence_threshold,
            health=self._health,
        )
        effective_tier, adjustments = self._apply_tier_policies(prediction.tier, ctx)
        evaluations = self._evaluate_candidates(effective_tier, ctx)
        eligible = [e for e in evaluations if e.eligible]

        chosen = None

        if not eligible:
            # If the chosen effective tier contains no eligible candidates
            # (for example because none of its providers are configured in
            # this environment), attempt to find an eligible candidate in a
            # nearby tier without bypassing candidate policies. This keeps the
            # routing engine authoritative while allowing the system to make
            # a best-effort selection when the configured environment lacks
            # models for the originally selected tier.
            other_tiers = [t for t in (Tier(1), Tier(2), Tier(3))]
            # Order other tiers by distance from the effective tier.
            other_tiers.sort(key=lambda t: abs(int(t) - int(effective_tier)))
            found = False
            for other in other_tiers:
                if other == effective_tier:
                    continue
                other_evals = self._evaluate_candidates(other, ctx)
                other_eligible = [e for e in other_evals if e.eligible]
                if other_eligible:
                    # Promote this alternative tier and pick its top eligible.
                    effective_tier = other
                    evaluations = tuple(
                        e.model_copy(update={"selected": True}) if e.model_id == other_eligible[0].model_id else e
                        for e in other_evals
                    )
                    chosen = other_eligible[0]
                    found = True
                    break
            if not found:
                raise NoCapableModelError(self._no_capable_message(effective_tier, evaluations))

            # Refresh eligible list to reflect the chosen/updated evaluations.
            eligible = [e for e in evaluations if e.eligible]

        if chosen is None:
            chosen = eligible[0]
        evaluations = tuple(
            e.model_copy(update={"selected": True}) if e.model_id == chosen.model_id else e
            for e in evaluations
        )
        explanation = self._explain(
            prediction=prediction,
            effective_tier=effective_tier,
            adjustments=adjustments,
            evaluations=evaluations,
            chosen=chosen,
            ctx=ctx,
            features=features,
        )
        decision = RoutingDecision(
            predicted_tier=prediction.tier,
            effective_tier=effective_tier,
            confidence=prediction.confidence,
            chosen_model_id=chosen.model_id,
            candidate_model_ids=tuple(e.model_id for e in eligible),
            confidence_bumped=any("confidence" in a for a in adjustments),
            classifier_version=prediction.classifier_version,
            routing_config_version=self._config.version,
            decision_latency_ms=round((self._clock.monotonic() - started) * 1000, 3),
            explanation=explanation,
        )
        _log.info(
            "routing.decided",
            predicted_tier=int(prediction.tier),
            effective_tier=int(effective_tier),
            confidence=prediction.confidence,
            chosen_model_id=decision.chosen_model_id,
            candidates=len(decision.candidate_model_ids),
            skipped=len(explanation.skipped),
            classifier_version=decision.classifier_version,
            decision_latency_ms=decision.decision_latency_ms,
            estimated_cost_usd=str(explanation.estimated_cost_usd),
        )
        if self._operational_store is not None:
            self._operational_store.record_routing_decision(decision)
        return decision

    def _predict(self, features: FeatureVector) -> Prediction:
        """Predict a tier, degrading to the heuristic if the artifact fails.

        A classifier that cannot load or cannot predict must not take the API
        down with it: routing conservatively is a far better failure mode than
        returning 500 for every request.
        """
        try:
            return self._classifier.predict(features)
        except ClassifierError as exc:
            _log.warning(
                "routing.classifier_failed",
                classifier_version=self._classifier.version,
                error=str(exc),
            )
            return HeuristicClassifier().predict(features)

    def _apply_tier_policies(
        self, tier: Tier, ctx: RoutingContext
    ) -> tuple[Tier, tuple[str, ...]]:
        """Run every tier policy in order, collecting the reasons for each move."""
        reasons: list[str] = []
        for policy in self._tier_policies:
            adjustment = policy.apply(tier, ctx)
            if adjustment.changed:
                reasons.append(f"{policy.name}: {adjustment.reason}")
                tier = adjustment.tier
        return tier, tuple(reasons)

    def _evaluate_candidates(
        self, tier: Tier, ctx: RoutingContext
    ) -> tuple[CandidateEvaluation, ...]:
        """Evaluate every candidate for ``tier``, in fallback-chain order."""
        evaluations: list[CandidateEvaluation] = []
        for model_id in self._config.escalation_chain(tier):
            try:
                model = self._registry.get(model_id)
            except ModelNotFoundError:
                # A stale routing config must not take out the whole chain; the
                # remaining candidates are still perfectly serviceable.
                _log.warning("routing.unknown_candidate", model_id=model_id)
                continue
            evaluations.append(self._evaluate(model, ctx))
        return tuple(evaluations)

    def _evaluate(self, model: ModelConfig, ctx: RoutingContext) -> CandidateEvaluation:
        """Run the candidate policies against one model and price the outcome."""
        reason = "eligible"
        eligible = True
        for policy in self._candidate_policies:
            verdict = policy.evaluate(model, ctx)
            if not verdict.eligible:
                eligible = False
                reason = verdict.reason
                break
        return CandidateEvaluation(
            model_id=model.id,
            provider=model.provider,
            tier=model.tier,
            eligible=eligible,
            reason=reason,
            estimated_cost_usd=self._estimate(model, ctx),
            expected_latency_ms=model.expected_latency_ms,
            provider_state=self._health.state(model.provider),
        )

    def _estimate(self, model: ModelConfig, ctx: RoutingContext) -> Decimal:
        """Project the cost of serving this request on ``model``."""
        return estimate_cost(
            input_tokens=ctx.prompt_tokens,
            expected_output_tokens=int(ctx.request.max_tokens * _OUTPUT_ESTIMATE_RATIO),
            model=model,
        )

    def _explain(
        self,
        *,
        prediction: Prediction,
        effective_tier: Tier,
        adjustments: tuple[str, ...],
        evaluations: tuple[CandidateEvaluation, ...],
        chosen: CandidateEvaluation,
        ctx: RoutingContext,
        features: FeatureVector,
    ) -> RoutingExplanation:
        """Assemble the full explanation for a completed decision."""
        baseline = self._baseline_estimate(ctx)
        return RoutingExplanation(
            predicted_tier=prediction.tier,
            effective_tier=effective_tier,
            confidence=prediction.confidence,
            estimated_cost_usd=chosen.estimated_cost_usd,
            baseline_cost_usd=baseline,
            estimated_latency_ms=chosen.expected_latency_ms,
            prompt_tokens=ctx.prompt_tokens,
            provider_state=chosen.provider_state,
            provider_healthy=chosen.provider_state is not BreakerState.OPEN,
            tier_adjustments=adjustments,
            candidates=evaluations,
            top_features=prediction.top_features or self._fallback_features(features),
            summary=self._summarize(prediction, effective_tier, chosen, evaluations, baseline),
        )

    def _baseline_estimate(self, ctx: RoutingContext) -> Decimal:
        """Project what the premium baseline model would have cost.

        Returns zero when the configured baseline is missing from the registry:
        a broken savings figure should not fail a request that routed perfectly
        well, so the number degrades rather than the decision.
        """
        try:
            baseline = self._registry.get(self._config.baseline_model_id)
        except ModelNotFoundError:
            _log.warning("routing.baseline_missing", model_id=self._config.baseline_model_id)
            return Decimal("0")
        return self._estimate(baseline, ctx)

    @staticmethod
    def _fallback_features(features: FeatureVector) -> tuple[FeatureContribution, ...]:
        """Return raw top features when the classifier offered no attribution."""
        return tuple(
            FeatureContribution(name=name, value=value, weight=0.0, contribution=0.0)
            for name, value in features.top()
        )

    @staticmethod
    def _summarize(
        prediction: Prediction,
        effective_tier: Tier,
        chosen: CandidateEvaluation,
        evaluations: tuple[CandidateEvaluation, ...],
        baseline: Decimal,
    ) -> str:
        """Render the one-sentence justification shown in the dashboard."""
        parts = [
            f"Classified tier {int(prediction.tier)} at {prediction.confidence:.0%} confidence"
        ]
        if effective_tier is not prediction.tier:
            parts.append(f"routed as tier {int(effective_tier)} after policy adjustment")
        skipped = [e for e in evaluations if not e.eligible]
        if skipped:
            parts.append(f"{len(skipped)} cheaper candidate(s) skipped")
        saving = baseline - chosen.estimated_cost_usd
        parts.append(
            f"selected {chosen.model_id} at an estimated ${chosen.estimated_cost_usd:f} "
            f"(${saving:f} below baseline)"
        )
        return "; ".join(parts) + "."

    @staticmethod
    def _no_capable_message(tier: Tier, evaluations: tuple[CandidateEvaluation, ...]) -> str:
        """Render an actionable message naming why each candidate was rejected."""
        if not evaluations:
            return f"No models are registered for tier {int(tier)} or above."
        detail = "; ".join(f"{e.model_id}: {e.reason}" for e in evaluations)
        return f"No capable model for tier {int(tier)}. Candidates rejected — {detail}."
