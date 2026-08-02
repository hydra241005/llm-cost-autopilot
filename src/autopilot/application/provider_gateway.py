"""The unified request interface.

One call site for every model, regardless of vendor: give it a registry model id
and a request, and it resolves the adapter, applies the tier's timeout budget,
retries transient failures, and prices the result.

This is deliberately *not* the router. It has no opinion about which model should
serve a request — it only makes calling any of them uniform. The routing engine
built in Phase 2 sits on top of this.
"""

from __future__ import annotations

from collections.abc import Mapping
from decimal import Decimal

from autopilot.domain.entities import (
    CompletionInput,
    CostBreakdown,
    ModelConfig,
    ProviderResponse,
    Usage,
)
from autopilot.domain.enums import Provider
from autopilot.domain.errors import ConfigurationError, NoCapableModelError
from autopilot.domain.interfaces import LLMProvider, ModelRegistry
from autopilot.domain.policies.pricing import build_cost_breakdown, compute_cost
from autopilot.infrastructure.observability.logging import get_logger
from autopilot.infrastructure.resilience.health import ProviderHealthManager
from autopilot.infrastructure.resilience.retry import RetryPolicy, call_with_retry

_log = get_logger(__name__)

#: Fixed synthetic workload used only to rank models when no baseline is configured.
_BASELINE_RANKING_USAGE = Usage(input_tokens=1_000, output_tokens=500)



class CompletionOutcome:
    """A provider response paired with its cost accounting.

    Attributes:
        response: The normalized provider response.
        cost: Actual versus counterfactual-baseline cost for this call.
        attempts: How many provider calls were made, including retries.
    """

    __slots__ = ("attempts", "cost", "response")

    def __init__(self, response: ProviderResponse, cost: CostBreakdown, attempts: int) -> None:
        """Bundle a response with its cost breakdown."""
        self.response = response
        self.cost = cost
        self.attempts = attempts

    @property
    def cost_usd(self) -> Decimal:
        """Actual USD cost of the call."""
        return self.cost.actual_usd

    @property
    def saved_usd(self) -> Decimal:
        """USD saved versus the counterfactual baseline model."""
        return self.cost.saved_usd


class ProviderGateway:
    """Executes completions against any registered model through one interface.

    Adapters are injected as domain ports, so tests substitute fakes without
    touching the network and the gateway never imports a vendor SDK.
    """

    def __init__(
        self,
        registry: ModelRegistry,
        adapters: Mapping[Provider, LLMProvider],
        *,
        retry_policy: RetryPolicy | None = None,
        timeouts_by_tier: Mapping[int, float] | None = None,
        baseline_model_id: str | None = None,
        health: ProviderHealthManager | None = None,
    ) -> None:
        """Wire the gateway.

        Args:
            registry: Catalogue of callable models.
            adapters: One adapter per configured provider.
            retry_policy: Backoff parameters; defaults to two retries.
            timeouts_by_tier: Per-tier wall-clock budgets in seconds.
            baseline_model_id: Premium model used as the savings counterfactual.
                Defaults to the most expensive registered model.
            health: Circuit-breaker and metrics tracker. One is created over the
                supplied adapters when not injected.

        Raises:
            ConfigurationError: No adapters were supplied.
        """
        if not adapters:
            raise ConfigurationError("ProviderGateway requires at least one provider adapter.")
        self._registry = registry
        self._adapters = dict(adapters)
        self._retry_policy = retry_policy or RetryPolicy()
        self._timeouts = dict(timeouts_by_tier or {1: 10.0, 2: 20.0, 3: 45.0})
        self._baseline_model_id = baseline_model_id
        self._health = health or ProviderHealthManager(self._adapters.keys())

    @property
    def health(self) -> ProviderHealthManager:
        """Circuit state and rolling metrics for every configured provider."""
        return self._health

    @property
    def available_providers(self) -> frozenset[Provider]:
        """Providers that have a configured adapter."""
        return frozenset(self._adapters)

    def adapter_for(self, provider: Provider) -> LLMProvider | None:
        """Return the adapter for ``provider``, or ``None`` when unconfigured."""
        return self._adapters.get(provider)

    def supports(self, model_id: str) -> bool:
        """Return whether ``model_id`` is registered, active, and has an adapter."""
        try:
            model = self._registry.get(model_id)
        except Exception:
            return False
        return model.active and model.provider in self._adapters

    async def complete(
        self,
        model_id: str,
        req: CompletionInput,
        *,
        timeout_s: float | None = None,
    ) -> CompletionOutcome:
        """Call ``model_id`` with ``req``, retrying transient failures.

        Args:
            model_id: Registry id of the model to call.
            req: The provider-agnostic request.
            timeout_s: Override for the tier's default timeout budget.

        Returns:
            The response together with its cost breakdown.

        Raises:
            ModelNotFoundError: ``model_id`` is not registered.
            NoCapableModelError: The model is inactive or has no configured adapter.
            CircuitOpenError: The provider's circuit is open.
            ProviderError: The call failed and retries were exhausted or the
                failure was not retryable.
        """
        model = self._registry.get(model_id)
        adapter = self._resolve_adapter(model)
        budget = timeout_s if timeout_s is not None else self._timeouts.get(int(model.tier), 30.0)

        attempts = 0

        async def _call() -> ProviderResponse:
            nonlocal attempts
            attempts += 1
            # Checked per attempt, not once up front: a retry must not slip past
            # a circuit that opened while an earlier attempt was in flight.
            self._health.check(model.provider)
            try:
                response = await adapter.complete(req, model, budget)
            except Exception as exc:
                self._health.record_failure(model.provider, exc)
                raise
            self._health.record_success(model.provider, response.latency_ms)
            return response

        response = await call_with_retry(
            _call,
            self._retry_policy,
            context={"model_id": model.id, "provider": model.provider.value},
        )

        cost = build_cost_breakdown(response.usage, model, self._baseline_model(model))
        _log.info(
            "provider.completed",
            model_id=model.id,
            provider=model.provider.value,
            tier=int(model.tier),
            attempts=attempts,
            latency_ms=response.latency_ms,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            cost_usd=str(cost.actual_usd),
            saved_usd=str(cost.saved_usd),
            finish_reason=response.finish_reason.value,
        )
        return CompletionOutcome(response=response, cost=cost, attempts=attempts)

    def _resolve_adapter(self, model: ModelConfig) -> LLMProvider:
        """Return the adapter for ``model``.

        Raises:
            NoCapableModelError: The model is inactive or its provider is unconfigured.
        """
        if not model.active:
            raise NoCapableModelError(f"Model {model.id!r} is deactivated in the registry.")
        adapter = self._adapters.get(model.provider)
        if adapter is None:
            raise NoCapableModelError(
                f"No adapter configured for provider {model.provider.value!r} "
                f"(required by model {model.id!r})."
            )
        return adapter

    def _baseline_model(self, fallback: ModelConfig) -> ModelConfig:
        """Return the counterfactual baseline model.

        Falls back to the most expensive registered model when no baseline is
        configured, and to the served model itself when the registry holds only
        one entry — which correctly reports zero savings rather than crashing.
        """
        if self._baseline_model_id is not None:
            return self._registry.get(self._baseline_model_id)
        priced = [m for m in self._registry.all() if not m.is_free]
        if not priced:
            return fallback
        return max(priced, key=lambda m: compute_cost(_BASELINE_RANKING_USAGE, m))

    async def aclose(self) -> None:
        """Close every adapter this gateway holds."""
        for adapter in self._adapters.values():
            await adapter.aclose()
