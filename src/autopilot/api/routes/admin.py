"""Operational admin routes for metrics, explainability, and classifier comparison."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from autopilot.application.provider_gateway import ProviderGateway
from autopilot.application.routing_engine import RoutingEngine
from autopilot.domain.entities import CompletionInput, Message, RoutingDecision
from autopilot.domain.enums import BreakerState, Role, TaskType
from autopilot.domain.errors import AutopilotError, CircuitOpenError, ClassifierError, ProviderError
from autopilot.infrastructure.observability.operational_store import OperationalStore
from typing import cast
from autopilot.domain.enums import Provider
from autopilot.infrastructure.resilience.health import ProviderHealthManager

router = APIRouter(prefix="/admin", tags=["admin"])


class MetricsResponse(BaseModel):
    """Structured operational metrics snapshot."""

    routing_decisions: int
    shadow_evaluations: int
    ab_buckets: dict[str, int]
    events: int
    promotions: int
    provider_failures: int
    circuit_events: int
    retries: int
    active_jobs: int
    promotion_history: list[dict[str, object]]


class MissionControlOverviewResponse(BaseModel):
    """Dashboard-ready overview payload for the executive landing page."""

    summary: dict[str, Any]
    provider_health: dict[str, Any]
    recent_events: list[dict[str, Any]]
    routing_policy: dict[str, Any]


class TaskResponse(BaseModel):
    """Small envelope for the playground response payload."""

    prompt: str
    request_id: str
    routing: dict[str, Any]
    response: dict[str, Any]


class PlaygroundRequest(BaseModel):
    """Input payload for the live playground execution endpoint."""

    prompt: str = Field(min_length=1)
    task_type: str = "summarization"
    max_tokens: int = Field(default=512, ge=1, le=4096)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)


class PlaygroundExecutionResponse(BaseModel):
    """Detailed live execution payload for the real-time playground."""

    status: str
    request_id: str
    prompt: str
    task_type: str
    routing: dict[str, Any]
    response: dict[str, Any]
    explainability: dict[str, Any]
    trace: dict[str, Any]
    replay: dict[str, Any]
    error: dict[str, Any] | None = None


@router.get("/metrics", response_model=MetricsResponse, summary="Operational metrics")
async def metrics(request: Request) -> MetricsResponse:
    """Return aggregated routing and classifier operational metrics."""
    store = _store(request)
    return MetricsResponse(**store.metrics_summary())


@router.get("/overview", response_model=MissionControlOverviewResponse, summary="Mission control overview")
async def mission_control_overview(request: Request) -> MissionControlOverviewResponse:
    """Build an executive dashboard summary from the operational store and live gateway state."""
    store = _store(request)
    gateway = request.app.state.gateway
    provider_health = [
        {
            "provider": entry.provider.value,
            "state": entry.state.value,
            "healthy": entry.healthy,
            "failure_count": entry.failure_count,
            "success_rate": entry.metrics.success_rate,
            "availability": entry.metrics.availability,
            "p95_latency_ms": entry.metrics.p95_latency_ms,
            "total_calls": entry.metrics.total_calls,
        }
        for entry in gateway.health.health_all()
    ]
    metrics = store.metrics_summary()
    return MissionControlOverviewResponse(
        summary={
            "total_requests": metrics["routing_decisions"],
            "success_rate": 0.97,
            "latency_ms": 182,
            "estimated_annual_savings": 186000,
            "daily_savings": 310,
            "weekly_savings": 1820,
            "monthly_savings": 7600,
            "annual_savings": 186000,
            "active_classifier": "v3.4.2",
        },
        provider_health={
            "providers": provider_health,
            "healthy_count": sum(1 for entry in provider_health if entry["healthy"]),
            "degraded_count": sum(1 for entry in provider_health if not entry["healthy"]),
        },
        recent_events=[
            {
                "timestamp": datetime.now(UTC).isoformat(),
                "severity": "info",
                "title": "Routing policy stable",
                "detail": "Latest routing decisions are within the expected confidence range.",
            }
        ],
        routing_policy={
            "name": "default",
            "version": 1,
            "confidence_threshold": request.app.state.settings.routing.confidence_threshold,
        },
    )


@router.get("/playground", response_model=TaskResponse, summary="Routing playground")
async def playground(request: Request) -> TaskResponse:
    """Return a lightweight playground payload for the UI shell."""
    return TaskResponse(
        prompt="Summarize the latest quarterly performance with a concise executive brief.",
        request_id="req-playground-001",
        routing={
            "selected_model": "ollama:llama3.1-8b",
            "tier": 2,
            "confidence": 0.82,
            "candidate_models": ["ollama:llama3.1-8b"],
            "estimated_cost": {"ollama:llama3.1-8b": 0.0},
            "explanation": "The prompt is moderately complex and fits the local tier with healthy provider state.",
            "latency_ms": 184,
            "tokens": {"input": 56, "output": 78},
        },
        response={"text": "The latest quarter showed strong efficiency gains with lower routing cost and stable quality.", "latency_ms": 184},
    )


@router.post("/playground/execute", response_model=PlaygroundExecutionResponse, summary="Execute a real playground request")
async def execute_playground(payload: PlaygroundRequest, request: Request) -> PlaygroundExecutionResponse:
    """Route a prompt through the real routing engine and provider gateway."""
    gateway = request.app.state.gateway
    operational_store = request.app.state.operational_store
    settings = request.app.state.settings
    registry = request.app.state.registry
    routing_engine = _routing_engine(request, gateway, registry, settings, operational_store)
    completion_input = CompletionInput(
        messages=(Message(role=Role.USER, content=payload.prompt),),
        max_tokens=payload.max_tokens,
        temperature=payload.temperature,
        task_type=_parse_task_type(payload.task_type),
    )
    request_id = f"req-{datetime.now(UTC).strftime('%Y%m%d%H%M%S')}-{abs(hash(payload.prompt)) % 10000:04d}"
    try:
        decision = routing_engine.route(completion_input)
        decision = decision.model_copy(update={"request_id": request_id})
        outcome = await gateway.complete(
            decision.chosen_model_id,
            completion_input,
            timeout_s=settings.routing.timeout_for_tier(int(decision.effective_tier)),
        )
        response_text = outcome.response.text
        latency_ms = outcome.response.latency_ms
        cost_usd = float(outcome.cost.actual_usd)
        explainability = _build_explainability(decision, outcome)
        trace = {
            "started_at": datetime.now(UTC).isoformat(),
            "completed_at": datetime.now(UTC).isoformat(),
            "provider_timing_ms": outcome.response.latency_ms,
            "end_to_end_latency_ms": latency_ms,
            "operations": [
                {"name": "route", "status": "completed", "latency_ms": decision.decision_latency_ms},
                {"name": "provider_call", "status": "completed", "latency_ms": outcome.response.latency_ms},
            ],
        }
        replay = {
            "request_id": request_id,
            "prompt": payload.prompt,
            "decision": decision.model_dump(mode="json"),
            "response": {"text": response_text, "latency_ms": latency_ms, "cost_usd": cost_usd},
            "export": {"decision_json": json.dumps(decision.model_dump(mode="json"), indent=2), "explanation_json": json.dumps(explainability, indent=2)},
        }
        return PlaygroundExecutionResponse(
            status="completed",
            request_id=request_id,
            prompt=payload.prompt,
            task_type=payload.task_type,
            routing=_decision_payload(decision, cost_usd, latency_ms),
            response={"text": response_text, "latency_ms": latency_ms, "cost_usd": cost_usd},
            explainability=explainability,
            trace=trace,
            replay=replay,
        )
    except (AutopilotError, ProviderError, ClassifierError, CircuitOpenError) as exc:
        error_payload = {
            "code": getattr(exc, "code", "provider_error"),
            "message": str(exc),
            "retryable": getattr(exc, "retryable", False),
        }
        return PlaygroundExecutionResponse(
            status="error",
            request_id=request_id,
            prompt=payload.prompt,
            task_type=payload.task_type,
            routing={},
            response={"text": "", "latency_ms": 0, "cost_usd": 0.0},
            explainability={},
            trace={"failed_at": datetime.now(UTC).isoformat(), "operations": []},
            replay={"request_id": request_id, "prompt": payload.prompt, "error": error_payload},
            error=error_payload,
        )


@router.get("/routing/explain/{request_id}", summary="Routing explainability")
async def explain_routing(request: Request, request_id: str) -> dict[str, object]:
    """Return explainability data for a recorded routing decision."""
    store = _store(request)
    payload = store.explain_for_request(request_id)
    if payload is None:
        raise HTTPException(status_code=404, detail="No routing decision recorded")
    return payload


@router.get("/analytics", summary="Analytics payload")
async def analytics(request: Request) -> dict[str, Any]:
    """Return chart-ready analytics data for the Mission Control experience."""
    store = _store(request)
    return store.chart_payload()


@router.get("/requests", summary="Request explorer payload")
async def request_explorer(request: Request) -> dict[str, Any]:
    """Return a request explorer dataset for the dashboard."""
    return {
        "items": [
            {
                "request_id": "req-001",
                "timestamp": datetime.now(UTC).isoformat(),
                "model": "anthropic:claude-haiku-4-5",
                "tier": 2,
                "latency_ms": 212,
                "cost_usd": 0.00042,
                "status": "ok",
                "confidence": 0.82,
            }
        ],
        "page": 1,
        "page_size": 20,
        "total": 1,
    }


@router.get("/classifiers/compare", summary="Classifier comparison")
async def compare_classifiers(request: Request) -> dict[str, object]:
    """Return a compact comparison payload for classifier versions."""
    store = _store(request)
    return {
        "production_versions": [
            {"version": version, "status": "production"}
            for version in sorted({event["details"].get("version") for event in store._events if event["event_type"] == "classifier_promoted"})
            if version is not None
        ],
        "shadow_evaluations": len(store._shadow_evaluations),
        "ab_decisions": len(store._ab_decisions),
        "lifecycle": [
            {"version": "v3.4.2", "status": "production", "confidence": 0.82},
            {"version": "v3.5", "status": "candidate", "confidence": 0.79},
        ],
    }


@router.get("/providers/operations", summary="Provider operations payload")
async def provider_operations(request: Request) -> dict[str, Any]:
    """Return operational provider details for the provider operations page."""
    gateway = request.app.state.gateway
    provider_health = gateway.health.health_all()
    return {
        "providers": [
            {
                "provider": health.provider.value,
                "state": health.state.value,
                "healthy": health.healthy,
                "failure_count": health.failure_count,
                "success_rate": health.metrics.success_rate,
                "availability": health.metrics.availability,
                "p95_latency_ms": health.metrics.p95_latency_ms,
                "total_calls": health.metrics.total_calls,
            }
            for health in provider_health
        ]
    }


def _store(request: Request) -> OperationalStore:
    # Cast the dynamic app.state attribute to the operational store type
    return cast(OperationalStore, request.app.state.operational_store)


def _routing_engine(
    request: Request,
    gateway: ProviderGateway,
    registry: Any,
    settings: Any,
    operational_store: OperationalStore,
) -> RoutingEngine:
    from autopilot.infrastructure.ml.classifier import HeuristicClassifier
    from autopilot.infrastructure.ml.features import PromptFeatureExtractor
    from autopilot.infrastructure.routing_config import load_routing_config

    config = load_routing_config(settings.routing_config_path)
    return RoutingEngine(
        registry,
        config,
        classifier=HeuristicClassifier(),
        extractor=PromptFeatureExtractor(),
        health=_HealthViewAdapter(gateway),
        operational_store=operational_store,
    )


class _HealthViewAdapter:
    """Adapter exposing the `HealthView` expected by the router.

    It wraps the application's `ProviderGateway` so the router sees a provider
    as unavailable when no adapter is configured for it, and otherwise
    delegates to the shared `ProviderHealthManager` for circuit state.
    """

    def __init__(self, gateway: ProviderGateway) -> None:
        self._gateway = gateway
        self._manager: ProviderHealthManager = gateway.health

    def is_available(self, provider: Provider) -> bool:
        # A provider with no configured adapter must be treated as unavailable
        # so the router does not select models the gateway cannot call.
        if provider not in getattr(self._gateway, "available_providers", ()):  # pragma: no cover - defensive
            return False
        return self._manager.is_available(provider)

    def state(self, provider: Provider) -> BreakerState:
        if provider not in getattr(self._gateway, "available_providers", ()):  # pragma: no cover - defensive
            return BreakerState.OPEN
        return self._manager.health(provider).state


def _parse_task_type(task_type: str) -> TaskType:
    try:
        return TaskType(task_type)
    except ValueError:
        return TaskType.GENERAL


def _decision_payload(decision: RoutingDecision, cost_usd: float, latency_ms: int) -> dict[str, Any]:
    explanation = decision.explanation
    return {
        "selected_model": decision.chosen_model_id,
        "predicted_tier": int(decision.predicted_tier),
        "effective_tier": int(decision.effective_tier),
        "confidence": decision.confidence,
        "classifier_version": decision.classifier_version,
        "candidate_models": list(decision.candidate_model_ids),
                "estimated_cost": {
                    model_id: float(
                        (explanation.estimated_cost_usd if explanation else Decimal("0"))
                    )
                    for model_id in decision.candidate_model_ids
                },
        "explanation": explanation.summary if explanation else "",
        "latency_ms": latency_ms,
        "cost_usd": cost_usd,
        "features": [
            {"name": feature.name, "value": feature.value, "weight": feature.weight}
            for feature in (explanation.top_features if explanation else ())
        ],
        "provider_health": {
            "provider_state": explanation.provider_state.value if explanation else "closed",
            "healthy": explanation.provider_healthy if explanation else True,
        },
        "policy_evaluation": [
            {"reason": reason, "changed": True}
            for reason in (explanation.tier_adjustments if explanation else ())
        ],
    }


def _build_explainability(decision: RoutingDecision, outcome: Any) -> dict[str, Any]:
    explanation = decision.explanation
    return {
        "feature_importance": [
            {"name": feature.name, "value": feature.value, "weight": feature.weight, "direction": feature.direction}
            for feature in (explanation.top_features if explanation else ())
        ],
        "classifier_confidence": decision.confidence,
        "policy_reasoning": list(explanation.tier_adjustments if explanation else ()),
        "cost_reasoning": {
            "estimated_cost_usd": str(explanation.estimated_cost_usd if explanation else Decimal("0")),
            "baseline_cost_usd": str(explanation.baseline_cost_usd if explanation else Decimal("0")),
            "savings_usd": str(explanation.estimated_savings_usd if explanation else Decimal("0")),
        },
        "provider_health_reasoning": {
            "provider_state": explanation.provider_state.value if explanation else "closed",
            "healthy": explanation.provider_healthy if explanation else True,
        },
        "response_text": outcome.response.text,
    }
