from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import UTC, datetime
from decimal import Decimal

import pytest
from asgi_lifespan import LifespanManager
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from autopilot.api.main import create_app
from autopilot.config import AppSettings
from autopilot.domain.entities import (
    CandidateEvaluation,
    ClassifierMetadata,
    FeatureContribution,
    RoutingDecision,
    RoutingExplanation,
)
from autopilot.domain.enums import BreakerState, ClassifierStatus, Tier
from autopilot.infrastructure.observability.operational_store import OperationalStore


@pytest.fixture
def settings() -> AppSettings:
    return AppSettings(environment="test", log_json=False)


@pytest.fixture
def app(settings: AppSettings) -> FastAPI:
    return create_app(settings)


@pytest.fixture
async def client(app: FastAPI) -> AsyncIterator[AsyncClient]:
    async with LifespanManager(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            yield c


def test_operational_store_tracks_shadow_and_ab_outcomes() -> None:
    store = OperationalStore()
    metadata = ClassifierMetadata(
        version="v3",
        status=ClassifierStatus.CANDIDATE,
        algorithm="logistic_regression",
        trained_at=datetime.now(UTC),
        dataset_hash="hash",
        dataset_size=20,
        feature_schema_version="fx-1",
    )

    store.record_event("classifier_trained", details={"version": "v3"})
    store.record_shadow_evaluation(
        request_id="req-1",
        production_version="v2",
        candidate_version="v3",
        predicted_tier=Tier.COMPLEX,
        confidence=0.91,
        latency_ms=12.5,
        feature_summary=(("reasoning_marker_count", 2.0),),
        agreement=False,
        routing_explanation="candidate flagged complexity",
        metadata=metadata,
    )
    store.record_ab_decision(
        request_id="req-2",
        split="95/5",
        bucket="control",
        routing_accuracy=0.75,
        disagreement_rate=0.1,
        quality_score=0.8,
        estimated_cost=Decimal("0.02"),
        cost_savings=Decimal("0.01"),
        latency_ms=150.0,
    )

    summary = store.metrics_summary()
    assert summary["routing_decisions"] == 2
    assert summary["shadow_evaluations"] == 1
    assert summary["ab_buckets"]["control"] == 1


@pytest.mark.asyncio
async def test_admin_metrics_and_explainability_endpoints(
    client: AsyncClient, app: FastAPI
) -> None:
    decision = RoutingDecision(
        request_id="req-100",
        predicted_tier=Tier.SIMPLE,
        effective_tier=Tier.SIMPLE,
        confidence=0.86,
        chosen_model_id="openai:gpt-4o",
        candidate_model_ids=("openai:gpt-4o",),
        classifier_version="v2",
        explanation=RoutingExplanation(
            predicted_tier=Tier.SIMPLE,
            effective_tier=Tier.SIMPLE,
            confidence=0.86,
            estimated_cost_usd=Decimal("0.01"),
            baseline_cost_usd=Decimal("0.02"),
            estimated_latency_ms=100,
            prompt_tokens=120,
            provider_state=BreakerState.CLOSED,
            provider_healthy=True,
            tier_adjustments=(),
            candidates=(
                CandidateEvaluation(
                    model_id="openai:gpt-4o",
                    provider="openai",
                    tier=Tier.SIMPLE,
                    eligible=True,
                    reason="eligible",
                    estimated_cost_usd=Decimal("0.01"),
                    expected_latency_ms=120,
                    provider_state=BreakerState.CLOSED,
                    selected=True,
                ),
            ),
            top_features=(FeatureContribution(name="question_count", value=1.0, weight=1.0, contribution=1.0),),
            summary="cheap and sufficient",
        ),
    )
    store = app.state.operational_store
    store.record_routing_decision(decision)

    response = await client.get("/admin/metrics")
    assert response.status_code == 200

    explain = await client.get("/admin/routing/explain/req-100")
    assert explain.status_code == 200
    payload = explain.json()
    assert payload["request_id"] == "req-100"
    assert payload["final_decision"]["chosen_model_id"] == "openai:gpt-4o"
