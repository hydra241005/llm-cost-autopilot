from __future__ import annotations

from collections.abc import AsyncIterator

import pytest
from asgi_lifespan import LifespanManager
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from autopilot import __version__
from autopilot.api.main import create_app
from autopilot.config import AppSettings
from autopilot.domain.errors import ProviderServerError


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


async def test_health_reports_a_ready_application(client):
    response = await client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["version"] == __version__
    assert body["model_count"] >= 6
    assert "ollama" in body["providers"]


async def test_models_endpoint_returns_the_catalogue(client):
    response = await client.get("/v1/models")

    assert response.status_code == 200
    body = response.json()
    assert body["count"] == len(body["data"])
    ids = {m["id"] for m in body["data"]}
    assert "openai:gpt-4o" in ids
    assert "ollama:llama3.1-8b" in ids


async def test_models_endpoint_filters_by_tier(client):
    response = await client.get("/v1/models", params={"tier": 1})

    assert response.status_code == 200
    body = response.json()
    assert body["count"] > 0
    assert all(m["tier"] == 1 for m in body["data"])


async def test_models_endpoint_rejects_an_invalid_tier(client):
    response = await client.get("/v1/models", params={"tier": 9})

    assert response.status_code == 422


async def test_local_model_is_callable_without_any_credentials(client):
    body = (await client.get("/v1/models")).json()
    local = next(m for m in body["data"] if m["id"] == "ollama:llama3.1-8b")

    assert local["callable"] is True


async def test_openapi_schema_is_served(client):
    response = await client.get("/openapi.json")

    assert response.status_code == 200
    schema = response.json()
    assert schema["info"]["title"] == "LLM Cost Autopilot"
    assert "/v1/models" in schema["paths"]
    assert "/health/providers" in schema["paths"]


async def test_provider_health_reports_every_configured_provider(client):
    response = await client.get("/health/providers")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert "ollama" in {p["provider"] for p in body["providers"]}
    assert all(p["state"] == "closed" and p["healthy"] for p in body["providers"])


async def test_idle_provider_reports_null_rates_not_a_perfect_score(client):
    body = (await client.get("/health/providers")).json()
    entry = next(p for p in body["providers"] if p["provider"] == "ollama")

    assert entry["metrics"]["total_calls"] == 0
    assert entry["metrics"]["success_rate"] is None
    assert entry["metrics"]["p95_latency_ms"] is None


async def test_provider_health_stays_200_when_a_circuit_is_open(client, app, settings):
    health = app.state.gateway.health
    provider = next(iter(health.providers))
    for _ in range(settings.routing.breaker_failure_threshold):
        health.record_failure(
            provider,
            ProviderServerError("boom", provider=provider.value, model_id="x"),
        )

    response = await client.get("/health/providers")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "degraded"
    entry = next(p for p in body["providers"] if p["provider"] == provider.value)
    assert entry["state"] == "open"
    assert entry["healthy"] is False


async def test_admin_playground_supports_get_requests(client):
    response = await client.get("/admin/playground")

    assert response.status_code == 200
    body = response.json()
    assert body["request_id"]
    assert body["routing"]["selected_model"]


async def test_admin_playground_executes_the_real_routing_engine(client):
    response = await client.post(
        "/admin/playground/execute",
        json={"prompt": "Summarize the latest quarterly performance.", "task_type": "summarization"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "completed"
    assert body["response"]["text"]
    assert body["routing"]["selected_model"]
