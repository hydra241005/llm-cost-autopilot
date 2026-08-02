"""FastAPI application factory.

The composition root: settings are read once, the registry and adapters are
built, and the resulting gateway is held on ``app.state`` for injection into
routes. Phase 1 exposes only health and model-catalogue endpoints; the routing
and completions surface arrives with the routing engine.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI

from autopilot import __version__
from autopilot.application.provider_gateway import ProviderGateway
from autopilot.config import AppSettings, get_settings
from autopilot.domain.enums import Provider
from autopilot.infrastructure.ml.model_store import FilesystemClassifierStore
from autopilot.infrastructure.observability.logging import configure_logging, get_logger
from autopilot.infrastructure.observability.operational_store import OperationalStore
from autopilot.infrastructure.observability.metrics import ProviderMetricsCollector
from autopilot.infrastructure.providers.factory import build_adapters
from autopilot.infrastructure.providers.ollama_adapter import OllamaAdapter
from autopilot.infrastructure.providers.ollama_setup import OllamaStatus, check_ollama
from autopilot.infrastructure.providers.registry import YamlModelRegistry
from autopilot.infrastructure.resilience.breaker import BreakerPolicy
from autopilot.infrastructure.resilience.health import ProviderHealthManager
from autopilot.infrastructure.resilience.retry import RetryPolicy

_log = get_logger(__name__)


def build_gateway(settings: AppSettings) -> ProviderGateway:
    """Construct the provider gateway from settings.

    Args:
        settings: Application settings.

    Returns:
        A gateway wired to the YAML registry and every configured adapter.
    """
    registry = YamlModelRegistry.from_yaml(settings.models_config_path)
    adapters = build_adapters(settings)
    routing = settings.routing
    health = ProviderHealthManager(
        adapters.keys(),
        breaker_policy=BreakerPolicy(
            failure_threshold=routing.breaker_failure_threshold,
            failure_rate=routing.breaker_failure_rate,
            window_s=routing.breaker_window_s,
            cooldown_s=routing.breaker_cooldown_s,
        ),
        metrics=ProviderMetricsCollector(window_s=routing.metrics_window_s),
    )
    return ProviderGateway(
        registry=registry,
        adapters=adapters,
        retry_policy=RetryPolicy(
            max_retries=routing.max_retries,
            base_delay_s=routing.retry_base_delay_s,
            max_delay_s=routing.retry_max_delay_s,
        ),
        timeouts_by_tier={
            1: routing.timeout_tier1_s,
            2: routing.timeout_tier2_s,
            3: routing.timeout_tier3_s,
        },
        baseline_model_id=routing.baseline_model_id,
        health=health,
    )


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Build shared resources on startup and release them on shutdown."""
    settings: AppSettings = app.state.settings
    registry = YamlModelRegistry.from_yaml(settings.models_config_path)
    gateway = build_gateway(settings)

    app.state.registry = registry
    app.state.gateway = gateway
    app.state.classifier_store = FilesystemClassifierStore(settings.artifacts_dir)
    app.state.operational_store = OperationalStore()
    app.state.ollama_status = await _preflight_ollama(settings, gateway, registry)
    _log.info(
        "app.started",
        environment=settings.environment,
        model_count=len(registry.all()),
        providers=sorted(p.value for p in gateway.available_providers),
    )
    try:
        yield
    finally:
        await gateway.aclose()
        _log.info("app.stopped")


async def _preflight_ollama(
    settings: AppSettings, gateway: ProviderGateway, registry: YamlModelRegistry
) -> OllamaStatus | None:
    """Check the local tier and surface actionable guidance at startup.

    Returns ``None`` when Ollama is not configured at all.
    """
    adapter = gateway.adapter_for(Provider.OLLAMA)
    if not isinstance(adapter, OllamaAdapter):
        return None
    status = await check_ollama(
        adapter, registry, auto_pull=settings.providers.ollama_auto_pull
    )
    guidance = status.guidance()
    if guidance is not None:
        _log.warning("ollama.setup_required", guidance=guidance)
    return status


def create_app(settings: AppSettings | None = None) -> FastAPI:
    """Create and configure the FastAPI application.

    Args:
        settings: Override settings, injected by tests.

    Returns:
        The configured application.
    """
    resolved = settings or get_settings()
    configure_logging(level=resolved.log_level, json_output=resolved.log_json)

    app = FastAPI(
        title="LLM Cost Autopilot",
        version=__version__,
        summary="Routes each LLM request to the cheapest model that can handle it.",
        description=(
            "An intelligent routing layer that classifies request complexity, routes to "
            "the cheapest capable model, verifies quality asynchronously on a sampled "
            "basis, and retrains its own router from verified failures."
        ),
        lifespan=lifespan,
    )
    app.state.settings = resolved

    from autopilot.api.routes import admin, classifier, health, models

    app.include_router(health.router)
    app.include_router(models.router)
    app.include_router(classifier.router)
    app.include_router(admin.router)
    return app


def get_app() -> FastAPI:
    """Return an application instance for ASGI servers.

    Used as the uvicorn factory target: ``uvicorn autopilot.api.main:get_app --factory``.
    """
    return create_app()


def __getattr__(name: str) -> Any:
    """Lazily expose a module-level ``app`` for tooling that expects one."""
    if name == "app":
        return create_app()
    raise AttributeError(name)
