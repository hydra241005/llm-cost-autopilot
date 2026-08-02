"""Health endpoints.

``/health`` drives container and platform liveness probes identically across
every deployment target, so no platform needs a bespoke probe configuration.
``/health/providers`` exposes per-provider circuit state and rolling call
metrics, which is what an operator actually needs when routing degrades.
"""

from __future__ import annotations

from fastapi import APIRouter, Request, Response, status
from pydantic import BaseModel, Field

from autopilot import __version__
from autopilot.infrastructure.observability.metrics import ProviderMetrics
from autopilot.infrastructure.resilience.health import ProviderHealth
from autopilot.infrastructure.providers.ollama_setup import check_ollama
from autopilot.domain.enums import Provider

router = APIRouter(tags=["ops"])


class HealthResponse(BaseModel):
    """Liveness payload."""

    status: str = Field(description="Always 'ok' when the process can serve traffic.")
    version: str = Field(description="Application version.")
    environment: str = Field(description="Deployment environment name.")
    providers: list[str] = Field(description="Providers with usable credentials.")
    model_count: int = Field(description="Models loaded in the registry.")
    setup_warning: str | None = Field(
        default=None,
        description="Actionable setup guidance when a provider is reachable but unusable.",
    )


@router.get("/health", response_model=HealthResponse, summary="Liveness probe")
async def health(request: Request) -> HealthResponse:
    """Report process health and what the router currently has available."""
    app = request.app
    ollama_status = getattr(app.state, "ollama_status", None)
    # If Ollama was previously reported missing models, try a one-time refresh
    # so the health endpoint reflects models pulled after startup.
    if ollama_status is not None and ollama_status.missing:
        try:
            registry = getattr(app.state, "registry", None)
            gateway = getattr(app.state, "gateway", None)
            if gateway is not None and registry is not None:
                adapter = gateway.adapter_for(Provider.OLLAMA)
                # Re-run the lightweight preflight (no auto-pull) to refresh state.
                new_status = await check_ollama(adapter, registry, auto_pull=False)
                app.state.ollama_status = new_status
                ollama_status = new_status
        except Exception:
            # Don't let a transient refresh error break the liveness check.
            pass
    return HealthResponse(
        status="ok",
        version=__version__,
        environment=app.state.settings.environment,
        providers=sorted(p.value for p in app.state.gateway.available_providers),
        model_count=len(app.state.registry.all()),
        setup_warning=ollama_status.guidance() if ollama_status is not None else None,
    )


@router.get("/health/ready", response_model=HealthResponse, summary="Readiness probe")
async def readiness(request: Request, response: Response) -> HealthResponse:
    """Readiness: ensures the process and key dependencies can serve traffic.

    This endpoint performs lightweight checks to determine whether the
    application is ready to receive production traffic. It verifies:
    - app composition (registry/gateway) has completed
    - database connectivity (when a non-sqlite URL is configured)
    - redis connectivity (when `REDIS_URL` is configured)
    - provider manager is present on the gateway

    Returns HTTP 200 when ready, HTTP 503 when any critical dependency is
    unavailable. The JSON payload retains the same shape as `/health`.
    """
    app = request.app
    settings = app.state.settings

    registry = getattr(app.state, "registry", None)
    gateway = getattr(app.state, "gateway", None)

    ready = True
    setup_warning = None

    # Basic composition checks
    if registry is None or gateway is None:
        ready = False

    # Provider manager availability
    try:
        has_health = gateway is not None and getattr(gateway, "health", None) is not None
        if not has_health:
            ready = False
    except Exception:
        ready = False

    # Database connectivity: SQLite is considered local and available by default.
    db_url = getattr(settings, "database_url", "")
    if db_url and not db_url.startswith("sqlite"):
        try:
            # Import SQLAlchemy only when needed; optional dependency may be absent.
            from sqlalchemy.ext.asyncio import create_async_engine
            from sqlalchemy import text

            engine = create_async_engine(db_url, pool_pre_ping=True)
            try:
                async with engine.connect() as conn:
                    await conn.execute(text("SELECT 1"))
            finally:
                await engine.dispose()
        except Exception as e:  # pragma: no cover - runtime external checks
            ready = False
            setup_warning = str(e)

    # Redis connectivity (optional)
    redis_url = getattr(settings, "redis_url", None)
    if redis_url:
        try:
            import redis.asyncio as redis_async

            client = redis_async.from_url(redis_url)
            try:
                await client.ping()
            finally:
                await client.close()
        except Exception as e:  # pragma: no cover - runtime external checks
            ready = False
            setup_warning = (setup_warning or "") + f"; redis: {e}"

    response.status_code = status.HTTP_200_OK if ready else status.HTTP_503_SERVICE_UNAVAILABLE

    return HealthResponse(
        status="ok" if ready else "degraded",
        version=__version__,
        environment=app.state.settings.environment,
        providers=sorted(p.value for p in app.state.gateway.available_providers)
        if gateway is not None
        else [],
        model_count=len(registry.all()) if registry is not None else 0,
        setup_warning=setup_warning,
    )


class ProviderMetricsResponse(BaseModel):
    """Rolling call statistics for one provider."""

    window_s: float = Field(description="How far back these figures reach, in seconds.")
    total_calls: int = Field(description="Calls observed in the window.")
    successes: int
    failures: int
    timeouts: int = Field(description="Calls that exceeded their timeout budget.")
    rate_limited: int = Field(description="Calls rejected with HTTP 429.")
    server_errors: int = Field(description="Retryable upstream failures (5xx, connection).")
    bad_requests: int = Field(description="Non-retryable failures caused by the request.")
    circuit_open_rejections: int = Field(
        description="Calls short-circuited locally without reaching the vendor."
    )
    success_rate: float | None = Field(
        description="Successes over calls that reached the vendor. Null when idle."
    )
    availability: float | None = Field(
        description="Share of calls not rejected by an open circuit. Null when idle."
    )
    p50_latency_ms: int | None
    p95_latency_ms: int | None

    @classmethod
    def from_metrics(cls, metrics: ProviderMetrics) -> ProviderMetricsResponse:
        """Build the response model from a collector snapshot."""
        return cls(
            window_s=metrics.window_s,
            total_calls=metrics.total_calls,
            successes=metrics.successes,
            failures=metrics.failures,
            timeouts=metrics.timeouts,
            rate_limited=metrics.rate_limited,
            server_errors=metrics.server_errors,
            bad_requests=metrics.bad_requests,
            circuit_open_rejections=metrics.circuit_open_rejections,
            success_rate=metrics.success_rate,
            availability=metrics.availability,
            p50_latency_ms=metrics.p50_latency_ms,
            p95_latency_ms=metrics.p95_latency_ms,
        )


class ProviderHealthResponse(BaseModel):
    """Circuit state and metrics for one provider."""

    provider: str
    state: str = Field(description="Circuit state: closed, open, or half_open.")
    healthy: bool = Field(description="Whether calls are currently admitted.")
    failure_count: int = Field(description="Failures inside the breaker's rolling window.")
    metrics: ProviderMetricsResponse

    @classmethod
    def from_health(cls, health: ProviderHealth) -> ProviderHealthResponse:
        """Build the response model from a health-manager view."""
        return cls(
            provider=health.provider.value,
            state=health.state.value,
            healthy=health.healthy,
            failure_count=health.failure_count,
            metrics=ProviderMetricsResponse.from_metrics(health.metrics),
        )


class ProviderHealthListResponse(BaseModel):
    """Envelope for provider health."""

    status: str = Field(
        description="'ok' when every provider is admitting calls, 'degraded' otherwise."
    )
    providers: list[ProviderHealthResponse]


@router.get(
    "/health/providers",
    response_model=ProviderHealthListResponse,
    summary="Per-provider circuit state and metrics",
)
async def provider_health(request: Request) -> ProviderHealthListResponse:
    """Report circuit state and rolling call metrics for every provider.

    Always returns 200, including when providers are degraded: this endpoint
    describes upstream health, and a non-200 would make an orchestrator restart
    a perfectly healthy process because a vendor was down.
    """
    entries = request.app.state.gateway.health.health_all()
    return ProviderHealthListResponse(
        status="ok" if all(e.healthy for e in entries) else "degraded",
        providers=[ProviderHealthResponse.from_health(e) for e in entries],
    )
