"""Model catalogue endpoint.

Exposes the registry so callers can see which models exist, what they cost, and
which are currently callable.
"""

from __future__ import annotations

from decimal import Decimal

from fastapi import APIRouter, Query, Request
from pydantic import BaseModel, Field

from autopilot.domain.entities import ModelConfig
from autopilot.domain.enums import Tier

router = APIRouter(prefix="/v1", tags=["models"])


class ModelSummary(BaseModel):
    """Public view of one registry entry."""

    id: str
    provider: str
    tier: int
    input_cost_per_mtok: Decimal
    output_cost_per_mtok: Decimal
    max_context_tokens: int
    max_output_tokens: int
    expected_latency_ms: int
    active: bool
    callable: bool = Field(description="Whether a configured adapter can reach this model.")

    @classmethod
    def from_config(cls, cfg: ModelConfig, *, is_callable: bool) -> ModelSummary:
        """Build a summary from a registry entry."""
        return cls(
            id=cfg.id,
            provider=cfg.provider.value,
            tier=int(cfg.tier),
            input_cost_per_mtok=cfg.input_cost_per_mtok,
            output_cost_per_mtok=cfg.output_cost_per_mtok,
            max_context_tokens=cfg.max_context_tokens,
            max_output_tokens=cfg.max_output_tokens,
            expected_latency_ms=cfg.expected_latency_ms,
            active=cfg.active,
            callable=is_callable,
        )


class ModelListResponse(BaseModel):
    """Envelope for the model catalogue."""

    data: list[ModelSummary]
    count: int


@router.get("/models", response_model=ModelListResponse, summary="List registered models")
async def list_models(
    request: Request,
    tier: Tier | None = Query(default=None, description="Filter to one complexity tier."),
) -> ModelListResponse:
    """Return the model catalogue, optionally filtered to a single tier."""
    registry = request.app.state.registry
    gateway = request.app.state.gateway

    entries = registry.by_tier(tier) if tier is not None else registry.all()
    data = [
        ModelSummary.from_config(cfg, is_callable=gateway.supports(cfg.id)) for cfg in entries
    ]
    return ModelListResponse(data=data, count=len(data))
