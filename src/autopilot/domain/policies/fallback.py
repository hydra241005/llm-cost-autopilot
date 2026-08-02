"""Routing configuration and the ordered candidate chain it produces.

A tier does not map to *a* model; it maps to a primary model plus an ordered
fallback chain that deliberately crosses providers, so a full outage at one
vendor degrades transparently instead of failing the request.

This module is pure data and ordering. Deciding whether a candidate is
*usable* belongs to the policies in :mod:`autopilot.domain.policies.routing`.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from autopilot.domain.enums import Tier
from autopilot.domain.errors import ConfigurationError


class TierRoute(BaseModel):
    """The primary model for one tier and its ordered fallbacks."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    primary: str = Field(min_length=1)
    fallbacks: tuple[str, ...] = ()

    @property
    def chain(self) -> tuple[str, ...]:
        """Primary first, then fallbacks, with duplicates removed."""
        seen: dict[str, None] = {}
        for model_id in (self.primary, *self.fallbacks):
            seen.setdefault(model_id, None)
        return tuple(seen)


class RoutingConfig(BaseModel):
    """A versioned tier-to-model map.

    Seeded from ``configs/routing.yaml`` and versioned in the database
    thereafter; every routing decision records the version that produced it.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    version: int = Field(default=1, ge=1)
    name: str = "default"
    baseline_model_id: str = Field(min_length=1)
    confidence_threshold: float = Field(default=0.6, ge=0.0, le=1.0)
    tiers: dict[Tier, TierRoute]

    @field_validator("tiers")
    @classmethod
    def _require_every_tier(cls, value: dict[Tier, TierRoute]) -> dict[Tier, TierRoute]:
        """Reject a config that leaves any tier unroutable."""
        missing = [int(t) for t in Tier if t not in value]
        if missing:
            raise ValueError(f"routing config is missing tier(s): {missing}")
        return value

    @classmethod
    def from_dict(cls, raw: Any, *, source: str = "<dict>") -> RoutingConfig:
        """Build a config from an already-parsed document.

        Args:
            raw: Parsed mapping, as produced by a YAML or JSON loader.
            source: Human-readable origin, used in error messages.

        Returns:
            The validated config.

        Raises:
            ConfigurationError: The document shape or any value is invalid.
        """
        if not isinstance(raw, dict):
            raise ConfigurationError(f"Routing config at {source} must be a mapping.")
        try:
            return cls.model_validate(raw)
        except Exception as exc:
            raise ConfigurationError(f"Invalid routing config at {source}: {exc}") from exc

    def chain_for(self, tier: Tier) -> tuple[str, ...]:
        """Return the ordered candidate model ids for ``tier``."""
        return self.tiers[tier].chain

    def escalation_chain(self, tier: Tier) -> tuple[str, ...]:
        """Return candidates for ``tier`` followed by those of every higher tier.

        Exhausting a tier is not a reason to fail while a more capable model is
        still healthy: paying more is strictly better than returning a 503.
        Duplicates are dropped so a model shared across tiers is tried once.
        """
        seen: dict[str, None] = {}
        for candidate_tier in Tier:
            if candidate_tier < tier:
                continue
            for model_id in self.chain_for(candidate_tier):
                seen.setdefault(model_id, None)
        return tuple(seen)
