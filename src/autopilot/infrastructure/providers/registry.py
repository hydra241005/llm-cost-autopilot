"""YAML-backed model registry.

Loads ``configs/models.yaml`` into validated :class:`ModelConfig` objects and
serves the lookups the router needs. In-memory and immutable: a config change
builds a new registry and swaps it atomically, so in-flight requests always see
a consistent catalogue.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from decimal import Decimal
from pathlib import Path
from typing import Any

import yaml

from autopilot.domain.entities import ModelConfig, Usage
from autopilot.domain.enums import Tier
from autopilot.domain.errors import ConfigurationError, ModelNotFoundError, NoCapableModelError
from autopilot.domain.interfaces import ModelRegistry
from autopilot.domain.policies.pricing import compute_cost

#: Representative workload used to rank models by price. A fixed synthetic mix is
#: used rather than input price alone so a model with cheap input and expensive
#: output cannot masquerade as the cheapest option.
_RANKING_USAGE = Usage(input_tokens=1_000, output_tokens=500)


class YamlModelRegistry(ModelRegistry):
    """A :class:`~autopilot.domain.interfaces.ModelRegistry` backed by a YAML document."""

    def __init__(self, models: Iterable[ModelConfig], *, version: int = 1) -> None:
        """Build a registry from already-validated model configs.

        Args:
            models: The catalogue entries.
            version: Config document version, recorded on cache keys and decisions.

        Raises:
            ConfigurationError: The catalogue is empty or contains duplicate ids.
        """
        entries = list(models)
        if not entries:
            raise ConfigurationError("Model registry is empty; at least one model is required.")

        by_id: dict[str, ModelConfig] = {}
        for model in entries:
            if model.id in by_id:
                raise ConfigurationError(f"Duplicate model id in registry: {model.id!r}")
            by_id[model.id] = model

        self._by_id = by_id
        self._version = version
        self._by_tier: dict[Tier, tuple[ModelConfig, ...]] = {
            tier: tuple(
                sorted(
                    (m for m in entries if m.tier is tier and m.active),
                    key=lambda m: (compute_cost(_RANKING_USAGE, m), m.expected_latency_ms),
                )
            )
            for tier in Tier
        }

    @classmethod
    def from_yaml(cls, path: Path) -> YamlModelRegistry:
        """Load and validate a registry from a YAML file.

        Args:
            path: Path to a ``models.yaml`` document.

        Returns:
            The populated registry.

        Raises:
            ConfigurationError: The file is missing, unparseable, or fails validation.
        """
        try:
            raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise ConfigurationError(f"Model config not found at {path}") from exc
        except yaml.YAMLError as exc:
            raise ConfigurationError(f"Model config at {path} is not valid YAML: {exc}") from exc

        return cls.from_dict(raw, source=str(path))

    @classmethod
    def from_dict(cls, raw: Any, *, source: str = "<dict>") -> YamlModelRegistry:
        """Build a registry from an already-parsed document.

        Args:
            raw: The parsed document; must be a mapping with a ``models`` list.
            source: Human-readable origin, used in error messages.

        Returns:
            The populated registry.

        Raises:
            ConfigurationError: The document shape or any entry is invalid.
        """
        if not isinstance(raw, dict):
            raise ConfigurationError(f"Model config at {source} must be a mapping.")

        entries = raw.get("models")
        if not isinstance(entries, list):
            raise ConfigurationError(f"Model config at {source} must contain a 'models' list.")

        models: list[ModelConfig] = []
        for index, entry in enumerate(entries):
            if not isinstance(entry, dict):
                raise ConfigurationError(f"Model entry #{index} in {source} must be a mapping.")
            models.append(cls._parse_entry(entry, index=index, source=source))

        version = raw.get("version", 1)
        if not isinstance(version, int) or version < 1:
            raise ConfigurationError(
                f"Model config version in {source} must be a positive integer."
            )

        return cls(models, version=version)

    @staticmethod
    def _parse_entry(entry: dict[str, Any], *, index: int, source: str) -> ModelConfig:
        """Validate one raw entry into a :class:`ModelConfig`."""
        payload = dict(entry)
        # Prices are quoted as strings in YAML so Decimal never sees a float.
        for field in ("input_cost_per_mtok", "output_cost_per_mtok"):
            if field in payload:
                payload[field] = Decimal(str(payload[field]))
        try:
            return ModelConfig.model_validate(payload)
        except Exception as exc:
            model_id = entry.get("id", f"#{index}")
            raise ConfigurationError(
                f"Invalid model entry {model_id!r} in {source}: {exc}"
            ) from exc

    @property
    def version(self) -> int:
        """Version of the loaded config document."""
        return self._version

    @property
    def map_version(self) -> str:
        """Cache-key component that changes whenever the catalogue changes."""
        return f"models-v{self._version}"

    def get(self, model_id: str) -> ModelConfig:
        """Return the config for ``model_id``.

        Raises:
            ModelNotFoundError: No such model is registered.
        """
        try:
            return self._by_id[model_id]
        except KeyError:
            raise ModelNotFoundError(model_id) from None

    def by_tier(self, tier: Tier) -> Sequence[ModelConfig]:
        """Return active models for ``tier``, cheapest first."""
        return self._by_tier[tier]

    def cheapest(self, tier: Tier) -> ModelConfig:
        """Return the cheapest active model for ``tier``.

        Raises:
            NoCapableModelError: The tier has no active models.
        """
        candidates = self._by_tier[tier]
        if not candidates:
            raise NoCapableModelError(f"No active model registered for tier {int(tier)}.")
        return candidates[0]

    def all(self) -> Sequence[ModelConfig]:
        """Return every registered model, active or not, in declaration order."""
        return tuple(self._by_id.values())

    def fits_context(self, model: ModelConfig, prompt_tokens: int, max_tokens: int) -> bool:
        """Return whether ``model`` can hold the prompt plus the requested completion.

        Args:
            model: Candidate model.
            prompt_tokens: Estimated prompt size.
            max_tokens: Completion tokens the caller asked for.

        Returns:
            ``True`` when the model's context window and output cap both suffice.
        """
        return (
            prompt_tokens + max_tokens <= model.max_context_tokens
            and max_tokens <= model.max_output_tokens
        )
