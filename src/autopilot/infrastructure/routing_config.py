"""YAML-backed routing configuration loader.

Mirrors the model registry: parse once at startup into a validated, immutable
:class:`RoutingConfig`. The domain type does the validating; this module only
knows how to read a file.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from autopilot.domain.errors import ConfigurationError
from autopilot.domain.policies.fallback import RoutingConfig


def load_routing_config(path: Path) -> RoutingConfig:
    """Load and validate a routing config from a YAML file.

    Args:
        path: Path to a ``routing.yaml`` document.

    Returns:
        The validated config.

    Raises:
        ConfigurationError: The file is missing, unparseable, or invalid.
    """
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigurationError(f"Routing config not found at {path}") from exc
    except yaml.YAMLError as exc:
        raise ConfigurationError(f"Routing config at {path} is not valid YAML: {exc}") from exc

    return RoutingConfig.from_dict(raw, source=str(path))
