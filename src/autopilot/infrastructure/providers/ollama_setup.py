"""Startup preflight for the local Ollama tier.

Ollama being *reachable* is not the same as Ollama being *usable*: a fresh
install answers every health check while having zero models pulled, so the first
real request fails with a vendor 404 that says nothing about how to fix it. This
module closes that gap by checking installed models at startup and naming the
exact command to run — or pulling them, when the operator has opted in.
"""

from __future__ import annotations

from dataclasses import dataclass

from autopilot.domain.enums import Provider
from autopilot.domain.errors import ProviderError
from autopilot.domain.interfaces import ModelRegistry
from autopilot.infrastructure.observability.logging import get_logger
from autopilot.infrastructure.providers.ollama_adapter import OllamaAdapter

_log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class OllamaStatus:
    """Result of the local-tier preflight.

    Attributes:
        reachable: Whether the Ollama server answered at all.
        installed: Vendor model names already pulled.
        missing: Registered Ollama models that are not yet pulled.
        pulled: Models downloaded during this preflight.
    """

    reachable: bool
    installed: frozenset[str]
    missing: tuple[str, ...]
    pulled: tuple[str, ...]

    @property
    def usable(self) -> bool:
        """Whether every registered local model can now be called."""
        return self.reachable and not self.missing

    def guidance(self) -> str | None:
        """Return an actionable message, or ``None`` when nothing needs doing."""
        if not self.reachable:
            return (
                "Ollama is not reachable. Install it from https://ollama.com/download, "
                "start it with `ollama serve`, then pull a model. The cloud providers "
                "still work without it; only the free local tier is unavailable."
            )
        if self.missing:
            commands = "\n".join(f"  ollama pull {name}" for name in self.missing)
            return (
                "Ollama is running but these registered models are not pulled:\n"
                f"{commands}\n"
                "Set OLLAMA_AUTO_PULL=true to download them automatically at startup."
            )
        return None


async def check_ollama(
    adapter: OllamaAdapter,
    registry: ModelRegistry,
    *,
    auto_pull: bool = False,
) -> OllamaStatus:
    """Verify the local tier is usable and report or repair what is missing.

    Args:
        adapter: The Ollama adapter to interrogate.
        registry: Catalogue whose Ollama entries must be available.
        auto_pull: Download missing models instead of only reporting them.

    Returns:
        What was found, and what was done about it. Never raises: a broken local
        tier degrades the router to its cloud providers rather than blocking
        startup entirely.
    """
    required = sorted(
        {m.vendor_model_id for m in registry.all() if m.provider is Provider.OLLAMA and m.active}
    )
    if not required:
        return OllamaStatus(reachable=True, installed=frozenset(), missing=(), pulled=())

    installed = await adapter.installed_models()
    if installed is None:
        status = OllamaStatus(
            reachable=False, installed=frozenset(), missing=tuple(required), pulled=()
        )
        _log.warning("ollama.unreachable", guidance=status.guidance())
        return status

    missing = [name for name in required if name not in installed]
    pulled: list[str] = []

    if missing and auto_pull:
        for name in list(missing):
            try:
                await adapter.pull(name)
            except ProviderError as exc:
                _log.warning("ollama.pull_failed", model=name, error=str(exc))
                continue
            pulled.append(name)
        installed = await adapter.installed_models() or frozenset()
        missing = [name for name in required if name not in installed]

    status = OllamaStatus(
        reachable=True,
        installed=installed,
        missing=tuple(missing),
        pulled=tuple(pulled),
    )
    if status.missing:
        _log.warning("ollama.models_missing", missing=list(status.missing))
    else:
        _log.info("ollama.ready", models=required, pulled=list(status.pulled))
    return status
