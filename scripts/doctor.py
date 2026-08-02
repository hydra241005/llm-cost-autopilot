"""Environment preflight.

Answers "why isn't this working?" before the first request rather than after:
which providers are configured, whether Ollama is running, and which local
models still need pulling — with the exact command to fix each gap.

Usage::

    uv run python scripts/doctor.py
    uv run python scripts/doctor.py --pull
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from autopilot.config import AppSettings, get_settings
from autopilot.domain.enums import Provider
from autopilot.infrastructure.providers.factory import build_adapters
from autopilot.infrastructure.providers.ollama_adapter import OllamaAdapter
from autopilot.infrastructure.providers.ollama_setup import check_ollama
from autopilot.infrastructure.providers.registry import YamlModelRegistry

_OK = "[ok]"
_WARN = "[!!]"


async def diagnose(settings: AppSettings, *, pull: bool) -> int:
    """Report environment readiness.

    Args:
        settings: Application settings.
        pull: Download missing Ollama models instead of only reporting them.

    Returns:
        ``0`` when every configured provider is usable, ``1`` otherwise.
    """
    registry = YamlModelRegistry.from_yaml(settings.models_config_path)
    adapters = build_adapters(settings)

    print(f"Models in registry: {len(registry.all())}")
    for provider in Provider:
        mark = _OK if provider in adapters else "[--]"
        detail = "configured" if provider in adapters else "no credentials, skipped"
        print(f"{mark} {provider.value:<10} {detail}")

    adapter = adapters.get(Provider.OLLAMA)
    if not isinstance(adapter, OllamaAdapter):
        return 0

    try:
        status = await check_ollama(
            adapter, registry, auto_pull=pull or settings.providers.ollama_auto_pull
        )
    finally:
        for built in adapters.values():
            await built.aclose()

    if status.pulled:
        print(f"{_OK} pulled: {', '.join(status.pulled)}")
    guidance = status.guidance()
    if guidance is None:
        print(f"{_OK} ollama     local tier ready ({len(status.installed)} models installed)")
        return 0
    print(f"{_WARN} {guidance}")
    return 1


def main() -> int:
    """Parse arguments and run the preflight."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--pull",
        action="store_true",
        help="Download any missing Ollama models (several GB each).",
    )
    args = parser.parse_args()
    return asyncio.run(diagnose(get_settings(), pull=args.pull))


if __name__ == "__main__":
    sys.exit(main())
