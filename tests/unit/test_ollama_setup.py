"""Ollama setup preflight tests."""

from __future__ import annotations

import httpx
import pytest
import respx

from autopilot.domain.enums import Provider, Tier
from autopilot.infrastructure.providers.ollama_adapter import OllamaAdapter
from autopilot.infrastructure.providers.ollama_setup import check_ollama
from autopilot.infrastructure.providers.registry import YamlModelRegistry
from tests.conftest import make_model

BASE_URL = "http://ollama.test:11434"
TAGS_URL = f"{BASE_URL}/api/tags"
PULL_URL = f"{BASE_URL}/api/pull"


@pytest.fixture
def adapter() -> OllamaAdapter:
    return OllamaAdapter(base_url=BASE_URL)


@pytest.fixture
def registry() -> YamlModelRegistry:
    return YamlModelRegistry(
        [
            make_model(
                "ollama:llama3.1-8b",
                provider=Provider.OLLAMA,
                tier=Tier.SIMPLE,
                input_cost="0",
                output_cost="0",
            ),
            make_model("openai:gpt-4o-mini", tier=Tier.MODERATE),
        ]
    )


def _tags(*names: str) -> dict:
    return {"models": [{"name": n} for n in names]}


@respx.mock
async def test_ready_when_the_required_model_is_installed(adapter, registry):
    respx.get(TAGS_URL).mock(return_value=httpx.Response(200, json=_tags("llama3.1-8b:latest")))

    status = await check_ollama(adapter, registry)

    assert status.usable is True
    assert status.guidance() is None


@respx.mock
async def test_missing_model_names_the_exact_pull_command(adapter, registry):
    respx.get(TAGS_URL).mock(return_value=httpx.Response(200, json=_tags("mistral:7b")))

    status = await check_ollama(adapter, registry)

    assert status.missing == ("llama3.1-8b",)
    guidance = status.guidance()
    assert "ollama pull llama3.1-8b" in guidance
    assert "OLLAMA_AUTO_PULL" in guidance


@respx.mock
async def test_unreachable_server_reports_install_guidance(adapter, registry):
    respx.get(TAGS_URL).mock(side_effect=httpx.ConnectError("refused"))

    status = await check_ollama(adapter, registry)

    assert status.reachable is False
    assert status.usable is False
    assert "ollama.com/download" in status.guidance()


@respx.mock
async def test_auto_pull_downloads_missing_models(adapter, registry):
    tags = respx.get(TAGS_URL).mock(
        side_effect=[
            httpx.Response(200, json=_tags()),
            httpx.Response(200, json=_tags("llama3.1-8b:latest")),
        ]
    )
    pull = respx.post(PULL_URL).mock(return_value=httpx.Response(200, json={"status": "success"}))

    status = await check_ollama(adapter, registry, auto_pull=True)

    assert pull.called
    assert tags.call_count == 2
    assert status.pulled == ("llama3.1-8b",)
    assert status.usable is True


@respx.mock
async def test_a_failed_pull_degrades_to_guidance(adapter, registry):
    respx.get(TAGS_URL).mock(return_value=httpx.Response(200, json=_tags()))
    respx.post(PULL_URL).mock(return_value=httpx.Response(500, json={"error": "disk full"}))

    status = await check_ollama(adapter, registry, auto_pull=True)

    assert status.pulled == ()
    assert status.missing == ("llama3.1-8b",)


@respx.mock
async def test_a_registry_without_local_models_needs_no_check(adapter):
    registry = YamlModelRegistry([make_model("openai:gpt-4o-mini", tier=Tier.MODERATE)])

    status = await check_ollama(adapter, registry)

    assert status.usable is True
    assert not respx.calls


@respx.mock
async def test_installed_models_include_bare_stems(adapter):
    respx.get(TAGS_URL).mock(return_value=httpx.Response(200, json=_tags("llama3.1:8b")))

    installed = await adapter.installed_models()

    assert {"llama3.1:8b", "llama3.1"} <= installed


@respx.mock
async def test_installed_models_is_none_when_unreachable(adapter):
    respx.get(TAGS_URL).mock(side_effect=httpx.ConnectError("refused"))

    assert await adapter.installed_models() is None
