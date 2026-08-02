from __future__ import annotations

from types import SimpleNamespace

import pytest
from asgi_lifespan import LifespanManager
from httpx import ASGITransport, AsyncClient
from pydantic import SecretStr

from autopilot.api.main import create_app
from autopilot.config import AppSettings, ProviderSettings
from autopilot.domain.errors import ConfigurationError


@pytest.mark.parametrize(
    "providers_cfg, expect_success",
    [
        ("ollama", True),
        ("openai", True),
        ("anthropic", True),
        ("mixed", True),
        ("none", False),
    ],
)
async def test_playground_execute_with_various_provider_combinations(providers_cfg, expect_success):
    """Ensure the playground exercise path works for different configured providers.

    The model registry contains models across providers (see configs/models.yaml).
    We construct AppSettings with different provider credentials so the
    application composes adapters accordingly and the routing engine only
    considers configured providers.
    """
    # Build minimal provider settings for each scenario.
    if providers_cfg == "ollama":
        prov = ProviderSettings(ollama_base_url="http://localhost:11434")
    elif providers_cfg == "openai":
        prov = ProviderSettings(openai_api_key=SecretStr("sk-test"))
    elif providers_cfg == "anthropic":
        prov = ProviderSettings(anthropic_api_key=SecretStr("anthropic-test"))
    elif providers_cfg == "mixed":
        prov = ProviderSettings(openai_api_key=SecretStr("sk-test"), anthropic_api_key=SecretStr("anthropic-test"))
    else:
        # Construct a lightweight providers object with no configured providers.
        prov = SimpleNamespace(
            openai_api_key=None,
            openai_base_url=None,
            anthropic_api_key=None,
            anthropic_base_url=None,
            ollama_base_url=None,
            ollama_auto_pull=False,
        )

    settings = AppSettings(environment="test", log_json=False)
    # Inject our provider settings override.
    settings.providers = prov

    app = create_app(settings)

    if not expect_success:
        # Expect the app startup to fail when no providers are configured.
        with pytest.raises(ConfigurationError):
            async with LifespanManager(app):
                pass
        return

    async with LifespanManager(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/admin/playground/execute",
                json={"prompt": "Summarize the latest quarterly performance.", "task_type": "summarization"},
            )
            assert resp.status_code == 200
            body = resp.json()
            assert body["status"] == "completed"
            assert body["response"]["text"] is not None and body["response"]["text"] != ""
