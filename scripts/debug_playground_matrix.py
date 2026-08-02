import asyncio

from asgi_lifespan import LifespanManager
from httpx import ASGITransport, AsyncClient
from pydantic import SecretStr

from autopilot.api.main import create_app
from autopilot.config import AppSettings, ProviderSettings


async def try_config(name, prov):
    settings = AppSettings(environment="test", log_json=False)
    settings.providers = prov
    app = create_app(settings)
    try:
        async with LifespanManager(app):
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as c:
                resp = await c.post("/admin/playground/execute", json={"prompt":"hello","task_type":"summarization"})
                print(name, 'STATUS', resp.status_code, resp.json())
    except Exception as e:
        print(name, 'ERROR', type(e).__name__, e)

async def main():
    matrix = [
        ('ollama', ProviderSettings(ollama_base_url="http://localhost:11434")),
        ('openai', ProviderSettings(openai_api_key=SecretStr('sk-test'))),
        ('anthropic', ProviderSettings(anthropic_api_key=SecretStr('anthropic'))),
        ('mixed', ProviderSettings(openai_api_key=SecretStr('sk-test'), anthropic_api_key=SecretStr('anthropic'))),
        ('none', ProviderSettings(openai_api_key=None, anthropic_api_key=None, ollama_base_url=None)),
    ]
    for name, prov in matrix:
        await try_config(name, prov)

if __name__ == '__main__':
    asyncio.run(main())
