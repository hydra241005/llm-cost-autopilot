from asgi_lifespan import LifespanManager
from httpx import AsyncClient, ASGITransport
import asyncio

from autopilot.api.main import create_app
from autopilot.config import AppSettings

async def main():
    settings = AppSettings(environment="test", log_json=False)
    app = create_app(settings)
    async with LifespanManager(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            resp = await c.post(
                "/admin/playground/execute",
                json={"prompt": "Summarize the latest quarterly performance.", "task_type": "summarization"},
            )
            print("STATUS:", resp.status_code)
            print("BODY:\n", resp.text)

if __name__ == "__main__":
    asyncio.run(main())
