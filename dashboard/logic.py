from __future__ import annotations

from urllib.parse import urljoin


def build_api_url(base_url: str, path: str) -> str:
    """Build a fully qualified API URL from a base URL and a path."""
    base = base_url.rstrip("/")
    if not path.startswith("/"):
        path = f"/{path}"
    return urljoin(f"{base}/", path.lstrip("/"))


def summarize_provider_status(providers: list[dict[str, object]]) -> dict[str, int | str]:
    """Summarize provider health into a simple operator-friendly payload."""
    healthy = sum(1 for provider in providers if provider.get("healthy"))
    total = len(providers)
    return {
        "healthy_count": healthy,
        "degraded_count": total - healthy,
        "overall_status": "healthy" if healthy == total else "degraded",
    }
