"""Application configuration.

A single ``AppSettings`` object is built once at startup from the environment and
injected everywhere. No module outside this file reads ``os.environ`` — that
keeps configuration auditable and makes tests trivially overridable.

Secrets (provider API keys, database URLs) live only in environment variables and
are wrapped in :class:`~pydantic.SecretStr` so they cannot leak into a log line
or a traceback repr.
"""

from __future__ import annotations

from decimal import Decimal
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

#: Repository root, resolved from this file's location.
PROJECT_ROOT = Path(__file__).resolve().parents[2]


class ProviderSettings(BaseSettings):
    """Credentials and endpoints for upstream LLM vendors.

    All three providers are optional: the stack runs with only Ollama (free,
    local) so a reviewer can clone the repo and see it work without any keys.
    """

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    openai_api_key: SecretStr | None = Field(default=None, alias="OPENAI_API_KEY")
    openai_base_url: str | None = Field(default=None, alias="OPENAI_BASE_URL")
    anthropic_api_key: SecretStr | None = Field(default=None, alias="ANTHROPIC_API_KEY")
    anthropic_base_url: str | None = Field(default=None, alias="ANTHROPIC_BASE_URL")
    ollama_base_url: str = Field(default="http://localhost:11434", alias="OLLAMA_BASE_URL")
    ollama_auto_pull: bool = Field(
        default=False,
        alias="OLLAMA_AUTO_PULL",
        description=(
            "Download missing Ollama models at startup. Off by default: a pull is "
            "several gigabytes and should be an explicit choice."
        ),
    )

    def is_configured(self, provider: str) -> bool:
        """Return whether ``provider`` has the credentials it needs to be called."""
        match provider:
            case "openai":
                return self.openai_api_key is not None
            case "anthropic":
                return self.anthropic_api_key is not None
            case "ollama":
                return bool(self.ollama_base_url)
            case _:
                return False


class RoutingSettings(BaseSettings):
    """Tunables for the routing engine and its resilience layer."""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", env_prefix="ROUTING_", extra="ignore"
    )

    confidence_threshold: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="Below this confidence the effective tier is bumped by one.",
    )
    baseline_model_id: str = Field(
        default="openai:gpt-4o",
        description="Premium model used as the counterfactual savings baseline.",
    )
    timeout_tier1_s: float = Field(default=10.0, gt=0)
    timeout_tier2_s: float = Field(default=20.0, gt=0)
    timeout_tier3_s: float = Field(default=45.0, gt=0)
    connect_timeout_s: float = Field(default=3.0, gt=0)
    max_retries: int = Field(default=2, ge=0, le=5)
    retry_base_delay_s: float = Field(default=0.5, gt=0)
    retry_max_delay_s: float = Field(default=8.0, gt=0)
    breaker_failure_threshold: int = Field(default=5, ge=1)
    breaker_failure_rate: float = Field(default=0.5, ge=0.0, le=1.0)
    breaker_window_s: float = Field(default=30.0, gt=0)
    breaker_cooldown_s: float = Field(default=20.0, gt=0)
    metrics_window_s: float = Field(
        default=300.0,
        gt=0,
        description="Rolling window for per-provider health metrics, in seconds.",
    )

    def timeout_for_tier(self, tier: int) -> float:
        """Return the total request timeout budget for ``tier``."""
        return {1: self.timeout_tier1_s, 2: self.timeout_tier2_s, 3: self.timeout_tier3_s}[tier]


class VerificationSettings(BaseSettings):
    """Sampling and budget controls for asynchronous quality verification."""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", env_prefix="VERIFY_", extra="ignore"
    )

    enabled: bool = True
    base_sample_rate: float = Field(default=0.10, ge=0.0, le=1.0)
    daily_budget_usd: Decimal = Field(default=Decimal("5.00"), ge=0)
    judge_model_id: str = Field(default="anthropic:claude-sonnet-5")
    reference_model_id: str = Field(default="openai:gpt-4o")
    pass_score_threshold: float = Field(default=4.0, ge=0.0, le=5.0)
    pass_agreement_threshold: float = Field(default=0.7, ge=0.0, le=1.0)


class CacheSettings(BaseSettings):
    """Response-cache behaviour."""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", env_prefix="CACHE_", extra="ignore"
    )

    enabled: bool = True
    ttl_s: int = Field(default=3600, gt=0)
    max_temperature: float = Field(
        default=0.3,
        ge=0.0,
        description="Responses above this temperature are never cached.",
    )
    semantic_enabled: bool = Field(
        default=False, description="Semantic cache is off by default; see docs 06 §A5."
    )


class AppSettings(BaseSettings):
    """Root configuration object for the whole application."""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    environment: Literal["local", "test", "staging", "production"] = Field(
        default="local", alias="ENVIRONMENT"
    )
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        default="INFO", alias="LOG_LEVEL"
    )
    log_json: bool = Field(
        default=True,
        alias="LOG_JSON",
        description="JSON logs in every environment except an interactive local shell.",
    )
    database_url: str = Field(
        default="sqlite+aiosqlite:///./data/autopilot.db",
        alias="DATABASE_URL",
        description="SQLite by default; set a postgresql+asyncpg URL to switch dialects.",
    )
    redis_url: str | None = Field(
        default=None,
        alias="REDIS_URL",
        description="Optional. When unset, in-process cache/queue/rate-limit adapters are used.",
    )
    models_config_path: Path = Field(
        default=PROJECT_ROOT / "configs" / "models.yaml", alias="MODELS_CONFIG_PATH"
    )
    routing_config_path: Path = Field(
        default=PROJECT_ROOT / "configs" / "routing.yaml", alias="ROUTING_CONFIG_PATH"
    )
    artifacts_dir: Path = Field(default=PROJECT_ROOT / "artifacts", alias="ARTIFACTS_DIR")
    api_auth_enabled: bool = Field(default=False, alias="API_AUTH_ENABLED")
    default_rate_limit_rpm: int = Field(default=60, ge=1, alias="DEFAULT_RATE_LIMIT_RPM")
    max_request_bytes: int = Field(default=256_000, gt=0, alias="MAX_REQUEST_BYTES")

    providers: ProviderSettings = Field(default_factory=ProviderSettings)
    routing: RoutingSettings = Field(default_factory=RoutingSettings)
    verification: VerificationSettings = Field(default_factory=VerificationSettings)
    cache: CacheSettings = Field(default_factory=CacheSettings)

    @field_validator("models_config_path", "routing_config_path", "artifacts_dir")
    @classmethod
    def _resolve_path(cls, value: Path) -> Path:
        """Resolve config paths against the project root when given relatively."""
        return value if value.is_absolute() else (PROJECT_ROOT / value)

    @property
    def use_redis(self) -> bool:
        """Whether Redis-backed adapters should be wired instead of in-process ones."""
        return self.redis_url is not None

    @property
    def is_sqlite(self) -> bool:
        """Whether the configured database is SQLite."""
        return self.database_url.startswith("sqlite")


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    """Return the process-wide settings singleton.

    Cached so the environment is read exactly once. Tests clear the cache with
    ``get_settings.cache_clear()``.
    """
    return AppSettings()
