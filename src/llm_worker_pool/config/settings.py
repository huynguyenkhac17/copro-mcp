"""Loads config: default.yaml (+ env override) and budgets.yaml.

AppConfig is loaded ONCE at server startup (server.py), then passed
down as a constructor parameter (dependency injection) - do not use
module-level global variables for this state.
"""

from __future__ import annotations

import os
from pathlib import Path

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)

from llm_worker_pool.domain.exceptions import ConfigurationError
from llm_worker_pool.domain.models import BudgetProfile, GenerationOptions

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_CONFIG_PATH = _REPO_ROOT / "config" / "default.yaml"
_DEFAULT_BUDGETS_PATH = _REPO_ROOT / "config" / "budgets.yaml"

CONFIG_PATH_ENV_VAR = "LLM_WORKER_POOL_CONFIG"
BUDGETS_PATH_ENV_VAR = "LLM_WORKER_POOL_BUDGETS_CONFIG"


def _resolve_path(env_var: str, default: Path) -> Path:
    override = os.environ.get(env_var)
    return Path(override) if override else default


class ProviderInstanceConfig(BaseModel):
    type: str
    endpoint: str
    model: str
    api_key_env: str | None = None
    api_key_envs: list[str] | None = None
    """Multiple env var names holding API keys, round-robined per request
    (see providers/openai_compatible_provider.py). Combined with the legacy
    single-key api_key_env if both are set."""
    concurrency: int | None = None
    """Max parallel generate() calls for this provider instance. None = use
    the type-based default (see providers/registry.py)."""


class RetryConfig(BaseModel):
    max_attempts: int = 3
    backoff_base_seconds: float = 0.5


class LoggingConfig(BaseModel):
    level: str = "INFO"
    debug_log_content: bool = False


class FileAccessConfig(BaseModel):
    """Allows delegate_task to read files directly via the file_paths parameter instead
    of Claude pasting content. Default allowed_roots is empty = feature DISABLED
    (fail-closed) - must be explicitly configured to read any files.
    """

    allowed_roots: list[str] = Field(default_factory=list)
    max_file_bytes: int = 200_000
    max_total_bytes: int = 800_000


class CacheConfig(BaseModel):
    """Caches provider responses by (final prompt + options), avoiding repeated
    inference/GPU costs for the exact same question."""

    enabled: bool = True
    max_entries: int = 256
    ttl_seconds: float | None = 3600


class JobsConfig(BaseModel):
    ttl_seconds: float = 3600
    max_entries: int = 1000


class RagConfig(BaseModel):
    enabled: bool = False
    embedding_model: str = "nomic-embed-text"
    chunk_size_tokens: int = 256
    overlap_tokens: int = 32
    top_k: int = 10


class AppConfig(BaseSettings):
    """Root configuration. Priority sources (high -> low):
    init kwargs > real environment variables (LLM_WORKER_POOL_..., nested with
    '__') > .env file in repo root > default.yaml.
    """

    model_config = SettingsConfigDict(
        env_prefix="LLM_WORKER_POOL_",
        env_nested_delimiter="__",
        env_file=_REPO_ROOT / ".env",
        yaml_file=_resolve_path(CONFIG_PATH_ENV_VAR, _DEFAULT_CONFIG_PATH),
        extra="ignore",
    )

    providers: dict[str, ProviderInstanceConfig] = Field(default_factory=dict)
    default_provider: str = "ollama"
    timeout_seconds: float = 60.0
    retry: RetryConfig = Field(default_factory=RetryConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    file_access: FileAccessConfig = Field(default_factory=FileAccessConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    jobs: JobsConfig = Field(default_factory=JobsConfig)
    rag: RagConfig = Field(default_factory=RagConfig)

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            YamlConfigSettingsSource(settings_cls),
            file_secret_settings,
        )


def load_budget_profiles(path: Path | None = None) -> dict[str, BudgetProfile]:
    """Reads budgets.yaml and returns {budget_name: BudgetProfile}."""

    resolved = path or _resolve_path(BUDGETS_PATH_ENV_VAR, _DEFAULT_BUDGETS_PATH)
    if not resolved.exists():
        raise ConfigurationError(f"Budgets config file not found: {resolved}")

    with open(resolved, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    budgets_raw = raw.get("budgets") or {}
    if not budgets_raw:
        raise ConfigurationError(f"Budgets config file empty or malformed: {resolved}")

    profiles: dict[str, BudgetProfile] = {}
    for name, body in budgets_raw.items():
        try:
            options = GenerationOptions(**(body.get("options") or {}))
            profiles[name] = BudgetProfile(
                name=name,
                description=body.get("description", ""),
                options=options,
                timeout_seconds=body.get("timeout_seconds"),
            )
        except Exception as exc:  # noqa: BLE001 - wrap in domain error with context
            raise ConfigurationError(f"Budget '{name}' is misconfigured: {exc}") from exc

    return profiles
