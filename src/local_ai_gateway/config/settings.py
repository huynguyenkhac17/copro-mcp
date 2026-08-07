"""Load cau hinh: default.yaml (+ env override) va budgets.yaml.

AppConfig duoc load MOT LAN luc server khoi dong (server.py), sau do duoc
truyen xuong duoi dang tham so constructor (dependency injection) - khong
dung bien module-level global cho state nay.
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

from local_ai_gateway.domain.exceptions import ConfigurationError
from local_ai_gateway.domain.models import BudgetProfile, GenerationOptions

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_CONFIG_PATH = _REPO_ROOT / "config" / "default.yaml"
_DEFAULT_BUDGETS_PATH = _REPO_ROOT / "config" / "budgets.yaml"

CONFIG_PATH_ENV_VAR = "LOCAL_AI_GATEWAY_CONFIG"
BUDGETS_PATH_ENV_VAR = "LOCAL_AI_GATEWAY_BUDGETS_CONFIG"


def _resolve_path(env_var: str, default: Path) -> Path:
    override = os.environ.get(env_var)
    return Path(override) if override else default


class OllamaProviderConfig(BaseModel):
    endpoint: str = "http://localhost:11434"
    model: str = "qwen3.5:9b"


class RetryConfig(BaseModel):
    max_attempts: int = 3
    backoff_base_seconds: float = 0.5


class ConcurrencyConfig(BaseModel):
    max_parallel_requests: int = 1


class LoggingConfig(BaseModel):
    level: str = "INFO"
    debug_log_content: bool = False


class FileAccessConfig(BaseModel):
    """Cho phep call_local doc file truc tiep qua tham so file_paths thay vi
    Claude phai paste noi dung. Mac dinh allowed_roots rong = tinh nang TAT
    (fail-closed) - phai cau hinh ro thi moi doc duoc file nao.
    """

    allowed_roots: list[str] = Field(default_factory=list)
    max_file_bytes: int = 200_000
    max_total_bytes: int = 800_000


class CacheConfig(BaseModel):
    """Cache response cua provider theo (prompt cuoi cung + options), giup
    khong tra gia inference/GPU lap lai cho cung mot cau hoi."""

    enabled: bool = True
    max_entries: int = 256
    ttl_seconds: float | None = 3600


class AppConfig(BaseSettings):
    """Cau hinh goc. Nguon uu tien (cao -> thap):
    init kwargs > bien moi truong that (LOCAL_AI_GATEWAY_..., nested bang
    '__') > file .env o repo root > default.yaml.
    """

    model_config = SettingsConfigDict(
        env_prefix="LOCAL_AI_GATEWAY_",
        env_nested_delimiter="__",
        env_file=_REPO_ROOT / ".env",
        yaml_file=_resolve_path(CONFIG_PATH_ENV_VAR, _DEFAULT_CONFIG_PATH),
        extra="ignore",
    )

    provider: str = "ollama"
    ollama: OllamaProviderConfig = Field(default_factory=OllamaProviderConfig)
    timeout_seconds: float = 60.0
    retry: RetryConfig = Field(default_factory=RetryConfig)
    concurrency: ConcurrencyConfig = Field(default_factory=ConcurrencyConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    file_access: FileAccessConfig = Field(default_factory=FileAccessConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)

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
    """Doc budgets.yaml va tra ve {ten_budget: BudgetProfile}."""

    resolved = path or _resolve_path(BUDGETS_PATH_ENV_VAR, _DEFAULT_BUDGETS_PATH)
    if not resolved.exists():
        raise ConfigurationError(f"Khong tim thay file budgets config: {resolved}")

    with open(resolved, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    budgets_raw = raw.get("budgets") or {}
    if not budgets_raw:
        raise ConfigurationError(f"File budgets config rong hoac sai dinh dang: {resolved}")

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
        except Exception as exc:  # noqa: BLE001 - goi thanh loi domain co ngu canh
            raise ConfigurationError(f"Budget '{name}' cau hinh sai: {exc}") from exc

    return profiles
