"""Loads config: default.yaml (+ env override) and budgets.yaml.

AppConfig is loaded ONCE at server startup (server.py), then passed
down as a constructor parameter (dependency injection) - do not use
module-level global variables for this state.
"""

from __future__ import annotations

import os
from pathlib import Path

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)

from llm_worker_pool.domain.exceptions import ConfigurationError
from llm_worker_pool.domain.models import BudgetProfile, GenerationOptions, TaskProfile

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_CONFIG_PATH = _REPO_ROOT / "config" / "default.yaml"
_DEFAULT_BUDGETS_PATH = _REPO_ROOT / "config" / "budgets.yaml"
_DEFAULT_TASK_PROFILES_PATH = _REPO_ROOT / "config" / "task_profiles.yaml"

# pydantic-settings' own dotenv source only feeds declared AppConfig fields - it never
# touches os.environ. But provider API keys (api_key_env/api_key_envs) are arbitrary
# variable names read directly via os.environ.get() in providers/registry.py, so without
# this they'd be invisible unless exported as real OS environment variables. override=False
# keeps real env vars taking priority over .env, matching the documented priority order.
load_dotenv(_REPO_ROOT / ".env", override=False)

CONFIG_PATH_ENV_VAR = "LLM_WORKER_POOL_CONFIG"
BUDGETS_PATH_ENV_VAR = "LLM_WORKER_POOL_BUDGETS_CONFIG"
TASK_PROFILES_PATH_ENV_VAR = "LLM_WORKER_POOL_TASK_PROFILES_CONFIG"


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
    per_key_concurrency: int | None = None
    """Max parallel generate() calls per API key. If set, overrides
    provider-level concurrency wrapping."""
    max_prompt_tokens: int | None = None
    """Skip this provider without calling it when the estimated prompt is larger. For
    vendors with a small per-request ceiling: Groq's free tier allows 8K tokens/minute
    per key and answers a bigger request with 413 (measured 2026-10-07: an 11K-token L3
    prompt was refused, a 3.5K-token L2 prompt passed). None = no limit."""


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
    max_dir_files: int = 300
    """Files kept per directory/glob item, most useful first (domain/file_selection.py)."""
    max_dir_depth: int = 12
    max_dir_entries: int = 20_000
    """Directory entries scanned per item before giving up - pointing at a whole drive
    must return quickly with a warning, not walk it."""


class UrlFetchConfig(BaseModel):
    """Lets delegate_task fetch `urls` (web pages, GitHub repos) server-side."""

    enabled: bool = True
    max_bytes_per_url: int = 1_000_000
    """Raw download cap per page - HTML shrinks a lot once reduced to text."""
    max_total_chars: int = 800_000
    """Cap on all fetched text combined, across every URL in one request."""
    timeout_seconds: float = 30.0
    allow_private_hosts: bool = False
    """Off = refuse localhost / private-network addresses (SSRF guard)."""
    repo_max_file_bytes: int = 100_000


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


class CircuitBreakerConfig(BaseModel):
    enabled: bool = True
    failure_threshold: int = 3
    quarantine_seconds: float = 60.0


class QuotaConfig(BaseModel):
    """Pre-flight quota checks (infra/quota_tracker.py): skip a tier that is known to be
    out of quota or hanging instead of paying a 429 or a full tier timeout for it."""

    enabled: bool = True
    safety_margin_tokens: int = 500
    """Headroom kept below a key's remaining token quota; the chars/4 estimate
    undercounts code and Vietnamese text."""
    timeouts_before_block: int = 2
    """Consecutive unanswered calls before the whole model is skipped for a while."""
    timeout_block_seconds: float = 120.0
    daily_quota_block_seconds: float = 3600.0
    """How long a key is parked after a 429 naming a per-day quota."""


class MetricsConfig(BaseModel):
    """Shared JSONL event log (infra/event_log.py) and the `worker-metrics` exporter."""

    enabled: bool = True
    event_log_path: str | None = None
    """None = ~/.llm_worker_pool/events.jsonl."""
    max_bytes: int = 50_000_000
    """Rotate to events.jsonl.1 past this size."""
    exporter_host: str = "127.0.0.1"
    exporter_port: int = 9464
    eval_latest_path: str | None = None
    """None = ~/.llm_worker_pool/eval_latest.json (written by `worker-eval`)."""


class GuardrailsConfig(BaseModel):
    """Input/output guardrails (domain/guardrails.py)."""

    deny_globs: list[str] | None = None
    """Paths never read through file_paths, even inside allowed_roots.
    None = the built-in list (DEFAULT_DENY_GLOBS); [] disables the deny-list."""
    extra_deny_globs: list[str] = Field(default_factory=list)
    """Added on top of deny_globs (or the defaults) - extend without restating them."""
    redact_secrets: bool = True
    """Replace secret-looking literals with [REDACTED:<kind>] before a cloud call."""
    extra_patterns: dict[str, str] = Field(default_factory=dict)
    """Extra {kind: regex} secret patterns."""
    local_providers: list[str] = Field(default_factory=lambda: ["ollama"])
    """Provider instance names whose traffic never leaves the machine (no redaction)."""


class EvalConfig(BaseModel):
    """`worker-eval` settings. The judge is called through the openai client against an
    OpenAI-compatible endpoint, reusing a provider entry's endpoint and keys."""

    judge_provider: str = "gemini-lite"
    judge_reasoning_effort: str | None = "low"
    judge_timeout_seconds: float = 90.0
    judge_max_retries: int = 2
    embedding_endpoint: str = "http://localhost:11434/v1"
    embedding_model: str = "nomic-embed-text"
    max_concurrency: int = 3


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
    budget_providers: dict[str, str] = Field(default_factory=dict)
    """Default provider per budget when the request names none (e.g. {"L1": "groq"}).
    A budget without an entry - or whose provider failed to start - uses
    default_provider. An explicit `provider` in the request always wins, so routing a
    confidential task to ollama is never overridden."""
    budget_fallbacks: dict[str, list[str]] = Field(default_factory=dict)
    """Fallback chain for a budget's own provider, replacing `fallbacks[provider]` for
    requests routed by budget_providers (e.g. L3: groq -> [gemini, gemini-lite] while
    L2 keeps groq -> [gemini-lite, gemini]). Ignored when the request names a provider."""
    fallbacks: dict[str, list[str]] = Field(default_factory=dict)
    """Per-provider fallback chain, tried in order when the provider fails even after
    retries (e.g. {"gemini": ["gemini-lite", "groq"]}). A provider with no entry never
    falls back - that is what keeps a request sent to local ollama for confidentiality
    from silently leaking to a cloud API."""
    tier_timeout_seconds: float | None = None
    """Cap on how long a tier may run while a later fallback tier is still available.
    The last available tier always gets the budget's full timeout. Exists because a
    saturated model can accept the call and then never answer: measured 2026-10-06,
    gemini-flash-latest held L3 calls for the whole 300s before flash-lite even started.
    None = no cap (every tier gets the full budget timeout)."""
    fallback_reasoning_effort: str | None = None
    """reasoning_effort used once the chain has moved past the requested provider, when
    the budget asks for more ("medium"/"high"). A weaker fallback model thinking at
    medium effort can burn the whole num_predict on reasoning and return a truncated
    answer (flash-lite at L3, 2026-10-06). None = keep the budget's value."""
    timeout_seconds: float = 60.0
    retry: RetryConfig = Field(default_factory=RetryConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    file_access: FileAccessConfig = Field(default_factory=FileAccessConfig)
    url_fetch: UrlFetchConfig = Field(default_factory=UrlFetchConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    jobs: JobsConfig = Field(default_factory=JobsConfig)
    rag: RagConfig = Field(default_factory=RagConfig)
    circuit_breaker: CircuitBreakerConfig = Field(default_factory=CircuitBreakerConfig)
    quota: QuotaConfig = Field(default_factory=QuotaConfig)
    metrics: MetricsConfig = Field(default_factory=MetricsConfig)
    guardrails: GuardrailsConfig = Field(default_factory=GuardrailsConfig)
    eval: EvalConfig = Field(default_factory=EvalConfig)

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


def load_task_profiles(path: Path | None = None) -> dict[str, TaskProfile]:
    """Reads task_profiles.yaml and returns {task_type: TaskProfile}. A missing file is
    not an error (task_type then only accepts None), unlike budgets.yaml."""

    resolved = path or _resolve_path(TASK_PROFILES_PATH_ENV_VAR, _DEFAULT_TASK_PROFILES_PATH)
    if not resolved.exists():
        return {}

    with open(resolved, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    profiles: dict[str, TaskProfile] = {}
    for name, body in (raw.get("task_types") or {}).items():
        try:
            profiles[name] = TaskProfile(name=name, **(body or {}))
        except Exception as exc:  # noqa: BLE001 - wrap in domain error with context
            raise ConfigurationError(f"Task profile '{name}' is misconfigured: {exc}") from exc

    return profiles
