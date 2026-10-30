"""DTO transferring data between MCP layer and application layer.

Note: 'budget' is typed as str (not a fixed Literal["L1","L2","L3"])
because the valid budget list comes from budgets.yaml at runtime - adding
new budgets must not require code/type changes here.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from llm_worker_pool.domain.models import OutputFormat


class DelegateTaskRequest(BaseModel):
    task: str
    provider: str | None = None
    budget: str = "L2"
    context: str | None = None
    file_paths: list[str] | None = None
    """File paths for the server to read directly (instead of pasting content into
    'context'), saving tokens on Claude's side. Requires configuration of
    file_access.allowed_roots to work - see domain/file_context.py."""
    urls: list[str] | None = None
    """Web pages or GitHub repos for the server to fetch and inline into the context
    (see infra/url_fetcher.py)."""
    output_format: OutputFormat = "text"
    json_schema: dict[str, Any] | None = None
    task_type: str | None = None
    """Kind of work (config/task_profiles.yaml): adds a system-prompt contract, default
    budget/output format/schema and validators. None = no profile, i.e. exactly the
    behaviour from before profiles existed. A field the caller sets explicitly
    (budget, output_format, json_schema) always beats the profile's default."""


class DelegateTaskResult(BaseModel):
    success: bool
    status: Literal["done", "running", "failed"] = "done"
    job_id: str | None = None
    content: Any = None
    raw_text: str = ""
    provider: str = ""
    model: str = ""
    budget_used: str = ""
    latency_ms: float = 0.0
    tokens_used: int | None = None
    """Grand total billed: prompt + thinking + completion."""

    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    thinking_tokens: int | None = None
    """Charged inside tokens_used but reported in neither of the other two."""

    truncated: bool = False
    """The answer was cut off at the output ceiling rather than finished."""

    cached: bool = False
    warnings: list[str] = Field(default_factory=list)
    error: str | None = None
    task_type: str | None = None
    guardrail_actions: list[str] = Field(default_factory=list)
    """What the guardrails did on this call: file_denied_sensitive:<path>,
    secret_redacted:<kind> xN (input, before a cloud call), secret_in_response:<kind> xN."""

    verification: dict[str, Any] | None = None
    """Citation check (task types with the `citations` validator): every path:line the
    answer cites, with status ok / file_not_in_context / line_out_of_range /
    quote_not_found. Lets the caller drop invented findings without opening files."""
