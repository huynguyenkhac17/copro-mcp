"""DTO transferring data between MCP layer and application layer.

Note: 'budget' is typed as str (not a fixed Literal["L1","L2","L3"])
because the valid budget list comes from budgets.yaml at runtime - adding
new budgets must not require code/type changes here.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from local_ai_gateway.domain.models import OutputFormat


class CallLocalRequest(BaseModel):
    task: str
    budget: str = "L2"
    context: str | None = None
    file_paths: list[str] | None = None
    """File paths for the server to read directly (instead of pasting content into
    'context'), saving tokens on Claude's side. Requires configuration of
    file_access.allowed_roots to work - see domain/file_context.py."""
    output_format: OutputFormat = "text"
    json_schema: dict[str, Any] | None = None


class CallLocalResult(BaseModel):
    success: bool
    content: Any = None
    raw_text: str = ""
    provider: str = ""
    model: str = ""
    budget_used: str = ""
    latency_ms: float = 0.0
    tokens_used: int | None = None
    cached: bool = False
    warnings: list[str] = Field(default_factory=list)
    error: str | None = None
