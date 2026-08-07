"""DTO chuyen giao giua MCP layer va application layer.

Luu y: 'budget' co kieu str (khong phai Literal["L1","L2","L3"] co dinh)
vi danh sach budget hop le den tu budgets.yaml luc runtime - them budget
moi khong duoc phep doi hoi sua code/type o day.
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
    """Duong dan file de server tu doc truc tiep (thay vi paste noi dung vao
    'context'), tiet kiem token phia Claude. Can cau hinh
    file_access.allowed_roots moi hoat dong - xem domain/file_context.py."""
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
