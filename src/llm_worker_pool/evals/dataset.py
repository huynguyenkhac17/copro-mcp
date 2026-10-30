"""Eval cases: one JSON object per line in evals/datasets/*.jsonl."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from llm_worker_pool.domain.models import OutputFormat


class ExpectedFinding(BaseModel):
    file: str
    """Basename or path suffix of the file the bug is in."""
    line: int


class EvalCase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    task: str
    task_type: str
    budget: str = "L2"
    file_paths: list[str] | None = None
    """Relative to the repository root (resolved by the runner)."""
    context: str | None = None
    output_format: OutputFormat = "text"
    json_schema: dict[str, Any] | None = None
    reference: str | None = None
    """Ground-truth answer, enables FactualCorrectness (and context metrics on RAG cases)."""
    expected_findings: list[ExpectedFinding] | None = None
    """Seeded bugs; recall counts a citation within +-3 lines of each."""
    expected_fields: dict[str, Any] | None = None
    """Dotted path into the JSON answer -> expected value ("errors.0.service")."""
    must_contain: list[str] | None = None
    must_not_contain: list[str] | None = None
    """Leak / injection checks: none of these may appear in the answer."""
    expect_file_denied: bool = False
    """The case reads a sensitive file the deny-list should refuse."""


def load_cases(paths: list[Path]) -> list[EvalCase]:
    cases: list[EvalCase] = []
    seen: set[str] = set()
    for path in paths:
        files = sorted(path.glob("*.jsonl")) if path.is_dir() else [path]
        for file in files:
            for lineno, line in enumerate(file.read_text(encoding="utf-8").splitlines(), 1):
                if not line.strip():
                    continue
                try:
                    case = EvalCase(**json.loads(line))
                except Exception as exc:  # noqa: BLE001 - point at the bad line
                    raise ValueError(f"{file}:{lineno}: {exc}") from exc
                if case.id in seen:
                    raise ValueError(f"{file}:{lineno}: duplicate case id {case.id!r}")
                seen.add(case.id)
                cases.append(case)
    return cases
