"""Reads file contents directly from specified paths, instead of making Claude
paste content via MCP protocol - saves tokens on Claude's side.

Unlike PromptBuilder (pure text processing), this service uses real I/O
but is still part of the system's business logic (file access security policy)
so it is placed in the domain group, not infra.

Strictly graceful-degradation: NEVER raise an exception due to a bad path - just
skip that path and return a clear warning, similar to PromptBuilder/OutputParser.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel


class ResolvedFileContext(BaseModel):
    text: str
    warnings: list[str] = []


class FileContextResolver:
    def __init__(
        self,
        allowed_roots: list[Path],
        max_file_bytes: int = 200_000,
        max_total_bytes: int = 800_000,
    ) -> None:
        self._allowed_roots = allowed_roots
        self._max_file_bytes = max_file_bytes
        self._max_total_bytes = max_total_bytes

    def resolve(self, paths: list[str]) -> ResolvedFileContext:
        if not self._allowed_roots:
            return ResolvedFileContext(
                text="",
                warnings=[
                    "file_access_disabled: configure file_access.allowed_roots "
                    "to enable direct file reading features"
                ],
            )

        text_parts: list[str] = []
        warnings: list[str] = []
        total_bytes = 0

        for raw in paths:
            try:
                resolved = Path(raw).resolve()
            except OSError:
                warnings.append(f"file_invalid_path:{raw}")
                continue

            if not self._is_allowed(resolved):
                warnings.append(f"file_access_denied:{raw}")
                continue

            if not resolved.is_file():
                warnings.append(f"file_not_found:{raw}")
                continue

            size = resolved.stat().st_size
            if size > self._max_file_bytes:
                warnings.append(f"file_too_large:{raw}")
                continue
            if total_bytes + size > self._max_total_bytes:
                warnings.append(f"file_budget_exceeded:{raw}")
                continue

            try:
                content = resolved.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                warnings.append(f"file_unreadable:{raw}")
                continue

            total_bytes += size
            text_parts.append(f"--- file: {raw} ---\n{content}")

        return ResolvedFileContext(text="\n\n".join(text_parts), warnings=warnings)

    def _is_allowed(self, resolved: Path) -> bool:
        return any(
            resolved == root or resolved.is_relative_to(root)
            for root in self._allowed_roots
        )
