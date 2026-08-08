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
from typing import Any

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

    async def aresolve(
        self, 
        paths: list[str], 
        query: str | None = None,
        rag_indexer: Any | None = None,
    ) -> ResolvedFileContext:
        if not self._allowed_roots:
            return ResolvedFileContext(
                text="",
                warnings=[
                    "file_access_disabled: configure file_access.allowed_roots "
                    "to enable direct file reading features"
                ],
            )

        texts_by_file: dict[str, str] = {}
        warnings: list[str] = []
        total_bytes = 0

        # Gather all files (including from directories)
        files_to_read = []
        for raw in paths:
            try:
                resolved = Path(raw).resolve()
            except OSError:
                warnings.append(f"file_invalid_path:{raw}")
                continue

            if not self._is_allowed(resolved):
                warnings.append(f"file_access_denied:{raw}")
                continue
                
            if resolved.is_dir():
                # Non-recursive directory traversal, limits to first 100 files
                count = 0
                for child in resolved.iterdir():
                    if child.is_file() and self._is_allowed(child):
                        if count >= 100:
                            warnings.append(f"file_dir_too_many_files:{raw}")
                            break
                        files_to_read.append(child)
                        count += 1
            elif resolved.is_file():
                files_to_read.append(resolved)
            else:
                warnings.append(f"file_not_found:{raw}")

        for resolved in files_to_read:
            raw = str(resolved)
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
            texts_by_file[raw] = content

        if not texts_by_file:
            return ResolvedFileContext(text="", warnings=warnings)

        if rag_indexer and query and self._has_directory(paths):
            try:
                text = await rag_indexer.index_and_search(texts_by_file, query)
                return ResolvedFileContext(text=text, warnings=warnings)
            except Exception as exc:
                warnings.append(f"rag_indexing_failed:{exc}")
                # Fallback to normal if RAG fails

        text_parts = [
            f"--- file: {raw} ---\n{content}" 
            for raw, content in texts_by_file.items()
        ]
        return ResolvedFileContext(text="\n\n".join(text_parts), warnings=warnings)
        
    def _has_directory(self, paths: list[str]) -> bool:
        for raw in paths:
            try:
                if Path(raw).is_dir():
                    return True
            except OSError:
                continue
        return False

    def _is_allowed(self, resolved: Path) -> bool:
        return any(
            resolved == root or resolved.is_relative_to(root)
            for root in self._allowed_roots
        )


async def resolve_request_context(
    resolver: FileContextResolver,
    *,
    context: str | None,
    file_paths: list[str] | None,
    query: str,
    rag_indexer: Any | None = None,
) -> tuple[str | None, list[str]]:
    """Shared by DelegateTaskUseCase and MapReduceOrchestrator so both build the
    combined (context + file_paths) text the exact same way."""

    if not file_paths:
        return context, []

    file_result = await resolver.aresolve(file_paths, query=query, rag_indexer=rag_indexer)
    combined = "\n\n".join(part for part in (context, file_result.text) if part)
    return combined, file_result.warnings
