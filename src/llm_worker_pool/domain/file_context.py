"""Reads file contents directly from specified paths, instead of making Claude
paste content via MCP protocol - saves tokens on Claude's side.

Unlike PromptBuilder (pure text processing), this service uses real I/O
but is still part of the system's business logic (file access security policy)
so it is placed in the domain group, not infra.

Strictly graceful-degradation: NEVER raise an exception due to a bad path - just
skip that path and return a clear warning, similar to PromptBuilder/OutputParser.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from llm_worker_pool.domain.guardrails import PathDenyList


class ResolvedFileContext(BaseModel):
    text: str
    warnings: list[str] = []


def _normalize_root(root: Path | str) -> Path:
    p = Path(root)
    s = str(p)
    if len(s) == 2 and s[1] == ":":
        return Path(f"{s}/").resolve()
    return p.resolve()


class FileContextResolver:
    def __init__(
        self,
        allowed_roots: list[Path],
        max_file_bytes: int = 200_000,
        max_total_bytes: int = 800_000,
        deny_list: PathDenyList | None = None,
    ) -> None:
        self._allowed_roots = [_normalize_root(r) for r in allowed_roots]
        self._deny_list = deny_list
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
            if self._is_sensitive(Path(raw), resolved):
                warnings.append(f"file_denied_sensitive:{raw}")
                continue
                
            if resolved.is_dir():
                # Non-recursive directory traversal, limits to first 100 files
                count = 0
                for child in resolved.iterdir():
                    if not child.is_file():
                        continue
                    child_resolved = child.resolve()
                    if self._is_sensitive(child, child_resolved):
                        warnings.append(f"file_denied_sensitive:{child}")
                        continue
                    if self._is_allowed(child_resolved):
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
            if resolved.suffix.lower() == ".ipynb":
                rendered = render_notebook(content)
                if rendered is None:
                    warnings.append(f"notebook_unparsed_sent_raw:{raw}")
                else:
                    content = rendered
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

    def _is_sensitive(self, given: Path, resolved: Path) -> bool:
        """Checked on both the name the caller used and the symlink target, so neither
        `notes.txt -> .env` nor `.env -> config.txt` slips through."""
        if self._deny_list is None:
            return False
        return bool(self._deny_list.match(given) or self._deny_list.match(resolved))

    def _is_allowed(self, resolved: Path) -> bool:
        return any(
            resolved == root or resolved.is_relative_to(root)
            for root in self._allowed_roots
        )


def render_notebook(raw_json: str) -> str | None:
    """Render a Jupyter notebook as explicitly numbered cells, or None if unparseable.

    Sent as raw JSON, a worker has to count elements of the "cells" array itself -
    and models do that badly: on a 47-cell lab notebook one answered with indices
    up to 50 and silently pointed edits at the wrong cells. Stating each index
    outright removes the counting. Outputs are dropped: they are noise for the
    worker and can be huge (base64 plots).
    """
    try:
        cells = json.loads(raw_json)["cells"]
        parts = [
            f"=== cell {i} ({cell['cell_type']}) ===\n{_cell_source(cell)}"
            for i, cell in enumerate(cells)
        ]
    except (ValueError, KeyError, TypeError):
        return None
    header = (
        f"[Jupyter notebook: {len(cells)} cells, numbered 0-{len(cells) - 1} in file "
        "order; cell outputs omitted]"
    )
    return "\n\n".join([header, *parts])


def _cell_source(cell: dict[str, Any]) -> str:
    source = cell.get("source", "")
    return "".join(source) if isinstance(source, list) else str(source)


async def resolve_request_context(
    resolver: FileContextResolver,
    *,
    context: str | None,
    file_paths: list[str] | None,
    query: str,
    rag_indexer: Any | None = None,
    urls: list[str] | None = None,
    url_fetcher: Any | None = None,
) -> tuple[str | None, list[str]]:
    """Shared by DelegateTaskUseCase and MapReduceOrchestrator so both build the
    combined (context + file_paths + urls) text the exact same way."""

    if not file_paths and not urls:
        return context, []

    parts: list[str | None] = [context]
    warnings: list[str] = []
    if file_paths:
        file_result = await resolver.aresolve(file_paths, query=query, rag_indexer=rag_indexer)
        parts.append(file_result.text)
        warnings.extend(file_result.warnings)
    if urls:
        if url_fetcher is None:
            warnings.append("url_fetch_disabled: enable url_fetch in config to use `urls`")
        else:
            fetched = await url_fetcher.fetch(urls)
            parts.append(fetched.text)
            warnings.extend(fetched.warnings)
    combined = "\n\n".join(part for part in parts if part)
    return combined or None, warnings
