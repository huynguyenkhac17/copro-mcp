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
import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from llm_worker_pool.domain.file_selection import (
    glob_files,
    is_glob,
    read_text_file,
    sort_key,
    split_glob,
    walk_files,
)
from llm_worker_pool.domain.exceptions import NoUsableContextError
from llm_worker_pool.domain.guardrails import PathDenyList
from llm_worker_pool.domain.token_estimate import estimate_tokens


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
        max_dir_files: int = 300,
        max_dir_depth: int = 12,
        max_dir_entries: int = 20_000,
    ) -> None:
        self._allowed_roots = [_normalize_root(r) for r in allowed_roots]
        self._deny_list = deny_list
        self._max_file_bytes = max_file_bytes
        self._max_total_bytes = max_total_bytes
        self._max_dir_files = max_dir_files
        self._max_dir_depth = max_dir_depth
        self._max_dir_entries = max_dir_entries

    async def aresolve(
        self, 
        paths: list[str], 
        query: str | None = None,
        rag_indexer: Any | None = None,
        rag_min_tokens: int | None = None,
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

        # Gather all files. A directory is walked recursively and a glob
        # ("D:/repo/src/**/*.py") is expanded; both prune node_modules/.venv/.git...
        # (domain/file_selection.py) and list their files most-useful-first, so when the
        # byte budget runs out it is config/CI noise that gets dropped, not source.
        # Each entry: (path, the file_paths item it came from if expanded, else None).
        files_to_read: list[tuple[Path, str | None]] = []
        seen: set[Path] = set()
        for raw in paths:
            if is_glob(raw):
                base, pattern = split_glob(raw)
                try:
                    base_resolved = base.resolve()
                except OSError:
                    warnings.append(f"file_invalid_path:{raw}")
                    continue
                if not self._is_allowed(base_resolved):
                    warnings.append(f"file_access_denied:{raw}")
                    continue
                if not base_resolved.is_dir():
                    warnings.append(f"file_not_found:{raw}")
                    continue
                try:
                    found, hit_limit = glob_files(
                        base_resolved, pattern,
                        max_depth=self._max_dir_depth, max_entries=self._max_dir_entries,
                    )
                except re.error as exc:  # e.g. a character class like [z-a]
                    warnings.append(f"file_invalid_glob:{raw} ({exc})")
                    continue
                if not found and not hit_limit:
                    warnings.append(f"file_glob_no_match:{raw}")
                self._add_expanded(raw, base_resolved, found, hit_limit, files_to_read, seen, warnings)
                continue

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
                found, hit_limit = walk_files(
                    resolved, max_depth=self._max_dir_depth, max_entries=self._max_dir_entries
                )
                self._add_expanded(raw, resolved, found, hit_limit, files_to_read, seen, warnings)
            elif resolved.is_file():
                if resolved not in seen:
                    seen.add(resolved)
                    files_to_read.append((resolved, None))
            else:
                warnings.append(f"file_not_found:{raw}")

        # Files found by expanding a directory/glob are reported per item as counts:
        # one warning per skipped file would bury the answer under hundreds of lines.
        counted: dict[tuple[str, str], int] = {}

        def report(reason: str, raw: str, origin: str | None) -> None:
            if origin is None:
                warnings.append(f"{reason}:{raw}")
            else:
                counted[(reason, origin)] = counted.get((reason, origin), 0) + 1

        for path, origin in files_to_read:
            raw = str(path)
            try:
                size = path.stat().st_size
            except OSError:
                size = None
            reason = None
            if size is None:
                reason = "file_unreadable"
            elif size > self._max_file_bytes:
                reason = "file_too_large"
            elif total_bytes + size > self._max_total_bytes:
                reason = "file_budget_exceeded"
            else:
                try:
                    decoded = read_text_file(path)
                except OSError:
                    decoded = None
                if decoded is None:
                    reason = "file_unreadable"
            if reason is not None:
                report(reason, raw, origin)
                continue

            content, lossy = decoded
            if lossy:
                report("file_decoded_lossy", raw, origin)
            total_bytes += size
            if path.suffix.lower() == ".ipynb":
                rendered = render_notebook(content)
                if rendered is None:
                    warnings.append(f"notebook_unparsed_sent_raw:{raw}")
                else:
                    content = rendered
            texts_by_file[raw] = content
        for (reason, origin), count in counted.items():
            warnings.append(f"{reason}:{origin} ({count} files)")

        if not texts_by_file:
            return ResolvedFileContext(text="", warnings=warnings)

        full_tokens = sum(estimate_tokens(content) for content in texts_by_file.values())
        if (
            rag_indexer and query and self._has_directory(paths)
            and (rag_min_tokens is None or full_tokens > rag_min_tokens)
        ):
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
        
    def _add_expanded(
        self,
        raw: str,
        root: Path,
        found: list[Path],
        hit_limit: bool,
        files_to_read: list[tuple[Path, str | None]],
        seen: set[Path],
        warnings: list[str],
    ) -> None:
        if hit_limit:
            warnings.append(
                f"file_dir_scan_limit:{raw} (stopped after {self._max_dir_entries} entries; "
                "narrow the path or use a glob)"
            )
        kept: list[Path] = []
        for child in found:
            # The walk starts from a resolved root and does not follow directory links,
            # so only a symlinked file can point elsewhere. Resolving every file costs
            # seconds on a large tree on Windows.
            try:
                child_resolved = child.resolve() if child.is_symlink() else child
            except OSError:
                continue
            if self._is_sensitive(child, child_resolved):
                warnings.append(f"file_denied_sensitive:{child}")
                continue
            # A symlink pointing outside the allowed roots is skipped like before.
            if not self._is_allowed(child_resolved) or child_resolved in seen:
                continue
            kept.append(child)
        kept.sort(key=lambda p: sort_key(p, root))
        if len(kept) > self._max_dir_files:
            warnings.append(
                f"file_dir_too_many_files:{raw} ({len(kept) - self._max_dir_files} of "
                f"{len(kept)} omitted, least useful first)"
            )
            kept = kept[: self._max_dir_files]
        for child in kept:
            seen.add(child.resolve() if child.is_symlink() else child)
            files_to_read.append((child, raw))

    def _has_directory(self, paths: list[str]) -> bool:
        for raw in paths:
            if is_glob(raw):
                return True
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
    rag_min_tokens: int | None = None,
) -> tuple[str | None, list[str]]:
    """Shared by DelegateTaskUseCase and MapReduceOrchestrator so both build the
    combined (context + file_paths + urls) text the exact same way."""

    if not file_paths and not urls:
        return context, []

    parts: list[str | None] = [context]
    warnings: list[str] = []
    if file_paths:
        file_result = await resolver.aresolve(
            file_paths, query=query, rag_indexer=rag_indexer, rag_min_tokens=rag_min_tokens
        )
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


def require_file_content(
    file_paths: list[str] | None, context: str | None, warnings: list[str]
) -> None:
    """Refuse before any model call when file_paths were the request's only material and
    none of them could be read (denied, missing, file access disabled...). The model
    would answer the bare task, and success=true hid the warnings behind that answer."""
    if file_paths and not context:
        reasons = "; ".join(warnings) or "no readable content"
        raise NoUsableContextError(
            f"None of file_paths could be read: {reasons}", warnings=warnings
        )
