"""Post-checks on a worker's answer, selected per task type (config/task_profiles.yaml).

CitationValidator: does each `path:line` (or JSON {file, line}) the worker cites point
at a file that was actually in its context, at a line that exists, and - when it
quoted evidence - does the quote appear near that line? Workers do invent plausible
line numbers; this lets Claude drop such findings without opening every file.

The JSON-repair retry needs a provider call, so it lives in DelegateTaskUseCase; only
its prompt is built here.
"""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel

from copro.domain.output_parser import OutputParser

_FILE_HEADER_RE = re.compile(r"^--- file: (.+?) ---$", re.MULTILINE)
# path with an extension, then :line - "src/a.py:42", "a.py:L42", "D:\\x\\a.py:7"
_INLINE_CITATION_RE = re.compile(
    r"(?<![\w/\\.-])((?:[A-Za-z]:)?[\w./\\-]*[\w-]\.[A-Za-z0-9]{1,8}):L?(\d{1,6})\b"
)
_PATH = r"((?:[A-Za-z]:)?[\w./\\-]*[\w-]\.[A-Za-z0-9]{1,8})"
# Markdown table row "| path | 42 |" - the shape workers most often answer reviews in.
_TABLE_CITATION_RE = re.compile(r"\|\s*`?" + _PATH + r"`?\s*\|\s*L?(\d{1,6})\s*\|")
# "orders.py line 42", "`orders.py` (line 42)", "orders.py, lines 42-44"
_WORDED_CITATION_RE = re.compile(_PATH + r"`?[\s,(:-]{1,4}lines?\s+(\d{1,6})\b", re.IGNORECASE)
QUOTE_WINDOW_LINES = 3
"""A quote counts as found if it appears within this many lines of the cited one."""
# A backslash before a quote or another backslash: what a worker leaves when it
# JSON-escapes the code it quotes one time too many.
_OVERESCAPED_RE = re.compile(r"""\\(["'\\])""")


def _quote_variants(quote: str) -> list[str]:
    """The quote as given, then with one layer of escaping removed. Seen live
    (2026-10-31): both review providers quoted `f"SELECT ..."` as `f\\"SELECT ...\\"`,
    so a real SQL injection finding came back quote_not_found. The literal form is
    tried first, so code that really contains `\\"` still matches as itself."""
    unescaped = _OVERESCAPED_RE.sub(r"\1", quote)
    return [quote] if unescaped == quote else [quote, unescaped]


class Citation(BaseModel):
    file: str
    line: int
    quote: str | None = None


class CitationCheck(BaseModel):
    file: str
    line: int
    status: str
    """ok | file_not_in_context | line_out_of_range | quote_not_found"""
    resolved_file: str | None = None
    found_at_line: int | None = None
    """When the quote is not near the cited line but exists elsewhere in the file: the
    line it actually starts on (workers often get the code right and the number wrong)."""


def split_context_files(context: str | None) -> dict[str, str]:
    """{path: content} for every `--- file: <path> ---` block FileContextResolver /
    the repo fetcher emitted. Content ends at the next block header of any kind."""
    if not context:
        return {}
    files: dict[str, str] = {}
    headers = list(_FILE_HEADER_RE.finditer(context))
    for i, match in enumerate(headers):
        start = match.end() + 1
        end = headers[i + 1].start() if i + 1 < len(headers) else len(context)
        body = context[start:end]
        # Drop a trailing "--- url: ..." / "--- repo: ..." block that is not a file.
        other = re.search(r"^--- (?:url|repo): ", body, re.MULTILINE)
        if other:
            body = body[: other.start()]
        files[match.group(1).strip()] = body.rstrip("\n")
    return files


def extract_citations(raw_text: str, content: Any = None) -> list[Citation]:
    """Citations from structured content ({file, line[, evidence_quote]} anywhere in
    the JSON) plus inline `path:line` mentions in the raw text. Deduplicated."""
    found: dict[tuple[str, int], Citation] = {}

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            file = node.get("file", node.get("path"))
            line = node.get("line")
            if isinstance(file, str) and isinstance(line, (int, str)) and str(line).isdigit():
                quote = node.get("evidence_quote")
                key = (file, int(line))
                found[key] = Citation(
                    file=file, line=int(line), quote=quote if isinstance(quote, str) else None
                )
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    if not isinstance(content, (dict, list)) and raw_text:
        # A text-format answer that is (or embeds) JSON - workers often reply that way.
        content = OutputParser().parse(raw_text).content
    walk(content)
    for pattern in (_INLINE_CITATION_RE, _TABLE_CITATION_RE, _WORDED_CITATION_RE):
        for match in pattern.finditer(raw_text or ""):
            key = (match.group(1), int(match.group(2)))
            found.setdefault(key, Citation(file=key[0], line=key[1]))
    return list(found.values())


def _normalize(path: str) -> str:
    return path.replace("\\", "/").strip().lstrip("./").lower()


class CitationValidator:
    def __init__(self, files: dict[str, str]) -> None:
        self._files = files
        self._lines = {path: content.splitlines() for path, content in files.items()}

    def _match_file(self, cited: str) -> list[str]:
        want = _normalize(cited)
        if not want:
            return []
        matches = []
        for path in self._files:
            have = _normalize(path)
            if have == want or have.endswith("/" + want):
                matches.append(path)
        return matches

    def check(self, citation: Citation) -> CitationCheck:
        candidates = self._match_file(citation.file)
        if not candidates:
            return CitationCheck(file=citation.file, line=citation.line, status="file_not_in_context")

        status = "line_out_of_range"
        resolved: str | None = candidates[0]
        for path in candidates:
            lines = self._lines[path]
            if not 1 <= citation.line <= len(lines):
                continue
            resolved = path
            if citation.quote and not any(
                self._quote_near(lines, citation.line, q) for q in _quote_variants(citation.quote)
            ):
                status = "quote_not_found"
                continue
            return CitationCheck(file=citation.file, line=citation.line, status="ok", resolved_file=path)
        found_at = None
        if citation.quote and resolved is not None:
            found_at = next(
                (
                    at for q in _quote_variants(citation.quote)
                    if (at := self._find_quote(self._lines[resolved], q)) is not None
                ),
                None,
            )
            if found_at is not None and status == "line_out_of_range":
                status = "quote_not_found"  # line is wrong, the code itself exists
        return CitationCheck(
            file=citation.file, line=citation.line, status=status, resolved_file=resolved,
            found_at_line=found_at,
        )

    @staticmethod
    def _find_quote(lines: list[str], quote: str) -> int | None:
        needle = " ".join(quote.split())
        if not needle:
            return None
        for i in range(len(lines)):
            window = " ".join(" ".join(lines[i : i + 1 + QUOTE_WINDOW_LINES]).split())
            if window.startswith(needle) or needle in " ".join(lines[i].split()):
                return i + 1
        return None

    @staticmethod
    def _quote_near(lines: list[str], line: int, quote: str) -> bool:
        needle = " ".join(quote.split())
        if not needle:
            return True
        lo = max(0, line - 1 - QUOTE_WINDOW_LINES)
        hi = min(len(lines), line + QUOTE_WINDOW_LINES)
        window = " ".join(" ".join(lines[lo:hi]).split())
        return needle in window

    def validate(self, citations: list[Citation]) -> dict[str, Any]:
        checks = [self.check(c) for c in citations]
        ok = sum(1 for c in checks if c.status == "ok")
        return {
            "citations_total": len(checks),
            "citations_verified": ok,
            "citations": [c.model_dump(exclude_none=True) for c in checks],
        }


def citation_warnings(verification: dict[str, Any]) -> list[str]:
    return [
        f"citation_unverified:{c['file']}:{c['line']} ({c['status']}"
        + (f", found at line {c['found_at_line']}" if c.get("found_at_line") else "")
        + ")"
        for c in verification.get("citations", [])
        if c["status"] != "ok"
    ]


def build_json_repair_prompt(raw_text: str, errors: list[str], schema: dict | None) -> str:
    parts = [
        "Your previous answer was supposed to be a single valid JSON value but it was "
        "rejected by the parser.",
        "Errors:\n" + "\n".join(f"- {e}" for e in errors),
    ]
    if schema is not None:
        parts.append(f"Required JSON schema:\n{schema}")
    parts.append("Previous answer:\n" + raw_text)
    parts.append(
        "Return ONLY the corrected JSON - same information, no explanations, no "
        "markdown code fences."
    )
    return "\n\n".join(parts)
