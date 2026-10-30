"""Merge review findings reported by several providers (cross-check).

Two findings are the same defect only when they sit in the same file within
`line_window` lines of each other. The evidence quote is NOT a merge key on its own:
the same snippet (`return None`, `i += 1`) can appear on many lines, and merging by
quote would fold two different bugs into one. It only breaks ties when several
findings fall inside the window.

Every merged finding keeps who reported it (`agreed_by`) and, when the files are at
hand, whether its citation checks out. A finding that fails the check is kept and
marked rather than dropped - one provider seeing a bug the other missed is exactly
what cross-checking is for, and a wrong line number does not make the bug invented.
"""

from __future__ import annotations

from typing import Any

from llm_worker_pool.domain.validators import Citation, CitationValidator

_SEVERITY_RANK = {"high": 3, "medium": 2, "low": 1}


def _norm_path(path: Any) -> str:
    return str(path or "").replace("\\", "/").strip().lower()


def _same_file(a: str, b: str) -> bool:
    if a == b:
        return True
    # One provider may cite "orders.py", the other the full context-header path.
    return a.endswith("/" + b) or b.endswith("/" + a)


def _norm_quote(quote: Any) -> str:
    return " ".join(str(quote or "").split())


def _line(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def findings_of(content: Any) -> list[dict[str, Any]] | None:
    """The findings list of a review answer, or None when the answer has none."""
    if isinstance(content, dict) and isinstance(content.get("findings"), list):
        return [f for f in content["findings"] if isinstance(f, dict)]
    return None


def merge_findings(
    by_arm: dict[str, list[dict[str, Any]]],
    *,
    line_window: int = 3,
    files: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    for arm, findings in by_arm.items():
        for finding in findings:
            line = _line(finding.get("line"))
            path = _norm_path(finding.get("file"))
            target = None
            if line is not None and path:
                near = [
                    m for m in merged
                    if arm not in m["agreed_by"]
                    and _same_file(m["_path"], path)
                    and abs(m["_line"] - line) <= line_window
                ]
                if near:
                    quote = _norm_quote(finding.get("evidence_quote"))
                    same_quote = [m for m in near if quote and _norm_quote(m.get("evidence_quote")) == quote]
                    pool = same_quote or near
                    target = min(pool, key=lambda m: abs(m["_line"] - line))
            if target is None:
                entry = dict(finding)
                entry["agreed_by"] = [arm]
                entry["_path"], entry["_line"] = path, line if line is not None else -10**9
                merged.append(entry)
                continue
            target["agreed_by"].append(arm)
            if _SEVERITY_RANK.get(str(finding.get("severity")).lower(), 0) > _SEVERITY_RANK.get(
                str(target.get("severity")).lower(), 0
            ):
                target["severity"] = finding.get("severity")
            if not _norm_quote(target.get("evidence_quote")) and finding.get("evidence_quote"):
                target["evidence_quote"] = finding["evidence_quote"]

    validator = CitationValidator(files) if files else None
    for entry in merged:
        del entry["_path"], entry["_line"]
        if validator is not None:
            line = _line(entry.get("line"))
            if line is not None and entry.get("file"):
                check = validator.check(
                    Citation(file=str(entry["file"]), line=line, quote=entry.get("evidence_quote"))
                )
                entry["citation"] = check.status
                if check.found_at_line is not None:
                    entry["found_at_line"] = check.found_at_line
            else:
                entry["citation"] = "missing_location"

    def rank(entry: dict[str, Any]) -> tuple:
        return (
            -len(entry["agreed_by"]),
            entry.get("citation", "ok") != "ok",
            -_SEVERITY_RANK.get(str(entry.get("severity")).lower(), 0),
            _norm_path(entry.get("file")),
            _line(entry.get("line")) or 0,
        )

    return sorted(merged, key=rank)
