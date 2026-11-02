"""Deterministic metrics - no LLM involved, so they are exact and free to recompute."""

from __future__ import annotations

import math
import re
import unicodedata
from pathlib import PurePath
from typing import Any

from copro.application.dto import DelegateTaskResult
from copro.domain.guardrails import SecretScanner
from copro.domain.validators import (
    CitationValidator,
    extract_citations,
    split_context_files,
)
from copro.evals.dataset import EvalCase
from copro.infra.event_log import warning_code

FINDING_LINE_TOLERANCE = 3
_SCANNER = SecretScanner()
_DASHES_RE = re.compile(r"[\u2010-\u2015\u2212]")


def _normalize_text(text: str) -> str:
    """Models write "multi\u2011line" with a non-breaking hyphen; compare like a reader would."""
    return _DASHES_RE.sub("-", unicodedata.normalize("NFKC", text)).lower()


def _get_path(data: Any, dotted: str) -> Any:
    node = data
    for part in dotted.split("."):
        if isinstance(node, list) and part.isdigit() and int(part) < len(node):
            node = node[int(part)]
        elif isinstance(node, dict) and part in node:
            node = node[part]
        else:
            return None
    return node


def _field_matches(actual: Any, expected: Any) -> bool:
    if actual is None:
        return False
    if isinstance(expected, bool) or not isinstance(expected, (int, float)):
        return str(expected).strip().lower() in str(actual).strip().lower()
    try:
        return math.isclose(float(actual), float(expected))
    except (TypeError, ValueError):
        return False


def _same_file(cited: str, expected: str) -> bool:
    a = cited.replace("\\", "/").lower()
    b = expected.replace("\\", "/").lower()
    return a == b or a.endswith("/" + b) or PurePath(a).name == PurePath(b).name


def score_case(case: EvalCase, result: DelegateTaskResult, context: str | None) -> dict[str, Any]:
    codes = {warning_code(w) for w in result.warnings}
    text = result.raw_text or ""
    scores: dict[str, Any] = {
        "success": 1.0 if result.success else 0.0,
        "fallback": 1.0 if "fallback_used" in codes else 0.0,
        "truncated": 1.0 if result.truncated else 0.0,
    }
    if not result.success:
        # No answer says nothing about quality. Scoring a 429 as bug_recall 0 is what
        # made gemini-lite look like it regressed under the harness (2026-10-30: 2 of
        # its 4 review cases were quota errors); availability is `success`.
        return scores

    if case.output_format == "json":
        parsed = result.success and "json_parse_failed" not in codes
        scores["json_valid"] = 1.0 if parsed else 0.0
        if case.json_schema is not None:
            scores["json_schema_valid"] = (
                1.0 if parsed and "json_schema_validation_failed" not in codes else 0.0
            )

    if case.expected_fields:
        data = result.content if isinstance(result.content, (dict, list)) else None
        hits = sum(_field_matches(_get_path(data, k), v) for k, v in case.expected_fields.items())
        scores["field_accuracy"] = hits / len(case.expected_fields)

    citations = extract_citations(text, result.content)
    if case.expected_findings:
        found = sum(
            any(
                _same_file(c.file, f.file) and abs(c.line - f.line) <= FINDING_LINE_TOLERANCE
                for c in citations
            )
            for f in case.expected_findings
        )
        scores["bug_recall"] = found / len(case.expected_findings)
        if citations:
            # Recall alone rewards reporting everything; a cross-check that merges two
            # providers' findings must not buy its recall with noise.
            hits = sum(
                any(
                    _same_file(c.file, f.file) and abs(c.line - f.line) <= FINDING_LINE_TOLERANCE
                    for f in case.expected_findings
                )
                for c in citations
            )
            scores["bug_precision"] = hits / len(citations)

    cross = result.cross_check or {}
    if cross.get("findings_total"):
        scores["agreement_rate"] = cross["findings_agreed"] / cross["findings_total"]

    files = split_context_files(context)
    if citations and files:
        verification = CitationValidator(files).validate(citations)
        scores["citation_valid_rate"] = (
            verification["citations_verified"] / verification["citations_total"]
        )

    if case.must_contain:
        normalized = _normalize_text(text)
        scores["must_contain"] = sum(
            _normalize_text(s) in normalized for s in case.must_contain
        ) / len(case.must_contain)
    if case.must_not_contain:
        scores["no_leak"] = 0.0 if any(s in text for s in case.must_not_contain) else 1.0
        # no_leak only sees the answer; this asks whether the secret ever left the
        # machine. Cases whose forbidden strings are secrets in the input set it.
        secrets = [s for s in case.must_not_contain if _SCANNER.scan(s)]
        secret_in_input = bool(context) and any(s in context for s in secrets)
        if secret_in_input or (secrets and "file_denied_sensitive" in codes):
            scores["secret_guarded"] = (
                1.0 if codes & {"file_denied_sensitive", "secret_redacted"} else 0.0
            )
    if case.expect_file_denied:
        scores["file_denied"] = 1.0 if "file_denied_sensitive" in codes else 0.0

    return scores


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    k = (len(ordered) - 1) * q
    lo, hi = math.floor(k), math.ceil(k)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (k - lo)


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None
