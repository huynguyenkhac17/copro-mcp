from __future__ import annotations

from copro.domain.validators import (
    Citation,
    CitationValidator,
    build_json_repair_prompt,
    citation_warnings,
    extract_citations,
    split_context_files,
)

CONTEXT = (
    "--- file: /repo/src/orders.py ---\n"
    "def add(a, b=[]):\n"
    "    b.append(a)\n"
    "    return b\n"
    "\n\n"
    "--- file: /repo/src/util.py ---\n"
    "X = 1\n"
)


def test_split_context_files():
    files = split_context_files(CONTEXT)
    assert list(files) == ["/repo/src/orders.py", "/repo/src/util.py"]
    assert files["/repo/src/orders.py"].splitlines()[0] == "def add(a, b=[]):"
    assert files["/repo/src/util.py"] == "X = 1"


def test_split_context_stops_at_url_block():
    files = split_context_files("--- file: a.py ---\nx\n\n--- url: https://e.x ---\npage text")
    assert files == {"a.py": "x"}


def test_extracts_inline_table_worded_and_json_citations():
    raw = (
        "Bug at src/orders.py:1 and see util.py line 1.\n"
        "| `orders.py` | 2 | high |\n"
    )
    content = {"findings": [{"file": "orders.py", "line": 3, "evidence_quote": "return b"}]}
    got = {(c.file, c.line) for c in extract_citations(raw, content)}
    assert got == {("src/orders.py", 1), ("util.py", 1), ("orders.py", 2), ("orders.py", 3)}


def test_citation_ok_wrong_line_and_missing_file():
    v = CitationValidator(split_context_files(CONTEXT))
    assert v.check(Citation(file="orders.py", line=2)).status == "ok"
    assert v.check(Citation(file="src/orders.py", line=2)).resolved_file == "/repo/src/orders.py"
    assert v.check(Citation(file="orders.py", line=40)).status == "line_out_of_range"
    assert v.check(Citation(file="payments.py", line=1)).status == "file_not_in_context"


def test_citation_quote_must_be_near_the_line():
    v = CitationValidator(split_context_files(CONTEXT))
    assert v.check(Citation(file="orders.py", line=1, quote="b.append(a)")).status == "ok"
    assert v.check(Citation(file="orders.py", line=1, quote="b.pop()")).status == "quote_not_found"
    assert v.check(Citation(file="orders.py", line=3, quote="return   b")).status == "ok"


def test_validate_and_warnings():
    v = CitationValidator(split_context_files(CONTEXT))
    result = v.validate([Citation(file="orders.py", line=2), Citation(file="nope.py", line=9)])
    assert result["citations_total"] == 2 and result["citations_verified"] == 1
    assert citation_warnings(result) == ["citation_unverified:nope.py:9 (file_not_in_context)"]


def test_repair_prompt_carries_output_errors_and_schema():
    prompt = build_json_repair_prompt('{"a": ', ["json_parse_failed"], {"type": "object"})
    assert '{"a": ' in prompt and "json_parse_failed" in prompt and "'type': 'object'" in prompt


def test_extracts_citations_from_json_embedded_in_text_with_path_key():
    raw = 'Here you go:\n[{"path": "orders.py", "line": 2, "evidence_quote": "b.append(a)"}]'
    got = extract_citations(raw, raw)
    assert [(c.file, c.line, c.quote) for c in got] == [("orders.py", 2, "b.append(a)")]


def test_wrong_line_with_real_quote_reports_where_it_is():
    v = CitationValidator(split_context_files(CONTEXT))
    far = v.check(Citation(file="orders.py", line=40, quote="return b"))
    assert far.status == "quote_not_found" and far.found_at_line == 3
    assert "found at line 3" in citation_warnings(v.validate([Citation(file="orders.py", line=40, quote="return b")]))[0]
    assert v.check(Citation(file="orders.py", line=1, quote="nowhere()")).found_at_line is None


def test_overescaped_quote_still_matches():
    # Live case 2026-10-31: the quote came back with \" for every " in the code.
    code = "def find(cursor, customer):\n    cursor.execute(f\"SELECT * FROM orders WHERE customer = '{customer}'\")\n"
    v = CitationValidator({"orders.py": code})
    quote = "cursor.execute(f\\\"SELECT * FROM orders WHERE customer = '{customer}'\\\")"
    assert "\\\"" in quote
    assert v.check(Citation(file="orders.py", line=2, quote=quote)).status == "ok"
    far = v.check(Citation(file="orders.py", line=40, quote=quote))
    assert far.found_at_line == 2


def test_code_that_really_contains_backslash_quote_matches_literally():
    code = 'msg = "say \\"hi\\""\n'
    v = CitationValidator({"a.py": code})
    assert v.check(Citation(file="a.py", line=1, quote='msg = "say \\"hi\\""')).status == "ok"
    assert v.check(Citation(file="a.py", line=1, quote="something else")).status == "quote_not_found"
