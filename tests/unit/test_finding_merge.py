from __future__ import annotations

from llm_worker_pool.domain.finding_merge import findings_of, merge_findings


def _f(file: str, line: int, severity: str = "medium", quote: str = "", claim: str = "bug") -> dict:
    return {"file": file, "line": line, "severity": severity, "claim": claim, "evidence_quote": quote}


def test_same_bug_within_window_is_merged_with_both_reporters():
    merged = merge_findings({"groq": [_f("orders.py", 17, "low")], "lite": [_f("orders.py", 19, "high")]})
    assert len(merged) == 1
    assert merged[0]["agreed_by"] == ["groq", "lite"]
    assert merged[0]["severity"] == "high"
    assert merged[0]["line"] == 17


def test_lines_too_far_apart_stay_separate():
    merged = merge_findings({"groq": [_f("orders.py", 17)], "lite": [_f("orders.py", 21)]})
    assert [m["agreed_by"] for m in merged] == [["groq"], ["lite"]]


def test_same_quote_on_distant_lines_is_not_merged():
    # `return None` on two different lines is two bugs, not one.
    merged = merge_findings(
        {"groq": [_f("a.py", 10, quote="return None")], "lite": [_f("a.py", 40, quote="return None")]}
    )
    assert len(merged) == 2


def test_quote_breaks_ties_inside_the_window():
    groq = [_f("a.py", 10, quote="x = 1"), _f("a.py", 12, quote="y = 2")]
    lite = [_f("a.py", 11, quote="y = 2")]
    merged = merge_findings({"groq": groq, "lite": lite})
    by_quote = {m["evidence_quote"]: m["agreed_by"] for m in merged}
    assert by_quote == {"y = 2": ["groq", "lite"], "x = 1": ["groq"]}


def test_findings_of_one_arm_are_never_merged_together():
    merged = merge_findings({"groq": [_f("a.py", 10), _f("a.py", 11)]})
    assert len(merged) == 2


def test_short_and_full_paths_match():
    merged = merge_findings(
        {"groq": [_f("orders.py", 5)], "lite": [_f("D:\\repo\\fixtures\\orders.py", 5)]}
    )
    assert len(merged) == 1


def test_different_files_never_merge():
    merged = merge_findings({"groq": [_f("a.py", 5)], "lite": [_f("b.py", 5)]})
    assert len(merged) == 2


def test_agreed_findings_rank_first_and_citations_are_checked():
    files = {"orders.py": "a\nb\ntotal = price * qty\nd\ne\nf\ng\nh\ni\nj"}
    merged = merge_findings(
        {
            "groq": [_f("orders.py", 3, "low", quote="total = price * qty"), _f("orders.py", 9, "high", quote="nope")],
            "lite": [_f("orders.py", 3, "low", quote="total = price * qty")],
        },
        files=files,
    )
    assert merged[0]["agreed_by"] == ["groq", "lite"]
    assert merged[0]["citation"] == "ok"
    # The unverified finding is kept and marked, not dropped.
    assert merged[1]["citation"] == "quote_not_found"


def test_finding_without_location_is_kept():
    merged = merge_findings({"groq": [{"claim": "global bug"}]}, files={"a.py": "x"})
    assert merged[0]["citation"] == "missing_location"
    assert "_path" not in merged[0]


def test_findings_of():
    assert findings_of({"findings": [_f("a", 1), "junk"]}) == [_f("a", 1)]
    assert findings_of({"other": []}) is None
    assert findings_of("text") is None


def test_suffix_names_and_malformed_findings():
    # "a.py" vs "aa.py"/"data.py" are different files: suffixes only match after a "/".
    assert len(merge_findings({"g": [_f("a.py", 10)], "l": [_f("aa.py", 10)]})) == 2
    assert len(merge_findings({"g": [_f("a.py", 10)], "l": [_f("data.py", 10)]})) == 2
    # No file: never merged, even on adjacent lines.
    assert len(merge_findings({"g": [{"line": 5}], "l": [{"line": 6}]})) == 2
    # Non-numeric line / non-string severity: no crash, marked as unlocatable.
    merged = merge_findings(
        {"g": [{"file": "a.py", "line": "ten", "severity": 5}], "l": [{"file": "a.py", "line": None}]},
        files={"a.py": "x"},
    )
    assert [m["citation"] for m in merged] == ["missing_location", "missing_location"]
