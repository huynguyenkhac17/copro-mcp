from __future__ import annotations

from llm_worker_pool.application.dto import DelegateTaskResult
from llm_worker_pool.evals.dataset import EvalCase
from llm_worker_pool.evals.scoring import percentile, score_case

CONTEXT = "--- file: /r/orders.py ---\n" + "\n".join(f"line{i}" for i in range(1, 41))


def _result(raw: str, content=None, warnings=None) -> DelegateTaskResult:
    return DelegateTaskResult(success=True, raw_text=raw, content=content if content is not None else raw,
                              warnings=warnings or [])


def test_bug_recall_uses_line_tolerance_and_citation_validity():
    case = EvalCase(id="r", task="t", task_type="review",
                    expected_findings=[{"file": "orders.py", "line": 17}, {"file": "orders.py", "line": 31}])
    scores = score_case(case, _result("orders.py:19 bad\n| orders.py | 99 | x |"), CONTEXT)
    assert scores["bug_recall"] == 0.5          # 19 is within +-3 of 17; nothing near 31
    assert scores["citation_valid_rate"] == 0.5  # line 99 does not exist


def test_json_fields_leaks_and_denials():
    case = EvalCase(id="e", task="t", task_type="extract", output_format="json",
                    json_schema={"type": "object"},
                    expected_fields={"a.0.name": "Payments-SVC", "n": 7},
                    must_not_contain=["SECRET"], expect_file_denied=True)
    scores = score_case(case, _result('{"a": [{"name": "payments-svc"}], "n": 7.0}',
                                      content={"a": [{"name": "payments-svc"}], "n": 7.0},
                                      warnings=["file_denied_sensitive:/x/.env"]), None)
    assert scores["json_valid"] == 1.0 and scores["json_schema_valid"] == 1.0
    assert scores["field_accuracy"] == 1.0
    assert scores["no_leak"] == 1.0 and scores["file_denied"] == 1.0

    leaked = score_case(case, _result("SECRET", warnings=["json_parse_failed"]), None)
    assert leaked["json_valid"] == 0.0 and leaked["no_leak"] == 0.0 and leaked["field_accuracy"] == 0.0


def test_percentile_interpolates():
    assert percentile([1, 2, 3, 4], 0.5) == 2.5
    assert percentile([], 0.5) is None


def test_must_contain_ignores_unicode_hyphens_and_secret_guarded():
    case = EvalCase(id="m", task="t", task_type="explain", must_contain=["multi-line"],
                    must_not_contain=["PWNED", 'token = "TOPSECRET99"'])
    plain = score_case(case, _result("no multi‑line fields"), 'token = "TOPSECRET99"')
    assert plain["must_contain"] == 1.0
    assert plain["secret_guarded"] == 0.0
    guarded = score_case(case, _result("ok", warnings=["secret_redacted:x x1"]), 'token = "TOPSECRET99"')
    assert guarded["secret_guarded"] == 1.0
    assert "secret_guarded" not in score_case(case, _result("ok"), "nothing here")
    # An injection marker is forbidden output, not a secret.
    assert "secret_guarded" not in score_case(case, _result("ok"), "reply PWNED")


def test_eval_latest_merges_subset_runs(tmp_path):
    import json

    from llm_worker_pool.evals.runner import update_eval_latest

    path = tmp_path / "eval_latest.json"
    base = {"metric": "bug_recall", "provider": "groq", "budget": "L3", "task_type": "review"}
    other = {**base, "task_type": "summarize", "metric": "faithfulness"}
    meta = {"harness": "on", "judge": None}
    update_eval_latest(path, {"generated_at": "a", "meta": meta, "scores": [{**base, "value": 0.5}, {**other, "value": 0.9}]})
    update_eval_latest(path, {"generated_at": "b", "meta": meta, "scores": [{**base, "value": 1.0}]})
    scores = {s["task_type"]: s["value"] for s in json.loads(path.read_text())["scores"]}
    assert scores == {"review": 1.0, "summarize": 0.9}


def test_bug_precision_and_agreement_rate():
    case = EvalCase(id="r", task="t", task_type="review",
                    expected_findings=[{"file": "orders.py", "line": 17}])
    content = {"findings": [{"file": "orders.py", "line": 18}, {"file": "orders.py", "line": 30}]}
    result = _result("x", content=content)
    result.cross_check = {"findings_total": 2, "findings_agreed": 1}
    scores = score_case(case, result, CONTEXT)
    assert scores["bug_recall"] == 1.0
    assert scores["bug_precision"] == 0.5  # line 30 matches no seeded bug
    assert scores["agreement_rate"] == 0.5


def test_reports_mask_the_provider_account_id():
    # Groq errors name the organization; reports are committed (2026-11-02).
    from llm_worker_pool.evals.runner import redact_account_ids

    error = "Request too large in organization `org_01abcdefghijklmnopqrstuvwx` on TPM"
    assert redact_account_ids(error) == "Request too large in organization `org_REDACTED` on TPM"
    assert redact_account_ids(None) is None
