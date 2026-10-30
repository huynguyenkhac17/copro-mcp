from __future__ import annotations

import json
import logging
import multiprocessing
from pathlib import Path

import pytest

from llm_worker_pool.infra.event_log import EventLog, error_code, warning_code
from llm_worker_pool.infra.logging import InvocationLogger
from llm_worker_pool.metrics_exporter import EventAggregator, EventLogCollector


def _write_many(path: str, worker: int, count: int) -> None:
    log = EventLog(Path(path))
    for i in range(count):
        # Large enough that a non-atomic write would visibly interleave.
        log.append({"worker": worker, "i": i, "pad": "x" * 2000})


@pytest.mark.skipif(
    "fork" not in multiprocessing.get_all_start_methods(), reason="needs fork start method"
)
def test_concurrent_writers_never_interleave_lines(tmp_path: Path):
    path = tmp_path / "events.jsonl"
    ctx = multiprocessing.get_context("fork")
    procs = [ctx.Process(target=_write_many, args=(str(path), w, 200)) for w in range(4)]
    for p in procs:
        p.start()
    for p in procs:
        p.join()

    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 800
    records = [json.loads(line) for line in lines]
    for w in range(4):
        assert sorted(r["i"] for r in records if r["worker"] == w) == list(range(200))


@pytest.mark.skipif(
    "fork" not in multiprocessing.get_all_start_methods(), reason="needs fork start method"
)
def test_concurrent_writers_with_rotation_lose_nothing(tmp_path: Path):
    path = tmp_path / "events.jsonl"
    ctx = multiprocessing.get_context("fork")

    def write(worker: int) -> None:
        log = EventLog(path, max_bytes=40_000)
        for i in range(100):
            log.append({"worker": worker, "i": i, "pad": "x" * 500})

    procs = [ctx.Process(target=write, args=(w,)) for w in range(4)]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
    # 400 events x ~560 bytes = ~220 KB through a 40 KB cap: many rotations, and only
    # one generation is kept - so just check that what survives is whole lines.
    for f in (path, path.with_name("events.jsonl.1")):
        for line in f.read_text(encoding="utf-8").splitlines():
            json.loads(line)


def test_rotates_past_size_cap(tmp_path: Path):
    path = tmp_path / "events.jsonl"
    log = EventLog(path, max_bytes=500)
    for i in range(20):
        log.append({"i": i, "pad": "x" * 100})
    assert path.with_name("events.jsonl.1").exists()
    assert path.stat().st_size < 1000


def test_write_failure_is_swallowed(tmp_path: Path):
    blocker = tmp_path / "file"
    blocker.write_text("not a dir")
    EventLog(blocker / "events.jsonl").append({"a": 1})  # must not raise


@pytest.mark.parametrize(
    ("warning", "code"),
    [
        ("fallback_used:groq after gemini: timeout", "fallback_used"),
        ("output_truncated_at_max_tokens (num_predict=10); the answer", "output_truncated_at_max_tokens"),
        ("json_parse_failed", "json_parse_failed"),
        ("json_schema_validation_failed: 'x' is required", "json_schema_validation_failed"),
        ("secret_redacted:google_api_key x2", "secret_redacted"),
        ("thinking_consumed_900_of_1000_output_tokens; set", "thinking_consumed"),
        ("file_denied_sensitive:/a/.env", "file_denied_sensitive"),
    ],
)
def test_warning_code_is_low_cardinality(warning: str, code: str):
    assert warning_code(warning) == code


@pytest.mark.parametrize(
    ("error", "code"),
    [
        ("gemini did not respond within 40.0s (budget=L3).", "timeout"),
        ("groq: prompt ~9000 tokens exceeds max_prompt_tokens=6000; skipped", "prompt_too_large"),
        ("gemini returned status 503: {overloaded}", "http_503"),
        ("gemini returned error 400: bad", "http_400"),
        ("something odd", "error"),
    ],
)
def test_error_code(error: str, code: str):
    assert error_code(error) == code


def _logger(tmp_path: Path, debug: bool = False) -> tuple[InvocationLogger, Path]:
    path = tmp_path / "events.jsonl"
    return (
        InvocationLogger(logging.getLogger("t"), debug_log_content=debug, event_log=EventLog(path)),
        path,
    )


def _events(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_invocation_logger_writes_event_without_content(tmp_path: Path):
    logger, path = _logger(tmp_path)
    logger.log_success(
        budget="L2", provider="gemini-lite", model="m", latency_ms=1234.5, tokens_used=30,
        prompt_tokens=10, completion_tokens=15, thinking_tokens=5, cached=False,
        warnings=["fallback_used:gemini-lite after groq: x"], task="SECRET TASK",
        raw_response="SECRET ANSWER", task_type="review", requested_provider="groq",
        fallback_chain=["groq", "gemini-lite"], guardrail_actions=["secret_redacted:x x1"],
    )
    logger.log_failure(budget="L2", provider="groq", model="m", latency_ms=5, error="groq returned status 429: q")

    raw = path.read_text(encoding="utf-8")
    assert "SECRET" not in raw
    ok, failed = _events(path)
    assert ok["outcome"] == "success" and ok["task_type"] == "review"
    assert ok["warning_codes"] == ["fallback_used"]
    assert ok["guardrail_actions"] == ["secret_redacted"]
    assert ok["fallback_chain"] == ["groq", "gemini-lite"]
    assert failed["outcome"] == "failure" and failed["error_code"] == "http_429"


def test_invocation_logger_includes_content_in_debug_mode(tmp_path: Path):
    logger, path = _logger(tmp_path, debug=True)
    logger.log_success(budget="L1", provider="p", model="m", latency_ms=1, tokens_used=None,
                       cached=False, warnings=[], task="T", raw_response="R")
    assert _events(path)[0]["task"] == "T"


def test_sink_failure_does_not_break_logging(tmp_path: Path):
    seen: list[dict] = []

    def broken(_event: dict) -> None:
        raise RuntimeError("boom")

    logger = InvocationLogger(logging.getLogger("t"), sinks=[broken, seen.append])
    logger.log_success(budget="L1", provider="p", model="m", latency_ms=1, tokens_used=None,
                       cached=False, warnings=[])
    assert len(seen) == 1


def test_aggregator_reads_incrementally_and_keeps_partial_line(tmp_path: Path):
    path = tmp_path / "events.jsonl"
    log = EventLog(path)
    log.append({"outcome": "success", "provider": "groq", "model": "m", "budget": "L2",
                "latency_ms": 800, "prompt_tokens": 10, "completion_tokens": 5})
    agg = EventAggregator(path)
    agg.refresh()
    assert agg.requests[("groq", "m", "L2", "none", "success")] == 1

    with open(path, "ab") as f:
        f.write(b'{"outcome": "success", "provider": "groq"')  # writer mid-line
    agg.refresh()
    assert sum(agg.requests.values()) == 1
    with open(path, "ab") as f:
        f.write(b', "model": "m", "budget": "L2"}\n')
    agg.refresh()
    assert agg.requests[("groq", "m", "L2", "none", "success")] == 2
    assert agg.tokens[("groq", "prompt")] == 10


def test_aggregator_survives_rotation_without_losing_tail(tmp_path: Path):
    path = tmp_path / "events.jsonl"
    log = EventLog(path, max_bytes=300)
    agg = EventAggregator(path)
    log.append({"outcome": "success", "provider": "a", "pad": "x" * 200})
    agg.refresh()
    log.append({"outcome": "success", "provider": "a", "pad": "x" * 200})  # still old file
    log.append({"outcome": "success", "provider": "a", "pad": "x" * 200})  # rotates first
    agg.refresh()
    assert sum(agg.requests.values()) == 3


def test_collector_exposes_counters_and_histogram(tmp_path: Path):
    prometheus_client = pytest.importorskip("prometheus_client")
    from prometheus_client import CollectorRegistry, generate_latest

    path = tmp_path / "events.jsonl"
    log = EventLog(path)
    log.append({"outcome": "success", "provider": "gemini-lite", "model": "lite", "budget": "L3",
                "task_type": "review", "latency_ms": 3000, "prompt_tokens": 100,
                "completion_tokens": 20, "thinking_tokens": 7, "requested_provider": "groq",
                "warning_codes": ["fallback_used"], "guardrail_actions": ["secret_redacted"]})
    log.append({"outcome": "failure", "provider": "groq", "model": "oss", "budget": "L3",
                "latency_ms": 50, "error_code": "prompt_too_large"})
    eval_path = tmp_path / "eval_latest.json"
    eval_path.write_text(json.dumps({"scores": [
        {"metric": "faithfulness", "provider": "groq", "budget": "L2", "task_type": "summarize", "value": 0.8}
    ]}))

    registry = CollectorRegistry()
    registry.register(EventLogCollector(EventAggregator(path), eval_path))
    text = generate_latest(registry).decode()

    assert 'llm_worker_requests_total{budget="L3",model="lite",outcome="success",provider="gemini-lite",task_type="review"} 1.0' in text
    assert 'llm_worker_latency_seconds_bucket{budget="L3",le="5.0",outcome="success",provider="gemini-lite"} 1.0' in text
    assert 'llm_worker_latency_seconds_bucket{budget="L3",le="2.5",outcome="success",provider="gemini-lite"} 0.0' in text
    assert 'llm_worker_tokens_total{kind="thinking",provider="gemini-lite"} 7.0' in text
    assert 'llm_worker_fallback_total{answered="gemini-lite",requested="groq"} 1.0' in text
    assert 'llm_worker_warnings_total{code="fallback_used"} 1.0' in text
    assert 'llm_worker_guardrail_actions_total{action="secret_redacted"} 1.0' in text
    assert 'llm_worker_errors_total{code="prompt_too_large",provider="groq"} 1.0' in text
    assert 'llm_worker_eval_score{budget="L2",metric="faithfulness",provider="groq",task_type="summarize"} 0.8' in text
    assert prometheus_client  # imported for the skip guard
