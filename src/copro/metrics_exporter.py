"""`copro metrics`: serves http://127.0.0.1:9464/metrics from the shared event log.

A separate process on purpose. The MCP server speaks stdio and often runs as several
instances at once (Claude Code, Gemini CLI...): an HTTP endpoint inside each would fight
over the port and only see its own share of the traffic. Every instance appends to
events.jsonl instead (infra/event_log.py), and this exporter folds the file into
counters, reading only the bytes added since the previous scrape.

Needs the [metrics] extra (prometheus_client); the aggregation itself does not, so it
is unit-testable without it.
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterator

LATENCY_BUCKETS = (0.5, 1.0, 2.5, 5.0, 10.0, 20.0, 40.0, 60.0, 120.0, 300.0, float("inf"))
TOKEN_KINDS = ("prompt", "completion", "thinking")


class EventAggregator:
    """Running totals over the event log. Counters only grow: a rotated file is
    finished from its `.1` copy, then the new file is read from the start."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._offset = 0
        self._inode: int | None = None
        self._lock = threading.Lock()
        self.requests: dict[tuple[str, ...], int] = defaultdict(int)
        self.latency_buckets: dict[tuple[str, ...], list[int]] = {}
        self.latency_sum: dict[tuple[str, ...], float] = defaultdict(float)
        self.latency_count: dict[tuple[str, ...], int] = defaultdict(int)
        self.tokens: dict[tuple[str, str], int] = defaultdict(int)
        self.fallbacks: dict[tuple[str, str], int] = defaultdict(int)
        self.warnings: dict[str, int] = defaultdict(int)
        self.guardrail_actions: dict[str, int] = defaultdict(int)
        self.errors: dict[tuple[str, str], int] = defaultdict(int)
        self.malformed_lines = 0

    def refresh(self) -> None:
        with self._lock:
            try:
                stat = self._path.stat()
            except FileNotFoundError:
                return
            if self._inode is not None and stat.st_ino != self._inode:
                self._finish_rotated()
                self._offset = 0
            elif stat.st_size < self._offset:
                self._offset = 0  # truncated in place
            self._inode = stat.st_ino
            self._offset = self._consume(self._path, self._offset)

    def _finish_rotated(self) -> None:
        rotated = self._path.with_name(self._path.name + ".1")
        try:
            if rotated.stat().st_ino == self._inode:
                self._consume(rotated, self._offset)
        except FileNotFoundError:
            pass

    def _consume(self, path: Path, offset: int) -> int:
        """Ingest complete lines after `offset`; return the offset after the last one.
        A half-written trailing line is left for the next scrape."""
        try:
            with open(path, "rb") as f:
                f.seek(offset)
                data = f.read()
        except OSError:
            return offset
        end = data.rfind(b"\n")
        if end < 0:
            return offset
        for line in data[: end + 1].splitlines():
            if not line.strip():
                continue
            try:
                self.ingest(json.loads(line))
            except (ValueError, TypeError, AttributeError):
                self.malformed_lines += 1
        return offset + end + 1

    def ingest(self, event: dict[str, Any]) -> None:
        provider = str(event.get("provider") or "unknown")
        budget = str(event.get("budget") or "")
        task_type = str(event.get("task_type") or "none")
        outcome = str(event.get("outcome") or "unknown")
        model = str(event.get("model") or "")

        self.requests[(provider, model, budget, task_type, outcome)] += 1

        latency = event.get("latency_ms")
        if isinstance(latency, (int, float)):
            key = (provider, budget, outcome)
            seconds = latency / 1000
            buckets = self.latency_buckets.setdefault(key, [0] * len(LATENCY_BUCKETS))
            for i, bound in enumerate(LATENCY_BUCKETS):
                if seconds <= bound:
                    buckets[i] += 1
            self.latency_sum[key] += seconds
            self.latency_count[key] += 1

        if outcome == "failure":
            self.errors[(provider, str(event.get("error_code") or "error"))] += 1
            return

        for kind in TOKEN_KINDS:
            value = event.get(f"{kind}_tokens")
            if isinstance(value, int) and value > 0:
                self.tokens[(provider, kind)] += value

        requested = str(event.get("requested_provider") or provider)
        if requested != provider:
            self.fallbacks[(requested, provider)] += 1

        for code in event.get("warning_codes") or []:
            self.warnings[str(code)] += 1
        for action in event.get("guardrail_actions") or []:
            self.guardrail_actions[str(action)] += 1


def load_eval_scores(path: Path) -> list[dict[str, Any]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    scores = data.get("scores") if isinstance(data, dict) else None
    return [s for s in scores or [] if isinstance(s, dict) and isinstance(s.get("value"), (int, float))]


class EventLogCollector:
    """prometheus_client custom collector: refreshes the aggregator on every scrape."""

    def __init__(self, aggregator: EventAggregator, eval_path: Path) -> None:
        self._agg = aggregator
        self._eval_path = eval_path

    def collect(self) -> Iterator[Any]:
        from prometheus_client.core import (
            CounterMetricFamily,
            GaugeMetricFamily,
            HistogramMetricFamily,
        )

        agg = self._agg
        agg.refresh()

        requests = CounterMetricFamily(
            "copro_requests",
            "delegate_task invocations (failure = one failed tier)",
            labels=["provider", "model", "budget", "task_type", "outcome"],
        )
        for labels, value in agg.requests.items():
            requests.add_metric(list(labels), value)
        yield requests

        latency = HistogramMetricFamily(
            "copro_latency_seconds",
            "Provider latency per call",
            labels=["provider", "budget", "outcome"],
        )
        for labels, buckets in agg.latency_buckets.items():
            pairs = [
                ("+Inf" if bound == float("inf") else str(bound), count)
                for bound, count in zip(LATENCY_BUCKETS, buckets)
            ]
            latency.add_metric(list(labels), pairs, agg.latency_sum[labels])
        yield latency

        tokens = CounterMetricFamily(
            "copro_tokens", "Tokens billed by kind", labels=["provider", "kind"]
        )
        for labels, value in agg.tokens.items():
            tokens.add_metric(list(labels), value)
        yield tokens

        fallbacks = CounterMetricFamily(
            "copro_fallback",
            "Answers given by a fallback tier instead of the requested provider",
            labels=["requested", "answered"],
        )
        for labels, value in agg.fallbacks.items():
            fallbacks.add_metric(list(labels), value)
        yield fallbacks

        errors = CounterMetricFamily(
            "copro_errors", "Failed tiers by error class", labels=["provider", "code"]
        )
        for labels, value in agg.errors.items():
            errors.add_metric(list(labels), value)
        yield errors

        warnings = CounterMetricFamily(
            "copro_warnings", "Warnings attached to results", labels=["code"]
        )
        for code, value in agg.warnings.items():
            warnings.add_metric([code], value)
        yield warnings

        guardrails = CounterMetricFamily(
            "copro_guardrail_actions", "Guardrail interventions", labels=["action"]
        )
        for action, value in agg.guardrail_actions.items():
            guardrails.add_metric([action], value)
        yield guardrails

        malformed = CounterMetricFamily(
            "copro_event_log_malformed_lines", "Event log lines that failed to parse"
        )
        malformed.add_metric([], agg.malformed_lines)
        yield malformed

        scores = GaugeMetricFamily(
            "copro_eval_score",
            "Latest copro eval score",
            labels=["metric", "provider", "budget", "task_type"],
        )
        for s in load_eval_scores(self._eval_path):
            scores.add_metric(
                [str(s.get(k) or "") for k in ("metric", "provider", "budget", "task_type")],
                float(s["value"]),
            )
        yield scores


def main(argv: list[str] | None = None) -> None:
    from prometheus_client import start_http_server
    from prometheus_client.core import REGISTRY

    from copro.config.settings import AppConfig
    from copro.infra.event_log import (
        configured_path,
        default_eval_latest_path,
        default_event_log_path,
    )

    config = AppConfig().metrics
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--host", default=config.exporter_host)
    parser.add_argument("--port", type=int, default=config.exporter_port)
    parser.add_argument(
        "--event-log",
        type=Path,
        default=configured_path(config.event_log_path, default_event_log_path()),
    )
    parser.add_argument(
        "--eval-latest",
        type=Path,
        default=configured_path(config.eval_latest_path, default_eval_latest_path()),
    )
    args = parser.parse_args(argv)

    REGISTRY.register(EventLogCollector(EventAggregator(args.event_log), args.eval_latest))
    start_http_server(args.port, addr=args.host)
    print(f"serving http://{args.host}:{args.port}/metrics from {args.event_log}", flush=True)
    while True:
        time.sleep(3600)


if __name__ == "__main__":
    main()
