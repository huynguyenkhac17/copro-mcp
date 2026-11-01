from __future__ import annotations

import pytest

from llm_worker_pool.infra.quota_tracker import QuotaTracker, parse_duration


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def _tracker(**kwargs) -> tuple[QuotaTracker, FakeClock]:
    clock = FakeClock()
    return QuotaTracker(clock=clock, **kwargs), clock


# Groq headers as captured from a real 200 response on 2026-10-30.
_GROQ_HEADERS = {
    "x-ratelimit-limit-requests": "1000",
    "x-ratelimit-limit-tokens": "8000",
    "x-ratelimit-remaining-requests": "999",
    "x-ratelimit-remaining-tokens": "7923",
    "x-ratelimit-reset-requests": "1m26.4s",
    "x-ratelimit-reset-tokens": "577ms",
}

# Shape of Google's 429 body (google.rpc error details); the live probe on 2026-10-30
# only produced 200/503/hangs, so this follows the documented format.
_GEMINI_429_PER_MINUTE = """[{
  "error": {
    "code": 429,
    "message": "You exceeded your current quota. Please retry in 37.5s.",
    "details": [
      {"@type": "type.googleapis.com/google.rpc.QuotaFailure",
       "violations": [{"quotaId": "GenerateRequestsPerMinutePerProjectPerModel-FreeTier"}]},
      {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "37s"}
    ]
  }
}]"""
_GEMINI_429_PER_DAY = _GEMINI_429_PER_MINUTE.replace("PerMinute", "PerDay")


@pytest.mark.parametrize(
    ("value", "expected"),
    [("577ms", 0.577), ("7.66s", 7.66), ("1m26.4s", 86.4), ("2h3m", 7380.0), ("30", 30.0)],
)
def test_parse_duration(value: str, expected: float):
    assert parse_duration(value) == pytest.approx(expected)


@pytest.mark.parametrize("value", [None, "", "soon", "5x", "1m garbage"])
def test_parse_duration_rejects_garbage(value):
    assert parse_duration(value) is None


def test_unknown_key_is_unrestricted():
    tracker, _ = _tracker()
    assert tracker.can_serve("k", 10_000_000)


def test_remaining_tokens_from_headers_gate_large_requests():
    tracker, _ = _tracker()
    tracker.observe_headers("k", _GROQ_HEADERS)
    assert tracker.can_serve("k", 7000)
    assert not tracker.can_serve("k", 8000)


def test_window_refills_to_limit_after_reset():
    tracker, clock = _tracker()
    tracker.observe_headers("k", {**_GROQ_HEADERS, "x-ratelimit-remaining-tokens": "100"})
    assert not tracker.can_serve("k", 5000)
    clock.now += 1.0  # past the 577ms reset
    assert tracker.can_serve("k", 5000)
    assert not tracker.can_serve("k", 9000)  # still bounded by the 8000 limit


def test_reservations_stop_a_concurrent_burst():
    tracker, _ = _tracker()
    tracker.observe_headers("k", _GROQ_HEADERS)
    tracker.reserve("k", 5000)
    assert not tracker.can_serve("k", 5000)
    tracker.release("k", 5000)
    assert tracker.can_serve("k", 5000)


def test_safety_margin_is_kept():
    tracker, _ = _tracker(safety_margin_tokens=500)
    tracker.observe_headers("k", _GROQ_HEADERS)
    assert not tracker.can_serve("k", 7500)


def test_exhausted_requests_block_until_reset():
    tracker, clock = _tracker()
    tracker.observe_headers("k", {"x-ratelimit-remaining-requests": "0", "x-ratelimit-reset-requests": "10s"})
    assert not tracker.can_serve("k", 1)
    clock.now += 11
    assert tracker.can_serve("k", 1)


def test_retry_delay_in_body_blocks_key_exactly_that_long():
    tracker, clock = _tracker()
    assert tracker.observe_rate_limited("k", {}, _GEMINI_429_PER_MINUTE) == 37.0
    assert not tracker.can_serve("k", 1)
    assert tracker.can_serve("other", 1)
    clock.now += 38
    assert tracker.can_serve("k", 1)


def test_retry_after_header_wins():
    tracker, _ = _tracker()
    assert tracker.observe_rate_limited("k", {"retry-after": "5"}, _GEMINI_429_PER_MINUTE) == 5.0


def test_retry_in_message_is_used_without_retry_info():
    tracker, _ = _tracker()
    body = '{"error": {"message": "Rate limit reached. Please retry in 750ms."}}'
    assert tracker.observe_rate_limited("k", {}, body) == pytest.approx(0.75)


def test_per_day_quota_parks_key_long():
    tracker, _ = _tracker(daily_quota_block_seconds=3600)
    assert tracker.observe_rate_limited("k", {}, _GEMINI_429_PER_DAY) == 3600


def test_429_without_hint_leaves_key_alone():
    tracker, _ = _tracker()
    assert tracker.observe_rate_limited("k", {}, "quota exceeded") is None
    assert tracker.can_serve("k", 1)


def test_repeated_timeouts_block_the_model_and_success_resets():
    tracker, clock = _tracker(timeouts_before_block=2, timeout_block_seconds=120)
    assert tracker.record_timeout() is False
    assert tracker.model_blocked_for() == 0
    assert tracker.record_timeout() is True
    assert tracker.model_blocked_for() == pytest.approx(120)

    clock.now += 121
    assert tracker.model_blocked_for() == 0
    # Still failing after the block: the very next hang re-blocks at once.
    assert tracker.record_timeout() is True

    tracker.record_success()
    clock.now += 121
    assert tracker.record_timeout() is False


def test_disabled_tracker_never_restricts():
    tracker, _ = _tracker(enabled=False, timeouts_before_block=1)
    tracker.observe_rate_limited("k", {}, _GEMINI_429_PER_DAY)
    tracker.record_timeout()
    assert tracker.can_serve("k", 10**9)
    assert tracker.model_blocked_for() == 0


def test_stats_never_contain_the_key():
    tracker, _ = _tracker()
    secret = "gsk_secret_value"
    tracker.observe_headers(secret, _GROQ_HEADERS)
    stats = tracker.get_stats([secret])
    assert secret not in repr(stats)
    assert stats["keys"][0]["key_index"] == 0
    assert stats["keys"][0]["remaining_tokens"] == 7923


def test_zero_token_estimate_still_needs_some_quota():
    tracker, _ = _tracker()
    tracker.observe_headers("k", {"x-ratelimit-remaining-tokens": "0", "x-ratelimit-reset-tokens": "30s"})
    assert not tracker.can_serve("k", 0)


def test_retry_after_http_date():
    from email.utils import format_datetime
    from datetime import datetime, timedelta, timezone

    from llm_worker_pool.infra.quota_tracker import parse_retry_after

    when = format_datetime(datetime.now(timezone.utc) + timedelta(seconds=90), usegmt=True)
    assert parse_retry_after(when) == pytest.approx(90, abs=2)
    assert parse_retry_after("Wed, 21 Oct 2015 07:28:00 GMT") == 0.0  # in the past
    assert parse_retry_after("not a date") is None


def test_rate_limit_returns_effective_block_not_raw_delay():
    tracker, _ = _tracker()
    tracker.observe_rate_limited("k", {}, '{"retryDelay": "300s"}')
    # A shorter hint later does not shorten the existing block - and says so.
    assert tracker.observe_rate_limited("k", {"retry-after": "5"}, "") == pytest.approx(300)


def test_wait_until_servable_is_zero_while_one_key_is_free():
    tracker, _ = _tracker()
    tracker.observe_rate_limited("a", {"retry-after": "30"}, "")
    assert tracker.wait_until_servable(["a", "b"], 100) == 0.0


def test_wait_until_servable_takes_the_earliest_unblocked_key():
    tracker, clock = _tracker()
    tracker.observe_rate_limited("a", {"retry-after": "30"}, "")
    tracker.observe_rate_limited("b", {"retry-after": "12"}, "")
    assert tracker.wait_until_servable(["a", "b"], 100) == pytest.approx(12)
    clock.now += 12
    assert tracker.wait_until_servable(["a", "b"], 100) == 0.0


def test_wait_until_servable_counts_a_key_drained_by_headers():
    tracker, _ = _tracker()
    tracker.observe_headers(
        "a", {"x-ratelimit-remaining-tokens": "50", "x-ratelimit-reset-tokens": "9s"}
    )
    # No 429 block, but no tokens left until the window resets.
    assert tracker.wait_until_servable(["a"], 100) == pytest.approx(9)


def test_wait_until_servable_none_without_a_timed_block():
    tracker, clock = _tracker()
    tracker.observe_headers(
        "a",
        {
            "x-ratelimit-limit-tokens": "500",
            "x-ratelimit-remaining-tokens": "500",
            "x-ratelimit-reset-tokens": "1s",
        },
    )
    clock.now += 2  # window already refilled; only the reservation holds the key back
    tracker.reserve("a", 450)
    assert tracker.wait_until_servable(["a"], 100) is None
