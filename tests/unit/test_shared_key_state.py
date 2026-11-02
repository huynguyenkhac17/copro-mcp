from __future__ import annotations

import json

import pytest

from copro.config.settings import AppConfig
from copro.infra.circuit_breaker import KeyCircuitBreaker
from copro.infra.quota_tracker import QuotaTracker
from copro.infra.shared_key_state import SharedKeyState, key_id
from copro.providers.registry import ProviderFactory


class Clock:
    def __init__(self, now: float) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


class Machine:
    """One wall clock; each "process" has its own monotonic clock with another origin,
    as real processes would after a reboot."""

    def __init__(self, path) -> None:
        self.path = path
        self.wall = Clock(1_700_000_000.0)
        self.processes: list[Clock] = []

    def process(self, origin: float) -> tuple[SharedKeyState, Clock]:
        clock = Clock(origin)
        self.processes.append(clock)
        return SharedKeyState(self.path, min_refresh_seconds=0, wall=self.wall), clock

    def tracker(self, origin: float, **kwargs) -> QuotaTracker:
        shared, clock = self.process(origin)
        return QuotaTracker(clock=clock, shared=shared, scope="groq", **kwargs)

    def breaker(self, origin: float) -> KeyCircuitBreaker:
        shared, clock = self.process(origin)
        return KeyCircuitBreaker(failure_threshold=2, quarantine_seconds=60, clock=clock, shared=shared, scope="groq")

    def advance(self, seconds: float) -> None:
        self.wall.now += seconds
        for clock in self.processes:
            clock.now += seconds


@pytest.fixture
def machine(tmp_path) -> Machine:
    return Machine(tmp_path / "key_state.json")


def test_a_429_seen_by_one_process_blocks_the_key_in_another(machine):
    a = machine.tracker(100.0)
    b = machine.tracker(987_654.0)  # unrelated monotonic origin
    assert b.can_serve("key-1", 100)

    a.observe_rate_limited("key-1", {"retry-after": "30"}, "")

    assert not b.can_serve("key-1", 100)
    assert b.wait_until_servable(["key-1"], 100) == pytest.approx(30)
    assert b.can_serve("key-2", 100)  # other keys untouched
    machine.advance(31)
    assert b.can_serve("key-1", 100)


def test_newest_rate_limit_headers_win_across_processes(machine):
    a = machine.tracker(0.0)
    b = machine.tracker(5_000.0)
    a.observe_headers(
        "key-1",
        {"x-ratelimit-limit-tokens": "8000", "x-ratelimit-remaining-tokens": "100", "x-ratelimit-reset-tokens": "20s"},
    )
    assert not b.can_serve("key-1", 1000)
    assert b.wait_until_servable(["key-1"], 1000) == pytest.approx(20)

    machine.advance(1)
    b.observe_headers("key-1", {"x-ratelimit-remaining-tokens": "7000", "x-ratelimit-reset-tokens": "20s"})
    assert a.can_serve("key-1", 1000)


def test_an_older_observation_does_not_overwrite_a_newer_one(machine):
    a = machine.tracker(0.0)
    b = machine.tracker(0.0)
    machine.advance(1)
    b.observe_headers("key-1", {"x-ratelimit-remaining-tokens": "50", "x-ratelimit-reset-tokens": "30s"})
    # a's own observation is older than what the file holds: a must not publish it.
    a._state("key-1").tokens_observed_at = -10  # noqa: SLF001
    a._publish_key("key-1", {"remaining_tokens": 9999, "tokens_observed_at": a._to_wall(-10)})  # noqa: SLF001
    entry = json.loads(machine.path.read_text())["providers"]["groq"]["keys"][key_id("key-1")]
    assert entry["remaining_tokens"] == 50


def test_model_hang_block_is_shared(machine):
    a = machine.tracker(0.0, timeouts_before_block=1, timeout_block_seconds=120)
    b = machine.tracker(42.0)
    assert a.record_timeout()
    assert b.model_blocked_for() == pytest.approx(120)


def test_breaker_quarantine_is_shared_and_a_success_lifts_it_in_the_file(machine):
    a = machine.breaker(0.0)
    b = machine.breaker(777.0)
    a.record_failure("key-1", 429)
    assert b.is_available("key-1")  # one failure: no quarantine yet, counts are local
    assert a.record_failure("key-1", 429)
    assert not b.is_available("key-1")

    a.record_success("key-1")
    entry = json.loads(machine.path.read_text())["providers"].get("groq", {}).get("keys", {})
    assert "quarantined_until" not in entry.get(key_id("key-1"), {})
    assert machine.breaker(5.0).is_available("key-1")  # a process started now


def test_file_never_holds_a_key_in_clear_and_drops_expired_entries(machine):
    a = machine.tracker(0.0)
    a.observe_rate_limited("gsk_secret_value", {"retry-after": "5"}, "")
    text = machine.path.read_text()
    assert "gsk_secret_value" not in text and key_id("gsk_secret_value") in text

    machine.advance(6)
    a.observe_rate_limited("other", {"retry-after": "5"}, "")
    keys = json.loads(machine.path.read_text())["providers"]["groq"]["keys"]
    assert list(keys) == [key_id("other")]


def test_an_unwritable_file_never_breaks_a_call(tmp_path):
    blocker = tmp_path / "not_a_dir"
    blocker.write_text("x")
    tracker = QuotaTracker(shared=SharedKeyState(blocker / "key_state.json"), scope="groq")
    assert tracker.observe_rate_limited("k", {"retry-after": "5"}, "") == pytest.approx(5)
    assert not tracker.can_serve("k", 1)


def test_factory_shares_one_file_unless_disabled(tmp_path):
    config = AppConfig()
    config.quota.shared_state_path = str(tmp_path / "state.json")
    assert ProviderFactory(config)._shared is not None  # noqa: SLF001
    config.quota.shared_state = False
    assert ProviderFactory(config)._shared is None  # noqa: SLF001


def test_same_size_rewrites_in_quick_succession_are_both_seen(machine):
    a = machine.tracker(0.0)
    b = machine.tracker(0.0)
    a.observe_rate_limited("key-1", {"retry-after": "11"}, "")
    assert b.wait_until_servable(["key-1"], 1) == pytest.approx(11)
    a.observe_rate_limited("key-1", {"retry-after": "19"}, "")  # same length on disk
    assert b.wait_until_servable(["key-1"], 1) == pytest.approx(19)


def test_a_key_first_seen_after_a_sync_still_gets_its_shared_entry(machine):
    a = machine.tracker(0.0)
    b = machine.tracker(0.0)
    a.observe_rate_limited("key-1", {"retry-after": "30"}, "")
    a.observe_rate_limited("key-3", {"retry-after": "30"}, "")
    assert not b.can_serve("key-1", 1)  # b synced, file unchanged since
    assert not b.can_serve("key-3", 1)  # first look at key-3: merged all the same


@pytest.mark.parametrize(
    "content",
    [
        '{"version": 1, "providers": {"groq": {"keys": []}}}',
        '{"version": 1, "providers": {"groq": {"keys": {"x": "y"}, "model_blocked_until": "soon"}}}',
        '{"version": 1, "providers": []}',
        '{"version": 1, "providers": {"groq": {"keys": {"%s": {"blocked_until": "later"}}}}}' % key_id("key-1"),
        "not json",
    ],
)
def test_a_malformed_file_is_ignored_on_read_and_replaced_on_write(machine, content):
    machine.path.write_text(content)
    a = machine.tracker(0.0)
    breaker = machine.breaker(0.0)
    assert a.can_serve("key-1", 1)
    assert a.model_blocked_for() == 0
    assert breaker.is_available("key-1")

    a.observe_rate_limited("key-1", {"retry-after": "30"}, "")
    assert not machine.tracker(0.0).can_serve("key-1", 1)


def test_a_held_lock_skips_the_write_quickly_instead_of_stalling(tmp_path):
    # 2026-11-02: msvcrt.locking(LK_LOCK) slept a full second per retry on the event
    # loop; three processes writing at once stalled a call for 1-7 s.
    import time

    from copro.infra import shared_key_state as sks

    path = tmp_path / "key_state.json"
    state = SharedKeyState(path)
    with open(path.with_name(path.name + ".lock"), "a+") as holder:
        sks._lock(holder)  # another "process" holds the lock
        started = time.monotonic()
        state.update("groq", lambda section: section.setdefault("model_blocked_until", time.time() + 60))
        elapsed = time.monotonic() - started
    assert elapsed < 0.9
    assert not path.exists()  # the write was skipped, not forced through

    state.update("groq", lambda section: section.setdefault("model_blocked_until", time.time() + 60))
    assert json.loads(path.read_text())["providers"]["groq"]["model_blocked_until"] > 0
