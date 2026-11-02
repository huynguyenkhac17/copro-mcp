"""Circuit breaker tracking health per API key.

When an API key encounters repeated errors (429 rate limit, 403 quota/denied,
503 overload, network errors), the key is quarantined for a configurable duration.
During quarantine, round-robin key selection skips the quarantined key.
When the quarantine period expires, the key is automatically reintroduced (half-open).

With a SharedKeyState, a quarantine also reaches the other server processes on the
machine (infra/shared_key_state.py); a success on the key lifts it in the file again.
Failure counts stay per process.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from llm_worker_pool.infra.shared_key_state import SharedKeyState, key_id


@dataclass
class KeyState:
    consecutive_failures: int = 0
    quarantined_until: float = 0.0
    total_quarantines: int = 0
    total_failures: int = 0
    total_successes: int = 0


class KeyCircuitBreaker:
    def __init__(
        self,
        failure_threshold: int = 3,
        quarantine_seconds: float = 60.0,
        enabled: bool = True,
        *,
        clock: Callable[[], float] = time.monotonic,
        shared: SharedKeyState | None = None,
        scope: str = "",
    ) -> None:
        self.failure_threshold = failure_threshold
        self.quarantine_seconds = quarantine_seconds
        self.enabled = enabled
        self._clock = clock
        self._states: dict[str, KeyState] = {}
        self._shared = shared
        self._scope = scope
        self._key_by_id: dict[str, str] = {}
        self._applied_generation = -1

    def _get_state(self, key: str) -> KeyState:
        if key not in self._states:
            self._states[key] = KeyState()
            self._key_by_id[key_id(key)] = key
            self._applied_generation = -1
        return self._states[key]

    def _sync(self) -> None:
        """Adopt quarantines other processes published (only ever lengthening ours)."""
        if self._shared is None:
            return
        section = self._shared.provider(self._scope)
        if self._shared.generation == self._applied_generation:
            return
        self._applied_generation = self._shared.generation
        offset = self._clock() - self._shared.now()
        for kid, entry in section.get("keys", {}).items():
            key = self._key_by_id.get(kid)
            until = entry.get("quarantined_until")
            if key is None or until is None:
                continue
            state = self._states[key]
            state.quarantined_until = max(state.quarantined_until, until + offset)

    def _publish(self, key: str, until: float | None) -> None:
        """Write this key's quarantine end (local clock), or lift it with None."""
        if self._shared is None:
            return
        kid = key_id(key)
        wall = None if until is None else until - self._clock() + self._shared.now()

        def mutate(section: dict[str, Any]) -> None:
            entry = section.setdefault("keys", {}).setdefault(kid, {})
            if wall is None:
                entry.pop("quarantined_until", None)
            else:
                entry["quarantined_until"] = max(wall, entry.get("quarantined_until", 0.0))

        self._shared.update(self._scope, mutate)

    def _shared_quarantine_on(self, key: str) -> bool:
        if self._shared is None:
            return False
        entry = self._shared.provider(self._scope).get("keys", {}).get(key_id(key), {})
        return entry.get("quarantined_until", 0.0) > self._shared.now()

    def is_available(self, key: str) -> bool:
        if not self.enabled:
            return True
        state = self._get_state(key)
        self._sync()
        now = self._clock()
        if state.quarantined_until > 0 and now < state.quarantined_until:
            return False
        return True

    def record_success(self, key: str) -> None:
        state = self._get_state(key)
        state.consecutive_failures = 0
        state.quarantined_until = 0.0
        state.total_successes += 1
        # Only when there is something to lift: a success is the common case, and it
        # should not cost a file write.
        if self._shared_quarantine_on(key):
            self._publish(key, None)

    def record_failure(self, key: str, status_code: int | None = None) -> bool:
        """Record a failure for the key. If status_code is given, only quarantine-worthy
        status codes (403, 429, 500, 502, 503, 504, or None for connection failure) increment failures.
        Returns True if the key was newly quarantined.
        """
        if not self.enabled:
            return False

        if status_code is not None and status_code not in (403, 429, 500, 502, 503, 504):
            return False

        state = self._get_state(key)
        state.consecutive_failures += 1
        state.total_failures += 1

        if state.consecutive_failures >= self.failure_threshold:
            now = self._clock()
            state.quarantined_until = now + self.quarantine_seconds
            state.total_quarantines += 1
            self._publish(key, state.quarantined_until)
            return True
        return False

    def filter_available_keys(self, keys: list[str]) -> list[str]:
        """Return available keys. If all keys are quarantined, returns the key
        closest to expiring its quarantine (so requests do not completely lock up).
        """
        if not self.enabled or not keys:
            return list(keys)

        available = [k for k in keys if self.is_available(k)]
        if available:
            return available

        now = self._clock()
        earliest_key = min(keys, key=lambda k: self._get_state(k).quarantined_until)
        return [earliest_key]

    def next_available_in(self, keys: list[str]) -> float:
        """Seconds until the first of `keys` leaves quarantine (0 = one is free now)."""
        if not self.enabled or not keys or any(self.is_available(k) for k in keys):
            return 0.0
        earliest = min(self._get_state(k).quarantined_until for k in keys)
        return max(0.0, earliest - self._clock())

    def stats_by_position(self, keys: list[str]) -> dict[str, dict[str, Any]]:
        """Stats for every configured key, labelled `key_<i>` by its position in
        `keys` - the keys themselves are secrets. A key never used yet reports zeros.
        `quarantined_until` is wall-clock (epoch seconds, 0 = not quarantined) so a
        reader of the on-disk snapshot can tell whether a quarantine is still on."""
        for key in keys:
            self._get_state(key)
        self._sync()
        now = self._clock()
        wall_offset = time.time() - now
        out: dict[str, dict[str, Any]] = {}
        for i, key in enumerate(keys):
            st = self._states.get(key) or KeyState()
            quarantined = st.quarantined_until > now
            out[f"key_{i}"] = {
                "consecutive_failures": st.consecutive_failures,
                "is_quarantined": quarantined,
                "quarantine_remaining_seconds": max(0.0, st.quarantined_until - now),
                "quarantined_until": st.quarantined_until + wall_offset if quarantined else 0.0,
                "total_quarantines": st.total_quarantines,
                "total_failures": st.total_failures,
                "total_successes": st.total_successes,
            }
        return out

    def get_stats(self) -> dict[str, Any]:
        now = self._clock()
        return {
            key: {
                "consecutive_failures": st.consecutive_failures,
                "is_quarantined": st.quarantined_until > now,
                "quarantine_remaining_seconds": max(0.0, st.quarantined_until - now),
                "total_quarantines": st.total_quarantines,
                "total_failures": st.total_failures,
                "total_successes": st.total_successes,
            }
            for key, st in self._states.items()
        }
