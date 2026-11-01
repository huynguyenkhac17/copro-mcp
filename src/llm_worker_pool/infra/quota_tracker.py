"""Per-key quota bookkeeping, so a provider can be skipped BEFORE it answers 429.

The circuit breaker (infra/circuit_breaker.py) only reacts after the fact: three
consecutive failures, then a fixed 60s quarantine. Two things were measured to go
wrong with that alone (2026-10-30):

- Groq reports its remaining quota on every response (`x-ratelimit-remaining-tokens`
  = 8K tokens/minute on the free tier) but nobody read it, so concurrent requests all
  went out and collected 429s.
- gemini-flash-latest mostly does not fail at all - it accepts the call and hangs. The
  tier deadline cancels the request (asyncio.wait_for), which the provider never saw as
  a failure, so every L3 request waited the full `tier_timeout_seconds` on flash again.

Gemini's OpenAI shim sends no rate-limit headers (checked on 200 and 503 responses), so
for Gemini the only quota signal is the body of a 429 (`retryDelay`, and a quotaId
naming a per-day limit).

One tracker per provider, like the breaker: quotas and overload are per model, and
gemini / gemini-lite share the same keys.
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any

_DURATION_PART_RE = re.compile(r"(\d+(?:\.\d+)?)(ms|h|m|s)")
_RETRY_DELAY_RE = re.compile(r'"retryDelay"\s*:\s*"(\d+(?:\.\d+)?)s"')
_RETRY_IN_RE = re.compile(r"retry in (\d+(?:\.\d+)?)\s*(ms|s)\b", re.IGNORECASE)
_PER_DAY_RE = re.compile(r"PerDay", re.IGNORECASE)

_UNIT_SECONDS = {"ms": 0.001, "s": 1.0, "m": 60.0, "h": 3600.0}


def parse_duration(value: str | None) -> float | None:
    """Groq reset headers: "577ms", "7.66s", "1m26.4s", "2h3m". None if unparseable."""
    if not value:
        return None
    text = value.strip()
    try:
        return float(text)  # plain seconds, e.g. a numeric Retry-After
    except ValueError:
        pass
    parts = _DURATION_PART_RE.findall(text)
    if not parts or "".join(n + u for n, u in parts) != text:
        return None
    return sum(float(n) * _UNIT_SECONDS[u] for n, u in parts)


def parse_retry_after(value: str | None) -> float | None:
    """Retry-After is either delta-seconds or an HTTP-date (RFC 9110 10.2.3)."""
    delay = parse_duration(value)
    if delay is not None or not value:
        return delay
    try:
        when = parsedate_to_datetime(value.strip())
    except (TypeError, ValueError):
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    return max(0.0, (when - datetime.now(timezone.utc)).total_seconds())


def _int_header(headers: Mapping[str, str], name: str) -> int | None:
    raw = headers.get(name)
    if raw is None:
        return None
    try:
        return int(float(raw))
    except ValueError:
        return None


@dataclass
class _KeyQuota:
    limit_requests: int | None = None
    remaining_requests: int | None = None
    requests_reset_at: float = 0.0
    limit_tokens: int | None = None
    remaining_tokens: int | None = None
    tokens_reset_at: float = 0.0
    blocked_until: float = 0.0
    reserved_tokens: int = 0
    reserved_requests: int = 0
    skips: int = 0


class QuotaTracker:
    def __init__(
        self,
        *,
        enabled: bool = True,
        safety_margin_tokens: int = 0,
        timeouts_before_block: int = 2,
        timeout_block_seconds: float = 120.0,
        daily_quota_block_seconds: float = 3600.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.enabled = enabled
        self._margin = safety_margin_tokens
        self._timeouts_before_block = timeouts_before_block
        self._timeout_block_seconds = timeout_block_seconds
        self._daily_block_seconds = daily_quota_block_seconds
        self._clock = clock
        self._keys: dict[str, _KeyQuota] = {}
        self._consecutive_timeouts = 0
        self._model_blocked_until = 0.0
        self._model_blocks = 0

    def _state(self, key: str) -> _KeyQuota:
        state = self._keys.get(key)
        if state is None:
            state = self._keys[key] = _KeyQuota()
        return state

    # --- decisions -------------------------------------------------------------

    def can_serve(self, key: str, est_tokens: int) -> bool:
        if not self.enabled:
            return True
        now = self._clock()
        state = self._state(key)
        if state.blocked_until > now:
            return False

        requests = state.remaining_requests if now < state.requests_reset_at else state.limit_requests
        if requests is not None and requests - state.reserved_requests <= 0:
            return False

        # Before the reset the header's remaining count applies; after it the window has
        # refilled to the limit. Unknown limit = no information = no restriction.
        tokens = state.remaining_tokens if now < state.tokens_reset_at else state.limit_tokens
        # Every call costs something; chars/4 rounds a tiny prompt down to 0, which would
        # make a key reporting 0 tokens left look able to serve it.
        if tokens is not None and tokens - state.reserved_tokens - self._margin < max(est_tokens, 1):
            return False
        return True

    def model_blocked_for(self) -> float:
        """Seconds left on a model-wide block (set after repeated hangs), else 0."""
        if not self.enabled:
            return 0.0
        return max(0.0, self._model_blocked_until - self._clock())

    def next_available_in(self, keys: list[str]) -> float:
        """Shortest wait until one of `keys` is unblocked (0 if one already is)."""
        now = self._clock()
        waits = [max(0.0, self._state(k).blocked_until - now) for k in keys]
        return min(waits, default=0.0)

    def wait_until_servable(self, keys: list[str], est_tokens: int) -> float | None:
        """Seconds until one of `keys` can take a request of `est_tokens` (0 if one can now),
        or None when no key is held back by anything with a known end - e.g. only by other
        requests' reservations - so there is no real time to wait for.

        Unlike next_available_in this also counts a key drained by the x-ratelimit-*
        headers (no explicit block, but no tokens until its window resets), which is how
        Groq's per-minute token limit shows up."""
        if not self.enabled or not keys:
            return 0.0
        now = self._clock()
        waits = []
        for key in keys:
            if self.can_serve(key, est_tokens):
                return 0.0
            state = self._state(key)
            ready_at = state.blocked_until
            requests = state.remaining_requests if now < state.requests_reset_at else state.limit_requests
            if requests is not None and requests - state.reserved_requests <= 0:
                ready_at = max(ready_at, state.requests_reset_at)
            tokens = state.remaining_tokens if now < state.tokens_reset_at else state.limit_tokens
            if tokens is not None and tokens - state.reserved_tokens - self._margin < max(est_tokens, 1):
                ready_at = max(ready_at, state.tokens_reset_at)
            if ready_at > now:
                waits.append(ready_at - now)
        return min(waits, default=None)

    def note_skip(self, key: str) -> None:
        self._state(key).skips += 1

    # --- reservations ------------------------------------------------------------
    # Taken synchronously when a key is picked, released when the response is in. Without
    # them, N concurrent requests all read the same "remaining" before the first answer
    # updates it, and all of them collect a 429.

    def reserve(self, key: str, est_tokens: int) -> None:
        state = self._state(key)
        state.reserved_tokens += est_tokens
        state.reserved_requests += 1

    def release(self, key: str, est_tokens: int) -> None:
        state = self._state(key)
        state.reserved_tokens = max(0, state.reserved_tokens - est_tokens)
        state.reserved_requests = max(0, state.reserved_requests - 1)

    # --- observations ------------------------------------------------------------

    def observe_headers(self, key: str, headers: Mapping[str, str]) -> None:
        """Read Groq-style x-ratelimit-* headers. Absent headers change nothing."""
        now = self._clock()
        state = self._state(key)

        limit = _int_header(headers, "x-ratelimit-limit-requests")
        remaining = _int_header(headers, "x-ratelimit-remaining-requests")
        reset = parse_duration(headers.get("x-ratelimit-reset-requests"))
        if limit is not None:
            state.limit_requests = limit
        if remaining is not None:
            state.remaining_requests = remaining
            state.requests_reset_at = now + (reset if reset is not None else 60.0)

        limit = _int_header(headers, "x-ratelimit-limit-tokens")
        remaining = _int_header(headers, "x-ratelimit-remaining-tokens")
        reset = parse_duration(headers.get("x-ratelimit-reset-tokens"))
        if limit is not None:
            state.limit_tokens = limit
        if remaining is not None:
            state.remaining_tokens = remaining
            state.tokens_reset_at = now + (reset if reset is not None else 60.0)

    def observe_rate_limited(
        self, key: str, headers: Mapping[str, str], body: str
    ) -> float | None:
        """Block `key` for as long as the 429 says. Returns how long the key is now
        blocked, or None when the response gave no hint (the circuit breaker then
        handles it as before)."""
        delay = parse_retry_after(headers.get("retry-after"))
        if delay is None:
            match = _RETRY_DELAY_RE.search(body)
            if match:
                delay = float(match.group(1))
        if delay is None:
            match = _RETRY_IN_RE.search(body)
            if match:
                delay = float(match.group(1)) * _UNIT_SECONDS[match.group(2).lower()]
        # A per-day quota reports a short retryDelay, yet retrying in a minute only
        # collects another 429 - the counter resets once a day.
        if _PER_DAY_RE.search(body):
            delay = max(delay or 0.0, self._daily_block_seconds)
        if delay is None:
            return None
        now = self._clock()
        state = self._state(key)
        state.blocked_until = max(state.blocked_until, now + delay)
        return state.blocked_until - now

    def record_success(self) -> None:
        self._consecutive_timeouts = 0

    def record_timeout(self) -> bool:
        """A request that never answered. Hangs are model-wide overload rather than a
        per-key quota, so enough of them in a row block the whole model for a while.
        Returns True when this call started a block."""
        self._consecutive_timeouts += 1
        if self._consecutive_timeouts < self._timeouts_before_block:
            return False
        now = self._clock()
        already = self._model_blocked_until > now
        self._model_blocked_until = now + self._timeout_block_seconds
        if not already:
            self._model_blocks += 1
        return not already

    # --- reporting ---------------------------------------------------------------

    def get_stats(self, keys: list[str]) -> dict[str, Any]:
        """Keyed by position in the provider's key list - never by the key itself."""
        now = self._clock()
        per_key = []
        for index, key in enumerate(keys):
            state = self._state(key)
            per_key.append(
                {
                    "key_index": index,
                    "remaining_tokens": state.remaining_tokens if now < state.tokens_reset_at else None,
                    "remaining_requests": state.remaining_requests if now < state.requests_reset_at else None,
                    "blocked_for_seconds": round(max(0.0, state.blocked_until - now), 1),
                    "skips": state.skips,
                }
            )
        return {
            "model_blocked_for_seconds": round(self.model_blocked_for(), 1),
            "model_blocks": self._model_blocks,
            "consecutive_timeouts": self._consecutive_timeouts,
            "keys": per_key,
        }
