"""Multi-turn follow-ups (`continuation_id`): keep what a call was given and answered,
so the next call can say "continue" or "now check X" without resending the files.

History is replayed as a transcript inside the user prompt rather than as chat
messages: every provider, wrapper (cache, concurrency, quota) and test double keeps
its single-prompt signature, and the transcript is redacted like any other task text.

The stored context is the text BEFORE secret redaction: each call redacts for its own
fallback chain, so a conversation started on a local model keeps its full text there
and is still redacted if a later turn goes to the cloud.

Two follow-ups sent in parallel on the same id both build on the same history and the
later one to finish wins (one turn is not recorded) - follow-ups are sequential in
practice, so this is not locked.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any

from llm_worker_pool.domain.token_estimate import estimate_tokens
from llm_worker_pool.infra.cache import InMemoryTTLCache

# Fields a follow-up inherits from the conversation unless the caller sets them.
INHERITED_FIELDS = ("provider", "budget", "task_type", "output_format", "json_schema")


@dataclass(frozen=True)
class Turn:
    task: str
    answer: str


@dataclass(frozen=True)
class Conversation:
    context: str | None
    settings: dict[str, Any]
    turns: tuple[Turn, ...] = field(default_factory=tuple)


class ConversationStore:
    def __init__(self, max_entries: int = 50, ttl_seconds: float = 3600, max_turns: int = 20) -> None:
        self._cache: InMemoryTTLCache[Conversation] = InMemoryTTLCache(
            max_entries=max_entries, ttl_seconds=ttl_seconds
        )
        self._max_turns = max_turns

    def get(self, conversation_id: str) -> Conversation | None:
        return self._cache.get(conversation_id)

    def append(
        self,
        conversation_id: str | None,
        *,
        context: str | None,
        settings: dict[str, Any],
        task: str,
        answer: str,
    ) -> str:
        """Record one answered turn; a new id when `conversation_id` is None. The
        TTL restarts on every turn."""
        previous = self._cache.get(conversation_id) if conversation_id else None
        turns = (*(previous.turns if previous else ()), Turn(task=task, answer=answer))
        conversation_id = conversation_id or uuid.uuid4().hex
        self._cache.set(
            conversation_id,
            Conversation(context=context, settings=settings, turns=turns[-self._max_turns :]),
        )
        return conversation_id


def render_history(turns: tuple[Turn, ...], task: str, max_tokens: int) -> tuple[str, int]:
    """The new task preceded by as many of the latest turns as fit in `max_tokens`.
    Returns (prompt task, number of oldest turns left out)."""
    blocks: list[str] = []
    used = estimate_tokens(task)
    for number, turn in reversed(list(enumerate(turns, start=1))):
        block = f"[Turn {number}] Request:\n{turn.task}\n\n[Turn {number}] Your answer:\n{turn.answer}"
        cost = estimate_tokens(block)
        if used + cost > max_tokens:
            break
        blocks.append(block)
        used += cost
    if not blocks:
        return task, len(turns)
    history = "\n\n".join(reversed(blocks))
    prompt = (
        "This continues an earlier conversation. Earlier turns, oldest first:\n\n"
        f"{history}\n\n"
        "New request (if it says to continue, pick up exactly where your last answer "
        f"stopped, without repeating it):\n{task}"
    )
    return prompt, len(turns) - len(blocks)
