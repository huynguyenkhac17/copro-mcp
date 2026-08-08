"""Token estimation heuristic for the gateway."""

from __future__ import annotations

# Rough heuristic: average ~4 characters / token for mixed English-Vietnamese text.
_CHARS_PER_TOKEN = 4

# Reserve a portion of a budget's context window for system prompt + task + output,
# shared by anything that needs to size context against num_ctx (PromptBuilder,
# MapReduceOrchestrator chunking, ...).
CONTEXT_RESERVE_RATIO = 0.25


def estimate_tokens(text: str) -> int:
    return len(text) // _CHARS_PER_TOKEN

def max_chars_for_tokens(tokens: int) -> int:
    return tokens * _CHARS_PER_TOKEN
