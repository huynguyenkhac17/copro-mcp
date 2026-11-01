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


_ASCII_ALNUM = b"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
_ASCII_SPACE = b" \t\n\r\f\v"


def estimate_prompt_tokens(text: str) -> int:
    """Closer estimate for checks against a vendor's per-request token limit
    (max_prompt_tokens, per-key TPM). Letters and digits run ~4 per token, while
    punctuation and non-ASCII characters (Vietnamese diacritics) cost far more, so
    len/4 under-counted JSON by ~21% and Vietnamese prose by ~17%. Fitted on 11
    samples measured against groq gpt-oss-120b usage.prompt_tokens (2026-11-01):
    0.89-1.15x real for code, diffs, English/Vietnamese Markdown, JSON and YAML
    (len/4: 0.79-1.17x), leaning high on purpose since it guards a hard limit."""
    raw = text.encode("ascii", "ignore")
    non_ascii = len(text) - len(raw)
    alnum = len(raw) - len(raw.translate(None, _ASCII_ALNUM))
    space = len(raw) - len(raw.translate(None, _ASCII_SPACE))
    punct = len(raw) - alnum - space
    return int(alnum / 4 + 0.75 * (punct + non_ascii))

def max_chars_for_tokens(tokens: int) -> int:
    return tokens * _CHARS_PER_TOKEN
