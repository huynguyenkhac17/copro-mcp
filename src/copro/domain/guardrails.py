"""Input/output guardrails for delegate_task.

PathDenyList: files that must never be read through file_paths, even inside
allowed_roots. With allowed_roots: ["D:/"], nothing else stopped the server's own
.env (Gemini/Groq keys) from being shipped to a cloud API on request.

SecretScanner: replaces secret-looking literals with [REDACTED:<kind>] before text
leaves the machine, and flags any that show up in an answer. Pattern-based, so it is a
net for the obvious cases (vendor key prefixes, PEM blocks, password="..."), not a
guarantee - provider='ollama' remains the answer for genuinely confidential data.
"""

from __future__ import annotations

import re
from collections import Counter
from fnmatch import fnmatchcase
from pathlib import Path

DEFAULT_DENY_GLOBS: tuple[str, ...] = (
    ".env*",
    "*.pem",
    "*.key",
    "*id_rsa*",
    "*id_ed25519*",
    "credentials*.json",
    "**/.git/**",
)

_REDACTED = "[REDACTED:{kind}]"

# Groq names the account in its errors ("in organization `org_01...`"), OpenAI uses
# "org-...". Not a secret, but it identifies the owner, and provider errors travel far:
# warnings and errors returned to the client (which users paste into issues), logs,
# eval reports.
_ACCOUNT_ID = re.compile(r"\borg[_-][0-9A-Za-z]{20,}")


def redact_account_ids(text: str | None) -> str | None:
    """Mask provider account ids (org_REDACTED); None and "" pass through."""
    return _ACCOUNT_ID.sub("org_REDACTED", text) if text else text

# (kind, pattern, group holding the secret - 0 = whole match)
_BUILTIN_PATTERNS: tuple[tuple[str, str, int], ...] = (
    ("private_key", r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----[\s\S]*?(?:-----END [A-Z0-9 ]*PRIVATE KEY-----|\Z)", 0),
    ("google_api_key", r"\bAIza[0-9A-Za-z_\-]{30,}", 0),
    ("groq_api_key", r"\bgsk_[0-9A-Za-z]{20,}", 0),
    ("github_token", r"\b(?:gh[pousr]_[0-9A-Za-z]{30,}|github_pat_[0-9A-Za-z_]{30,})", 0),
    ("openai_api_key", r"\bsk-(?:proj-|ant-)?[0-9A-Za-z_\-]{20,}", 0),
    ("aws_access_key", r"\bAKIA[0-9A-Z]{16}\b", 0),
    ("url_password", r"://[^/\s:@]+:([^/\s@]{3,})@", 1),
    # password = "literal" / token: 'literal' - quoted literals only, so code such as
    # `token = get_token()` is left alone.
    (
        "assigned_secret",
        r"(?i)\b(?:password|passwd|pwd|secret|token|api[_-]?key|access[_-]?key|client[_-]?secret)\b"
        r"[\"']?\s*[:=]\s*([\"'])([^\"'\s]{8,})\1",
        2,
    ),
    # SOME_TOKEN=value in env files (unquoted, whole line).
    (
        "env_secret",
        r"(?m)^\s*(?:export\s+)?[A-Z0-9_]*(?:PASSWORD|SECRET|TOKEN|API_KEY|ACCESS_KEY)[A-Z0-9_]*"
        r"\s*=\s*[\"']?([^\s\"']{8,})[\"']?\s*$",
        1,
    ),
)


class PathDenyList:
    """Glob deny-list. A glob without '/' is matched against the file name
    (".env*" denies /any/dir/.env.local); one with '/' against the whole path, where
    "**/.git/**" denies anything inside a .git directory. Case-insensitive, because
    the same config also runs on Windows."""

    def __init__(self, globs: list[str] | tuple[str, ...] = DEFAULT_DENY_GLOBS) -> None:
        self._name_globs = [g.lower() for g in globs if "/" not in g]
        self._path_globs = [g.lower() for g in globs if "/" in g]

    def match(self, path: Path | str) -> str | None:
        """The glob that denies `path`, or None."""
        posix = str(path).replace("\\", "/").lower()
        name = posix.rstrip("/").rsplit("/", 1)[-1]
        for glob in self._name_globs:
            if fnmatchcase(name, glob):
                return glob
        for glob in self._path_globs:
            # "x/.git" itself (a directory passed directly) must match "**/.git/**" too.
            if fnmatchcase(posix, glob) or fnmatchcase(posix + "/", glob):
                return glob
            if glob.startswith("**/") and fnmatchcase("/" + posix.lstrip("/"), glob[2:]):
                return glob
        return None


class SecretScanner:
    def __init__(self, extra_patterns: dict[str, str] | None = None) -> None:
        patterns = list(_BUILTIN_PATTERNS)
        for kind, regex in (extra_patterns or {}).items():
            patterns.append((kind, regex, 0))
        self._patterns = [(kind, re.compile(regex), group) for kind, regex, group in patterns]

    def redact(self, text: str) -> tuple[str, Counter[str]]:
        """Text with every secret replaced, and how many of each kind were found."""
        counts: Counter[str] = Counter()
        for kind, pattern, group in self._patterns:
            replacement = _REDACTED.format(kind=kind)

            def _sub(match: re.Match[str], kind: str = kind, group: int = group, replacement: str = replacement) -> str:
                value = match.group(group)
                # Never re-match our own placeholders ("://[REDACTED:github_token]@"
                # would otherwise look like user:password@).
                if not value or "[REDACTED:" in match.group(0):
                    return match.group(0)
                counts[kind] += 1
                if group == 0:
                    return replacement
                start, end = match.span(group)
                offset = match.start()
                whole = match.group(0)
                return whole[: start - offset] + replacement + whole[end - offset :]

            text = pattern.sub(_sub, text)
        return text, counts

    def scan(self, text: str) -> Counter[str]:
        return self.redact(text)[1]


def format_counts(prefix: str, counts: Counter[str]) -> list[str]:
    """["secret_redacted:google_api_key x2", ...] - kinds sorted for stable output."""
    return [f"{prefix}:{kind} x{n}" for kind, n in sorted(counts.items())]


UNTRUSTED_OPEN = "<untrusted_context>"
UNTRUSTED_CLOSE = "</untrusted_context>"


def wrap_untrusted(text: str) -> str:
    """Fence context so the model treats it as data. A closing tag inside the data is
    defused, otherwise a file could end the fence early and speak as the user."""
    safe = re.sub(r"</\s*untrusted_context\s*>", "</untrusted_context_escaped>", text, flags=re.IGNORECASE)
    return f"{UNTRUSTED_OPEN}\n{safe}\n{UNTRUSTED_CLOSE}"
