"""Which files of a directory tree are worth sending to a text model, and in what order.

Shared by FileContextResolver (local directories and globs in `file_paths`) and the
GitHub repo fetcher (infra/url_fetcher.py), so both skip the same noise and, when the
byte budget runs out, drop the same low-value tail.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

SKIP_DIRS = frozenset({
    ".git", "node_modules", "dist", "build", "__pycache__", ".venv", "venv",
    ".tox", ".mypy_cache", ".pytest_cache", "vendor", "target",
    # Windows system folders: walking a drive root otherwise reads the recycle bin,
    # i.e. content the user deleted.
    "$RECYCLE.BIN", "System Volume Information",
})
SKIP_SUFFIXES = (".svg", ".min.js", ".map", ".ico", ".lock")
SKIP_FILES = frozenset({
    "package-lock.json", "yarn.lock", "pnpm-lock.yaml", "poetry.lock",
    "uv.lock", "Cargo.lock", "go.sum",
})
CODE_SUFFIXES = frozenset({
    ".py", ".pyi", ".c", ".h", ".cc", ".cpp", ".hpp", ".rs", ".go", ".java",
    ".kt", ".js", ".jsx", ".ts", ".tsx", ".rb", ".php", ".cs", ".swift",
    ".scala", ".sh", ".sql", ".vue", ".svelte",
})
DOC_SUFFIXES = frozenset({".md", ".rst", ".txt", ".adoc"})
GLOB_CHARS = frozenset("*?[")


def wanted(path: Path, root: Path) -> bool:
    """Not inside a skipped directory, not a lockfile / minified / map / icon."""
    rel = path.relative_to(root).parts
    return (
        not any(part in SKIP_DIRS for part in rel[:-1])
        and path.name not in SKIP_FILES
        and not path.name.lower().endswith(SKIP_SUFFIXES)
    )


def priority(rel: Path) -> int:
    """Lower = more useful: top-level README, source, tests, docs, then the rest."""
    name = rel.name.lower()
    if name.startswith("readme") and len(rel.parts) == 1:
        return 0
    is_test = any(part in ("tests", "test", "__tests__", "spec") for part in rel.parts[:-1])
    if rel.suffix.lower() in CODE_SUFFIXES:
        return 2 if is_test else 1
    if rel.suffix.lower() in DOC_SUFFIXES:
        return 3
    return 4


def sort_key(path: Path, root: Path) -> tuple:
    rel = path.relative_to(root)
    return (priority(rel), len(rel.parts), str(rel).lower())


BINARY_SNIFF_BYTES = 8192
MAX_REPLACED_RATIO = 0.05


def read_text_file(path: Path) -> tuple[str, bool] | None:
    """(text, lossy), or None for a binary file. OSError propagates.

    A strict utf-8 read drops the whole file over one bad byte: a file cut mid-
    character by `head -c`, or one PowerShell 5.1 wrote in cp1252. So a file is
    binary only if it has a NUL in the first few KB, or if decoding replaces more
    than a few percent of it (a PDF's compressed streams can go 8KB without a NUL).
    Anything else is sent, with undecodable bytes replaced and `lossy` set so the
    caller can warn."""
    data = path.read_bytes()
    if b"\x00" in data[:BINARY_SNIFF_BYTES]:
        return None
    try:
        return data.decode("utf-8-sig"), False
    except UnicodeDecodeError:
        text = data.decode("utf-8", errors="replace")
    if text.count("�") > MAX_REPLACED_RATIO * len(text):
        return None
    return text, True


def is_glob(raw: str) -> bool:
    return any(ch in GLOB_CHARS for ch in raw)


def split_glob(raw: str) -> tuple[Path, str]:
    """`D:/repo/src/**/*.py` -> (D:/repo/src, "**/*.py"): the longest leading part
    without glob characters is the directory the pattern is evaluated in."""
    parts = Path(raw).parts
    for i, part in enumerate(parts):
        if is_glob(part):
            base = Path(*parts[:i]) if i else Path(".")
            return base, "/".join(parts[i:])
    return Path(raw), ""


_REPARSE_POINT = 0x400  # FILE_ATTRIBUTE_REPARSE_POINT


def is_link_dir(path: str | Path) -> bool:
    """A directory entry that redirects elsewhere: symlink, NTFS junction or other
    reparse point. os.walk(followlinks=False) only skips symlinks - it still descends
    into a junction, which is_symlink() reports as False (found in review 2026-10-31:
    a junction inside an allowed root exposed files outside it)."""
    path = os.fspath(path)
    if os.path.islink(path):
        return True
    isjunction = getattr(os.path, "isjunction", None)  # Python 3.12+
    if isjunction is not None and isjunction(path):
        return True
    try:
        attrs = getattr(os.lstat(path), "st_file_attributes", 0)
    except OSError:
        return True  # cannot tell - do not descend
    return bool(attrs & _REPARSE_POINT)


def walk_files(
    root: Path, *, max_depth: int, max_entries: int
) -> tuple[list[Path], bool]:
    """Files under `root`, pruning SKIP_DIRS while walking (rglob would still descend
    into node_modules/.venv and only filter afterwards - minutes on a big tree).
    Directory symlinks and junctions are not followed. Returns (files, hit_limit): hit_limit is True
    when `max_entries` stopped the walk early."""
    files: list[Path] = []
    seen = 0
    base_depth = len(root.parts)
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        current = Path(dirpath)
        depth = len(current.parts) - base_depth
        dirnames[:] = sorted(
            d for d in dirnames
            if d not in SKIP_DIRS and depth < max_depth and not is_link_dir(current / d)
        )
        for name in sorted(filenames):
            seen += 1
            if seen > max_entries:
                return files, True
            path = current / name
            if wanted(path, root):
                files.append(path)
    return files, False


def glob_regex(pattern: str) -> re.Pattern[str]:
    """Compile a glob relative to its base dir, matched against posix relative paths:
    `**/` = any number of directories (including none), `*` / `?` stay within one
    path segment, `[...]` is a character class. Case-insensitive (Windows paths)."""
    out: list[str] = []
    i = 0
    while i < len(pattern):
        if pattern.startswith("**/", i):
            out.append("(?:.*/)?")
            i += 3
        elif pattern.startswith("**", i):
            out.append(".*")
            i += 2
        elif pattern[i] == "*":
            out.append("[^/]*")
            i += 1
        elif pattern[i] == "?":
            out.append("[^/]")
            i += 1
        elif pattern[i] == "[":
            end = pattern.find("]", i + 1)
            if end == -1:
                out.append(re.escape(pattern[i]))
                i += 1
            else:
                body = pattern[i + 1 : end]
                if body.startswith("!"):
                    body = "^" + body[1:]
                out.append("[" + body.replace("\\", "\\\\") + "]")
                i = end + 1
        else:
            out.append(re.escape(pattern[i]))
            i += 1
    return re.compile("".join(out) + r"\Z", re.IGNORECASE)


def glob_files(
    base: Path, pattern: str, *, max_depth: int, max_entries: int
) -> tuple[list[Path], bool]:
    """Files under `base` whose posix path relative to `base` matches `pattern`.
    Walks with the same pruning and limits as walk_files."""
    regex = glob_regex(pattern.replace("\\", "/"))
    files, hit_limit = walk_files(base, max_depth=max_depth, max_entries=max_entries)
    return [f for f in files if regex.match(f.relative_to(base).as_posix())], hit_limit
