from __future__ import annotations

from pathlib import Path

import pytest

from copro.domain.file_context import FileContextResolver
from copro.domain.file_selection import glob_regex, split_glob, walk_files
from copro.domain.guardrails import PathDenyList


def _tree(root: Path, files: dict[str, str]) -> None:
    for rel, body in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body)


@pytest.mark.parametrize(
    ("raw", "base", "pattern"),
    [
        ("D:/repo/src/**/*.py", Path("D:/repo/src"), "**/*.py"),
        ("D:/repo/*.md", Path("D:/repo"), "*.md"),
        ("/r/a/b?/c.txt", Path("/r/a"), "b?/c.txt"),
    ],
)
def test_split_glob(raw, base, pattern):
    assert split_glob(raw) == (base, pattern)


@pytest.mark.parametrize(
    ("pattern", "path", "match"),
    [
        ("**/*.py", "a.py", True),
        ("**/*.py", "x/y/a.py", True),
        ("*.py", "x/a.py", False),
        ("src/*.py", "src/x/a.py", False),
        ("a?.py", "ab.py", True),
        ("[!ab].py", "b.py", False),
        ("*.PY", "a.py", True),
        ("*.py", "a.pyc", False),
    ],
)
def test_glob_regex(pattern, path, match):
    assert bool(glob_regex(pattern).match(path)) is match


def test_walk_prunes_noise_and_respects_depth(tmp_path: Path):
    _tree(tmp_path, {
        "a.py": "", "x/b.py": "", "x/y/z/deep.py": "",
        "node_modules/m.js": "", ".venv/lib.py": "", "pkg/package-lock.json": "",
    })
    files, hit = walk_files(tmp_path, max_depth=2, max_entries=1000)
    names = sorted(p.relative_to(tmp_path).as_posix() for p in files)
    assert names == ["a.py", "x/b.py"]  # deep.py is 3 levels down; noise pruned
    assert hit is False


def test_walk_stops_at_entry_limit(tmp_path: Path):
    _tree(tmp_path, {f"f{i}.txt": "" for i in range(10)})
    files, hit = walk_files(tmp_path, max_depth=5, max_entries=4)
    assert hit is True and len(files) == 4


# --- FileContextResolver with directories and globs ---------------------------------

@pytest.mark.asyncio
async def test_glob_selects_matching_files_recursively(tmp_path: Path):
    _tree(tmp_path, {
        "src/a.py": "BODY_A", "src/sub/b.py": "BODY_B", "src/notes.md": "BODY_NOTES",
        "node_modules/c.py": "BODY_VENDORED",
    })
    resolver = FileContextResolver(allowed_roots=[tmp_path])

    result = await resolver.aresolve([str(tmp_path / "**" / "*.py")])

    assert "BODY_A" in result.text and "BODY_B" in result.text
    assert "BODY_NOTES" not in result.text
    assert "BODY_VENDORED" not in result.text


@pytest.mark.asyncio
async def test_glob_outside_allowed_root_is_denied(tmp_path: Path):
    allowed = tmp_path / "allowed"
    _tree(tmp_path, {"allowed/a.py": "A", "outside/b.py": "B"})
    resolver = FileContextResolver(allowed_roots=[allowed])

    result = await resolver.aresolve([str(tmp_path / "outside" / "*.py")])

    assert result.text == ""
    assert result.warnings == [f"file_access_denied:{tmp_path / 'outside' / '*.py'}"]


@pytest.mark.asyncio
async def test_glob_without_match_says_so(tmp_path: Path):
    resolver = FileContextResolver(allowed_roots=[tmp_path])
    raw = str(tmp_path / "*.rs")
    result = await resolver.aresolve([raw])
    assert result.warnings == [f"file_glob_no_match:{raw}"]


@pytest.mark.asyncio
async def test_budget_drops_least_useful_files_with_one_summary_warning(tmp_path: Path):
    _tree(tmp_path, {
        "README.md": "R" * 100,
        "src/app.py": "S" * 100,
        "tests/test_app.py": "T" * 100,
        "ci/config.yml": "Y" * 100,
        "ci/other.yml": "Z" * 100,
    })
    resolver = FileContextResolver(allowed_roots=[tmp_path], max_total_bytes=310)

    result = await resolver.aresolve([str(tmp_path)])

    assert "R" * 100 in result.text and "S" * 100 in result.text and "T" * 100 in result.text
    assert "Y" * 100 not in result.text and "Z" * 100 not in result.text
    assert result.warnings == [f"file_budget_exceeded:{tmp_path} (2 files)"]


@pytest.mark.asyncio
async def test_file_cap_keeps_the_most_useful(tmp_path: Path):
    _tree(tmp_path, {"main.py": "MAIN", "a.cfg": "a", "b.cfg": "b", "c.cfg": "c"})
    resolver = FileContextResolver(allowed_roots=[tmp_path], max_dir_files=2)

    result = await resolver.aresolve([str(tmp_path)])

    assert "MAIN" in result.text
    assert any(w.startswith(f"file_dir_too_many_files:{tmp_path} (2 of 4 omitted") for w in result.warnings)


@pytest.mark.asyncio
async def test_sensitive_file_deep_in_tree_is_denied(tmp_path: Path):
    _tree(tmp_path, {"app.py": "APP", "deploy/prod/.env": "SECRET=1"})
    resolver = FileContextResolver(allowed_roots=[tmp_path], deny_list=PathDenyList())

    result = await resolver.aresolve([str(tmp_path)])

    assert "APP" in result.text and "SECRET" not in result.text
    assert f"file_denied_sensitive:{tmp_path / 'deploy' / 'prod' / '.env'}" in result.warnings


@pytest.mark.asyncio
async def test_unreadable_files_in_a_tree_are_counted_not_listed(tmp_path: Path):
    _tree(tmp_path, {"ok.py": "OK"})
    (tmp_path / "img1.png").write_bytes(b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR")
    (tmp_path / "img2.png").write_bytes(b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR")
    resolver = FileContextResolver(allowed_roots=[tmp_path])

    result = await resolver.aresolve([str(tmp_path)])

    assert "OK" in result.text
    assert result.warnings == [f"file_unreadable:{tmp_path} (2 files)"]


@pytest.mark.asyncio
async def test_file_listed_twice_is_read_once(tmp_path: Path):
    _tree(tmp_path, {"a.py": "ONLY_ONCE"})
    resolver = FileContextResolver(allowed_roots=[tmp_path])

    result = await resolver.aresolve([str(tmp_path / "a.py"), str(tmp_path), str(tmp_path / "*.py")])

    assert result.text.count("ONLY_ONCE") == 1


@pytest.mark.asyncio
async def test_glob_counts_as_directory_for_rag(tmp_path: Path):
    resolver = FileContextResolver(allowed_roots=[tmp_path])
    assert resolver._has_directory([str(tmp_path / "*.py")])


def _junction_or_skip(link: Path, target: Path) -> None:
    """NTFS junction: creatable without admin rights, unlike a symlink."""
    try:
        import _winapi
    except ImportError:
        pytest.skip("junctions are Windows-only")
    _winapi.CreateJunction(str(target), str(link))


@pytest.mark.asyncio
async def test_junction_out_of_root_is_not_followed(tmp_path: Path):
    # Found in review 2026-10-31: os.walk(followlinks=False) still descends into a
    # junction (is_symlink() is False for it), so files outside the root leaked.
    root = tmp_path / "root"
    _tree(tmp_path, {"root/ok.txt": "INSIDE", "outside/secret.txt": "TOP_SECRET_OUTSIDE"})
    _junction_or_skip(root / "jlink", tmp_path / "outside")
    resolver = FileContextResolver(allowed_roots=[root])

    for item in (str(root), str(root / "**" / "*.txt")):
        result = await resolver.aresolve([item])
        assert "INSIDE" in result.text
        assert "TOP_SECRET_OUTSIDE" not in result.text


@pytest.mark.asyncio
async def test_invalid_glob_is_a_warning_not_a_crash(tmp_path: Path):
    resolver = FileContextResolver(allowed_roots=[tmp_path])
    raw = str(tmp_path / "[z-a].py")
    result = await resolver.aresolve([raw])
    assert result.text == ""
    assert result.warnings[0].startswith(f"file_invalid_glob:{raw}")


def test_read_text_file_treats_mostly_undecodable_bytes_as_binary(tmp_path: Path):
    """A PDF's compressed stream can run past the NUL sniff window; it must not be
    sent as a page of replacement characters."""
    from copro.domain.file_selection import read_text_file

    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"%PDF-1.7\nstream\n" + bytes(range(0x80, 0x100)) * 40)
    note = tmp_path / "note.md"
    note.write_bytes(b"long enough line of text \x97 then more text\n" * 3)

    assert read_text_file(pdf) is None
    assert read_text_file(note) == ("long enough line of text � then more text\n" * 3, True)
