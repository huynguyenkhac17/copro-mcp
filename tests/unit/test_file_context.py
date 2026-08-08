from __future__ import annotations

from pathlib import Path

import pytest

from local_ai_gateway.domain.file_context import FileContextResolver


@pytest.mark.asyncio
async def test_allowed_roots_empty_gives_single_disabled_warning(tmp_path: Path):
    f = tmp_path / "a.txt"
    f.write_text("hello")

    resolver = FileContextResolver(allowed_roots=[])
    result = await resolver.aresolve([str(f)], query="task")

    assert result.text == ""
    assert len(result.warnings) == 1
    assert result.warnings[0].startswith("file_access_disabled")


@pytest.mark.asyncio
async def test_reads_file_within_allowed_root(tmp_path: Path):
    f = tmp_path / "a.txt"
    f.write_text("file content")

    resolver = FileContextResolver(allowed_roots=[tmp_path])
    result = await resolver.aresolve([str(f)], query="task")

    assert "file content" in result.text
    assert str(f) in result.text
    assert result.warnings == []


@pytest.mark.asyncio
async def test_denies_path_outside_allowed_root(tmp_path: Path):
    outside_root = tmp_path / "allowed"
    outside_root.mkdir()
    outsider = tmp_path / "outside.txt"
    outsider.write_text("cannot be read")

    resolver = FileContextResolver(allowed_roots=[outside_root])
    result = await resolver.aresolve([str(outsider)], query="task")

    assert result.text == ""
    assert any(w.startswith("file_access_denied") for w in result.warnings)


@pytest.mark.asyncio
async def test_denies_path_traversal_out_of_allowed_root(tmp_path: Path):
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    secret = tmp_path / "secret.txt"
    secret.write_text("secret")
    traversal_path = allowed / ".." / "secret.txt"

    resolver = FileContextResolver(allowed_roots=[allowed])
    result = await resolver.aresolve([str(traversal_path)], query="task")

    assert result.text == ""
    assert any(w.startswith("file_access_denied") for w in result.warnings)


@pytest.mark.asyncio
async def test_missing_file_warns_not_found(tmp_path: Path):
    resolver = FileContextResolver(allowed_roots=[tmp_path])
    result = await resolver.aresolve([str(tmp_path / "does_not_exist.txt")], query="task")

    assert any(w.startswith("file_not_found") for w in result.warnings)


@pytest.mark.asyncio
async def test_file_too_large_is_skipped(tmp_path: Path):
    f = tmp_path / "big.txt"
    f.write_text("x" * 100)

    resolver = FileContextResolver(allowed_roots=[tmp_path], max_file_bytes=10)
    result = await resolver.aresolve([str(f)], query="task")

    assert result.text == ""
    assert any(w.startswith("file_too_large") for w in result.warnings)


@pytest.mark.asyncio
async def test_total_budget_exceeded_stops_further_files(tmp_path: Path):
    f1 = tmp_path / "a.txt"
    f2 = tmp_path / "b.txt"
    f1.write_text("x" * 60)
    f2.write_text("y" * 60)

    resolver = FileContextResolver(
        allowed_roots=[tmp_path], max_file_bytes=100, max_total_bytes=100
    )
    result = await resolver.aresolve([str(f1), str(f2)], query="task")

    assert "x" * 60 in result.text
    assert "y" * 60 not in result.text
    assert any(w.startswith("file_budget_exceeded") for w in result.warnings)


@pytest.mark.asyncio
async def test_binary_file_is_unreadable(tmp_path: Path):
    f = tmp_path / "bin.dat"
    f.write_bytes(b"\xff\xfe\x00\x01\x80\x81")

    resolver = FileContextResolver(allowed_roots=[tmp_path])
    result = await resolver.aresolve([str(f)], query="task")

    assert result.text == ""
    assert any(w.startswith("file_unreadable") for w in result.warnings)


@pytest.mark.asyncio
async def test_directory_reads_all_allowed_files_non_recursively(tmp_path: Path):
    (tmp_path / "a.txt").write_text("content A")
    (tmp_path / "b.txt").write_text("content B")
    nested_dir = tmp_path / "nested"
    nested_dir.mkdir()
    (nested_dir / "c.txt").write_text("content C cannot be read")

    resolver = FileContextResolver(allowed_roots=[tmp_path])
    result = await resolver.aresolve([str(tmp_path)], query="task")

    assert "content A" in result.text
    assert "content B" in result.text
    assert "content C cannot be read" not in result.text
