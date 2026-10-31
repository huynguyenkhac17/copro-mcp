from __future__ import annotations

from pathlib import Path

import pytest

from llm_worker_pool.domain.file_context import FileContextResolver


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
async def test_directory_is_read_recursively_skipping_noise_dirs(tmp_path: Path):
    # Was non-recursive until roadmap item 2 (2026-10-31): a module with
    # subpackages had to be listed file by file.
    (tmp_path / "a.txt").write_text("content A")
    (tmp_path / "b.txt").write_text("content B")
    nested_dir = tmp_path / "nested"
    nested_dir.mkdir()
    (nested_dir / "c.txt").write_text("content C nested")
    for noise in ("node_modules", ".git", "__pycache__"):
        (tmp_path / noise).mkdir()
        (tmp_path / noise / "junk.txt").write_text(f"junk from {noise}")

    resolver = FileContextResolver(allowed_roots=[tmp_path])
    result = await resolver.aresolve([str(tmp_path)], query="task")

    assert "content A" in result.text
    assert "content B" in result.text
    assert "content C nested" in result.text
    assert "junk from" not in result.text


@pytest.mark.asyncio
async def test_notebook_is_rendered_as_numbered_cells_without_outputs(tmp_path: Path):
    import json

    nb = tmp_path / "lab.ipynb"
    nb.write_text(json.dumps({"cells": [
        {"cell_type": "markdown", "source": ["# Title\n", "intro"]},
        {"cell_type": "code", "source": "def f():\n    pass",
         "outputs": [{"output_type": "stream", "text": "HUGE_OUTPUT"}]},
    ]}), encoding="utf-8")

    resolver = FileContextResolver(allowed_roots=[tmp_path])
    result = await resolver.aresolve([str(nb)], query="task")

    assert "2 cells, numbered 0-1" in result.text
    assert "=== cell 0 (markdown) ===\n# Title\nintro" in result.text
    assert "=== cell 1 (code) ===\ndef f():\n    pass" in result.text
    assert "HUGE_OUTPUT" not in result.text
    assert '"cell_type"' not in result.text
    assert result.warnings == []


@pytest.mark.asyncio
async def test_malformed_notebook_falls_back_to_raw_text_with_warning(tmp_path: Path):
    nb = tmp_path / "broken.ipynb"
    nb.write_text("{not json", encoding="utf-8")

    resolver = FileContextResolver(allowed_roots=[tmp_path])
    result = await resolver.aresolve([str(nb)], query="task")

    assert "{not json" in result.text
    assert result.warnings == [f"notebook_unparsed_sent_raw:{nb.resolve()}"]


@pytest.mark.asyncio
async def test_drive_root_normalization():
    resolver = FileContextResolver(allowed_roots=[Path("D:")])
    assert resolver._allowed_roots[0] == Path("D:/").resolve()


class _FakeRag:
    def __init__(self) -> None:
        self.calls = 0

    async def index_and_search(self, texts_by_file, query):
        self.calls += 1
        return "RAG_CHUNKS"


@pytest.mark.asyncio
async def test_rag_only_when_files_exceed_the_context_window(tmp_path: Path):
    # RAG swaps whole files for top_k chunks; when everything fits the budget's
    # window the model must get the files verbatim (review needs all the code).
    (tmp_path / "a.py").write_text("x" * 400)  # ~100 tokens
    resolver = FileContextResolver(allowed_roots=[tmp_path])
    rag = _FakeRag()

    fits = await resolver.aresolve([str(tmp_path)], query="q", rag_indexer=rag, rag_min_tokens=1000)
    assert "x" * 400 in fits.text and rag.calls == 0

    too_big = await resolver.aresolve([str(tmp_path)], query="q", rag_indexer=rag, rag_min_tokens=50)
    assert too_big.text == "RAG_CHUNKS" and rag.calls == 1


def test_rag_embeddings_use_ollama_not_the_default_chat_provider():
    from llm_worker_pool.config.settings import AppConfig

    config = AppConfig()
    assert config.rag.embedding_endpoint == "http://localhost:11434"
    assert config.providers[config.default_provider].endpoint != config.rag.embedding_endpoint
