from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from copro.domain.file_context import FileContextResolver, resolve_request_context
from copro.infra.url_fetcher import UrlContextFetcher, html_to_text

PAGE = (
    "<html><head><title>Demo Page</title><style>.x{}</style></head><body>"
    "<script>alert('x')</script><h1>Heading</h1><p>Hello <b>world</b>.</p>"
    "<ul><li>one</li><li>two</li></ul></body></html>"
)


def _fetcher(handler, *, hosts=None, clone=None, **kwargs) -> UrlContextFetcher:
    hosts = hosts or {}

    async def resolve(host: str) -> list[str]:
        return hosts.get(host, ["93.184.216.34"])  # any public address

    return UrlContextFetcher(
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        host_resolver=resolve,
        clone_runner=clone,
        **kwargs,
    )


def test_html_to_text_keeps_structure_and_drops_scripts():
    text = html_to_text(PAGE)

    assert text.startswith("Title: Demo Page")
    assert "# Heading" in text
    assert "Hello world." in text
    assert "- one" in text and "- two" in text
    assert "alert" not in text and ".x{}" not in text


@pytest.mark.asyncio
async def test_html_page_is_fetched_as_text():
    fetcher = _fetcher(lambda req: httpx.Response(200, html=PAGE))

    result = await fetcher.fetch(["https://example.com/a"])

    assert result.text.startswith("--- url: https://example.com/a ---")
    assert "Hello world." in result.text
    assert result.warnings == []


@pytest.mark.asyncio
async def test_private_host_is_blocked_before_any_request():
    calls = []
    fetcher = _fetcher(lambda req: calls.append(req) or httpx.Response(200, text="secret"),
                       hosts={"router.local": ["192.168.1.1"]})

    result = await fetcher.fetch(["http://router.local/admin"])

    assert result.text == ""
    assert result.warnings == ["url_private_host_blocked:http://router.local/admin"]
    assert calls == []


@pytest.mark.asyncio
async def test_redirect_into_private_network_is_blocked():
    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.host == "example.com":
            return httpx.Response(302, headers={"location": "http://169.254.169.254/latest"})
        return httpx.Response(200, text="cloud metadata")

    fetcher = _fetcher(handler, hosts={"169.254.169.254": ["169.254.169.254"]})

    result = await fetcher.fetch(["https://example.com/r"])

    assert "cloud metadata" not in result.text
    assert result.warnings == ["url_private_host_blocked_on_redirect:https://example.com/r"]


@pytest.mark.asyncio
async def test_public_redirect_is_followed():
    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path == "/old":
            return httpx.Response(301, headers={"location": "https://example.com/new"})
        return httpx.Response(200, text="moved here")

    result = await _fetcher(handler).fetch(["https://example.com/old"])

    assert "moved here" in result.text


@pytest.mark.asyncio
async def test_non_http_scheme_and_binary_and_http_error_are_skipped_with_warnings():
    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path == "/img.png":
            return httpx.Response(200, content=b"\x89PNG", headers={"content-type": "image/png"})
        return httpx.Response(404)

    result = await _fetcher(handler).fetch(
        ["file:///etc/passwd", "https://example.com/img.png", "https://example.com/missing"]
    )

    assert result.text == ""
    assert result.warnings == [
        "url_unsupported_scheme:file:///etc/passwd",
        "url_unsupported_content_type:image/png:https://example.com/img.png",
        "url_http_404:https://example.com/missing",
    ]


@pytest.mark.asyncio
async def test_github_blob_url_is_fetched_from_raw_host():
    seen = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append(str(req.url))
        return httpx.Response(200, text="print('hi')", headers={"content-type": "text/plain"})

    result = await _fetcher(handler).fetch(["https://github.com/o/r/blob/main/src/a.py"])

    assert seen == ["https://raw.githubusercontent.com/o/r/main/src/a.py"]
    assert "print('hi')" in result.text


def _fake_clone(files: dict[str, str], record: list | None = None):
    async def clone(url: str, ref: str | None, dest: Path, timeout: float) -> None:
        if record is not None:
            record.append((url, ref))
        for rel, content in files.items():
            path = dest / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")

    return clone


@pytest.mark.asyncio
async def test_github_repo_is_cloned_readme_first_and_junk_skipped():
    calls: list = []
    clone = _fake_clone(
        {
            "src/pkg/core.py": "def core(): ...",
            "README.md": "# Repo",
            "node_modules/dep/index.js": "junk",
            "package-lock.json": "{}",
        },
        record=calls,
    )
    fetcher = _fetcher(lambda req: httpx.Response(500), clone=clone)

    result = await fetcher.fetch(["https://github.com/owner/repo/tree/dev"])

    assert calls == [("https://github.com/owner/repo.git", "dev")]
    assert result.text.index("owner/repo/README.md") < result.text.index("owner/repo/src/pkg/core.py")
    assert "junk" not in result.text and "package-lock" not in result.text
    assert result.warnings == []


@pytest.mark.asyncio
async def test_github_repo_over_budget_is_truncated_with_warning():
    clone = _fake_clone({"README.md": "r" * 50, "a.py": "a" * 500, "b.py": "b" * 500})
    fetcher = _fetcher(lambda req: httpx.Response(500), clone=clone, max_total_chars=700)

    result = await fetcher.fetch(["https://github.com/owner/repo"])

    assert "owner/repo/README.md" in result.text
    assert result.warnings == ["url_repo_truncated:https://github.com/owner/repo:1_files_omitted"]


@pytest.mark.asyncio
async def test_failed_clone_becomes_a_warning():
    async def clone(url, ref, dest, timeout):
        raise RuntimeError("fatal: repository not found")

    fetcher = _fetcher(lambda req: httpx.Response(500), clone=clone)

    result = await fetcher.fetch(["https://github.com/owner/private"])

    assert result.text == ""
    assert result.warnings[0].startswith("url_repo_clone_failed:https://github.com/owner/private")


@pytest.mark.asyncio
async def test_urls_without_fetcher_warn_instead_of_failing(tmp_path: Path):
    combined, warnings = await resolve_request_context(
        FileContextResolver(allowed_roots=[tmp_path]),
        context="ctx",
        file_paths=None,
        query="task",
        urls=["https://example.com"],
        url_fetcher=None,
    )

    assert combined == "ctx"
    assert warnings[0].startswith("url_fetch_disabled")


@pytest.mark.asyncio
async def test_urls_are_appended_after_context():
    fetcher = _fetcher(lambda req: httpx.Response(200, text="page body"))

    combined, warnings = await resolve_request_context(
        FileContextResolver(allowed_roots=[]),
        context="ctx",
        file_paths=None,
        query="task",
        urls=["https://example.com/p"],
        url_fetcher=fetcher,
    )

    assert combined.startswith("ctx\n\n--- url: https://example.com/p ---")
    assert warnings == []


def test_svg_titles_do_not_leak_into_page_title():
    html = "<html><head><title>Real</title></head><body><svg><title>icon</title></svg>x</body></html>"

    assert html_to_text(html).startswith("Title: Real\n")


@pytest.mark.asyncio
async def test_repo_orders_source_before_tests_docs_and_config():
    clone = _fake_clone({
        ".github/workflows/ci.yaml": "ci", "docs/guide.md": "doc", "tests/test_a.py": "t",
        "src/pkg/a.py": "code", "README.md": "r", "logo.svg": "<svg/>",
    })
    result = await _fetcher(lambda req: httpx.Response(500), clone=clone).fetch(
        ["https://github.com/o/r"]
    )

    order = [line for line in result.text.splitlines() if line.startswith("--- file:")]
    assert order == [
        "--- file: o/r/README.md ---", "--- file: o/r/src/pkg/a.py ---",
        "--- file: o/r/tests/test_a.py ---", "--- file: o/r/docs/guide.md ---",
        "--- file: o/r/.github/workflows/ci.yaml ---",
    ]


def test_repo_render_skips_symlinked_files(tmp_path):
    """A cloned repo is untrusted: a symlink in it must not pull in a file from
    elsewhere on the machine."""
    from copro.infra.url_fetcher import UrlContextFetcher

    outside = tmp_path / "outside.txt"
    outside.write_text("TOP_SECRET_OUTSIDE")
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "main.py").write_text("print('hi')")
    try:
        (repo / "notes.txt").symlink_to(outside)
    except OSError as exc:
        if getattr(exc, "winerror", None) != 1314:
            raise
        pytest.skip(f"cannot create symlinks here: {exc}")

    text, _ = UrlContextFetcher()._render_repo("https://github.com/o/r", "o/r", repo, repo, 100_000)

    assert "print('hi')" in text
    assert "TOP_SECRET_OUTSIDE" not in text


def test_repo_render_keeps_non_utf8_text_and_skips_binary(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "main.py").write_text("print('hi')")
    (repo / "notes.md").write_bytes(b"cp1252 \x97 dash in a normal sentence")
    (repo / "cut.txt").write_bytes(("diff line\n" * 5 + "đú").encode("utf-8")[:-1])
    (repo / "blob.dat").write_bytes(b"\x00\x01\x02 BINARYMARK")

    text, warnings = UrlContextFetcher()._render_repo(
        "https://github.com/o/r", "o/r", repo, repo, 100_000
    )

    assert "cp1252 � dash" in text
    assert "--- file: o/r/cut.txt ---" in text
    assert "BINARYMARK" not in text
    assert "url_repo_decoded_lossy:https://github.com/o/r:2" in warnings
