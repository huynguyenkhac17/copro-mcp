"""Fetches `urls` server-side so Claude never pastes web pages or repos into `context`.

Two kinds of URL are understood:
- a GitHub repository (github.com/<owner>/<repo>[/tree/<ref>[/<subdir>]]) is shallow-
  cloned into a temp dir and its text files are inlined, README first;
- anything else is an HTTP GET. HTML is reduced to readable text, other text-like
  bodies are passed through, binaries are skipped with a warning.

Same contract as FileContextResolver: NEVER raise on a bad URL - skip it and return a
warning, so one dead link does not sink the whole task. This lives in infra because
it is network and subprocess I/O, not business policy.
"""

from __future__ import annotations

import asyncio
import ipaddress
import os
import re
import socket
import tempfile
from collections.abc import Awaitable, Callable
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlparse

import httpx
from pydantic import BaseModel

from llm_worker_pool.domain.file_selection import read_text_file, sort_key, wanted

_GITHUB_REPO_RE = re.compile(
    r"^https?://(?:www\.)?github\.com/(?P<owner>[\w.-]+)/(?P<repo>[\w.-]+?)(?:\.git)?"
    r"(?:/tree/(?P<ref>[^/]+)(?:/(?P<subdir>.+?))?)?/?$"
)
_GITHUB_BLOB_RE = re.compile(
    r"^https?://(?:www\.)?github\.com/(?P<owner>[\w.-]+)/(?P<repo>[\w.-]+)/blob/(?P<rest>.+)$"
)
_TEXT_TYPES = ("text/", "application/json", "application/xml", "application/javascript",
               "application/x-yaml", "application/yaml", "application/toml")
_MAX_REDIRECTS = 5
HostResolver = Callable[[str], Awaitable[list[str]]]
CloneRunner = Callable[[str, "str | None", Path, float], Awaitable[None]]


class FetchedContext(BaseModel):
    text: str
    warnings: list[str] = []


class UrlContextFetcher:
    def __init__(
        self,
        *,
        max_bytes_per_url: int = 1_000_000,
        max_total_chars: int = 800_000,
        timeout_seconds: float = 30.0,
        allow_private_hosts: bool = False,
        repo_max_file_bytes: int = 100_000,
        http_client: httpx.AsyncClient | None = None,
        host_resolver: HostResolver | None = None,
        clone_runner: CloneRunner | None = None,
    ) -> None:
        self._max_bytes_per_url = max_bytes_per_url
        self._max_total_chars = max_total_chars
        self._timeout = timeout_seconds
        self._allow_private = allow_private_hosts
        self._repo_max_file_bytes = repo_max_file_bytes
        self._client = http_client or httpx.AsyncClient(
            timeout=timeout_seconds,
            headers={"User-Agent": "llm-worker-pool/1 (+url fetch for delegate_task)"},
        )
        self._resolve_host = host_resolver or _default_resolve_host
        self._clone = clone_runner or _git_clone

    async def fetch(self, urls: list[str]) -> FetchedContext:
        parts: list[str] = []
        warnings: list[str] = []
        budget = self._max_total_chars

        for url in urls:
            if budget <= 0:
                warnings.append(f"url_budget_exceeded:{url}")
                continue
            problem = await self._check_url(url)
            if problem:
                warnings.append(f"{problem}:{url}")
                continue

            repo = _GITHUB_REPO_RE.match(url)
            if repo:
                text, warns = await self._fetch_repo(url, repo, budget)
            else:
                blob = _GITHUB_BLOB_RE.match(url)
                target = (
                    f"https://raw.githubusercontent.com/{blob['owner']}/{blob['repo']}/{blob['rest']}"
                    if blob
                    else url
                )
                text, warns = await self._fetch_page(url, target, budget)
            warnings.extend(warns)
            if text:
                parts.append(text)
                budget -= len(text)

        return FetchedContext(text="\n\n".join(parts), warnings=warnings)

    async def aclose(self) -> None:
        await self._client.aclose()

    async def _check_url(self, url: str) -> str | None:
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https") or not parsed.hostname:
            return "url_unsupported_scheme"
        if self._allow_private:
            return None
        try:
            addresses = await self._resolve_host(parsed.hostname)
        except OSError:
            return "url_unresolvable_host"
        for addr in addresses:
            ip = ipaddress.ip_address(addr.split("%", 1)[0])
            if not ip.is_global:
                # Block SSRF into the user's own network (router admin pages, local
                # services, cloud metadata endpoints) from a prompt-supplied URL.
                return "url_private_host_blocked"
        return None

    async def _fetch_page(self, url: str, target: str, budget: int) -> tuple[str, list[str]]:
        warnings: list[str] = []
        try:
            # Redirects are followed by hand so every hop passes the private-host check;
            # letting httpx follow them would let a public page bounce us to localhost.
            for _ in range(_MAX_REDIRECTS + 1):
                resp = await self._client.send(
                    self._client.build_request("GET", target), stream=True, follow_redirects=False
                )
                if not resp.is_redirect:
                    break
                await resp.aclose()
                target = str(resp.next_request.url) if resp.next_request else ""
                problem = await self._check_url(target)
                if problem:
                    return "", [f"{problem}_on_redirect:{url}"]
            else:
                return "", [f"url_too_many_redirects:{url}"]

            try:
                if resp.status_code >= 400:
                    return "", [f"url_http_{resp.status_code}:{url}"]
                ctype = resp.headers.get("content-type", "").split(";")[0].strip().lower()
                if ctype and not ctype.startswith(_TEXT_TYPES) and "html" not in ctype:
                    return "", [f"url_unsupported_content_type:{ctype}:{url}"]
                body = bytearray()
                async for chunk in resp.aiter_bytes():
                    body.extend(chunk)
                    if len(body) > self._max_bytes_per_url:
                        warnings.append(f"url_truncated_at_{self._max_bytes_per_url}_bytes:{url}")
                        del body[self._max_bytes_per_url:]
                        break
                encoding = resp.encoding or "utf-8"
            finally:
                await resp.aclose()
        except httpx.HTTPError as exc:
            return "", [f"url_fetch_failed:{url}:{type(exc).__name__}"]

        raw = body.decode(encoding, errors="replace")
        looks_html = "html" in ctype or raw.lstrip()[:15].lower().startswith(("<!doctype html", "<html"))
        text = html_to_text(raw) if looks_html else raw
        if len(text) > budget:
            text = text[:budget]
            warnings.append(f"url_budget_truncated:{url}")
        return f"--- url: {url} ---\n{text.strip()}", warnings

    async def _fetch_repo(self, url: str, m: re.Match[str], budget: int) -> tuple[str, list[str]]:
        clone_url = f"https://github.com/{m['owner']}/{m['repo']}.git"
        with tempfile.TemporaryDirectory(prefix="lwp-repo-") as tmp:
            dest = Path(tmp) / m["repo"]
            try:
                await self._clone(clone_url, m["ref"], dest, self._timeout * 4)
            except (OSError, RuntimeError, asyncio.TimeoutError) as exc:
                return "", [f"url_repo_clone_failed:{url}:{str(exc)[:120]}"]

            root = dest / m["subdir"] if m["subdir"] else dest
            if not root.is_dir():
                return "", [f"url_repo_path_not_found:{url}"]
            return self._render_repo(url, f"{m['owner']}/{m['repo']}", dest, root, budget)

    def _render_repo(
        self, url: str, label: str, dest: Path, root: Path, budget: int
    ) -> tuple[str, list[str]]:
        files = sorted(
            # A cloned repo is untrusted: a symlinked file in it can point anywhere on
            # this machine (~/.ssh/..., another project's .env) and read_text() follows it.
            (p for p in root.rglob("*") if p.is_file() and not p.is_symlink() and wanted(p, dest)),
            # When the budget runs out the tail is dropped, so order by value: README,
            # source, tests, docs, then config/CI noise.
            key=lambda p: sort_key(p, dest),
        )
        parts = [f"--- repo: {url} ({len(files)} text files) ---"]
        used = len(parts[0])
        skipped_large = omitted = lossy_count = 0
        for i, path in enumerate(files):
            if path.stat().st_size > self._repo_max_file_bytes:
                skipped_large += 1
                continue
            try:
                decoded = read_text_file(path)
            except OSError:
                continue
            if decoded is None:
                continue  # binary - not useful to a text model
            content, lossy = decoded
            block = f"--- file: {label}/{path.relative_to(dest).as_posix()} ---\n{content}"
            if used + len(block) > budget:
                omitted = len(files) - i
                break
            parts.append(block)
            used += len(block)
            lossy_count += lossy

        warnings = []
        if omitted:
            warnings.append(f"url_repo_truncated:{url}:{omitted}_files_omitted")
        if skipped_large:
            warnings.append(f"url_repo_large_files_skipped:{url}:{skipped_large}")
        if lossy_count:
            warnings.append(f"url_repo_decoded_lossy:{url}:{lossy_count}")
        return "\n\n".join(parts), warnings


async def _default_resolve_host(host: str) -> list[str]:
    infos = await asyncio.get_running_loop().getaddrinfo(host, None, type=socket.SOCK_STREAM)
    return [info[4][0] for info in infos]


async def _git_clone(url: str, ref: str | None, dest: Path, timeout: float) -> None:
    args = ["git", "clone", "--depth", "1", "--single-branch"]
    if ref:
        args += ["--branch", ref]
    args += [url, str(dest)]
    env = {
        **os.environ,
        "GIT_TERMINAL_PROMPT": "0",  # private repo -> fail fast, never hang on a prompt
        "GIT_LFS_SKIP_SMUDGE": "1",  # don't pull large binaries we would skip anyway
    }
    proc = await asyncio.create_subprocess_exec(
        *args, env=env, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE
    )
    try:
        _, stderr = await asyncio.wait_for(proc.communicate(), timeout)
    except asyncio.TimeoutError:
        proc.kill()
        raise
    if proc.returncode != 0:
        raise RuntimeError(stderr.decode(errors="replace").strip() or f"git exit {proc.returncode}")


class _TextExtractor(HTMLParser):
    _SKIP = {"script", "style", "noscript", "svg", "template", "head"}
    _BLOCK = {"p", "div", "br", "li", "tr", "section", "article", "header", "footer",
              "h1", "h2", "h3", "h4", "h5", "h6", "pre", "blockquote", "table", "ul", "ol", "dt", "dd"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.out: list[str] = []
        self.title = ""
        self._skip_depth = 0
        self._in_title = False

    def handle_starttag(self, tag: str, attrs) -> None:  # noqa: ANN001
        if tag == "title" and not self.title:
            self._in_title = True
        if tag in self._SKIP:
            self._skip_depth += 1
        elif tag in self._BLOCK:
            self.out.append("\n")
        if tag in ("h1", "h2", "h3", "h4"):
            self.out.append("#" * int(tag[1]) + " ")
        elif tag == "li":
            self.out.append("- ")

    def handle_endtag(self, tag: str) -> None:
        if tag == "title":
            self._in_title = False
        if tag in self._SKIP and self._skip_depth:
            self._skip_depth -= 1
        elif tag in self._BLOCK:
            self.out.append("\n")

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self.title += data
        elif not self._skip_depth:
            self.out.append(data)


def html_to_text(html: str) -> str:
    """Readable text from HTML: drops scripts/styles, keeps headings and list structure."""
    parser = _TextExtractor()
    parser.feed(html)
    text = "".join(parser.out)
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r" *\n[ \n]*", "\n", text)
    title = parser.title.strip()
    return (f"Title: {title}\n\n" if title else "") + text.strip()
