"""Smoke test for the composition root (server.py).

This is exactly the kind of wiring bug (missing import, provider startup
crash, ...) that mocked unit tests for individual classes cannot catch -
those never actually call build_server(). Runs against the real
config/default.yaml, but never talks to Ollama/network: constructing a
provider only sets up an httpx.AsyncClient, it does not connect anywhere.
"""

from __future__ import annotations

import pytest

from llm_worker_pool.server import build_server


def test_build_server_does_not_raise():
    server = build_server()
    assert server is not None


@pytest.mark.asyncio
async def test_build_server_registers_expected_tools():
    server = build_server()
    tools = await server.list_tools()
    tool_names = {tool.name for tool in tools}

    assert {"delegate_task", "check_job_status", "get_gateway_stats"} <= tool_names
