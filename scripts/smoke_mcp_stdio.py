"""Start an installed `copro-mcp` over stdio, as an MCP client would, and check
it answers: three tools listed, get_gateway_stats returns. No API keys are needed
(cloud providers without a key are skipped), so CI can run it on every OS.

    python scripts/smoke_mcp_stdio.py <path to the copro-mcp executable>
"""

from __future__ import annotations

import asyncio
import json
import os
import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

EXPECTED_TOOLS = {"delegate_task", "check_job_status", "get_gateway_stats"}


async def main(command: str) -> None:
    env = dict(os.environ)
    async with stdio_client(StdioServerParameters(command=command, env=env)) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = {tool.name for tool in (await session.list_tools()).tools}
            missing = EXPECTED_TOOLS - tools
            if missing:
                raise SystemExit(f"missing tools: {sorted(missing)} (got {sorted(tools)})")
            stats = json.loads((await session.call_tool("get_gateway_stats", {})).content[0].text)
            if "total_requests" not in stats:
                raise SystemExit(f"unexpected get_gateway_stats answer: {stats}")
    print(f"ok: {sorted(tools)}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    asyncio.run(main(sys.argv[1]))
