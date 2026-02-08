# mcp_stdio_client.py
from __future__ import annotations

import json
from contextlib import AsyncExitStack
from typing import Any, Dict, Optional

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


def mcp_result_to_text(result: Any) -> str:
    """
    Convert MCP tool results to a readable string.

    FastMCP commonly returns an object with:
      result.content = [TextContent(text="..."), ...]
    """
    content = getattr(result, "content", None)
    if isinstance(content, list):
        texts = []
        for item in content:
            text = getattr(item, "text", None)
            if isinstance(text, str) and text:
                texts.append(text)
        if texts:
            return "\n".join(texts)

    # Fallback: show something useful
    try:
        return json.dumps(result, indent=2, default=str)
    except Exception:
        return str(result)


class MCPStdioClient:
    """
    Stdio MCP client that spawns a local MCP server process (your mcp_stdio_server.py)
    and communicates with it via stdin/stdout.

    Usage:
        mcp = MCPStdioClient(server_script_path="mcp_stdio_server.py", python_exe="./.venv/bin/python")
        await mcp.start()
        tools = await mcp.list_tools()
        out = await mcp.call_tool("add_numbers", {"a": 1, "b": 2})
        await mcp.close()
    """

    def __init__(
        self,
        server_script_path: str,
        python_exe: str = "python3",
        env: Optional[Dict[str, str]] = None,
    ):
        self.server_script_path = server_script_path
        self.python_exe = python_exe
        self.env = env

        self._stack = AsyncExitStack()
        self.session: Optional[ClientSession] = None

    async def start(self) -> None:
        """
        Spawn the server process and complete MCP initialization handshake.
        """
        params = StdioServerParameters(
            command=self.python_exe,
            args=[self.server_script_path],
            env=self.env,
        )

        # stdio_client spawns the process and gives us async read/write streams
        read_stream, write_stream = await self._stack.enter_async_context(stdio_client(params))

        # ClientSession handles MCP JSON-RPC over those streams
        self.session = await self._stack.enter_async_context(ClientSession(read_stream, write_stream))

        # MCP handshake
        await self.session.initialize()

    async def close(self) -> None:
        """
        Close session and terminate server subprocess cleanly.
        """
        await self._stack.aclose()

    async def list_tools(self) -> Any:
        """
        List tools exposed by the MCP server.
        The SDK returns a typed response; your agent can inspect `.tools`.
        """
        if not self.session:
            raise RuntimeError("MCPStdioClient not started. Call await start() first.")
        return await self.session.list_tools()

    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> Any:
        """
        Call a tool by name with arguments.
        Returns the typed MCP result object (convert with mcp_result_to_text if desired).
        """
        if not self.session:
            raise RuntimeError("MCPStdioClient not started. Call await start() first.")
        return await self.session.call_tool(name, arguments)

