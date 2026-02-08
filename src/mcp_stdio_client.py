import json
from contextlib import AsyncExitStack
from typing import Any, Dict, Optional

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


def mcp_result_to_text(result: Any) -> str:
    # FastMCP commonly returns result.content = [TextContent(...)]
    content = getattr(result, "content", None)
    if isinstance(content, list):
        texts = []
        for item in content:
            text = getattr(item, "text", None)
            if isinstance(text, str) and text:
                texts.append(text)
        if texts:
            return "\n".join(texts)

    # fallback
    try:
        return json.dumps(result, indent=2, default=str)
    except Exception:
        return str(result)


class MCPStdioClient:
    def __init__(self, server_script_path: str, python_exe: str = "python3", env: Optional[Dict[str, str]] = None):
        self.server_script_path = server_script_path
        self.python_exe = python_exe
        self.env = env

        self._stack = AsyncExitStack()
        self.session: ClientSession | None = None

    async def start(self) -> None:
        params = StdioServerParameters(
            command=self.python_exe,
            args=[self.server_script_path],
            env=self.env,
        )
        read_stream, write_stream = await self._stack.enter_async_context(stdio_client(params))
        self.session = await self._stack.enter_async_context(ClientSession(read_stream, write_stream))
        await self.session.initialize()

    async def close(self) -> None:
        await self._stack.aclose()

    async def list_tools(self):
        assert self.session is not None
        return await self.session.list_tools()

    async def call_tool(self, name: str, arguments: Dict[str, Any]):
        assert self.session is not None
        return await self.session.call_tool(name, arguments)

