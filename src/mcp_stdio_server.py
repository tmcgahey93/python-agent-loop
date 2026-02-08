import asyncio
import requests
from mcp.server.fastmcp import FastMCP
from mcp.types import TextContent

mcp = FastMCP("python-mcp-demo")

GO_APP_BASE_URL = "http://localhost:8080"

@mcp.tool()
async def add_numbers(a: float, b: float) -> TextContent:
    """
    Add two numbers and return the result.
    """
    resp = requests.post(
        f"{GO_APP_BASE_URL}/sum",
        json={"a": a, "b": b},
        timeout=5
    )
    resp.raise_for_status()
    data = resp.json() #expects {"result": <number>}
    return TextContent(text=f"Result from Go: {data['result']}")

@mcp.tool()
async def echo_message(message: str) -> TextContent:
    """
    Echo a message and return metadata about it.
    """
    resp = requests.post (
        f"{GO_APP_BASE_URL}/echo",
        json={"message": message},
        timeout=5,
    )
    resp.raise_for_status()
    data = resp.json() #expects {echoed, length, uppercase}

    text = (
        f"Echoed: {data['echoed']}\n"
        f"Length: {data['length']}\n"
        f"Uppercase: {data['uppercase']}"
    )
    return TextContent(text=text)

if __name__ == "__main__":
    # stdio transport so an MCP host can spawn this process
    mcp.run(transport="stdio")