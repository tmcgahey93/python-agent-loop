import json
import uuid 
from typing import Any, Dict, List, Optional, Tuple

import requests


class MCPHttpClient:
    """
    Minimal MCP client for Streamable HTTP transport.

    - Sends JSON-RPC via HTTP POST to the MCP endpoint.  [oai_citation:2‡Model Context Protocol](https://modelcontextprotocol.io/specification/2025-03-26/basic/transports)
    - Uses tools/list and tools/call JSON-RPC methods.  [oai_citation:3‡Model Context Protocol](https://modelcontextprotocol.io/specification/2025-06-18/server/tools)
    - Stores Mcp-Session-Id if server provides it.  [oai_citation:4‡Model Context Protocol](https://modelcontextprotocol.io/specification/2025-03-26/basic/transports)
    """

    def __init__(self, endpoint_url: str, bearer_token: Optional[str] = None, timeout_s: int = 60):
        self.endpoint_url = endpoint_url
        self.timeout_s = timeout_s
        self.session_id: Optional[str] = None
        
        self._session = requests.Session()
        self._session.headers.update(
            {
                # Spec says client MUST include Accept with application/json and text/event-stream.  [oai_citation:5‡Model Context Protocol](https://modelcontextprotocol.io/specification/2025-03-26/basic/transports)
                "Accept": "application/json, text/event-stream",
                "Content-Type": "application/json",
            }
        )
        if bearer_token:
            self._session.headers["Authorization"] = f"Bearer {bearer_token}"

    def _post_jsonrpc(self, method: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        msg_id = str(uuid.uuid4())
        payload: Dict[str, Any] = {"jsonrpc": "2.0", "id": msg_id, "method": method}
        if params is not None:
            payload["params"] = params

        headers = {}
        if self.session_id:
            headers["Mcp-Session-Id"] = self.session_id # session management header  [oai_citation:6‡Model Context Protocol](https://modelcontextprotocol.io/specification/2025-03-26/basic/transports)

        resp = self._session.post(
            self.endpoint_url,
            data=json.dumps(payload),
            headers=headers,
            timeout=self.timeout_s,
        )
        resp.raise_for_status()

        # If the server returns a session id, store it for future calls.  [oai_citation:7‡Model Context Protocol](https://modelcontextprotocol.io/specification/2025-03-26/basic/transports)
        sid = resp.headers.get("Mcp-Session-Id")
        if sid:
            self.session_id = sid

        ctype = (resp.headers.get("Content-Type") or "").lower()

        # Many servers will just return JSON.
        if "application/json" in ctype:
            data = resp.json()
            if "error" in data:
                raise RuntimeError(f"MCP error: {data['error']}")
            return data
        
        # Some servers may reply with SSE (text/event-stream) for streaming responses.  [oai_citation:8‡Model Context Protocol](https://modelcontextprotocol.io/specification/2025-03-26/basic/transports)
        # This tiny client supports a simple case: read events until we see a JSON-RPC response with matching id.
        if "text/event-stream" in ctype:
            return self._read_sse_for_response(expected_id=msg_id, resp=resp)
        
        raise RuntimeError(f"Unexpected Content-Type from MCP server: {ctype}")
    
    def _reead_sse_for_response(self, expected_id: str, resp: requests.Response) -> Dict[str, Any]:
        # Very small SSE parser: looks for lines starting with "data:" and parses JSON.
        for raw_line in resp.iter_lines(decode_unicode=True):
            if not raw_line:
                continue
            line = raw_line.strip()
            if not line.startswith("data:"):
                continue
            data_str = line[len("data") :].strip()
            try:
                evt = json.loads(data_str)
            except Exception:
                continue

            if isinstance(evt, dict) and evt.get("jsonrpc") == "2.0" and evt.get("id") == expected_id:
                if "error" in evt:
                    raise RuntimeError(f"MCP error: {evt['error']}")
                return evt
        
        raise RuntimeError("SSE stream ended before receiving expected JSON-RPC response")
    
    def intialization(self) -> None:
        """
        MCP servers typically require initialization (lifecycle handshake).
        Many servers accept it; some are stateless and may ignore it.

        If your server requires it, you MUST do it before tools/list/tools/call.
        """
        # NOTE: The exact initialize params depend on the server + protocol revision.
        # If your server complains, we’ll adjust based on its error response.
        self._post_jsonrpc("initialization", {"clientInfor": {"name": "tiny-agent", "version": "0.1"}})

    def list_tools(self, cursor: Optional[str] = None) -> Dict[str, Any]:
        params = {"cursor": cursor} if cursor else {}
        return self._post_jsonrpc("tools/list", params) # tools/list request  [oai_citation:9‡Model Context Protocol](https://modelcontextprotocol.io/specification/2025-06-18/server/tools)

    def call_tool(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        # tools/call is how tools are invoked.  [oai_citation:10‡Model Context Protocol](https://modelcontextprotocol.io/specification/2025-06-18/server/tools)
        return self._post_jsonrpc("tools/call", {"name": name, "arguments": arguments})

