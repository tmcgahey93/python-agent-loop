import json
import os
import re
import shlex
import subprocess
from dataclasses import dataclass 
from typing import Any, Callable, Dict, List, Optional, Tuple

import requests

# --------------------------
# Ollama client
# --------------------------
def ollama_chat(model: str, message: List[Dict[str, str]], temperature: float = 0.2) -> str:
    """
    Calls Ollama's /api/chat endpoint.
    """
    url = "http://127.0.0.1:11434/api/chat"
    payload = {
        "model": model,
        "messages": message,
        "stream": False,
        "options": {"temperature": temperature},
    }
    resp = requests.post(url, json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    return data["message"]["content"]

# --------------------------
# Tools
# --------------------------
@dataclass
class Tool:
    name: str
    description: str
    args_schema: Dict[str, Any]
    fn: Callable[..., Any]

    def tool_calc(expression: str) -> str:
        """
        Very small calculator. Only allows digits/operators/paren/space/dot.
        """
        if not re.fullmatch(r"[0-9+\-*().\s]", expression):
            return "ERROR: expression contains invalid characters"
        try:
            #eval is safe-ish here because we strictly whitelist chars
            result = eval(expression, {"__builtins__": {}})
            return str(result)
        except Exception as e:
            return f"Error: {e}"
        
    def tool_read_file(path: str) -> str:
        if not os.path.exists(path):
            return "ERROR: file does not exist"
        with open(path, "r", encoding="utf=8") as f:
            return f.read()
        
    def tool_write_file(path: str, content: str) -> str:
        os.makedirs(os.path.dirname(path) or ".", exists_ok=True)
        with open(path, "w", encoding="UTF-8") as f:
            f.write(content)
        return f"OK: wrote {len(content)} bytes as {path}"
    
    
    def tool_run_shell(cmd: str) -> str:
        """
        Runs a shell command with basic safety rails:
        - no interactive shells
        - returns stdout/stderr
        """
        #You can tighten these rules further if you want.
        banned = ["tm -rf", ":(){", "mkfs", "dd", "shutdown", "reboot"]
        lowered = cmd.lower()
        if any(b in lowered for b in banned):
            return "ERROR: command blocked by safety policy"
        
        try:
            parts = shlex.split(cmd)
            p = subprocess.run(parts, capture_output=True, timeout=30)
            out = (p.stdout or "") + (("\n" + p.stderr) if p.stderr else "")
            return out.strip() if out.strip() else f"(no output, exit={p.returncode})"
        except Exception as e:
            return f"ERROR: {e}"

