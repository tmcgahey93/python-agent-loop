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

TOOLS: Dict[str, Tool] = {
    "calc": Tool(
        name="calc",
        description="Evaluate a simple arithmetic expression.",
        args_schema={"expression": "string"},
        fn=tool_calc,
    ),
    "read_file": Tool(
        name="read_file",
        description="Read a UTF-8 text file from disk.",
        args_schema={"path": "string"},
        fn=tool_read_file,
    ),
    "write_file": Tool(
        name="write_file",
        description="Write a UTF-8 text file to disk (overwrites).",
        args_schema={"path": "string", "content": "string"},
        fn=tool_run_shell,
    ),
}

# --------------------------
# Agent protocal
# --------------------------
SYSTEM = """You are a tiny tool-using agent.

You MUST respond with exactly one JSON object (no markdown, no extra text).
    Choose one of these shapes:

1) Tool call:
{"type":"tool","name":"<tool_name>","args":{...}}

2) Final answer:
{"type":"final","answer":"..."}

Rules:
- Only call tools that exist in the tool list.
- If you need info from the environment/files, call a tool rather than guessing.
- Keep steps small and verify with tools when relevant.
"""

def tools_manifest() -> str:
    tool_list = [
        {"name": t.name, "description": t.description, "args_schema": t.args_schema}
        for t in TOOLS.values()
    ]
    return json.dumps(tool_list, indent=2)

def parse_agent_json(text: str) -> Tuple[Optional[Dict[str, Any]], str]:
    """
    Tries hard to find a JSON object in model output.
    Returns (obj, error_messsage)
    """
    text=text.strip()
    try:
        return json.loads(text), ""
    except Exception:
        #attempt to extract first {...} block
        m = re.search(r"\{.*}", text, re.DOTALL)
        if not m:
            return None, "No JSON object found in response."
        try:
            return json.loads(m.group(0)), ""
        except Exception as e:
            return None, f"Failed to parse extracted JSON: {e}"
        
def run_agent(task: str, model: str = "llama3.2", max_steps: int = 12) -> str:
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"Available tools:\n{tools_manifest()}"},
        {"role": "user", "content": f"Task: {task}"},
    ]

    for step in range(1, max_steps + 1):
        raw = ollama_chat(model=model, messages=messages)
        obj, err = parse_agent_json(raw)

        if obj is None:
            # Tell the model it violated protocol and try again
            messages.append({"role": "assistant", "content": raw})
            messages.append({"role": "user", "content": f"Your last response was invalid JSON. Error: {err}. Respond again with ONLY one JSON object."})
            continue

        if obj.get("type") == "final":
            return obj.get("answer", "")
        
        if obj.get("type") != "tool":
            messages.append({"role": "assistant", "content": raw})
            messages.append({"role": "user", "content": 'Invalid "type". Use {"type":"tool",...} or {"type":"final",...}.'})
            continue

        name = obj.get("name")
        args = obj.get("args", {})

        if name not in TOOLS:
            messages.append({"role": "assistant", "content": raw})
            messages.append({"role": "user", "content": f"Tool {name!r} does not exist. Choose from: {list(TOOLS.keys())}."})
            continue

        tool = TOOLS[name]
        try:
            result = tool.fn(**args)
        except TypeError as e:
            result = f"ERROR: bad args for tool {name}: {e}"
        except Exception as e:
            result = f"ERROR: tool {name} failed: {e}"

        # Feed tool result back to the model as "observation"
        messages.append({"role": "assistant", "content": raw})
        messages.append({"role": "user", "content": f"Observation from tool {name}: {result}"})

    return f"Stopped after {max_steps} steps without a final answer."

if __name__ == "__main__":
    # Try a couple tasks:
    print(run_agent("Compute (17*3) + 2, and return the number only."))
    print(run_agent("Write a file ./out/hello.txt that contains 'hello from the agent' then read it to confirm contents."))