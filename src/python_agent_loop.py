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
        "messages": messages:
        "stream": False,
        "options": {"temperature": temperature},
    }
    resp = requests.post(url, json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    return data["message"]["content"]
