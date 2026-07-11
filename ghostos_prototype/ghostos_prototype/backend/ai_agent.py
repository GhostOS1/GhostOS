"""
agents/ai_agent.py
AI Agent - the final generation step. Every other agent only gathers or
organizes information (files, timeline, browser, memory, system); this is
the only one that actually talks to the LLM. Builds the Ollama /api/chat
payload (system prompt + rolling conversation history + this turn's
context-injected prompt) and streams the reply back token by token.

Split out of app.py's old inline generate() closure - same request/
response contract, just relocated so the Flask route doesn't also have to
own "how do I talk to Ollama."
"""

import json
import requests
from config import OLLAMA_BASE_URL, CHAT_MODEL, REQUEST_TIMEOUT_SECONDS

OLLAMA_CHAT_URL = f"{OLLAMA_BASE_URL}/api/chat"


def stream_reply(system_prompt: str, history: list, full_prompt: str):
    """Yields response tokens as they stream in from Ollama. `history` is
    the rolling conversation window (clean role/content pairs, see
    agents/memory_agent.py) - sent as-is, never re-wrapped in the "User
    Message / Indexed Context" template that's only for the *current*
    turn's fresh retrieval. Replaying old context blocks every turn would
    balloon the prompt and re-inject stale search results as if they were
    still current.
    """
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(history)
    messages.append({"role": "user", "content": full_prompt})

    payload = {"model": CHAT_MODEL, "messages": messages, "stream": True}

    with requests.post(OLLAMA_CHAT_URL, json=payload, stream=True, timeout=REQUEST_TIMEOUT_SECONDS) as r:
        r.raise_for_status()
        for line in r.iter_lines(decode_unicode=True):
            if not line:
                continue
            try:
                chunk = json.loads(line)
            except Exception as e:
                print("[ai_agent] JSON error:", e)
                continue
            token = chunk.get("message", {}).get("content", "")
            if token:
                yield token
