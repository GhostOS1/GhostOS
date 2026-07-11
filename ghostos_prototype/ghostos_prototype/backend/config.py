"""Central runtime configuration for GhostOS."""

import os


HOST = os.getenv("GHOSTOS_HOST", "127.0.0.1")
PORT = int(os.getenv("GHOSTOS_PORT", "5000"))
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
CHAT_MODEL = os.getenv("GHOSTOS_CHAT_MODEL", "gemma4:e2b")
EMBED_MODEL = os.getenv("GHOSTOS_EMBED_MODEL", "nomic-embed-text")
REQUEST_TIMEOUT_SECONDS = int(os.getenv("GHOSTOS_REQUEST_TIMEOUT", "120"))

