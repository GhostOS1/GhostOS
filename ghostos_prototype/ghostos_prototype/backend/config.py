"""Central runtime configuration for GhostOS."""

import os

from settings_store import get_settings


def _bounded_int(name: str, default: int, minimum: int, maximum: int) -> int:
    """Read a consumer-facing integer setting without allowing unsafe extremes."""
    try:
        value = int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        value = default
    return max(minimum, min(value, maximum))


_local_settings = get_settings()

HOST = os.getenv("GHOSTOS_HOST", "127.0.0.1")
PORT = int(os.getenv("GHOSTOS_PORT", "5000"))
OLLAMA_BASE_URL = os.getenv(
    "OLLAMA_BASE_URL", str(_local_settings["ollama_url"])
).rstrip("/")
CHAT_MODEL = os.getenv("GHOSTOS_CHAT_MODEL", str(_local_settings["chat_model"]))
EMBED_MODEL = os.getenv("GHOSTOS_EMBED_MODEL", str(_local_settings["embedding_model"]))
REQUEST_TIMEOUT_SECONDS = int(os.getenv("GHOSTOS_REQUEST_TIMEOUT", "120"))

# Indexing defaults are deliberately conservative. File workers overlap disk
# reads/text extraction, while the smaller embedding-worker limit protects a
# consumer Ollama instance from being flooded with concurrent model requests.
INDEX_MAX_FILE_BYTES = _bounded_int(
    "GHOSTOS_INDEX_MAX_FILE_BYTES", 25 * 1024 * 1024, 1 * 1024 * 1024, 512 * 1024 * 1024
)
INDEX_MAX_CHUNKS_PER_FILE = _bounded_int("GHOSTOS_INDEX_MAX_CHUNKS_PER_FILE", 64, 1, 512)
INDEX_FILE_WORKERS = _bounded_int("GHOSTOS_INDEX_FILE_WORKERS", 4, 1, 16)
INDEX_EMBEDDING_WORKERS = _bounded_int("GHOSTOS_INDEX_EMBEDDING_WORKERS", 2, 1, 4)
INDEX_MAX_PENDING_FILES = _bounded_int(
    "GHOSTOS_INDEX_MAX_PENDING_FILES", INDEX_FILE_WORKERS * 2, INDEX_FILE_WORKERS, 64
)
INDEX_FAILED_FILES_LIMIT = _bounded_int("GHOSTOS_INDEX_FAILED_FILES_LIMIT", 100, 1, 1000)
