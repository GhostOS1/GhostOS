"""
embeddings.py
Calls the local Ollama API to turn text into vector embeddings.
Requires: ollama pull nomic-embed-text
"""

import requests
from config import OLLAMA_BASE_URL, EMBED_MODEL, REQUEST_TIMEOUT_SECONDS

OLLAMA_URL = f"{OLLAMA_BASE_URL}/api/embeddings"


def get_embedding(text: str) -> list[float]:
    """
    Sends text to the local Ollama embeddings endpoint and returns
    the resulting vector. Raises an exception if Ollama isn't running
    or the model isn't pulled.
    """
    response = requests.post(
        OLLAMA_URL,
        json={"model": EMBED_MODEL, "prompt": text},
        timeout=min(REQUEST_TIMEOUT_SECONDS, 60),
    )
    response.raise_for_status()
    data = response.json()
    return data["embedding"]


def get_embeddings_batch(texts: list[str]) -> list[list[float]]:
    """
    Naive batch wrapper - Ollama's embeddings endpoint takes one
    prompt at a time, so we loop. Fine for a prototype; if indexing
    becomes slow on large folders, this is the first place to optimize
    (e.g. run a small thread pool).
    """
    return [get_embedding(t) for t in texts]
