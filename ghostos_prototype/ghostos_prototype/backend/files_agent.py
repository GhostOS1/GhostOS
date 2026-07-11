"""
agents/files_agent.py
Files Agent - structured (non-embedding) file lookups: "what's in my
Downloads/Desktop/..." folder listings, and exact filename/path matches
("find resume.pdf", "where's the invoice"). Split out of app.py's old
folder_agent()/file_agent() so file-lookup logic lives in one importable
place instead of being two of many functions in one large module.

Neither function here ever touches embeddings or Ollama - both are plain
SQLite lookups (see vectorstore.py), which is exactly why the Intent
Router (router.py) checks for these before falling through to the more
expensive semantic_query path.
"""

import re

from vectorstore import search_files_keywords, search_files_by_name, get_files_by_folder

STANDARD_FOLDERS = ["Desktop", "Documents", "Downloads", "Pictures", "Videos", "Music"]
FOLDER_ALIASES = {
    "downloads": "Downloads", "download": "Downloads", "downloads folder": "Downloads",
    "desktop": "Desktop",
    "documents": "Documents", "docs folder": "Documents",
    "pictures": "Pictures", "photos folder": "Pictures",
    "videos": "Videos", "videos folder": "Videos",
    "music": "Music",
}

EXACT_FILE_EXT_RE = re.compile(r"\.\w{2,5}\b")  # e.g. ".pdf", ".docx", ".exe"
EXACT_FILE_TRIGGER_PHRASES = ("find ", "where is", "where's", "locate ")


def detect_folder(text: str) -> str | None:
    """Returns the canonical standard-folder name (e.g. 'Downloads') if the
    message is asking to browse/list one of them, else None."""
    for alias, folder in FOLDER_ALIASES.items():
        if alias in text:
            return folder
    return None


def looks_like_exact_file_lookup(text: str) -> bool:
    """A filename with an extension, or an explicit find/locate/where-is
    phrasing, means the user already knows (part of) the name - that's a
    direct SQLite name/path match, never a semantic search."""
    return bool(EXACT_FILE_EXT_RE.search(text)) or any(p in text for p in EXACT_FILE_TRIGGER_PHRASES)


def folder_agent(query: str) -> tuple[list, str | None]:
    """Direct structured lookup for 'what's in my Downloads/Desktop/...'
    requests - a plain SQLite path match, never touches embeddings or the
    vector store. Returns (files, folder_name)."""
    folder_name = detect_folder(query.lower())
    if not folder_name:
        return [], None
    return get_files_by_folder(folder_name), folder_name


def file_agent(query: str) -> list:
    """Filename/path keyword matches - catches files whose content was
    never embedded (installers, unsupported extensions, etc.), and still
    works even if Ollama's embedding model is down."""
    matches = search_files_keywords(query.casefold(), limit=5)
    if matches:
        return matches
    # Reuse the typo-tolerant metadata search for chat questions after
    # stripping conversational filler and file extensions.
    words = re.findall(r"[a-z0-9_-]+", query.casefold())
    stop = {"where", "what", "which", "find", "locate", "show", "open", "file", "files", "is", "my", "the", "a", "an", "please"}
    meaningful = [w for w in words if w not in stop and len(w) > 2]
    fuzzy = []
    seen = set()
    for term in meaningful[:6]:
        for item in search_files_by_name(term, limit=5):
            if item["path"] not in seen:
                seen.add(item["path"])
                fuzzy.append(item)
    return fuzzy[:5]
