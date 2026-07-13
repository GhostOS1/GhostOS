"""
agents/memory_agent.py
Memory Agent - two related jobs bundled together, matching the v2.0
architecture doc's own description of this agent ("Memory Search, Recall,
Summaries, Context"):

  1. search_agent(): hybrid content search (dense vector + BM25 keyword),
     fused with Reciprocal Rank Fusion and lexically re-ranked before
     anything reaches Gemma - see vectorstore.hybrid_search()/rerank().
  2. Conversation memory: a rolling dialogue history plus a small set of
     session slots (last file/folder/browser tab/topic) used to resolve
     pronouns like "it"/"that"/"this" across turns ("Find report.pdf" ->
     "Open it").

Both are "memory" in the sense the architecture doc means it - recalling
either previously-indexed content or recent conversational context - so
they live in one module rather than being split arbitrarily.

State here is deliberately global/in-process, not per-session: GhostOS is
a single-user, single-machine assistant, not a multi-tenant chatbot - see
watcher.py and connect_system.py for the same pattern of module-level
state elsewhere in this codebase.
"""

import copy
import ntpath
import re
import threading
from collections import deque
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

from embeddings import get_embedding
from vectorstore import bm25_search, hybrid_search, rerank

TOP_K_CHUNKS = 5
RETRIEVAL_POOL_SIZE = 20  # candidates pulled per retriever before fusion + re-ranking narrows to TOP_K_CHUNKS
MIN_RERANK_SCORE = 0.22   # calibrated cutoff; rank alone can no longer make an unrelated result pass

CONVERSATION_HISTORY_MAXLEN = 12  # ~6 user/assistant turn pairs kept as rolling context sent to Gemma
_lock = threading.Lock()
_conversation_history: deque = deque(maxlen=CONVERSATION_HISTORY_MAXLEN)
def _empty_session_context() -> dict:
    return {
        "last_file": None,          # {"name": ..., "path": ...}
        "last_pdf": None,           # most recently surfaced PDF
        "recent_files": [],         # newest first, unique by path
        "last_folder": None,        # canonical/custom folder name
        "last_folder_path": None,   # concrete folder path for safe actions
        "last_browser_tab": None,   # compatibility name used by app.py
        "last_browser_page": None,  # clearer alias for prompts/tests
        "last_url": None,
        "recent_browser_pages": [],
        "last_application": None,
        "last_timeline_event": None,
        "last_project": None,
        "last_topic": None,
        "last_user_intent": None,
    }


_session_context = _empty_session_context()

# Words short enough and pronoun-y enough that, on their own, they can only
# be resolved against something already established earlier in the
# conversation - "open it" is meaningless as a fresh retrieval query, but
# meaningful if we remember what "it" was.
REFERENCE_PATTERNS = (
    re.compile(r"\b(?:it|this|that|them|that one|this one)\b", re.IGNORECASE),
    re.compile(
        r"\b(?:that|this|the previous|previous|the last|last|earlier)\s+"
        r"(?:file|document|pdf|folder|directory|page|website|site|link|url)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\b(?:file|folder|document|pdf|page|website|link)\s+(?:we|i)\s+(?:discussed|mentioned|found|visited|opened)\b", re.IGNORECASE),
    re.compile(r"\b(?:page|website|site)\s+i\s+visited\s+earlier\b", re.IGNORECASE),
)


def _canonical_source(value: str) -> str:
    raw = (value or "").strip()
    if raw.casefold().startswith(("http://", "https://")):
        parts = urlsplit(raw)
        # Fragments and common trailing-slash differences are not distinct
        # sources for answer attribution.
        return urlunsplit((parts.scheme.casefold(), parts.netloc.casefold(), parts.path.rstrip("/") or "/", parts.query, ""))
    return raw.replace("/", "\\").casefold()


def deduplicate_results(results: list[dict], limit: int = TOP_K_CHUNKS) -> list[dict]:
    """Keep the best result per local file/URL while preserving rank."""
    ordered = sorted(results, key=lambda item: float(item.get("score", 0.0)), reverse=True)
    seen: set[str] = set()
    unique = []
    for item in ordered:
        key = _canonical_source(str(item.get("source_path") or ""))
        if not key:
            # A missing source cannot be safely attributed, so it should not
            # be given to the model as retrieved local evidence.
            continue
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
        if len(unique) >= limit:
            break
    return unique


def _derive_folder_path(folder_matches: list[dict], folder_name: str | None) -> str | None:
    """Derive the concrete directory represented by a folder listing.

    Prefer the named path segment (so a listing that includes nested files
    still resolves to ``Downloads`` itself), then fall back to the common
    parent of the returned files.  No path is invented when there are no
    actual results.
    """
    paths = [str(item.get("path") or "") for item in folder_matches]
    paths = [path for path in paths if path]
    if not paths:
        return None

    wanted = (folder_name or "").casefold()
    if wanted:
        for raw_path in paths:
            normalized = raw_path.replace("/", "\\")
            parts = list(Path(normalized).parts)
            for index, part in enumerate(parts):
                if part.strip("\\/").casefold() == wanted:
                    return str(Path(*parts[:index + 1]))

    try:
        parents = [ntpath.dirname(path.replace("/", "\\")) for path in paths]
        return ntpath.commonpath(parents) or None
    except ValueError:
        # Results from different drives have no safe common directory.
        return ntpath.dirname(paths[0].replace("/", "\\")) or None


def search_agent(query: str) -> list:
    """Hybrid content search: vector + BM25, fused via RRF, re-ranked
    against the raw query text. Covers indexed files AND browser history
    alike - browser_connector.py stores visited pages as chunks in the
    same table (see vectorstore.search()), tagged with
    source_type='browser_history_<browser>'."""
    normalized_query = " ".join((query or "").split())
    if not normalized_query:
        return []

    fused = []
    try:
        query_embedding = get_embedding(normalized_query)
        fused = hybrid_search(
            query_embedding, normalized_query,
            top_k=TOP_K_CHUNKS, pool_size=RETRIEVAL_POOL_SIZE,
        )
    except Exception as embedding_error:
        # Keyword recall is still useful and fully local when Ollama or the
        # embedding model is unavailable.  Previously one embedding error
        # disabled all content retrieval, including BM25.
        print(f"[memory_agent.search_agent] embedding unavailable; using BM25: {embedding_error}")

    try:
        if not fused:
            fused = bm25_search(normalized_query, top_k=RETRIEVAL_POOL_SIZE)
        reranked = rerank(normalized_query, fused, top_k=RETRIEVAL_POOL_SIZE)
        relevant = [item for item in reranked if float(item.get("score", 0.0)) >= MIN_RERANK_SCORE]
        kept = deduplicate_results(relevant, limit=TOP_K_CHUNKS)
        print(f"[memory_agent.search_agent] query={normalized_query!r} kept={len(kept)}/{len(reranked)}")
        return kept
    except Exception as retrieval_error:
        print(f"[memory_agent.search_agent] retrieval failed: {retrieval_error}")
        return []


def wants_reference_resolution(text: str) -> bool:
    """True if the message looks like it's pointing back at something
    already discussed rather than describing a new search from scratch."""
    normalized = " ".join((text or "").casefold().split())
    words = normalized.split()
    if not words or len(words) > 14:
        return False
    return any(pattern.search(normalized) for pattern in REFERENCE_PATTERNS)


def snapshot_session() -> dict:
    """A point-in-time copy of session memory, safe to read from outside
    the lock afterward."""
    with _lock:
        return copy.deepcopy(_session_context)


def get_history() -> list:
    with _lock:
        return list(_conversation_history)


def record_turn(user_message: str, assistant_reply: str):
    """Stores the clean (un-templated) turn - not the full retrieval-
    injected prompt - so old context blocks don't get replayed forever and
    balloon future prompts."""
    with _lock:
        _conversation_history.append({"role": "user", "content": user_message})
        _conversation_history.append({"role": "assistant", "content": assistant_reply})


def update_session_from_turn(intent: str, folder_matches: list, folder_name: str | None,
                              filename_matches: list, file_content_matches: list,
                              browser_matches: list, user_message: str):
    """Updates session slots from this turn's results. Only overwrites a
    slot when this turn actually surfaced something for it - a
    reference_query turn (or one with no matches) leaves previous slots
    alone, so "it" still means the same thing two turns later if nothing
    new was found in between."""
    with _lock:
        if folder_matches:
            _session_context["last_folder"] = folder_name
            _session_context["last_folder_path"] = _derive_folder_path(folder_matches, folder_name)

        surfaced_files = []
        for match in filename_matches:
            path = str(match.get("path") or "")
            if path:
                surfaced_files.append({"name": str(match.get("name") or Path(path).name), "path": path})
        for match in file_content_matches:
            path = str(match.get("source_path") or "")
            if path:
                surfaced_files.append({"name": Path(path).name, "path": path})

        if surfaced_files:
            combined = surfaced_files + list(_session_context.get("recent_files") or [])
            recent_files = []
            seen_files: set[str] = set()
            for item in combined:
                key = _canonical_source(item["path"])
                if key and key not in seen_files:
                    seen_files.add(key)
                    recent_files.append(item)
                if len(recent_files) >= 12:
                    break
            _session_context["recent_files"] = recent_files
            _session_context["last_file"] = recent_files[0]
            latest_pdf = next(
                (item for item in recent_files if Path(item["name"]).suffix.casefold() == ".pdf"),
                None,
            )
            if latest_pdf:
                _session_context["last_pdf"] = latest_pdf

        if browser_matches:
            top = browser_matches[0]
            page = {
                "title": top["content"].split("\n")[0].removeprefix("Visited: "),
                "url": top["source_path"],
            }
            _session_context["last_browser_tab"] = page
            _session_context["last_browser_page"] = page
            _session_context["last_url"] = page["url"]
            pages = [page] + list(_session_context.get("recent_browser_pages") or [])
            unique_pages = []
            seen_pages: set[str] = set()
            for item in pages:
                key = _canonical_source(str(item.get("url") or ""))
                if key and key not in seen_pages:
                    seen_pages.add(key)
                    unique_pages.append(item)
                if len(unique_pages) >= 12:
                    break
            _session_context["recent_browser_pages"] = unique_pages
        if intent not in ("greeting", "thanks", "farewell", "reference_query"):
            _session_context["last_topic"] = user_message
        _session_context["last_user_intent"] = intent


def build_reference_context(session_snapshot: dict) -> str | None:
    """Turns remembered session slots into a context block the LLM can use
    to resolve "it"/"that"/"this". Does NOT imply GhostOS can act on
    whatever's referenced - the system prompt's "no file operations" rule
    still applies regardless of whether the reference is known."""
    parts = []
    last_file = session_snapshot.get("last_file")
    last_pdf = session_snapshot.get("last_pdf")
    last_folder = session_snapshot.get("last_folder")
    last_folder_path = session_snapshot.get("last_folder_path")
    last_tab = session_snapshot.get("last_browser_page") or session_snapshot.get("last_browser_tab")

    if last_file:
        parts.append(f'Most recently surfaced file: "{last_file["name"]}" (path: {last_file["path"]})')
    if last_pdf:
        parts.append(f'The previous/most recently discussed PDF: "{last_pdf["name"]}" (path: {last_pdf["path"]})')
    if last_folder:
        folder_detail = f' (path: {last_folder_path})' if last_folder_path else ""
        parts.append(f'The most recently discussed folder was: {last_folder}{folder_detail}')
    if last_tab:
        parts.append(f'The most recently discussed browser page was: '
                      f'"{last_tab["title"]}" ({last_tab["url"]})')

    if not parts:
        return None
    return (
        "[Local conversation entities - resolve references only from these values; "
        "do not invent another file/page]\n" + "\n".join(f"- {part}" for part in parts)
    )


def reset():
    """Clears conversation history and session memory - the backend
    equivalent of starting a new chat."""
    with _lock:
        _conversation_history.clear()
        _session_context.clear()
        _session_context.update(_empty_session_context())
