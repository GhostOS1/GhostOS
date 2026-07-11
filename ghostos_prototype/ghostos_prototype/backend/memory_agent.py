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

import threading
from collections import deque
from pathlib import Path

from embeddings import get_embedding
from vectorstore import hybrid_search, rerank

TOP_K_CHUNKS = 5
RETRIEVAL_POOL_SIZE = 20  # candidates pulled per retriever before fusion + re-ranking narrows to TOP_K_CHUNKS
MIN_RERANK_SCORE = 0.08   # final post-rerank cutoff (0-1 scale) - drops fused candidates that still aren't a real match

CONVERSATION_HISTORY_MAXLEN = 12  # ~6 user/assistant turn pairs kept as rolling context sent to Gemma
_lock = threading.Lock()
_conversation_history: deque = deque(maxlen=CONVERSATION_HISTORY_MAXLEN)
_session_context = {
    "last_file": None,         # {"name": ..., "path": ...}
    "last_folder": None,       # canonical folder name, e.g. "Downloads"
    "last_browser_tab": None,  # {"title": ..., "url": ...}
    "last_topic": None,        # last non-small-talk user message
}

# Words short enough and pronoun-y enough that, on their own, they can only
# be resolved against something already established earlier in the
# conversation - "open it" is meaningless as a fresh retrieval query, but
# meaningful if we remember what "it" was.
REFERENCE_PRONOUNS = {"it", "that", "this", "that one", "this one"}


def search_agent(query: str) -> list:
    """Hybrid content search: vector + BM25, fused via RRF, re-ranked
    against the raw query text. Covers indexed files AND browser history
    alike - browser_connector.py stores visited pages as chunks in the
    same table (see vectorstore.search()), tagged with
    source_type='browser_history_<browser>'."""
    try:
        query_embedding = get_embedding(query)
        fused = hybrid_search(query_embedding, query, top_k=TOP_K_CHUNKS, pool_size=RETRIEVAL_POOL_SIZE)
        reranked = rerank(query, fused, top_k=TOP_K_CHUNKS)
        kept = [m for m in reranked if m["score"] >= MIN_RERANK_SCORE]
        print(f"[memory_agent.search_agent] query={query!r} kept={len(kept)}/{len(reranked)}")
        return kept
    except Exception as e:
        print(f"[memory_agent.search_agent] skipped (embedding failed): {e}")
        return []


def wants_reference_resolution(text: str) -> bool:
    """True if the message looks like it's pointing back at something
    already discussed rather than describing a new search from scratch."""
    words = text.split()
    if not words or len(words) > 6:
        return False
    return bool(set(words) & REFERENCE_PRONOUNS)


def snapshot_session() -> dict:
    """A point-in-time copy of session memory, safe to read from outside
    the lock afterward."""
    with _lock:
        return dict(_session_context)


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
        if filename_matches:
            top = filename_matches[0]
            _session_context["last_file"] = {"name": top["name"], "path": top["path"]}
        if file_content_matches:
            top = file_content_matches[0]
            _session_context["last_file"] = {"name": Path(top["source_path"]).name, "path": top["source_path"]}
        if browser_matches:
            top = browser_matches[0]
            _session_context["last_browser_tab"] = {
                "title": top["content"].split("\n")[0].removeprefix("Visited: "),
                "url": top["source_path"],
            }
        if intent not in ("greeting", "thanks", "farewell", "reference_query"):
            _session_context["last_topic"] = user_message


def build_reference_context(session_snapshot: dict) -> str | None:
    """Turns remembered session slots into a context block the LLM can use
    to resolve "it"/"that"/"this". Does NOT imply GhostOS can act on
    whatever's referenced - the system prompt's "no file operations" rule
    still applies regardless of whether the reference is known."""
    parts = []
    last_file = session_snapshot.get("last_file")
    last_folder = session_snapshot.get("last_folder")
    last_tab = session_snapshot.get("last_browser_tab")

    if last_file:
        parts.append(f'"it"/"that"/"this" most recently referred to the file '
                      f'"{last_file["name"]}", located at: {last_file["path"]}')
    if last_folder:
        parts.append(f'The most recently discussed folder was: {last_folder}')
    if last_tab:
        parts.append(f'The most recently discussed browser page was: '
                      f'"{last_tab["title"]}" ({last_tab["url"]})')

    if not parts:
        return None
    return "[Conversation memory - use this to resolve pronouns like \"it\"/\"that\"/\"this\"]\n" + "\n".join(parts)


def reset():
    """Clears conversation history and session memory - the backend
    equivalent of starting a new chat."""
    with _lock:
        _conversation_history.clear()
        _session_context.update({
            "last_file": None, "last_folder": None,
            "last_browser_tab": None, "last_topic": None,
        })
