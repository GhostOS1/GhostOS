"""
vectorstore.py
A minimal local vector store built on SQLite + numpy.
Good enough for a prototype handling up to tens of thousands of chunks.
If you outgrow this (100k+ chunks, need faster search), swap this
module out for sqlite-vec, LanceDB, or Chroma without touching the
rest of the app.
"""

import sqlite3
import json
import math
import re
from difflib import SequenceMatcher
from collections import Counter
import numpy as np
from pathlib import Path

DB_PATH = Path(__file__).parent / "ghostos_memory.db"


def _connect():
    """SQLite connection tuned for concurrent Flask, watcher and indexer use."""
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.execute("PRAGMA busy_timeout=30000")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db():
    conn = _connect()
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_path TEXT,
            source_type TEXT,
            content TEXT,
            embedding TEXT,
            file_hash TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_file_hash ON chunks(file_hash)
    """)

    # One row per real file on disk (metadata only - separate from the
    # per-chunk embeddings above, since not every file gets chunked/embedded,
    # e.g. images/videos are catalogued but not RAG-searchable yet).
    conn.execute("""
        CREATE TABLE IF NOT EXISTS files (
            file_hash TEXT PRIMARY KEY,
            path TEXT,
            name TEXT,
            extension TEXT,
            category TEXT,
            collection TEXT,
            size_bytes INTEGER,
            modified_at TEXT,
            indexed_at TEXT DEFAULT CURRENT_TIMESTAMP,
            embedded INTEGER DEFAULT 0
        )
    """)

    # A flat activity log - the source of truth for the Timeline screen.
    # Populated by the indexer (file events) and the browser connector
    # (browsing events). This is an approximation of "ambient" activity
    # capture using only what this prototype actually observes (files
    # being indexed + browser history) - not live window/OCR tracking yet.
    conn.execute("""
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_type TEXT,
            title TEXT,
            subtitle TEXT,
            app_label TEXT,
            badge_type TEXT,
            path_or_url TEXT,
            timestamp TEXT
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events(timestamp)
    """)
    conn.commit()
    conn.close()


def remove_source(path: str):
    """Remove stale index, metadata and timeline entries after deletion/rename."""
    conn = _connect()
    conn.execute("DELETE FROM chunks WHERE source_path = ?", (path,))
    conn.execute("DELETE FROM files WHERE path = ?", (path,))
    conn.execute("DELETE FROM events WHERE path_or_url = ? AND event_type = 'file_indexed'", (path,))
    conn.commit()
    conn.close()


def file_already_indexed(file_hash: str) -> bool:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.execute("SELECT 1 FROM chunks WHERE file_hash = ? LIMIT 1", (file_hash,))
    result = cur.fetchone()
    conn.close()
    return result is not None


def add_chunk(source_path: str, source_type: str, content: str, embedding: list[float], file_hash: str):
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO chunks (source_path, source_type, content, embedding, file_hash) VALUES (?, ?, ?, ?, ?)",
        (source_path, source_type, content, json.dumps(embedding), file_hash),
    )
    conn.commit()
    conn.close()


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def search(query_embedding: list[float], top_k: int = 5) -> list[dict]:
    """
    Brute-force cosine similarity search over all stored chunks.
    Fine for a prototype; fast enough up to ~50-100k chunks on a
    modern CPU. Beyond that, consider a proper ANN index.
    """
    conn = sqlite3.connect(DB_PATH)
    cur = conn.execute("SELECT source_path, source_type, content, embedding FROM chunks")
    rows = cur.fetchall()
    conn.close()

    if not rows:
        return []

    q = np.array(query_embedding)
    scored = []
    for source_path, source_type, content, embedding_json in rows:
        emb = np.array(json.loads(embedding_json))
        score = _cosine_sim(q, emb)
        scored.append({
            "source_path": source_path,
            "source_type": source_type,
            "content": content,
            "score": score,
        })

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]


# ---------------------------------------------------------------------------
# Hybrid search: BM25 keyword search + RRF fusion + lexical re-ranking
# ---------------------------------------------------------------------------
# search() above is pure dense retrieval (cosine similarity over
# embeddings) - great for "what did I write about X" style fuzzy recall,
# but embeddings blur together documents that are topically similar even
# when the user's actual words point at one specific file (e.g. "lithium
# pump report" scoring Lithium Notes.pdf and Report.docx close together).
# BM25 is the classic fix: exact/near-exact term matching. Fusing both
# and then re-ranking against the raw query catches what either one
# misses alone.

_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "of", "in",
    "on", "at", "to", "for", "and", "or", "my", "me", "i", "it", "this",
    "that", "with", "as", "from", "by", "show", "find", "please",
}


def _tokenize(text: str) -> list[str]:
    tokens = re.findall(r"[a-z0-9]+", (text or "").lower())
    return [t for t in tokens if len(t) > 1 and t not in _STOPWORDS]


def bm25_search(query: str, top_k: int = 20) -> list[dict]:
    """
    Pure-Python BM25 over chunk content, brute-force across every stored
    chunk - same "good enough for a prototype" tradeoff the module
    docstring already makes for cosine similarity in search(). This is
    the keyword-matching leg of hybrid_search(): it catches exact terms
    (filenames, project names, jargon) a dense embedding can wash out.
    """
    conn = sqlite3.connect(DB_PATH)
    cur = conn.execute("SELECT source_path, source_type, content, file_hash FROM chunks")
    rows = cur.fetchall()
    conn.close()

    query_tokens = _tokenize(query)
    if not rows or not query_tokens:
        return []

    doc_tokens = [_tokenize(content) for _, _, content, _ in rows]
    doc_lens = [len(t) for t in doc_tokens]
    avg_len = (sum(doc_lens) / len(doc_lens)) if doc_lens else 0.0
    n_docs = len(rows)

    df = Counter()
    for tokens in doc_tokens:
        present = set(tokens)
        for term in query_tokens:
            if term in present:
                df[term] += 1

    k1, b = 1.5, 0.75
    scored = []
    for (source_path, source_type, content, file_hash), tokens, dl in zip(rows, doc_tokens, doc_lens):
        if dl == 0:
            continue
        tf = Counter(tokens)
        score = 0.0
        for term in query_tokens:
            freq = tf.get(term, 0)
            if freq == 0:
                continue
            idf = math.log(1 + (n_docs - df[term] + 0.5) / (df[term] + 0.5))
            denom = freq + k1 * (1 - b + b * dl / avg_len) if avg_len else freq
            score += idf * (freq * (k1 + 1)) / denom
        if score > 0:
            scored.append({
                "source_path": source_path, "source_type": source_type,
                "content": content, "file_hash": file_hash, "bm25_score": score,
            })

    scored.sort(key=lambda x: x["bm25_score"], reverse=True)
    return scored[:top_k]


def hybrid_search(query_embedding: list[float], query_text: str, top_k: int = 5,
                   pool_size: int = 20, rrf_k: int = 60) -> list[dict]:
    """
    Fuses dense vector search and BM25 keyword search with Reciprocal
    Rank Fusion (RRF) rather than averaging their raw scores - cosine
    similarity (0-1ish) and BM25 (unbounded) live on incompatible
    scales, and RRF sidesteps that entirely by only caring about each
    result's *rank* within each list. rrf_k=60 is the standard constant
    from the original RRF paper; it dampens how much rank 1 vs rank 2
    matters so neither list can dominate on tiny score gaps alone.
    Returns the fused candidate pool (not yet re-ranked - see rerank()).
    """
    vector_results = search(query_embedding, top_k=pool_size)
    keyword_results = bm25_search(query_text, top_k=pool_size)

    def _key(r):
        return (r["source_path"], r["content"][:200])

    rrf_scores: dict = {}
    combined: dict = {}

    for rank, r in enumerate(vector_results):
        k = _key(r)
        rrf_scores[k] = rrf_scores.get(k, 0.0) + 1.0 / (rrf_k + rank + 1)
        combined.setdefault(k, dict(r))
        combined[k].setdefault("bm25_score", 0.0)
        combined[k]["vector_score"] = r["score"]

    for rank, r in enumerate(keyword_results):
        k = _key(r)
        rrf_scores[k] = rrf_scores.get(k, 0.0) + 1.0 / (rrf_k + rank + 1)
        if k in combined:
            combined[k]["bm25_score"] = r["bm25_score"]
        else:
            entry = dict(r)
            entry["vector_score"] = 0.0
            combined[k] = entry

    fused = []
    for k, data in combined.items():
        data["fused_score"] = rrf_scores[k]
        fused.append(data)

    fused.sort(key=lambda x: x["fused_score"], reverse=True)
    return fused[:pool_size]


def rerank(query_text: str, candidates: list[dict], top_k: int = 5) -> list[dict]:
    """
    Lightweight lexical re-ranker. There's no cross-encoder model
    available in this stack (Ollama here only serves the embedding model
    and Gemma for chat), so instead of a neural reranker this scores
    each fused candidate directly against the raw query text - the
    actual point of re-ranking: catching cases where the *retriever*
    put things in a suboptimal order (e.g. "Lithium Notes.pdf" outranking
    "Lithium Pump Report.pdf" for a "show my lithium pump report" query,
    because embedding similarity nudges toward the broader topical match
    rather than the specific document).

    Combines three normalized (0-1) signals:
      - fused_score from hybrid_search (RRF rank across both retrievers)
      - content term overlap (do the query's actual words appear in the
        text, not just something embedding-adjacent to them)
      - filename term overlap, weighted heaviest - the query's words
        showing up in the filename itself is a strong, cheap relevance
        signal a pure content search underweights.

    Sets "score" on each result (matching the field name the rest of the
    app already expects from search()), plus "rerank_score" for clarity.
    """
    if not candidates:
        return []

    query_tokens = set(_tokenize(query_text))
    if not query_tokens:
        return candidates[:top_k]

    max_fused = max((c.get("fused_score", 0.0) for c in candidates), default=0.0)

    rescored = []
    for c in candidates:
        content_tokens = set(_tokenize(c.get("content", "")))
        filename = Path(c.get("source_path", "") or "").name
        filename_tokens = set(_tokenize(filename))

        content_overlap = len(query_tokens & content_tokens) / len(query_tokens)
        filename_overlap = len(query_tokens & filename_tokens) / len(query_tokens)
        norm_fused = (c.get("fused_score", 0.0) / max_fused) if max_fused > 0 else 0.0

        final_score = 0.5 * norm_fused + 0.2 * content_overlap + 0.3 * filename_overlap
        entry = dict(c)
        entry["score"] = final_score
        entry["rerank_score"] = final_score
        rescored.append(entry)

    rescored.sort(key=lambda x: x["score"], reverse=True)
    return rescored[:top_k]


def get_stats() -> dict:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.execute("SELECT COUNT(*) FROM chunks")
    total_chunks = cur.fetchone()[0]
    cur = conn.execute("SELECT COUNT(*), COALESCE(SUM(size_bytes), 0) FROM files")
    total_files, total_bytes = cur.fetchone()
    conn.close()
    return {
        "total_chunks": total_chunks or 0,
        "total_files": total_files or 0,
        "total_bytes": total_bytes or 0,
    }


# ---------------------------------------------------------------------------
# File metadata (Files & Data screen)
# ---------------------------------------------------------------------------

def search_files_by_name(query: str, limit: int = 8) -> list[dict]:
    """
    Simple substring match on filename/path. Powers the search bar's
    instant "files" results - deliberately independent of embeddings/
    Ollama, so search still works (for filenames at least) even if the
    local LLM stack isn't running.
    """
    conn = sqlite3.connect(DB_PATH)
    like = f"%{query}%"
    cur = conn.execute(
        """SELECT path, name, extension, category, size_bytes, modified_at
           FROM files WHERE name LIKE ? OR path LIKE ?
           ORDER BY modified_at DESC LIMIT ?""",
        (like, like, limit),
    )
    rows = cur.fetchall()
    if len(rows) < limit:
        # Typo-tolerant fallback (e.g. "metrologology" vs "metrology").
        # This uses metadata only and remains available without Ollama.
        all_rows = conn.execute(
            "SELECT path, name, extension, category, size_bytes, modified_at FROM files"
        ).fetchall()
        q = query.casefold()
        existing = {row[0] for row in rows}
        fuzzy = []
        for row in all_rows:
            if row[0] in existing:
                continue
            name = (row[1] or "").casefold()
            path = (row[0] or "").casefold()
            score = max(SequenceMatcher(None, q, name).ratio(), SequenceMatcher(None, q, Path(path).stem).ratio())
            if score >= 0.48:
                fuzzy.append((score, row))
        fuzzy.sort(key=lambda item: item[0], reverse=True)
        rows.extend(row for _, row in fuzzy[:max(0, limit - len(rows))])
    conn.close()
    return [
        {"path": path, "name": name, "extension": ext, "category": category,
         "size_bytes": size_bytes, "modified_at": modified_at}
        for path, name, ext, category, size_bytes, modified_at in rows
    ]


_CHAT_STOPWORDS = {
    "where", "is", "my", "the", "a", "an", "find", "show", "me", "file", "files",
    "indexed", "for", "of", "in", "on", "open", "locate", "search", "please",
    "can", "you", "do", "have", "i", "did", "and", "with", "about",
}


def search_files_keywords(message: str, limit: int = 5) -> list[dict]:
    """
    Chat-facing counterpart to search_files_by_name(). /api/chat's RAG only
    searches embedded chunk *content*, which means anything the indexer
    doesn't embed (installers, .exe, anything outside SUPPORTED_EXTENSIONS)
    is invisible to it, and even embedded files won't surface for a "where
    is X" question unless X's wording happens to also appear inside the
    file's content. This pulls significant words out of the user's message
    and matches them against filenames/paths directly, so "where is my
    python installer" can find python-3.11.9-amd64.exe even though that
    file was never embedded.
    """
    words = [w.strip('.,?!"\'') for w in message.lower().split()]
    keywords = [w for w in words if len(w) > 2 and w not in _CHAT_STOPWORDS]
    if not keywords:
        return []

    conn = sqlite3.connect(DB_PATH)
    seen_hashes = set()
    results = []
    for kw in keywords[:6]:  # cap how many keywords we probe, this is a heuristic not a search engine
        like = f"%{kw}%"
        cur = conn.execute(
            """SELECT file_hash, path, name, extension, category, size_bytes, modified_at
               FROM files WHERE name LIKE ? OR path LIKE ?
               ORDER BY modified_at DESC LIMIT ?""",
            (like, like, limit),
        )
        for file_hash, path, name, ext, category, size_bytes, modified_at in cur.fetchall():
            if file_hash in seen_hashes:
                continue
            seen_hashes.add(file_hash)
            results.append({"path": path, "name": name, "extension": ext, "category": category,
                             "size_bytes": size_bytes, "modified_at": modified_at})
        if len(results) >= limit:
            break
    conn.close()
    return results[:limit]


def get_files_by_folder(folder_name: str, limit: int = 50) -> list[dict]:
    """
    Structured lookup for "what's in my Downloads/Desktop/..." style requests.
    Pure SQLite path-substring match - no embeddings, no LLM call. Files
    aren't stored with an explicit 'folder' column, only their full path,
    so this matches on the folder name appearing as a path segment
    (handles both \\Downloads\\ and /Downloads/ separators, so it also
    works if GhostOS ever runs cross-platform).
    """
    conn = sqlite3.connect(DB_PATH)
    like_backslash = f"%\\{folder_name}\\%"
    like_forward = f"%/{folder_name}/%"
    cur = conn.execute(
        """SELECT path, name, extension, category, size_bytes, modified_at
           FROM files WHERE path LIKE ? OR path LIKE ?
           ORDER BY modified_at DESC LIMIT ?""",
        (like_backslash, like_forward, limit),
    )
    rows = cur.fetchall()
    conn.close()
    return [
        {"path": path, "name": name, "extension": ext, "category": category,
         "size_bytes": size_bytes, "modified_at": modified_at}
        for path, name, ext, category, size_bytes, modified_at in rows
    ]


def file_already_catalogued(file_hash: str) -> bool:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.execute("SELECT 1 FROM files WHERE file_hash = ? LIMIT 1", (file_hash,))
    result = cur.fetchone()
    conn.close()
    return result is not None


def is_catalogued_path(path: str) -> bool:
    conn = _connect()
    result = conn.execute("SELECT 1 FROM files WHERE path = ? LIMIT 1", (path,)).fetchone()
    conn.close()
    return result is not None


def upsert_file(file_hash: str, path: str, name: str, extension: str, category: str,
                 collection: str, size_bytes: int, modified_at: str, embedded: bool):
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """INSERT INTO files (file_hash, path, name, extension, category, collection,
                               size_bytes, modified_at, embedded)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
           ON CONFLICT(file_hash) DO UPDATE SET
             path=excluded.path, name=excluded.name, extension=excluded.extension,
             category=excluded.category, collection=excluded.collection,
             size_bytes=excluded.size_bytes, modified_at=excluded.modified_at,
             embedded=excluded.embedded""",
        (file_hash, path, name, extension, category, collection,
         size_bytes, modified_at, int(embedded)),
    )
    conn.commit()
    conn.close()


def get_categories() -> list[dict]:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.execute(
        "SELECT category, COUNT(*) as n FROM files GROUP BY category ORDER BY n DESC"
    )
    rows = cur.fetchall()
    conn.close()
    return [{"name": name, "count": n} for name, n in rows]


def get_collections() -> list[dict]:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.execute(
        "SELECT collection, COUNT(*) as n FROM files GROUP BY collection ORDER BY n DESC"
    )
    rows = cur.fetchall()
    conn.close()
    return [{"name": name, "count": n} for name, n in rows]


def get_recent_files(limit: int = 20, category: str | None = None, collection: str | None = None) -> list[dict]:
    """
    category/collection are optional filters powering the "Files & Data"
    category tiles and AI Collections tiles on the frontend - clicking a
    tile calls this same endpoint with the tile's name so it shows exactly
    the files in that bucket instead of just the newest files overall.
    """
    conn = sqlite3.connect(DB_PATH)
    where = []
    params: list = []
    if category:
        where.append("category = ?")
        params.append(category)
    if collection:
        where.append("collection = ?")
        params.append(collection)
    where_clause = f"WHERE {' AND '.join(where)}" if where else ""
    params.append(limit)
    cur = conn.execute(
        f"""SELECT path, name, extension, category, size_bytes, modified_at
           FROM files {where_clause} ORDER BY modified_at DESC LIMIT ?""",
        params,
    )
    rows = cur.fetchall()
    conn.close()
    return [
        {
            "path": path, "name": name, "extension": ext, "category": category,
            "size_bytes": size_bytes, "modified_at": modified_at,
        }
        for path, name, ext, category, size_bytes, modified_at in rows
    ]


def get_storage_breakdown() -> list[dict]:
    """Buckets every file's size into the four Storage Overview groups
    used on the Files & Data screen (Documents / Images / Videos / Others)."""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.execute(
        "SELECT category, COALESCE(SUM(size_bytes),0) FROM files GROUP BY category"
    )
    rows = cur.fetchall()
    conn.close()

    buckets = {"Documents": 0, "Images": 0, "Videos": 0, "Others": 0}
    doc_categories = {"Documents", "PDFs", "Word", "Excel", "Presentations", "Code", "Databases"}
    for category, total_bytes in rows:
        if category == "Images":
            buckets["Images"] += total_bytes
        elif category == "Videos":
            buckets["Videos"] += total_bytes
        elif category in doc_categories:
            buckets["Documents"] += total_bytes
        else:
            buckets["Others"] += total_bytes
    return [{"name": k, "bytes": v} for k, v in buckets.items()]


# ---------------------------------------------------------------------------
# Activity timeline
# ---------------------------------------------------------------------------

def add_event(event_type: str, title: str, subtitle: str, app_label: str,
              badge_type: str, path_or_url: str, timestamp: str):
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """INSERT INTO events (event_type, title, subtitle, app_label, badge_type,
                                path_or_url, timestamp)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (event_type, title, subtitle, app_label, badge_type, path_or_url, timestamp),
    )
    conn.commit()
    conn.close()


def get_timeline(date_prefix: str | None = None, limit: int = 200) -> list[dict]:
    """date_prefix like '2024-05-14' filters to that day; None returns most recent."""
    conn = sqlite3.connect(DB_PATH)
    if date_prefix:
        cur = conn.execute(
            """SELECT event_type, title, subtitle, app_label, badge_type, path_or_url, timestamp
               FROM events WHERE timestamp LIKE ? ORDER BY timestamp ASC LIMIT ?""",
            (f"{date_prefix}%", limit),
        )
    else:
        cur = conn.execute(
            """SELECT event_type, title, subtitle, app_label, badge_type, path_or_url, timestamp
               FROM events ORDER BY timestamp DESC LIMIT ?""",
            (limit,),
        )
    rows = cur.fetchall()
    conn.close()
    return [
        {
            "event_type": et, "title": title, "subtitle": subtitle,
            "app_label": app_label, "badge_type": badge_type,
            "path_or_url": path_or_url, "timestamp": ts,
        }
        for et, title, subtitle, app_label, badge_type, path_or_url, ts in rows
    ]
