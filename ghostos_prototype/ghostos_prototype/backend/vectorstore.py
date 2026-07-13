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

TIMELINE_EVENT_KINDS = frozenset({"all", "apps", "documents", "web", "system"})


def normalize_timeline_event_kind(
    event_kind: str | None = None, *, kind: str | None = None,
) -> str:
    """Validate and normalize either public name for a Timeline category.

    Keeping this at the storage boundary means callers other than Flask cannot
    accidentally turn an arbitrary string into a SQL fragment.  When both
    aliases are supplied, they must describe the same category.
    """
    normalized: list[str] = []
    for value in (event_kind, kind):
        if value is None:
            continue
        candidate = str(value).strip().casefold()
        if candidate not in TIMELINE_EVENT_KINDS:
            allowed = ", ".join(sorted(TIMELINE_EVENT_KINDS))
            raise ValueError(f"Unknown Timeline kind {value!r}; expected one of: {allowed}")
        normalized.append(candidate)
    if len(set(normalized)) > 1:
        raise ValueError("Timeline query parameters 'kind' and 'event_kind' conflict")
    return normalized[0] if normalized else "all"


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
    # Additive migrations keep existing user databases usable. Nanosecond
    # mtimes make unchanged-file checks cheap and reliable, while status/error
    # fields let the indexer report partial failures without a separate log DB.
    file_columns = {row[1] for row in conn.execute("PRAGMA table_info(files)")}
    for column, definition in {
        "mtime_ns": "INTEGER",
        "index_status": "TEXT DEFAULT 'catalogued'",
        "index_error": "TEXT",
        "chunks_count": "INTEGER DEFAULT 0",
        # Null on legacy/catalog-only rows. For embedded rows this records
        # which local model produced the vectors, allowing unchanged files
        # to be refreshed after a configured embedding-model change.
        "embedding_model": "TEXT",
    }.items():
        if column not in file_columns:
            conn.execute(f"ALTER TABLE files ADD COLUMN {column} {definition}")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_files_path ON files(path)")

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
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_events_type_timestamp
        ON events(event_type, timestamp)
    """)
    event_columns = {row[1] for row in conn.execute("PRAGMA table_info(events)")}
    if "event_key" not in event_columns:
        conn.execute("ALTER TABLE events ADD COLUMN event_key TEXT")
    conn.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_events_event_key ON events(event_key) WHERE event_key IS NOT NULL"
    )
    conn.commit()
    conn.close()


def remove_source(path: str):
    """Remove stale searchable state while preserving historical Timeline memory."""
    conn = _connect()
    conn.execute("DELETE FROM chunks WHERE source_path = ?", (path,))
    conn.execute("DELETE FROM files WHERE path = ?", (path,))
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


def upsert_browser_record(
    *, search_hash: str, event_key: str, source_path: str, source_type: str,
    content: str, embedding: list[float] | None, event: dict | None,
) -> dict[str, bool]:
    """Atomically store one page-level search chunk and one concrete visit."""
    conn = _connect()
    try:
        conn.execute("BEGIN IMMEDIATE")
        chunk_exists = conn.execute(
            "SELECT 1 FROM chunks WHERE file_hash = ? LIMIT 1", (search_hash,)
        ).fetchone() is not None
        chunk_added = False
        if not chunk_exists and embedding:
            # Browser search hashes include the embedding-model identity.
            # Once a replacement vector is ready, discard only the older
            # vector for this exact browser page/type so model switches do
            # not leave duplicate search results behind.
            if source_type.startswith("browser_history_"):
                conn.execute(
                    """DELETE FROM chunks
                       WHERE source_path = ? AND source_type = ? AND file_hash <> ?""",
                    (source_path, source_type, search_hash),
                )
            conn.execute(
                """INSERT INTO chunks
                       (source_path, source_type, content, embedding, file_hash)
                   VALUES (?, ?, ?, ?, ?)""",
                (source_path, source_type, content, json.dumps(embedding), search_hash),
            )
            chunk_added = True

        event_added = False
        if event:
            before = conn.total_changes
            conn.execute(
                """INSERT OR IGNORE INTO events
                       (event_type, title, subtitle, app_label, badge_type,
                        path_or_url, timestamp, event_key)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    event.get("event_type"), event.get("title"), event.get("subtitle"),
                    event.get("app_label"), event.get("badge_type"),
                    event.get("path_or_url"), event.get("timestamp"), event_key,
                ),
            )
            event_added = conn.total_changes > before
        conn.commit()
        return {"chunk_added": chunk_added, "event_added": event_added}
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def get_file_index_state(path: str) -> dict | None:
    """Return the lightweight state needed for an unchanged-file fast path."""
    conn = _connect()
    row = conn.execute(
        """SELECT file_hash, size_bytes, modified_at, mtime_ns, embedded,
                  index_status, index_error, chunks_count, embedding_model
           FROM files WHERE path = ? ORDER BY indexed_at DESC LIMIT 1""",
        (path,),
    ).fetchone()
    conn.close()
    if row is None:
        return None
    return {
        "file_hash": row[0], "size_bytes": row[1], "modified_at": row[2],
        "mtime_ns": row[3], "embedded": bool(row[4]),
        "index_status": row[5], "index_error": row[6], "chunks_count": row[7] or 0,
        "embedding_model": row[8],
    }


def get_catalogued_path_for_hash(file_hash: str) -> str | None:
    """Find the first catalogued source for duplicate-content detection."""
    conn = _connect()
    row = conn.execute(
        "SELECT path FROM files WHERE file_hash = ? LIMIT 1", (file_hash,)
    ).fetchone()
    conn.close()
    return row[0] if row else None


def replace_file_index(
    *, file_hash: str, path: str, name: str, extension: str, category: str,
    collection: str, size_bytes: int, modified_at: str, mtime_ns: int,
    embedded_chunks: list[tuple[str, list[float]]], index_status: str,
    index_error: str | None = None,
    source_type: str | None = None,
    embedding_model: str | None = None,
) -> str | None:
    """Atomically replace one path's metadata and chunks.

    Returns the existing path when the same content hash is already owned by
    another file. No partial chunks are exposed if embedding or a DB write
    fails midway.
    """
    conn = _connect()
    try:
        conn.execute("BEGIN IMMEDIATE")
        duplicate = conn.execute(
            "SELECT path FROM files WHERE file_hash = ? AND path <> ? LIMIT 1",
            (file_hash, path),
        ).fetchone()
        if duplicate:
            conn.rollback()
            return duplicate[0]

        conn.execute("DELETE FROM chunks WHERE source_path = ?", (path,))
        conn.execute("DELETE FROM files WHERE path = ?", (path,))
        conn.execute(
            """INSERT INTO files (
                   file_hash, path, name, extension, category, collection,
                   size_bytes, modified_at, embedded, mtime_ns, index_status,
                   index_error, chunks_count, embedding_model
               ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                file_hash, path, name, extension, category, collection,
                size_bytes, modified_at, int(bool(embedded_chunks)), mtime_ns,
                index_status, index_error, len(embedded_chunks),
                embedding_model if embedded_chunks else None,
            ),
        )
        if embedded_chunks:
            conn.executemany(
                """INSERT INTO chunks
                       (source_path, source_type, content, embedding, file_hash)
                   VALUES (?, ?, ?, ?, ?)""",
                [
                    (path, source_type or extension, content, json.dumps(embedding), file_hash)
                    for content, embedding in embedded_chunks
                ],
            )
        conn.commit()
        return None
    except Exception:
        conn.rollback()
        raise
    finally:
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
    # ``[^\W_]`` is the Unicode-aware equivalent of letters/digits.  It
    # keeps search case-insensitive while no longer discarding non-English
    # filenames or document text.
    tokens = re.findall(r"[^\W_]+", (text or "").casefold(), flags=re.UNICODE)
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
        source = str(r.get("source_path") or "").replace("/", "\\").casefold()
        content = re.sub(r"\s+", " ", str(r.get("content") or "")).strip().casefold()
        return (source, content)

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

    Combines calibrated vector, BM25, content, filename and phrase signals.
    RRF rank is intentionally a small tie-breaker: the old implementation
    normalized the first candidate's RRF score to 1.0, which gave every
    query a result above the relevance threshold even when that result had
    zero lexical or semantic evidence.

    Sets "score" on each result (matching the field name the rest of the
    app already expects from search()), plus "rerank_score" for clarity.
    """
    if not candidates:
        return []

    query_tokens = set(_tokenize(query_text))
    if not query_tokens:
        empty_scored = []
        for candidate in candidates[:top_k]:
            entry = dict(candidate)
            entry["score"] = 0.0
            entry["rerank_score"] = 0.0
            entry["matched_terms"] = []
            empty_scored.append(entry)
        return empty_scored

    max_fused = max((c.get("fused_score", 0.0) for c in candidates), default=0.0)
    query_phrase = " ".join(_tokenize(query_text))

    rescored = []
    for c in candidates:
        content_tokens = set(_tokenize(c.get("content", "")))
        filename = Path(c.get("source_path", "") or "").name
        filename_tokens = set(_tokenize(filename))

        matched_content = query_tokens & content_tokens
        matched_filename = query_tokens & filename_tokens
        content_overlap = len(matched_content) / len(query_tokens)
        filename_overlap = len(matched_filename) / len(query_tokens)

        raw_vector = float(c.get("vector_score", c.get("score", 0.0)) or 0.0)
        # Nomic cosine scores below ~0.20 carry little evidence.  A score of
        # ~0.50 is just strong enough to pass without lexical overlap, while
        # high-confidence semantic matches scale smoothly toward 1.
        vector_relevance = min(1.0, max(0.0, (raw_vector - 0.20) / 0.65))
        raw_bm25 = max(0.0, float(c.get("bm25_score", 0.0) or 0.0))
        bm25_relevance = 1.0 - math.exp(-raw_bm25 / 3.0)
        rank_signal = (float(c.get("fused_score", 0.0)) / max_fused) if max_fused > 0 else 0.0

        filename_phrase = " ".join(_tokenize(filename))
        filename_fuzzy = SequenceMatcher(None, query_phrase, filename_phrase).ratio() if filename_phrase else 0.0
        normalized_content = " ".join(_tokenize(c.get("content", "")))
        phrase_bonus = 0.0
        if query_phrase and query_phrase in filename_phrase:
            phrase_bonus += 0.06
        if query_phrase and query_phrase in normalized_content:
            phrase_bonus += 0.04

        final_score = (
            0.44 * vector_relevance
            + 0.24 * content_overlap
            + 0.16 * filename_overlap
            + 0.10 * bm25_relevance
            + 0.02 * filename_fuzzy
            + 0.04 * rank_signal
            + phrase_bonus
        )
        final_score = min(1.0, max(0.0, final_score))
        entry = dict(c)
        entry["score"] = final_score
        entry["rerank_score"] = final_score
        entry["matched_terms"] = sorted(matched_content | matched_filename)
        rescored.append(entry)

    rescored.sort(key=lambda x: x["score"], reverse=True)
    return rescored[:top_k]


def get_stats() -> dict:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.execute("SELECT COUNT(*) FROM chunks")
    total_chunks = cur.fetchone()[0]
    cur = conn.execute(
        """SELECT COUNT(*), COALESCE(SUM(size_bytes), 0),
                  COALESCE(SUM(CASE WHEN embedded = 1 THEN 1 ELSE 0 END), 0),
                  COALESCE(SUM(CASE WHEN index_status = 'failed' THEN 1 ELSE 0 END), 0)
           FROM files"""
    )
    total_files, total_bytes, embedded_files, failed_files = cur.fetchone()
    conn.close()
    return {
        "total_chunks": total_chunks or 0,
        "total_files": total_files or 0,
        "total_bytes": total_bytes or 0,
        "embedded_files": embedded_files or 0,
        "failed_files": failed_files or 0,
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
                 collection: str, size_bytes: int, modified_at: str, embedded: bool,
                 mtime_ns: int | None = None, index_status: str = "catalogued",
                 index_error: str | None = None, chunks_count: int | None = None,
                 embedding_model: str | None = None):
    """Create/update metadata while preserving the original public call shape."""
    conn = _connect()
    conn.execute(
        """INSERT INTO files (file_hash, path, name, extension, category, collection,
                               size_bytes, modified_at, embedded, mtime_ns,
                               index_status, index_error, chunks_count,
                               embedding_model)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
           ON CONFLICT(file_hash) DO UPDATE SET
             path=excluded.path, name=excluded.name, extension=excluded.extension,
             category=excluded.category, collection=excluded.collection,
             size_bytes=excluded.size_bytes, modified_at=excluded.modified_at,
             embedded=excluded.embedded, mtime_ns=excluded.mtime_ns,
             index_status=excluded.index_status, index_error=excluded.index_error,
             chunks_count=COALESCE(excluded.chunks_count, files.chunks_count),
             embedding_model=COALESCE(excluded.embedding_model, files.embedding_model)""",
        (file_hash, path, name, extension, category, collection,
         size_bytes, modified_at, int(embedded), mtime_ns, index_status,
         index_error, chunks_count, embedding_model),
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


def refresh_file_collections(classifier) -> dict[str, int]:
    """Recompute derived AI-collection labels for every catalogued file.

    Collection names are presentation metadata derived from a path, not user
    data.  Keeping the database operation here avoids importing the indexer
    (and therefore avoids a circular dependency), while accepting the small
    pure classifier callback lets a newer GhostOS release safely repair rows
    written by an older classifier.
    """
    conn = _connect()
    try:
        rows = conn.execute("SELECT path, collection FROM files").fetchall()
        updates: list[tuple[str, str]] = []
        for path, current_collection in rows:
            collection = str(classifier(path) or "Other").strip() or "Other"
            if collection != current_collection:
                updates.append((collection, path))
        if updates:
            conn.executemany(
                "UPDATE files SET collection = ? WHERE path = ?",
                updates,
            )
            conn.commit()
        return {"scanned": len(rows), "updated": len(updates)}
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def get_recent_files(
    limit: int = 20,
    category: str | None = None,
    collection: str | None = None,
    offset: int = 0,
) -> list[dict]:
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
    params.extend((limit, offset))
    cur = conn.execute(
        f"""SELECT path, name, extension, category, size_bytes, modified_at,
                  embedded, index_status, index_error, chunks_count
           FROM files {where_clause} ORDER BY modified_at DESC LIMIT ? OFFSET ?""",
        params,
    )
    rows = cur.fetchall()
    conn.close()
    return [
        {
            "path": path, "name": name, "extension": ext, "category": category,
            "size_bytes": size_bytes, "modified_at": modified_at,
            "embedded": bool(embedded), "index_status": index_status,
            "index_error": index_error, "chunks_count": chunks_count or 0,
        }
        for (
            path, name, ext, category, size_bytes, modified_at,
            embedded, index_status, index_error, chunks_count,
        ) in rows
    ]


def clear_file_index() -> dict:
    """Delete rebuildable file metadata and chunks, preserving activity history."""
    conn = _connect()
    try:
        conn.execute("BEGIN IMMEDIATE")
        chunks = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        files = conn.execute("SELECT COUNT(*) FROM files").fetchone()[0]
        conn.execute("DELETE FROM chunks")
        conn.execute("DELETE FROM files")
        conn.commit()
        return {"files": files, "chunks": chunks, "events": 0}
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def clear_all_local_data() -> dict:
    """Delete all user-derived index and timeline rows from the local database."""
    conn = _connect()
    try:
        conn.execute("BEGIN IMMEDIATE")
        counts = {
            "chunks": conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0],
            "files": conn.execute("SELECT COUNT(*) FROM files").fetchone()[0],
            "events": conn.execute("SELECT COUNT(*) FROM events").fetchone()[0],
        }
        conn.execute("DELETE FROM chunks")
        conn.execute("DELETE FROM files")
        conn.execute("DELETE FROM events")
        conn.commit()
        return counts
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


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
              badge_type: str, path_or_url: str, timestamp: str,
              event_key: str | None = None):
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """INSERT OR IGNORE INTO events
               (event_type, title, subtitle, app_label, badge_type,
                path_or_url, timestamp, event_key)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (event_type, title, subtitle, app_label, badge_type, path_or_url, timestamp, event_key),
    )
    conn.commit()
    conn.close()


def get_timeline(
    date_prefix: str | None = None,
    limit: int = 200,
    event_kind: str | None = None,
    *,
    kind: str | None = None,
) -> list[dict]:
    """Return Timeline events, optionally restricted to one honest UI category.

    ``event_kind`` and ``kind`` are equivalent aliases. Categories are derived
    from the event source rather than titles or badges: ``app_focus`` is Apps,
    ``file_*`` is Documents, ``browser_*`` is Web, and everything else is
    System. A dated query remains oldest-first; the undated feed is newest-first.
    """
    selected_kind = normalize_timeline_event_kind(event_kind, kind=kind)
    bounded_limit = min(max(int(limit), 1), 5000)

    conditions: list[str] = []
    params: list[object] = []
    if date_prefix:
        conditions.append("timestamp LIKE ?")
        params.append(f"{date_prefix}%")

    if selected_kind == "apps":
        conditions.append("event_type = ?")
        params.append("app_focus")
    elif selected_kind == "documents":
        conditions.append("event_type GLOB ?")
        params.append("file_*")
    elif selected_kind == "web":
        conditions.append("event_type GLOB ?")
        params.append("browser_*")
    elif selected_kind == "system":
        conditions.append(
            "NOT (COALESCE(event_type, '') = ? "
            "OR COALESCE(event_type, '') GLOB ? "
            "OR COALESCE(event_type, '') GLOB ?)"
        )
        params.extend(("app_focus", "file_*", "browser_*"))

    where_clause = f" WHERE {' AND '.join(conditions)}" if conditions else ""
    direction = "ASC" if date_prefix else "DESC"
    params.append(bounded_limit)
    conn = _connect()
    cur = conn.execute(
        f"""SELECT event_type, title, subtitle, app_label, badge_type, path_or_url, timestamp
            FROM events{where_clause} ORDER BY timestamp {direction} LIMIT ?""",
        params,
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
