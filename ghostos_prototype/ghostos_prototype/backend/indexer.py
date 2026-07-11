"""
indexer.py
Walks a folder, extracts text from supported file types, chunks it,
embeds it, and stores it in the local vector store.

Hardcoded safety blacklist: paths/filenames matching these patterns
are always skipped, regardless of what folder you point the indexer
at. This is intentional and should not be silently removed - it's
the one thing standing between "personal memory tool" and "accidentally
indexed my saved passwords."
"""

import hashlib
from datetime import datetime
from pathlib import Path

from pypdf import PdfReader
import docx

from embeddings import get_embedding
from vectorstore import (
    init_db, add_chunk, file_already_indexed, get_stats,
    upsert_file, file_already_catalogued, add_event,
)

# Extensions GhostOS can extract text from and embed for RAG search.
SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf", ".docx", ".py", ".js", ".json", ".csv"}

# Every other recognized extension GhostOS still catalogues (for the
# Files & Data screen: categories, storage, recent files) even though it
# can't embed/search the content yet.
CATEGORY_BY_EXTENSION = {
    ".pdf": "PDFs",
    ".docx": "Word", ".doc": "Word",
    ".xlsx": "Excel", ".xls": "Excel", ".csv": "Excel",
    ".pptx": "Presentations", ".ppt": "Presentations",
    ".txt": "Documents", ".md": "Documents",
    ".png": "Images", ".jpg": "Images", ".jpeg": "Images", ".gif": "Images", ".webp": "Images",
    ".mp4": "Videos", ".mov": "Videos", ".avi": "Videos", ".mkv": "Videos",
    ".mp3": "Audio", ".wav": "Audio", ".m4a": "Audio",
    ".py": "Code", ".js": "Code", ".ts": "Code", ".json": "Code", ".html": "Code", ".css": "Code",
    ".zip": "Archives", ".rar": "Archives", ".7z": "Archives",
    ".db": "Databases", ".sqlite": "Databases",
}

# Simple path-keyword heuristic for "AI Collections" - a real system would
# use the LLM/embeddings to cluster files by meaning; this is a placeholder
# rule-based classifier that's easy to replace later without touching the
# rest of the app.
COLLECTION_KEYWORDS = {
    "Work": ["work", "project", "client", "report", "invoice", "meeting"],
    "College": ["college", "university", "course", "assignment", "lecture", "thesis"],
    "Finance": ["finance", "budget", "invoice", "tax", "bank", "expense"],
    "Personal": ["personal", "photo", "family", "vacation", "diary"],
}


def categorize_extension(ext: str) -> str:
    return CATEGORY_BY_EXTENSION.get(ext, "Others")


def classify_collection(path: Path) -> str:
    path_str = str(path).lower()
    for collection, keywords in COLLECTION_KEYWORDS.items():
        if any(kw in path_str for kw in keywords):
            return collection
    return "Projects"


def badge_type_for_extension(ext: str) -> str:
    """Maps a file extension to the small set of icon badges the frontend knows about."""
    category = categorize_extension(ext)
    mapping = {
        "PDFs": "pdf", "Word": "docx", "Excel": "xlsx", "Presentations": "pptx",
        "Images": "image", "Videos": "video", "Audio": "audio", "Code": "code",
        "Archives": "archive", "Databases": "db", "Documents": "txt",
    }
    return mapping.get(category, "txt")

# Hardcoded exclusions - always skipped, no toggle needed.
SENSITIVE_PATTERNS = [
    "login data",
    "1password",
    "keepass",
    "wallet.dat",
    "cookies",
    ".ssh",
    "private key",
    "id_rsa",
    ".env",
    "credentials",
]

CHUNK_SIZE_WORDS = 400
CHUNK_OVERLAP_WORDS = 50


def is_sensitive_path(path: Path) -> bool:
    path_str = str(path).lower()
    return any(pattern in path_str for pattern in SENSITIVE_PATTERNS)


def hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8192), b""):
            h.update(block)
    return h.hexdigest()


def extract_text(path: Path) -> str:
    ext = path.suffix.lower()
    try:
        if ext == ".pdf":
            reader = PdfReader(str(path))
            return "\n".join(page.extract_text() or "" for page in reader.pages)
        elif ext == ".docx":
            d = docx.Document(str(path))
            return "\n".join(p.text for p in d.paragraphs)
        else:  # txt, md, py, js, json, csv - plain text read
            return path.read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        print(f"[skip] Could not read {path}: {e}")
        return ""


def chunk_text(text: str) -> list[str]:
    words = text.split()
    if not words:
        return []
    chunks = []
    start = 0
    while start < len(words):
        end = start + CHUNK_SIZE_WORDS
        chunk = " ".join(words[start:end])
        if chunk.strip():
            chunks.append(chunk)
        start = end - CHUNK_OVERLAP_WORDS
    return chunks


class FileProcessResult:
    """Simple result marker for process_file, used by both bulk indexing and the watcher."""
    SENSITIVE = "sensitive"
    UNSUPPORTED = "unsupported"
    ALREADY_INDEXED = "already_indexed"
    EMPTY = "empty"
    PROCESSED = "processed"
    NOT_FOUND = "not_found"


def process_file(path: Path) -> tuple[str, int]:
    """
    Processes a single file end-to-end: safety check, dedup check,
    cataloguing (always), extraction + embedding (only for supported
    text-extractable types), and timeline logging.

    Returns (status, chunks_added). Status is one of FileProcessResult's
    constants. This is the single entry point both index_folder() and
    the background watcher use, so the safety blacklist and dedup logic
    can never be accidentally bypassed by one of the two code paths.
    """
    path = Path(path)
    ext = path.suffix.lower()

    if not path.exists() or not path.is_file():
        return FileProcessResult.NOT_FOUND, 0

    if is_sensitive_path(path):
        return FileProcessResult.SENSITIVE, 0

    try:
        file_hash = hash_file(path)
    except Exception as e:
        print(f"[skip] Could not hash {path}: {e}")
        return FileProcessResult.NOT_FOUND, 0

    if file_already_catalogued(file_hash):
        return FileProcessResult.ALREADY_INDEXED, 0

    category = categorize_extension(ext)
    collection = classify_collection(path)

    try:
        stat = path.stat()
        size_bytes = stat.st_size
        modified_at = datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds")
    except Exception as e:
        print(f"[skip] Could not stat {path}: {e}")
        return FileProcessResult.NOT_FOUND, 0

    chunks_added = 0
    embedded = False

    if ext in SUPPORTED_EXTENSIONS and not file_already_indexed(file_hash):
        text = extract_text(path)
        if text.strip():
            for chunk in chunk_text(text):
                try:
                    embedding = get_embedding(chunk)
                    add_chunk(str(path), ext, chunk, embedding, file_hash)
                    chunks_added += 1
                except Exception as e:
                    print(f"[error] Embedding failed for chunk in {path}: {e}")
            embedded = chunks_added > 0

    upsert_file(
        file_hash=file_hash, path=str(path), name=path.name, extension=ext,
        category=category, collection=collection, size_bytes=size_bytes,
        modified_at=modified_at, embedded=embedded,
    )
    add_event(
        event_type="file_indexed",
        title=f"Indexed {path.name}",
        subtitle=str(path.parent),
        app_label="GhostOS Indexer",
        badge_type=badge_type_for_extension(ext),
        path_or_url=str(path),
        timestamp=modified_at,
    )

    return FileProcessResult.PROCESSED, chunks_added


def index_folder(folder_path: str, progress_callback=None) -> dict:
    """
    Walks folder_path recursively and processes every file through
    process_file(). Returns a summary dict. Used for the initial/manual
    bulk index; the watcher handles ongoing changes after this.
    """
    init_db()
    root = Path(folder_path)
    if not root.exists():
        return {"error": f"Folder not found: {folder_path}"}

    counts = {
        FileProcessResult.PROCESSED: 0,
        FileProcessResult.SENSITIVE: 0,
        FileProcessResult.ALREADY_INDEXED: 0,
        FileProcessResult.NOT_FOUND: 0,
    }
    chunks_added_total = 0

    for path in root.rglob("*"):
        if not path.is_file():
            continue
        status, chunks_added = process_file(path)
        counts[status] = counts.get(status, 0) + 1
        chunks_added_total += chunks_added
        if status == FileProcessResult.PROCESSED and progress_callback:
            progress_callback(str(path))

    return {
        "files_processed": counts[FileProcessResult.PROCESSED],
        "files_skipped_sensitive": counts[FileProcessResult.SENSITIVE],
        "files_skipped_already_catalogued": counts[FileProcessResult.ALREADY_INDEXED],
        "chunks_added": chunks_added_total,
        "stats": get_stats(),
    }


def _walk_files(root: Path, is_excluded_dir=None):
    """
    Yields every file under root, using os.walk (not Path.rglob) so that
    excluded directories (node_modules, .git, venv, ...) can be pruned
    from the walk itself instead of merely filtered out after the fact -
    rglob has no way to stop descending into a subtree, so on a real
    Documents/Downloads folder that contains a node_modules or .venv,
    rglob would still walk every file inside it before discarding them.
    Pruning during the walk is what actually protects scan speed.
    """
    import os

    for dirpath, dirnames, filenames in os.walk(root):
        if is_excluded_dir is not None:
            dirnames[:] = [d for d in dirnames if not is_excluded_dir(Path(dirpath) / d)]
        for filename in filenames:
            yield Path(dirpath) / filename


def index_folders(folder_paths: list[str], is_excluded_dir=None,
                   on_folder_start=None, on_folder_done=None) -> dict:
    """
    Multi-folder counterpart to index_folder(), built for
    core/connect_system.py's "Connect to System" flow, which indexes
    several standard folders (Desktop, Documents, Downloads, ...) in one
    pass and wants per-folder progress callbacks plus a noise/system
    directory filter layered on top of process_file()'s existing
    sensitive-path blacklist.

    Args:
        folder_paths: list of folder path strings to index, one after another.
        is_excluded_dir: optional callable(Path) -> bool. Any directory it
            returns True for is pruned from the walk entirely (not just
            skipped) - this is the noise-folder filter (node_modules, .git,
            etc.), a separate concern from indexer.py's own
            SENSITIVE_PATTERNS blacklist, which still applies underneath
            via process_file() regardless of this parameter.
        on_folder_start: optional callable(Path) called before each folder
            begins scanning.
        on_folder_done: optional callable(Path, dict) called after each
            folder finishes, with that folder's own summary dict (same
            shape as index_folder()'s return value).

    Returns a combined summary dict across all folders, same keys as
    index_folder()'s return value.
    """
    init_db()

    counts = {
        FileProcessResult.PROCESSED: 0,
        FileProcessResult.SENSITIVE: 0,
        FileProcessResult.ALREADY_INDEXED: 0,
        FileProcessResult.NOT_FOUND: 0,
    }
    chunks_added_total = 0

    for folder_path in folder_paths:
        root = Path(folder_path)
        folder_counts = {
            FileProcessResult.PROCESSED: 0,
            FileProcessResult.SENSITIVE: 0,
            FileProcessResult.ALREADY_INDEXED: 0,
            FileProcessResult.NOT_FOUND: 0,
        }
        folder_chunks_added = 0

        if on_folder_start:
            on_folder_start(root)

        if not root.exists():
            if on_folder_done:
                on_folder_done(root, {
                    "files_processed": 0, "files_skipped_sensitive": 0,
                    "files_skipped_already_catalogued": 0, "chunks_added": 0,
                })
            continue

        for path in _walk_files(root, is_excluded_dir=is_excluded_dir):
            if not path.is_file():
                continue
            status, chunks_added = process_file(path)
            folder_counts[status] = folder_counts.get(status, 0) + 1
            folder_chunks_added += chunks_added

        for key, value in folder_counts.items():
            counts[key] = counts.get(key, 0) + value
        chunks_added_total += folder_chunks_added

        if on_folder_done:
            on_folder_done(root, {
                "files_processed": folder_counts[FileProcessResult.PROCESSED],
                "files_skipped_sensitive": folder_counts[FileProcessResult.SENSITIVE],
                "files_skipped_already_catalogued": folder_counts[FileProcessResult.ALREADY_INDEXED],
                "chunks_added": folder_chunks_added,
            })

    return {
        "files_processed": counts[FileProcessResult.PROCESSED],
        "files_skipped_sensitive": counts[FileProcessResult.SENSITIVE],
        "files_skipped_already_catalogued": counts[FileProcessResult.ALREADY_INDEXED],
        "chunks_added": chunks_added_total,
        "stats": get_stats(),
    }