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
import re
import threading
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable, Iterator

from pypdf import PdfReader
import docx

from embeddings import get_embedding
from ocr_service import IMAGE_EXTENSIONS, extract_ocr_text, get_ocr_status
from settings_store import get_settings
from config import (
    EMBED_MODEL,
    INDEX_EMBEDDING_WORKERS,
    INDEX_FAILED_FILES_LIMIT,
    INDEX_FILE_WORKERS,
    INDEX_MAX_CHUNKS_PER_FILE,
    INDEX_MAX_FILE_BYTES,
    INDEX_MAX_PENDING_FILES,
)
from vectorstore import (
    add_event,
    get_catalogued_path_for_hash,
    get_file_index_state,
    get_stats,
    init_db,
    replace_file_index,
    refresh_file_collections,
    upsert_file,
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
    ".bmp": "Images", ".tif": "Images", ".tiff": "Images",
    ".mp4": "Videos", ".mov": "Videos", ".avi": "Videos", ".mkv": "Videos",
    ".mp3": "Audio", ".wav": "Audio", ".m4a": "Audio",
    ".py": "Code", ".js": "Code", ".ts": "Code", ".json": "Code", ".html": "Code", ".css": "Code",
    ".zip": "Archives", ".rar": "Archives", ".7z": "Archives",
    ".db": "Databases", ".sqlite": "Databases",
}

# Simple path-keyword heuristic for "AI Collections".  Topic-specific
# collections are checked before the broader Projects/Work buckets so, for
# example, ``College/semester-project`` remains College rather than becoming a
# generic project.  Anything without a real signal belongs in Other; treating
# every unmatched file as a Project made that collection effectively useless.
COLLECTION_KEYWORDS = {
    "Finance": {
        "finance", "financial", "budget", "invoice", "tax", "bank",
        "expense", "expenses", "payslip", "salary", "investment", "receipt",
    },
    "College": {
        "college", "university", "course", "assignment", "lecture", "thesis",
        "semester", "campus", "study", "classwork",
    },
    "Personal": {
        "personal", "family", "vacation", "holiday", "photo", "photos",
        "pictures", "diary", "journal", "home",
    },
    "Projects": {
        "project", "projects", "prototype", "repo", "repository", "workspace",
        "github", "gitlab",
    },
    "Work": {
        "work", "office", "client", "clients", "meeting", "meetings", "report",
        "proposal", "business", "company", "job",
    },
}


def categorize_extension(ext: str) -> str:
    return CATEGORY_BY_EXTENSION.get(ext, "Others")


def classify_collection(path: Path | str) -> str:
    # Token matching is deliberately case-insensitive and separator agnostic.
    # It avoids substring mistakes such as classifying ``homework.py`` merely
    # because it contains "work".
    path_tokens = {
        token for token in re.split(r"[^\w]+", str(path).casefold())
        if token
    }
    for collection, keywords in COLLECTION_KEYWORDS.items():
        if path_tokens.intersection(keywords):
            return collection
    return "Other"


def refresh_catalog_collections() -> dict[str, int]:
    """Apply the current classifier to legacy and already-indexed file rows."""
    return refresh_file_collections(classify_collection)


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
_DEFAULT_EMBEDDING_SEMAPHORE = threading.BoundedSemaphore(INDEX_EMBEDDING_WORKERS)


def is_sensitive_path(path: Path) -> bool:
    path_str = str(path).lower()
    return any(pattern in path_str for pattern in SENSITIVE_PATTERNS)


def hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8192), b""):
            h.update(block)
    return h.hexdigest()


def _extract_text_checked(path: Path) -> str:
    ext = path.suffix.lower()
    if ext == ".pdf":
        reader = PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    if ext == ".docx":
        document = docx.Document(str(path))
        return "\n".join(paragraph.text for paragraph in document.paragraphs)
    return path.read_text(encoding="utf-8", errors="ignore")


def _extract_text_with_optional_ocr(path: Path) -> tuple[str, str]:
    """Extract normal text first, then use local OCR only when configured."""
    path = Path(path)
    ext = path.suffix.lower()
    settings = get_settings()
    normal_text = "" if ext in IMAGE_EXTENSIONS else _extract_text_checked(path)
    source_type = ext or "text"
    if settings.get("ocr_enabled") and (
        ext in IMAGE_EXTENSIONS or (ext == ".pdf" and not normal_text.strip())
    ):
        ocr_text = extract_ocr_text(path)
        if ocr_text.strip():
            return ocr_text, "ocr"
    return normal_text, source_type


def extract_text(path: Path) -> str:
    """Backward-compatible extraction helper; bulk indexing records errors separately."""
    try:
        return _extract_text_with_optional_ocr(Path(path))[0]
    except Exception as e:
        print(f"[skip] Could not read {path}: {e}")
        return ""


def chunk_text(text: str, max_chunks: int = INDEX_MAX_CHUNKS_PER_FILE) -> list[str]:
    """Split text into overlapping chunks with a hard per-file upper bound."""
    words = text.split()
    if not words:
        return []
    chunks = []
    start = 0
    while start < len(words) and len(chunks) < max(1, max_chunks):
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
    TOO_LARGE = "too_large"
    DUPLICATE_CONTENT = "duplicate_content"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(frozen=True)
class FileProcessOutcome:
    """Detailed internal result while `process_file` keeps its legacy tuple API."""

    path: str
    status: str
    chunks_added: int = 0
    size_bytes: int = 0
    stage: str | None = None
    error: str | None = None
    duplicate_of: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def _metadata_fingerprint(path: Path, size_bytes: int, mtime_ns: int) -> str:
    """Cheap identity for catalog-only files that should never be fully read."""
    payload = f"metadata\0{path.resolve()}\0{size_bytes}\0{mtime_ns}"
    return hashlib.sha256(payload.encode("utf-8", errors="surrogatepass")).hexdigest()


def _cancelled(cancel_event: threading.Event | None) -> bool:
    return bool(cancel_event and cancel_event.is_set())


def _acquire_embedding_slot(
    semaphore: threading.BoundedSemaphore,
    cancel_event: threading.Event | None,
) -> bool:
    """Wait in short intervals so cancellation is noticed between Ollama calls."""
    while not semaphore.acquire(timeout=0.1):
        if _cancelled(cancel_event):
            return False
    return True


def _record_catalog_state(
    *, path: Path, file_hash: str, size_bytes: int, modified_at: str,
    mtime_ns: int, status: str, error: str | None = None,
) -> str | None:
    return replace_file_index(
        file_hash=file_hash,
        path=str(path),
        name=path.name,
        extension=path.suffix.lower(),
        category=categorize_extension(path.suffix.lower()),
        collection=classify_collection(path),
        size_bytes=size_bytes,
        modified_at=modified_at,
        mtime_ns=mtime_ns,
        embedded_chunks=[],
        index_status=status,
        index_error=error,
    )


def _log_index_event(path: Path, modified_at: str) -> None:
    try:
        add_event(
            event_type="file_indexed",
            title=f"Indexed {path.name}",
            subtitle=str(path.parent),
            app_label="GhostOS Indexer",
            badge_type=badge_type_for_extension(path.suffix.lower()),
            path_or_url=str(path),
            timestamp=modified_at,
        )
    except Exception as exc:
        # The searchable file is already committed; a timeline write should
        # not turn a successful index into a retry storm.
        print(f"[indexer] timeline event failed for {path}: {exc}")


def process_file_detailed(
    path: Path,
    *,
    cancel_event: threading.Event | None = None,
    embedding_semaphore: threading.BoundedSemaphore | None = None,
    max_file_bytes: int = INDEX_MAX_FILE_BYTES,
    max_chunks: int = INDEX_MAX_CHUNKS_PER_FILE,
) -> FileProcessOutcome:
    """Index one file with bounded work and an atomic final DB commit."""
    path = Path(path)
    ext = path.suffix.lower()
    path_string = str(path)
    ocr_enabled = bool(get_settings().get("ocr_enabled"))
    ocr_image = ocr_enabled and ext in IMAGE_EXTENSIONS
    extractable = ext in SUPPORTED_EXTENSIONS or ocr_image

    if _cancelled(cancel_event):
        return FileProcessOutcome(path_string, FileProcessResult.CANCELLED)

    if not path.exists() or not path.is_file():
        return FileProcessOutcome(path_string, FileProcessResult.NOT_FOUND)

    if is_sensitive_path(path):
        return FileProcessOutcome(path_string, FileProcessResult.SENSITIVE)

    try:
        stat = path.stat()
        size_bytes = stat.st_size
        mtime_ns = stat.st_mtime_ns
        modified_at = datetime.fromtimestamp(stat.st_mtime).astimezone().isoformat(timespec="seconds")
    except OSError as exc:
        return FileProcessOutcome(
            path_string, FileProcessResult.FAILED, stage="stat", error=str(exc)
        )

    # The common watcher/resume path never even re-hashes an unchanged file.
    existing_state = get_file_index_state(path_string)
    embedding_model_stale = bool(
        existing_state
        and extractable
        and existing_state.get("embedded")
        and existing_state.get("embedding_model") != EMBED_MODEL
    )
    retryable_statuses = {
        "failed", "cancelled", "skipped_too_large", "duplicate_content",
        "ocr_unavailable",
    }
    if ocr_enabled and ext == ".pdf":
        # A scanned PDF indexed while OCR was disabled is recorded as
        # ``empty``.  Once the user enables local OCR, that unchanged file
        # must be reconsidered instead of being permanently hidden by the
        # fast mtime/size skip below.
        retryable_statuses.add("empty")
    if (
        existing_state
        and extractable
        and existing_state.get("index_status") == "catalogued"
        and not existing_state.get("embedded")
    ):
        # A file catalogued by an older GhostOS version may now have a new
        # extractor; do not let metadata-only state permanently suppress it.
        retryable_statuses.add("catalogued")
    if (
        existing_state
        and existing_state.get("mtime_ns") == mtime_ns
        and existing_state.get("size_bytes") == size_bytes
        and existing_state.get("index_status") not in retryable_statuses
        and not embedding_model_stale
    ):
        return FileProcessOutcome(
            path_string, FileProcessResult.ALREADY_INDEXED,
            size_bytes=size_bytes,
        )

    if _cancelled(cancel_event):
        return FileProcessOutcome(path_string, FileProcessResult.CANCELLED, size_bytes=size_bytes)

    # Binary/unknown formats are useful metadata but should never be read in
    # full merely to calculate a hash. The same applies to oversized text.
    if not extractable:
        file_hash = _metadata_fingerprint(path, size_bytes, mtime_ns)
        _record_catalog_state(
            path=path, file_hash=file_hash, size_bytes=size_bytes,
            modified_at=modified_at, mtime_ns=mtime_ns, status="catalogued",
        )
        _log_index_event(path, modified_at)
        return FileProcessOutcome(
            path_string, FileProcessResult.UNSUPPORTED, size_bytes=size_bytes,
            stage="catalogue",
        )

    if ocr_image and not get_ocr_status()["available"]:
        file_hash = _metadata_fingerprint(path, size_bytes, mtime_ns)
        message = "OCR is enabled, but local Tesseract OCR is unavailable."
        _record_catalog_state(
            path=path, file_hash=file_hash, size_bytes=size_bytes,
            modified_at=modified_at, mtime_ns=mtime_ns,
            status="ocr_unavailable", error=message,
        )
        return FileProcessOutcome(
            path_string, FileProcessResult.UNSUPPORTED, size_bytes=size_bytes,
            stage="ocr", error=message,
        )

    if size_bytes > max_file_bytes:
        file_hash = _metadata_fingerprint(path, size_bytes, mtime_ns)
        _record_catalog_state(
            path=path, file_hash=file_hash, size_bytes=size_bytes,
            modified_at=modified_at, mtime_ns=mtime_ns, status="skipped_too_large",
            error=f"File exceeds indexing limit of {max_file_bytes} bytes",
        )
        _log_index_event(path, modified_at)
        return FileProcessOutcome(
            path_string, FileProcessResult.TOO_LARGE, size_bytes=size_bytes,
            stage="size_limit",
            error=f"File exceeds indexing limit of {max_file_bytes} bytes",
        )

    try:
        file_hash = hash_file(path)
    except OSError as exc:
        return FileProcessOutcome(
            path_string, FileProcessResult.FAILED, size_bytes=size_bytes,
            stage="hash", error=str(exc),
        )

    # A touched-but-byte-identical file only needs its metadata refreshed.
    if (
        existing_state
        and existing_state.get("file_hash") == file_hash
        and existing_state.get("index_status") not in retryable_statuses
        and not embedding_model_stale
    ):
        upsert_file(
            file_hash=file_hash, path=path_string, name=path.name, extension=ext,
            category=categorize_extension(ext), collection=classify_collection(path),
            size_bytes=size_bytes, modified_at=modified_at,
            embedded=existing_state.get("embedded", False), mtime_ns=mtime_ns,
            index_status=existing_state.get("index_status") or "catalogued",
            index_error=existing_state.get("index_error"),
            chunks_count=existing_state.get("chunks_count", 0),
            embedding_model=existing_state.get("embedding_model"),
        )
        return FileProcessOutcome(
            path_string, FileProcessResult.ALREADY_INDEXED, size_bytes=size_bytes,
        )

    duplicate_path = get_catalogued_path_for_hash(file_hash)
    if duplicate_path and duplicate_path != path_string:
        _record_catalog_state(
            path=path,
            file_hash=_metadata_fingerprint(path, size_bytes, mtime_ns),
            size_bytes=size_bytes,
            modified_at=modified_at,
            mtime_ns=mtime_ns,
            status="duplicate_content",
            error=f"Duplicate content of {duplicate_path}",
        )
        return FileProcessOutcome(
            path_string, FileProcessResult.DUPLICATE_CONTENT,
            size_bytes=size_bytes, duplicate_of=duplicate_path,
        )

    try:
        text, source_type = _extract_text_with_optional_ocr(path)
    except Exception as exc:
        error = str(exc)
        _record_catalog_state(
            path=path, file_hash=file_hash, size_bytes=size_bytes,
            modified_at=modified_at, mtime_ns=mtime_ns, status="failed", error=error,
        )
        return FileProcessOutcome(
            path_string, FileProcessResult.FAILED, size_bytes=size_bytes,
            stage="extract", error=error,
        )

    if not text.strip():
        _record_catalog_state(
            path=path, file_hash=file_hash, size_bytes=size_bytes,
            modified_at=modified_at, mtime_ns=mtime_ns, status="empty",
        )
        _log_index_event(path, modified_at)
        return FileProcessOutcome(
            path_string, FileProcessResult.EMPTY, size_bytes=size_bytes,
            stage="extract",
        )

    chunks = chunk_text(text, max_chunks=max_chunks)
    truncated = len(text.split()) > (
        CHUNK_SIZE_WORDS + max(0, len(chunks) - 1) * (CHUNK_SIZE_WORDS - CHUNK_OVERLAP_WORDS)
    )
    semaphore = embedding_semaphore or _DEFAULT_EMBEDDING_SEMAPHORE
    embedded_chunks: list[tuple[str, list[float]]] = []

    for chunk in chunks:
        if _cancelled(cancel_event):
            return FileProcessOutcome(
                path_string, FileProcessResult.CANCELLED,
                size_bytes=size_bytes, stage="embedding",
            )
        if not _acquire_embedding_slot(semaphore, cancel_event):
            return FileProcessOutcome(
                path_string, FileProcessResult.CANCELLED,
                size_bytes=size_bytes, stage="embedding",
            )
        try:
            embedding = get_embedding(chunk)
            if not isinstance(embedding, list) or not embedding:
                raise ValueError("Embedding model returned an empty vector")
            embedded_chunks.append((chunk, embedding))
        except Exception as exc:
            error = str(exc)
            _record_catalog_state(
                path=path, file_hash=file_hash, size_bytes=size_bytes,
                modified_at=modified_at, mtime_ns=mtime_ns,
                status="failed", error=error,
            )
            return FileProcessOutcome(
                path_string, FileProcessResult.FAILED, size_bytes=size_bytes,
                stage="embedding", error=error,
            )
        finally:
            semaphore.release()

    duplicate_path = replace_file_index(
        file_hash=file_hash,
        path=path_string,
        name=path.name,
        extension=ext,
        category=categorize_extension(ext),
        collection=classify_collection(path),
        size_bytes=size_bytes,
        modified_at=modified_at,
        mtime_ns=mtime_ns,
        embedded_chunks=embedded_chunks,
        index_status="embedded_truncated" if truncated else "embedded",
        source_type=source_type,
        embedding_model=EMBED_MODEL,
    )
    if duplicate_path:
        _record_catalog_state(
            path=path,
            file_hash=_metadata_fingerprint(path, size_bytes, mtime_ns),
            size_bytes=size_bytes,
            modified_at=modified_at,
            mtime_ns=mtime_ns,
            status="duplicate_content",
            error=f"Duplicate content of {duplicate_path}",
        )
        return FileProcessOutcome(
            path_string, FileProcessResult.DUPLICATE_CONTENT,
            size_bytes=size_bytes, duplicate_of=duplicate_path,
        )

    _log_index_event(path, modified_at)
    return FileProcessOutcome(
        path_string, FileProcessResult.PROCESSED,
        chunks_added=len(embedded_chunks), size_bytes=size_bytes,
    )


def process_file(path: Path, **kwargs) -> tuple[str, int]:
    """Compatibility wrapper used by the watcher and existing callers."""
    outcome = process_file_detailed(path, **kwargs)
    return outcome.status, outcome.chunks_added


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


def _empty_counts() -> dict[str, int]:
    return {
        value: 0
        for value in (
            FileProcessResult.PROCESSED,
            FileProcessResult.SENSITIVE,
            FileProcessResult.UNSUPPORTED,
            FileProcessResult.ALREADY_INDEXED,
            FileProcessResult.EMPTY,
            FileProcessResult.NOT_FOUND,
            FileProcessResult.TOO_LARGE,
            FileProcessResult.DUPLICATE_CONTENT,
            FileProcessResult.FAILED,
            FileProcessResult.CANCELLED,
        )
    }


def _build_summary(
    counts: dict[str, int],
    chunks_added: int,
    failed_files: list[dict],
    *,
    include_stats: bool = True,
) -> dict:
    # `files_processed` remains a backwards-compatible count of files that
    # were successfully catalogued in this pass, including non-embeddable,
    # empty, and intentionally size-limited files.
    files_processed = sum(
        counts.get(status, 0)
        for status in (
            FileProcessResult.PROCESSED,
            FileProcessResult.UNSUPPORTED,
            FileProcessResult.EMPTY,
            FileProcessResult.TOO_LARGE,
        )
    )
    result = {
        "files_processed": files_processed,
        "files_completed": sum(counts.values()),
        "files_embedded": counts.get(FileProcessResult.PROCESSED, 0),
        "files_catalogued_only": counts.get(FileProcessResult.UNSUPPORTED, 0),
        "files_empty": counts.get(FileProcessResult.EMPTY, 0),
        "files_too_large": counts.get(FileProcessResult.TOO_LARGE, 0),
        "files_failed": counts.get(FileProcessResult.FAILED, 0),
        "files_cancelled": counts.get(FileProcessResult.CANCELLED, 0),
        "files_skipped_sensitive": counts.get(FileProcessResult.SENSITIVE, 0),
        "files_skipped_already_catalogued": counts.get(FileProcessResult.ALREADY_INDEXED, 0),
        "files_skipped_duplicate_content": counts.get(FileProcessResult.DUPLICATE_CONTENT, 0),
        "files_missing": counts.get(FileProcessResult.NOT_FOUND, 0),
        "chunks_added": chunks_added,
        "cancelled": counts.get(FileProcessResult.CANCELLED, 0) > 0,
        "failed_files": failed_files[:INDEX_FAILED_FILES_LIMIT],
    }
    if include_stats:
        result["stats"] = get_stats()
    return result


def _process_paths_bounded(
    paths: Iterable[Path],
    *,
    cancel_event: threading.Event | None,
    file_workers: int,
    max_pending: int,
    embedding_semaphore: threading.BoundedSemaphore,
    max_file_bytes: int,
    max_chunks: int,
    on_file_start: Callable[[Path], None] | None,
) -> Iterator[FileProcessOutcome]:
    """Process a streaming iterator without ever materializing the whole tree."""
    workers = max(1, file_workers)
    pending_limit = max(workers, max_pending)
    iterator = iter(paths)

    def run(path: Path) -> FileProcessOutcome:
        if on_file_start:
            on_file_start(path)
        return process_file_detailed(
            path,
            cancel_event=cancel_event,
            embedding_semaphore=embedding_semaphore,
            max_file_bytes=max_file_bytes,
            max_chunks=max_chunks,
        )

    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="ghostos-index") as executor:
        pending: dict = {}
        exhausted = False

        def fill_queue() -> None:
            nonlocal exhausted
            while not exhausted and len(pending) < pending_limit and not _cancelled(cancel_event):
                try:
                    path = next(iterator)
                except StopIteration:
                    exhausted = True
                    break
                pending[executor.submit(run, path)] = path

        fill_queue()
        while pending:
            done, _ = wait(tuple(pending), return_when=FIRST_COMPLETED)
            for future in done:
                path = pending.pop(future)
                try:
                    yield future.result()
                except Exception as exc:
                    yield FileProcessOutcome(
                        str(path), FileProcessResult.FAILED,
                        stage="unexpected", error=str(exc),
                    )
            fill_queue()


def index_folder(
    folder_path: str,
    progress_callback=None,
    *,
    cancel_event: threading.Event | None = None,
    file_workers: int = INDEX_FILE_WORKERS,
    embedding_workers: int = INDEX_EMBEDDING_WORKERS,
) -> dict:
    """Backward-compatible single-folder entry point using the bounded pipeline."""
    root = Path(folder_path)
    if not root.exists():
        return {"error": f"Folder not found: {folder_path}"}

    def on_file_done(path: Path, outcome: dict) -> None:
        if progress_callback and outcome["status"] == FileProcessResult.PROCESSED:
            progress_callback(str(path))

    return index_folders(
        [folder_path],
        on_file_done=on_file_done,
        cancel_event=cancel_event,
        file_workers=file_workers,
        embedding_workers=embedding_workers,
    )


def index_folders(
    folder_paths: list[str],
    is_excluded_dir=None,
    on_folder_start=None,
    on_folder_done=None,
    *,
    on_file_start: Callable[[Path], None] | None = None,
    on_file_done: Callable[[Path, dict], None] | None = None,
    cancel_event: threading.Event | None = None,
    file_workers: int = INDEX_FILE_WORKERS,
    embedding_workers: int = INDEX_EMBEDDING_WORKERS,
    max_pending: int = INDEX_MAX_PENDING_FILES,
    max_file_bytes: int = INDEX_MAX_FILE_BYTES,
    max_chunks: int = INDEX_MAX_CHUNKS_PER_FILE,
) -> dict:
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

    File enumeration is lazy and the pending-future queue is bounded, so even
    a whole-drive scan does not load every path or document into memory.
    """
    init_db()
    counts = _empty_counts()
    chunks_added_total = 0
    failed_files: list[dict] = []
    embedding_semaphore = (
        _DEFAULT_EMBEDDING_SEMAPHORE
        if embedding_workers == INDEX_EMBEDDING_WORKERS
        else threading.BoundedSemaphore(max(1, embedding_workers))
    )

    for folder_path in folder_paths:
        if _cancelled(cancel_event):
            break
        root = Path(folder_path)
        folder_counts = _empty_counts()
        folder_chunks_added = 0
        folder_failures: list[dict] = []

        if on_folder_start:
            on_folder_start(root)

        if not root.exists():
            folder_counts[FileProcessResult.NOT_FOUND] += 1
            if on_folder_done:
                on_folder_done(root, _build_summary(
                    folder_counts, 0, [], include_stats=False
                ))
            counts[FileProcessResult.NOT_FOUND] += 1
            continue

        paths = _walk_files(root, is_excluded_dir=is_excluded_dir)
        for outcome in _process_paths_bounded(
            paths,
            cancel_event=cancel_event,
            file_workers=file_workers,
            max_pending=max_pending,
            embedding_semaphore=embedding_semaphore,
            max_file_bytes=max_file_bytes,
            max_chunks=max_chunks,
            on_file_start=on_file_start,
        ):
            folder_counts[outcome.status] = folder_counts.get(outcome.status, 0) + 1
            folder_chunks_added += outcome.chunks_added
            if outcome.status == FileProcessResult.FAILED:
                failure = {
                    "path": outcome.path,
                    "stage": outcome.stage or "unknown",
                    "error": outcome.error or "Unknown indexing error",
                }
                if len(folder_failures) < INDEX_FAILED_FILES_LIMIT:
                    folder_failures.append(failure)
                if len(failed_files) < INDEX_FAILED_FILES_LIMIT:
                    failed_files.append(failure)
            if on_file_done:
                on_file_done(Path(outcome.path), outcome.to_dict())

        for key, value in folder_counts.items():
            counts[key] = counts.get(key, 0) + value
        chunks_added_total += folder_chunks_added

        if on_folder_done:
            on_folder_done(root, _build_summary(
                folder_counts, folder_chunks_added, folder_failures, include_stats=False
            ))

    # Cancellation can happen before another file is submitted, leaving no
    # per-file CANCELLED outcome. Reflect the job-level signal explicitly.
    summary = _build_summary(counts, chunks_added_total, failed_files)
    if _cancelled(cancel_event):
        summary["cancelled"] = True
    return summary
