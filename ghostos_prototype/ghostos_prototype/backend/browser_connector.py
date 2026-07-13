"""Local Chrome/Edge historical-data connector for GhostOS.

The connector discovers persistent Chrome and Edge profiles, snapshots their
locked local files, and indexes three explicit record types:

* history (from the profile ``History`` SQLite database)
* bookmarks (from the profile ``Bookmarks`` JSON file)
* downloads (from the ``downloads`` tables when that Chromium schema exists)

No network requests are made and this module does not claim live-tab access.
``sync_browser_history()`` keeps its original no-argument call and result keys
so the existing watcher remains compatible; additional result fields provide
honest per-profile/per-type diagnostics.
"""

from __future__ import annotations

import json
import hashlib
import os
import shutil
import sqlite3
import tempfile
import threading
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import urlsplit

from browser_intelligence import (
    BrowserProfile,
    BrowserRecord,
    active_tab_support_status,
    chromium_time_to_iso,
    deduplicate_records,
    extract_youtube_metadata,
    normalize_url,
    summarize_domain_groups,
)
from embeddings import get_embedding
from config import EMBED_MODEL
from indexer import SENSITIVE_PATTERNS
from vectorstore import file_already_indexed, init_db, upsert_browser_record


# Extra patterns specific to browsing (on top of the file blacklist).  These
# are intentionally local string checks and never sent to an external service.
BROWSER_SENSITIVE_PATTERNS = SENSITIVE_PATTERNS + [
    "paypal.com",
    "chase.com",
    "bankofamerica",
    "wellsfargo",
    "accounts.google.com",
    "login.microsoftonline.com",
]

# Retained for existing diagnostics/tests which inspect Default-profile paths.
# Profile discovery derives each browser's User Data root from these paths and
# then scans every persistent profile underneath it.
HISTORY_PATHS = {
    "chrome": Path(os.path.expandvars(r"%LOCALAPPDATA%\Google\Chrome\User Data\Default\History")),
    "edge": Path(os.path.expandvars(r"%LOCALAPPDATA%\Microsoft\Edge\User Data\Default\History")),
}

MAX_ENTRIES_PER_RUN = 2000
BACKFILL_ENTRIES_PER_RUN = 500
MAX_BOOKMARKS_PER_PROFILE = 5000
MAX_DOWNLOADS_PER_PROFILE = 2000
BROWSER_STATE_PATH = Path(__file__).with_name("ghostos_browser_state.json")
_state_lock = threading.Lock()
_browser_write_lock = threading.Lock()

_SKIPPED_PROFILE_NAMES = {"System Profile", "Guest Profile"}
_SUPPORTED_HISTORY_SCHEMES = {"http", "https", "file"}


def _model_search_hash(record: BrowserRecord) -> str:
    """Bind a page-level search identity to the vector model that made it."""
    identity = f"{record.search_fingerprint}\0{EMBED_MODEL}"
    return hashlib.sha256(identity.encode("utf-8", errors="replace")).hexdigest()


def is_sensitive_url(url: str) -> bool:
    """Return true for sensitive patterns or credential-bearing URLs."""

    value = (url or "").casefold()
    if any(pattern.casefold() in value for pattern in BROWSER_SENSITIVE_PATTERNS):
        return True
    try:
        parsed = urlsplit(url)
        return bool(parsed.username or parsed.password)
    except ValueError:
        return True


def _is_sensitive_record(record: BrowserRecord) -> bool:
    candidates = [record.url, record.local_path, record.folder_path]
    return any(value and is_sensitive_url(value) for value in candidates)


def _is_supported_history_url(url: str | None) -> bool:
    if not url:
        return False
    try:
        return urlsplit(url).scheme.casefold() in _SUPPORTED_HISTORY_SCHEMES
    except ValueError:
        return False


def chrome_time_to_datetime(chrome_timestamp: int | str | None) -> str:
    """Backward-compatible wrapper returning the existing fallback string."""

    return chromium_time_to_iso(chrome_timestamp) or "unknown time"


def extract_youtube_id(url: str) -> str | None:
    """Backward-compatible helper now covering watch, short, embed and live URLs."""

    return extract_youtube_metadata(url).get("video_id")


def _copy_locked_db(source_path: Path) -> Path | None:
    """Snapshot a Chromium SQLite file so the running browser can keep its lock."""

    if not source_path.exists():
        return None
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".sqlite")
    os.close(tmp_fd)
    try:
        shutil.copy2(source_path, tmp_path)
        # Chromium may keep its newest committed rows in WAL mode.  Copy the
        # sidecars under the temp database's matching basename when present;
        # SQLite will replay them when the snapshot is opened.  A missing or
        # briefly locked sidecar is non-fatal—the main snapshot is still useful.
        for suffix in ("-wal", "-shm"):
            source_sidecar = Path(f"{source_path}{suffix}")
            if source_sidecar.exists():
                try:
                    shutil.copy2(source_sidecar, f"{tmp_path}{suffix}")
                except (OSError, shutil.Error):
                    pass
        return Path(tmp_path)
    except (OSError, shutil.Error) as exc:
        print(f"[browser] could not snapshot {source_path}: {exc}")
        Path(tmp_path).unlink(missing_ok=True)
        return None


def _load_profile_names(user_data_root: Path) -> dict[str, str]:
    local_state = user_data_root / "Local State"
    try:
        if not local_state.exists():
            return {}
        payload = json.loads(local_state.read_text(encoding="utf-8"))
        info_cache = payload.get("profile", {}).get("info_cache", {})
        if not isinstance(info_cache, dict):
            return {}
        return {
            str(profile_id): str(details.get("name") or profile_id)
            for profile_id, details in info_cache.items()
            if isinstance(details, dict)
        }
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        print(f"[browser] could not read profile metadata at {local_state}: {exc}")
        return {}


def discover_browser_profiles(
    history_paths: dict[str, Path] | None = None,
) -> list[BrowserProfile]:
    """Discover all persistent Chrome/Edge profiles with local browser data.

    ``history_paths`` exists for deterministic offline tests and preserves the
    established ``HISTORY_PATHS`` configuration surface.  Directories listed
    in Chromium's Local State are considered, as are on-disk profile folders
    containing a History or Bookmarks file.
    """

    configured = history_paths or HISTORY_PATHS
    discovered: list[BrowserProfile] = []

    for browser, default_history in configured.items():
        default_history = Path(default_history)
        user_data_root = default_history.parent.parent
        profile_names = _load_profile_names(user_data_root)
        candidate_ids = set(profile_names)

        try:
            if default_history.exists() or (default_history.parent / "Bookmarks").exists():
                candidate_ids.add(default_history.parent.name)
        except OSError as exc:
            print(f"[browser] could not inspect default profile in {user_data_root}: {exc}")

        try:
            for child in user_data_root.iterdir() if user_data_root.exists() else ():
                if not child.is_dir() or child.name in _SKIPPED_PROFILE_NAMES:
                    continue
                if (child / "History").exists() or (child / "Bookmarks").exists():
                    candidate_ids.add(child.name)
        except OSError as exc:
            print(f"[browser] could not enumerate profiles in {user_data_root}: {exc}")

        for profile_id in sorted(candidate_ids, key=lambda value: (value != "Default", value.casefold())):
            if profile_id in _SKIPPED_PROFILE_NAMES:
                continue
            directory = user_data_root / profile_id
            history_path = directory / "History"
            bookmarks_path = directory / "Bookmarks"
            try:
                has_history = history_path.exists()
                has_bookmarks = bookmarks_path.exists()
            except OSError as exc:
                print(f"[browser] could not inspect profile {directory}: {exc}")
                continue
            if not has_history and not has_bookmarks:
                continue
            discovered.append(BrowserProfile(
                browser=browser,
                profile_id=profile_id,
                display_name=profile_names.get(profile_id, profile_id),
                directory=str(directory),
                history_path=str(history_path),
                bookmarks_path=str(bookmarks_path),
            ))

    return discovered


def _table_names(conn: sqlite3.Connection) -> set[str]:
    return {
        row[0]
        for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    }


def _table_columns(conn: sqlite3.Connection, table_name: str) -> set[str]:
    # table_name comes only from fixed constants in this module.
    return {row[1] for row in conn.execute(f"PRAGMA table_info({table_name})").fetchall()}


def _history_records(
    conn: sqlite3.Connection,
    profile: BrowserProfile,
    limit: int,
    offset: int = 0,
) -> tuple[list[BrowserRecord], int]:
    table_names = _table_names(conn)
    if "urls" not in table_names:
        return [], 0
    columns = _table_columns(conn, "urls")
    required = {"url", "title", "last_visit_time"}
    if not required.issubset(columns):
        return [], 0

    if "visits" in table_names and {"id", "url", "visit_time"}.issubset(
        _table_columns(conn, "visits")
    ) and "id" in columns:
        hidden = " AND urls.hidden = 0" if "hidden" in columns else ""
        visit_count = "urls.visit_count" if "visit_count" in columns else "NULL"
        typed_count = "urls.typed_count" if "typed_count" in columns else "NULL"
        sql = f"""SELECT visits.id, urls.url, urls.title, visits.visit_time,
                         {visit_count}, {typed_count}
                  FROM visits JOIN urls ON visits.url = urls.id
                  WHERE 1=1{hidden}
                  ORDER BY visits.visit_time DESC LIMIT ? OFFSET ?"""
    else:
        expressions = [
            "id" if "id" in columns else "rowid AS id",
            "url", "title", "last_visit_time",
            "visit_count" if "visit_count" in columns else "NULL AS visit_count",
            "typed_count" if "typed_count" in columns else "NULL AS typed_count",
        ]
        where = " WHERE hidden = 0" if "hidden" in columns else ""
        sql = f"SELECT {', '.join(expressions)} FROM urls{where} ORDER BY last_visit_time DESC LIMIT ? OFFSET ?"
    rows = conn.execute(sql, (limit, max(0, offset))).fetchall()

    records: list[BrowserRecord] = []
    for native_id, url, title, last_visit_time, visit_count, typed_count in rows:
        if not _is_supported_history_url(url):
            continue
        safe_url = normalize_url(url)
        if not safe_url:
            continue
        records.append(BrowserRecord(
            record_type="history",
            browser=profile.browser,
            profile_id=profile.profile_id,
            profile_name=profile.display_name,
            native_id=str(native_id),
            title=(title or safe_url or "(no title)").strip(),
            url=safe_url,
            timestamp=chromium_time_to_iso(last_visit_time),
            visit_count=int(visit_count) if visit_count is not None else None,
            metadata={
                "typed_count": int(typed_count) if typed_count is not None else None,
                "historical": True,
            },
        ))
    return records, len(rows)


def _download_url_chains(conn: sqlite3.Connection) -> dict[int, list[str]]:
    if "downloads_url_chains" not in _table_names(conn):
        return {}
    columns = _table_columns(conn, "downloads_url_chains")
    if not {"id", "url"}.issubset(columns):
        return {}
    order = ", chain_index" if "chain_index" in columns else ""
    chains: dict[int, list[str]] = defaultdict(list)
    for download_id, url in conn.execute(
        f"SELECT id, url FROM downloads_url_chains ORDER BY id{order}"
    ).fetchall():
        if url:
            chains[int(download_id)].append(url)
    return dict(chains)


def _downloads_records(
    conn: sqlite3.Connection,
    profile: BrowserProfile,
    limit: int,
) -> list[BrowserRecord]:
    if "downloads" not in _table_names(conn):
        return []
    columns = _table_columns(conn, "downloads")
    if "id" not in columns:
        return []

    wanted = [
        "id", "current_path", "target_path", "start_time", "end_time",
        "received_bytes", "total_bytes", "state", "opened", "mime_type",
        "tab_url", "site_url", "referrer",
    ]
    expressions = [column if column in columns else f"NULL AS {column}" for column in wanted]
    order_by = "start_time" if "start_time" in columns else "id"
    rows = conn.execute(
        f"SELECT {', '.join(expressions)} FROM downloads ORDER BY {order_by} DESC LIMIT ?",
        (limit,),
    ).fetchall()
    chains = _download_url_chains(conn)

    records: list[BrowserRecord] = []
    for row in rows:
        data = dict(zip(wanted, row))
        download_id = int(data["id"])
        local_path = data.get("target_path") or data.get("current_path")
        chain = [safe for item in chains.get(download_id, []) if (safe := normalize_url(item))]
        url = data.get("tab_url") or data.get("site_url") or (chain[-1] if chain else None) \
            or data.get("referrer")
        url = normalize_url(url)
        if not local_path and not url:
            continue
        title = Path(str(local_path)).name if local_path else ""
        if not title:
            title = (urlsplit(str(url)).path.rsplit("/", 1)[-1] if url else "") or "Downloaded item"
        records.append(BrowserRecord(
            record_type="download",
            browser=profile.browser,
            profile_id=profile.profile_id,
            profile_name=profile.display_name,
            native_id=str(download_id),
            title=title,
            url=str(url) if url else None,
            timestamp=chromium_time_to_iso(data.get("start_time")),
            local_path=str(local_path) if local_path else None,
            bytes_received=int(data["received_bytes"]) if data.get("received_bytes") is not None else None,
            total_bytes=int(data["total_bytes"]) if data.get("total_bytes") is not None else None,
            metadata={
                "state": data.get("state"),
                "opened": bool(data.get("opened")) if data.get("opened") is not None else None,
                "mime_type": data.get("mime_type"),
                "end_time": chromium_time_to_iso(data.get("end_time")),
                "url_chain": chain,
                "historical": True,
            },
        ))
    return records


def _read_history_and_downloads(
    profile: BrowserProfile,
    history_limit: int = MAX_ENTRIES_PER_RUN,
    download_limit: int = MAX_DOWNLOADS_PER_PROFILE,
    history_offset: int = 0,
) -> list[BrowserRecord]:
    source_path = Path(profile.history_path)
    snapshot = _copy_locked_db(source_path)
    if snapshot is None:
        return []
    conn: sqlite3.Connection | None = None
    try:
        conn = sqlite3.connect(str(snapshot))
        history, _ = _history_records(conn, profile, history_limit, history_offset)
        return history + _downloads_records(conn, profile, download_limit)
    except sqlite3.Error as exc:
        print(f"[browser] could not read {profile.label} History database: {exc}")
        return []
    finally:
        if conn is not None:
            conn.close()
        snapshot.unlink(missing_ok=True)
        Path(f"{snapshot}-wal").unlink(missing_ok=True)
        Path(f"{snapshot}-shm").unlink(missing_ok=True)


def _read_history_page(
    profile: BrowserProfile, limit: int, offset: int,
) -> tuple[list[BrowserRecord], int]:
    """Read one raw-row-counted page for bounded historical backfill."""
    snapshot = _copy_locked_db(Path(profile.history_path))
    if snapshot is None:
        return [], 0
    conn: sqlite3.Connection | None = None
    try:
        conn = sqlite3.connect(str(snapshot))
        return _history_records(conn, profile, limit, offset)
    except sqlite3.Error as exc:
        print(f"[browser] could not backfill {profile.label}: {exc}")
        return [], 0
    finally:
        if conn is not None:
            conn.close()
        snapshot.unlink(missing_ok=True)
        Path(f"{snapshot}-wal").unlink(missing_ok=True)
        Path(f"{snapshot}-shm").unlink(missing_ok=True)


def _walk_bookmark_nodes(
    node: object,
    folder_parts: tuple[str, ...] = (),
) -> list[tuple[dict, tuple[str, ...]]]:
    found: list[tuple[dict, tuple[str, ...]]] = []
    if not isinstance(node, dict):
        return found
    children = node.get("children")
    node_type = node.get("type")
    if node_type == "url" or (node.get("url") and not children):
        found.append((node, folder_parts))
    if isinstance(children, list):
        folder_name = str(node.get("name") or "").strip()
        child_parts = folder_parts + ((folder_name,) if folder_name else ())
        for child in children:
            found.extend(_walk_bookmark_nodes(child, child_parts))
    return found


def _bookmark_records(
    profile: BrowserProfile,
    limit: int = MAX_BOOKMARKS_PER_PROFILE,
) -> list[BrowserRecord]:
    configured_path = Path(profile.bookmarks_path)
    bookmarks_path = next(
        (candidate for candidate in (configured_path, configured_path.with_name("Bookmarks.bak"))
         if candidate.exists()),
        None,
    )
    if bookmarks_path is None:
        return []
    try:
        payload = json.loads(bookmarks_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        print(f"[browser] could not read {profile.label} bookmarks: {exc}")
        return []

    nodes: list[tuple[dict, tuple[str, ...]]] = []
    roots = payload.get("roots", {}) if isinstance(payload, dict) else {}
    if isinstance(roots, dict):
        for root in roots.values():
            nodes.extend(_walk_bookmark_nodes(root))

    records: list[BrowserRecord] = []
    for node, folder_parts in nodes:
        url = str(node.get("url") or "")
        if not _is_supported_history_url(url):
            continue
        url = normalize_url(url)
        if not url:
            continue
        records.append(BrowserRecord(
            record_type="bookmark",
            browser=profile.browser,
            profile_id=profile.profile_id,
            profile_name=profile.display_name,
            native_id=str(node.get("guid") or node.get("id") or "") or None,
            title=str(node.get("name") or url).strip(),
            url=url,
            timestamp=chromium_time_to_iso(node.get("date_added") or node.get("date_last_used")),
            folder_path=" / ".join(folder_parts) or None,
            metadata={"historical": True},
        ))

    records.sort(key=lambda record: record.timestamp or "", reverse=True)
    return records[:limit]


def collect_profile_records(profile: BrowserProfile) -> list[BrowserRecord]:
    """Read all supported historical record types for one local profile."""

    records = _read_history_and_downloads(profile)
    records.extend(_bookmark_records(profile))
    return deduplicate_records(records)


def collect_browser_records(
    history_paths: dict[str, Path] | None = None,
) -> tuple[list[BrowserProfile], list[BrowserRecord]]:
    """Collect and cross-profile-dedupe local records without indexing them."""

    profiles = discover_browser_profiles(history_paths)
    records: list[BrowserRecord] = []
    for profile in profiles:
        records.extend(collect_profile_records(profile))
    return profiles, deduplicate_records(records)


def _profile_state_key(profile: BrowserProfile) -> str:
    return f"{profile.browser}:{profile.profile_id}"


def _load_backfill_state() -> dict[str, int]:
    with _state_lock:
        try:
            payload = json.loads(BROWSER_STATE_PATH.read_text(encoding="utf-8"))
            offsets = payload.get("history_backfill_offsets", {})
            return {
                str(key): int(value) if int(value) == -1 else max(0, int(value))
                for key, value in offsets.items()
                if isinstance(value, (int, float, str))
            }
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            return {}


def _save_backfill_state(offsets: dict[str, int]) -> None:
    with _state_lock:
        BROWSER_STATE_PATH.write_text(
            json.dumps({"history_backfill_offsets": offsets}, indent=2),
            encoding="utf-8",
        )


def reset_browser_sync_state() -> None:
    """Forget only the local backfill cursor so a cleared index can rebuild."""
    with _state_lock:
        BROWSER_STATE_PATH.unlink(missing_ok=True)


def wait_for_browser_writes() -> None:
    """Wait only for the current short atomic browser database commit."""
    with _browser_write_lock:
        pass


def sync_browser_history(cancel_event: threading.Event | None = None) -> dict:
    """Index new local browser history/bookmark/download records.

    The original contract keys (``browsers_found``, ``entries_added``,
    ``skipped_sensitive``, and ``skipped_already_indexed``) remain present.
    Stable canonical fingerprints prevent duplicates across repeated runs,
    tracking-parameter URL variants, browser profiles, and Chrome/Edge.
    """

    init_db()
    profiles = discover_browser_profiles()
    offsets = _load_backfill_state()
    next_offsets = dict(offsets)
    records: list[BrowserRecord] = []
    for profile in profiles:
        if cancel_event and cancel_event.is_set():
            break
        recent = collect_profile_records(profile)
        records.extend(recent)
        key = _profile_state_key(profile)
        offset = offsets.get(key, 0)
        if offset >= MAX_ENTRIES_PER_RUN:
            older, rows_consumed = _read_history_page(
                profile, BACKFILL_ENTRIES_PER_RUN, offset
            )
            records.extend(older)
            next_offsets[key] = (
                -1 if rows_consumed < BACKFILL_ENTRIES_PER_RUN
                else offset + rows_consumed
            )
        elif offset != -1:
            next_offsets[key] = MAX_ENTRIES_PER_RUN
    records = deduplicate_records(records)
    added_by_type: Counter[str] = Counter()
    found_by_type: Counter[str] = Counter(record.record_type for record in records)
    skipped_sensitive = 0
    skipped_existing = 0
    failed = 0
    safe_records: list[BrowserRecord] = []
    cancelled = False

    for record in records:
        if cancel_event and cancel_event.is_set():
            cancelled = True
            break
        if _is_sensitive_record(record):
            skipped_sensitive += 1
            continue
        safe_records.append(record)
        content = record.to_search_text()
        event = record.to_timeline_event()
        search_hash = _model_search_hash(record)
        chunk_exists = file_already_indexed(search_hash)
        embedding = None
        if not chunk_exists:
            try:
                embedding = get_embedding(content)
            except Exception as exc:
                # Timeline collection must remain useful while Ollama is
                # stopped.  Store the visit without a search chunk and keep
                # the backfill cursor unchanged so a later sync can retry the
                # missing embedding.
                failed += 1
                print(f"[browser] embedding deferred for {record.source_path}: {exc}")
        if cancel_event and cancel_event.is_set():
            cancelled = True
            break
        try:
            with _browser_write_lock:
                if cancel_event and cancel_event.is_set():
                    cancelled = True
                    break
                stored = upsert_browser_record(
                    search_hash=search_hash,
                    event_key=record.fingerprint,
                    source_path=record.source_path,
                    source_type=record.source_type,
                    content=content,
                    embedding=embedding,
                    event=event,
                )
            if stored["chunk_added"] or stored["event_added"]:
                added_by_type[record.record_type] += 1
            else:
                skipped_existing += 1
        except Exception as exc:
            failed += 1
            print(f"[browser] indexing failed for {record.source_path}: {exc}")

    browsers_found = sorted({profile.browser for profile in profiles})
    if failed == 0 and not cancelled:
        try:
            _save_backfill_state(next_offsets)
        except OSError as exc:
            print(f"[browser] could not save backfill state: {exc}")
    return {
        # Stable legacy contract:
        "browsers_found": browsers_found,
        "entries_added": sum(added_by_type.values()),
        "skipped_sensitive": skipped_sensitive,
        "skipped_already_indexed": skipped_existing,
        # Structured Phase 6 diagnostics:
        "profiles_found": [
            {
                "browser": profile.browser,
                "profile_id": profile.profile_id,
                "profile_name": profile.display_name,
            }
            for profile in profiles
        ],
        "records_found": len(records),
        "records_found_by_type": {
            kind: found_by_type.get(kind, 0) for kind in ("history", "bookmark", "download")
        },
        "entries_added_by_type": {
            kind: added_by_type.get(kind, 0) for kind in ("history", "bookmark", "download")
        },
        "failed": failed,
        "search_queries_found": sum(bool(record.search_query) for record in safe_records),
        "youtube_videos_found": sum(bool(record.youtube["video_id"]) for record in safe_records),
        "domain_groups": summarize_domain_groups(safe_records)[:25],
        "data_scope": "historical_browser_data",
        "cancelled": cancelled,
        "live_tabs": active_tab_support_status(),
    }


if __name__ == "__main__":
    print(sync_browser_history())
