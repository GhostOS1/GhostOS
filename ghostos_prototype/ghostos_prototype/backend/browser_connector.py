"""
browser_connector.py
Reads local browser history (Chrome and Edge) and adds it to GhostOS's
memory, using the same vectorstore as files.

How it works:
- Chrome/Edge store history in a local SQLite file under
  %LocalAppData%\\...\\User Data\\Default\\History
- That file is locked while the browser is running, so we copy it to a
  temp file first (a read-only snapshot) instead of opening it directly.
- Each visited URL becomes one memory entry: title + url + visit time
  (+ video ID if it's a YouTube link). Already-seen URLs are skipped on
  future runs (deduped by URL hash), so this is safe to run repeatedly.

Privacy notes:
- Browsers themselves already don't log Incognito/Private browsing to
  this History file - that's a browser guarantee, not something this
  script has to enforce.
- URLs matching the same SENSITIVE_PATTERNS blacklist used for files
  (banking, password managers, etc.) are skipped here too.
"""

import os
import sqlite3
import shutil
import tempfile
import hashlib
from pathlib import Path
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from urllib.parse import urlparse, parse_qs

from embeddings import get_embedding
from vectorstore import init_db, add_chunk, file_already_indexed, add_event
from indexer import SENSITIVE_PATTERNS

# Extra patterns specific to browsing (on top of the file blacklist)
BROWSER_SENSITIVE_PATTERNS = SENSITIVE_PATTERNS + [
    "paypal.com",
    "chase.com",
    "bankofamerica",
    "wellsfargo",
    "accounts.google.com",
    "login.microsoftonline.com",
]

HISTORY_PATHS = {
    "chrome": Path(os.path.expandvars(r"%LOCALAPPDATA%\Google\Chrome\User Data\Default\History")),
    "edge": Path(os.path.expandvars(r"%LOCALAPPDATA%\Microsoft\Edge\User Data\Default\History")),
}

MAX_ENTRIES_PER_RUN = 2000  # most recent N visits per browser, per run


def is_sensitive_url(url: str) -> bool:
    url_lower = url.lower()
    return any(pattern in url_lower for pattern in BROWSER_SENSITIVE_PATTERNS)


def chrome_time_to_datetime(chrome_timestamp: int) -> str:
    """Chrome/Edge store timestamps as microseconds since 1601-01-01 (WebKit epoch)."""
    if not chrome_timestamp:
        return "unknown time"
    try:
        dt_utc = datetime(1601, 1, 1, tzinfo=timezone.utc) + timedelta(microseconds=chrome_timestamp)
        return dt_utc.astimezone(ZoneInfo("Asia/Kolkata")).isoformat(timespec="seconds")
    except (OverflowError, ValueError):
        return "unknown time"


def extract_youtube_id(url: str) -> str | None:
    parsed = urlparse(url)
    if "youtube.com" in parsed.netloc:
        qs = parse_qs(parsed.query)
        if "v" in qs:
            return qs["v"][0]
    if "youtu.be" in parsed.netloc:
        return parsed.path.lstrip("/")
    return None


def _copy_locked_db(source_path: Path) -> Path | None:
    """Copies the browser's History file to a temp location so we can
    read it even while the browser is open and holding a lock on it."""
    if not source_path.exists():
        return None
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".sqlite")
    os.close(tmp_fd)
    try:
        shutil.copy2(source_path, tmp_path)
        return Path(tmp_path)
    except Exception as e:
        print(f"[browser] could not copy {source_path}: {e}")
        return None


def sync_browser_history() -> dict:
    """
    Reads history from all detected browsers and adds new (not yet
    seen) entries to the memory index. Safe to call repeatedly - already
    indexed URLs are skipped.
    """
    init_db()
    total_added = 0
    total_skipped_sensitive = 0
    total_skipped_existing = 0
    browsers_found = []

    for browser_name, history_path in HISTORY_PATHS.items():
        if not history_path.exists():
            continue
        browsers_found.append(browser_name)

        tmp_copy = _copy_locked_db(history_path)
        if tmp_copy is None:
            continue

        try:
            conn = sqlite3.connect(str(tmp_copy))
            cur = conn.execute(
                "SELECT url, title, last_visit_time, visit_count "
                "FROM urls ORDER BY last_visit_time DESC LIMIT ?",
                (MAX_ENTRIES_PER_RUN,),
            )
            rows = cur.fetchall()
            conn.close()
        except Exception as e:
            print(f"[browser] could not read {browser_name} history: {e}")
            continue
        finally:
            try:
                os.remove(tmp_copy)
            except OSError:
                pass

        for url, title, last_visit_time, visit_count in rows:
            if not url:
                continue

            if is_sensitive_url(url):
                total_skipped_sensitive += 1
                continue

            url_hash = hashlib.sha256(url.encode("utf-8")).hexdigest()
            if file_already_indexed(url_hash):
                total_skipped_existing += 1
                continue

            visited_at = chrome_time_to_datetime(last_visit_time)
            youtube_id = extract_youtube_id(url)

            content_lines = [
                f"Visited: {title or '(no title)'}",
                f"URL: {url}",
                f"Visited at: {visited_at}",
                f"Visit count: {visit_count}",
            ]
            if youtube_id:
                content_lines.append(f"YouTube Video ID: {youtube_id}")
            content = "\n".join(content_lines)

            try:
                embedding = get_embedding(content)
                add_chunk(
                    source_path=url,
                    source_type=f"browser_history_{browser_name}",
                    content=content,
                    embedding=embedding,
                    file_hash=url_hash,
                )
                badge_type = "video" if youtube_id else "txt"
                app_label = {"chrome": "Google Chrome", "edge": "Microsoft Edge"}.get(browser_name, browser_name)
                add_event(
                    event_type="browser_visit",
                    title=f"Visited {title or url}",
                    subtitle=url,
                    app_label=app_label,
                    badge_type=badge_type,
                    path_or_url=url,
                    timestamp=visited_at,
                )
                total_added += 1
            except Exception as e:
                print(f"[browser] embedding failed for {url}: {e}")

    return {
        "browsers_found": browsers_found,
        "entries_added": total_added,
        "skipped_sensitive": total_skipped_sensitive,
        "skipped_already_indexed": total_skipped_existing,
    }


if __name__ == "__main__":
    result = sync_browser_history()
    print(result)
