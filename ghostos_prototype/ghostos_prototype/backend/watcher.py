"""
watcher.py
Background file-watching for GhostOS.

CHANGED: this used to read its folder list from watched_folders.json and
only run as a separate `python watcher.py` process, updated whenever
app.py wrote a newly-indexed folder path to that file. That's gone now.

watch_folders() is called directly, in-process, by
core/connect_system.py right after the initial scan finishes, with the
exact folder list connect_system() just discovered and indexed. There is
no JSON file to fall out of sync with, and no second process to remember
to start - the Flask app IS the watcher process now.

`python watcher.py` standalone is kept only as a fallback for restarting
the watcher without redoing "Connect to System" (e.g. after a crash) - it
runs the same folder discovery connect_system() uses, so it still doesn't
touch a JSON file.
"""

import time
import threading
from pathlib import Path

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from indexer import process_file, FileProcessResult, init_db, SUPPORTED_EXTENSIONS
from vectorstore import get_stats, remove_source
from browser_connector import sync_browser_history
from activity_tracker import start_activity_tracker

BROWSER_SYNC_INTERVAL_SECONDS = 120  # how often to re-check browser history

# Files are often written in multiple small disk operations (especially
# large PDFs/docs). DEBOUNCE_SECONDS waits for writes to settle before
# processing, so we don't try to read a half-written file.
DEBOUNCE_SECONDS = 3

# Module-level state so watch_folders() can be called more than once
# (e.g. if connect_system() ever re-runs) without spawning duplicate
# Observers or duplicate browser-sync threads. This replaces
# watched_folders.json as the "source of truth" - it's simply in-memory,
# owned by whichever process called watch_folders() (normally the Flask
# app started by app.py).
_observer = None
_handler = None
_watched_folders: list[str] = []
_browser_thread_started = False
_state_lock = threading.Lock()


class DebouncedHandler(FileSystemEventHandler):
    """
    Collects file change events and processes each changed file once,
    a few seconds after its last modification, instead of reacting to
    every individual write event immediately.
    """

    def __init__(self):
        self._pending = {}  # path -> timer
        self._lock = threading.Lock()

    def _schedule(self, path: str):
        p = Path(path)
        if not p.is_file():
            return
        if p.suffix.lower() not in SUPPORTED_EXTENSIONS:
            return

        with self._lock:
            existing = self._pending.get(path)
            if existing:
                existing.cancel()
            timer = threading.Timer(DEBOUNCE_SECONDS, self._process, args=(path,))
            self._pending[path] = timer
            timer.start()

    def _process(self, path: str):
        with self._lock:
            self._pending.pop(path, None)

        status, chunks_added = process_file(path)
        if status == FileProcessResult.PROCESSED:
            print(f"[watcher] indexed: {path} (+{chunks_added} chunks)")
        elif status == FileProcessResult.SENSITIVE:
            print(f"[watcher] skipped (sensitive path): {path}")
        # ALREADY_INDEXED / UNSUPPORTED / EMPTY -> silent, not worth logging every time

    def on_created(self, event):
        if not event.is_directory:
            self._schedule(event.src_path)

    def on_modified(self, event):
        if not event.is_directory:
            self._schedule(event.src_path)

    def on_deleted(self, event):
        if not event.is_directory:
            remove_source(event.src_path)
            print(f"[watcher] removed deleted file: {event.src_path}")

    def on_moved(self, event):
        if not event.is_directory:
            remove_source(event.src_path)
            self._schedule(event.dest_path)


def _browser_sync_loop():
    """Runs in its own thread, periodically pulling new browser history
    into memory alongside the file watcher."""
    while True:
        try:
            result = sync_browser_history()
            if result["entries_added"] > 0:
                print(f"[browser] synced: +{result['entries_added']} new visits "
                      f"from {result['browsers_found']} "
                      f"(skipped {result['skipped_sensitive']} sensitive)")
        except Exception as e:
            print(f"[browser] sync error: {e}")
        time.sleep(BROWSER_SYNC_INTERVAL_SECONDS)


def watch_folders(folders: list[str]) -> dict:
    """
    Starts (or extends) watching on exactly the given folders - the
    replacement for the old JSON-file-driven workflow. Safe to call more
    than once: already-watched folders are skipped, and the background
    Observer + browser-sync threads are only ever started once per
    process. Non-blocking - watchdog's Observer runs its own thread, so
    this returns immediately and is safe to call from a Flask request
    handler (which is exactly how core/connect_system.py uses it).
    """
    global _observer, _handler, _browser_thread_started
    init_db()
    start_activity_tracker()

    with _state_lock:
        if _observer is None:
            _handler = DebouncedHandler()
            _observer = Observer()
            _observer.start()

        newly_added = 0
        for folder in folders:
            if folder in _watched_folders:
                continue
            if Path(folder).exists():
                _observer.schedule(_handler, folder, recursive=True)
                _watched_folders.append(folder)
                newly_added += 1
                print(f"[watcher] watching: {folder}")
            else:
                print(f"[watcher] skipping missing folder: {folder}")

        if not _browser_thread_started:
            t = threading.Thread(target=_browser_sync_loop, daemon=True)
            t.start()
            _browser_thread_started = True
            print(f"[browser] history sync thread started (checks every {BROWSER_SYNC_INTERVAL_SECONDS}s)")

    return {"watching": list(_watched_folders), "newly_added": newly_added}


def stop_watcher():
    """Optional cleanup hook - not required for the Flask dev server
    (daemon threads die with the process), but useful for tests/scripts
    that want a clean shutdown."""
    global _observer
    with _state_lock:
        if _observer:
            _observer.stop()
            _observer.join()
            _observer = None


if __name__ == "__main__":
    # Standalone fallback: reuse connect_system's own folder discovery so
    # this still never touches a JSON file.
    from connect_system import discover_standard_folders

    init_db()
    folders = [str(f) for f in discover_standard_folders()]
    if not folders:
        print("[watcher] No standard folders found on this system.")

    watch_folders(folders)
    stats = get_stats()
    print(f"[watcher] running. {len(folders)} folder(s) watched. "
          f"Current memory: {stats['total_files']} files, {stats['total_chunks']} chunks.")
    print("[watcher] Press Ctrl+C to stop.")

    try:
        while True:
            time.sleep(5)
    except KeyboardInterrupt:
        stop_watcher()
