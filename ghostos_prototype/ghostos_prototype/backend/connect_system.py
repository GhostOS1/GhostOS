"""
core/connect_system.py
The single entry point for GhostOS's "Connect to System" experience.

REPLACES the old manual-folder-path workflow: instead of asking the user
to type a path into a popup, GhostOS detects the OS, discovers the user's
standard folders (Desktop, Documents, Downloads, Pictures, Videos, Music)
plus their coding folders (VS Code recent projects, Git repositories),
indexes all of it, and starts the file watcher on those same folders -
all in one step, no textbox, no path typed by the user.

This module owns:
- OS detection
- standard folder discovery (per-OS)
- precise "dev folder" discovery (VS Code projects, Git repos) - see below
- opt-in whole-drive discovery (see SCAN_ENTIRE_DRIVES note below)
- noise/system folder exclusion (separate from indexer.py's sensitive-data
  blacklist, which still applies underneath this)
- orchestrating indexing across every discovered folder
- registering those folders with the in-process watcher

It deliberately does NOT reimplement text extraction, embeddings, vector
search, or file watching - it calls straight into indexer.py and
watcher.py, which already own that logic. This file is orchestration only.

---------------------------------------------------------------------------
On "C Drive / D Drive / VS Code Projects / Git Repositories / Emails"
(the original Files & Data mockup listed these as things GhostOS scans)
---------------------------------------------------------------------------
Whole-drive scanning is still off by default, and still opt-in only when
turned on - that reasoning hasn't changed (installers/system files/caches
drown out real content and it's a much bigger privacy surface). But two of
the items from that list don't actually require a whole-drive scan to do
precisely and cheaply, so they're now real:

- VS Code Projects: VS Code itself keeps a list of recently opened folders
  in its own local storage.json. Reading that gives an exact, small list
  of real project folders - no drive walk needed.
- Git Repositories: found by a *shallow, bounded* walk (default depth 3)
  starting from the already-discovered folders (Documents, Desktop,
  standard dev roots like ~/projects, ~/code, plus the VS Code list
  above) - not a whole-drive walk.

Email is NOT implemented here. Indexing email needs a real integration
(Outlook COM API / Microsoft Graph, or IMAP + OAuth for Gmail/etc.) - it
isn't something a filesystem scan can provide, and faking it would be
worse than admitting it's not built yet. See discover_email_accounts()
below for the honest stub and where that work would plug in later (the
Email Agent from the v2.0 architecture).

Whole-drive scanning (C:\\, D:\\, external drives) remains a separate,
explicit opt-in (scan_entire_drives=True) - see discover_drives() and
SCAN_ENTIRE_DRIVES below.
"""

import os
import json
import platform
import threading
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

from config import INDEX_FAILED_FILES_LIMIT
from indexer import FileProcessResult, index_folders
from watcher import watch_folders

# Whole-drive scanning (entire C:\, D:\, external drives) is intentionally
# OFF by default. It's now fully implemented (see discover_drives()) but
# stays opt-in: callers must explicitly pass scan_entire_drives=True to
# connect_system(). The default "Connect to System" flow never touches it -
# this constant is just the documented default for that opt-in.
SCAN_ENTIRE_DRIVES = False

# Bounded depth for the Git-repository search below. Keeps repo discovery
# fast and predictable instead of an unbounded walk.
GIT_SEARCH_MAX_DEPTH = 3

# Extra folder names, beyond the OS-standard ones, that commonly hold code
# projects. Only used to *seed* the (bounded) Git-repo search - these are
# still real, specific folders, not a drive-wide walk.
COMMON_DEV_ROOT_NAMES = ["projects", "code", "dev", "src", "workspace", "repos", "github"]

# Directory *names* skipped everywhere during scanning - noise/system
# folders, not sensitive-data folders. This is a different concern from
# indexer.py's SENSITIVE_PATTERNS (which protects secrets like .env or
# id_rsa wherever they appear): this list exists purely to protect scan
# speed and retrieval quality by keeping dependency/cache/system junk out
# of the index. Matched case-insensitively against each path segment.
EXCLUDED_DIR_NAMES = {
    "node_modules", "__pycache__", ".git", ".svn", ".venv", "venv", "env",
    "$recycle.bin", "system volume information", "windows",
    "program files", "program files (x86)", "programdata", "appdata",
    ".cache", ".npm", "site-packages", "dist", "build", ".next", ".gradle",
}

# Extra names only pruned during a *whole-drive* scan (scan_entire_drives=True) -
# kept separate from EXCLUDED_DIR_NAMES so the standard-folder scan doesn't
# accidentally start skipping a legitimately-named user folder like
# "~/Documents/temp" or "~/Downloads/backup".
DRIVE_SCAN_EXTRA_EXCLUDED_DIR_NAMES = {
    "temp", "tmp", "recovery", "boot", "perflogs", "intel", "amd", "nvidia",
    "drivers", "winsxs", "$windows.~bt", "$windows.~ws",
}


def detect_os() -> str:
    """Returns 'windows', 'macos', or 'linux'."""
    system = platform.system().lower()
    if system == "windows":
        return "windows"
    if system == "darwin":
        return "macos"
    return "linux"


# Windows Known Folder GUIDs (FOLDERID_*) for the six standard folders.
# These are the *canonical, stable* IDs Windows itself uses internally -
# looking a folder up this way returns wherever it *actually* lives right
# now, which matters a lot in practice: OneDrive's "Backup your folder"
# feature (on by default on most new Windows 11 setups, and common on
# Windows 10) moves Desktop/Documents/Pictures to
# C:\Users\<you>\OneDrive\Desktop (etc.) and the plain
# C:\Users\<you>\Desktop path is then either missing or empty. Guessing
# the path from Path.home() silently misses that; asking Windows via
# SHGetKnownFolderPath does not.
_WINDOWS_KNOWN_FOLDER_GUIDS = {
    "Desktop": "{B4BFCC3A-DB2C-424C-B029-7FE99A87C641}",
    "Documents": "{FDD39AD0-238F-46AF-ADB4-6C85480369C7}",
    "Downloads": "{374DE290-123F-4565-9164-39C4925E467B}",
    "Pictures": "{33E28130-4E1E-4676-835A-98395C3BC3BB}",
    "Videos": "{18989B1D-99B5-455B-841C-AB7C74E4DDFC}",
    "Music": "{4BD8D571-6D19-48D3-BE97-422220080E43}",
}


def _resolve_windows_known_folder(name: str) -> Optional[Path]:
    """Asks Windows for a known folder's *current* real path via
    SHGetKnownFolderPath, instead of assuming it lives at
    ~\\<name>. Returns None if the call fails or the path doesn't exist
    (falls back to the home-relative guess in the caller)."""
    import ctypes
    from ctypes import wintypes

    guid = _WINDOWS_KNOWN_FOLDER_GUIDS.get(name)
    if not guid:
        return None
    try:
        class GUID(ctypes.Structure):
            _fields_ = [
                ("Data1", wintypes.DWORD), ("Data2", wintypes.WORD),
                ("Data3", wintypes.WORD), ("Data4", ctypes.c_byte * 8),
            ]

        rfid = GUID()
        ctypes.windll.ole32.CLSIDFromString(guid, ctypes.byref(rfid))

        path_ptr = ctypes.c_wchar_p()
        result = ctypes.windll.shell32.SHGetKnownFolderPath(
            ctypes.byref(rfid), 0, None, ctypes.byref(path_ptr)
        )
        if result != 0:
            return None
        path_str = path_ptr.value
        ctypes.windll.ole32.CoTaskMemFree(path_ptr)
        return Path(path_str) if path_str else None
    except Exception as e:
        print(f"[connect_system] known-folder lookup failed for {name}: {e}")
        return None


def discover_standard_folders() -> list[Path]:
    """
    Returns the user's standard folders for the current machine, filtered
    to ones that actually exist.

    On Windows, each folder's *real, current* location is resolved via the
    Known Folder API (SHGetKnownFolderPath) rather than assumed to be
    ~\\<name> - this is what correctly finds OneDrive-redirected
    Desktop/Documents/Pictures instead of silently returning nothing for
    them. macOS and Linux don't have this redirection concept, so
    Path.home()-relative paths are used there, same as before.
    """
    home = Path.home()
    names = ["Desktop", "Documents", "Downloads", "Pictures", "Videos", "Music"]

    candidates: list[Path] = []
    if detect_os() == "windows":
        for name in names:
            resolved = _resolve_windows_known_folder(name)
            candidates.append(resolved if resolved is not None else home / name)
    else:
        candidates = [home / name for name in names]

    return [p for p in candidates if p.exists() and p.is_dir()]


def _vscode_storage_paths() -> list[Path]:
    """Locations VS Code (and common forks) keep their global storage.json,
    per OS. We read this rather than guessing folder names - it's the
    actual list VS Code itself maintains of recently opened folders."""
    home = Path.home()
    paths = []
    if platform.system().lower() == "windows":
        appdata = os.environ.get("APPDATA", str(home / "AppData" / "Roaming"))
        bases = [Path(appdata) / "Code", Path(appdata) / "Code - Insiders"]
    elif platform.system().lower() == "darwin":
        bases = [home / "Library" / "Application Support" / "Code",
                  home / "Library" / "Application Support" / "Code - Insiders"]
    else:
        bases = [home / ".config" / "Code", home / ".config" / "Code - Insiders"]

    for base in bases:
        paths.append(base / "User" / "globalStorage" / "storage.json")
    return paths


def discover_vscode_projects() -> list[Path]:
    """
    Reads VS Code's own record of recently opened folders instead of
    guessing where "code projects" live. Returns real, existing folders
    only. Safe and cheap - this is a small JSON file read, not a scan.
    """
    found: list[Path] = []
    for storage_path in _vscode_storage_paths():
        if not storage_path.exists():
            continue
        try:
            data = json.loads(storage_path.read_text(encoding="utf-8", errors="ignore"))
        except Exception:
            continue

        entries = (
            data.get("openedPathsList", {}).get("entries", [])
            or data.get("windowsState", {}).get("openedWindows", [])
        )
        for entry in entries:
            folder_uri = entry.get("folderUri") or entry.get("workspace", {}).get("configPath")
            if not folder_uri:
                continue
            path_str = folder_uri
            if path_str.startswith("file:///"):
                # "file://" is 7 chars; on Windows the remainder starts
                # with a drive letter ("C:/Users/..."), so strip one more
                # char (the extra leading slash). On POSIX the remainder
                # must KEEP its leading slash ("/home/you/...") or it
                # turns into a relative path that silently never matches.
                if platform.system().lower() == "windows":
                    path_str = path_str[8:]
                else:
                    path_str = path_str[7:]
            path_str = path_str.replace("%3A", ":")
            try:
                p = Path(path_str)
                if p.exists() and p.is_dir():
                    found.append(p)
            except Exception:
                continue

    # de-dupe while preserving order
    seen = set()
    unique = []
    for p in found:
        rp = str(p.resolve())
        if rp not in seen:
            seen.add(rp)
            unique.append(p)
    return unique


def discover_git_repositories(
    search_roots: list[Path],
    max_depth: int = GIT_SEARCH_MAX_DEPTH,
    excluded_folders: list[str | Path] | None = None,
) -> list[Path]:
    """
    Finds Git repositories with a *bounded* walk from search_roots (not a
    whole-drive scan). A folder counts as a repo if it directly contains a
    .git directory. max_depth limits how far below each root we look, so
    this stays fast even on a large Documents folder.
    """
    home = Path.home()
    roots = list(search_roots)
    for name in COMMON_DEV_ROOT_NAMES:
        candidate = home / name
        if candidate.exists() and candidate.is_dir():
            roots.append(candidate)

    configured_exclusions = [Path(path) for path in (excluded_folders or [])]
    repos: list[Path] = []
    seen = set()

    def walk(path: Path, depth: int):
        if depth > max_depth:
            return
        if _is_within_any(path, configured_exclusions):
            return
        try:
            if (path / ".git").exists():
                rp = str(path.resolve())
                if rp not in seen:
                    seen.add(rp)
                    repos.append(path)
                return  # don't descend into a repo's own internals looking for nested repos
            for child in path.iterdir():
                if not child.is_dir():
                    continue
                if child.name.lower() in EXCLUDED_DIR_NAMES:
                    continue
                walk(child, depth + 1)
        except (PermissionError, OSError):
            return

    for root in roots:
        if root.exists() and root.is_dir():
            walk(root, 0)

    return repos


def discover_drives() -> list[Path]:
    """
    Whole-drive discovery - only ever called when scan_entire_drives=True
    is explicitly passed to connect_system(). Returns drive/mount roots to
    scan (e.g. C:\\, D:\\ on Windows; /, /mnt/*, /media/* on Linux; /Volumes/*
    on macOS). This is intentionally its own function so it's obvious in
    a diff/review exactly what "opt-in whole-drive scan" touches.
    """
    system = platform.system().lower()
    drives: list[Path] = []

    if system == "windows":
        import string
        from ctypes import windll  # available on Windows only

        bitmask = windll.kernel32.GetLogicalDrives()
        for i, letter in enumerate(string.ascii_uppercase):
            if bitmask & (1 << i):
                drive = Path(f"{letter}:\\")
                if drive.exists():
                    drives.append(drive)
    elif system == "darwin":
        volumes = Path("/Volumes")
        if volumes.exists():
            drives.extend([p for p in volumes.iterdir() if p.is_dir()])
        drives.append(Path.home())
    else:  # linux
        for base in [Path("/mnt"), Path("/media")]:
            if base.exists():
                for user_dir in base.iterdir():
                    if user_dir.is_dir():
                        drives.extend([p for p in user_dir.iterdir() if p.is_dir()])
        drives.append(Path.home())

    return drives


def discover_email_accounts() -> dict:
    """
    Honest stub. Email indexing needs a real integration (Outlook COM API /
    Microsoft Graph on Windows, or IMAP+OAuth for Gmail/etc.) - it can't be
    done via filesystem discovery like the folders above, so this
    deliberately does NOT pretend to scan anything. It exists so
    connect_system()'s summary can report accurately instead of silently
    omitting email with no explanation. This is where the future Email
    Agent (from the v2.0 architecture) would plug in.
    """
    return {"supported": False, "reason": "Email indexing requires a separate account integration (not yet built)."}


def is_excluded_dir(path: Path) -> bool:
    """Passed into indexer.index_folders() as the noise/system-folder filter
    for the standard (non-drive) scan."""
    return path.name.lower() in EXCLUDED_DIR_NAMES


def is_excluded_dir_for_drive_scan(path: Path) -> bool:
    """Stricter filter used only for the opt-in whole-drive scan - prunes
    everything the standard filter does, plus extra OS/system noise that's
    only a concern once you're walking an entire drive."""
    name = path.name.lower()
    return name in EXCLUDED_DIR_NAMES or name in DRIVE_SCAN_EXTRA_EXCLUDED_DIR_NAMES


def _resolved_existing_folders(paths: list[str] | None) -> list[Path]:
    """Return unique, existing configured folders without inventing paths."""
    folders: list[Path] = []
    seen: set[str] = set()
    for raw_path in paths or []:
        try:
            folder = Path(raw_path).expanduser().resolve(strict=True)
        except (OSError, RuntimeError):
            continue
        if not folder.is_dir():
            continue
        key = str(folder).casefold()
        if key not in seen:
            seen.add(key)
            folders.append(folder)
    return folders


def _is_within_any(path: Path, roots: list[Path]) -> bool:
    try:
        resolved = path.resolve(strict=False)
    except (OSError, RuntimeError):
        return True
    for root in roots:
        try:
            resolved.relative_to(root)
            return True
        except ValueError:
            continue
    return False


# ---------------------------------------------------------------------------
# Background indexing job (Phase 2)
# ---------------------------------------------------------------------------
# connect_system() below only does the fast, synchronous part now (OS
# detect + standard-folder discovery + starting the watcher). Everything
# expensive - VS Code/Git/drive discovery and the actual indexing/embedding
# pass - runs in a daemon thread kicked off at the end of connect_system(),
# so the function (and whatever HTTP request called it) returns in well
# under a second instead of waiting minutes for a full scan.
#
# Since nothing is blocking on that thread, progress has to be something a
# caller can *poll* rather than something streamed back on the original
# request - get_background_status() is that poll target (wired up by
# app.py's GET /api/index-status). _bg_lock guards _bg_state since the
# background thread writes to it while a Flask request thread may read it
# concurrently.
_bg_lock = threading.Lock()
_bg_cancel_event = threading.Event()
def _idle_background_state() -> dict:
    return {
        "active": False,       # a background job is currently running
        "done": True,          # the most recent background job has finished (or none has run yet)
        "phase": "idle",       # idle | discovering_vscode_projects | discovering_git_repos | discovering_drives | indexing | finished | error
        "current_folder": None,
        "current_file": None,
        "folders_total": 0,
        "folders_completed": 0,
        "files_processed": 0,
        "files_seen": 0,
        "files_completed": 0,
        "files_failed": 0,
        "files_skipped": 0,
        "files_too_large": 0,
        "chunks_added": 0,
        "vscode_projects_found": 0,
        "git_repos_found": 0,
        "drive_folders_found": 0,
        "error": None,
        "failed_files": [],
        "cancel_requested": False,
        "cancelled": False,
        "started_at": None,
        "finished_at": None,
        "stats": {},
    }


_bg_state: dict = _idle_background_state()


def get_background_status() -> dict:
    """Snapshot of the background indexing job's current progress. Safe to
    call from any thread/request; returns a copy so callers can't mutate
    the shared state."""
    with _bg_lock:
        snapshot = dict(_bg_state)
        snapshot["failed_files"] = list(_bg_state.get("failed_files", []))
        return snapshot


def reset_background_status() -> bool:
    """Return an inactive job to its honest initial state after data clear.

    Refuse to reset while work is active so a clear/status request can never
    hide or detach a running indexing thread from its cancellation/progress
    state.
    """
    with _bg_lock:
        if _bg_state.get("active"):
            return False
        _bg_cancel_event.clear()
        _bg_state.clear()
        _bg_state.update(_idle_background_state())
        return True


def cancel_background_indexing() -> dict:
    """Request graceful cancellation; an in-flight Ollama call may finish first."""
    with _bg_lock:
        if not _bg_state["active"]:
            return {"accepted": False, "reason": "No indexing job is running."}
        _bg_cancel_event.set()
        _bg_state["cancel_requested"] = True
        _bg_state["phase"] = "cancelling"
    return {"accepted": True, "message": "Indexing cancellation requested."}


def _dedupe_folders(folders: list[Path]) -> list[Path]:
    seen = set()
    unique: list[Path] = []
    for f in folders:
        rp = str(f.resolve())
        if rp not in seen:
            seen.add(rp)
            unique.append(f)
    return unique


def _run_background_discovery_and_indexing(
    standard_folders: list[Path],
    include_dev_folders: bool,
    scan_entire_drives: bool,
    excluded_folders: list[Path] | None = None,
) -> None:
    """
    The actual heavy lifting, run entirely off the request thread: VS
    Code/Git/drive discovery, then indexing + embedding of every discovered
    folder, updating _bg_state as it goes. standard_folders is already
    being watched and is usable by the time this even starts (see
    connect_system()) - this thread only adds more folders to the watcher
    and to the search index on top of that.
    """
    try:
        excluded_folders = excluded_folders or []

        dev_folders: list[Path] = []
        if include_dev_folders and not _bg_cancel_event.is_set():
            with _bg_lock:
                _bg_state["phase"] = "discovering_vscode_projects"
            vscode_projects = discover_vscode_projects()
            with _bg_lock:
                _bg_state["vscode_projects_found"] = len(vscode_projects)
            dev_folders.extend(vscode_projects)

            if not _bg_cancel_event.is_set():
                with _bg_lock:
                    _bg_state["phase"] = "discovering_git_repos"
                git_repos = discover_git_repositories(
                    standard_folders + vscode_projects,
                    excluded_folders=excluded_folders,
                )
                with _bg_lock:
                    _bg_state["git_repos_found"] = len(git_repos)
                dev_folders.extend(git_repos)

        # Discovery can return a root that is itself inside a configured
        # exclusion.  Directory-pruning callbacks only see descendants, so
        # reject excluded roots before either the watcher or indexer receives
        # them.
        all_folders = [
            folder
            for folder in _dedupe_folders(standard_folders + dev_folders)
            if not _is_within_any(folder, excluded_folders)
        ]

        drive_folders: list[Path] = []
        if scan_entire_drives and not _bg_cancel_event.is_set():
            with _bg_lock:
                _bg_state["phase"] = "discovering_drives"
            drive_folders = [
                folder
                for folder in discover_drives()
                if not _is_within_any(folder, excluded_folders)
            ]
            with _bg_lock:
                _bg_state["drive_folders_found"] = len(drive_folders)
            all_folders = _dedupe_folders(all_folders + drive_folders)

        # Extend the watcher (already running on standard_folders since
        # connect_system() started it) to also cover the newly discovered
        # dev/drive folders. watch_folders() skips folders it's already
        # watching, so this is additive, not a restart.
        watch_folders([str(f) for f in all_folders])

        with _bg_lock:
            _bg_state["phase"] = "indexing"
            _bg_state["folders_total"] = len(all_folders)

        def on_folder_start(folder: Path):
            with _bg_lock:
                _bg_state["current_folder"] = folder.name

        def on_folder_done(folder: Path, result: dict):
            with _bg_lock:
                _bg_state["folders_completed"] += 1

        def on_file_start(path: Path):
            with _bg_lock:
                _bg_state["files_seen"] += 1
                _bg_state["current_file"] = str(path)

        def on_file_done(path: Path, result: dict):
            status = result["status"]
            with _bg_lock:
                _bg_state["files_completed"] += 1
                _bg_state["chunks_added"] += result.get("chunks_added", 0)
                if status in {
                    FileProcessResult.PROCESSED,
                    FileProcessResult.UNSUPPORTED,
                    FileProcessResult.EMPTY,
                    FileProcessResult.TOO_LARGE,
                }:
                    _bg_state["files_processed"] += 1
                if status == FileProcessResult.TOO_LARGE:
                    _bg_state["files_too_large"] += 1
                if status == FileProcessResult.FAILED:
                    _bg_state["files_failed"] += 1
                    if len(_bg_state["failed_files"]) < INDEX_FAILED_FILES_LIMIT:
                        _bg_state["failed_files"].append({
                            "path": str(path),
                            "stage": result.get("stage") or "unknown",
                            "error": result.get("error") or "Unknown indexing error",
                        })
                if status in {
                    FileProcessResult.SENSITIVE,
                    FileProcessResult.ALREADY_INDEXED,
                    FileProcessResult.DUPLICATE_CONTENT,
                    FileProcessResult.NOT_FOUND,
                }:
                    _bg_state["files_skipped"] += 1

        # Whole-drive folders need the stricter exclusion filter;
        # standard/dev folders use the normal one - same two-pass split as
        # before, just running in the background now instead of blocking
        # the original caller.
        def configured_exclusion(path: Path) -> bool:
            return is_excluded_dir(path) or _is_within_any(path, excluded_folders)

        def configured_drive_exclusion(path: Path) -> bool:
            return is_excluded_dir_for_drive_scan(path) or _is_within_any(path, excluded_folders)

        summary = index_folders(
            [str(f) for f in all_folders if f not in drive_folders],
            is_excluded_dir=configured_exclusion,
            on_folder_start=on_folder_start,
            on_folder_done=on_folder_done,
            on_file_start=on_file_start,
            on_file_done=on_file_done,
            cancel_event=_bg_cancel_event,
        )

        if drive_folders and not _bg_cancel_event.is_set():
            drive_summary = index_folders(
                [str(f) for f in drive_folders],
                is_excluded_dir=configured_drive_exclusion,
                on_folder_start=on_folder_start,
                on_folder_done=on_folder_done,
                on_file_start=on_file_start,
                on_file_done=on_file_done,
                cancel_event=_bg_cancel_event,
            )
            summary["stats"] = drive_summary["stats"]

        with _bg_lock:
            was_cancelled = _bg_cancel_event.is_set() or summary.get("cancelled", False)
            _bg_state["phase"] = "cancelled" if was_cancelled else "finished"
            _bg_state["done"] = True
            _bg_state["active"] = False
            _bg_state["current_folder"] = None
            _bg_state["current_file"] = None
            _bg_state["cancelled"] = was_cancelled
            _bg_state["finished_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
            _bg_state["stats"] = summary.get("stats", {})

    except Exception as e:
        print(f"[connect_system] background indexing failed: {e}")
        with _bg_lock:
            _bg_state["error"] = str(e)
            _bg_state["phase"] = "error"
            _bg_state["done"] = True
            _bg_state["active"] = False
            _bg_state["current_file"] = None
            _bg_state["finished_at"] = datetime.now().astimezone().isoformat(timespec="seconds")


def connect_system(
    progress_callback: Optional[Callable[[dict], None]] = None,
    include_dev_folders: bool = True,
    scan_entire_drives: bool = False,
    additional_folders: list[str] | None = None,
    excluded_folders: list[str] | None = None,
) -> dict:
    """
    The "Connect to System" flow, split into two phases so the user isn't
    stuck waiting on a modal for however long a full scan takes:

    Phase 1 (this function, synchronous, target: well under a second):
      1. detect OS
      2. discover standard folders (Desktop/Documents/Downloads/Pictures/
         Videos/Music - resolved via the Windows Known Folder API where
         applicable, see discover_standard_folders())
      3. start the watcher on exactly those folders immediately, so no
         file change is ever missed even while Phase 2 is still running
      4. return - the caller can treat status "connected" as "usable now"

    Phase 2 (background daemon thread, started at the end of this
    function, keeps running after this function - and the request that
    called it - has already returned):
      1. VS Code project discovery + Git repository discovery (if
         include_dev_folders)
      2. whole-drive discovery (only if scan_entire_drives=True)
      3. index + embed everything discovered (the actually slow part -
         one Ollama call per chunk)
      4. extend the watcher to cover the newly discovered folders too

    Progress for Phase 2 isn't streamed back on this call (there's nothing
    to stream it on, since this function has already returned) - poll
    get_background_status() instead. progress_callback here only covers
    Phase 1's few, fast steps.
    """
    def emit(event_type: str, **kwargs):
        if progress_callback:
            progress_callback({"type": event_type, **kwargs})

    os_name = detect_os()
    emit("progress", message=f"Detected OS: {os_name}")

    configured_exclusions = _resolved_existing_folders(excluded_folders)
    discovered = discover_standard_folders()
    configured = _resolved_existing_folders(additional_folders)
    standard_folders = _dedupe_folders(discovered + configured)
    standard_folders = [
        folder for folder in standard_folders
        if not _is_within_any(folder, configured_exclusions)
    ]
    if not standard_folders and not scan_entire_drives:
        emit("progress", message="No standard folders found on this system.")
        return {"status": "no_folders", "folders": 0, "files": 0}

    emit("progress", message=f"Found {len(standard_folders)} standard folders.")

    # Start watching the standard folders right away - this just registers
    # watchdog handlers, it's cheap and doesn't wait on anything below.
    watch_folders([str(f) for f in standard_folders])
    emit("progress", message="Watching standard folders for changes.")

    email_status = discover_email_accounts()

    # Kick off Phase 2 in the background and return immediately. If a
    # background job is already running (e.g. the user double-clicked
    # Connect), don't start a second overlapping one.
    with _bg_lock:
        already_running = _bg_state["active"]
        if not already_running:
            _bg_cancel_event.clear()
            _bg_state.update({
                "active": True, "done": False, "phase": "starting",
                "current_folder": None, "current_file": None,
                "folders_total": len(standard_folders), "folders_completed": 0,
                "files_processed": 0, "files_seen": 0, "files_completed": 0,
                "files_failed": 0, "files_skipped": 0, "files_too_large": 0,
                "chunks_added": 0,
                "vscode_projects_found": 0, "git_repos_found": 0,
                "drive_folders_found": 0, "error": None, "failed_files": [],
                "cancel_requested": False, "cancelled": False,
                "started_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                "finished_at": None, "stats": {},
            })

    if already_running:
        emit("progress", message="Background indexing is already running.")
    else:
        threading.Thread(
            target=_run_background_discovery_and_indexing,
            args=(
                standard_folders, include_dev_folders, scan_entire_drives,
                configured_exclusions,
            ),
            daemon=True,
        ).start()
        emit("progress", message="Indexing continues in the background.")

    return {
        "status": "connected",
        "folders": len(standard_folders),
        "standard_folders": len(standard_folders),
        "folder_list": [str(f) for f in standard_folders],
        "email": email_status,
        "background_indexing": True,
    }
