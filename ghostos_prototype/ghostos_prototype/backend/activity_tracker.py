"""Windows foreground application tracker for the GhostOS Timeline."""

import ctypes
import threading
import time
from datetime import datetime

try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    psutil = None
    _HAS_PSUTIL = False

from vectorstore import add_event
from settings_store import get_settings

_started = False
_lock = threading.Lock()
_operation_lock = threading.Lock()
_paused = threading.Event()


def _foreground_window():
    user32 = ctypes.windll.user32
    hwnd = user32.GetForegroundWindow()
    if not hwnd:
        return None
    length = user32.GetWindowTextLengthW(hwnd)
    buf = ctypes.create_unicode_buffer(length + 1)
    user32.GetWindowTextW(hwnd, buf, length + 1)
    pid = ctypes.c_ulong()
    user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
    if not buf.value.strip() or not pid.value:
        return None
    if _HAS_PSUTIL:
        try:
            process = psutil.Process(pid.value)
            app_name = process.name()
        except (psutil.Error, OSError):
            app_name = "Unknown application"
    else:
        app_name = "Windows application"
    return app_name, buf.value.strip()


def _loop():
    previous = None
    while True:
        if _paused.is_set() or not get_settings()["activity_tracking_enabled"]:
            previous = None
            time.sleep(2)
            continue
        try:
            with _operation_lock:
                if not _paused.is_set():
                    current = _foreground_window()
                    if current and current != previous:
                        app_name, window_title = current
                        add_event(
                            event_type="app_focus",
                            title=window_title,
                            subtitle=f"Active window in {app_name}",
                            app_label=app_name,
                            badge_type="app",
                            path_or_url="",
                            timestamp=datetime.now().astimezone().isoformat(timespec="seconds"),
                        )
                        previous = current
        except Exception as exc:
            print(f"[activity] foreground tracking error: {exc}")
        time.sleep(1)


def start_activity_tracker():
    global _started
    _paused.clear()
    with _lock:
        if _started:
            return
        threading.Thread(target=_loop, daemon=True, name="ghostos-activity-tracker").start()
        _started = True
        print("[activity] Windows foreground application tracking started")


def pause_activity_tracker() -> None:
    """Pause collection and wait for an in-flight local event write to finish."""
    _paused.set()
    with _operation_lock:
        pass
