"""Local-only health and setup diagnostics for GhostOS."""

from pathlib import Path

import requests

import activity_tracker
import browser_connector
import watcher
from config import CHAT_MODEL, EMBED_MODEL, OLLAMA_BASE_URL
from vectorstore import DB_PATH, get_stats
from settings_store import get_settings
from ocr_service import get_ocr_status
from voice_service import get_voice_status


def _model_present(configured: str, installed: set[str]) -> bool:
    return configured in installed or (":" not in configured and f"{configured}:latest" in installed)


def get_diagnostics() -> dict:
    problems: list[str] = []
    commands: list[str] = []
    ollama = {"available": False, "chat_model": False, "embedding_model": False, "error": None}
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=3)
        response.raise_for_status()
        installed = {m.get("name") for m in response.json().get("models", []) if m.get("name")}
        ollama.update({
            "available": True,
            "chat_model": _model_present(CHAT_MODEL, installed),
            "embedding_model": _model_present(EMBED_MODEL, installed),
            "installed_models": sorted(installed),
        })
    except requests.RequestException as exc:
        ollama["error"] = str(exc)
        problems.append(f"Ollama is unavailable at {OLLAMA_BASE_URL}")

    if ollama["available"] and not ollama["chat_model"]:
        problems.append(f"Ollama chat model {CHAT_MODEL} is not installed")
        commands.append(f"ollama pull {CHAT_MODEL}")
    if ollama["available"] and not ollama["embedding_model"]:
        problems.append(f"Ollama embedding model {EMBED_MODEL} is not installed")
        commands.append(f"ollama pull {EMBED_MODEL}")

    try:
        database_stats = get_stats()
        database = {"available": True, "path": str(DB_PATH), **database_stats}
    except Exception as exc:
        database = {"available": False, "path": str(DB_PATH), "error": str(exc)}
        problems.append("SQLite database is unavailable")

    observer = getattr(watcher, "_observer", None)
    collector_pause_event = getattr(watcher, "_collectors_paused", None)
    collectors_paused = bool(collector_pause_event and collector_pause_event.is_set())
    watcher_status = {
        "active": bool(observer and observer.is_alive() and not collectors_paused),
        "paused": collectors_paused,
        "folders": list(getattr(watcher, "_watched_folders", [])),
    }
    settings = get_settings()
    browser_paths = {}
    browser_profiles = []
    try:
        browser_profiles = browser_connector.discover_browser_profiles()
        for profile in browser_profiles:
            if Path(profile.history_path).exists():
                browser_paths[profile.label] = profile.history_path
    except (OSError, PermissionError):
        pass
    browser_status = {
        "enabled": bool(settings["browser_history_enabled"]),
        "active": bool(
            settings["browser_history_enabled"]
            and getattr(watcher, "_browser_thread_started", False)
            and not collectors_paused
        ),
        "paused": collectors_paused,
        "detected_browsers": sorted({profile.browser for profile in browser_profiles}),
        "profiles": len(browser_profiles),
        "history_paths": browser_paths,
    }
    ocr_status = {**get_ocr_status(), "enabled": bool(settings["ocr_enabled"])}
    if ocr_status["enabled"] and not ocr_status["available"]:
        problems.append("OCR is enabled, but local Tesseract OCR is unavailable")
        commands.append(r".\setup_backend.ps1 -WithOCR")

    voice_status = {**get_voice_status(), "enabled": bool(settings["voice_enabled"])}
    if voice_status["enabled"] and not voice_status["available"]:
        problems.append("Voice input is enabled, but the local speech engine is unavailable")
        commands.append(r".\setup_backend.ps1 -WithVoice")

    activity_pause_event = getattr(activity_tracker, "_paused", None)
    activity_paused = bool(activity_pause_event and activity_pause_event.is_set())
    ready = not problems
    return {
        "status": "ready" if ready else "degraded",
        "backend": {"available": True, "local_only": True},
        "ollama": ollama,
        "database": database,
        "watcher": watcher_status,
        "browser_connector": browser_status,
        "activity_tracker": {
            "enabled": bool(settings["activity_tracking_enabled"]),
            "active": bool(
                settings["activity_tracking_enabled"]
                and getattr(activity_tracker, "_started", False)
                and not activity_paused
            ),
            "paused": activity_paused,
        },
        "ocr": ocr_status,
        "voice": voice_status,
        "problems": problems,
        "commands": commands,
    }