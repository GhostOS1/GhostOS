"""Validated, local JSON settings for GhostOS."""

import json
import threading
from copy import deepcopy
from pathlib import Path
from urllib.parse import urlparse

SETTINGS_PATH = Path(__file__).with_name("ghostos_settings.json")
_lock = threading.Lock()

DEFAULT_SETTINGS = {
    "ollama_url": "http://127.0.0.1:11434",
    "chat_model": "gemma4:e2b",
    "embedding_model": "nomic-embed-text",
    "indexed_folders": [],
    "excluded_folders": [],
    "ocr_enabled": False,
    "voice_enabled": False,
    "activity_tracking_enabled": True,
    "browser_history_enabled": True,
    "scan_entire_drives": False,
    "action_permissions": {
        "open_file": True,
        "open_folder": True,
        "open_url": True,
        "open_app": True,
        "reveal_in_explorer": True,
        "create_text_note": True,
        "create_folder": True,
    },
}


def _merged(data: dict | None) -> dict:
    result = deepcopy(DEFAULT_SETTINGS)
    if not isinstance(data, dict):
        return result
    for key in DEFAULT_SETTINGS:
        if key == "action_permissions" and isinstance(data.get(key), dict):
            result[key].update({k: bool(v) for k, v in data[key].items() if k in result[key]})
        elif key in data:
            result[key] = data[key]
    return result


def get_settings() -> dict:
    with _lock:
        if not SETTINGS_PATH.exists():
            return deepcopy(DEFAULT_SETTINGS)
        try:
            return _merged(json.loads(SETTINGS_PATH.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            return deepcopy(DEFAULT_SETTINGS)


def update_settings(updates: dict) -> dict:
    if not isinstance(updates, dict):
        raise ValueError("Settings payload must be an object")
    unknown = sorted(set(updates) - set(DEFAULT_SETTINGS))
    if unknown:
        raise ValueError(f"Unknown settings: {', '.join(unknown)}")
    current = get_settings()
    for key, value in updates.items():
        if key in {"ocr_enabled", "voice_enabled", "activity_tracking_enabled", "browser_history_enabled", "scan_entire_drives"}:
            if not isinstance(value, bool):
                raise ValueError(f"{key} must be true or false")
        elif key in {"indexed_folders", "excluded_folders"}:
            if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
                raise ValueError(f"{key} must be a list of paths")
            value = list(dict.fromkeys(item.strip() for item in value if item.strip()))
        elif key == "ollama_url":
            parsed = urlparse(str(value))
            if parsed.scheme not in {"http", "https"} or not parsed.hostname:
                raise ValueError("ollama_url must be a valid HTTP(S) URL")
            if parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
                raise ValueError("GhostOS only accepts a local Ollama URL")
        elif key in {"chat_model", "embedding_model"}:
            if not isinstance(value, str) or not value.strip() or len(value) > 100:
                raise ValueError(f"{key} must be a model name")
        elif key == "action_permissions":
            if not isinstance(value, dict):
                raise ValueError("action_permissions must be an object")
            unknown_actions = sorted(set(value) - set(current[key]))
            if unknown_actions:
                raise ValueError(f"Unknown action permissions: {', '.join(unknown_actions)}")
            if not all(isinstance(enabled, bool) for enabled in value.values()):
                raise ValueError("action permission values must be true or false")
            current[key].update(value)
            continue
        current[key] = value
    with _lock:
        SETTINGS_PATH.write_text(json.dumps(current, indent=2), encoding="utf-8")
    return current