"""Honest proactive insights derived only from local GhostOS records."""

import hashlib
import json
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

from vectorstore import get_recent_files, get_timeline

STATE_PATH = Path(__file__).with_name("ghostos_insights_state.json")


def _id(kind: str, key: str) -> str:
    return hashlib.sha256(f"{kind}:{key}".encode("utf-8")).hexdigest()[:16]


def _dismissed() -> set[str]:
    try:
        data = json.loads(STATE_PATH.read_text(encoding="utf-8"))
        return set(data.get("dismissed", []))
    except (OSError, json.JSONDecodeError):
        return set()


def dismiss_insight(insight_id: str) -> None:
    dismissed = _dismissed()
    dismissed.add(insight_id)
    STATE_PATH.write_text(json.dumps({"dismissed": sorted(dismissed)}, indent=2), encoding="utf-8")


def clear_insight_state() -> None:
    STATE_PATH.unlink(missing_ok=True)


def build_insights(now: datetime | None = None) -> list[dict]:
    now = now or datetime.now().astimezone()
    insights: list[dict] = []
    recent_files = get_recent_files(limit=100)
    important = [f for f in recent_files if f.get("category") not in {"Others", "Archives"}]
    if important:
        file = important[0]
        insight_id = _id("recent_file", file["path"])
        insights.append({
            "id": insight_id,
            "type": "recent_file",
            "title": "Recently modified file",
            "message": f"{file['name']} is one of your most recently modified indexed files.",
            "target": file["path"],
            "evidence": {"modified_at": file.get("modified_at"), "category": file.get("category")},
        })

    today = now.date().isoformat()
    events = get_timeline(date_prefix=today, limit=2000)
    apps = Counter(event.get("app_label") for event in events if event.get("app_label"))
    if apps:
        app, count = apps.most_common(1)[0]
        insight_id = _id("frequent_app", f"{today}:{app}")
        insights.append({
            "id": insight_id,
            "type": "frequent_application",
            "title": "Frequently used today",
            "message": f"{app} appears in {count} locally recorded timeline events today.",
            "target": None,
            "evidence": {"date": today, "event_count": count},
        })

    dismissed = _dismissed()
    return [insight for insight in insights if insight["id"] not in dismissed]
