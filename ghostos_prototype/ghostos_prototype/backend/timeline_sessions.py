"""Deterministic grouping and summaries for locally stored timeline events."""

from collections import Counter
from datetime import datetime, timedelta, timezone

INDIA_TIMEZONE = timezone(timedelta(hours=5, minutes=30))


def _parse_timestamp(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value)
        if parsed.tzinfo is None:
            # Older indexer rows were stored as naive Windows local time.
            # GhostOS's configured local calendar is India time, so attach
            # +05:30 before normalizing instead of treating them as UTC.
            parsed = parsed.replace(tzinfo=INDIA_TIMEZONE)
        parsed = parsed.astimezone(timezone.utc).replace(tzinfo=None)
        return parsed
    except (TypeError, ValueError):
        return datetime.min


def group_events(events: list[dict], inactivity_minutes: int = 20) -> list[dict]:
    """Group real events into sessions separated by inactivity gaps."""
    ordered = sorted(events, key=lambda event: _parse_timestamp(event.get("timestamp", "")))
    if not ordered:
        return []
    gap = timedelta(minutes=max(1, inactivity_minutes))
    raw_sessions: list[list[dict]] = [[ordered[0]]]
    for event in ordered[1:]:
        previous = _parse_timestamp(raw_sessions[-1][-1].get("timestamp", ""))
        current = _parse_timestamp(event.get("timestamp", ""))
        if current - previous > gap:
            raw_sessions.append([event])
        else:
            raw_sessions[-1].append(event)

    sessions = []
    for number, items in enumerate(raw_sessions, start=1):
        compressed = []
        duplicate_counts: Counter = Counter()
        for item in items:
            key = (item.get("event_type"), item.get("title"), item.get("app_label"), item.get("path_or_url"))
            duplicate_counts[key] += 1
            if duplicate_counts[key] == 1:
                compressed.append(dict(item))
        for item in compressed:
            key = (item.get("event_type"), item.get("title"), item.get("app_label"), item.get("path_or_url"))
            item["occurrences"] = duplicate_counts[key]

        apps = Counter(item.get("app_label") for item in items if item.get("app_label"))
        kinds = Counter(item.get("event_type") for item in items if item.get("event_type"))
        dominant_app = apps.most_common(1)[0][0] if apps else "Computer activity"
        sessions.append({
            "id": number,
            "start": items[0].get("timestamp"),
            "end": items[-1].get("timestamp"),
            "label": f"Activity in {dominant_app}",
            "event_count": len(items),
            "applications": [{"name": name, "count": count} for name, count in apps.most_common()],
            "categories": dict(kinds),
            "events": compressed,
        })
    return sessions


def summarize_day(events: list[dict], date: str | None = None) -> dict:
    sessions = group_events(events)
    types = Counter(event.get("event_type") for event in events if event.get("event_type"))
    apps = Counter(event.get("app_label") for event in events if event.get("app_label"))
    return {
        "date": date,
        "event_count": len(events),
        "session_count": len(sessions),
        "file_events": sum(count for kind, count in types.items() if str(kind).startswith("file_")),
        "browser_events": sum(count for kind, count in types.items() if str(kind).startswith("browser_")),
        "application_events": types.get("app_focus", 0),
        "top_applications": [{"name": name, "count": count} for name, count in apps.most_common(5)],
        "sessions": sessions,
        "message": "No locally recorded activity for this day." if not events else (
            f"GhostOS recorded {len(events)} events across {len(sessions)} activity sessions."
        ),
    }
