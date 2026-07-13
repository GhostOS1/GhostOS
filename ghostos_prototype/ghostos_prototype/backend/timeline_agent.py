"""
agents/timeline_agent.py
Timeline Agent - recent activity from the events table (populated by
indexer.py and browser_connector.py, see vectorstore.py's `events` table).
Only pulled when the question actually sounds like it's asking about
*when*/*recently* something happened, not on every request. Split out of
app.py; logic unchanged, just relocated into its own agent module.
"""

import re
from difflib import SequenceMatcher
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from vectorstore import get_timeline

TIME_INTENT_KEYWORDS = {
    "today", "yesterday", "recent", "recently", "earlier", "this morning",
    "last night", "last week", "past week", "this week", "just now", "my day",
    "my activity", "what did i do", "activity timeline", "timeline",
}

try:
    LOCAL_TIMEZONE = ZoneInfo("Asia/Kolkata")
except ZoneInfoNotFoundError:
    # Windows Python installations do not always include the IANA tzdata
    # package. India has no daylight-saving transitions, so +05:30 is an
    # exact offline fallback rather than a guessed timezone.
    LOCAL_TIMEZONE = timezone(timedelta(hours=5, minutes=30))
_BROWSER_REFERENCE_RE = re.compile(
    r"\b(?:that|the|previous)\s+(?:page|website|site|link|url)\b.*\b(?:earlier|visited|discussed)\b",
    re.IGNORECASE,
)

_CONTEXT_STOPWORDS = {
    "a", "about", "activity", "an", "and", "at", "did", "do", "earlier",
    "find", "for", "from", "happened", "i", "in", "last", "me", "morning",
    "my", "night", "on", "open", "opened", "page", "past", "recent",
    "recently", "remember", "saw", "see", "show", "site", "that", "the",
    "this", "timeline", "today", "visit", "visited", "was", "website",
    "week", "what", "when", "where", "which", "yesterday",
}


def _context_terms(query: str) -> list[str]:
    """Return only terms that identify *what* the user is remembering."""
    tokens = re.findall(r"[a-z0-9][a-z0-9._-]*", (query or "").casefold())
    return [
        token for token in tokens
        if token not in _CONTEXT_STOPWORDS
        and not re.fullmatch(r"\d{1,4}(?:[-/]\d{1,2}){0,2}", token)
    ]


def _event_relevance(event: dict, terms: list[str]) -> float:
    """Small deterministic lexical/typo score; never needs Ollama."""
    weighted_fields = (
        (str(event.get("title") or "").casefold(), 5.0),
        (str(event.get("subtitle") or "").casefold(), 3.0),
        (str(event.get("path_or_url") or "").casefold(), 4.0),
        (str(event.get("app_label") or "").casefold(), 1.5),
        (str(event.get("event_type") or "").casefold(), 1.0),
    )
    score = 0.0
    for term in terms:
        matched = False
        for field, weight in weighted_fields:
            if term in field:
                score += weight
                matched = True
                continue
            # Context is often recalled with a small spelling error.  Bound
            # fuzzy comparison to field tokens so unrelated long text cannot
            # accumulate noise.
            if not matched and any(
                SequenceMatcher(None, term, token).ratio() >= 0.84
                for token in re.findall(r"[a-z0-9][a-z0-9._-]*", field)[:80]
            ):
                score += weight * 0.65
                matched = True
        if matched:
            score += 0.5
    return score


def wants_timeline(text: str) -> bool:
    t = " ".join((text or "").casefold().split())
    # This phrasing normally points to a page already surfaced in the
    # conversation.  Let Memory Agent resolve it; without session memory it
    # falls through to semantic browser-history search instead of returning
    # an arbitrary recent event.
    if _BROWSER_REFERENCE_RE.search(t):
        return False
    return any(re.search(rf"(?<!\w){re.escape(kw)}(?!\w)", t) for kw in TIME_INTENT_KEYWORDS)


def resolve_date_prefixes(query: str, now: datetime | None = None) -> list[str] | None:
    """Resolve natural/explicit dates into local ``YYYY-MM-DD`` prefixes.

    ``None`` means no date filter.  "last week" follows the usual calendar
    meaning (the previous Monday through Sunday); "past week" means the
    rolling seven-day window including today.
    """
    current = now or datetime.now(LOCAL_TIMEZONE)
    if current.tzinfo is None:
        current = current.replace(tzinfo=LOCAL_TIMEZONE)
    else:
        current = current.astimezone(LOCAL_TIMEZONE)
    today = current.date()
    text = " ".join((query or "").casefold().split())

    iso = re.search(r"\b(20\d{2}-\d{2}-\d{2})\b", text)
    if iso:
        try:
            return [date.fromisoformat(iso.group(1)).isoformat()]
        except ValueError:
            return []

    numeric = re.search(r"\b(\d{1,2})[/-](\d{1,2})[/-](20\d{2})\b", text)
    if numeric:
        try:
            parsed = date(int(numeric.group(3)), int(numeric.group(2)), int(numeric.group(1)))
            return [parsed.isoformat()]
        except ValueError:
            return []

    if re.search(r"\byesterday\b|\blast night\b", text):
        return [(today - timedelta(days=1)).isoformat()]
    if re.search(r"\btoday\b|\bthis morning\b", text):
        return [today.isoformat()]
    if re.search(r"\blast week\b", text):
        this_monday = today - timedelta(days=today.weekday())
        previous_monday = this_monday - timedelta(days=7)
        return [(previous_monday + timedelta(days=offset)).isoformat() for offset in range(7)]
    if re.search(r"\bpast week\b|\blast 7 days\b", text):
        return [(today - timedelta(days=offset)).isoformat() for offset in range(6, -1, -1)]
    if re.search(r"\bthis week\b", text):
        monday = today - timedelta(days=today.weekday())
        return [(monday + timedelta(days=offset)).isoformat() for offset in range((today - monday).days + 1)]
    return None


def _parse_event_time(value: str) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(str(value))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=LOCAL_TIMEZONE)
        return parsed.astimezone(LOCAL_TIMEZONE)
    except (TypeError, ValueError):
        return None


def deduplicate_timeline(events: list[dict], within_seconds: int = 90) -> list[dict]:
    """Remove event-import duplicates while preserving genuine revisits."""
    ordered = sorted(events, key=lambda event: str(event.get("timestamp") or ""), reverse=True)
    seen_at: dict[tuple[str, str, str], datetime | None] = {}
    unique = []
    for event in ordered:
        target = str(event.get("path_or_url") or event.get("title") or "").strip().casefold()
        key = (
            str(event.get("event_type") or "").casefold(),
            target,
            str(event.get("app_label") or "").casefold(),
        )
        occurred = _parse_event_time(event.get("timestamp"))
        previous = seen_at.get(key)
        if key in seen_at and previous is not None and occurred is not None:
            if abs((previous - occurred).total_seconds()) <= within_seconds:
                continue
        elif key in seen_at and previous is None and occurred is None:
            continue
        seen_at[key] = occurred
        unique.append(event)
    return unique


def timeline_agent(query: str, limit: int = 8) -> list:
    if not wants_timeline(query):
        return []
    date_prefixes = resolve_date_prefixes(query)
    if date_prefixes == []:
        return []
    if date_prefixes is None:
        events = get_timeline(limit=max(limit * 25, 500))
    else:
        events = []
        # Busy days can contain thousands of focus changes.  Pull a bounded
        # but broad day window, then rank by the remembered subject below.
        per_day_limit = 5000
        for prefix in date_prefixes:
            events.extend(get_timeline(date_prefix=prefix, limit=per_day_limit))
    events = deduplicate_timeline(events)
    terms = _context_terms(query)
    if terms:
        ranked = [(_event_relevance(event, terms), event) for event in events]
        ranked = [(score, event) for score, event in ranked if score > 0]
        ranked.sort(
            key=lambda item: (item[0], str(item[1].get("timestamp") or "")),
            reverse=True,
        )
        return [event for _, event in ranked[:limit]]

    query_text = (query or "").casefold()
    if re.search(r"\b(?:website|site|page|url|link|browser)\b", query_text):
        events = [event for event in events if str(event.get("event_type") or "").startswith("browser_")]
    elif re.search(r"\b(?:file|document|pdf|image|folder)\b", query_text):
        events = [
            event for event in events
            if str(event.get("event_type") or "") in {"file_indexed", "file_changed", "file_opened"}
        ]
    return events[:limit]
