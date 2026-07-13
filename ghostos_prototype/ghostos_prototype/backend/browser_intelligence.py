"""Pure helpers and data models for GhostOS browser intelligence.

This module deliberately models *historical* browser data only.  Chrome and
Edge history, bookmarks, and downloads can be read from their local profile
files without contacting the internet.  Live tabs need a separately installed
browser extension or an explicitly configured local CDP connection; the
``ActiveTabProvider`` protocol below is the extension point for that future
capability and is not implemented by the current connector.

Keeping URL parsing, deduplication, and record rendering independent from the
filesystem/SQLite reader makes the privacy-sensitive behavior easy to test
with completely offline fixtures.
"""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import PurePath
from typing import Iterable, Protocol, Sequence
from urllib.parse import parse_qsl, parse_qs, urlencode, urlsplit, urlunsplit
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


CHROMIUM_EPOCH = datetime(1601, 1, 1, tzinfo=timezone.utc)
LOCAL_TIMEZONE = "Asia/Kolkata"

_TRACKING_PARAMETERS = {
    "fbclid", "gclid", "dclid", "msclkid", "mc_cid", "mc_eid",
    "igshid", "yclid", "_hsenc", "_hsmi", "vero_conv", "vero_id",
}
_SECRET_QUERY_PARAMETERS = {
    "code", "token", "access_token", "refresh_token", "id_token",
    "oauth_token", "auth", "authorization", "credential", "credentials",
    "password", "passwd", "secret", "client_secret", "state", "ticket",
    "session", "session_id", "sid", "jwt", "signature", "sig", "api_key",
    "apikey", "key", "reset_token",
}
_YOUTUBE_TRACKING_PARAMETERS = {"feature", "si", "pp"}
_YOUTUBE_ID_RE = re.compile(r"^[A-Za-z0-9_-]{6,20}$")
_CONTROL_CHARACTERS_RE = re.compile(r"[\x00-\x1f\x7f]+")


def chromium_time_to_iso(value: int | str | None, timezone_name: str = LOCAL_TIMEZONE) -> str | None:
    """Convert Chromium's microseconds-since-1601 timestamp to local ISO.

    Malformed, zero, or out-of-range values return ``None``.  Callers can
    still index records without a time, but should not fabricate a timeline
    timestamp for them.
    """

    if value in (None, "", 0, "0"):
        return None
    try:
        microseconds = int(value)
        converted = CHROMIUM_EPOCH + timedelta(microseconds=microseconds)
        try:
            local_zone = ZoneInfo(timezone_name)
        except ZoneInfoNotFoundError:
            # Windows Python installations often lack the optional ``tzdata``
            # wheel. India has no daylight-saving transitions, so this exact
            # fixed offset is a safe offline fallback for GhostOS's configured
            # local timezone rather than silently falling back to UTC.
            local_zone = timezone(timedelta(hours=5, minutes=30)) \
                if timezone_name in {"Asia/Kolkata", "Asia/Calcutta"} else timezone.utc
        return converted.astimezone(local_zone).isoformat(timespec="seconds")
    except (OverflowError, TypeError, ValueError):
        return None


def extract_domain(url: str | None) -> str | None:
    """Return a stable, human-readable hostname for domain grouping."""

    if not url:
        return None
    try:
        hostname = (urlsplit(url).hostname or "").casefold().rstrip(".")
    except ValueError:
        return None
    if hostname.startswith("www."):
        hostname = hostname[4:]
    return hostname or None


def _is_host(host: str, domain: str) -> bool:
    return host == domain or host.endswith(f".{domain}")


def normalize_url(url: str | None) -> str:
    """Canonicalize and redact a URL before deduplication or local storage.

    Fragments and well-known tracking parameters are removed, query ordering
    is stabilized, host/scheme casing is normalized, and default ports are
    collapsed.  Search terms and other meaningful query parameters remain.
    """

    raw = (url or "").strip()
    if not raw:
        return ""
    try:
        parts = urlsplit(raw)
    except ValueError:
        return raw.casefold()

    scheme = parts.scheme.casefold()
    host = (parts.hostname or "").casefold().rstrip(".")
    try:
        port = parts.port
    except ValueError:
        port = None
    if port and not ((scheme == "http" and port == 80) or (scheme == "https" and port == 443)):
        netloc = f"{host}:{port}"
    else:
        netloc = host

    kept_query: list[tuple[str, str]] = []
    for key, value in parse_qsl(parts.query, keep_blank_values=True):
        folded = key.casefold()
        if (
            folded in _SECRET_QUERY_PARAMETERS
            or "token" in folded
            or "secret" in folded
            or "signature" in folded
            or "credential" in folded
        ):
            continue
        if folded.startswith("utm_") or folded in _TRACKING_PARAMETERS:
            continue
        if (_is_host(host, "youtube.com") or _is_host(host, "youtu.be")) \
                and folded in _YOUTUBE_TRACKING_PARAMETERS:
            continue
        kept_query.append((key, value))
    kept_query.sort(key=lambda item: (item[0].casefold(), item[1]))

    path = parts.path or ("/" if scheme in {"http", "https"} else "")
    if path != "/":
        path = path.rstrip("/")
    return urlunsplit((scheme, netloc, path, urlencode(kept_query, doseq=True), ""))


def extract_search_query(url: str | None) -> str | None:
    """Extract a user's explicit query from common search-result URLs."""

    if not url:
        return None
    try:
        parsed = urlsplit(url)
        host = (parsed.hostname or "").casefold().removeprefix("www.")
        path = parsed.path.rstrip("/") or "/"
        query = parse_qs(parsed.query)
    except ValueError:
        return None

    parameter: str | None = None
    if (host == "google.com" or host.startswith("google.") or ".google." in host) and path == "/search":
        parameter = "q"
    elif _is_host(host, "bing.com") and path == "/search":
        parameter = "q"
    elif _is_host(host, "duckduckgo.com"):
        parameter = "q"
    elif _is_host(host, "search.yahoo.com") and path.startswith("/search"):
        parameter = "p"
    elif _is_host(host, "search.brave.com") and path.startswith("/search"):
        parameter = "q"
    elif _is_host(host, "ecosia.org") and path.startswith("/search"):
        parameter = "q"
    elif (host == "yandex.com" or host.startswith("yandex.")) and path.startswith("/search"):
        parameter = "text"
    elif _is_host(host, "youtube.com") and path == "/results":
        parameter = "search_query"
    elif _is_host(host, "github.com") and path == "/search":
        parameter = "q"

    values = query.get(parameter or "", [])
    if not values:
        return None
    cleaned = _CONTROL_CHARACTERS_RE.sub(" ", str(values[0])).strip()
    cleaned = " ".join(cleaned.split())
    return cleaned[:500] or None


def extract_youtube_metadata(url: str | None, page_title: str | None = None) -> dict[str, str | None]:
    """Return local YouTube metadata for watch/short/embed/live URLs."""

    video_id: str | None = None
    if url:
        try:
            parsed = urlsplit(url)
            host = (parsed.hostname or "").casefold().removeprefix("www.")
            path_parts = [part for part in parsed.path.split("/") if part]
            if _is_host(host, "youtu.be") and path_parts:
                video_id = path_parts[0]
            elif _is_host(host, "youtube.com"):
                if parsed.path.rstrip("/") == "/watch":
                    video_id = (parse_qs(parsed.query).get("v") or [None])[0]
                elif len(path_parts) >= 2 and path_parts[0] in {"shorts", "embed", "live"}:
                    video_id = path_parts[1]
        except ValueError:
            video_id = None

    if video_id and not _YOUTUBE_ID_RE.fullmatch(video_id):
        video_id = None

    title = (page_title or "").strip() or None
    if title:
        title = re.sub(r"\s*(?:-|\|)\s*YouTube\s*$", "", title, flags=re.IGNORECASE).strip() or None
    return {"video_id": video_id, "title": title if video_id else None}


def _display_path(path: str) -> str:
    """Return just the filename without assuming host OS path semantics."""

    normalized = path.replace("\\", "/")
    return PurePath(normalized).name or path


@dataclass(frozen=True, slots=True)
class BrowserProfile:
    """One persistent Chromium profile discovered on the local machine."""

    browser: str
    profile_id: str
    display_name: str
    directory: str
    history_path: str
    bookmarks_path: str

    @property
    def label(self) -> str:
        browser_label = {"chrome": "Google Chrome", "edge": "Microsoft Edge"}.get(
            self.browser, self.browser.title()
        )
        return f"{browser_label} ({self.display_name})"


@dataclass(frozen=True, slots=True)
class BrowserRecord:
    """Normalized historical browser datum ready for search and Timeline."""

    record_type: str
    browser: str
    profile_id: str
    profile_name: str
    title: str
    url: str | None = None
    timestamp: str | None = None
    local_path: str | None = None
    folder_path: str | None = None
    native_id: str | None = None
    visit_count: int | None = None
    bytes_received: int | None = None
    total_bytes: int | None = None
    metadata: dict[str, str | int | bool | None] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.record_type not in {"history", "bookmark", "download"}:
            raise ValueError(f"Unsupported browser record type: {self.record_type}")

    @property
    def domain(self) -> str | None:
        return extract_domain(self.url)

    @property
    def search_query(self) -> str | None:
        return extract_search_query(self.url)

    @property
    def youtube(self) -> dict[str, str | None]:
        return extract_youtube_metadata(self.url, self.title)

    @property
    def fingerprint(self) -> str:
        """Stable cross-profile dedupe marker stored in ``chunks.file_hash``."""

        canonical_url = normalize_url(self.url)
        if self.record_type == "history" and canonical_url:
            # A URL can be visited repeatedly. Include the actual visit time
            # (or native visit id as a fallback) so a later revisit becomes a
            # new searchable/timeline event while repeat syncs stay idempotent.
            identity = "|".join((canonical_url, self.timestamp or self.native_id or ""))
        elif self.record_type == "bookmark" and canonical_url:
            identity = canonical_url
        elif self.record_type == "download":
            normalized_path = (self.local_path or "").replace("/", "\\").casefold()
            identity = "|".join((canonical_url, normalized_path, self.timestamp or self.native_id or ""))
        else:
            identity = "|".join((self.native_id or "", self.title.casefold(), self.timestamp or ""))
        raw = f"ghostos-browser-v2|{self.record_type}|{identity}"
        return hashlib.sha256(raw.encode("utf-8", errors="replace")).hexdigest()

    @property
    def search_fingerprint(self) -> str:
        """Page-level identity so repeat visits never duplicate embeddings."""
        canonical_url = normalize_url(self.url)
        identity = canonical_url or (self.local_path or "").replace("/", "\\").casefold()
        if not identity:
            identity = self.fingerprint
        raw = f"ghostos-browser-search-v2|{self.record_type}|{identity}"
        return hashlib.sha256(raw.encode("utf-8", errors="replace")).hexdigest()

    @property
    def source_type(self) -> str:
        # ``browser_history_`` is the existing Browser Agent routing
        # namespace.  Keep that prefix so bookmarks/downloads are not
        # accidentally presented as local files, while the suffix and the
        # structured ``record_type`` retain their real historical category.
        if self.record_type == "history":
            return f"browser_history_{self.browser}"
        return f"browser_history_{self.record_type}_{self.browser}"

    @property
    def source_path(self) -> str:
        return self.url or self.local_path or f"browser-record:{self.fingerprint}"

    @property
    def app_label(self) -> str:
        browser_label = {"chrome": "Google Chrome", "edge": "Microsoft Edge"}.get(
            self.browser, self.browser.title()
        )
        return f"{browser_label} ({self.profile_name})"

    def to_search_text(self) -> str:
        """Render deterministic, source-attributed local context for RAG."""

        verb = {"history": "Visited", "bookmark": "Bookmarked", "download": "Downloaded"}[
            self.record_type
        ]
        data_label = {
            "history": "browser history", "bookmark": "browser bookmark",
            "download": "browser download",
        }[self.record_type]
        lines = [
            f"{verb}: {self.title or '(no title)'}",
            f"Browser data type: Historical {data_label} (not a live tab)",
            f"Browser: {self.app_label}",
        ]
        if self.url:
            lines.append(f"URL: {self.url}")
        if self.domain:
            lines.append(f"Domain: {self.domain}")
        if self.timestamp:
            time_label = {"history": "Visited at", "bookmark": "Bookmarked at", "download": "Downloaded at"}[
                self.record_type
            ]
            lines.append(f"{time_label}: {self.timestamp}")
        if self.folder_path:
            lines.append(f"Bookmark folder: {self.folder_path}")
        if self.local_path:
            lines.append(f"Local download path: {self.local_path}")
        if self.visit_count is not None:
            lines.append(f"Visit count: {self.visit_count}")
        if self.search_query:
            lines.append(f"Search query: {self.search_query}")
        youtube = self.youtube
        if youtube["video_id"]:
            lines.append(f"YouTube Video ID: {youtube['video_id']}")
        if youtube["title"]:
            lines.append(f"YouTube title: {youtube['title']}")
        if self.bytes_received is not None:
            lines.append(f"Downloaded bytes: {self.bytes_received}")
        if self.total_bytes is not None:
            lines.append(f"Total bytes: {self.total_bytes}")
        mime_type = self.metadata.get("mime_type")
        if mime_type:
            lines.append(f"MIME type: {mime_type}")
        return "\n".join(lines)

    def to_timeline_event(self) -> dict[str, str] | None:
        """Convert the record to the existing flat Timeline event contract."""

        if not self.timestamp:
            return None
        verb = {"history": "Visited", "bookmark": "Bookmarked", "download": "Downloaded"}[
            self.record_type
        ]
        subtitle = self.url or self.local_path or self.domain or "Historical browser data"
        return {
            "event_type": f"browser_{'visit' if self.record_type == 'history' else self.record_type}",
            "title": f"{verb} {self.title or subtitle}",
            "subtitle": subtitle,
            "app_label": self.app_label,
            "badge_type": "video" if self.youtube["video_id"] else "txt",
            "path_or_url": self.source_path,
            "timestamp": self.timestamp,
        }


def deduplicate_records(records: Iterable[BrowserRecord]) -> list[BrowserRecord]:
    """Keep the newest representative for every stable browser identity."""

    selected: dict[str, BrowserRecord] = {}
    for record in records:
        existing = selected.get(record.fingerprint)
        if existing is None or (record.timestamp or "") > (existing.timestamp or ""):
            selected[record.fingerprint] = record
    return sorted(selected.values(), key=lambda item: item.timestamp or "", reverse=True)


def group_records_by_domain(records: Iterable[BrowserRecord]) -> dict[str, list[BrowserRecord]]:
    """Group historical records by hostname for local browsing summaries."""

    groups: dict[str, list[BrowserRecord]] = defaultdict(list)
    for record in records:
        groups[record.domain or "(local/no domain)"].append(record)
    for domain_records in groups.values():
        domain_records.sort(key=lambda item: item.timestamp or "", reverse=True)
    return dict(sorted(groups.items(), key=lambda item: (-len(item[1]), item[0])))


def summarize_domain_groups(records: Iterable[BrowserRecord]) -> list[dict[str, object]]:
    """Return JSON-friendly domain counts for future UI/API consumers."""

    summaries: list[dict[str, object]] = []
    for domain, domain_records in group_records_by_domain(records).items():
        counts = {kind: 0 for kind in ("history", "bookmark", "download")}
        for record in domain_records:
            counts[record.record_type] += 1
        summaries.append({
            "domain": domain,
            "total": len(domain_records),
            "history": counts["history"],
            "bookmarks": counts["bookmark"],
            "downloads": counts["download"],
            "most_recent": domain_records[0].timestamp if domain_records else None,
        })
    return summaries


class BrowserRecordProvider(Protocol):
    """Extension-ready interface for an explicit local browser provider."""

    provider_name: str
    historical: bool

    def is_available(self) -> bool:
        """Return whether the local provider is configured and reachable."""

    def collect(self, limit: int) -> Sequence[BrowserRecord]:
        """Return structured local records without making cloud requests."""


class ActiveTabProvider(Protocol):
    """Future opt-in live-tab interface; no implementation is bundled."""

    provider_name: str

    def is_available(self) -> bool:
        """Only true when an extension/local CDP provider is configured."""

    def active_tabs(self) -> Sequence[dict[str, str]]:
        """Return explicitly shared active tabs from that provider."""


def active_tab_support_status() -> dict[str, object]:
    """Honest status for the current MVP: historical data only."""

    return {
        "available": False,
        "provider": None,
        "message": "Live-tab access is not configured; GhostOS currently reads historical browser data only.",
    }
