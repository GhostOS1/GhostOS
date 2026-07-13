"""Offline tests for multi-profile Chromium historical-data ingestion."""

from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import browser_connector
from browser_intelligence import (
    CHROMIUM_EPOCH,
    BrowserRecord,
    active_tab_support_status,
    extract_search_query,
    extract_youtube_metadata,
    group_records_by_domain,
    normalize_url,
)


def _chrome_timestamp(iso_timestamp: str) -> int:
    value = datetime.fromisoformat(iso_timestamp).astimezone(timezone.utc)
    return int((value - CHROMIUM_EPOCH).total_seconds() * 1_000_000)


def _create_history_db(path: Path, urls: list[tuple], downloads: list[tuple] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    try:
        conn.execute(
            """CREATE TABLE urls (
                   id INTEGER PRIMARY KEY, url TEXT, title TEXT,
                   last_visit_time INTEGER, visit_count INTEGER,
                   typed_count INTEGER, hidden INTEGER DEFAULT 0
               )"""
        )
        conn.executemany(
            "INSERT INTO urls (id, url, title, last_visit_time, visit_count, typed_count, hidden) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            urls,
        )
        if downloads is not None:
            conn.execute(
                """CREATE TABLE downloads (
                       id INTEGER PRIMARY KEY, current_path TEXT, target_path TEXT,
                       start_time INTEGER, end_time INTEGER, received_bytes INTEGER,
                       total_bytes INTEGER, state INTEGER, opened INTEGER,
                       mime_type TEXT, tab_url TEXT, site_url TEXT, referrer TEXT
                   )"""
            )
            conn.execute(
                "CREATE TABLE downloads_url_chains (id INTEGER, chain_index INTEGER, url TEXT)"
            )
            conn.executemany(
                """INSERT INTO downloads (
                       id, current_path, target_path, start_time, end_time,
                       received_bytes, total_bytes, state, opened, mime_type,
                       tab_url, site_url, referrer
                   ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                downloads,
            )
            conn.execute(
                "INSERT INTO downloads_url_chains VALUES (?, ?, ?)",
                (1, 0, "https://cdn.example.com/files/paper.pdf"),
            )
        conn.commit()
    finally:
        conn.close()


class BrowserIntelligencePureTests(unittest.TestCase):
    def test_url_metadata_and_tracking_dedupe(self):
        first = "HTTPS://WWW.Example.com:443/path/?b=2&utm_source=news&a=1#section"
        second = "https://www.example.com/path?a=1&b=2"
        self.assertEqual(normalize_url(first), normalize_url(second))
        self.assertEqual(
            extract_search_query("https://www.google.co.in/search?q=local+private+AI&sourceid=chrome"),
            "local private AI",
        )
        self.assertEqual(
            extract_search_query("https://www.youtube.com/results?search_query=ghostos+demo"),
            "ghostos demo",
        )
        metadata = extract_youtube_metadata(
            "https://www.youtube.com/shorts/AbCdEf12345?si=tracking",
            "GhostOS Demo - YouTube",
        )
        self.assertEqual(metadata, {"video_id": "AbCdEf12345", "title": "GhostOS Demo"})

    def test_browser_page_search_hash_changes_with_embedding_model(self):
        record = BrowserRecord(
            record_type="history",
            browser="chrome",
            profile_id="Default",
            profile_name="Personal",
            title="Local AI notes",
            url="https://example.com/local-ai",
            timestamp="2026-07-13T10:00:00+05:30",
        )
        with patch.object(browser_connector, "EMBED_MODEL", "embed-model-a"):
            first = browser_connector._model_search_hash(record)
        with patch.object(browser_connector, "EMBED_MODEL", "embed-model-b"):
            second = browser_connector._model_search_hash(record)

        self.assertNotEqual(first, second)
        self.assertEqual(len(first), 64)
        self.assertEqual(len(second), 64)

    def test_structured_record_dedupe_grouping_and_honest_scope(self):
        base = dict(
            record_type="history", browser="chrome", profile_id="Default",
            profile_name="Personal", title="Example", timestamp="2026-07-13T10:00:00+05:30",
        )
        one = BrowserRecord(url="https://example.com/page?utm_source=a", **base)
        two = BrowserRecord(
            url="https://example.com/page?utm_source=b",
            **{**base, "profile_id": "Profile 1", "profile_name": "Work"},
        )
        revisit = BrowserRecord(
            url="https://example.com/page?utm_source=c",
            **{**base, "timestamp": "2026-07-13T10:05:00+05:30"},
        )
        bookmark = BrowserRecord(
            record_type="bookmark", browser="edge", profile_id="Default",
            profile_name="Default", title="Example bookmark", url="https://example.com/page",
        )
        self.assertEqual(one.fingerprint, two.fingerprint)
        self.assertNotEqual(one.fingerprint, revisit.fingerprint)
        self.assertEqual(one.search_fingerprint, revisit.search_fingerprint)
        self.assertNotEqual(one.fingerprint, bookmark.fingerprint)
        self.assertEqual(bookmark.source_type, "browser_history_bookmark_edge")
        self.assertIn("Historical browser history (not a live tab)", one.to_search_text())
        groups = group_records_by_domain([one, bookmark])
        self.assertEqual(len(groups["example.com"]), 2)
        self.assertFalse(active_tab_support_status()["available"])


class ChromiumFixtureTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name) / "Chrome" / "User Data"
        self.default = self.root / "Default"
        self.work = self.root / "Profile 1"
        timestamp_1 = _chrome_timestamp("2026-07-13T04:30:00+00:00")
        timestamp_2 = _chrome_timestamp("2026-07-13T05:30:00+00:00")

        _create_history_db(
            self.default / "History",
            [
                (1, "https://www.google.com/search?q=local+ai&utm_source=test", "local ai - Google Search", timestamp_1, 3, 1, 0),
                (2, "https://www.youtube.com/watch?v=AbCdEf12345&si=tracking", "GhostOS Demo - YouTube", timestamp_2, 2, 0, 0),
                (3, "https://accounts.google.com/signin", "Sign in", timestamp_2, 1, 0, 0),
                (4, "chrome://settings/", "Settings", timestamp_2, 1, 0, 0),
            ],
            downloads=[(
                1,
                r"C:\Users\Tester\Downloads\paper.pdf.crdownload",
                r"C:\Users\Tester\Downloads\paper.pdf",
                timestamp_1,
                timestamp_2,
                1024,
                2048,
                1,
                0,
                "application/pdf",
                "https://example.com/research/article",
                "https://example.com",
                "https://search.example.com",
            )],
        )
        _create_history_db(
            self.work / "History",
            [
                # Same logical search as Default; newer profile record should win.
                (1, "https://www.google.com/search?utm_medium=x&q=local+ai", "Local AI", timestamp_2, 5, 2, 0),
                (2, "https://docs.python.org/3/library/sqlite3.html", "sqlite3 docs", timestamp_2, 1, 0, 0),
            ],
        )
        self.default.joinpath("Bookmarks").write_text(json.dumps({
            "roots": {
                "bookmark_bar": {
                    "type": "folder",
                    "name": "Bookmarks bar",
                    "children": [{
                        "type": "folder",
                        "name": "Research",
                        "children": [{
                            "type": "url",
                            "id": "7",
                            "guid": "bookmark-guid",
                            "name": "GhostOS short",
                            "url": "https://youtu.be/ZyXwVu98765?si=tracking",
                            "date_added": str(timestamp_1),
                        }],
                    }],
                }
            }
        }), encoding="utf-8")
        self.root.joinpath("Local State").write_text(json.dumps({
            "profile": {"info_cache": {
                "Default": {"name": "Personal"},
                "Profile 1": {"name": "Work"},
                "System Profile": {"name": "System"},
            }}
        }), encoding="utf-8")
        self.edge_root = Path(self.temp_dir.name) / "Edge" / "User Data"
        self.edge_default = self.edge_root / "Default"
        _create_history_db(
            self.edge_default / "History",
            [(1, "https://www.bing.com/search?q=private+second+brain", "private second brain", timestamp_2, 1, 1, 0)],
        )
        self.edge_root.joinpath("Local State").write_text(json.dumps({
            "profile": {"info_cache": {"Default": {"name": "Edge Personal"}}}
        }), encoding="utf-8")
        self.paths = {
            "chrome": self.default / "History",
            "edge": self.edge_default / "History",
        }

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_multiple_profiles_history_bookmarks_and_optional_downloads(self):
        profiles, records = browser_connector.collect_browser_records(self.paths)
        self.assertEqual([(p.browser, p.profile_id, p.display_name) for p in profiles], [
            ("chrome", "Default", "Personal"),
            ("chrome", "Profile 1", "Work"),
            ("edge", "Default", "Edge Personal"),
        ])
        counts = {kind: sum(record.record_type == kind for record in records)
                  for kind in ("history", "bookmark", "download")}
        # The same Google page was visited at two different times, so both
        # concrete visits remain available to Timeline; chrome:// is ignored.
        self.assertEqual(counts, {"history": 6, "bookmark": 1, "download": 1})
        google_visits = [record for record in records if record.search_query == "local ai"]
        self.assertEqual(len(google_visits), 2)
        self.assertEqual({record.profile_name for record in google_visits}, {"Personal", "Work"})
        self.assertEqual(len({record.search_fingerprint for record in google_visits}), 1)
        self.assertEqual(len({record.fingerprint for record in google_visits}), 2)
        bookmark = next(record for record in records if record.record_type == "bookmark")
        self.assertEqual(bookmark.folder_path, "Bookmarks bar / Research")
        self.assertEqual(bookmark.youtube["video_id"], "ZyXwVu98765")
        download = next(record for record in records if record.record_type == "download")
        self.assertEqual(download.title, "paper.pdf")
        self.assertEqual(download.domain, "example.com")
        self.assertIn("Local download path:", download.to_search_text())

    def test_sync_contract_sensitive_filter_and_repeat_dedupe(self):
        indexed: set[str] = set()
        chunks: list[dict] = []
        events: list[dict] = []

        def already_indexed(file_hash: str) -> bool:
            return file_hash in indexed

        def upsert_browser_record(**kwargs) -> dict[str, bool]:
            chunk_added = False
            if kwargs["search_hash"] not in indexed and kwargs.get("embedding"):
                indexed.add(kwargs["search_hash"])
                chunks.append({
                    "file_hash": kwargs["search_hash"],
                    "source_path": kwargs["source_path"],
                    "source_type": kwargs["source_type"],
                    "content": kwargs["content"],
                    "embedding": kwargs["embedding"],
                })
                chunk_added = True

            event_added = False
            event = kwargs.get("event")
            event_key = kwargs["event_key"]
            if event and all(stored["event_key"] != event_key for stored in events):
                events.append({"event_key": event_key, **event})
                event_added = True
            return {"chunk_added": chunk_added, "event_added": event_added}

        with patch.object(browser_connector, "HISTORY_PATHS", self.paths), \
                patch.object(
                    browser_connector,
                    "BROWSER_STATE_PATH",
                    Path(self.temp_dir.name) / "browser-state.json",
                ), \
                patch.object(browser_connector, "init_db"), \
                patch.object(
                    browser_connector, "get_embedding", return_value=[0.1, 0.2]
                ) as get_embedding, \
                patch.object(browser_connector, "file_already_indexed", side_effect=already_indexed), \
                patch.object(
                    browser_connector,
                    "upsert_browser_record",
                    side_effect=upsert_browser_record,
                ):
            first = browser_connector.sync_browser_history()
            second = browser_connector.sync_browser_history()

        for key in ("browsers_found", "entries_added", "skipped_sensitive", "skipped_already_indexed"):
            self.assertIn(key, first)
        self.assertEqual(first["browsers_found"], ["chrome", "edge"])
        self.assertEqual(first["profiles_found"][1]["profile_name"], "Work")
        self.assertEqual(first["records_found_by_type"], {"history": 6, "bookmark": 1, "download": 1})
        self.assertEqual(first["skipped_sensitive"], 1)
        self.assertEqual(first["entries_added"], 7)
        self.assertEqual(second["entries_added"], 0)
        self.assertEqual(second["skipped_already_indexed"], 7)
        self.assertEqual(len(chunks), 6)
        self.assertEqual(get_embedding.call_count, 6)
        self.assertEqual(len(events), 7)
        google_events = [
            event for event in events
            if event["path_or_url"] == "https://www.google.com/search?q=local+ai"
        ]
        self.assertEqual(len(google_events), 2)
        self.assertEqual(len({event["event_key"] for event in google_events}), 2)
        self.assertFalse(first["live_tabs"]["available"])
        self.assertEqual(first["data_scope"], "historical_browser_data")


if __name__ == "__main__":
    unittest.main()
