"""Focused coverage for source-based Timeline category filtering."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import app as app_module
import vectorstore


class TimelineStorageFilteringTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_patch = patch.object(
            vectorstore, "DB_PATH", Path(self.temp_dir.name) / "timeline.db"
        )
        self.db_patch.start()
        vectorstore.init_db()

        event_types = [
            "app_focus",
            "file_indexed",
            "file_deleted",
            "browser_visit",
            "browser_download",
            "system_status",
            "filesystem_probe",
            "browserish",
        ]
        for position, event_type in enumerate(event_types, start=1):
            vectorstore.add_event(
                event_type=event_type,
                title=event_type,
                subtitle="fixture",
                app_label="test",
                badge_type="txt",
                path_or_url=f"fixture:{event_type}",
                timestamp=f"2026-07-13T09:{position:02d}:00+05:30",
            )

    def tearDown(self):
        self.db_patch.stop()
        self.temp_dir.cleanup()

    def _types(self, **kwargs):
        return [row["event_type"] for row in vectorstore.get_timeline(**kwargs)]

    def test_each_kind_maps_to_its_real_event_source(self):
        self.assertEqual(self._types(kind="apps"), ["app_focus"])
        self.assertEqual(
            self._types(event_kind="documents"), ["file_deleted", "file_indexed"]
        )
        self.assertEqual(
            self._types(kind="web"), ["browser_download", "browser_visit"]
        )
        self.assertEqual(
            self._types(kind="system"),
            ["browserish", "filesystem_probe", "system_status"],
        )

    def test_all_is_default_and_aliases_must_not_conflict(self):
        self.assertEqual(len(vectorstore.get_timeline()), 8)
        self.assertEqual(len(vectorstore.get_timeline(kind=" ALL ")), 8)
        with self.assertRaises(ValueError):
            vectorstore.get_timeline(kind="apps", event_kind="web")
        with self.assertRaises(ValueError):
            vectorstore.get_timeline(kind="titles-containing-files")

    def test_date_filter_keeps_chronological_order_and_limit_is_bounded(self):
        rows = vectorstore.get_timeline(
            date_prefix="2026-07-13", kind="documents", limit=0
        )
        self.assertEqual([row["event_type"] for row in rows], ["file_indexed"])


class TimelineApiFilteringTests(unittest.TestCase):
    def setUp(self):
        self.client = app_module.app.test_client()

    def test_kind_alias_is_normalized_and_limit_is_capped(self):
        with patch.object(app_module, "get_timeline", return_value=[]) as timeline:
            response = self.client.get("/api/timeline?kind=APPS&limit=999999")

        self.assertEqual(response.status_code, 200)
        timeline.assert_called_once_with(
            date_prefix=None, limit=5000, event_kind="apps"
        )

    def test_event_kind_alias_is_supported(self):
        with patch.object(app_module, "get_timeline", return_value=[]) as timeline:
            response = self.client.get(
                "/api/timeline?event_kind=documents&date=2026-07-13&limit=-9"
            )

        self.assertEqual(response.status_code, 200)
        timeline.assert_called_once_with(
            date_prefix="2026-07-13", limit=1, event_kind="documents"
        )

    def test_unknown_or_conflicting_kind_is_rejected_before_database_access(self):
        for query in (
            "kind=unknown",
            "kind=apps&event_kind=web",
        ):
            with self.subTest(query=query), patch.object(
                app_module, "get_timeline"
            ) as timeline:
                response = self.client.get(f"/api/timeline?{query}")

            self.assertEqual(response.status_code, 400)
            self.assertEqual(
                response.get_json()["allowed_kinds"],
                ["all", "apps", "documents", "system", "web"],
            )
            timeline.assert_not_called()


if __name__ == "__main__":
    unittest.main()
