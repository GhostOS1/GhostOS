import unittest

from timeline_sessions import group_events, summarize_day


class TimelineSessionTests(unittest.TestCase):
    def test_groups_by_inactivity_and_compresses_duplicates(self):
        events = [
            {"timestamp": "2026-07-13T09:00:00+05:30", "event_type": "app_focus", "title": "app.py", "app_label": "VS Code", "path_or_url": ""},
            {"timestamp": "2026-07-13T09:01:00+05:30", "event_type": "app_focus", "title": "app.py", "app_label": "VS Code", "path_or_url": ""},
            {"timestamp": "2026-07-13T10:00:00+05:30", "event_type": "browser_visit", "title": "Ollama docs", "app_label": "Chrome", "path_or_url": "https://example.test"},
        ]
        sessions = group_events(events)
        self.assertEqual(len(sessions), 2)
        self.assertEqual(sessions[0]["events"][0]["occurrences"], 2)
        self.assertEqual(sessions[0]["event_count"], 2)

    def test_empty_summary_is_honest(self):
        summary = summarize_day([], "2026-07-13")
        self.assertEqual(summary["event_count"], 0)
        self.assertIn("No locally recorded", summary["message"])

    def test_naive_and_aware_india_timestamps_share_the_same_session(self):
        events = [
            {
                # Legacy GhostOS rows stored Windows local time without an
                # offset.  On an India-local installation this is 09:00 IST.
                "timestamp": "2026-07-13T09:00:00",
                "event_type": "app_focus",
                "title": "GhostOS",
                "app_label": "VS Code",
                "path_or_url": "",
            },
            {
                "timestamp": "2026-07-13T09:05:00+05:30",
                "event_type": "browser_visit",
                "title": "Local documentation",
                "app_label": "Chrome",
                "path_or_url": "https://example.test/docs",
            },
            {
                # The same India-local morning expressed explicitly as UTC.
                "timestamp": "2026-07-13T03:40:00+00:00",
                "event_type": "file_indexed",
                "title": "notes.txt",
                "app_label": "GhostOS Indexer",
                "path_or_url": "C:/notes.txt",
            },
        ]

        sessions = group_events(events)

        self.assertEqual(len(sessions), 1)
        self.assertEqual(sessions[0]["event_count"], 3)
        self.assertEqual(sessions[0]["start"], "2026-07-13T09:00:00")
        self.assertEqual(sessions[0]["end"], "2026-07-13T03:40:00+00:00")


if __name__ == "__main__":
    unittest.main()
