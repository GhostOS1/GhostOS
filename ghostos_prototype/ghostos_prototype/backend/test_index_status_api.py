"""Offline API tests for background indexing progress and cancellation."""

import unittest
from unittest.mock import patch

import app as app_module


class IndexStatusApiTests(unittest.TestCase):
    def setUp(self):
        self.client = app_module.app.test_client()

    def test_index_status_returns_an_isolated_progress_snapshot(self):
        snapshot = {
            "active": True,
            "done": False,
            "phase": "indexing",
            "current_folder": "Documents",
            "current_file": "report.pdf",
            "folders_total": 4,
            "folders_completed": 1,
            "files_processed": 17,
            "files_failed": 1,
            "failed_files": [{"path": "bad.pdf", "reason": "unreadable"}],
            "cancel_requested": False,
            "cancelled": False,
        }
        with patch.object(app_module, "get_background_status", return_value=snapshot) as mocked:
            response = self.client.get("/api/index-status")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json(), snapshot)
        mocked.assert_called_once_with()

    def test_cancel_requests_graceful_stop_for_active_job(self):
        expected = {"accepted": True, "message": "Indexing cancellation requested."}
        with patch.object(app_module, "cancel_background_indexing", return_value=expected) as mocked:
            response = self.client.post("/api/index-cancel")

        self.assertEqual(response.status_code, 202)
        self.assertEqual(response.get_json(), expected)
        mocked.assert_called_once_with()

    def test_cancel_reports_conflict_when_no_job_is_active(self):
        expected = {"accepted": False, "reason": "No indexing job is running."}
        with patch.object(app_module, "cancel_background_indexing", return_value=expected) as mocked:
            response = self.client.post("/api/index-cancel")

        self.assertEqual(response.status_code, 409)
        self.assertEqual(response.get_json(), expected)
        mocked.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
