"""Offline tests for the health/diagnostics contract.

No test in this module contacts Ollama, scans browser profiles, or reads the
production GhostOS database.
"""

import unittest
from unittest.mock import MagicMock, patch

import requests

import app as app_module
import diagnostics


def _report(status: str) -> dict:
    return {
        "status": status,
        "backend": {"available": True, "local_only": True},
        "ollama": {"available": status == "ready"},
        "database": {"available": True, "total_files": 0},
        "watcher": {"active": False, "folders": []},
        "browser_connector": {"enabled": False, "detected_browsers": []},
        "activity_tracker": {"active": False},
        "ocr": {"available": False, "enabled": False},
        "problems": [] if status == "ready" else ["Ollama unavailable"],
        "commands": [],
    }


class HealthApiTests(unittest.TestCase):
    def setUp(self):
        self.client = app_module.app.test_client()

    def test_ready_health_returns_200_without_external_calls(self):
        with patch.object(app_module, "get_diagnostics", return_value=_report("ready")) as mocked:
            response = self.client.get("/api/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["status"], "ready")
        mocked.assert_called_once_with()

    def test_degraded_health_returns_503_without_external_calls(self):
        with patch.object(app_module, "get_diagnostics", return_value=_report("degraded")) as mocked:
            response = self.client.get("/api/health")

        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.get_json()["problems"], ["Ollama unavailable"])
        mocked.assert_called_once_with()

    def test_diagnostics_route_returns_same_snapshot(self):
        expected = _report("ready")
        with patch.object(app_module, "get_diagnostics", return_value=expected):
            response = self.client.get("/api/diagnostics")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json(), expected)


class DiagnosticsBuilderTests(unittest.TestCase):
    def _local_patches(self):
        response = MagicMock()
        response.raise_for_status.return_value = None
        response.json.return_value = {
            "models": [
                {"name": diagnostics.CHAT_MODEL},
                {"name": diagnostics.EMBED_MODEL},
            ]
        }
        return (
            patch.object(diagnostics.requests, "get", return_value=response),
            patch.object(diagnostics, "get_stats", return_value={"total_files": 12, "chunks": 25}),
            patch.object(
                diagnostics,
                "get_settings",
                return_value={
                    "ocr_enabled": True,
                    "browser_history_enabled": True,
                    "activity_tracking_enabled": True,
                },
            ),
            patch.object(diagnostics, "get_ocr_status", return_value={"available": True, "engine": "tesseract"}),
            patch.object(diagnostics.browser_connector, "HISTORY_PATHS", {}),
            patch.object(diagnostics.watcher, "_observer", None, create=True),
            patch.object(diagnostics.watcher, "_watched_folders", [], create=True),
            patch.object(diagnostics.watcher, "_browser_thread_started", False, create=True),
            patch.object(diagnostics.activity_tracker, "_started", False, create=True),
        )

    def test_ready_snapshot_is_built_entirely_offline(self):
        patches = self._local_patches()
        for active_patch in patches:
            active_patch.start()
        try:
            report = diagnostics.get_diagnostics()
        finally:
            for active_patch in reversed(patches):
                active_patch.stop()

        self.assertEqual(report["status"], "ready")
        self.assertTrue(report["backend"]["local_only"])
        self.assertTrue(report["ollama"]["chat_model"])
        self.assertTrue(report["ollama"]["embedding_model"])
        self.assertEqual(report["database"]["total_files"], 12)
        self.assertTrue(report["ocr"]["enabled"])
        self.assertEqual(report["problems"], [])

    def test_ollama_failure_is_reported_as_degraded(self):
        with (
            patch.object(diagnostics.requests, "get", side_effect=requests.ConnectionError("offline")),
            patch.object(diagnostics, "get_stats", return_value={"total_files": 0}),
            patch.object(
                diagnostics,
                "get_settings",
                return_value={
                    "ocr_enabled": False,
                    "browser_history_enabled": False,
                    "activity_tracking_enabled": False,
                },
            ),
            patch.object(diagnostics, "get_ocr_status", return_value={"available": False}),
            patch.object(diagnostics.browser_connector, "HISTORY_PATHS", {}),
            patch.object(diagnostics.watcher, "_observer", None, create=True),
            patch.object(diagnostics.watcher, "_watched_folders", [], create=True),
            patch.object(diagnostics.watcher, "_browser_thread_started", False, create=True),
            patch.object(diagnostics.activity_tracker, "_started", False, create=True),
        ):
            report = diagnostics.get_diagnostics()

        self.assertEqual(report["status"], "degraded")
        self.assertFalse(report["ollama"]["available"])
        self.assertIn("Ollama is unavailable", report["problems"][0])


if __name__ == "__main__":
    unittest.main()
