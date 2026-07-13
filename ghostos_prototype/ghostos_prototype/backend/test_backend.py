"""Fast, offline smoke tests for the GhostOS API."""

import unittest
from unittest.mock import patch

import app as app_module
from router import classify_intent


class GhostOSSmokeTests(unittest.TestCase):
    def setUp(self):
        self.client = app_module.app.test_client()

    def test_router(self):
        self.assertEqual(classify_intent("hello"), "greeting")
        self.assertEqual(classify_intent("what is my cpu usage"), "system_query")
        self.assertEqual(classify_intent("find report.pdf"), "exact_file_query")

    def test_stats(self):
        response = self.client.get("/api/stats")
        self.assertEqual(response.status_code, 200)
        self.assertIn("total_files", response.get_json())

    @patch.object(app_module, "get_diagnostics")
    def test_health_contract(self, diagnostics_mock):
        diagnostics_mock.return_value = {
            "status": "ready",
            "backend": {"available": True, "local_only": True},
            "ollama": {"available": True},
            "problems": [],
        }
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["status"], "ready")
        diagnostics_mock.assert_called_once_with()

    def test_chat_validation_and_fast_path(self):
        self.assertEqual(self.client.post("/api/chat", json={"message": ""}).status_code, 400)
        self.assertEqual(self.client.post("/api/chat", json={"message": "hello"}).status_code, 200)


if __name__ == "__main__":
    unittest.main()
