"""Fast, offline smoke tests for the GhostOS API."""

import unittest

from app import app
from router import classify_intent


class GhostOSSmokeTests(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def test_router(self):
        self.assertEqual(classify_intent("hello"), "greeting")
        self.assertEqual(classify_intent("what is my cpu usage"), "system_query")
        self.assertEqual(classify_intent("find report.pdf"), "exact_file_query")

    def test_stats(self):
        response = self.client.get("/api/stats")
        self.assertEqual(response.status_code, 200)
        self.assertIn("total_files", response.get_json())

    def test_health_contract(self):
        response = self.client.get("/api/health")
        self.assertIn(response.status_code, (200, 503))
        self.assertIn(response.get_json()["status"], ("ready", "degraded"))

    def test_chat_validation_and_fast_path(self):
        self.assertEqual(self.client.post("/api/chat", json={"message": ""}).status_code, 400)
        self.assertEqual(self.client.post("/api/chat", json={"message": "hello"}).status_code, 200)


if __name__ == "__main__":
    unittest.main()

