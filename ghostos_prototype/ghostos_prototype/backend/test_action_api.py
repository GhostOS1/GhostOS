import unittest
from unittest.mock import patch

import memory_agent
from app import app


class ActionApiTests(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def tearDown(self):
        memory_agent.reset()

    def test_unknown_action_is_rejected(self):
        response = self.client.post("/api/actions/execute", json={"action": "run_shell", "arguments": {}})
        self.assertIn(response.status_code, (400, 403))
        self.assertFalse(response.get_json()["success"])

    def test_open_it_uses_remembered_file_without_model(self):
        memory_agent._session_context["last_file"] = {"name": "report.pdf", "path": "C:/Users/test/report.pdf"}
        result = {"success": True, "action": "open_file", "target": "C:/Users/test/report.pdf", "message": "File opened successfully."}
        with patch("app.execute_action", return_value=result) as execute:
            response = self.client.post("/api/chat", json={"message": "open it"})
        self.assertEqual(response.status_code, 200)
        execute.assert_called_once_with("open_file", {"path": "C:/Users/test/report.pdf"})
        self.assertIn(b"File opened successfully", response.data)


if __name__ == "__main__":
    unittest.main()
