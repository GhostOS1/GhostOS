"""Local API boundary and attachment safety regressions."""

import io
import unittest
from copy import deepcopy
from unittest.mock import patch

import app as app_module
import memory_agent
import settings_store


class ApiSecurityTests(unittest.TestCase):
    def setUp(self):
        self.client = app_module.app.test_client()

    def tearDown(self):
        memory_agent.reset()

    def test_cross_site_mutation_is_rejected_before_handler(self):
        with patch.object(app_module, "update_settings") as update:
            response = self.client.put(
                "/api/settings",
                json={"ocr_enabled": True},
                headers={
                    "Origin": "https://attacker.invalid",
                    "Sec-Fetch-Site": "cross-site",
                },
            )

        self.assertEqual(response.status_code, 403)
        update.assert_not_called()

    def test_same_origin_mutation_is_allowed(self):
        settings = deepcopy(settings_store.DEFAULT_SETTINGS)
        with patch.object(app_module, "update_settings", return_value=settings) as update:
            response = self.client.put(
                "/api/settings",
                json={"ocr_enabled": True},
                headers={"Origin": "http://127.0.0.1:5000", "Sec-Fetch-Site": "same-origin"},
            )

        self.assertEqual(response.status_code, 200)
        update.assert_called_once_with({"ocr_enabled": True})

    def test_disabled_permission_blocks_legacy_and_registry_action_routes(self):
        settings = deepcopy(settings_store.DEFAULT_SETTINGS)
        settings["action_permissions"]["open_url"] = False
        with (
            patch.object(app_module, "get_settings", return_value=settings),
            patch.object(app_module, "execute_action") as execute,
        ):
            legacy = self.client.post(
                "/api/open", json={"target": "https://example.invalid/page"}
            )
            registry = self.client.post(
                "/api/actions/execute",
                json={"action": "open_url", "arguments": {"url": "https://example.invalid/page"}},
            )

        self.assertEqual(legacy.status_code, 403)
        self.assertEqual(registry.status_code, 403)
        self.assertEqual(legacy.get_json()["error"]["code"], "permission_disabled")
        execute.assert_not_called()

    def test_unsupported_binary_attachment_is_not_extracted(self):
        with (
            patch.object(app_module, "get_settings", return_value=deepcopy(settings_store.DEFAULT_SETTINGS)),
            patch.object(app_module, "extract_text") as extract,
            patch.object(app_module, "stream_reply", return_value=iter(["handled locally"])) as stream,
        ):
            response = self.client.post(
                "/api/chat",
                data={
                    "message": "What is attached?",
                    "attachment": (io.BytesIO(b"opaque-binary"), "recording.exe"),
                },
                content_type="multipart/form-data",
            )
            body = response.get_data(as_text=True)

        self.assertEqual(response.status_code, 200)
        self.assertIn("handled locally", body)
        extract.assert_not_called()
        prompt = stream.call_args.args[2]
        self.assertIn("format is not supported", prompt)
        self.assertIn("transcription is not implemented", prompt)

    def test_attachment_request_size_is_bounded(self):
        original = app_module.app.config["MAX_CONTENT_LENGTH"]
        app_module.app.config["MAX_CONTENT_LENGTH"] = 128
        try:
            response = self.client.post(
                "/api/chat",
                data={
                    "message": "large attachment",
                    "attachment": (io.BytesIO(b"x" * 1024), "large.txt"),
                },
                content_type="multipart/form-data",
            )
        finally:
            app_module.app.config["MAX_CONTENT_LENGTH"] = original

        self.assertEqual(response.status_code, 413)


if __name__ == "__main__":
    unittest.main()
