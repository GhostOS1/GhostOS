import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import app as app_module
import settings_store


class SettingsTests(unittest.TestCase):
    def test_rejects_remote_ollama(self):
        with self.assertRaises(ValueError):
            settings_store.update_settings({"ollama_url": "https://example.com"})

    def test_persists_valid_local_settings(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(settings_store, "SETTINGS_PATH", Path(tmp) / "settings.json"):
            saved = settings_store.update_settings({"ocr_enabled": True, "chat_model": "local:test"})
            self.assertTrue(saved["ocr_enabled"])
            self.assertEqual(settings_store.get_settings()["chat_model"], "local:test")

    def test_invalid_json_falls_back_to_fresh_defaults(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "settings.json"
            path.write_text("{broken", encoding="utf-8")
            with patch.object(settings_store, "SETTINGS_PATH", path):
                loaded = settings_store.get_settings()

        self.assertEqual(loaded, settings_store.DEFAULT_SETTINGS)
        self.assertIsNot(loaded, settings_store.DEFAULT_SETTINGS)

    def test_partial_permissions_merge_without_dropping_other_defaults(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(
            settings_store, "SETTINGS_PATH", Path(tmp) / "settings.json"
        ):
            saved = settings_store.update_settings({"action_permissions": {"open_url": False}})
            reloaded = settings_store.get_settings()

        self.assertFalse(saved["action_permissions"]["open_url"])
        self.assertTrue(reloaded["action_permissions"]["open_file"])
        self.assertTrue(reloaded["action_permissions"]["create_folder"])

    def test_permissions_reject_string_booleans_and_unknown_actions(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(
            settings_store, "SETTINGS_PATH", Path(tmp) / "settings.json"
        ):
            with self.assertRaisesRegex(ValueError, "true or false"):
                settings_store.update_settings(
                    {"action_permissions": {"open_url": "false"}}
                )
            with self.assertRaisesRegex(ValueError, "Unknown action permissions"):
                settings_store.update_settings(
                    {"action_permissions": {"run_shell": True}}
                )

    def test_settings_api_round_trip_uses_local_file(self):
        client = app_module.app.test_client()
        with tempfile.TemporaryDirectory() as tmp, patch.object(
            settings_store, "SETTINGS_PATH", Path(tmp) / "settings.json"
        ):
            put_response = client.put(
                "/api/settings",
                json={"ocr_enabled": True, "browser_history_enabled": False},
            )
            get_response = client.get("/api/settings")

        self.assertEqual(put_response.status_code, 200)
        self.assertTrue(put_response.get_json()["restart_required"])
        self.assertTrue(get_response.get_json()["settings"]["ocr_enabled"])
        self.assertFalse(get_response.get_json()["settings"]["browser_history_enabled"])

    def test_settings_api_rejects_unknown_keys_without_writing(self):
        client = app_module.app.test_client()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "settings.json"
            with patch.object(settings_store, "SETTINGS_PATH", path):
                response = client.put("/api/settings", json={"cloud_sync": True})

            self.assertEqual(response.status_code, 400)
            self.assertFalse(path.exists())


if __name__ == "__main__":
    unittest.main()
