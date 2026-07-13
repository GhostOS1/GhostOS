"""Offline integration coverage for OCR, local storage, and destructive API guards.

Every dependency that could contact Ollama, scan the real computer, or launch a
file is replaced with a deterministic local stub.  The storage tests use a
temporary SQLite database and temporary fixture files only.
"""

import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import app as app_module
import indexer
import vectorstore


class TemporaryVectorStoreTestCase(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.db_patch = patch.object(vectorstore, "DB_PATH", self.root / "memory.db")
        self.db_patch.start()
        vectorstore.init_db()

    def tearDown(self):
        self.db_patch.stop()
        self.temp_dir.cleanup()

    def _table_count(self, table: str) -> int:
        connection = sqlite3.connect(vectorstore.DB_PATH)
        try:
            return connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        finally:
            connection.close()


class OcrIndexingIntegrationTests(TemporaryVectorStoreTestCase):
    def _image_fixture(self, name: str = "receipt.png") -> Path:
        path = self.root / name
        # The OCR adapter is mocked, so a tiny opaque fixture is enough and
        # keeps the test independent of Pillow/Tesseract availability.
        path.write_bytes(b"offline-image-fixture")
        return path

    def test_enabled_image_ocr_is_embedded_with_ocr_source_type(self):
        path = self._image_fixture()
        with (
            patch.object(indexer, "get_settings", return_value={"ocr_enabled": True}),
            patch.object(indexer, "get_ocr_status", return_value={"available": True}),
            patch.object(indexer, "extract_ocr_text", return_value="Taxi receipt total 420 rupees") as extract,
            patch.object(indexer, "get_embedding", return_value=[0.25, 0.75]) as embed,
        ):
            outcome = indexer.process_file_detailed(path)

        self.assertEqual(outcome.status, indexer.FileProcessResult.PROCESSED)
        self.assertEqual(outcome.chunks_added, 1)
        extract.assert_called_once_with(path)
        embed.assert_called_once()

        connection = sqlite3.connect(vectorstore.DB_PATH)
        try:
            row = connection.execute(
                "SELECT source_type, content FROM chunks WHERE source_path = ?",
                (str(path),),
            ).fetchone()
        finally:
            connection.close()
        self.assertEqual(row[0], "ocr")
        self.assertIn("420 rupees", row[1])

    def test_disabled_image_is_catalogued_without_ocr_or_embedding(self):
        path = self._image_fixture("screenshot.jpg")
        with (
            patch.object(indexer, "get_settings", return_value={"ocr_enabled": False}),
            patch.object(indexer, "extract_ocr_text") as extract,
            patch.object(indexer, "get_embedding") as embed,
        ):
            outcome = indexer.process_file_detailed(path)

        self.assertEqual(outcome.status, indexer.FileProcessResult.UNSUPPORTED)
        self.assertEqual(outcome.stage, "catalogue")
        extract.assert_not_called()
        embed.assert_not_called()
        state = vectorstore.get_file_index_state(str(path))
        self.assertEqual(state["index_status"], "catalogued")
        self.assertFalse(state["embedded"])
        self.assertEqual(self._table_count("chunks"), 0)

    def test_missing_local_ocr_engine_is_reported_in_file_status(self):
        path = self._image_fixture("scan.tiff")
        with (
            patch.object(indexer, "get_settings", return_value={"ocr_enabled": True}),
            patch.object(indexer, "get_ocr_status", return_value={"available": False}),
            patch.object(indexer, "extract_ocr_text") as extract,
            patch.object(indexer, "get_embedding") as embed,
        ):
            outcome = indexer.process_file_detailed(path)

        self.assertEqual(outcome.status, indexer.FileProcessResult.UNSUPPORTED)
        self.assertEqual(outcome.stage, "ocr")
        self.assertIn("unavailable", outcome.error.lower())
        extract.assert_not_called()
        embed.assert_not_called()
        state = vectorstore.get_file_index_state(str(path))
        self.assertEqual(state["index_status"], "ocr_unavailable")
        self.assertIn("Tesseract", state["index_error"])


class VectorStoreStatusAndClearTests(TemporaryVectorStoreTestCase):
    def _insert_embedded_file(self, path: str = "C:/offline/report.txt") -> None:
        vectorstore.replace_file_index(
            file_hash="embedded-hash",
            path=path,
            name=Path(path).name,
            extension=".txt",
            category="Documents",
            collection="Projects",
            size_bytes=42,
            modified_at="2026-07-13T09:00:00",
            mtime_ns=123,
            embedded_chunks=[("offline report text", [1.0, 0.0])],
            index_status="embedded",
            source_type="text",
        )

    def _insert_failed_file(self) -> None:
        vectorstore.upsert_file(
            file_hash="failed-hash",
            path="C:/offline/broken.pdf",
            name="broken.pdf",
            extension=".pdf",
            category="PDFs",
            collection="Projects",
            size_bytes=99,
            modified_at="2026-07-13T10:00:00",
            embedded=False,
            mtime_ns=456,
            index_status="failed",
            index_error="offline extractor failure",
            chunks_count=0,
        )

    def test_recent_files_exposes_embedding_and_failure_status(self):
        self._insert_embedded_file()
        self._insert_failed_file()

        files = vectorstore.get_recent_files(limit=10)
        by_name = {item["name"]: item for item in files}
        self.assertTrue(by_name["report.txt"]["embedded"])
        self.assertEqual(by_name["report.txt"]["index_status"], "embedded")
        self.assertEqual(by_name["report.txt"]["chunks_count"], 1)
        self.assertIsNone(by_name["report.txt"]["index_error"])
        self.assertFalse(by_name["broken.pdf"]["embedded"])
        self.assertEqual(by_name["broken.pdf"]["index_status"], "failed")
        self.assertEqual(by_name["broken.pdf"]["chunks_count"], 0)
        self.assertEqual(by_name["broken.pdf"]["index_error"], "offline extractor failure")

    def test_clear_file_index_preserves_activity_events(self):
        self._insert_embedded_file()
        self._insert_failed_file()
        vectorstore.add_event(
            "browser_visit", "Local page", "offline", "Browser", "web",
            "https://example.invalid", "2026-07-13T10:30:00",
        )

        removed = vectorstore.clear_file_index()

        self.assertEqual(removed, {"files": 2, "chunks": 1, "events": 0})
        self.assertEqual(self._table_count("files"), 0)
        self.assertEqual(self._table_count("chunks"), 0)
        self.assertEqual(self._table_count("events"), 1)

    def test_clear_all_local_data_removes_files_chunks_and_events(self):
        self._insert_embedded_file()
        vectorstore.add_event(
            "file_indexed", "Indexed report.txt", "offline", "GhostOS", "txt",
            "C:/offline/report.txt", "2026-07-13T09:00:00",
        )

        removed = vectorstore.clear_all_local_data()

        self.assertEqual(removed, {"files": 1, "chunks": 1, "events": 1})
        self.assertEqual(self._table_count("files"), 0)
        self.assertEqual(self._table_count("chunks"), 0)
        self.assertEqual(self._table_count("events"), 0)

    def test_removing_deleted_source_preserves_historical_timeline_event(self):
        path = "C:/offline/report.txt"
        self._insert_embedded_file(path)
        vectorstore.add_event(
            "file_indexed", "Indexed report.txt", "offline", "GhostOS", "txt",
            path, "2026-07-13T09:00:00+05:30",
        )

        vectorstore.remove_source(path)

        self.assertEqual(self._table_count("files"), 0)
        self.assertEqual(self._table_count("chunks"), 0)
        self.assertEqual(self._table_count("events"), 1)


class LocalDataApiGuardTests(unittest.TestCase):
    def setUp(self):
        self.client = app_module.app.test_client()

    def test_clear_requires_exact_confirmation_without_mutating_data(self):
        with patch.object(app_module, "clear_all_local_data") as clear:
            response = self.client.post("/api/data/clear", json={"confirm": "yes"})
        self.assertEqual(response.status_code, 400)
        clear.assert_not_called()

    def test_clear_rejects_active_index_job_without_mutating_data(self):
        with (
            patch.object(app_module, "get_background_status", return_value={"active": True}),
            patch.object(app_module, "clear_all_local_data") as clear,
        ):
            response = self.client.post(
                "/api/data/clear", json={"confirm": "clear-local-data"}
            )
        self.assertEqual(response.status_code, 409)
        clear.assert_not_called()

    def test_confirmed_clear_resets_storage_and_conversation_memory(self):
        removed = {"files": 3, "chunks": 8, "events": 2}
        with (
            patch.object(app_module, "get_background_status", return_value={"active": False}),
            patch.object(app_module, "clear_all_local_data", return_value=removed) as clear,
            patch.object(app_module, "reset_background_status") as reset_status,
            patch.object(app_module, "reset_memory") as reset,
        ):
            response = self.client.post(
                "/api/data/clear", json={"confirm": "clear-local-data"}
            )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json(), {"status": "cleared", "removed": removed})
        clear.assert_called_once_with()
        reset_status.assert_called_once_with()
        reset.assert_called_once_with()

    def test_rebuild_requires_confirmation_and_rejects_active_job(self):
        with patch.object(app_module, "clear_file_index") as clear:
            missing = self.client.post("/api/index/rebuild", json={})
        self.assertEqual(missing.status_code, 400)
        clear.assert_not_called()

        with (
            patch.object(app_module, "get_background_status", return_value={"active": True}),
            patch.object(app_module, "clear_file_index") as clear,
            patch.object(app_module, "connect_system") as connect,
        ):
            active = self.client.post(
                "/api/index/rebuild", json={"confirm": "rebuild-index"}
            )
        self.assertEqual(active.status_code, 409)
        clear.assert_not_called()
        connect.assert_not_called()

    def test_confirmed_rebuild_uses_local_settings_and_starts_background_connect(self):
        settings = {
            "scan_entire_drives": False,
            "indexed_folders": ["C:/offline/Documents"],
            "excluded_folders": ["C:/offline/Documents/private"],
        }
        removed = {"files": 4, "chunks": 9, "events": 0}
        connected = {"connected": True, "background": {"active": True}}
        with (
            patch.object(app_module, "get_background_status", return_value={"active": False}),
            patch.object(app_module, "clear_file_index", return_value=removed) as clear,
            patch.object(app_module, "get_settings", return_value=settings),
            patch.object(app_module, "connect_system", return_value=connected) as connect,
        ):
            response = self.client.post(
                "/api/index/rebuild", json={"confirm": "rebuild-index"}
            )

        self.assertEqual(response.status_code, 202)
        self.assertEqual(response.get_json()["removed"], removed)
        self.assertEqual(response.get_json()["connect"], connected)
        clear.assert_called_once_with()
        connect.assert_called_once_with(
            scan_entire_drives=False,
            additional_folders=settings["indexed_folders"],
            excluded_folders=settings["excluded_folders"],
        )


if __name__ == "__main__":
    unittest.main()
