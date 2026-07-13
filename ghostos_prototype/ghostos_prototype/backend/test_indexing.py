"""Offline tests for GhostOS's bounded, resumable indexing pipeline."""

import os
import sqlite3
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import indexer
import vectorstore


class IndexingPipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.scan_root = self.root / "files"
        self.scan_root.mkdir()
        self.db_patch = patch.object(vectorstore, "DB_PATH", self.root / "test_memory.db")
        self.db_patch.start()
        vectorstore.init_db()

    def tearDown(self):
        self.db_patch.stop()
        self.temp_dir.cleanup()

    def _chunk_count(self, path: Path) -> int:
        conn = sqlite3.connect(vectorstore.DB_PATH)
        count = conn.execute(
            "SELECT COUNT(*) FROM chunks WHERE source_path = ?", (str(path),)
        ).fetchone()[0]
        conn.close()
        return count

    def test_chunk_limit_and_unchanged_file_skip_reembedding(self):
        path = self.scan_root / "large-notes.txt"
        path.write_text(" ".join(f"word{i}" for i in range(2000)), encoding="utf-8")

        with patch.object(indexer, "get_embedding", return_value=[0.1, 0.2]) as embed:
            first = indexer.process_file_detailed(path, max_chunks=2)
            second = indexer.process_file_detailed(path, max_chunks=2)

        self.assertEqual(first.status, indexer.FileProcessResult.PROCESSED)
        self.assertEqual(first.chunks_added, 2)
        self.assertEqual(second.status, indexer.FileProcessResult.ALREADY_INDEXED)
        self.assertEqual(embed.call_count, 2)
        self.assertEqual(self._chunk_count(path), 2)

    def test_embedding_model_change_reembeds_unchanged_file_once(self):
        path = self.scan_root / "model-aware.txt"
        path.write_text("remember this unchanged local note", encoding="utf-8")

        with (
            patch.object(indexer, "EMBED_MODEL", "embed-model-a"),
            patch.object(indexer, "get_embedding", return_value=[0.1]) as first_embed,
        ):
            first = indexer.process_file_detailed(path)
            unchanged = indexer.process_file_detailed(path)

        with (
            patch.object(indexer, "EMBED_MODEL", "embed-model-b"),
            patch.object(indexer, "get_embedding", return_value=[0.9]) as second_embed,
        ):
            refreshed = indexer.process_file_detailed(path)
            refreshed_unchanged = indexer.process_file_detailed(path)

        self.assertEqual(first.status, indexer.FileProcessResult.PROCESSED)
        self.assertEqual(unchanged.status, indexer.FileProcessResult.ALREADY_INDEXED)
        self.assertEqual(refreshed.status, indexer.FileProcessResult.PROCESSED)
        self.assertEqual(
            refreshed_unchanged.status, indexer.FileProcessResult.ALREADY_INDEXED
        )
        first_embed.assert_called_once()
        second_embed.assert_called_once()
        self.assertEqual(self._chunk_count(path), 1)
        self.assertEqual(
            vectorstore.get_file_index_state(str(path))["embedding_model"],
            "embed-model-b",
        )

    def test_legacy_database_migrates_embedding_model_as_stale(self):
        legacy_db = self.root / "legacy-memory.db"
        connection = sqlite3.connect(legacy_db)
        connection.execute(
            """CREATE TABLE files (
                   file_hash TEXT PRIMARY KEY,
                   path TEXT,
                   name TEXT,
                   extension TEXT,
                   category TEXT,
                   collection TEXT,
                   size_bytes INTEGER,
                   modified_at TEXT,
                   indexed_at TEXT DEFAULT CURRENT_TIMESTAMP,
                   embedded INTEGER DEFAULT 0
               )"""
        )
        connection.execute(
            """INSERT INTO files
                   (file_hash, path, name, extension, category, collection,
                    size_bytes, modified_at, embedded)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                "legacy-hash", "C:/legacy/note.txt", "note.txt", ".txt",
                "Documents", "Projects", 12, "2026-07-13T09:00:00", 1,
            ),
        )
        connection.commit()
        connection.close()

        with patch.object(vectorstore, "DB_PATH", legacy_db):
            vectorstore.init_db()
            state = vectorstore.get_file_index_state("C:/legacy/note.txt")
            connection = sqlite3.connect(legacy_db)
            columns = {
                row[1] for row in connection.execute("PRAGMA table_info(files)")
            }
            connection.close()

        self.assertIn("embedding_model", columns)
        self.assertIsNone(state["embedding_model"])

    def test_browser_model_refresh_replaces_old_page_vector(self):
        common = {
            "event_key": "visit-key",
            "source_path": "https://example.com/local-ai",
            "source_type": "browser_history_chrome",
            "content": "Local AI browser page",
            "event": None,
        }
        vectorstore.upsert_browser_record(
            search_hash="old-model-hash", embedding=[0.1], **common
        )
        vectorstore.upsert_browser_record(
            search_hash="new-model-hash", embedding=[0.9], **common
        )

        connection = sqlite3.connect(vectorstore.DB_PATH)
        rows = connection.execute(
            """SELECT file_hash, embedding FROM chunks
               WHERE source_path = ? AND source_type = ?""",
            (common["source_path"], common["source_type"]),
        ).fetchall()
        connection.close()

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0][0], "new-model-hash")

    def test_changed_file_atomically_replaces_old_chunks(self):
        path = self.scan_root / "changing.txt"
        path.write_text("old " * 500, encoding="utf-8")
        with patch.object(indexer, "get_embedding", return_value=[1.0]):
            first = indexer.process_file_detailed(path, max_chunks=2)

        old_mtime = path.stat().st_mtime_ns
        path.write_text("new content " * 20, encoding="utf-8")
        os.utime(path, ns=(old_mtime + 2_000_000_000, old_mtime + 2_000_000_000))
        with patch.object(indexer, "get_embedding", return_value=[2.0]):
            second = indexer.process_file_detailed(path, max_chunks=2)

        self.assertEqual(first.chunks_added, 2)
        self.assertEqual(second.status, indexer.FileProcessResult.PROCESSED)
        self.assertEqual(second.chunks_added, 1)
        self.assertEqual(self._chunk_count(path), 1)
        conn = sqlite3.connect(vectorstore.DB_PATH)
        content = conn.execute(
            "SELECT content FROM chunks WHERE source_path = ?", (str(path),)
        ).fetchone()[0]
        file_rows = conn.execute(
            "SELECT COUNT(*) FROM files WHERE path = ?", (str(path),)
        ).fetchone()[0]
        conn.close()
        self.assertIn("new content", content)
        self.assertEqual(file_rows, 1)

    def test_size_limit_catalogues_without_hashing_or_embedding(self):
        path = self.scan_root / "oversized.txt"
        path.write_text("x" * 100, encoding="utf-8")
        with (
            patch.object(indexer, "hash_file", side_effect=AssertionError("must not hash")),
            patch.object(indexer, "get_embedding", side_effect=AssertionError("must not embed")),
        ):
            outcome = indexer.process_file_detailed(path, max_file_bytes=10)

        self.assertEqual(outcome.status, indexer.FileProcessResult.TOO_LARGE)
        state = vectorstore.get_file_index_state(str(path))
        self.assertEqual(state["index_status"], "skipped_too_large")
        self.assertFalse(state["embedded"])

    def test_duplicate_content_is_not_embedded_twice(self):
        first_path = self.scan_root / "copy-a.txt"
        second_path = self.scan_root / "copy-b.txt"
        first_path.write_text("same local content", encoding="utf-8")
        second_path.write_text("same local content", encoding="utf-8")

        with patch.object(indexer, "get_embedding", return_value=[0.5]) as embed:
            first = indexer.process_file_detailed(first_path)
            second = indexer.process_file_detailed(second_path)

        self.assertEqual(first.status, indexer.FileProcessResult.PROCESSED)
        self.assertEqual(second.status, indexer.FileProcessResult.DUPLICATE_CONTENT)
        self.assertEqual(second.duplicate_of, str(first_path))
        self.assertEqual(embed.call_count, 1)
        self.assertEqual(
            vectorstore.get_file_index_state(str(second_path))["index_status"],
            "duplicate_content",
        )

    def test_failed_file_report_and_retryable_database_state(self):
        path = self.scan_root / "failure.txt"
        path.write_text("text that requires an embedding", encoding="utf-8")

        with patch.object(indexer, "get_embedding", side_effect=RuntimeError("Ollama offline")):
            summary = indexer.index_folders([str(self.scan_root)], file_workers=2)

        self.assertEqual(summary["files_failed"], 1)
        self.assertEqual(summary["failed_files"][0]["path"], str(path))
        self.assertEqual(summary["failed_files"][0]["stage"], "embedding")
        self.assertIn("Ollama offline", summary["failed_files"][0]["error"])
        self.assertEqual(
            vectorstore.get_file_index_state(str(path))["index_status"], "failed"
        )

        with patch.object(indexer, "get_embedding", return_value=[0.9]) as embed:
            retry = indexer.process_file_detailed(path)
        self.assertEqual(retry.status, indexer.FileProcessResult.PROCESSED)
        self.assertEqual(embed.call_count, 1)

    def test_unchanged_empty_scanned_pdf_retries_after_ocr_is_enabled(self):
        path = self.scan_root / "scanned-receipt.pdf"
        path.write_bytes(b"offline scanned pdf fixture")
        ocr_enabled = [False]

        with (
            patch.object(
                indexer,
                "get_settings",
                side_effect=lambda: {"ocr_enabled": ocr_enabled[0]},
            ),
            patch.object(indexer, "_extract_text_checked", return_value=""),
            patch.object(
                indexer,
                "extract_ocr_text",
                return_value="Receipt from Bengaluru for 420 rupees",
            ) as extract_ocr,
            patch.object(indexer, "get_embedding", return_value=[0.4, 0.2]) as embed,
        ):
            first = indexer.process_file_detailed(path)
            ocr_enabled[0] = True
            second = indexer.process_file_detailed(path)

        self.assertEqual(first.status, indexer.FileProcessResult.EMPTY)
        self.assertEqual(second.status, indexer.FileProcessResult.PROCESSED)
        extract_ocr.assert_called_once_with(path)
        embed.assert_called_once()
        state = vectorstore.get_file_index_state(str(path))
        self.assertEqual(state["index_status"], "embedded")
        self.assertTrue(state["embedded"])

        conn = sqlite3.connect(vectorstore.DB_PATH)
        row = conn.execute(
            "SELECT source_type, content FROM chunks WHERE source_path = ?",
            (str(path),),
        ).fetchone()
        conn.close()
        self.assertEqual(row[0], "ocr")
        self.assertIn("Bengaluru", row[1])

    def test_bounded_embedding_workers_progress_and_cancellation(self):
        for number in range(4):
            (self.scan_root / f"file-{number}.txt").write_text(
                f"unique-{number} " + "content " * 500, encoding="utf-8"
            )

        lock = threading.Lock()
        active = 0
        max_active = 0

        def fake_embedding(_text):
            nonlocal active, max_active
            with lock:
                active += 1
                max_active = max(max_active, active)
            time.sleep(0.02)
            with lock:
                active -= 1
            return [1.0]

        started: list[str] = []
        completed: list[str] = []
        with patch.object(indexer, "get_embedding", side_effect=fake_embedding):
            summary = indexer.index_folders(
                [str(self.scan_root)],
                file_workers=4,
                embedding_workers=1,
                max_pending=4,
                max_chunks=1,
                on_file_start=lambda path: started.append(str(path)),
                on_file_done=lambda path, _outcome: completed.append(str(path)),
            )

        self.assertEqual(max_active, 1)
        self.assertEqual(summary["files_embedded"], 4)
        self.assertEqual(len(started), 4)
        self.assertEqual(len(completed), 4)

        cancelled = threading.Event()
        cancelled.set()
        cancelled_summary = indexer.index_folders(
            [str(self.scan_root)], cancel_event=cancelled
        )
        self.assertTrue(cancelled_summary["cancelled"])
        self.assertEqual(cancelled_summary["files_completed"], 0)


if __name__ == "__main__":
    unittest.main()
