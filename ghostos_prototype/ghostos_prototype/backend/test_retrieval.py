"""Offline tests for Phase 3 routing, retrieval and conversation memory.

All database and Ollama boundaries are mocked where they would otherwise
perform I/O, so this suite is safe to run on a fresh clone without models.
"""

import unittest
from datetime import datetime
from unittest.mock import patch

import files_agent
import memory_agent
import timeline_agent
from router import classify_intent
from vectorstore import rerank


def _file(path: str, name: str, modified: str = "2026-07-12T12:00:00") -> dict:
    return {
        "path": path,
        "name": name,
        "extension": "." + name.rsplit(".", 1)[-1] if "." in name else "",
        "category": "Documents",
        "size_bytes": 100,
        "modified_at": modified,
    }


class RouterTests(unittest.TestCase):
    def test_small_talk_is_case_insensitive_but_does_not_swallow_task(self):
        self.assertEqual(classify_intent("HELLO!!"), "greeting")
        self.assertEqual(classify_intent("hey find Report.PDF"), "exact_file_query")

    def test_structured_and_semantic_precedence(self):
        self.assertEqual(classify_intent("find Report.PDF"), "exact_file_query")
        self.assertEqual(classify_intent("show files in College folder"), "folder_query")
        self.assertEqual(classify_intent("find a document about lithium batteries"), "semantic_query")
        self.assertEqual(classify_intent("what did I do yesterday?"), "timeline_query")

    def test_reference_phrases_reach_memory_capable_path(self):
        self.assertEqual(classify_intent("open it"), "semantic_query")
        self.assertEqual(classify_intent("where is that file"), "semantic_query")
        self.assertEqual(classify_intent("the previous PDF"), "semantic_query")
        self.assertEqual(classify_intent("the page I visited earlier"), "semantic_query")

    def test_file_keyword_uses_word_boundaries(self):
        self.assertEqual(classify_intent("Explain Python profile semantics"), "general")

    def test_personal_context_recall_routes_to_retrieval(self):
        for message in (
            "What did I write about metrology?",
            "remind me what I said about the lithium pump",
            "the thing about embeddings",
            "what was that blog about local AI?",
        ):
            with self.subTest(message=message):
                self.assertEqual(classify_intent(message), "semantic_query")
        self.assertEqual(classify_intent("explain quantum computing"), "general")


class FileRetrievalTests(unittest.TestCase):
    def test_exact_result_ranks_before_partial_and_is_deduplicated(self):
        exact = _file(r"C:\Docs\Report.pdf", "Report.pdf")
        duplicate = dict(exact, path=r"c:/docs/report.pdf")
        partial = _file(r"C:\Docs\Report old.pdf", "Report old.pdf", "2026-07-13T12:00:00")

        with patch.object(files_agent, "search_files_keywords", return_value=[partial, exact, duplicate]), \
             patch.object(files_agent, "search_files_by_name", return_value=[partial, exact]):
            results = files_agent.file_agent("FIND report.PDF")

        self.assertEqual(results[0]["name"], "Report.pdf")
        self.assertEqual(results[0]["match_type"], "exact")
        self.assertEqual(sum(item["name"] == "Report.pdf" for item in results), 1)

    def test_typo_tolerant_name_match_rejects_unrelated_noise(self):
        intended = _file(r"C:\Docs\metrology.pdf", "metrology.pdf")
        noise = _file(r"C:\Docs\banana.txt", "banana.txt")
        with patch.object(files_agent, "search_files_keywords", return_value=[]), \
             patch.object(files_agent, "search_files_by_name", return_value=[noise, intended]):
            results = files_agent.file_agent("find metrolology")

        self.assertEqual([item["name"] for item in results], ["metrology.pdf"])
        self.assertEqual(results[0]["match_type"], "typo")

    def test_custom_folder_lookup_is_structured(self):
        expected = [_file(r"C:\Work\College\notes.txt", "notes.txt")]
        with patch.object(files_agent, "get_files_by_folder", return_value=expected) as lookup:
            results, folder = files_agent.folder_agent("show files in College folder")
        self.assertEqual(folder, "college")
        self.assertEqual(results, expected)
        lookup.assert_called_once_with("college")


class TimelineRetrievalTests(unittest.TestCase):
    def test_natural_dates_use_india_local_calendar(self):
        now = datetime(2026, 7, 13, 10, 30, tzinfo=timeline_agent.LOCAL_TIMEZONE)
        self.assertEqual(timeline_agent.resolve_date_prefixes("yesterday", now), ["2026-07-12"])
        self.assertEqual(
            timeline_agent.resolve_date_prefixes("last week", now),
            [f"2026-07-{day:02d}" for day in range(6, 13)],
        )
        self.assertEqual(timeline_agent.resolve_date_prefixes("on 11/07/2026", now), ["2026-07-11"])

    def test_date_filter_is_forwarded_and_duplicate_events_are_compressed(self):
        first = {
            "event_type": "browser_history_chrome", "title": "GhostOS docs",
            "subtitle": "", "app_label": "Chrome", "badge_type": "web",
            "path_or_url": "https://example.test/docs", "timestamp": "2026-07-12T10:00:00+05:30",
        }
        duplicate = dict(first, timestamp="2026-07-12T10:00:30+05:30")
        with patch.object(timeline_agent, "resolve_date_prefixes", return_value=["2026-07-12"]), \
             patch.object(timeline_agent, "get_timeline", return_value=[first, duplicate]) as get_events:
            results = timeline_agent.timeline_agent("what did I visit yesterday", limit=8)

        self.assertEqual(results, [duplicate])
        get_events.assert_called_once_with(date_prefix="2026-07-12", limit=5000)

    def test_date_and_remembered_context_rank_the_matching_older_event(self):
        noise = [
            {
                "event_type": "app_focus", "title": f"Window {index}",
                "subtitle": "", "app_label": "Editor", "badge_type": "app",
                "path_or_url": "", "timestamp": f"2026-07-12T18:{index % 60:02d}:00+05:30",
            }
            for index in range(120)
        ]
        relevant = {
            "event_type": "browser_visit", "title": "Metrology gauge calibration guide",
            "subtitle": "Precision measurement blog", "app_label": "Chrome", "badge_type": "web",
            "path_or_url": "https://example.test/metrology/calibration",
            "timestamp": "2026-07-12T09:15:00+05:30",
        }
        with patch.object(timeline_agent, "resolve_date_prefixes", return_value=["2026-07-12"]), \
             patch.object(timeline_agent, "get_timeline", return_value=noise + [relevant]):
            results = timeline_agent.timeline_agent(
                "which website yesterday was about metrolology calibration?", limit=5
            )

        self.assertEqual(results[0], relevant)
        self.assertEqual(len(results), 1)


class HybridRetrievalTests(unittest.TestCase):
    def test_reranker_rejects_rank_only_noise(self):
        candidates = [
            {
                "source_path": r"C:\Docs\random.txt", "source_type": "text",
                "content": "banana recipe and kitchen list", "vector_score": 0.35,
                "bm25_score": 0.0, "fused_score": 0.04,
            },
            {
                "source_path": r"C:\Docs\Lithium Report.pdf", "source_type": "pdf",
                "content": "quarterly lithium battery safety report", "vector_score": 0.65,
                "bm25_score": 4.0, "fused_score": 0.03,
            },
        ]
        results = rerank("lithium battery report", candidates, top_k=2)
        by_name = {item["source_path"]: item for item in results}
        self.assertGreater(by_name[r"C:\Docs\Lithium Report.pdf"]["score"], memory_agent.MIN_RERANK_SCORE)
        self.assertLess(by_name[r"C:\Docs\random.txt"]["score"], memory_agent.MIN_RERANK_SCORE)

    def test_embedding_failure_falls_back_to_bm25_and_deduplicates_source(self):
        relevant = {
            "source_path": r"C:\Docs\metrology.txt", "source_type": "text",
            "content": "metrology gauge calibration", "file_hash": "one", "bm25_score": 5.0,
        }
        duplicate = dict(relevant, content="metrology gauge calibration duplicate", bm25_score=4.0)
        with patch.object(memory_agent, "get_embedding", side_effect=RuntimeError("offline")), \
             patch.object(memory_agent, "bm25_search", return_value=[relevant, duplicate]) as bm25:
            results = memory_agent.search_agent("metrology gauge")

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["source_path"], relevant["source_path"])
        bm25.assert_called_once_with("metrology gauge", top_k=memory_agent.RETRIEVAL_POOL_SIZE)


class ConversationMemoryTests(unittest.TestCase):
    def setUp(self):
        memory_agent.reset()

    def tearDown(self):
        memory_agent.reset()

    def test_structured_entities_resolve_previous_pdf_folder_and_page(self):
        pdf = _file(r"C:\Docs\Design.pdf", "Design.pdf")
        browser = [{
            "source_path": "https://example.test/ghostos",
            "source_type": "browser_history_chrome",
            "content": "Visited: GhostOS Architecture\nURL: https://example.test/ghostos",
            "score": 0.9,
        }]
        memory_agent.update_session_from_turn(
            "semantic_query", [pdf], "Documents", [pdf], [], browser,
            "find the GhostOS design PDF",
        )
        snapshot = memory_agent.snapshot_session()
        context = memory_agent.build_reference_context(snapshot)

        self.assertEqual(snapshot["last_file"]["path"], pdf["path"])
        self.assertEqual(snapshot["last_pdf"]["name"], "Design.pdf")
        self.assertEqual(snapshot["last_folder_path"], r"C:\Docs")
        self.assertEqual(snapshot["last_browser_page"]["url"], browser[0]["source_path"])
        self.assertIn("previous/most recently discussed PDF", context)
        self.assertIn("GhostOS Architecture", context)

    def test_reference_phrase_detection(self):
        positives = (
            "open it", "that file", "the previous PDF", "the folder we discussed",
            "the page I visited earlier",
        )
        for phrase in positives:
            with self.subTest(phrase=phrase):
                self.assertTrue(memory_agent.wants_reference_resolution(phrase))
        self.assertFalse(memory_agent.wants_reference_resolution("find budget report"))

    def test_snapshot_is_isolated_and_reset_clears_entities(self):
        pdf = _file(r"C:\Docs\Design.pdf", "Design.pdf")
        memory_agent.update_session_from_turn("exact_file_query", [], None, [pdf], [], [], "find Design.pdf")
        snapshot = memory_agent.snapshot_session()
        snapshot["recent_files"].clear()
        self.assertTrue(memory_agent.snapshot_session()["recent_files"])
        memory_agent.reset()
        self.assertIsNone(memory_agent.snapshot_session()["last_file"])


if __name__ == "__main__":
    unittest.main()
