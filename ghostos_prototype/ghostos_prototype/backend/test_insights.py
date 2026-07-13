import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import insights


class InsightTests(unittest.TestCase):
    def test_uses_local_evidence_and_respects_dismissal(self):
        files = [{"name": "report.pdf", "path": "C:/Users/test/report.pdf", "category": "PDFs", "modified_at": "2026-07-13T09:00:00"}]
        with tempfile.TemporaryDirectory() as tmp, patch.object(insights, "STATE_PATH", Path(tmp) / "state.json"), patch.object(insights, "get_recent_files", return_value=files), patch.object(insights, "get_timeline", return_value=[]):
            first = insights.build_insights()
            self.assertEqual(first[0]["target"], files[0]["path"])
            insights.dismiss_insight(first[0]["id"])
            self.assertEqual(insights.build_insights(), [])


if __name__ == "__main__":
    unittest.main()
