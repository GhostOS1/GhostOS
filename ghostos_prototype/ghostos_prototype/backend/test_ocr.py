import unittest
from unittest.mock import patch

import ocr_service


class OcrTests(unittest.TestCase):
    def test_missing_tesseract_is_optional(self):
        with patch("ocr_service.shutil.which", return_value=None):
            status = ocr_service.get_ocr_status()
        self.assertFalse(status["available"])


if __name__ == "__main__":
    unittest.main()
