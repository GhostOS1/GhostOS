"""Optional, offline OCR for images and scanned PDFs.

GhostOS never requires OCR to start. OCR runs only when enabled in local
settings and when Tesseract plus its Python binding are installed.
"""

import importlib.util
import shutil
from pathlib import Path

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
MAX_IMAGE_PIXELS = 25_000_000
MAX_PDF_PAGES = 10


def get_ocr_status() -> dict:
    executable = shutil.which("tesseract")
    binding = importlib.util.find_spec("pytesseract") is not None
    pillow = importlib.util.find_spec("PIL") is not None
    pdf_renderer = importlib.util.find_spec("fitz") is not None
    return {
        "available": bool(executable and binding and pillow),
        "executable": executable,
        "python_binding": binding,
        "image_support": bool(pillow),
        "scanned_pdf_support": bool(pdf_renderer and pillow),
    }


def extract_ocr_text(path: Path) -> str:
    status = get_ocr_status()
    if not status["available"]:
        return ""
    import pytesseract
    from PIL import Image

    path = Path(path)
    if path.suffix.casefold() in IMAGE_EXTENSIONS:
        with Image.open(path) as image:
            if image.width * image.height > MAX_IMAGE_PIXELS:
                return ""
            return pytesseract.image_to_string(image).strip()
    if path.suffix.casefold() == ".pdf" and status["scanned_pdf_support"]:
        import fitz
        texts = []
        with fitz.open(path) as document:
            for page in document[:MAX_PDF_PAGES]:
                pixmap = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5), alpha=False)
                if pixmap.width * pixmap.height > MAX_IMAGE_PIXELS:
                    continue
                image = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
                text = pytesseract.image_to_string(image).strip()
                if text:
                    texts.append(text)
        return "\n\n".join(texts)
    return ""

