"""Optional OCR hook. Only used when `cfg.ingest.ocr_enabled` and `ocr_available()`."""

from __future__ import annotations

from pathlib import Path


def ocr_available() -> bool:
    try:
        import pytesseract  # noqa: F401
    except ImportError:
        return False
    return True


def ocr_page(path: Path, page_number: int) -> str:
    """Best-effort OCR of a single PDF page (1-indexed). Caller decides whether to use it."""
    import pdfplumber
    import pytesseract  # local import: optional dependency

    with pdfplumber.open(path) as pdf:
        page = pdf.pages[page_number - 1]
        image = page.to_image(resolution=300).original
        return pytesseract.image_to_string(image)
