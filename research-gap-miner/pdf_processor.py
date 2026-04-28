"""Download and extract text from PDFs."""

from __future__ import annotations

import fitz
import requests


def extract_text_from_bytes(pdf_bytes: bytes, max_pages: int | None = None) -> str:
    """Extract text from in-memory PDF bytes."""

    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        page_limit = len(doc) if max_pages is None else min(max_pages, len(doc))
        pages = [doc[page_index].get_text() for page_index in range(page_limit)]

    return "\n".join(pages).strip()


def extract_text_from_url(pdf_url: str, max_pages: int | None = None) -> str:
    """Download a PDF URL and extract text with PyMuPDF."""

    response = requests.get(pdf_url, timeout=60)
    response.raise_for_status()
    return extract_text_from_bytes(response.content, max_pages=max_pages)
