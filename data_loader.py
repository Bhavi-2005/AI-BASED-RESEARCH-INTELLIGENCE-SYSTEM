"""Document loading and chunking utilities for the RAG system."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Iterable

from pypdf import PdfReader


SUPPORTED_EXTENSIONS = {".pdf", ".txt"}


@dataclass(frozen=True)
class DocumentChunk:
    """A searchable section of a source document."""

    chunk_id: str
    source: str
    text: str
    chunk_index: int
    word_count: int


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def extract_text_from_pdf(file_obj: BinaryIO) -> str:
    """Extract text from a PDF file-like object."""

    reader = PdfReader(file_obj)
    pages = []

    for page in reader.pages:
        pages.append(page.extract_text() or "")

    return _normalize_whitespace("\n".join(pages))


def extract_text_from_txt(file_obj: BinaryIO) -> str:
    """Extract text from a UTF-8 text file-like object."""

    raw = file_obj.read()
    if isinstance(raw, bytes):
        text = raw.decode("utf-8", errors="ignore")
    else:
        text = raw

    return _normalize_whitespace(text)


def load_document(file_obj: BinaryIO, filename: str) -> str:
    """Load supported document types and return plain text."""

    extension = Path(filename).suffix.lower()

    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {extension}. Use PDF or TXT.")

    if extension == ".pdf":
        return extract_text_from_pdf(file_obj)

    return extract_text_from_txt(file_obj)


def split_text_into_chunks(
    text: str,
    source: str,
    chunk_size: int = 500,
    overlap: int = 50,
) -> list[DocumentChunk]:
    """Split text into overlapping word chunks.

    The default chunk size follows the project requirement: roughly 500 words
    with 50 words of overlap so related ideas are less likely to be separated.
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    if overlap < 0:
        raise ValueError("overlap cannot be negative.")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size.")

    words = text.split()
    chunks: list[DocumentChunk] = []
    start = 0
    index = 0

    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk_text = " ".join(words[start:end])

        if chunk_text:
            chunk_hash = _hash_text(f"{source}:{index}:{chunk_text}")[:16]
            chunks.append(
                DocumentChunk(
                    chunk_id=chunk_hash,
                    source=source,
                    text=chunk_text,
                    chunk_index=index,
                    word_count=len(chunk_text.split()),
                )
            )

        if end == len(words):
            break

        start = end - overlap
        index += 1

    return chunks


def load_and_chunk_documents(
    files: Iterable[tuple[BinaryIO, str]],
    chunk_size: int = 500,
    overlap: int = 50,
) -> list[DocumentChunk]:
    """Load many files and return a combined chunk list."""

    all_chunks: list[DocumentChunk] = []

    for file_obj, filename in files:
        text = load_document(file_obj, filename)
        if not text:
            continue
        all_chunks.extend(split_text_into_chunks(text, filename, chunk_size, overlap))

    return all_chunks
