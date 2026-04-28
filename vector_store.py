"""FAISS vector store for chunk similarity search."""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np

from data_loader import DocumentChunk


@dataclass(frozen=True)
class SearchResult:
    """A retrieved chunk and its cosine similarity score."""

    chunk: DocumentChunk
    score: float


class FaissVectorStore:
    """Stores embeddings in FAISS and maps result ids back to chunks."""

    def __init__(self) -> None:
        self.index: faiss.IndexFlatIP | None = None
        self.chunks: list[DocumentChunk] = []

    @property
    def is_ready(self) -> bool:
        return self.index is not None and len(self.chunks) > 0

    def build(self, chunks: list[DocumentChunk], embeddings: np.ndarray) -> None:
        if len(chunks) == 0:
            raise ValueError("Cannot build a vector store with no chunks.")
        if embeddings.ndim != 2:
            raise ValueError("Embeddings must be a 2D matrix.")
        if len(chunks) != embeddings.shape[0]:
            raise ValueError("Number of chunks must match number of embeddings.")

        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings.astype("float32"))
        self.chunks = chunks

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> list[SearchResult]:
        if not self.is_ready or self.index is None:
            raise ValueError("Vector store is empty. Upload and index documents first.")

        query = query_embedding.reshape(1, -1).astype("float32")
        limit = min(top_k, len(self.chunks))
        scores, indexes = self.index.search(query, limit)

        results: list[SearchResult] = []
        for score, index in zip(scores[0], indexes[0]):
            if index == -1:
                continue
            results.append(SearchResult(chunk=self.chunks[index], score=float(score)))

        return results

    def save(self, directory: str | Path) -> None:
        if not self.is_ready or self.index is None:
            raise ValueError("Cannot save an empty vector store.")

        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, str(directory / "index.faiss"))
        with (directory / "chunks.pkl").open("wb") as file:
            pickle.dump(self.chunks, file)

    @classmethod
    def load(cls, directory: str | Path) -> "FaissVectorStore":
        directory = Path(directory)
        store = cls()
        store.index = faiss.read_index(str(directory / "index.faiss"))
        with (directory / "chunks.pkl").open("rb") as file:
            store.chunks = pickle.load(file)
        return store
