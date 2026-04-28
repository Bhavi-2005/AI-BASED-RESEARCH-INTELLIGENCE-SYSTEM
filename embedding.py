"""Sentence-BERT embedding service with lightweight disk caching."""

from __future__ import annotations

import hashlib
import pickle
from pathlib import Path
from typing import Iterable

import numpy as np
from sentence_transformers import SentenceTransformer


DEFAULT_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


class EmbeddingModel:
    """Wraps SentenceTransformer and caches text embeddings by content hash."""

    def __init__(
        self,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        cache_dir: str | Path = ".rag_cache/embeddings",
    ) -> None:
        self.model_name = model_name
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.model = SentenceTransformer(model_name)

    def _cache_key(self, text: str) -> Path:
        digest = hashlib.sha256(f"{self.model_name}:{text}".encode("utf-8")).hexdigest()
        return self.cache_dir / f"{digest}.pkl"

    def _load_cached_embedding(self, text: str) -> np.ndarray | None:
        path = self._cache_key(text)
        if not path.exists():
            return None

        with path.open("rb") as file:
            return pickle.load(file)

    def _save_cached_embedding(self, text: str, embedding: np.ndarray) -> None:
        path = self._cache_key(text)
        with path.open("wb") as file:
            pickle.dump(embedding.astype("float32"), file)

    def embed_texts(self, texts: Iterable[str], batch_size: int = 32) -> np.ndarray:
        """Embed texts and return L2-normalized float32 vectors.

        Normalized vectors let FAISS inner-product search behave like cosine
        similarity, which is intuitive for RAG confidence scoring.
        """

        text_list = list(texts)
        embeddings: list[np.ndarray | None] = []
        missing_texts: list[str] = []
        missing_indexes: list[int] = []

        for index, text in enumerate(text_list):
            cached = self._load_cached_embedding(text)
            embeddings.append(cached)
            if cached is None:
                missing_texts.append(text)
                missing_indexes.append(index)

        if missing_texts:
            computed = self.model.encode(
                missing_texts,
                batch_size=batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            ).astype("float32")

            for index, text, embedding in zip(missing_indexes, missing_texts, computed):
                embeddings[index] = embedding
                self._save_cached_embedding(text, embedding)

        return np.vstack([embedding for embedding in embeddings if embedding is not None]).astype(
            "float32"
        )

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single user query."""

        return self.embed_texts([query])[0]
