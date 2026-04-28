"""Sentence embedding helpers."""

from __future__ import annotations

from functools import lru_cache

from sentence_transformers import SentenceTransformer


DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def get_model(model_name: str = DEFAULT_EMBEDDING_MODEL) -> SentenceTransformer:
    return SentenceTransformer(model_name)


def get_embeddings(sentences: list[str], model_name: str = DEFAULT_EMBEDDING_MODEL):
    """Convert sentences to normalized dense vectors."""

    if not sentences:
        return []

    model = get_model(model_name)
    return model.encode(
        sentences,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
