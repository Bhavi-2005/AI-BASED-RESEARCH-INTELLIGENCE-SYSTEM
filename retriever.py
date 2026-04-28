"""Retrieval layer that connects query embeddings to FAISS search."""

from __future__ import annotations

from dataclasses import dataclass

from embedding import EmbeddingModel
from vector_store import FaissVectorStore, SearchResult


@dataclass(frozen=True)
class RetrievalOutput:
    """Search results plus aggregate confidence."""

    results: list[SearchResult]
    confidence: float
    is_relevant: bool


class Retriever:
    """Embeds queries and returns the most relevant document chunks."""

    def __init__(
        self,
        embedding_model: EmbeddingModel,
        vector_store: FaissVectorStore,
        relevance_threshold: float = 0.25,
    ) -> None:
        self.embedding_model = embedding_model
        self.vector_store = vector_store
        self.relevance_threshold = relevance_threshold

    def retrieve(self, query: str, top_k: int = 5) -> RetrievalOutput:
        query_embedding = self.embedding_model.embed_query(query)
        results = self.vector_store.search(query_embedding, top_k=top_k)
        confidence = self._confidence_from_scores([result.score for result in results])
        is_relevant = bool(results) and results[0].score >= self.relevance_threshold

        return RetrievalOutput(
            results=results,
            confidence=confidence,
            is_relevant=is_relevant,
        )

    @staticmethod
    def _confidence_from_scores(scores: list[float]) -> float:
        """Convert cosine similarities to a readable 0-100 confidence score."""

        if not scores:
            return 0.0

        top_scores = scores[:3]
        clipped = [max(0.0, min(1.0, score)) for score in top_scores]
        return round((sum(clipped) / len(clipped)) * 100, 2)
