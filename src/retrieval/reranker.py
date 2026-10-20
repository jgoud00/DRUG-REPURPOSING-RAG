"""Cross-encoder reranker for re-ranking hybrid retrieval results."""

import logging
from typing import List, Optional, Tuple

import numpy as np
from sentence_transformers import CrossEncoder

from src.config import get_settings
from src.retrieval.hybrid_retriever import RetrievalResult

logger = logging.getLogger(__name__)


class MockCrossEncoder:
    """Mock cross-encoder for fast unit testing."""

    def __init__(self, model_name: str = "mock"):
        self.model_name = model_name

    def predict(self, pairs: List[List[str]], batch_size: int = 16, show_progress_bar: bool = False) -> np.ndarray:
        """Return deterministic scores based on query-document overlap."""
        scores = []
        for query, doc in pairs:
            # Simple heuristic: score based on token overlap
            query_tokens = set(query.lower().split())
            doc_tokens = set(doc.lower().split())
            if not query_tokens:
                scores.append(0.0)
            else:
                overlap = len(query_tokens & doc_tokens) / len(query_tokens)
                scores.append(min(overlap * 2.0, 1.0))  # Scale and cap at 1.0
        return np.array(scores, dtype=np.float32)


class CrossEncoderReranker:
    """Cross-encoder reranker for (query, document) pairs."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        use_mock: bool = False,
        settings=None,
    ):
        self.settings = settings or get_settings()
        self.model_name = model_name or self.settings.reranking.model_name
        self.use_mock = use_mock
        self.batch_size = self.settings.reranking.batch_size
        self.max_length = self.settings.reranking.max_length
        self.top_k = self.settings.reranking.top_k
        self._model = None

    @property
    def model(self):
        if self._model is None:
            if self.use_mock:
                logger.info("Using MockCrossEncoder for testing")
                self._model = MockCrossEncoder(self.model_name)
            else:
                logger.info(f"Loading CrossEncoder: {self.model_name}")
                self._model = CrossEncoder(self.model_name, max_length=self.max_length)
        return self._model

    def rerank(
        self,
        query: str,
        results: List[RetrievalResult],
        top_k: Optional[int] = None,
    ) -> List[RetrievalResult]:
        """Rerank results using cross-encoder."""
        if not results:
            return []

        top_k = top_k or self.top_k
        pairs = [[query, r.text] for r in results]

        logger.info(f"Reranking {len(pairs)} query-document pairs")

        scores = self.model.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=False,
        )

        # Attach rerank scores and sort
        for result, score in zip(results, scores):
            result.rerank_score = float(score)

        # Sort by rerank score descending
        results.sort(key=lambda x: x.rerank_score, reverse=True)

        return results[:top_k]

    def rerank_with_scores(
        self,
        query: str,
        texts: List[str],
        top_k: Optional[int] = None,
    ) -> List[Tuple[str, float]]:
        """Rerank raw texts and return (text, score) pairs."""
        pairs = [[query, text] for text in texts]
        scores = self.model.predict(pairs, batch_size=self.batch_size, show_progress_bar=False)

        ranked = list(zip(texts, scores))
        ranked.sort(key=lambda x: x[1], reverse=True)

        return ranked[:top_k] if top_k else ranked