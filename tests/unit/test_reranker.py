"""Unit tests for cross-encoder reranker."""

import pytest
import numpy as np

from src.retrieval.reranker import CrossEncoderReranker, MockCrossEncoder
from src.retrieval.hybrid_retriever import RetrievalResult


class TestMockCrossEncoder:
    """Test mock cross-encoder."""

    def test_mock_predict(self):
        """Test mock prediction."""
        model = MockCrossEncoder()
        pairs = [
            ["Alzheimer's disease PDE5A", "Sildenafil inhibits PDE5A in Alzheimer Disease"],
            ["Aspirin COX-1", "Aspirin inhibits COX-1 in cardiovascular disease"],
            ["Unrelated query", "Completely different text about cooking"],
        ]
        scores = model.predict(pairs)

        assert len(scores) == 3
        # First two should have higher overlap
        assert scores[0] > scores[2]
        assert scores[1] > scores[2]
        # All scores should be in [0, 1]
        assert all(0 <= s <= 1 for s in scores)


class TestCrossEncoderReranker:
    """Test cross-encoder reranker."""

    def test_rerank_with_mock(self):
        """Test reranking with mock model."""
        reranker = CrossEncoderReranker(use_mock=True)

        results = [
            RetrievalResult(
                doc_id="1",
                text="Sildenafil inhibits PDE5A in Alzheimer Disease",
                metadata={},
                fused_score=0.5,
            ),
            RetrievalResult(
                doc_id="2",
                text="Aspirin inhibits COX-1 in cardiovascular disease",
                metadata={},
                fused_score=0.4,
            ),
            RetrievalResult(
                doc_id="3",
                text="Metformin activates AMPK in diabetes",
                metadata={},
                fused_score=0.3,
            ),
        ]

        query = "Alzheimer's disease PDE5A"
        reranked = reranker.rerank(query, results, top_k=2)

        assert len(reranked) == 2
        # Sildenafil/PDE5A/Alzheimer should be top
        assert reranked[0].doc_id == "1"
        assert hasattr(reranked[0], 'rerank_score')
        assert reranked[0].rerank_score > reranked[1].rerank_score

    def test_rerank_empty(self):
        """Test reranking empty results."""
        reranker = CrossEncoderReranker(use_mock=True)
        results = reranker.rerank("query", [])
        assert results == []

    def test_rerank_preserves_metadata(self):
        """Test that reranking preserves metadata."""
        reranker = CrossEncoderReranker(use_mock=True)

        results = [
            RetrievalResult(
                doc_id="1",
                text="Test text",
                metadata={"source": "chembl", "drug_id": "CHEMBL25"},
                fused_score=0.5,
            ),
        ]

        reranked = reranker.rerank("query", results)
        assert reranked[0].metadata["source"] == "chembl"
        assert reranked[0].metadata["drug_id"] == "CHEMBL25"

    def test_rerank_with_scores(self):
        """Test rerank_with_scores method."""
        reranker = CrossEncoderReranker(use_mock=True)

        texts = [
            "Sildenafil inhibits PDE5A in Alzheimer Disease",
            "Aspirin inhibits COX-1",
        ]
        ranked = reranker.rerank_with_scores("Alzheimer PDE5A", texts, top_k=1)

        assert len(ranked) == 1
        assert "Sildenafil" in ranked[0][0]
        assert ranked[0][1] > 0