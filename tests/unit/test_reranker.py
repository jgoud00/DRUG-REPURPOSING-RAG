"""Unit tests for cross-encoder reranker."""

import pytest
import numpy as np

from src.retrieval.reranker import CrossEncoderReranker, MockCrossEncoder
from src.retrieval.hybrid_retriever import RetrievalResult


class DeterministicMockCrossEncoder:
    """Deterministic mock cross-encoder for testing with explicit scores."""

    def __init__(self, score_map=None):
        """Initialize with optional score map: {(query, doc): score}."""
        self.score_map = score_map or {}

    def predict(self, pairs, batch_size=16, show_progress_bar=False):
        """Return deterministic scores based on score_map or token overlap fallback."""
        scores = []
        for query, doc in pairs:
            key = (query, doc)
            if key in self.score_map:
                scores.append(self.score_map[key])
            else:
                # Fallback: token overlap
                query_tokens = set(query.lower().split())
                doc_tokens = set(doc.lower().split())
                if not query_tokens:
                    scores.append(0.0)
                else:
                    overlap = len(query_tokens & doc_tokens) / len(query_tokens)
                    scores.append(min(overlap * 2.0, 1.0))
        return np.array(scores, dtype=np.float32)


class TestDeterministicMockCrossEncoder:
    """Test deterministic mock cross-encoder."""

    def test_mock_predict_deterministic(self):
        """Test deterministic mock prediction."""
        score_map = {
            ("query1", "doc1 relevant"): 0.9,
            ("query1", "doc2 less relevant"): 0.5,
            ("query1", "doc3 irrelevant"): 0.1,
        }
        model = DeterministicMockCrossEncoder(score_map)
        pairs = list(score_map.keys())
        scores = model.predict(pairs)

        assert len(scores) == 3
        assert scores[0] == 0.9
        assert scores[1] == 0.5
        assert scores[2] == 0.1

    def test_mock_predict_fallback(self):
        """Test fallback to token overlap when no score map entry."""
        model = DeterministicMockCrossEncoder()
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
        # Use deterministic mock with explicit scores
        score_map = {
            ("Alzheimer's disease PDE5A", "Sildenafil inhibits PDE5A in Alzheimer Disease"): 0.95,
            ("Alzheimer's disease PDE5A", "Aspirin inhibits COX-1 in cardiovascular disease"): 0.1,
            ("Alzheimer's disease PDE5A", "Metformin activates AMPK in diabetes"): 0.05,
        }
        reranker = CrossEncoderReranker(use_mock=True)
        reranker._model = DeterministicMockCrossEncoder(score_map)

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
        assert abs(reranked[0].rerank_score - 0.95) < 1e-6
        assert reranked[1].doc_id == "2"
        assert abs(reranked[1].rerank_score - 0.1) < 1e-6

    def test_rerank_empty(self):
        """Test reranking empty results."""
        reranker = CrossEncoderReranker(use_mock=True)
        results = reranker.rerank("query", [])
        assert results == []

    def test_rerank_preserves_metadata(self):
        """Test that reranking preserves metadata."""
        score_map = {
            ("query", "doc with metadata"): 0.8,
        }
        reranker = CrossEncoderReranker(use_mock=True)
        reranker._model = DeterministicMockCrossEncoder(score_map)

        results = [
            RetrievalResult(
                doc_id="1",
                text="doc with metadata",
                metadata={"source": "chembl", "drug_id": "CHEMBL25"},
                fused_score=0.5,
            ),
        ]

        reranked = reranker.rerank("query", results)
        assert reranked[0].metadata["source"] == "chembl"
        assert reranked[0].metadata["drug_id"] == "CHEMBL25"
        assert abs(reranked[0].rerank_score - 0.8) < 1e-6

    def test_rerank_with_scores(self):
        """Test rerank_with_scores method."""
        score_map = {
            ("Alzheimer PDE5A", "Sildenafil inhibits PDE5A in Alzheimer Disease"): 0.9,
            ("Alzheimer PDE5A", "Aspirin inhibits COX-1"): 0.2,
        }
        reranker = CrossEncoderReranker(use_mock=True)
        reranker._model = DeterministicMockCrossEncoder(score_map)

        texts = [
            "Sildenafil inhibits PDE5A in Alzheimer Disease",
            "Aspirin inhibits COX-1",
        ]
        ranked = reranker.rerank_with_scores("Alzheimer PDE5A", texts, top_k=1)

        assert len(ranked) == 1
        assert "Sildenafil" in ranked[0][0]
        assert ranked[0][1] == 0.9

    def test_rerank_ties_broken_by_fused_score(self):
        """Test that ties in rerank score are broken by fused_score."""
        score_map = {
            ("query", "doc1"): 0.5,
            ("query", "doc2"): 0.5,  # Same rerank score
        }
        reranker = CrossEncoderReranker(use_mock=True)
        reranker._model = DeterministicMockCrossEncoder(score_map)

        results = [
            RetrievalResult(doc_id="doc1", text="doc1", metadata={}, fused_score=0.9),
            RetrievalResult(doc_id="doc2", text="doc2", metadata={}, fused_score=0.5),
        ]

        reranked = reranker.rerank("query", results, top_k=2)
        # doc1 should come first due to higher fused_score
        assert reranked[0].doc_id == "doc1"
        assert reranked[1].doc_id == "doc2"

    def test_rerank_top_k_limits_results(self):
        """Test that top_k limits the number of returned results."""
        score_map = {
            ("q", f"doc{i}"): 0.9 - i * 0.1 for i in range(5)
        }
        reranker = CrossEncoderReranker(use_mock=True)
        reranker._model = DeterministicMockCrossEncoder(score_map)

        results = [
            RetrievalResult(doc_id=f"doc{i}", text=f"doc{i}", metadata={}, fused_score=0.5)
            for i in range(5)
        ]

        reranked = reranker.rerank("q", results, top_k=3)
        assert len(reranked) == 3
        assert [r.doc_id for r in reranked] == ["doc0", "doc1", "doc2"]