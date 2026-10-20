"""Unit tests for evaluation metrics."""

import pytest
from src.evaluation.metrics import (
    RetrievalEvaluator,
    GenerationEvaluator,
    PipelineEvaluator,
    RetrievalMetrics,
    GenerationMetrics,
    reciprocal_rank,
    average_precision,
    compute_metrics_at_k,
)


class TestRetrievalEvaluator:
    """Test retrieval evaluation metrics."""

    def test_recall_at_k(self):
        """Test Recall@k computation."""
        evaluator = RetrievalEvaluator(k_values=[1, 3, 5])
        retrieved = ["doc1", "doc2", "doc3", "doc4", "doc5"]
        relevant = {"doc2", "doc4", "doc6"}  # doc6 not retrieved

        metrics = evaluator.evaluate(retrieved, relevant)

        assert metrics.recall_at_k[1] == 0.0  # doc1 not relevant
        assert metrics.recall_at_k[3] == 1/3  # doc2 in top 3
        assert metrics.recall_at_k[5] == 2/3  # doc2, doc4 in top 5

    def test_mrr(self):
        """Test Mean Reciprocal Rank."""
        evaluator = RetrievalEvaluator()

        # First relevant at rank 2
        retrieved = ["doc1", "doc2", "doc3"]
        relevant = {"doc2", "doc4"}
        metrics = evaluator.evaluate(retrieved, relevant)
        assert metrics.mrr == 0.5  # 1/2

        # First relevant at rank 1
        retrieved = ["doc2", "doc1"]
        metrics = evaluator.evaluate(retrieved, relevant)
        assert metrics.mrr == 1.0  # 1/1

        # No relevant found
        retrieved = ["doc1", "doc3"]
        relevant = {"doc2", "doc4"}
        metrics = evaluator.evaluate(retrieved, relevant)
        assert metrics.mrr == 0.0

    def test_ndcg(self):
        """Test NDCG@k."""
        evaluator = RetrievalEvaluator(k_values=[3])
        retrieved = ["doc1", "doc2", "doc3"]
        relevant = {"doc2", "doc3"}

        metrics = evaluator.evaluate(retrieved, relevant)
        # NDCG should be > 0
        assert metrics.ndcg_at_k[3] > 0.0

    def test_precision_at_k(self):
        """Test Precision@k."""
        evaluator = RetrievalEvaluator(k_values=[3, 5])
        retrieved = ["doc1", "doc2", "doc3", "doc4", "doc5"]
        relevant = {"doc2", "doc4"}

        metrics = evaluator.evaluate(retrieved, relevant)
        assert metrics.precision_at_k[3] == 1/3  # doc2 in top 3
        assert metrics.precision_at_k[5] == 2/5  # doc2, doc4 in top 5


class TestGenerationEvaluator:
    """Test generation evaluation metrics."""

    def test_evaluate_answer(self):
        """Test generation metrics evaluation."""
        from src.generation.answer_synthesizer import SynthesizedAnswer
        from src.retrieval.citation_tracker import Citation

        evaluator = GenerationEvaluator()

        citations = [
            Citation(doc_id="doc1", source="pubmed", text_snippet="...", retrieval_score=1.0, pmid="123"),
            Citation(doc_id="doc2", source="chembl", text_snippet="...", retrieval_score=1.0, drug_id="CHEMBL25"),
        ]

        answer = SynthesizedAnswer(
            answer="Test answer with citations [PMID: 123] and [CHEMBL: CHEMBL25].",
            citations=citations,
            citation_precision=1.0,
            hallucination_rate=0.0,
        )

        metrics = evaluator.evaluate_answer(answer)

        assert metrics.citation_precision == 1.0
        assert metrics.hallucination_rate == 0.0
        assert metrics.num_citations == 2
        assert metrics.num_verified_citations == 2

    def test_citation_coverage(self):
        """Test citation coverage evaluation."""
        evaluator = GenerationEvaluator()
        from src.retrieval.citation_tracker import Citation

        answer = "Sentence one [PMID: 123]. Sentence two without citation. Sentence three [CHEMBL: CHEMBL25]."
        citations = [
            Citation(doc_id="1", source="pubmed", text_snippet="", retrieval_score=1.0, pmid="123"),
            Citation(doc_id="2", source="chembl", text_snippet="", retrieval_score=1.0, drug_id="CHEMBL25"),
        ]

        coverage = evaluator.evaluate_citation_coverage(answer, citations)

        assert coverage["total_sentences"] == 3
        assert coverage["cited_sentences"] == 2
        assert coverage["coverage"] == 2/3


class TestReciprocalRank:
    """Test reciprocal rank function."""

    def test_first_relevant(self):
        assert reciprocal_rank(["doc1", "doc2"], {"doc1"}) == 1.0

    def test_second_relevant(self):
        assert reciprocal_rank(["doc1", "doc2"], {"doc2"}) == 0.5

    def test_no_relevant(self):
        assert reciprocal_rank(["doc1", "doc2"], {"doc3"}) == 0.0


class TestAveragePrecision:
    """Test average precision function."""

    def test_perfect(self):
        retrieved = ["doc1", "doc2", "doc3"]
        relevant = {"doc1", "doc2", "doc3"}
        ap = average_precision(retrieved, relevant)
        assert ap == 1.0

    def test_partial(self):
        retrieved = ["doc1", "doc2", "doc3", "doc4"]
        relevant = {"doc2", "doc4"}
        ap = average_precision(retrieved, relevant)
        # Precision at rank 2: 1/2, rank 4: 2/4 -> avg = (0.5 + 0.5)/2 = 0.5
        assert ap == 0.5

    def test_none_relevant(self):
        retrieved = ["doc1", "doc2"]
        relevant = set()
        assert average_precision(retrieved, relevant) == 0.0


class TestComputeMetricsAtK:
    """Test compute_metrics_at_k function."""

    def test_basic(self):
        retrieved = ["doc1", "doc2", "doc3", "doc4", "doc5"]
        relevant = {"doc2", "doc4"}

        metrics = compute_metrics_at_k(retrieved, relevant, [3, 5])

        assert metrics["recall@3"] == 1/2  # doc2 in top 3
        assert metrics["recall@5"] == 2/2  # both in top 5
        assert metrics["precision@3"] == 1/3
        assert metrics["precision@5"] == 2/5
        assert metrics["f1@3"] == 2/5  # 2*(1/3)*(1/2) / (1/3+1/2) = (1/3)/(5/6) = 2/5


class TestPipelineEvaluator:
    """Test end-to-end pipeline evaluator."""

    def test_evaluate_pipeline(self):
        """Test pipeline evaluation with mock data."""
        from src.generation.answer_synthesizer import SynthesizedAnswer
        from src.retrieval.citation_tracker import Citation
        from src.retrieval.hybrid_retriever import RetrievalResult

        evaluator = PipelineEvaluator()

        # Mock retrieved/reranked results
        retrieved = [
            RetrievalResult(doc_id="doc1", text="...", metadata={}, fused_score=0.9),
            RetrievalResult(doc_id="doc2", text="...", metadata={}, fused_score=0.8),
        ]
        reranked = [
            RetrievalResult(doc_id="doc2", text="...", metadata={}, fused_score=0.85),
            RetrievalResult(doc_id="doc1", text="...", metadata={}, fused_score=0.8),
        ]

        # Mock synthesized answer
        synthesized = SynthesizedAnswer(
            answer="Test [PMID: 123]",
            citations=[Citation(doc_id="doc1", source="pubmed", text_snippet="", retrieval_score=1.0, pmid="123")],
            drug_candidates=[{"drug_id": "CHEMBL25", "drug_name": "Aspirin"}],
            citation_precision=1.0,
            hallucination_rate=0.0,
        )

        ground_truth = {
            "relevant_doc_ids": ["doc2"],
            "expected_drugs": ["CHEMBL25"],
        }

        metrics = evaluator.evaluate_pipeline(
            query="test query",
            retrieved=retrieved,
            reranked=reranked,
            synthesized=synthesized,
            ground_truth=ground_truth,
        )

        assert isinstance(metrics.retrieval, RetrievalMetrics)
        assert isinstance(metrics.generation, GenerationMetrics)
        assert "drug_recall" in metrics.end_to_end