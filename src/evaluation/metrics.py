"""Evaluation metrics for retrieval and generation."""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Set
from collections import defaultdict

from src.retrieval.hybrid_retriever import RetrievalResult
from src.retrieval.citation_tracker import Citation
from src.generation.answer_synthesizer import SynthesizedAnswer


@dataclass
class RetrievalMetrics:
    """Retrieval evaluation metrics."""
    recall_at_k: Dict[int, float] = field(default_factory=dict)
    mrr: float = 0.0
    ndcg_at_k: Dict[int, float] = field(default_factory=dict)
    precision_at_k: Dict[int, float] = field(default_factory=dict)


@dataclass
class GenerationMetrics:
    """Generation evaluation metrics."""
    citation_precision: float = 0.0
    citation_recall: float = 0.0
    hallucination_rate: float = 0.0
    answer_length: int = 0
    num_citations: int = 0
    num_verified_citations: int = 0
    num_unverified_citations: int = 0


@dataclass
class PipelineMetrics:
    """Combined pipeline metrics."""
    retrieval: RetrievalMetrics = field(default_factory=RetrievalMetrics)
    generation: GenerationMetrics = field(default_factory=GenerationMetrics)
    end_to_end: Dict[str, float] = field(default_factory=dict)


class RetrievalEvaluator:
    """Evaluate retrieval quality against ground truth."""

    def __init__(self, k_values: List[int] = None):
        self.k_values = k_values or [1, 3, 5, 10, 20]

    def evaluate(
        self,
        retrieved: List[Any],
        relevant_doc_ids: Set[str],
        relevant_pairs: Optional[Set[tuple]] = None,
    ) -> RetrievalMetrics:
        """Evaluate retrieval against known relevant documents."""
        # Handle both list of RetrievalResult and list of doc_id strings
        if retrieved and hasattr(retrieved[0], 'doc_id'):
            retrieved_ids = [r.doc_id for r in retrieved]
        else:
            retrieved_ids = retrieved
        metrics = RetrievalMetrics()

        # Recall@k
        for k in self.k_values:
            top_k_ids = set(retrieved_ids[:k])
            if relevant_doc_ids:
                recall = len(top_k_ids & relevant_doc_ids) / len(relevant_doc_ids)
            else:
                recall = 0.0
            metrics.recall_at_k[k] = recall

        # MRR
        mrr = 0.0
        for rank, doc_id in enumerate(retrieved_ids, 1):
            if doc_id in relevant_doc_ids:
                mrr = 1.0 / rank
                break
        metrics.mrr = mrr

        # NDCG@k
        for k in self.k_values:
            metrics.ndcg_at_k[k] = self._ndcg_at_k(retrieved_ids[:k], relevant_doc_ids, k)

        # Precision@k
        for k in self.k_values:
            top_k = retrieved_ids[:k]
            if top_k:
                precision = len(set(top_k) & relevant_doc_ids) / len(top_k)
            else:
                precision = 0.0
            metrics.precision_at_k[k] = precision

        return metrics

    def _ndcg_at_k(self, retrieved_ids: List[str], relevant: Set[str], k: int) -> float:
        """Compute NDCG@k."""
        if not relevant:
            return 0.0

        # Ideal DCG
        ideal_relevance = [1] * min(len(relevant), k) + [0] * max(0, k - len(relevant))
        idcg = sum(rel / (i + 1) for i, rel in enumerate(ideal_relevance))

        # Actual DCG
        dcg = 0.0
        for i, doc_id in enumerate(retrieved_ids[:k]):
            rel = 1.0 if doc_id in relevant else 0.0
            dcg += rel / (i + 1)

        return dcg / idcg if idcg > 0 else 0.0


class GenerationEvaluator:
    """Evaluate generation quality."""

    def evaluate_answer(self, answer: SynthesizedAnswer) -> GenerationMetrics:
        """Evaluate a synthesized answer."""
        total_citations = answer.num_citations if hasattr(answer, 'num_citations') else len(answer.citations)
        verified = answer.num_verified_citations if hasattr(answer, 'num_verified_citations') else len(answer.citations)
        unverified = answer.num_unverified_citations if hasattr(answer, 'num_unverified_citations') else len(answer.unverified_claims)

        return GenerationMetrics(
            citation_precision=answer.citation_precision,
            citation_recall=verified / max(total_citations, 1) if total_citations > 0 else 1.0,
            hallucination_rate=answer.hallucination_rate,
            answer_length=len(answer.answer),
            num_citations=total_citations,
            num_verified_citations=verified,
            num_unverified_citations=unverified,
        )

    def evaluate_citation_coverage(
        self,
        answer_text: str,
        citations: List[Citation],
    ) -> Dict[str, float]:
        """Evaluate how well citations cover the answer."""
        import re
        sentences = [s.strip() for s in answer_text.split('.') if s.strip()]
        if not sentences:
            return {"coverage": 0.0, "sentences_per_citation": 0.0}

        # Count sentences with citations
        citation_pattern = re.compile(r'\[(PMID|CHEMBL):\s*([^\]]+)\]')
        cited_sentences = 0
        for sent in sentences:
            if citation_pattern.search(sent):
                cited_sentences += 1

        coverage = cited_sentences / len(sentences)
        return {
            "coverage": coverage,
            "sentences_per_citation": len(sentences) / max(len(set(c.doc_id for c in citations)), 1),
            "total_sentences": len(sentences),
            "cited_sentences": cited_sentences,
        }


class PipelineEvaluator:
    """End-to-end pipeline evaluation."""

    def __init__(self, retrieval_evaluator: RetrievalEvaluator = None, generation_evaluator: GenerationEvaluator = None):
        self.retrieval_evaluator = retrieval_evaluator or RetrievalEvaluator()
        self.generation_evaluator = generation_evaluator or GenerationEvaluator()

    def evaluate_pipeline(
        self,
        query: str,
        retrieved: List[RetrievalResult],
        reranked: List[RetrievalResult],
        synthesized: Any,
        ground_truth: Dict[str, Any],
    ) -> PipelineMetrics:
        """Evaluate full pipeline against ground truth."""
        # Ground truth expected keys: relevant_doc_ids, relevant_pairs, expected_drugs
        relevant_doc_ids = set(ground_truth.get("relevant_doc_ids", []))
        expected_drugs = set(ground_truth.get("expected_drugs", []))

        # Retrieval metrics on reranked results
        retrieval_metrics = self.retrieval_evaluator.evaluate(reranked, relevant_doc_ids)

        # Generation metrics
        gen_metrics = self.generation_evaluator.evaluate_answer(synthesized)

        # End-to-end: did we find expected drugs?
        found_drugs = set()
        for dc in synthesized.drug_candidates:
            found_drugs.add(dc.get("drug_id", ""))

        drug_recall = len(found_drugs & expected_drugs) / len(expected_drugs) if expected_drugs else 0.0

        return PipelineMetrics(
            retrieval=retrieval_metrics,
            generation=gen_metrics,
            end_to_end={
                "drug_recall": drug_recall,
                "answer_length": len(synthesized.answer),
            },
        )


def compute_metrics_at_k(
    retrieved: List[str],
    relevant: Set[str],
    k_values: List[int],
) -> Dict[str, float]:
    """Compute standard IR metrics at multiple k values."""
    metrics = {}
    for k in k_values:
        top_k = retrieved[:k]
        rel_in_top_k = len(set(top_k) & relevant)

        # Recall@k
        metrics[f"recall@{k}"] = rel_in_top_k / len(relevant) if relevant else 0.0

        # Precision@k
        metrics[f"precision@{k}"] = rel_in_top_k / k if k > 0 else 0.0

        # F1@k
        p = metrics[f"precision@{k}"]
        r = metrics[f"recall@{k}"]
        metrics[f"f1@{k}"] = 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    return metrics


def reciprocal_rank(retrieved: List[str], relevant: Set[str]) -> float:
    """Compute Reciprocal Rank (1/rank of first relevant)."""
    for rank, doc_id in enumerate(retrieved, 1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


def average_precision(retrieved: List[str], relevant: Set[str]) -> float:
    """Compute Average Precision (AP)."""
    if not relevant:
        return 0.0

    precisions = []
    relevant_found = 0
    for i, doc_id in enumerate(retrieved):
        if doc_id in relevant:
            relevant_found += 1
            precisions.append(relevant_found / (i + 1))

    return sum(precisions) / len(relevant) if relevant else 0.0