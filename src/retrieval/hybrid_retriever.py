"""Hybrid retriever with RRF fusion."""

import logging
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

import numpy as np

from src.config import get_settings
from src.embeddings.hybrid_indexer import HybridIndexer
from src.embeddings.model_loader import load_embedding_model
from src.retrieval.citation_tracker import Citation, CitationTracker

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """Result from hybrid retrieval with citation tracking."""
    doc_id: str
    text: str
    metadata: Dict[str, Any]
    dense_score: Optional[float] = None
    sparse_score: Optional[float] = None
    fused_score: float = 0.0
    rank_dense: Optional[int] = None
    rank_sparse: Optional[int] = None
    citation: Optional[Citation] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "text": self.text,
            "metadata": self.metadata,
            "dense_score": self.dense_score,
            "sparse_score": self.sparse_score,
            "fused_score": self.fused_score,
            "rank_dense": self.rank_dense,
            "rank_sparse": self.rank_sparse,
        }


class HybridRetriever:
    """Hybrid retrieval with Weighted Reciprocal Rank Fusion (RRF)."""

    def __init__(
        self,
        indexer: Optional[HybridIndexer] = None,
        embedder_model=None,
        settings=None,
        citation_tracker: Optional[CitationTracker] = None,
    ):
        self.settings = settings or get_settings()
        self.indexer = indexer
        self.embedder_model = embedder_model or load_embedding_model(settings=self.settings)
        self.citation_tracker = citation_tracker or CitationTracker(
            citation_format=self.settings.generation.prompt.citation_format
        )

        # RRF parameters
        hybrid_config = self.settings.vector_store.hybrid
        self.dense_weight = hybrid_config.get("dense_weight", 0.7)
        self.sparse_weight = hybrid_config.get("sparse_weight", 0.3)
        self.rrf_k = hybrid_config.get("rrf_k", 60)
        self.top_k_initial = self.settings.retrieval.top_k_initial
        self.top_k_final = self.settings.retrieval.top_k_final

    def _ensure_indexer(self) -> HybridIndexer:
        """Lazy initialize indexer."""
        if self.indexer is None:
            self.indexer = HybridIndexer(settings=self.settings)
            self.indexer.initialize()
            # Try to load BM25 index
            self.indexer.load_bm25_index()
        return self.indexer

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> List[RetrievalResult]:
        """Perform hybrid retrieval with RRF fusion."""
        top_k = top_k or self.top_k_initial
        indexer = self._ensure_indexer()

        # Get query embedding
        query_embedding = self.embedder_model.encode([query], batch_size=1, show_progress_bar=False)[0]

        # Dense search - returns (doc_id, score, {"text": doc, "metadata": meta})
        dense_hits = indexer.dense_search(
            query_embedding,
            top_k=top_k,
            metadata_filter=metadata_filter,
        )

        # Sparse search - returns (doc_id, score)
        sparse_hits = indexer.sparse_search(query, top_k=top_k)

        # Convert to ranked dicts for RRF
        dense_ranked = {doc_id: rank + 1 for rank, (doc_id, score, _) in enumerate(dense_hits)}
        sparse_ranked = {doc_id: rank + 1 for rank, (doc_id, score) in enumerate(sparse_hits)}

        # Build lookup for dense results (doc_id -> (score, text, metadata))
        dense_lookup = {doc_id: (score, meta["text"], meta["metadata"]) for doc_id, score, meta in dense_hits}

        # Get all unique doc_ids
        all_doc_ids = set(dense_ranked.keys()) | set(sparse_ranked.keys())

        # Compute RRF scores
        results = []
        for doc_id in all_doc_ids:
            rank_d = dense_ranked.get(doc_id)
            rank_s = sparse_ranked.get(doc_id)

            rrf_score = 0.0
            if rank_d is not None:
                rrf_score += self.dense_weight * (1.0 / (self.rrf_k + rank_d))
            if rank_s is not None:
                rrf_score += self.sparse_weight * (1.0 / (self.rrf_k + rank_s))

            # Get document details from dense results if available
            doc_text = ""
            metadata = {}
            dense_score = None
            if doc_id in dense_lookup:
                dense_score, doc_text, metadata = dense_lookup[doc_id]

            results.append({
                "doc_id": doc_id,
                "rrf_score": rrf_score,
                "rank_dense": rank_d,
                "rank_sparse": rank_s,
                "dense_score": dense_score,
                "doc_text": doc_text,
                "metadata": metadata,
            })

        # Sort by RRF score descending
        results.sort(key=lambda x: x["rrf_score"], reverse=True)
        results = results[:top_k]

        # Create RetrievalResult objects (batch fetch for sparse-only hits)
        final_results = []
        
        # Collect sparse-only doc_ids that need fetching
        sparse_only_ids = [r["doc_id"] for r in results if not r["doc_text"]]
        
        # Batch fetch sparse-only documents from ChromaDB
        sparse_lookup = {}
        if sparse_only_ids and self.indexer.collection:
            try:
                fetched = self.indexer.collection.get(ids=sparse_only_ids, include=["documents", "metadatas"])
                if fetched["documents"]:
                    for doc_id, doc_text, meta in zip(fetched["ids"], fetched["documents"], fetched["metadatas"]):
                        sparse_lookup[doc_id] = (doc_text, meta or {})
            except Exception as e:
                logger.warning(f"Failed to batch fetch sparse-only documents: {e}")

        # Create RetrievalResult objects
        for result in results:
            doc_id = result["doc_id"]
            doc_text = result["doc_text"]
            metadata = result["metadata"]

            # Use batched fetch for sparse-only hits
            if not doc_text and doc_id in sparse_lookup:
                doc_text, metadata = sparse_lookup[doc_id]

            # Create citation
            chunk_data = {
                "doc_id": doc_id,
                "text": doc_text,
                "metadata": metadata,
                "source": metadata.get("source", "unknown"),
            }
            citation = Citation.from_chunk(chunk_data, result["rrf_score"], "fused")

            retrieval_result = RetrievalResult(
                doc_id=doc_id,
                text=doc_text,
                metadata=metadata,
                dense_score=result["dense_score"],
                sparse_score=None,  # BM25 score not directly stored
                fused_score=result["rrf_score"],
                rank_dense=result["rank_dense"],
                rank_sparse=result["rank_sparse"],
                citation=citation,
            )

            final_results.append(retrieval_result)
            self.citation_tracker.add_citation(citation)

        return final_results

    def retrieve_with_citations(
        self,
        query: str,
        top_k: Optional[int] = None,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> Tuple[List[RetrievalResult], CitationTracker]:
        """Retrieve and return results with citation tracker."""
        self.citation_tracker.clear()
        results = self.retrieve(query, top_k, metadata_filter)
        return results, self.citation_tracker


def weighted_rrf_fusion(
    dense_results: List[Tuple[str, float]],
    sparse_results: List[Tuple[str, float]],
    dense_weight: float = 0.7,
    sparse_weight: float = 0.3,
    k: int = 60,
) -> List[Tuple[str, float]]:
    """Standalone Weighted RRF fusion function."""
    dense_ranked = {doc_id: rank + 1 for rank, (doc_id, _) in enumerate(dense_results)}
    sparse_ranked = {doc_id: rank + 1 for rank, (doc_id, _) in enumerate(sparse_results)}

    all_doc_ids = set(dense_ranked.keys()) | set(sparse_ranked.keys())

    fused = []
    for doc_id in all_doc_ids:
        rank_d = dense_ranked.get(doc_id)
        rank_s = sparse_ranked.get(doc_id)

        score = 0.0
        if rank_d is not None:
            score += dense_weight * (1.0 / (k + rank_d))
        if rank_s is not None:
            score += sparse_weight * (1.0 / (k + rank_s))

        fused.append((doc_id, score))

    fused.sort(key=lambda x: x[1], reverse=True)
    return fused