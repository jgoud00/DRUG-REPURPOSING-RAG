"""Unit tests for hybrid retriever."""

import pytest
import numpy as np

from src.embeddings.hybrid_indexer import HybridIndexer
from src.embeddings.model_loader import MockEmbeddingModel
from src.retrieval.hybrid_retriever import HybridRetriever, weighted_rrf_fusion
from src.retrieval.citation_tracker import CitationTracker, Citation


class TestWeightedRRF:
    """Test weighted RRF fusion."""

    def test_rrf_basic(self):
        """Test basic RRF fusion."""
        dense = [("doc1", 0.9), ("doc2", 0.8), ("doc3", 0.7)]
        sparse = [("doc2", 10.0), ("doc4", 8.0), ("doc1", 5.0)]

        fused = weighted_rrf_fusion(dense, sparse, dense_weight=0.7, sparse_weight=0.3, k=60)

        # doc1 and doc2 should be in top results
        doc_ids = [d for d, _ in fused]
        assert "doc1" in doc_ids[:2]
        assert "doc2" in doc_ids[:2]

    def test_rrf_empty(self):
        """Test RRF with empty inputs."""
        assert weighted_rrf_fusion([], []) == []
        # Use approximate equality for floating point
        result = weighted_rrf_fusion([("a", 1.0)], [])
        assert result == [("a", 0.7/61)] or abs(result[0][1] - 0.7/61) < 1e-10
        result = weighted_rrf_fusion([], [("b", 1.0)])
        assert result == [("b", 0.3/61)] or abs(result[0][1] - 0.3/61) < 1e-10


class TestCitationTracker:
    """Test citation tracker."""

    def test_add_citation(self):
        """Test adding citations."""
        tracker = CitationTracker()
        citation = Citation(
            doc_id="test_1",
            source="pubmed",
            text_snippet="Aspirin inhibits COX-1",
            retrieval_score=0.9,
            pmid="12345",
        )
        tracker.add_citation(citation)

        assert len(tracker.citations) == 1
        assert tracker.get_citation("test_1") == citation

    def test_format_citations(self):
        """Test formatting citations."""
        tracker = CitationTracker(citation_format="[{doc_id}]")
        tracker.add_citation(Citation(doc_id="a", source="pubmed", text_snippet="", retrieval_score=1))
        tracker.add_citation(Citation(doc_id="b", source="chembl", text_snippet="", retrieval_score=1))

        formatted = tracker.format_citations(["a", "b"])
        assert formatted == "[a] [b]"

    def test_from_chunk_pubmed(self):
        """Test creating citation from PubMed chunk."""
        chunk = {
            "doc_id": "pubmed_123_chunk_0",
            "source": "pubmed",
            "text": "Aspirin resistance study...",
            "metadata": {
                "pmid": "12345",
                "title": "Aspirin study",
                "mesh_terms": "Aspirin,COX-1",
            },
        }
        citation = Citation.from_chunk(chunk, 0.85)
        assert citation.pmid == "12345"
        assert citation.source == "pubmed"

    def test_from_chunk_chembl(self):
        """Test creating citation from ChEMBL chunk."""
        chunk = {
            "doc_id": "chembl_CHEMBL25_CHEMBL240",
            "source": "chembl",
            "text": "Aspirin inhibits COX-1...",
            "metadata": {
                "drug_id": "CHEMBL25",
                "drug_name": "Aspirin",
                "target_id": "CHEMBL240",
                "target_name": "COX-1",
                "target_uniprot": "P23219",
                "action_type": "inhibitor",
                "confidence_score": 9.0,
            },
        }
        citation = Citation.from_chunk(chunk, 0.9)
        assert citation.drug_id == "CHEMBL25"
        assert citation.target_uniprot == "P23219"
        assert citation.confidence_score == 9.0


class TestHybridRetriever:
    """Test hybrid retriever."""

    @pytest.fixture
    def retriever(self, tmp_path):
        """Create retriever with mock model and test index."""
        indexer = HybridIndexer(persist_dir=tmp_path / "chromadb", collection_name="test")
        indexer.initialize()

        # Add test chunks
        chunks = [
            {
                "doc_id": "pubmed_1",
                "text": "Aspirin inhibits COX-1 in cardiovascular disease",
                "metadata": {"source": "pubmed", "pmid": "123", "title": "Aspirin study"},
            },
            {
                "doc_id": "chembl_1",
                "text": "Aspirin (CHEMBL25) inhibits COX-1 (P23219) implicated in Myocardial Infarction",
                "metadata": {
                    "source": "chembl",
                    "drug_id": "CHEMBL25",
                    "drug_name": "Aspirin",
                    "target_id": "CHEMBL240",
                    "target_name": "COX-1",
                    "target_uniprot": "P23219",
                    "action_type": "inhibitor",
                    "confidence_score": 9.0,
                },
            },
            {
                "doc_id": "chembl_2",
                "text": "Sildenafil (CHEMBL132) inhibits PDE5A (O76083) implicated in Alzheimer Disease",
                "metadata": {
                    "source": "chembl",
                    "drug_id": "CHEMBL132",
                    "drug_name": "Sildenafil",
                    "target_id": "CHEMBL3150",
                    "target_name": "PDE5A",
                    "target_uniprot": "O76083",
                    "action_type": "inhibitor",
                    "confidence_score": 7.0,
                },
            },
        ]
        embeddings = np.random.randn(3, 768).astype(np.float32)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        indexer.index_chunks(chunks, embeddings)

        model = MockEmbeddingModel(dimension=768)
        retriever = HybridRetriever(indexer=indexer, embedder_model=model)
        return retriever

    def test_retrieve_basic(self, retriever):
        """Test basic retrieval."""
        results = retriever.retrieve("Aspirin COX-1", top_k=3)

        assert len(results) > 0
        # Should find both aspirin-related chunks
        doc_ids = [r.doc_id for r in results]
        assert "pubmed_1" in doc_ids
        assert "chembl_1" in doc_ids

    def test_retrieve_with_filter(self, retriever):
        """Test retrieval with metadata filter."""
        results = retriever.retrieve("Aspirin", top_k=3, metadata_filter={"source": "chembl"})

        assert len(results) > 0
        for r in results:
            assert r.metadata.get("source") == "chembl"

    def test_retrieve_citation_tracking(self, retriever):
        """Test that citations are tracked."""
        results = retriever.retrieve("Aspirin COX-1", top_k=3)

        assert len(retriever.citation_tracker.citations) == len(results)
        for result in results:
            assert result.citation is not None
            assert result.citation.doc_id == result.doc_id

    def test_retrieve_fused_scores(self, retriever):
        """Test that fused scores are computed."""
        results = retriever.retrieve("Sildenafil PDE5A Alzheimer", top_k=3)

        for r in results:
            assert r.fused_score > 0
            assert r.rank_dense is not None or r.rank_sparse is not None


class TestHybridRetrieverIntegration:
    """Integration-style tests using the full pipeline."""

    def test_query_alzheimer_pde5a(self, tmp_path):
        """Test query for Alzheimer's + PDE5A retrieves Sildenafil."""
        indexer = HybridIndexer(persist_dir=tmp_path / "chromadb", collection_name="test")
        indexer.initialize()

        chunks = [
            {
                "doc_id": "chembl_sildenafil",
                "text": "Sildenafil (CHEMBL132) inhibits PDE5A (O76083) implicated in Alzheimer Disease via competitive PDE5A inhibition",
                "metadata": {
                    "source": "chembl",
                    "drug_id": "CHEMBL132",
                    "drug_name": "Sildenafil",
                    "target_name": "PDE5A",
                    "target_uniprot": "O76083",
                    "disease_name": "Alzheimer Disease",
                    "action_type": "inhibitor",
                },
            },
            {
                "doc_id": "pubmed_alz",
                "text": "Sildenafil reduced amyloid beta by 45% via PDE5A inhibition in Alzheimer's disease models",
                "metadata": {"source": "pubmed", "pmid": "38412346", "title": "Drug repurposing for Alzheimer's"},
            },
        ]
        embeddings = np.random.randn(2, 768).astype(np.float32)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        indexer.index_chunks(chunks, embeddings)

        model = MockEmbeddingModel(dimension=768)
        retriever = HybridRetriever(indexer=indexer, embedder_model=model)

        results = retriever.retrieve("Alzheimer's disease PDE5A", top_k=5)

        assert len(results) >= 1
        # Should find both chunks
        doc_ids = [r.doc_id for r in results]
        assert "chembl_sildenafil" in doc_ids
        assert "pubmed_alz" in doc_ids