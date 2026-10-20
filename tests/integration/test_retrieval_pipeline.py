"""Integration tests for retrieval pipeline (M4-M8)."""

import pytest
import numpy as np
from pathlib import Path

from src.config import get_settings
from src.logging_config import setup_logging
from src.pipeline.ingest import run_full_ingestion
from src.pipeline.index import run_full_index_pipeline
from src.embeddings.hybrid_indexer import HybridIndexer
from src.embeddings.model_loader import MockEmbeddingModel
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.reranker import CrossEncoderReranker
from src.retrieval.pathway_linker import PathwayLinker
from src.retrieval.citation_tracker import CitationTracker
from src.ingestion.chembl_parser import DrugTargetInteraction


@pytest.fixture(scope="session")
def test_settings():
    """Settings for integration tests using fixtures."""
    settings = get_settings("config/settings.yaml")
    settings.paths.pubmed_raw = Path("tests/fixtures")
    settings.paths.chembl_raw = Path("tests/fixtures")
    settings.paths.abstracts_processed = Path("tests/output/abstracts")
    settings.paths.interactions_processed = Path("tests/output/interactions")
    settings.paths.unified_corpus = Path("tests/output/unified")
    settings.paths.data_embeddings = Path("tests/output/embeddings")
    settings.paths.logs = Path("tests/output/logs")
    settings.vector_store.dense["persist_directory"] = "tests/output/embeddings/chromadb"
    settings.vector_store.sparse["persist_path"] = "tests/output/embeddings/bm25_index.pkl"
    return settings


@pytest.fixture(scope="session")
def built_index(test_settings):
    """Build index from fixtures for retrieval tests (session-scoped to avoid Windows locking)."""
    import shutil
    import uuid

    # Clean output dirs except chromadb (to avoid Windows locking)
    for path in [
        test_settings.paths.abstracts_processed,
        test_settings.paths.interactions_processed,
        test_settings.paths.unified_corpus,
        test_settings.paths.data_embeddings,
    ]:
        if path.exists():
            shutil.rmtree(path, ignore_errors=True)

    # Run ingestion + indexing with mock embeddings
    run_full_ingestion(test_settings)

    mock_model = MockEmbeddingModel(dimension=test_settings.embedding.model_dim)

    from src.pipeline.index import run_embedding, run_indexing
    from src.embeddings.embedder import Embedder, embed_corpus_file

    embedder = Embedder(model=mock_model, settings=test_settings)
    embed_corpus_file(
        test_settings.paths.unified_corpus / "corpus.jsonl",
        test_settings.paths.data_embeddings / "corpus_embeddings.npy",
        embedder,
        settings=test_settings,
    )

    # Use unique collection name to avoid conflicts
    test_settings.vector_store.dense["collection_name"] = f"test_drug_repurposing_{uuid.uuid4().hex[:8]}"

    run_indexing(test_settings)

    indexer = HybridIndexer(settings=test_settings)
    indexer.initialize()
    indexer.load_bm25_index()

    retriever = HybridRetriever(indexer=indexer, embedder_model=mock_model, settings=test_settings)
    return retriever


class TestRetrievalPipeline:
    """Test end-to-end retrieval pipeline."""

    def test_query_asp_cox1(self, built_index):
        """Test query for Aspirin COX-1."""
        results = built_index.retrieve("Aspirin COX-1 inhibition", top_k=20)

        assert len(results) > 0
        doc_ids = [r.doc_id for r in results]
        # Should find both PubMed and ChEMBL aspirin chunks
        assert any("pubmed" in d and "38412345" in d for d in doc_ids)
        # ChEMBL target_id is "240" (tid), not "CHEMBL240"
        assert any("chembl" in d and "CHEMBL25" in d and "240" in d for d in doc_ids)

    def test_query_alzheimer_pde5a(self, built_index):
        """Test query for Alzheimer's + PDE5A."""
        results = built_index.retrieve("Alzheimer's disease PDE5A", top_k=5)

        assert len(results) > 0
        doc_ids = [r.doc_id for r in results]
        # Should find Sildenafil ChEMBL + PubMed abstract
        assert any("chembl" in d and "CHEMBL132" in d for d in doc_ids)
        assert any("pubmed" in d and "38412346" in d for d in doc_ids)

    def test_query_metformin_ampk(self, built_index):
        """Test query for Metformin AMPK."""
        results = built_index.retrieve("Metformin AMPK activation", top_k=20)

        assert len(results) > 0
        doc_ids = [r.doc_id for r in results]
        assert any("chembl" in d and "CHEMBL1100" in d for d in doc_ids)

    def test_query_thalidomide_cereblon(self, built_index):
        """Test query for Thalidomide Cereblon Myeloma."""
        results = built_index.retrieve("Thalidomide Cereblon Multiple Myeloma", top_k=20)

        assert len(results) > 0
        doc_ids = [r.doc_id for r in results]
        assert any("chembl" in d and "CHEMBL382" in d for d in doc_ids)

    def test_citation_tracking(self, built_index):
        """Test citation tracking works."""
        results = built_index.retrieve("Aspirin COX-1", top_k=3)

        # Clear tracker before test to avoid accumulation from other tests
        built_index.citation_tracker.clear()
        results = built_index.retrieve("Aspirin COX-1", top_k=3)

        assert len(built_index.citation_tracker.citations) == len(results)
        for result in results:
            assert result.citation is not None
            assert result.citation.doc_id == result.doc_id
            assert result.citation.source in ("pubmed", "chembl")

    def test_citation_formatting(self, built_index):
        """Test citation formatting."""
        built_index.citation_tracker.clear()
        results = built_index.retrieve("Aspirin COX-1", top_k=3)
        formatted = built_index.citation_tracker.format_citations([r.doc_id for r in results])

        assert len(formatted) > 0
        assert "[" in formatted and "]" in formatted


class TestRerankingIntegration:
    """Test reranking integration."""

    def test_rerank_alzheimer_query(self, built_index):
        """Test reranking for Alzheimer's query."""
        retriever = built_index
        results = retriever.retrieve("Alzheimer's disease PDE5A", top_k=5)

        reranker = CrossEncoderReranker(use_mock=True)
        reranked = reranker.rerank("Alzheimer's disease PDE5A", results, top_k=3)

        assert len(reranked) == 3
        assert all(hasattr(r, 'rerank_score') for r in reranked)
        # Sildenafil should be top after reranking
        assert "Sildenafil" in reranked[0].text or "PDE5A" in reranked[0].text


class TestPathwayLinkingIntegration:
    """Test pathway linking integration."""

    @pytest.fixture
    def linker_with_corpus(self, built_index, test_settings):
        """Create pathway linker with interactions loaded from built corpus."""
        linker = PathwayLinker(
            pathway_tsv=Path("tests/fixtures/reactome_disease_pathways.tsv")
        )
        linker.load_pathway_mapping()

        from src.ingestion.merger import read_jsonl
        from src.ingestion.chembl_parser import DrugTargetInteraction

        interactions = []
        corpus_path = test_settings.paths.unified_corpus / "corpus.jsonl"
        if corpus_path.exists():
            for chunk in read_jsonl(corpus_path):
                if chunk.get("source") == "chembl":
                    meta = chunk.get("metadata", {})
                    interaction = DrugTargetInteraction(
                        drug_id=meta.get("drug_id", ""),
                        drug_name=meta.get("drug_name", ""),
                        target_id=meta.get("target_id", ""),
                        target_name=meta.get("target_name", ""),
                        target_uniprot=meta.get("target_uniprot", ""),
                        action_type=meta.get("action_type", ""),
                        mechanism=meta.get("mechanism", ""),
                        disease_id=meta.get("disease_id", ""),
                        disease_name=meta.get("disease_name", ""),
                        pathway_ids=meta.get("pathway_ids", "").split(",") if meta.get("pathway_ids") else [],
                        pathway_names=meta.get("pathway_names", "").split(",") if meta.get("pathway_names") else [],
                        confidence_score=meta.get("confidence_score", 0.0),
                        source=meta.get("source", "chembl"),
                    )
                    if interaction.target_uniprot:
                        interactions.append(interaction)

        linker.load_interactions(interactions)
        return linker

    def test_pathway_linker_alzheimer(self, linker_with_corpus):
        """Test pathway linker finds Sildenafil/Metformin for Alzheimer's."""
        candidates = linker_with_corpus.find_candidate_drugs(
            disease_id="EFO_0000249",
            disease_name="Alzheimer's disease",
            min_shared_pathways=1,
            top_k=10,
        )

        assert len(candidates) > 0
        drug_names = [c.drug_name for c in candidates]
        assert "Sildenafil" in drug_names
        assert "Metformin" in drug_names

    def test_pathway_linker_myeloma(self, linker_with_corpus):
        """Test pathway linker finds Thalidomide for Myeloma."""
        candidates = linker_with_corpus.find_candidate_drugs(
            disease_id="EFO_0000305",
            disease_name="Multiple Myeloma",
            min_shared_pathways=1,
            top_k=10,
        )

        assert len(candidates) > 0
        drug_names = [c.drug_name for c in candidates]
        assert "Thalidomide" in drug_names


class TestFullPipelineQueries:
    """Test known query patterns from the plan."""

    def test_known_repurposing_cases(self, built_index):
        """Test known drug repurposing cases are retrievable."""
        test_cases = [
            ("Aspirin COX-1 cardiovascular", "Aspirin", "COX-1"),
            ("Sildenafil PDE5A Alzheimer", "Sildenafil", "PDE5A"),
            ("Metformin AMPK Alzheimer", "Metformin", "AMPK"),
            ("Thalidomide Cereblon Myeloma", "Thalidomide", "Cereblon"),
            ("Rofecoxib COX-2 osteoarthritis", "Rofecoxib", "COX-2"),
            ("Celecoxib COX-2", "Celecoxib", "COX-2"),
        ]

        for query, drug, target in test_cases:
            results = built_index.retrieve(query, top_k=5)
            doc_ids = [r.doc_id for r in results]
            texts = " ".join([r.text for r in results])

            # At least one result should mention the drug and target
            found = any(drug.lower() in t.lower() and target.lower() in t.lower() for t in [r.text for r in results])
            assert found, f"Query '{query}' should retrieve {drug} + {target}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])