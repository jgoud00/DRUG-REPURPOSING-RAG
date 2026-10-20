"""End-to-end integration test for full query-to-answer pipeline."""

import pytest
from pathlib import Path
from src.config import get_settings
from src.pipeline.query import QueryPipeline, run_query
from src.pipeline.ingest import run_full_ingestion
from src.pipeline.index import run_full_index_pipeline
from src.retrieval.reranker import CrossEncoderReranker
from src.generation.answer_synthesizer import AnswerSynthesizer


@pytest.fixture
def e2e_settings():
    """Settings for end-to-end test using fixtures."""
    settings = get_settings("config/settings.yaml")
    settings.paths.pubmed_raw = Path("tests/fixtures")
    settings.paths.chembl_raw = Path("tests/fixtures")
    settings.paths.abstracts_processed = Path("tests/output/abstracts")
    settings.paths.interactions_processed = Path("tests/output/interactions")
    settings.paths.unified_corpus = Path("tests/output/unified")
    settings.paths.data_embeddings = Path("tests/output/embeddings")
    settings.paths.logs = Path("tests/output/logs")
    settings.vector_store.dense["persist_directory"] = "tests/output/embeddings/chromadb"
    settings.vector_store.dense["collection_name"] = "test_e2e"
    settings.vector_store.sparse["persist_path"] = "tests/output/embeddings/bm25_index.pkl"
    # Use mock embeddings for fast, reliable tests
    settings.embedding.model_name = "mock"
    return settings


@pytest.fixture
def e2e_pipeline(e2e_settings):
    """Build complete pipeline for end-to-end test."""
    import shutil
    import uuid

    # Clean output
    for path in [
        e2e_settings.paths.abstracts_processed,
        e2e_settings.paths.interactions_processed,
        e2e_settings.paths.unified_corpus,
        e2e_settings.paths.data_embeddings,
    ]:
        if path.exists():
            shutil.rmtree(path, ignore_errors=True)

    # Full ingestion + indexing with mock embeddings
    run_full_ingestion(e2e_settings)

    # Index with mock embeddings
    from src.embeddings.model_loader import MockEmbeddingModel
    from src.pipeline.index import run_embedding, run_indexing
    from src.embeddings.embedder import Embedder, embed_corpus_file

    mock_model = MockEmbeddingModel(dimension=e2e_settings.embedding.model_dim)
    embedder = Embedder(model=mock_model, settings=e2e_settings)
    embed_corpus_file(
        e2e_settings.paths.unified_corpus / "corpus.jsonl",
        e2e_settings.paths.data_embeddings / "corpus_embeddings.npy",
        embedder,
        settings=e2e_settings,
    )

    # Use unique collection name
    e2e_settings.vector_store.dense["collection_name"] = f"test_e2e_{uuid.uuid4().hex[:8]}"

    run_indexing(e2e_settings)

    # Build pipeline with mock components
    pipeline = QueryPipeline(
        settings=e2e_settings,
        reranker=CrossEncoderReranker(use_mock=True, settings=e2e_settings),
        synthesizer=AnswerSynthesizer(use_mock=True, settings=e2e_settings),
    )
    return pipeline


class TestEndToEndPipeline:
    """End-to-end integration tests."""

    def test_full_query_pipeline_alzheimer(self, e2e_pipeline):
        """Test complete query pipeline for Alzheimer's disease."""
        result = e2e_pipeline.query(
            query="Alzheimer's disease PDE5A",
            top_k=10,
            use_rerank=True,
            use_pathway=True,
        )

        # Verify all stages executed
        assert len(result.retrieved_chunks) > 0
        assert len(result.reranked_chunks) > 0
        assert len(result.pathway_candidates) > 0
        assert result.synthesized_answer is not None
        assert len(result.synthesized_answer.answer) > 0

        # Verify citations present
        assert result.synthesized_answer.citation_precision >= 0.0
        assert result.synthesized_answer.hallucination_rate >= 0.0

        # Verify drug candidates found
        drug_names = [dc.get("drug_name") for dc in result.synthesized_answer.drug_candidates]
        assert "Sildenafil" in drug_names or "Metformin" in drug_names

        # Verify pathway candidates
        assert len(result.pathway_candidates) > 0
        for pc in result.pathway_candidates:
            assert pc.drug_name
            assert pc.target_name
            assert len(pc.shared_pathways) > 0

    def test_full_query_pipeline_aspirin(self, e2e_pipeline):
        """Test complete query pipeline for Aspirin/COX-1."""
        result = e2e_pipeline.query(
            query="Aspirin COX-1 inhibition",
            top_k=10,
            use_rerank=True,
            use_pathway=True,
        )

        assert len(result.retrieved_chunks) > 0
        assert len(result.reranked_chunks) > 0

        # Check for Aspirin in results
        drug_names = [dc.get("drug_name") for dc in result.synthesized_answer.drug_candidates]
        assert "Aspirin" in drug_names

        # Verify citations
        assert result.synthesized_answer.citation_precision >= 0.0
        assert result.synthesized_answer.hallucination_rate >= 0.0

    def test_pipeline_without_rerank(self, e2e_pipeline):
        """Test pipeline with reranking disabled."""
        result = e2e_pipeline.query(
            query="Aspirin COX-1",
            top_k=5,
            use_rerank=False,
            use_pathway=True,
        )

        assert len(result.retrieved_chunks) > 0
        # Without rerank, reranked should equal retrieved (or be similar)
        assert len(result.reranked_chunks) > 0
        assert result.synthesized_answer is not None

    def test_pipeline_without_pathway(self, e2e_pipeline):
        """Test pipeline with pathway linking disabled."""
        result = e2e_pipeline.query(
            query="Alzheimer's disease",
            top_k=5,
            use_rerank=True,
            use_pathway=False,
        )

        assert len(result.retrieved_chunks) > 0
        assert len(result.reranked_chunks) > 0
        # Pathway candidates should be empty
        assert len(result.pathway_candidates) == 0
        assert result.synthesized_answer is not None

    def test_pipeline_empty_query(self, e2e_pipeline):
        """Test pipeline behavior with edge case queries."""
        result = e2e_pipeline.query(
            query="nonexistent disease xyz123",
            top_k=5,
            use_rerank=True,
            use_pathway=True,
        )

        # Should handle gracefully
        assert result.synthesized_answer is not None
        # May have low retrieval but should not crash

    def test_multiple_queries_same_pipeline(self, e2e_pipeline):
        """Test that pipeline can handle multiple queries."""
        queries = [
            "Aspirin COX-1",
            "Sildenafil PDE5A Alzheimer",
            "Metformin AMPK",
            "Thalidomide Cereblon Myeloma",
        ]

        for query in queries:
            result = e2e_pipeline.query(query=query, top_k=5, use_rerank=True, use_pathway=True)
            assert result.synthesized_answer is not None
            assert len(result.synthesized_answer.answer) > 0


class TestRunQueryConvenience:
    """Test the run_query convenience function."""

    def test_run_query_direct(self, e2e_settings):
        """Test run_query function directly."""
        from src.pipeline.query import run_query

        result = run_query(
            query="Aspirin COX-1",
            top_k=5,
            use_rerank=True,
            use_pathway=True,
            use_mock=True,
            settings=e2e_settings,
        )

        assert result.synthesized_answer is not None
        assert len(result.synthesized_answer.answer) > 0


class TestAPIModels:
    """Test FastAPI request/response models."""

    def test_query_request_validation(self):
        """Test QueryRequest model validation."""
        from src.api.app import QueryRequest

        # Valid request
        req = QueryRequest(query="Alzheimer's disease")
        assert req.query == "Alzheimer's disease"
        assert req.top_k is None
        assert req.use_rerank is True

        # Custom top_k
        req = QueryRequest(query="test", top_k=20)
        assert req.top_k == 20

    def test_query_response_serialization(self):
        """Test QueryResponse model can be serialized."""
        from src.api.app import QueryResponse, CitationResponse, DrugCandidateResponse

        citation = CitationResponse(
            doc_id="test",
            source="chembl",
            text_snippet="test",
            retrieval_score=0.9,
            drug_id="CHEMBL25",
        )
        drug = DrugCandidateResponse(
            drug_name="Aspirin",
            drug_id="CHEMBL25",
        )

        response = QueryResponse(
            query="test",
            answer="test answer",
            citations=[citation],
            drug_candidates=[drug],
            pathway_candidates=[],
            hallucination_rate=0.0,
            citation_precision=1.0,
            num_retrieved=5,
            num_reranked=3,
        )

        # Should be JSON serializable
        import json
        json_str = response.model_dump_json()
        assert "test answer" in json_str
        assert "Aspirin" in json_str


if __name__ == "__main__":
    pytest.main([__file__, "-v"])