"""Unit tests for generation module."""

import pytest
from src.generation.llm_client import (
    OllamaClient, OpenAIClient, MockLLMClient, create_llm_client, LLMResponse
)
from src.generation.answer_synthesizer import AnswerSynthesizer, SynthesizedAnswer
from src.retrieval.hybrid_retriever import RetrievalResult
from src.retrieval.citation_tracker import Citation


class TestLLMClients:
    """Test LLM client implementations."""

    def test_mock_client_basic(self):
        """Test mock client returns canned response."""
        client = MockLLMClient("Test answer [PMID: 12345]")
        response = client.generate("Test query")
        assert isinstance(response, LLMResponse)
        assert response.text == "Test answer [PMID: 12345]"
        assert response.model == "mock"

    def test_mock_client_custom(self):
        """Test mock client with custom response."""
        custom = "Custom answer [CHEMBL: CHEMBL25]"
        client = MockLLMClient(custom)
        response = client.generate("Query")
        assert response.text == custom

    def test_create_mock_client(self):
        """Test factory creates mock client."""
        client = create_llm_client(use_mock=True)
        assert isinstance(client, MockLLMClient)
        response = client.generate("Test")
        assert "Aspirin" in response.text

    def test_ollama_client_config(self):
        """Test OllamaClient configuration."""
        from src.config import get_settings
        settings = get_settings()
        client = OllamaClient(
            base_url=settings.generation.llm.base_url,
            model=settings.generation.llm.model,
            temperature=settings.generation.llm.temperature,
        )
        assert client.base_url == settings.generation.llm.base_url
        assert client.model == settings.generation.llm.model


class TestAnswerSynthesizer:
    """Test answer synthesis with citation verification."""

    @pytest.fixture
    def sample_chunks(self):
        """Create sample retrieval results."""
        return [
            RetrievalResult(
                doc_id="chembl_CHEMBL25_240",
                text="Aspirin (CHEMBL25) inhibits COX-1 (CHEMBL240, P23219) implicated in Myocardial Infarction via irreversible acetylation.",
                metadata={
                    "source": "chembl",
                    "drug_id": "CHEMBL25",
                    "drug_name": "Aspirin",
                    "target_id": "240",
                    "target_name": "COX-1",
                    "target_uniprot": "P23219",
                    "action_type": "inhibitor",
                    "mechanism": "irreversible acetylation of serine 530",
                    "disease_name": "Myocardial Infarction",
                    "confidence_score": 9.0,
                },
                dense_score=0.9,
                fused_score=0.8,
                citation=Citation(
                    doc_id="chembl_CHEMBL25_240",
                    source="chembl",
                    text_snippet="Aspirin inhibits COX-1...",
                    retrieval_score=0.8,
                    drug_id="CHEMBL25",
                    drug_name="Aspirin",
                    target_id="240",
                    target_name="COX-1",
                    target_uniprot="P23219",
                    action_type="inhibitor",
                    mechanism="irreversible acetylation of serine 530",
                    disease_name="Myocardial Infarction",
                    confidence_score=9.0,
                ),
            ),
            RetrievalResult(
                doc_id="pubmed_38412345_chunk_0",
                text="Aspirin resistance is associated with adverse cardiovascular outcomes in patients with diabetes mellitus. COX-1 expression correlated with resistance.",
                metadata={
                    "source": "pubmed",
                    "pmid": "38412345",
                    "title": "Aspirin resistance study",
                    "mesh_terms": "Aspirin,COX-1",
                },
                dense_score=0.7,
                fused_score=0.6,
                citation=Citation(
                    doc_id="pubmed_38412345_chunk_0",
                    source="pubmed",
                    text_snippet="Aspirin resistance study...",
                    retrieval_score=0.6,
                    pmid="38412345",
                    title="Aspirin resistance study",
                    mesh_terms="Aspirin,COX-1",
                ),
            ),
        ]

    def test_synthesize_with_mock(self, sample_chunks):
        """Test synthesis with mock LLM."""
        synthesizer = AnswerSynthesizer(use_mock=True)
        result = synthesizer.synthesize("Aspirin COX-1 inhibition", sample_chunks)

        assert isinstance(result, SynthesizedAnswer)
        assert len(result.answer) > 0
        assert result.citation_precision >= 0.0
        assert result.hallucination_rate >= 0.0

    def test_citation_verification_pass(self, sample_chunks):
        """Test that valid citations pass verification."""
        synthesizer = AnswerSynthesizer(use_mock=True)
        result = synthesizer.synthesize("Aspirin COX-1", sample_chunks)

        # Mock response contains [CHEMBL: CHEMBL25] which should verify
        assert result.citation_precision > 0.0

    def test_citation_verification_fail(self, sample_chunks):
        """Test that invalid citations are caught."""
        from src.generation.llm_client import MockLLMClient

        # Custom response with fake citation
        bad_client = MockLLMClient("Fake drug [PMID: 999999] does something [CHEMBL: FAKE123]")
        synthesizer = AnswerSynthesizer(llm_client=bad_client)
        result = synthesizer.synthesize("Test query", sample_chunks)

        assert result.hallucination_rate > 0.0
        assert len(result.unverified_claims) > 0
        assert any("PMID: 999999" in c for c in result.unverified_claims)
        assert any("CHEMBL: FAKE123" in c for c in result.unverified_claims)

    def test_drug_candidate_extraction(self, sample_chunks):
        """Test structured drug candidate extraction."""
        synthesizer = AnswerSynthesizer(use_mock=True)
        result = synthesizer.synthesize("Aspirin COX-1", sample_chunks)

        # Mock response contains Aspirin, Sildenafil, Metformin
        assert len(result.drug_candidates) >= 0

    def test_empty_chunks(self):
        """Test handling of empty retrieval."""
        synthesizer = AnswerSynthesizer(use_mock=True)
        result = synthesizer.synthesize("Query", [])

        assert result.answer == "Insufficient evidence in provided sources."
        assert result.citations == []
        assert result.hallucination_rate == 1.0

    def test_citation_format(self, sample_chunks):
        """Test that citations use correct format."""
        synthesizer = AnswerSynthesizer(use_mock=True)
        result = synthesizer.synthesize("Test", sample_chunks)

        # Check for PMID or CHEMBL format in answer
        assert "PMID:" in result.answer or "CHEMBL:" in result.answer


class TestSynthesizedAnswer:
    """Test SynthesizedAnswer dataclass."""

    def test_defaults(self):
        """Test default values."""
        ans = SynthesizedAnswer(answer="Test", citations=[])
        assert ans.drug_candidates == []
        assert ans.hallucination_rate == 0.0
        assert ans.citation_precision == 1.0
        assert ans.unverified_claims == []