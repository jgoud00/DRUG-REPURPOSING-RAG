"""Integration tests for full ingestion pipeline."""

import pytest
from pathlib import Path
import shutil

from src.config import get_settings
from src.pipeline.ingest import run_pubmed_ingestion, run_chembl_ingestion, run_merger


@pytest.fixture
def test_settings():
    """Create settings pointing to test fixtures."""
    settings = get_settings("config/settings.yaml")
    # Override paths to use test fixtures
    settings.paths.pubmed_raw = Path("tests/fixtures")
    settings.paths.chembl_raw = Path("tests/fixtures")
    settings.paths.abstracts_processed = Path("tests/output/abstracts")
    settings.paths.interactions_processed = Path("tests/output/interactions")
    settings.paths.unified_corpus = Path("tests/output/unified")
    return settings


def test_full_ingestion_pipeline(test_settings):
    """Test complete ingestion pipeline with fixtures."""
    # Clean output directories
    for path in [
        test_settings.paths.abstracts_processed,
        test_settings.paths.interactions_processed,
        test_settings.paths.unified_corpus,
    ]:
        if path.exists():
            shutil.rmtree(path)

    # Step 1: PubMed ingestion
    pubmed_chunks = run_pubmed_ingestion(test_settings)
    assert len(pubmed_chunks) > 0

    # Verify output files created
    abs_files = list(test_settings.paths.abstracts_processed.glob("*.jsonl"))
    assert len(abs_files) > 0

    # Step 2: ChEMBL ingestion
    chembl_chunks = run_chembl_ingestion(test_settings)
    assert len(chembl_chunks) > 0

    int_files = list(test_settings.paths.interactions_processed.glob("*.jsonl"))
    assert len(int_files) > 0

    # Step 3: Merger
    total = run_merger(test_settings)
    assert total == len(pubmed_chunks) + len(chembl_chunks)

    # Verify unified corpus
    corpus_file = test_settings.paths.unified_corpus / "corpus.jsonl"
    assert corpus_file.exists()

    # Read and verify corpus
    import orjson
    with open(corpus_file, "r") as f:
        lines = f.readlines()
    assert len(lines) == total

    # Check both sources present
    sources = set()
    for line in lines:
        chunk = orjson.loads(line)
        sources.add(chunk["source"])
    assert sources == {"pubmed", "chembl"}


def test_pubmed_chunks_have_required_fields(test_settings):
    """Test that PubMed chunks have all required metadata."""
    pubmed_chunks = run_pubmed_ingestion(test_settings)

    for chunk in pubmed_chunks:
        assert "doc_id" in chunk
        assert chunk["source"] == "pubmed"
        assert "text" in chunk
        assert "metadata" in chunk

        meta = chunk["metadata"]
        assert "pmid" in meta
        assert "title" in meta
        assert "mesh_terms" in meta
        assert "entities" in meta
        assert "chunk_idx" in meta


def test_chembl_chunks_have_required_fields(test_settings):
    """Test that ChEMBL chunks have all required metadata."""
    chembl_chunks = run_chembl_ingestion(test_settings)

    for chunk in chembl_chunks:
        assert "doc_id" in chunk
        assert chunk["source"] == "chembl"
        assert "text" in chunk
        assert "metadata" in chunk

        meta = chunk["metadata"]
        assert "drug_id" in meta
        assert "drug_name" in meta
        assert "target_id" in meta
        assert "target_name" in meta
        assert "target_uniprot" in meta
        assert "action_type" in meta
        assert "disease_name" in meta
        assert "confidence_score" in meta


def test_known_interactions_present(test_settings):
    """Test that known drug-target-disease triples are in the corpus."""
    run_pubmed_ingestion(test_settings)
    run_chembl_ingestion(test_settings)
    run_merger(test_settings)

    corpus_file = test_settings.paths.unified_corpus / "corpus.jsonl"
    import orjson
    with open(corpus_file, "r") as f:
        chunks = [orjson.loads(line) for line in f]

    # Check for Aspirin -> COX-1
    aspirin_chunks = [c for c in chunks if "Aspirin" in c.get("text", "") and "COX-1" in c.get("text", "")]
    assert len(aspirin_chunks) > 0

    # Check for Sildenafil -> PDE5A -> Alzheimer's
    sildenafil_chunks = [c for c in chunks if "Sildenafil" in c.get("text", "") and "PDE5A" in c.get("text", "")]
    assert len(sildenafil_chunks) > 0

    # Check for Metformin -> AMPK -> Alzheimer's
    metformin_chunks = [c for c in chunks if "Metformin" in c.get("text", "") and "AMPK" in c.get("text", "")]
    assert len(metformin_chunks) > 0

    # Check for Thalidomide -> Cereblon -> Myeloma
    thalidomide_chunks = [c for c in chunks if "Thalidomide" in c.get("text", "") and "Cereblon" in c.get("text", "")]
    assert len(thalidomide_chunks) > 0


def test_unified_corpus_jsonl_format(test_settings):
    """Test that unified corpus is valid JSONL."""
    run_pubmed_ingestion(test_settings)
    run_chembl_ingestion(test_settings)
    run_merger(test_settings)

    corpus_file = test_settings.paths.unified_corpus / "corpus.jsonl"
    import orjson
    with open(corpus_file, "r") as f:
        for i, line in enumerate(f):
            line = line.strip()
            assert line, f"Empty line at {i}"
            chunk = orjson.loads(line)
            assert "doc_id" in chunk
            assert "source" in chunk
            assert "text" in chunk
            assert "metadata" in chunk