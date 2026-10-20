"""Unit tests for merger."""

import pytest
import tempfile
from pathlib import Path

from src.ingestion.merger import CorpusMerger, write_jsonl, read_jsonl
from src.ingestion.chunker import Chunk


@pytest.fixture
def sample_chunks():
    return [
        {
            "doc_id": "pubmed_123_chunk_0",
            "source": "pubmed",
            "text": "Aspirin inhibits COX-1 in patients with cardiovascular disease.",
            "metadata": {"pmid": "123", "title": "Aspirin study", "mesh_terms": "Aspirin,COX-1"},
        },
        {
            "doc_id": "chembl_CHEMBL25_CHEMBL240",
            "source": "chembl",
            "text": "Aspirin (CHEMBL25) inhibits COX-1 (CHEMBL240, P23219) implicated in Myocardial Infarction.",
            "metadata": {"drug_id": "CHEMBL25", "target_uniprot": "P23219", "confidence_score": 9.0},
        },
    ]


def test_write_jsonl(sample_chunks):
    """Test writing chunks to JSONL."""
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
        temp_path = Path(f.name)

    try:
        count = write_jsonl(sample_chunks, temp_path)
        assert count == 2

        # Read back
        read_chunks = list(read_jsonl(temp_path))
        assert len(read_chunks) == 2
        assert read_chunks[0]["doc_id"] == "pubmed_123_chunk_0"
        assert read_chunks[1]["source"] == "chembl"
    finally:
        temp_path.unlink(missing_ok=True)


def test_read_jsonl(sample_chunks):
    """Test reading chunks from JSONL."""
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
        temp_path = Path(f.name)

    try:
        write_jsonl(sample_chunks, temp_path)
        chunks = list(read_jsonl(temp_path))
        assert len(chunks) == 2
    finally:
        temp_path.unlink(missing_ok=True)


def test_corpus_merger_merge_files():
    """Test merging multiple JSONL files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        abstracts_dir = tmpdir / "abstracts"
        interactions_dir = tmpdir / "interactions"
        abstracts_dir.mkdir()
        interactions_dir.mkdir()

        # Create abstracts file
        abs_file = abstracts_dir / "pubmed_chunks.jsonl"
        write_jsonl([{"doc_id": "a1", "source": "pubmed", "text": "Abstract 1", "metadata": {}}], abs_file)

        # Create interactions file
        int_file = interactions_dir / "chembl_chunks.jsonl"
        write_jsonl([{"doc_id": "c1", "source": "chembl", "text": "Interaction 1", "metadata": {}}], int_file)

        # Merge
        output_path = tmpdir / "unified" / "corpus.jsonl"
        merger = CorpusMerger()
        count = merger.merge_files(abstracts_dir, interactions_dir, output_path)

        assert count == 2

        # Verify merged content
        merged = list(read_jsonl(output_path))
        assert len(merged) == 2
        sources = {c["source"] for c in merged}
        assert sources == {"pubmed", "chembl"}


def test_corpus_merger_merge_chunks(sample_chunks):
    """Test merging in-memory chunks."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "corpus.jsonl"

        abstract_chunks = [sample_chunks[0]]
        interaction_chunks = [sample_chunks[1]]

        merger = CorpusMerger()
        count = merger.merge_chunks(abstract_chunks, interaction_chunks, output_path)

        assert count == 2

        merged = list(read_jsonl(output_path))
        assert len(merged) == 2