"""Unit tests for embeddings module."""

import pytest
import numpy as np

from src.embeddings.model_loader import MockEmbeddingModel, SentenceTransformerEmbeddingModel, load_embedding_model
from src.embeddings.embedder import Embedder, iter_jsonl_chunks
from src.embeddings.hybrid_indexer import HybridIndexer, BiomedicalTokenizer


class TestMockEmbeddingModel:
    """Test mock embedding model."""

    def test_mock_embedding_deterministic(self):
        """Test that mock embeddings are deterministic."""
        model = MockEmbeddingModel(dimension=768)
        texts = ["Aspirin inhibits COX-1", "Sildenafil inhibits PDE5A"]

        emb1 = model.encode(texts)
        emb2 = model.encode(texts)

        assert emb1.shape == (2, 768)
        assert np.allclose(emb1, emb2)

    def test_mock_embedding_normalized(self):
        """Test that mock embeddings are normalized."""
        model = MockEmbeddingModel(dimension=768)
        emb = model.encode(["test text"])
        norms = np.linalg.norm(emb, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-6)

    def test_mock_embedding_dimension(self):
        """Test mock embedding dimension."""
        for dim in [384, 768, 1024]:
            model = MockEmbeddingModel(dimension=dim)
            assert model.get_dimension() == dim
            emb = model.encode(["test"])
            assert emb.shape == (1, dim)


class TestBiomedicalTokenizer:
    """Test biomedical-aware tokenizer."""

    def test_tokenize_genes(self):
        """Test tokenization of gene/protein symbols."""
        tokenizer = BiomedicalTokenizer()
        text = "EGFR and TP53 and COX-1 are important proteins."
        tokens = tokenizer.tokenize(text)
        assert "egfr" in tokens
        assert "tp53" in tokens
        assert "cox-1" in tokens

    def test_tokenize_chembl_ids(self):
        """Test tokenization of ChEMBL IDs."""
        tokenizer = BiomedicalTokenizer()
        text = "CHEMBL25 and CHEMBL132 are drug IDs."
        tokens = tokenizer.tokenize(text)
        assert "chembl25" in tokens
        assert "chembl132" in tokens

    def test_tokenize_preserves_hyphens(self):
        """Test that hyphenated terms are preserved."""
        tokenizer = BiomedicalTokenizer()
        text = "COX-1 COX-2 PDE5A BACE1"
        tokens = tokenizer.tokenize(text)
        assert "cox-1" in tokens
        assert "cox-2" in tokens
        assert "pde5a" in tokens
        assert "bace1" in tokens

    def test_tokenize_filters_short(self):
        """Test that very short tokens are filtered."""
        tokenizer = BiomedicalTokenizer()
        text = "a b c COX-1"
        tokens = tokenizer.tokenize(text)
        assert "a" not in tokens
        assert "b" not in tokens
        assert "c" not in tokens
        assert "cox-1" in tokens


class TestHybridIndexer:
    """Test hybrid indexer."""

    def test_indexer_initialization(self, tmp_path):
        """Test indexer initialization."""
        indexer = HybridIndexer(persist_dir=tmp_path / "chromadb", collection_name="test")
        indexer.initialize()
        assert indexer.collection is not None
        stats = indexer.get_collection_stats()
        assert stats["dense_count"] == 0

    def test_upsert_chunks(self, tmp_path):
        """Test upserting chunks."""
        indexer = HybridIndexer(persist_dir=tmp_path / "chromadb", collection_name="test")
        indexer.initialize()

        chunks = [
            {"doc_id": "test_1", "text": "Aspirin inhibits COX-1", "metadata": {"source": "chembl"}},
            {"doc_id": "test_2", "text": "Sildenafil inhibits PDE5A", "metadata": {"source": "chembl"}},
        ]
        embeddings = np.random.randn(2, 768).astype(np.float32)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        indexer.index_chunks(chunks, embeddings)

        stats = indexer.get_collection_stats()
        assert stats["dense_count"] == 2
        assert stats["sparse_count"] == 2

    def test_dense_search(self, tmp_path):
        """Test dense vector search."""
        indexer = HybridIndexer(persist_dir=tmp_path / "chromadb", collection_name="test")
        indexer.initialize()

        chunks = [
            {"doc_id": "test_1", "text": "Aspirin inhibits COX-1", "metadata": {"source": "chembl"}},
            {"doc_id": "test_2", "text": "Sildenafil inhibits PDE5A", "metadata": {"source": "chembl"}},
        ]
        embeddings = np.random.randn(2, 768).astype(np.float32)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        indexer.index_chunks(chunks, embeddings)

        # Search with first embedding
        hits = indexer.dense_search(embeddings[0], top_k=2)
        assert len(hits) == 2
        assert hits[0][0] == "test_1"  # Should be most similar to itself

    def test_sparse_search(self, tmp_path):
        """Test BM25 sparse search."""
        indexer = HybridIndexer(persist_dir=tmp_path / "chromadb", collection_name="test")
        indexer.initialize()

        chunks = [
            {"doc_id": "test_1", "text": "Aspirin inhibits COX-1 uniquely", "metadata": {"source": "chembl"}},
            {"doc_id": "test_2", "text": "Sildenafil inhibits PDE5A specifically", "metadata": {"source": "chembl"}},
            {"doc_id": "test_3", "text": "Metformin activates AMPK pathway", "metadata": {"source": "chembl"}},
        ]
        embeddings = np.random.randn(3, 768).astype(np.float32)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        indexer.index_chunks(chunks, embeddings)

        hits = indexer.sparse_search("COX-1 uniquely", top_k=2)
        assert len(hits) >= 1
        assert hits[0][0] == "test_1"  # Should match COX-1 uniquely


class TestEmbedder:
    """Test embedder."""

    def test_embed_chunks(self):
        """Test embedding chunks with mock model."""
        model = MockEmbeddingModel(dimension=768)
        embedder = Embedder(model=model, batch_size=2)

        chunks = [
            {"doc_id": "1", "text": "Aspirin inhibits COX-1"},
            {"doc_id": "2", "text": "Sildenafil inhibits PDE5A"},
        ]
        embeddings = embedder.embed_chunks(chunks)

        assert embeddings.shape == (2, 768)

    def test_embed_query(self):
        """Test embedding a single query."""
        model = MockEmbeddingModel(dimension=768)
        embedder = Embedder(model=model)

        query_emb = embedder.embed_query("Alzheimer's disease PDE5A")
        assert query_emb.shape == (768,)


class TestLoadEmbeddingModel:
    """Test embedding model factory."""

    def test_load_mock_model(self):
        """Test loading mock model."""
        model = load_embedding_model(use_mock=True)
        assert isinstance(model, MockEmbeddingModel)
        assert model.get_dimension() == 768