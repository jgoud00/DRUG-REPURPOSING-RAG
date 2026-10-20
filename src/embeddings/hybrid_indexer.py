"""Hybrid indexer: ChromaDB dense vectors + BM25 sparse index."""

import logging
import pickle
import re
from collections import defaultdict
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import chromadb
import numpy as np
from rank_bm25 import BM25Okapi

from src.config import get_settings

logger = logging.getLogger(__name__)


class BiomedicalTokenizer:
    """Biomedical-aware tokenizer for BM25."""

    # Pattern: alphanumeric with hyphens/slashes, preserves case-insensitive matching
    # Matches: COX-1, PDE5A, CHEMBL25, EGFR, TP53, TNF-α, IL-6, IL1/IL-1
    TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+(?:[-/][A-Za-z0-9]+)*")

    def __init__(self):
        self.pattern = self.TOKEN_PATTERN

    def tokenize(self, text: str) -> List[str]:
        """Tokenize text preserving hyphenated/slashed biomedical terms."""
        tokens = self.pattern.findall(text.lower())
        # Filter out very short tokens and pure numbers
        return [t for t in tokens if len(t) > 1 and not t.isdigit()]

    def tokenize_corpus(self, texts: List[str]) -> List[List[str]]:
        """Tokenize a corpus of texts."""
        return [self.tokenize(text) for text in texts]


class HybridIndexer:
    """Hybrid index combining ChromaDB dense vectors and BM25 sparse index."""

    def __init__(
        self,
        persist_dir: Optional[Path] = None,
        collection_name: Optional[str] = None,
        settings=None,
    ):
        self.settings = settings or get_settings()
        dense_config = self.settings.vector_store.dense
        self.persist_dir = persist_dir or Path(dense_config.get("persist_directory", "data/embeddings/chromadb"))
        self.collection_name = collection_name or dense_config.get("collection_name", "drug_repurposing")
        self.dense_client: Optional[chromadb.PersistentClient] = None
        self.collection = None
        self.bm25: Optional[BM25Okapi] = None
        self.bm25_doc_ids: List[str] = []
        self.tokenizer = BiomedicalTokenizer()
        sparse_config = self.settings.vector_store.sparse
        self._bm25_path = self.persist_dir / sparse_config.get("persist_path", "bm25_index.pkl")

    def initialize(self) -> None:
        """Initialize ChromaDB client and collection."""
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.dense_client = chromadb.PersistentClient(path=str(self.persist_dir))

        self.collection = self.dense_client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(f"Initialized ChromaDB collection: {self.collection_name}")

    def load_bm25_index(self) -> bool:
        """Load persisted BM25 index from disk."""
        if self._bm25_path.exists():
            logger.info(f"Loading BM25 index from {self._bm25_path}")
            with open(self._bm25_path, "rb") as f:
                data = pickle.load(f)
            self.bm25 = data["bm25"]
            self.bm25_doc_ids = data["doc_ids"]
            logger.info(f"Loaded BM25 index with {len(self.bm25_doc_ids)} documents")
            return True
        return False

    def save_bm25_index(self) -> None:
        """Persist BM25 index to disk."""
        if self.bm25 is not None:
            self._bm25_path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "bm25": self.bm25,
                "doc_ids": self.bm25_doc_ids,
            }
            with open(self._bm25_path, "wb") as f:
                pickle.dump(data, f)
            logger.info(f"Saved BM25 index to {self._bm25_path}")

    def index_chunks(
        self,
        chunks: List[Dict[str, Any]],
        embeddings: np.ndarray,
        batch_size: Optional[int] = None,
    ) -> None:
        """Index chunks with their embeddings (upsert for idempotency)."""
        if self.collection is None:
            self.initialize()

        batch_size = batch_size or self.settings.embedding.batch_size

        # Load existing BM25 tokenized corpus if it exists
        existing_tokenized = {}
        if self._bm25_path.exists():
            try:
                with open(self._bm25_path, "rb") as f:
                    data = pickle.load(f)
                existing_tokenized = {doc_id: tokens for doc_id, tokens in zip(data["doc_ids"], data["bm25"].corpus)}
                self.bm25_doc_ids = data["doc_ids"]
                logger.info(f"Loaded existing BM25 index with {len(self.bm25_doc_ids)} documents")
            except Exception as e:
                logger.warning(f"Failed to load existing BM25 index: {e}")

        # Process in batches for ChromaDB upsert
        total_chunks = len(chunks)
        for i in range(0, total_chunks, batch_size):
            batch_chunks = chunks[i:i + batch_size]
            batch_embeddings = embeddings[i:i + batch_size]

            ids = [chunk["doc_id"] for chunk in batch_chunks]
            documents = [chunk["text"] for chunk in batch_chunks]
            metadatas = []

            for chunk in batch_chunks:
                meta = chunk.get("metadata", {}).copy()
                clean_meta = {}
                for k, v in meta.items():
                    if isinstance(v, list):
                        clean_meta[k] = ",".join(str(x) for x in v if x)
                    elif v is None:
                        clean_meta[k] = ""
                    else:
                        clean_meta[k] = str(v)
                if "source" not in clean_meta:
                    clean_meta["source"] = chunk.get("source", "unknown")
                metadatas.append(clean_meta)

            self.collection.upsert(
                ids=ids,
                documents=documents,
                embeddings=batch_embeddings.tolist(),
                metadatas=metadatas,
            )

            # Accumulate tokenized documents for BM25
            for doc_id, doc_text in zip(ids, documents):
                tokens = self.tokenizer.tokenize(doc_text)
                existing_tokenized[doc_id] = tokens

            logger.info(f"Upserted batch {i // batch_size + 1}/{(total_chunks + batch_size - 1) // batch_size} ({len(ids)} chunks) to ChromaDB")

        # Rebuild BM25 index from all tokenized documents
        if existing_tokenized:
            self.bm25_doc_ids = list(existing_tokenized.keys())
            tokenized_corpus = list(existing_tokenized.values())
            sparse_config = self.settings.vector_store.sparse
            self.bm25 = BM25Okapi(
                tokenized_corpus,
                k1=sparse_config.get("k1", 1.2),
                b=sparse_config.get("b", 0.75),
            )
            self.save_bm25_index()
            logger.info(f"Built BM25 index with {len(self.bm25_doc_ids)} documents")

    def index_from_files(
        self,
        jsonl_files: List[Path],
        embedding_files: List[Path],
    ) -> int:
        """Index chunks from JSONL files and their corresponding embedding .npy files."""
        total = 0
        for jsonl_file, emb_file in zip(jsonl_files, embedding_files):
            if not jsonl_file.exists() or not emb_file.exists():
                logger.warning(f"Missing files: {jsonl_file} or {emb_file}")
                continue

            # Load chunks
            chunks = []
            import orjson
            with open(jsonl_file, "rb") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        chunks.append(orjson.loads(line))

            # Load embeddings
            embeddings = np.load(emb_file)

            if len(chunks) != len(embeddings):
                logger.warning(f"Mismatch: {len(chunks)} chunks vs {len(embeddings)} embeddings")
                continue

            self.index_chunks(chunks, embeddings)
            total += len(chunks)

        return total

    def dense_search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 50,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Search dense vector index."""
        if self.collection is None:
            self.initialize()

        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
            where=metadata_filter,
            include=["documents", "metadatas", "distances"],
        )

        hits = []
        if results["ids"] and results["ids"][0]:
            for doc_id, doc, meta, dist in zip(
                results["ids"][0],
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            ):
                # Convert distance to similarity (cosine: 1 - distance)
                score = 1.0 - dist
                hits.append((doc_id, score, {"text": doc, "metadata": meta}))

        return hits

    def sparse_search(
        self,
        query: str,
        top_k: int = 50,
    ) -> List[Tuple[str, float]]:
        """Search BM25 sparse index."""
        if self.bm25 is None:
            if not self.load_bm25_index():
                logger.warning("BM25 index not available")
                return []

        query_tokens = self.tokenizer.tokenize(query)
        if not query_tokens:
            return []

        scores = self.bm25.get_scores(query_tokens)
        top_indices = np.argsort(scores)[::-1][:top_k]

        hits = []
        for idx in top_indices:
            if scores[idx] > 0:
                hits.append((self.bm25_doc_ids[idx], float(scores[idx])))

        return hits

    def get_collection_stats(self) -> Dict[str, Any]:
        """Get index statistics."""
        stats = {"dense_count": 0, "sparse_count": 0}
        if self.collection:
            stats["dense_count"] = self.collection.count()
        if self.bm25_doc_ids:
            stats["sparse_count"] = len(self.bm25_doc_ids)
        return stats