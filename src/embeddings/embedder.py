"""Batch embedding generation for corpus chunks."""

import json
import logging
from pathlib import Path
from typing import Iterator, List, Optional

import numpy as np
import orjson
from tqdm import tqdm

from src.config import get_settings
from src.embeddings.model_loader import EmbeddingModel, load_embedding_model

logger = logging.getLogger(__name__)


class Embedder:
    """Generate embeddings for text chunks in batches."""

    def __init__(
        self,
        model: Optional[EmbeddingModel] = None,
        batch_size: Optional[int] = None,
        settings=None,
    ):
        self.settings = settings or get_settings()
        self.model = model or load_embedding_model(settings=self.settings)
        self.batch_size = batch_size or self.settings.embedding.batch_size

    def embed_chunks(
        self,
        chunks: List[dict],
        text_field: str = "text",
    ) -> np.ndarray:
        """Embed a list of chunk dictionaries."""
        texts = [chunk[text_field] for chunk in chunks]
        logger.info(f"Embedding {len(texts)} chunks with batch_size={self.batch_size}")

        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=True,
        )

        logger.info(f"Generated embeddings shape: {embeddings.shape}")
        return embeddings

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Embed a list of raw texts."""
        return self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
        )

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query string."""
        return self.model.encode([query], batch_size=1, show_progress_bar=False)[0]


def iter_jsonl_chunks(filepath: Path) -> Iterator[dict]:
    """Iterate over chunks in a JSONL file."""
    with open(filepath, "rb") as f:
        for line in f:
            line = line.strip()
            if line:
                yield orjson.loads(line)


def embed_corpus_file(
    input_path: Path,
    output_path: Path,
    embedder: Optional[Embedder] = None,
    text_field: str = "text",
    batch_size: Optional[int] = None,
    settings=None,
) -> int:
    """Embed all chunks in a JSONL file and save embeddings alongside (streaming)."""
    if embedder is None:
        embedder = Embedder(batch_size=batch_size, settings=settings)

    settings = settings or get_settings()
    batch_size = batch_size or embedder.batch_size

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # First pass: count chunks and collect doc_ids for metadata
    doc_ids = []
    with open(input_path, "rb") as f:
        for line in f:
            line = line.strip()
            if line:
                import orjson
                chunk = orjson.loads(line)
                doc_ids.append(chunk.get("doc_id", ""))

    num_chunks = len(doc_ids)
    if num_chunks == 0:
        logger.warning("No chunks to embed")
        return 0

    logger.info(f"Embedding {num_chunks} chunks from {input_path} in batches of {batch_size}")

    # Second pass: stream embeddings in batches, write to memmap
    dim = embedder.model.get_dimension()
    embeddings_memmap = np.lib.format.open_memmap(
        output_path, mode='w+', dtype=np.float32, shape=(num_chunks, dim)
    )

    idx = 0
    batch_texts = []
    with open(input_path, "rb") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            import orjson
            chunk = orjson.loads(line)
            batch_texts.append(chunk[text_field])

            if len(batch_texts) >= batch_size:
                batch_embeddings = embedder.model.encode(
                    batch_texts,
                    batch_size=batch_size,
                    show_progress_bar=False,
                )
                embeddings_memmap[idx:idx + len(batch_texts)] = batch_embeddings.astype(np.float32)
                idx += len(batch_texts)
                batch_texts = []

    # Final batch
    if batch_texts:
        batch_embeddings = embedder.model.encode(
            batch_texts,
            batch_size=batch_size,
            show_progress_bar=False,
        )
        embeddings_memmap[idx:idx + len(batch_texts)] = batch_embeddings.astype(np.float32)

    # Flush to disk
    embeddings_memmap.flush()

    # Save metadata
    meta_path = output_path.with_suffix('.meta.json')
    model_name = getattr(embedder.model, 'model_name', 'mock')
    meta = {
        "model_name": model_name,
        "dimension": dim,
        "num_chunks": num_chunks,
        "doc_ids": doc_ids,
        "text_field": text_field,
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    logger.info(f"Saved embeddings to {output_path} ({(num_chunks, dim)})")
    logger.info(f"Saved metadata to {meta_path}")

    return num_chunks


def embed_corpus_directory(
    input_dir: Path,
    output_dir: Path,
    pattern: str = "*.jsonl",
    embedder: Optional[Embedder] = None,
    settings=None,
) -> int:
    """Embed all JSONL files in a directory."""
    if embedder is None:
        embedder = Embedder(settings=settings)

    settings = settings or get_settings()
    total = 0

    for jsonl_file in input_dir.glob(pattern):
        if jsonl_file.suffix == ".jsonl":
            output_file = output_dir / f"{jsonl_file.stem}_embeddings.npy"
            count = embed_corpus_file(jsonl_file, output_file, embedder, settings=settings)
            total += count

    logger.info(f"Embedded {total} total chunks from {input_dir}")
    return total