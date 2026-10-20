"""Indexing pipeline: embed corpus and build hybrid index."""

import argparse
from pathlib import Path

import numpy as np

from src.config import get_settings
from src.logging_config import setup_logging, get_logger
from src.embeddings.embedder import Embedder, embed_corpus_file, iter_jsonl_chunks
from src.embeddings.hybrid_indexer import HybridIndexer
from src.embeddings.model_loader import load_embedding_model

logger = get_logger(__name__)


def run_embedding(settings=None) -> int:
    """Generate embeddings for unified corpus."""
    if settings is None:
        settings = get_settings()

    embedder = Embedder(settings=settings)
    input_path = settings.paths.unified_corpus / "corpus.jsonl"
    output_path = settings.paths.data_embeddings / "corpus_embeddings.npy"

    if not input_path.exists():
        logger.error(f"Corpus file not found: {input_path}")
        return 0

    count = embed_corpus_file(input_path, output_path, embedder, settings=settings)
    return count


def run_indexing(settings=None) -> int:
    """Build hybrid index from corpus and embeddings (streaming)."""
    if settings is None:
        settings = get_settings()

    indexer = HybridIndexer(settings=settings)
    indexer.initialize()

    # Load unified corpus
    corpus_path = settings.paths.unified_corpus / "corpus.jsonl"
    emb_path = settings.paths.data_embeddings / "corpus_embeddings.npy"

    if not corpus_path.exists():
        logger.error(f"Corpus not found: {corpus_path}")
        return 0

    if not emb_path.exists():
        logger.error(f"Embeddings not found: {emb_path}")
        return 0

    # Stream chunks and embeddings in batches
    batch_size = settings.embedding.batch_size
    chunks_iter = iter_jsonl_chunks(corpus_path)
    embeddings_memmap = np.load(emb_path, mmap_mode='r')

    if len(list(iter_jsonl_chunks(corpus_path))) != len(embeddings_memmap):
        logger.error("Chunk/embedding count mismatch")
        return 0

    # Re-iterate since we consumed the iterator
    chunks_iter = iter_jsonl_chunks(corpus_path)
    embeddings_memmap = np.load(emb_path, mmap_mode='r')

    batch_chunks = []
    batch_embeddings = []
    total_indexed = 0

    for idx, chunk in enumerate(chunks_iter):
        batch_chunks.append(chunk)
        batch_embeddings.append(embeddings_memmap[idx])

        if len(batch_chunks) >= settings.embedding.batch_size:
            indexer.index_chunks(batch_chunks, np.array(batch_embeddings))
            total_indexed += len(batch_chunks)
            batch_chunks = []
            batch_embeddings = []

    # Final batch
    if batch_chunks:
        indexer.index_chunks(batch_chunks, np.array(batch_embeddings))
        total_indexed += len(batch_chunks)

    stats = indexer.get_collection_stats()
    logger.info(f"Index stats: {stats}")

    return stats["dense_count"]


def run_full_index_pipeline(settings=None) -> int:
    """Run complete indexing pipeline: embed + index."""
    if settings is None:
        settings = get_settings()

    setup_logging(settings)
    logger.info("=" * 50)
    logger.info("INDEXING PIPELINE STARTED")
    logger.info("=" * 50)

    # Step 1: Embed
    logger.info("Step 1: Generating embeddings")
    embed_count = run_embedding(settings)
    logger.info(f"Embedded {embed_count} chunks")

    # Step 2: Index
    logger.info("Step 2: Building hybrid index")
    index_count = run_indexing(settings)
    logger.info(f"Indexed {index_count} chunks")

    logger.info("=" * 50)
    logger.info("INDEXING PIPELINE COMPLETE")
    logger.info("=" * 50)

    return index_count


def main():
    parser = argparse.ArgumentParser(description="Drug Repurposing RAG - Indexing Pipeline")
    parser.add_argument("--step", choices=["embed", "index", "all"], default="all")
    parser.add_argument("--config", default="config/settings.yaml")
    args = parser.parse_args()

    settings = get_settings(args.config)

    if args.step == "embed":
        run_embedding(settings)
    elif args.step == "index":
        run_indexing(settings)
    else:
        run_full_index_pipeline(settings)


if __name__ == "__main__":
    main()