"""Full ingestion pipeline orchestrator."""

import argparse
from pathlib import Path

from src.config import get_settings
from src.logging_config import setup_logging, get_logger
from src.ingestion.pubmed_parser import PubMedParser, parse_pubmed_file
from src.ingestion.chembl_parser import ChEMBLParser, parse_chembl_directory
from src.ingestion.chunker import chunk_pubmed_abstracts, chunk_chembl_interactions
from src.ingestion.merger import write_jsonl, CorpusMerger

logger = get_logger(__name__)


def run_pubmed_ingestion(settings=None) -> list[dict]:
    """Run PubMed ingestion: parse -> chunk -> write."""
    if settings is None:
        settings = get_settings()

    parser = PubMedParser(settings)
    chunker_config = settings.chunking

    raw_dir = settings.paths.pubmed_raw
    processed_dir = settings.paths.abstracts_processed
    processed_dir.mkdir(parents=True, exist_ok=True)

    all_chunks = []

    # Parse all PubMed files
    for filepath in raw_dir.glob("*.xml"):
        logger.info(f"Processing PubMed file: {filepath}")
        abstracts = parse_pubmed_file(filepath, settings)
        logger.info(f"Parsed {len(abstracts)} abstracts from {filepath.name}")

        # Convert to dict and chunk
        abstract_dicts = [a.to_dict() for a in abstracts]
        chunks = chunk_pubmed_abstracts(abstract_dicts, settings)
        logger.info(f"Created {len(chunks)} chunks from {len(abstracts)} abstracts")

        # Write per-file chunks
        output_file = processed_dir / f"{filepath.stem}_chunks.jsonl"
        write_jsonl([c.to_dict() for c in chunks], output_file)
        all_chunks.extend([c.to_dict() for c in chunks])

    for filepath in raw_dir.glob("*.json"):
        logger.info(f"Processing PubMed file: {filepath}")
        abstracts = parse_pubmed_file(filepath, settings)
        logger.info(f"Parsed {len(abstracts)} abstracts from {filepath.name}")

        abstract_dicts = [a.to_dict() for a in abstracts]
        chunks = chunk_pubmed_abstracts(abstract_dicts, settings)
        logger.info(f"Created {len(chunks)} chunks from {len(abstracts)} abstracts")

        output_file = processed_dir / f"{filepath.stem}_chunks.jsonl"
        write_jsonl([c.to_dict() for c in chunks], output_file)
        all_chunks.extend([c.to_dict() for c in chunks])

    return all_chunks


def run_chembl_ingestion(settings=None) -> list[dict]:
    """Run ChEMBL ingestion: parse -> chunk -> write."""
    if settings is None:
        settings = get_settings()

    raw_dir = settings.paths.chembl_raw
    processed_dir = settings.paths.interactions_processed
    processed_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading ChEMBL tables from: {raw_dir}")
    tables = ChEMBLParser(settings).load_tables(raw_dir)

    if not any(not df.empty for df in tables.values()):
        logger.warning("No ChEMBL tables found or all empty")
        return []

    interactions = list(ChEMBLParser(settings).parse_interactions(tables))
    logger.info(f"Parsed {len(interactions)} drug-target interactions")

    interaction_dicts = [i.to_dict() for i in interactions]
    chunks = chunk_chembl_interactions(interaction_dicts, settings)
    logger.info(f"Created {len(chunks)} chunks from {len(interactions)} interactions")

    output_file = processed_dir / "chembl_interactions_chunks.jsonl"
    write_jsonl([c.to_dict() for c in chunks], output_file)

    return [c.to_dict() for c in chunks]


def run_merger(settings=None) -> int:
    """Merge all processed chunks into unified corpus."""
    if settings is None:
        settings = get_settings()

    merger = CorpusMerger(settings)
    output_path = settings.paths.unified_corpus / "corpus.jsonl"

    count = merger.merge_files(
        settings.paths.abstracts_processed,
        settings.paths.interactions_processed,
        output_path,
    )

    return count


def run_full_ingestion(settings=None):
    """Run complete ingestion pipeline."""
    if settings is None:
        settings = get_settings()

    setup_logging(settings)
    logger.info("Starting full ingestion pipeline")

    # PubMed
    logger.info("=" * 50)
    logger.info("STEP 1: PubMed Ingestion")
    logger.info("=" * 50)
    pubmed_chunks = run_pubmed_ingestion(settings)

    # ChEMBL
    logger.info("=" * 50)
    logger.info("STEP 2: ChEMBL Ingestion")
    logger.info("=" * 50)
    chembl_chunks = run_chembl_ingestion(settings)

    # Merge
    logger.info("=" * 50)
    logger.info("STEP 3: Corpus Merger")
    logger.info("=" * 50)
    total = run_merger(settings)

    logger.info("=" * 50)
    logger.info(f"INGESTION COMPLETE: {total} total chunks")
    logger.info(f"  PubMed chunks: {len(pubmed_chunks)}")
    logger.info(f"  ChEMBL chunks: {len(chembl_chunks)}")
    logger.info("=" * 50)

    return total


def main():
    parser = argparse.ArgumentParser(description="Drug Repurposing RAG - Ingestion Pipeline")
    parser.add_argument("--step", choices=["pubmed", "chembl", "merge", "all"], default="all")
    parser.add_argument("--config", default="config/settings.yaml")
    args = parser.parse_args()

    settings = get_settings(args.config)

    if args.step == "pubmed":
        run_pubmed_ingestion(settings)
    elif args.step == "chembl":
        run_chembl_ingestion(settings)
    elif args.step == "merge":
        run_merger(settings)
    else:
        run_full_ingestion(settings)


if __name__ == "__main__":
    main()