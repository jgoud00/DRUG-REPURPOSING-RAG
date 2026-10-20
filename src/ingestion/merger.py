"""Merge PubMed abstracts and ChEMBL interactions into unified corpus."""

import json
from pathlib import Path
from typing import Iterator

import orjson

from src.config import get_settings
from src.logging_config import get_logger

logger = get_logger(__name__)


class CorpusMerger:
    """Merge chunked abstracts and structured data into unified JSONL corpus."""

    def __init__(self, settings=None):
        self.settings = settings or get_settings()
        self.config = self.settings.chunking

    def merge_files(
        self,
        abstracts_dir: Path,
        interactions_dir: Path,
        output_path: Path,
    ) -> int:
        """Merge all chunked files into unified corpus."""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        count = 0
        with open(output_path, "wb") as f:
            # Process abstracts
            for jsonl_file in abstracts_dir.glob("*.jsonl"):
                logger.info(f"Merging abstracts from: {jsonl_file}")
                with open(jsonl_file, "r") as inf:
                    for line in inf:
                        if line.strip():
                            f.write(line.encode("utf-8"))
                            count += 1

            # Process interactions
            for jsonl_file in interactions_dir.glob("*.jsonl"):
                logger.info(f"Merging interactions from: {jsonl_file}")
                with open(jsonl_file, "r") as inf:
                    for line in inf:
                        if line.strip():
                            f.write(line.encode("utf-8"))
                            count += 1

        logger.info(f"Merged {count} total chunks into {output_path}")
        return count

    def merge_chunks(
        self,
        abstract_chunks: list[dict],
        interaction_chunks: list[dict],
        output_path: Path,
    ) -> int:
        """Merge in-memory chunk lists into unified corpus."""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        count = 0
        with open(output_path, "wb") as f:
            for chunk in abstract_chunks:
                f.write(orjson.dumps(chunk) + b"\n")
                count += 1
            for chunk in interaction_chunks:
                f.write(orjson.dumps(chunk) + b"\n")
                count += 1

        logger.info(f"Merged {count} total chunks into {output_path}")
        return count


def write_jsonl(chunks: list[dict], output_path: Path) -> int:
    """Write chunks to JSONL file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(output_path, "wb") as f:
        for chunk in chunks:
            f.write(orjson.dumps(chunk) + b"\n")
            count += 1
    return count


def read_jsonl(input_path: Path) -> Iterator[dict]:
    """Read chunks from JSONL file."""
    with open(input_path, "r") as f:
        for line in f:
            if line.strip():
                yield orjson.loads(line)