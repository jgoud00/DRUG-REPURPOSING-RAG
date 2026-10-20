#!/usr/bin/env python
"""CLI entry point for Drug Repurposing RAG Engine."""

import argparse
import json
import sys
from pathlib import Path

from src.config import get_settings
from src.logging_config import setup_logging
from src.pipeline.query import run_query


def main():
    parser = argparse.ArgumentParser(
        description="Drug Repurposing RAG Engine - Query Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m scripts.run_pipeline "Alzheimer's disease PDE5A"
  python -m scripts.run_pipeline "Aspirin COX-1" --top-k 10 --no-rerank
  python -m scripts.run_pipeline "Multiple Myeloma" --use-mock --json
        """,
    )

    parser.add_argument("query", help="Disease or target query string")
    parser.add_argument("--top-k", type=int, default=None, help="Number of results to retrieve")
    parser.add_argument("--no-rerank", action="store_true", help="Disable cross-encoder reranking")
    parser.add_argument("--no-pathway", action="store_true", help="Disable pathway linking")
    parser.add_argument("--use-mock", action="store_true", help="Use mock LLM/embeddings (no external services)")
    parser.add_argument("--json", action="store_true", help="Output JSON instead of formatted text")
    parser.add_argument("--config", default="config/settings.yaml", help="Path to config file")
    parser.add_argument("--output", help="Output file path (default: stdout)")

    args = parser.parse_args()

    # Load settings
    settings = get_settings(args.config)

    # Setup logging
    setup_logging(settings)

    # Run query
    try:
        result = run_query(
            query=args.query,
            top_k=args.top_k,
            use_rerank=not args.no_rerank,
            use_pathway=not args.no_pathway,
            use_mock=args.use_mock,
            settings=settings,
        )

        # Format output
        if args.json:
            output = format_json_output(result)
        else:
            output = format_text_output(result)

        # Write output
        if args.output:
            Path(args.output).write_text(output, encoding="utf-8")
            print(f"Output written to {args.output}")
        else:
            print(output)

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def format_text_output(result) -> str:
    """Format result as human-readable text."""
    lines = [
        f"Query: {result.query}",
        f"Retrieved: {len(result.retrieved_chunks)} chunks",
        f"Reranked: {len(result.reranked_chunks)} chunks",
        f"Pathway candidates: {len(result.pathway_candidates)}",
        "",
        "=== ANSWER ===",
        result.synthesized_answer.answer,
        "",
        "=== METRICS ===",
        f"Citation Precision: {result.synthesized_answer.citation_precision:.2%}",
        f"Hallucination Rate: {result.synthesized_answer.hallucination_rate:.2%}",
        "",
    ]

    if result.pathway_candidates:
        lines.extend([
            "=== PATHWAY CANDIDATES ===",
        ])
        for pc in result.pathway_candidates:
            lines.append(
                f"  - {pc.drug_name} ({pc.drug_id}) -> {pc.target_name} ({pc.target_uniprot}) "
                f"[{len(pc.shared_pathways)} shared pathways] conf={pc.confidence_score:.1f}"
            )

    lines.extend([
        "",
        "=== CITATIONS ===",
    ])
    for c in result.synthesized_answer.citations:
        lines.append(f"  [{c.doc_id}] {c.source} - {c.text_snippet[:80]}...")

    return "\n".join(lines)


def format_json_output(result) -> str:
    """Format result as JSON."""
    import dataclasses

    def to_dict(obj):
        if dataclasses.is_dataclass(obj):
            return dataclasses.asdict(obj)
        elif isinstance(obj, list):
            return [to_dict(item) for item in obj]
        elif isinstance(obj, dict):
            return {k: to_dict(v) for k, v in obj.items()}
        return obj

    return json.dumps(to_dict(result), indent=2)


if __name__ == "__main__":
    main()