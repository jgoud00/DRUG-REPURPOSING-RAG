"""Answer synthesis with citation verification."""

import logging
import re
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

from src.config import get_settings
from src.generation.llm_client import LLMClient, create_llm_client
from src.retrieval.hybrid_retriever import RetrievalResult
from src.retrieval.citation_tracker import Citation

logger = logging.getLogger(__name__)


@dataclass
class SynthesizedAnswer:
    """Final synthesized answer with verified citations."""
    answer: str
    citations: List[Citation]
    drug_candidates: List[Dict[str, Any]] = field(default_factory=list)
    hallucination_rate: float = 0.0
    citation_precision: float = 1.0
    unverified_claims: List[str] = field(default_factory=list)


class AnswerSynthesizer:
    """Synthesize final answer from retrieved chunks with citation verification."""

    CITATION_PATTERN = re.compile(r'\[(PMID|CHEMBL):\s*([^\]]+)\]')

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        use_mock: bool = False,
        settings=None,
    ):
        self.settings = settings or get_settings()
        self.llm_client = llm_client or create_llm_client(use_mock=use_mock, settings=self.settings)
        self.citation_format = self.settings.generation.prompt.citation_format
        self.require_citations = self.settings.generation.prompt.require_citations
        self.max_context_chunks = self.settings.generation.prompt.max_context_chunks

        # Load prompt templates
        self.system_template = self._load_template(self.settings.generation.prompt.system_template)
        self.user_template = self._load_template(self.settings.generation.prompt.user_template)

    def _load_template(self, path: str) -> str:
        """Load prompt template from file."""
        from pathlib import Path
        template_path = Path(path)
        if template_path.exists():
            return template_path.read_text(encoding="utf-8")
        # Fallback inline templates
        if "system" in path:
            return (
                "You are a biomedical research assistant. Answer using only provided context. "
                "Cite every claim with [PMID: <id>] or [CHEMBL: <id>]."
            )
        return "Query: {query}\n\nContext:\n{context}\n\nAnswer with citations."

    def _format_context(self, results: List[RetrievalResult]) -> str:
        """Format retrieved chunks for LLM context."""
        context_parts = []
        for i, result in enumerate(results[:self.max_context_chunks]):
            source = result.metadata.get("source", "unknown")
            if source == "pubmed":
                pmid = result.metadata.get("pmid", "unknown")
                identifier = f"PMID: {pmid}"
            else:
                drug_id = result.metadata.get("drug_id", "unknown")
                identifier = f"CHEMBL: {drug_id}"

            context_parts.append(
                f"[{identifier}] {result.text}"
            )
        return "\n\n".join(context_parts)

    def _extract_citations(self, text: str) -> List[tuple]:
        """Extract citation IDs from text."""
        return self.CITATION_PATTERN.findall(text)

    def _verify_citations(
        self,
        extracted_citations: List[tuple],
        retrieved_chunks: List[RetrievalResult],
    ) -> tuple:
        """Verify all citations exist in retrieved chunks."""
        available_pmids = set()
        available_chembls = set()
        available_doc_ids = set()

        for chunk in retrieved_chunks:
            meta = chunk.metadata
            if meta.get("source") == "pubmed":
                pmid = meta.get("pmid")
                if pmid:
                    available_pmids.add(str(pmid))
            elif meta.get("source") == "chembl":
                drug_id = meta.get("drug_id")
                if drug_id:
                    available_chembls.add(str(drug_id))
            # Also track full doc_id for fuzzy matching
            doc_id = chunk.doc_id
            if doc_id:
                available_doc_ids.add(doc_id)

        verified = []
        unverified = []

        for source, cid in extracted_citations:
            cid_str = str(cid).strip()
            if source == "PMID" and cid_str in available_pmids:
                verified.append((source, cid_str))
            elif source == "CHEMBL" and cid_str in available_chembls:
                verified.append((source, cid_str))
            # Fallback: check if cid appears in any doc_id (handles format variations)
            elif any(cid_str in doc_id for doc_id in available_doc_ids):
                verified.append((source, cid_str))
            else:
                unverified.append((source, cid_str))

        return verified, unverified

    def _extract_drug_candidates(self, text: str, chunks: List[RetrievalResult]) -> List[Dict[str, Any]]:
        """Extract structured drug candidates from answer text."""
        candidates = []
        # Match **Drug Name (ID)** or **Drug Name (ID)** with flexible formatting
        drug_pattern = re.compile(r'\*\*?([^*(]+?)\s*\(([^)]+)\)\*?')
        for match in drug_pattern.finditer(text):
            drug_name = match.group(1).strip()
            drug_id = match.group(2).strip()
            # Find matching chunk for metadata
            for chunk in chunks:
                meta = chunk.metadata
                if meta.get("drug_name") == drug_name or meta.get("drug_id") == drug_id:
                    candidates.append({
                        "drug_name": drug_name,
                        "drug_id": drug_id,
                        "target_name": meta.get("target_name"),
                        "target_id": meta.get("target_id"),
                        "target_uniprot": meta.get("target_uniprot"),
                        "action_type": meta.get("action_type"),
                        "mechanism": meta.get("mechanism"),
                        "disease_name": meta.get("disease_name"),
                        "confidence_score": meta.get("confidence_score"),
                    })
                    break
        return candidates

    def synthesize(
        self,
        query: str,
        retrieved_chunks: List[RetrievalResult],
    ) -> SynthesizedAnswer:
        """Generate answer with citation verification."""
        if not retrieved_chunks:
            return SynthesizedAnswer(
                answer="Insufficient evidence in provided sources.",
                citations=[],
                hallucination_rate=1.0,
                citation_precision=0.0,
                unverified_claims=["No chunks retrieved"],
            )

        # Build context
        context = self._format_context(retrieved_chunks)
        prompt = self.user_template.format(query=query, context=context)

        # Generate answer
        response = self.llm_client.generate(prompt, system_prompt=self.system_template)
        answer_text = response.text

        # Verify citations
        extracted = self._extract_citations(answer_text)
        verified, unverified = self._verify_citations(extracted, retrieved_chunks)

        # Calculate metrics
        total_citations = len(extracted)
        citation_precision = len(verified) / total_citations if total_citations > 0 else 1.0
        hallucination_rate = len(unverified) / total_citations if total_citations > 0 else 0.0

        # Build citation objects for verified ones
        citation_map = {c.doc_id: c for chunk in retrieved_chunks for c in [chunk.citation] if c}
        verified_citations = []
        for source, cid in verified:
            # Find matching citation
            for cit in citation_map.values():
                if source == "PMID" and cit.pmid == cid:
                    verified_citations.append(cit)
                    break
                elif source == "CHEMBL" and cit.drug_id == cid:
                    verified_citations.append(cit)
                    break

        # Extract drug candidates
        drug_candidates = self._extract_drug_candidates(answer_text, retrieved_chunks)

        unverified_claims = [f"[{src}: {cid}]" for src, cid in unverified]

        return SynthesizedAnswer(
            answer=answer_text,
            citations=verified_citations,
            drug_candidates=drug_candidates,
            hallucination_rate=hallucination_rate,
            citation_precision=citation_precision,
            unverified_claims=unverified_claims,
        )