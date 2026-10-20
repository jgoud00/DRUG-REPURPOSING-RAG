"""Citation tracker for provenance tracking."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any


@dataclass
class Citation:
    """Structured citation for a retrieved chunk."""
    doc_id: str
    source: str  # "pubmed" or "chembl"
    text_snippet: str
    retrieval_score: float
    rerank_score: Optional[float] = None
    fused_score: Optional[float] = None
    # Source-specific metadata
    pmid: Optional[str] = None
    title: Optional[str] = None
    mesh_terms: Optional[str] = None
    drug_id: Optional[str] = None
    drug_name: Optional[str] = None
    target_id: Optional[str] = None
    target_name: Optional[str] = None
    target_uniprot: Optional[str] = None
    action_type: Optional[str] = None
    mechanism: Optional[str] = None
    disease_id: Optional[str] = None
    disease_name: Optional[str] = None
    pathway_ids: Optional[str] = None
    pathway_names: Optional[str] = None
    confidence_score: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, excluding None values."""
        return {k: v for k, v in self.__dict__.items() if v is not None}

    def format_citation(self, format_str: str = "[{doc_id}]") -> str:
        """Format citation string."""
        return format_str.format(doc_id=self.doc_id)

    @classmethod
    def from_chunk(cls, chunk: Dict[str, Any], score: float, score_type: str = "retrieval") -> "Citation":
        """Create citation from retrieved chunk."""
        meta = chunk.get("metadata", {})
        source = chunk.get("source", "unknown")

        citation = cls(
            doc_id=chunk.get("doc_id", ""),
            source=source,
            text_snippet=chunk.get("text", "")[:500],  # Truncate for display
            retrieval_score=score,
        )

        if source == "pubmed":
            citation.pmid = meta.get("pmid")
            citation.title = meta.get("title")
            citation.mesh_terms = meta.get("mesh_terms")
        elif source == "chembl":
            citation.drug_id = meta.get("drug_id")
            citation.drug_name = meta.get("drug_name")
            citation.target_id = meta.get("target_id")
            citation.target_name = meta.get("target_name")
            citation.target_uniprot = meta.get("target_uniprot")
            citation.action_type = meta.get("action_type")
            citation.mechanism = meta.get("mechanism")
            citation.disease_id = meta.get("disease_id")
            citation.disease_name = meta.get("disease_name")
            citation.pathway_ids = meta.get("pathway_ids")
            citation.pathway_names = meta.get("pathway_names")
            citation.confidence_score = meta.get("confidence_score")

        return citation


class CitationTracker:
    """Track and manage citations for retrieved results."""

    def __init__(self, citation_format: str = "[{doc_id}]"):
        self.citation_format = citation_format
        self.citations: List[Citation] = []
        self._doc_id_to_citation: Dict[str, Citation] = {}

    def add_citation(self, citation: Citation) -> None:
        """Add a citation to the tracker."""
        self.citations.append(citation)
        self._doc_id_to_citation[citation.doc_id] = citation

    def add_citations(self, citations: List[Citation]) -> None:
        """Add multiple citations."""
        for c in citations:
            self.add_citation(c)

    def get_citation(self, doc_id: str) -> Optional[Citation]:
        """Get citation by doc_id."""
        return self._doc_id_to_citation.get(doc_id)

    def get_all_citations(self) -> List[Citation]:
        """Get all citations."""
        return self.citations

    def format_citations(self, doc_ids: List[str]) -> str:
        """Format multiple citations as a string."""
        return " ".join(
            self._doc_id_to_citation[doc_id].format_citation(self.citation_format)
            for doc_id in doc_ids
            if doc_id in self._doc_id_to_citation
        )

    def clear(self) -> None:
        """Clear all citations."""
        self.citations.clear()
        self._doc_id_to_citation.clear()

    def to_dict_list(self) -> List[Dict[str, Any]]:
        """Export all citations as list of dicts."""
        return [c.to_dict() for c in self.citations]