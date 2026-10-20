"""Reactome pathway linker for disease-to-drug candidate mapping."""

import csv
import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from src.config import get_settings
from src.ingestion.chembl_parser import DrugTargetInteraction

logger = logging.getLogger(__name__)


@dataclass
class PathwayLink:
    """Represents a disease-drug link via shared pathways."""
    disease_id: str
    disease_name: str
    drug_id: str
    drug_name: str
    target_id: str
    target_name: str
    target_uniprot: str
    shared_pathways: List[str]
    pathway_names: List[str]
    confidence_score: float
    mechanism: str
    action_type: str


@dataclass
class PathwayGraph:
    """In-memory graph for pathway-based linking."""
    # disease_id -> {pathway_id}
    disease_pathways: Dict[str, Set[str]] = field(default_factory=lambda: defaultdict(set))
    # pathway_id -> pathway_name
    pathway_names: Dict[str, str] = field(default_factory=dict)
    # uniprot_id -> {pathway_id}
    target_pathways: Dict[str, Set[str]] = field(default_factory=lambda: defaultdict(set))
    # uniprot_id -> target_name
    target_names: Dict[str, str] = field(default_factory=dict)


class PathwayLinker:
    """Link diseases to drug candidates via shared biological pathways."""

    def __init__(
        self,
        pathway_tsv: Optional[Path] = None,
        settings=None,
    ):
        self.settings = settings or get_settings()
        self.pathway_tsv = pathway_tsv or Path(self.settings.pathway_linker.disease_pathway_map)
        self.graph = PathwayGraph()
        self._interactions: List[DrugTargetInteraction] = []
        self._interaction_map: Dict[str, List[DrugTargetInteraction]] = defaultdict(list)  # uniprot -> interactions
        self._drug_interactions: Dict[str, List[DrugTargetInteraction]] = defaultdict(list)  # drug_id -> interactions

    def load_pathway_mapping(self) -> int:
        """Load Reactome disease-pathway and target-pathway mappings from TSV with explicit headers."""
        if not self.pathway_tsv.exists():
            logger.warning(f"Pathway mapping file not found: {self.pathway_tsv}")
            return 0

        count = 0
        with open(self.pathway_tsv, "r") as f:
            reader = csv.reader(f, delimiter="\t")
            header = None
            for line_num, row in enumerate(reader, 1):
                if not row:
                    continue
                if row[0].startswith("#"):
                    # Check for explicit header: # entity_type	entity_id	entity_name	pathway_id	pathway_name
                    if header is None:
                        first_col = row[0].lstrip("#").strip()
                        if first_col in ("entity_type", "entity_id"):
                            # Parse header: # entity_type	entity_id	entity_name	pathway_id	pathway_name
                            header = [col.strip() for col in row]
                            continue
                    continue

                if len(row) < 4:
                    logger.warning(f"Line {line_num}: insufficient columns (expected >=4, got {len(row)}): {row}")
                    continue

                # Support both old format (4 cols) and new format (5 cols with entity_type)
                if len(row) >= 5 and header:
                    entity_type, entity_id, entity_name, pathway_id, pathway_name = row[:5]
                elif len(row) >= 4:
                    # Legacy format: infer entity type from ID prefix
                    entity_id, entity_name, pathway_id, pathway_name = row[:4]
                    entity_type = self._infer_entity_type(entity_id)
                else:
                    logger.warning(f"Line {line_num}: insufficient columns: {row}")
                    continue

                if not entity_type or not entity_id or not pathway_id:
                    logger.warning(f"Line {line_num}: missing required fields: {row}")
                    continue

                count += 1

                if entity_type == "disease":
                    self.graph.disease_pathways[entity_id].add(pathway_id)
                    self.graph.pathway_names[pathway_id] = pathway_name
                elif entity_type == "target":
                    self.graph.target_pathways[entity_id].add(pathway_id)
                    self.graph.target_names[entity_id] = entity_name
                    self.graph.pathway_names[pathway_id] = pathway_name
                else:
                    logger.warning(f"Line {line_num}: unknown entity type '{entity_type}': {row}")
                    continue

        logger.info(f"Loaded {count} pathway mappings")
        logger.info(f"  Diseases: {len(self.graph.disease_pathways)}")
        logger.info(f"  Targets: {len(self.graph.target_pathways)}")
        logger.info(f"  Pathways: {len(self.graph.pathway_names)}")
        return count

    def _infer_entity_type(self, entity_id: str) -> str:
        """Infer entity type from ID prefix (legacy support)."""
        if entity_id.startswith(("EFO", "MONDO", "DOID", "HP", "ORPHA")):
            return "disease"
        if re.match(r'^[OPQ][0-9][A-Z0-9]{3}[0-9]$', entity_id) or \
           re.match(r'^[A-NR-Z][0-9](?:[A-Z][A-Z0-9]{2}[0-9]){1,2}$', entity_id):
            return "target"
        logger.warning(f"Cannot infer entity type for ID: {entity_id}")
        return "unknown"

    def load_interactions(self, interactions: List[DrugTargetInteraction]) -> None:
        """Load drug-target interactions for pathway-based linking."""
        self._interactions = interactions
        self._interaction_map.clear()
        self._drug_interactions.clear()

        for interaction in interactions:
            uniprot = interaction.target_uniprot
            if uniprot:
                self._interaction_map[uniprot].append(interaction)
            self._drug_interactions[interaction.drug_id].append(interaction)

        logger.info(f"Loaded {len(interactions)} interactions for pathway linking")
        logger.info(f"  Unique targets (UniProt): {len(self._interaction_map)}")
        logger.info(f"  Unique drugs: {len(self._drug_interactions)}")

    def find_candidate_drugs(
        self,
        disease_id: str,
        disease_name: str = "",
        min_shared_pathways: int = 1,
        top_k: int = 20,
    ) -> List[PathwayLink]:
        """Find drug candidates for a disease via shared pathways."""
        if not self.graph.disease_pathways.get(disease_id):
            logger.warning(f"No pathway mapping for disease: {disease_id}")
            return []

        disease_pathways = self.graph.disease_pathways[disease_id]
        if not disease_name:
            disease_name = disease_id

        candidates = []

        # For each target with pathway info, check overlap
        for uniprot, target_pathways in self.graph.target_pathways.items():
            shared = disease_pathways & target_pathways
            if len(shared) < min_shared_pathways:
                continue

            # Get interactions for this target
            interactions = self._interaction_map.get(uniprot, [])
            for interaction in interactions:
                pathway_names = [self.graph.pathway_names.get(p, p) for p in shared]

                link = PathwayLink(
                    disease_id=disease_id,
                    disease_name=disease_name,
                    drug_id=interaction.drug_id,
                    drug_name=interaction.drug_name,
                    target_id=interaction.target_id,
                    target_name=interaction.target_name,
                    target_uniprot=uniprot,
                    shared_pathways=list(shared),
                    pathway_names=pathway_names,
                    confidence_score=interaction.confidence_score,
                    mechanism=interaction.mechanism,
                    action_type=interaction.action_type,
                )
                candidates.append(link)

        # Deduplicate by drug (keep highest confidence)
        drug_best = {}
        for link in candidates:
            key = (link.drug_id, link.target_uniprot)
            if key not in drug_best or link.confidence_score > drug_best[key].confidence_score:
                drug_best[key] = link

        # Sort by: shared pathways count, confidence score
        sorted_links = sorted(
            drug_best.values(),
            key=lambda x: (len(x.shared_pathways), x.confidence_score),
            reverse=True,
        )

        return sorted_links[:top_k]

    def get_pathway_info(self, pathway_id: str) -> Optional[str]:
        """Get pathway name by ID."""
        return self.graph.pathway_names.get(pathway_id)

    def get_disease_pathways(self, disease_id: str) -> Set[str]:
        """Get pathway IDs for a disease."""
        return self.graph.disease_pathways.get(disease_id, set())

    def get_target_pathways(self, uniprot: str) -> Set[str]:
        """Get pathway IDs for a target protein."""
        return self.graph.target_pathways.get(uniprot, set())

    def rank_repurposing_candidates(
        self,
        disease_id: str,
        disease_name: str = "",
        retrieval_results: Optional[List] = None,  # For PubMed support
    ) -> List[PathwayLink]:
        """Rank repurposing candidates combining pathway, confidence, and retrieval evidence."""
        pathway_links = self.find_candidate_drugs(disease_id, disease_name, top_k=50)

        if not retrieval_results:
            return pathway_links

        # Enhance with retrieval support (PubMed co-occurrence)
        # Create a simple lookup: (drug_name, target_name) -> retrieval count
        retrieval_support = defaultdict(int)
        for result in retrieval_results:
            meta = result.metadata
            drug = meta.get("drug_name", "").lower()
            target = meta.get("target_name", "").lower()
            if drug and target:
                retrieval_support[(drug, target)] += 1

        # Re-rank with retrieval support
        for link in pathway_links:
            key = (link.drug_name.lower(), link.target_name.lower())
            link.retrieval_support = retrieval_support.get(key, 0)

        # Final ranking: shared pathways > confidence > retrieval support
        pathway_links.sort(
            key=lambda x: (
                len(x.shared_pathways),
                x.confidence_score,
                getattr(x, 'retrieval_support', 0),
            ),
            reverse=True,
        )

        return pathway_links


def load_pathway_linker_from_corpus(
    corpus_path: Path,
    pathway_tsv: Optional[Path] = None,
    settings=None,
) -> PathwayLinker:
    """Load pathway linker with interactions from corpus."""
    from src.ingestion.merger import read_jsonl
    from src.ingestion.chembl_parser import DrugTargetInteraction

    linker = PathwayLinker(pathway_tsv=pathway_tsv, settings=settings)
    linker.load_pathway_mapping()

    # Extract ChEMBL interactions from corpus
    interactions = []
    for chunk in read_jsonl(corpus_path):
        if chunk.get("source") == "chembl":
            meta = chunk.get("metadata", {})
            interaction = DrugTargetInteraction(
                drug_id=meta.get("drug_id", ""),
                drug_name=meta.get("drug_name", ""),
                target_id=meta.get("target_id", ""),
                target_name=meta.get("target_name", ""),
                target_uniprot=meta.get("target_uniprot", ""),
                action_type=meta.get("action_type", ""),
                mechanism=meta.get("mechanism", ""),
                disease_id=meta.get("disease_id", ""),
                disease_name=meta.get("disease_name", ""),
                pathway_ids=meta.get("pathway_ids", "").split(",") if meta.get("pathway_ids") else [],
                pathway_names=meta.get("pathway_names", "").split(",") if meta.get("pathway_names") else [],
                confidence_score=meta.get("confidence_score", 0.0),
                source=meta.get("source", "chembl"),
            )
            if interaction.target_uniprot:
                interactions.append(interaction)

    linker.load_interactions(interactions)
    return linker