"""ChEMBL data parser for drug-target interactions."""

import csv
import gzip
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional

import pandas as pd

from src.config import get_settings
from src.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class DrugTargetInteraction:
    """Unified drug-target-disease interaction record."""
    drug_id: str
    drug_name: str
    target_id: str
    target_name: str
    target_uniprot: str
    action_type: str
    mechanism: str
    disease_id: str
    disease_name: str
    pathway_ids: list[str] = field(default_factory=list)
    pathway_names: list[str] = field(default_factory=list)
    confidence_score: float = 0.0
    source: str = "chembl"

    def to_dict(self) -> dict:
        return {
            "drug_id": self.drug_id,
            "drug_name": self.drug_name,
            "target_id": self.target_id,
            "target_name": self.target_name,
            "target_uniprot": self.target_uniprot,
            "action_type": self.action_type,
            "mechanism": self.mechanism,
            "disease_id": self.disease_id,
            "disease_name": self.disease_name,
            "pathway_ids": self.pathway_ids,
            "pathway_names": self.pathway_names,
            "confidence_score": self.confidence_score,
            "source": self.source,
        }


class ChEMBLParser:
    """Parse ChEMBL CSV exports into unified drug-target interactions."""

    def __init__(self, settings=None):
        self.settings = settings or get_settings()
        self.config = self.settings.ingestion.chembl
        self._cache = {}

    def load_tables(self, data_dir: Path) -> dict[str, pd.DataFrame]:
        """Load all required ChEMBL tables from CSV files."""
        # Required columns per table (ChEMBL 33)
        required_columns = {
            "activities": ["molregno", "tid", "confidence_score"],
            "target_dictionary": ["tid", "pref_name", "target_type"],
            "compound_records": ["molregno", "chembl_id", "pref_name"],
            "mechanism_of_action": ["molregno", "tid", "action_type"],
            "target_components": ["tid", "component_id"],
        }

        tables = {}
        for table_name in self.config.tables:
            filepath = data_dir / f"sample_chembl_{table_name}.csv"
            if not filepath.exists():
                filepath = data_dir / f"{table_name}.csv"
            if not filepath.exists():
                filepath = data_dir / f"{table_name}.csv.gz"
            if filepath.exists():
                logger.info(f"Loading ChEMBL table: {table_name}")
                if filepath.suffix == ".gz":
                    df = pd.read_csv(filepath, compression="gzip", low_memory=False)
                else:
                    df = pd.read_csv(filepath, low_memory=False)

                # Validate required columns
                req_cols = required_columns.get(table_name, [])
                missing = [c for c in req_cols if c not in df.columns]
                if missing:
                    raise ValueError(f"ChEMBL table '{table_name}' missing required columns: {missing}")

                tables[table_name] = df
            else:
                logger.warning(f"ChEMBL table not found: {filepath}")
                tables[table_name] = pd.DataFrame()
        return tables

    def parse_interactions(self, tables: dict[str, pd.DataFrame]) -> Iterator[DrugTargetInteraction]:
        """Parse drug-target-disease interactions from loaded tables."""
        activities = tables.get("activities", pd.DataFrame())
        targets = tables.get("target_dictionary", pd.DataFrame())
        compounds = tables.get("compound_records", pd.DataFrame())
        mechanisms = tables.get("mechanism_of_action", pd.DataFrame())
        target_components = tables.get("target_components", pd.DataFrame())

        if activities.empty or targets.empty or compounds.empty:
            logger.warning("Missing required ChEMBL tables for interaction parsing")
            return

        # Build lookup dictionaries
        target_lookup = self._build_target_lookup(targets, target_components)
        compound_lookup = self._build_compound_lookup(compounds)
        mechanism_lookup = self._build_mechanism_lookup(mechanisms)

        # Filter activities by confidence
        if "confidence_score" in activities.columns:
            activities = activities[activities["confidence_score"] >= self.config.confidence_threshold]

        # Deduplicate by (normalized_drug_id, normalized_target_uniprot) keeping highest confidence
        # First, collect all valid interactions with their confidence scores
        interactions_by_key = {}  # (drug_id, uniprot) -> (interaction, confidence)
        
        for _, activity in activities.iterrows():
            molregno = str(activity.get("molregno", ""))
            tid = str(activity.get("tid", ""))
            if not molregno or not tid:
                continue

            compound = compound_lookup.get(molregno, {})
            target = target_lookup.get(tid, {})
            mechanism = mechanism_lookup.get((str(activity.get("molregno", "")), tid), {})

            # Normalize drug ID: prefer ChEMBL ID, don't fabricate from molregno
            drug_id = compound.get("chembl_id")
            if not drug_id:
                # Skip if no valid ChEMBL ID - we can't reliably identify this compound
                logger.debug(f"Skipping activity with molregno={molregno}: no ChEMBL ID")
                continue
            drug_name = compound.get("pref_name", "") or drug_id

            # Normalize target: use UniProt as canonical identifier
            target_uniprot = target.get("uniprot", "")
            if not target_uniprot:
                # Skip if no UniProt - we can't reliably link to pathways
                logger.debug(f"Skipping target tid={tid}: no UniProt accession")
                continue
            target_name = target.get("target_name", "") or tid
            target_id = tid  # Keep original tid for reference

            action_type = mechanism.get("action_type", "") or self._infer_action_type(activity)
            mech_text = mechanism.get("mechanism", "") or ""

            disease_id = mechanism.get("disease_effector", "") or ""
            disease_name = self._map_disease_id(disease_id) if disease_id else target_name

            confidence = float(activity.get("confidence_score", 0))

            # Canonical key: (normalized_drug_id, target_uniprot)
            canonical_key = (drug_id, target_uniprot)

            # Create interaction object
            interaction = DrugTargetInteraction(
                drug_id=drug_id,
                drug_name=drug_name,
                target_id=target_id,
                target_name=target_name,
                target_uniprot=target_uniprot,
                action_type=action_type,
                mechanism=mech_text,
                disease_id=disease_id,
                disease_name=disease_name,
                pathway_ids=[],  # Will be enriched later via pathway linker
                pathway_names=[],
                confidence_score=confidence,
                source="chembl",
            )

            # Keep highest confidence for each canonical key
            existing = interactions_by_key.get(canonical_key)
            if existing is None or confidence > existing[1]:
                interactions_by_key[canonical_key] = (interaction, confidence)

        # Yield deduplicated interactions
        for interaction, _ in interactions_by_key.values():
            yield interaction

    def _build_target_lookup(self, targets: pd.DataFrame, components: pd.DataFrame) -> dict:
        """Build target ID -> target info mapping."""
        lookup = {}

        # Target dictionary
        for _, row in targets.iterrows():
            tid = str(row.get("tid", ""))
            if tid:
                lookup[tid] = {
                    "target_name": row.get("pref_name", ""),
                    "target_type": row.get("target_type", ""),
                    "organism": row.get("organism", ""),
                    "uniprot": "",
                }

        # Target components for UniProt mapping
        for _, row in components.iterrows():
            tid = str(row.get("tid", ""))
            if tid and tid in lookup:
                uniprot = row.get("component_id", "") or row.get("accession", "")
                if uniprot and uniprot.startswith(("P", "Q", "O")):  # UniProt accession pattern
                    lookup[tid]["uniprot"] = uniprot

        return lookup

    def _build_compound_lookup(self, compounds: pd.DataFrame) -> dict:
        """Build compound ID -> compound info mapping."""
        lookup = {}
        for _, row in compounds.iterrows():
            cid = str(row.get("molregno", ""))
            if cid:
                lookup[cid] = {
                    "chembl_id": row.get("chembl_id", ""),
                    "pref_name": row.get("pref_name", ""),
                    "molecule_type": row.get("molecule_type", ""),
                }
        return lookup

    def _build_mechanism_lookup(self, mechanisms: pd.DataFrame) -> dict:
        """Build (molregno, tid) -> mechanism mapping."""
        lookup = {}
        for _, row in mechanisms.iterrows():
            molregno = str(row.get("molregno", ""))
            tid = str(row.get("tid", ""))
            if molregno and tid:
                key = (molregno, tid)
                if key not in lookup:
                    lookup[key] = {
                        "action_type": row.get("action_type", ""),
                        "mechanism": row.get("mechanism_of_action", ""),
                        "disease_effector": row.get("disease_effector", ""),
                        "max_phase": row.get("max_phase", 0),
                    }
        return lookup

    def _create_interaction(
        self,
        activity: pd.Series,
        target_lookup: dict,
        compound_lookup: dict,
        mechanism_lookup: dict,
    ) -> Optional[DrugTargetInteraction]:
        """Create a DrugTargetInteraction from an activity record."""
        try:
            molregno = str(activity.get("molregno", ""))
            tid = str(activity.get("tid", ""))

            if not molregno or not tid:
                return None

            compound = compound_lookup.get(molregno, {})
            target = target_lookup.get(tid, {})
            mechanism = mechanism_lookup.get((molregno, tid), {})

            drug_id = compound.get("chembl_id", f"CHEMBL{molregno}")
            drug_name = compound.get("pref_name", "") or drug_id
            target_id = tid
            target_name = target.get("target_name", "") or target_id
            target_uniprot = target.get("uniprot", "")

            action_type = mechanism.get("action_type", "") or self._infer_action_type(activity)
            mech_text = mechanism.get("mechanism", "") or ""

            # Disease mapping - ChEMBL doesn't have direct disease links in activities
            # Use mechanism disease_effector or infer from target
            disease_id = mechanism.get("disease_effector", "") or ""
            disease_name = self._map_disease_id(disease_id) if disease_id else target_name

            confidence = float(activity.get("confidence_score", 0))

            return DrugTargetInteraction(
                drug_id=drug_id,
                drug_name=drug_name,
                target_id=target_id,
                target_name=target_name,
                target_uniprot=target_uniprot,
                action_type=action_type,
                mechanism=mech_text,
                disease_id=disease_id,
                disease_name=disease_name,
                pathway_ids=[],  # Will be enriched later via pathway linker
                pathway_names=[],
                confidence_score=confidence,
                source="chembl",
            )
        except Exception as e:
            logger.debug(f"Failed to create interaction: {e}")
            return None

    def _infer_action_type(self, activity: pd.Series) -> str:
        """Infer action type from activity data."""
        standard_type = str(activity.get("standard_type", "")).lower()
        if "ic50" in standard_type or "ec50" in standard_type or "ki" in standard_type:
            return "inhibitor"
        elif "kd" in standard_type:
            return "binder"
        elif "potency" in standard_type:
            return "modulator"
        return "unknown"

    def _map_disease_id(self, disease_id: str) -> str:
        """Map disease ID to readable name (placeholder for EFO mapping)."""
        # In practice, this would map EFO/MONDO IDs to labels
        return disease_id


def parse_chembl_directory(data_dir: Path, settings=None) -> list[DrugTargetInteraction]:
    """Convenience function to parse all ChEMBL tables in a directory."""
    parser = ChEMBLParser(settings)
    tables = parser.load_tables(data_dir)
    return list(parser.parse_interactions(tables))