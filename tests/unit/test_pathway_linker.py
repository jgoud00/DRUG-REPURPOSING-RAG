"""Unit tests for pathway linker."""

import pytest
from pathlib import Path

from src.retrieval.pathway_linker import PathwayLinker, PathwayGraph
from src.ingestion.chembl_parser import DrugTargetInteraction


@pytest.fixture
def sample_interactions():
    """Create sample drug-target interactions."""
    return [
        DrugTargetInteraction(
            drug_id="CHEMBL25",
            drug_name="Aspirin",
            target_id="CHEMBL240",
            target_name="COX-1",
            target_uniprot="P23219",
            action_type="inhibitor",
            mechanism="irreversible acetylation of serine 530",
            disease_id="EFO_0001379",
            disease_name="Myocardial Infarction",
            pathway_ids=["R-HSA-216083"],
            pathway_names=["Prostaglandin synthesis"],
            confidence_score=9.0,
            source="chembl",
        ),
        DrugTargetInteraction(
            drug_id="CHEMBL132",
            drug_name="Sildenafil",
            target_id="CHEMBL3150",
            target_name="PDE5A",
            target_uniprot="O76083",
            action_type="inhibitor",
            mechanism="competitive PDE5A inhibition",
            disease_id="EFO_0000249",
            disease_name="Alzheimer Disease",
            pathway_ids=["R-HSA-112316"],
            pathway_names=["NO-cGMP signaling"],
            confidence_score=7.0,
            source="chembl",
        ),
        DrugTargetInteraction(
            drug_id="CHEMBL1100",
            drug_name="Metformin",
            target_id="CHEMBL4200",
            target_name="AMPK",
            target_uniprot="Q13131",
            action_type="activator",
            mechanism="AMPK activation via LKB1",
            disease_id="EFO_0000249",
            disease_name="Alzheimer Disease",
            pathway_ids=["R-HSA-196854"],
            pathway_names=["AMPK signaling"],
            confidence_score=6.0,
            source="chembl",
        ),
        DrugTargetInteraction(
            drug_id="CHEMBL382",
            drug_name="Thalidomide",
            target_id="CHEMBL5600",
            target_name="Cereblon",
            target_uniprot="Q96SW2",
            action_type="modulator",
            mechanism="CRBN binding and E3 ligase modulation",
            disease_id="EFO_0000305",
            disease_name="Multiple Myeloma",
            pathway_ids=["R-HSA-5673001"],
            pathway_names=["Ubiquitin-mediated proteolysis"],
            confidence_score=7.0,
            source="chembl",
        ),
    ]


@pytest.fixture
def pathway_linker():
    """Create pathway linker with fixture data."""
    linker = PathwayLinker(
        pathway_tsv=Path("tests/fixtures/reactome_disease_pathways.tsv")
    )
    linker.load_pathway_mapping()
    return linker


class TestPathwayGraph:
    """Test pathway graph."""

    def test_load_mapping(self, pathway_linker):
        """Test loading pathway mapping from TSV."""
        assert "EFO_0000249" in pathway_linker.graph.disease_pathways  # Alzheimer's
        assert "EFO_0001379" in pathway_linker.graph.disease_pathways  # MI
        assert "EFO_0000305" in pathway_linker.graph.disease_pathways  # Myeloma

        # Check target pathways
        assert "P23219" in pathway_linker.graph.target_pathways  # COX-1
        assert "O76083" in pathway_linker.graph.target_pathways  # PDE5A
        assert "Q13131" in pathway_linker.graph.target_pathways  # AMPK
        assert "Q96SW2" in pathway_linker.graph.target_pathways  # Cereblon

        # Check pathway names
        assert pathway_linker.graph.pathway_names.get("R-HSA-112316") == "NO-cGMP signaling"
        assert pathway_linker.graph.pathway_names.get("R-HSA-216083") == "Prostaglandin synthesis"


class TestPathwayLinker:
    """Test pathway linker."""

    def test_load_interactions(self, pathway_linker, sample_interactions):
        """Test loading interactions."""
        pathway_linker.load_interactions(sample_interactions)

        assert "P23219" in pathway_linker._interaction_map
        assert "O76083" in pathway_linker._interaction_map
        assert "CHEMBL25" in pathway_linker._drug_interactions
        assert "CHEMBL132" in pathway_linker._drug_interactions

    def test_find_candidates_alzheimer(self, pathway_linker, sample_interactions):
        """Test finding candidates for Alzheimer's disease."""
        pathway_linker.load_interactions(sample_interactions)

        candidates = pathway_linker.find_candidate_drugs(
            disease_id="EFO_0000249",
            disease_name="Alzheimer's disease",
            min_shared_pathways=1,
            top_k=10,
        )

        assert len(candidates) > 0
        # Should find Sildenafil and Metformin
        drug_names = [c.drug_name for c in candidates]
        assert "Sildenafil" in drug_names
        assert "Metformin" in drug_names

        # Sildenafil should have shared pathway
        sild = next(c for c in candidates if c.drug_name == "Sildenafil")
        assert len(sild.shared_pathways) > 0
        assert "R-HSA-112316" in sild.shared_pathways or "NO-cGMP signaling" in sild.pathway_names

    def test_find_candidates_myeloma(self, pathway_linker, sample_interactions):
        """Test finding candidates for Multiple Myeloma."""
        pathway_linker.load_interactions(sample_interactions)

        candidates = pathway_linker.find_candidate_drugs(
            disease_id="EFO_0000305",
            disease_name="Multiple Myeloma",
            min_shared_pathways=1,
            top_k=10,
        )

        assert len(candidates) > 0
        drug_names = [c.drug_name for c in candidates]
        assert "Thalidomide" in drug_names

        thal = next(c for c in candidates if c.drug_name == "Thalidomide")
        assert len(thal.shared_pathways) > 0
        assert thal.confidence_score == 7.0

    def test_find_candidates_mi(self, pathway_linker, sample_interactions):
        """Test finding candidates for Myocardial Infarction."""
        pathway_linker.load_interactions(sample_interactions)

        candidates = pathway_linker.find_candidate_drugs(
            disease_id="EFO_0001379",
            disease_name="Myocardial Infarction",
            min_shared_pathways=1,
            top_k=10,
        )

        assert len(candidates) > 0
        drug_names = [c.drug_name for c in candidates]
        assert "Aspirin" in drug_names

    def test_no_pathway_mapping(self, pathway_linker, sample_interactions):
        """Test disease with no pathway mapping."""
        pathway_linker.load_interactions(sample_interactions)

        candidates = pathway_linker.find_candidate_drugs(
            disease_id="EFO_9999999",
            disease_name="Unknown Disease",
            min_shared_pathways=1,
        )

        assert candidates == []

    def test_rank_with_retrieval_support(self, pathway_linker, sample_interactions):
        """Test ranking with retrieval support."""
        pathway_linker.load_interactions(sample_interactions)

        # Mock retrieval results
        class MockResult:
            def __init__(self, drug_name, target_name):
                self.metadata = {"drug_name": drug_name, "target_name": target_name}

        retrieval_results = [
            MockResult("Sildenafil", "PDE5A"),
            MockResult("Sildenafil", "PDE5A"),  # Two hits
            MockResult("Metformin", "AMPK"),
        ]

        ranked = pathway_linker.rank_repurposing_candidates(
            disease_id="EFO_0000249",
            disease_name="Alzheimer's disease",
            retrieval_results=retrieval_results,
        )

        # Sildenafil should have higher retrieval support
        sild = next(c for c in ranked if c.drug_name == "Sildenafil")
        met = next(c for c in ranked if c.drug_name == "Metformin")
        assert sild.retrieval_support == 2
        assert met.retrieval_support == 1


class TestPathwayLinkerFromCorpus:
    """Test loading pathway linker from corpus."""

    def test_load_from_corpus(self, tmp_path):
        """Test loading from corpus (requires corpus file)."""
        # This would need a real corpus file - skip for unit test
        pass