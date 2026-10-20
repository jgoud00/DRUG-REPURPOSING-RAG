"""Unit tests for ChEMBL parser."""

import pytest
from pathlib import Path

from src.ingestion.chembl_parser import ChEMBLParser, parse_chembl_directory, DrugTargetInteraction


@pytest.fixture
def sample_chembl_dir():
    return Path("tests/fixtures")


@pytest.fixture
def parser():
    return ChEMBLParser()


def test_load_tables(parser, sample_chembl_dir):
    """Test loading ChEMBL CSV tables."""
    tables = parser.load_tables(sample_chembl_dir)

    assert "activities" in tables
    assert "target_dictionary" in tables
    assert "compound_records" in tables
    assert "mechanism_of_action" in tables
    assert "target_components" in tables

    assert len(tables["activities"]) == 10
    assert len(tables["target_dictionary"]) == 5
    assert len(tables["compound_records"]) == 8


def test_parse_interactions(parser, sample_chembl_dir):
    """Test parsing interactions from tables."""
    tables = parser.load_tables(sample_chembl_dir)
    interactions = list(parser.parse_interactions(tables))

    # Should create interactions for each activity-mechanism pair
    assert len(interactions) > 0

    # Check first interaction (Aspirin -> COX-1)
    aspirin_interactions = [i for i in interactions if i.drug_id == "CHEMBL25"]
    assert len(aspirin_interactions) > 0

    first = aspirin_interactions[0]
    assert first.drug_name == "Aspirin"
    assert first.target_name == "COX-1"
    assert first.target_uniprot == "P23219"
    assert first.action_type == "inhibitor"
    assert first.confidence_score >= 5


def test_interaction_fields(parser, sample_chembl_dir):
    """Test that all required fields are present."""
    tables = parser.load_tables(sample_chembl_dir)
    interactions = list(parser.parse_interactions(tables))

    for interaction in interactions:
        assert interaction.drug_id
        assert interaction.drug_name
        assert interaction.target_id
        assert interaction.target_name
        assert interaction.action_type
        assert interaction.disease_name
        assert isinstance(interaction.pathway_ids, list)
        assert isinstance(interaction.pathway_names, list)
        assert interaction.source == "chembl"


def test_sildenafil_interaction(parser, sample_chembl_dir):
    """Test Sildenafil -> PDE5A interaction for Alzheimer's."""
    tables = parser.load_tables(sample_chembl_dir)
    interactions = list(parser.parse_interactions(tables))

    sildenafil = [i for i in interactions if i.drug_id == "CHEMBL132"]
    assert len(sildenafil) > 0

    s = sildenafil[0]
    assert s.target_name == "PDE5A"
    assert s.disease_name == "Alzheimer Disease"
    assert "PDE5A" in s.mechanism or "inhibition" in s.mechanism.lower()


def test_metformin_interaction(parser, sample_chembl_dir):
    """Test Metformin -> AMPK interaction."""
    tables = parser.load_tables(sample_chembl_dir)
    interactions = list(parser.parse_interactions(tables))

    metformin = [i for i in interactions if i.drug_id == "CHEMBL1100"]
    assert len(metformin) > 0

    m = metformin[0]
    assert m.target_name == "AMPK"
    assert m.action_type == "activator"
    assert "AMPK" in m.mechanism


def test_thalidomide_interaction(parser, sample_chembl_dir):
    """Test Thalidomide -> Cereblon for Multiple Myeloma."""
    tables = parser.load_tables(sample_chembl_dir)
    interactions = list(parser.parse_interactions(tables))

    thalidomide = [i for i in interactions if i.drug_id == "CHEMBL382"]
    assert len(thalidomide) > 0

    t = thalidomide[0]
    assert t.target_name == "Cereblon"
    assert t.disease_name == "Multiple Myeloma"
    assert "CRBN" in t.mechanism or "cereblon" in t.mechanism.lower()


def test_confidence_filtering(parser, sample_chembl_dir):
    """Test that confidence threshold is applied."""
    tables = parser.load_tables(sample_chembl_dir)
    interactions = list(parser.parse_interactions(tables))

    # All interactions should have confidence >= 5 (default threshold)
    for i in interactions:
        assert i.confidence_score >= 5


def test_parse_chembl_directory_convenience(sample_chembl_dir):
    """Test convenience function."""
    interactions = parse_chembl_directory(sample_chembl_dir)
    assert len(interactions) > 0
    assert all(isinstance(i, DrugTargetInteraction) for i in interactions)