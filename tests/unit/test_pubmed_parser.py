"""Unit tests for PubMed parser."""

import pytest
from pathlib import Path

from src.ingestion.pubmed_parser import PubMedParser, parse_pubmed_file


@pytest.fixture
def sample_xml_path():
    return Path("tests/fixtures/sample_pubmed.xml")


@pytest.fixture
def parser():
    return PubMedParser()


def test_parse_pubmed_xml(parser, sample_xml_path):
    """Test parsing PubMed XML file."""
    abstracts = list(parser.parse_xml_file(sample_xml_path))

    assert len(abstracts) == 5

    # Check first abstract
    first = abstracts[0]
    assert first.pmid == "38412345"
    assert "Aspirin resistance" in first.title
    assert "COX-1" in first.abstract
    assert "EGFR" in first.abstract
    assert len(first.mesh_terms) == 5
    assert "Acetylsalicylic Acid" in first.mesh_terms
    assert "COX-1" in first.mesh_terms
    assert first.doi == "10.1161/ATVBAHA.123.318456"


def test_parse_pubmed_xml_entities(parser, sample_xml_path):
    """Test that entities are present in abstracts."""
    abstracts = list(parser.parse_xml_file(sample_xml_path))

    # Check second abstract has drug names
    second = abstracts[1]
    assert "Sildenafil" in second.abstract or "CHEMBL132" in second.abstract
    assert "Metformin" in second.abstract or "CHEMBL1100" in second.abstract


def test_parse_pubmed_json(parser, sample_xml_path):
    """Test parsing via convenience function (XML)."""
    abstracts = parse_pubmed_file(sample_xml_path)
    assert len(abstracts) == 5
    assert all(hasattr(a, 'pmid') for a in abstracts)


def test_abstract_to_dict(parser, sample_xml_path):
    """Test conversion to dictionary."""
    abstracts = list(parser.parse_xml_file(sample_xml_path))
    d = abstracts[0].to_dict()

    assert d["pmid"] == "38412345"
    assert "mesh_terms" in d
    assert isinstance(d["mesh_terms"], list)
    assert "publication_date" in d


def test_required_fields_validation(parser):
    """Test validation of required fields."""
    # Create a mock abstract with missing fields
    from src.ingestion.pubmed_parser import PubMedAbstract
    from datetime import datetime

    abstract = PubMedAbstract(
        pmid="123",
        title="Test",
        abstract="",  # Missing required field
        mesh_terms=[],
        publication_date=datetime.now(),
    )
    assert not parser._validate_abstract(abstract)

    abstract.abstract = "Valid abstract"
    assert parser._validate_abstract(abstract)