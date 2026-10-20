"""PubMed abstract parser for XML/JSON formats."""

import json
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional
from datetime import datetime

from src.config import get_settings
from src.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class PubMedAbstract:
    """Parsed PubMed abstract with metadata."""
    pmid: str
    title: str
    abstract: str
    mesh_terms: list[str] = field(default_factory=list)
    publication_date: Optional[datetime] = None
    authors: list[str] = field(default_factory=list)
    journal: str = ""
    doi: str = ""
    keywords: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "pmid": self.pmid,
            "title": self.title,
            "abstract": self.abstract,
            "mesh_terms": self.mesh_terms,
            "publication_date": self.publication_date.isoformat() if self.publication_date else None,
            "authors": self.authors,
            "journal": self.journal,
            "doi": self.doi,
            "keywords": self.keywords,
        }


class PubMedParser:
    """Parse PubMed XML and JSON files into structured abstracts."""

    def __init__(self, settings=None):
        self.settings = settings or get_settings()
        self.config = self.settings.ingestion.pubmed
        self._parse_stats = {"parsed": 0, "skipped": 0, "failed": 0}

    def reset_stats(self):
        """Reset parsing statistics."""
        self._parse_stats = {"parsed": 0, "skipped": 0, "failed": 0}

    def get_stats(self) -> dict:
        """Get parsing statistics."""
        return self._parse_stats.copy()

    def log_summary(self):
        """Log ingestion summary."""
        stats = self.get_stats()
        logger.info(
            f"PubMed ingestion summary: parsed={stats['parsed']}, "
            f"skipped={stats['skipped']}, failed={stats['failed']}"
        )

    def parse_xml_file(self, filepath: Path) -> Iterator[PubMedAbstract]:
        """Parse a PubMed XML file (single or multiple records)."""
        self.reset_stats()
        logger.info(f"Parsing PubMed XML: {filepath}")
        tree = ET.parse(filepath)
        root = tree.getroot()

        # Handle both single PubmedArticle and PubmedArticleSet
        articles = root.findall(".//PubmedArticle")
        if not articles and root.tag == "PubmedArticle":
            articles = [root]

        for article in articles:
            abstract = self._parse_article_xml(article)
            if abstract and self._validate_abstract(abstract):
                self._parse_stats["parsed"] += 1
                yield abstract
            elif abstract is not None:
                self._parse_stats["skipped"] += 1
            else:
                self._parse_stats["failed"] += 1

        self.log_summary()

    def parse_json_file(self, filepath: Path) -> Iterator[PubMedAbstract]:
        """Parse a PubMed JSON file (array of articles or single article)."""
        self.reset_stats()
        logger.info(f"Parsing PubMed JSON: {filepath}")
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        articles = data if isinstance(data, list) else [data]

        for article in articles:
            abstract = self._parse_article_json(article)
            if abstract and self._validate_abstract(abstract):
                self._parse_stats["parsed"] += 1
                yield abstract
            elif abstract is not None:
                self._parse_stats["skipped"] += 1
            else:
                self._parse_stats["failed"] += 1

        self.log_summary()

    def parse_directory(self, directory: Path, pattern: str = "*.xml") -> Iterator[PubMedAbstract]:
        """Parse all matching files in a directory."""
        for filepath in directory.glob(pattern):
            if filepath.suffix.lower() == ".xml":
                yield from self.parse_xml_file(filepath)
            elif filepath.suffix.lower() == ".json":
                yield from self.parse_json_file(filepath)

    def _parse_article_xml(self, article: ET.Element) -> Optional[PubMedAbstract]:
        """Extract data from a single PubmedArticle XML element."""
        try:
            # PMID
            pmid_elem = article.find(".//PMID")
            pmid = pmid_elem.text if pmid_elem is not None else ""
            if not pmid:
                return None

            # Title
            title_elem = article.find(".//ArticleTitle")
            title = title_elem.text if title_elem is not None else ""

            # Abstract
            abstract_text = self._extract_abstract_xml(article)

            # MeSH terms
            mesh_terms = []
            for mesh in article.findall(".//MeshHeading/DescriptorName"):
                if mesh.text:
                    mesh_terms.append(mesh.text)

            # Publication date
            pub_date = self._extract_pub_date_xml(article)

            # Authors
            authors = []
            for author in article.findall(".//Author"):
                last = author.find("LastName")
                fore = author.find("ForeName")
                if last is not None and fore is not None:
                    authors.append(f"{last.text} {fore.text}")
                elif last is not None:
                    authors.append(last.text)

            # Journal
            journal_elem = article.find(".//Journal/Title")
            journal = journal_elem.text if journal_elem is not None else ""

            # DOI
            doi = ""
            for id_elem in article.findall(".//ArticleId"):
                if id_elem.get("IdType") == "doi":
                    doi = id_elem.text or ""
                    break

            # Keywords
            keywords = []
            for kw in article.findall(".//Keyword"):
                if kw.text:
                    keywords.append(kw.text)

            return PubMedAbstract(
                pmid=pmid,
                title=title,
                abstract=abstract_text,
                mesh_terms=mesh_terms,
                publication_date=pub_date,
                authors=authors,
                journal=journal,
                doi=doi,
                keywords=keywords,
            )
        except ET.ParseError as e:
            logger.warning(f"XML parse error in article: {e}")
            return None
        except Exception as e:
            logger.warning(f"Failed to parse article: {e}")
            return None

    def _parse_article_json(self, article: dict) -> Optional[PubMedAbstract]:
        """Extract data from a single PubMed JSON article."""
        try:
            # Handle different JSON structures (PubMed API, Europe PMC, etc.)
            pmid = str(article.get("pmid") or article.get("PMID") or "")
            if not pmid:
                return None

            title = article.get("title") or article.get("ArticleTitle") or ""

            # Abstract handling varies
            abstract = ""
            if "abstract" in article:
                abs_data = article["abstract"]
                if isinstance(abs_data, str):
                    abstract = abs_data
                elif isinstance(abs_data, dict):
                    abstract = abs_data.get("text") or abs_data.get("AbstractText") or ""
            elif "AbstractText" in article:
                abs_text = article["AbstractText"]
                abstract = " ".join(abs_text) if isinstance(abs_text, list) else str(abs_text)

            # MeSH terms
            mesh_terms = []
            mesh_data = article.get("mesh_terms") or article.get("MeshHeadingList") or []
            for mesh in mesh_data:
                if isinstance(mesh, dict):
                    desc = mesh.get("DescriptorName") or mesh.get("descriptor_name")
                    if desc:
                        mesh_terms.append(desc)
                elif isinstance(mesh, str):
                    mesh_terms.append(mesh)

            # Publication date
            pub_date = None
            date_fields = ["pub_date", "PublicationDate", "DateCompleted", "DateRevised"]
            for field in date_fields:
                if field in article:
                    pub_date = self._parse_date(article[field])
                    if pub_date:
                        break

            # Authors
            authors = []
            author_data = article.get("authors") or article.get("AuthorList") or []
            for author in author_data:
                if isinstance(author, dict):
                    last = author.get("LastName") or author.get("last_name") or ""
                    fore = author.get("ForeName") or author.get("fore_name") or ""
                    if last:
                        authors.append(f"{last} {fore}".strip())
                elif isinstance(author, str):
                    authors.append(author)

            # Journal
            journal = article.get("journal") or article.get("Journal") or ""

            # DOI
            doi = article.get("doi") or article.get("DOI") or ""

            # Keywords
            keywords = article.get("keywords") or article.get("KeywordList") or []

            return PubMedAbstract(
                pmid=pmid,
                title=title,
                abstract=abstract,
                mesh_terms=mesh_terms,
                publication_date=pub_date,
                authors=authors,
                journal=journal,
                doi=doi,
                keywords=keywords,
            )
        except (KeyError, ValueError, TypeError) as e:
            logger.warning(f"JSON article missing required field: {e}")
            return None
        except Exception as e:
            logger.warning(f"Failed to parse JSON article: {e}")
            return None

    def _extract_abstract_xml(self, article: ET.Element) -> str:
        """Extract abstract text from XML, handling structured abstracts."""
        abstract_parts = []

        # Structured abstract with labels
        for abstract_text in article.findall(".//Abstract/AbstractText"):
            label = abstract_text.get("Label", "")
            text = abstract_text.text or ""
            if label and text:
                abstract_parts.append(f"{label}: {text}")
            elif text:
                abstract_parts.append(text)

        # Fallback to simple abstract
        if not abstract_parts:
            simple_abstract = article.find(".//AbstractText")
            if simple_abstract is not None and simple_abstract.text:
                abstract_parts.append(simple_abstract.text)

        return " ".join(abstract_parts)

    def _extract_pub_date_xml(self, article: ET.Element) -> Optional[datetime]:
        """Extract publication date from XML."""
        for date_elem in article.findall(".//PubDate"):
            year = date_elem.find("Year")
            month = date_elem.find("Month")
            day = date_elem.find("Day")

            try:
                y = int(year.text) if year is not None and year.text else 1900
                m = int(month.text) if month is not None and month.text and month.text.isdigit() else 1
                d = int(day.text) if day is not None and day.text and day.text.isdigit() else 1
                return datetime(y, m, d)
            except (ValueError, TypeError):
                continue

        # Fallback to DateCompleted or DateRevised
        for field in ["DateCompleted", "DateRevised"]:
            elem = article.find(f".//{field}")
            if elem is not None:
                try:
                    y = int(elem.find("Year").text) if elem.find("Year") is not None else 1900
                    m = int(elem.find("Month").text) if elem.find("Month") is not None else 1
                    d = int(elem.find("Day").text) if elem.find("Day") is not None else 1
                    return datetime(y, m, d)
                except (ValueError, TypeError, AttributeError):
                    continue

        return None

    def _parse_date(self, date_value: any) -> Optional[datetime]:
        """Parse date from various formats."""
        if isinstance(date_value, datetime):
            return date_value
        if isinstance(date_value, str):
            for fmt in ["%Y-%m-%d", "%Y/%m/%d", "%Y-%m-%dT%H:%M:%S", "%Y"]:
                try:
                    return datetime.strptime(date_value, fmt)
                except ValueError:
                    continue
        if isinstance(date_value, dict):
            try:
                y = int(date_value.get("Year", 1900))
                m = int(date_value.get("Month", 1))
                d = int(date_value.get("Day", 1))
                return datetime(y, m, d)
            except (ValueError, TypeError):
                pass
        return None

    def _validate_abstract(self, abstract: PubMedAbstract) -> bool:
        """Validate abstract has required fields."""
        for field_name in self.config.required_fields:
            value = getattr(abstract, field_name, None)
            if not value:
                logger.debug(f"Abstract {abstract.pmid} missing required field: {field_name}")
                return False
        return True


def parse_pubmed_file(filepath: Path, settings=None) -> list[PubMedAbstract]:
    """Convenience function to parse a single PubMed file."""
    parser = PubMedParser(settings)
    if filepath.suffix.lower() == ".xml":
        return list(parser.parse_xml_file(filepath))
    elif filepath.suffix.lower() == ".json":
        return list(parser.parse_json_file(filepath))
    else:
        raise ValueError(f"Unsupported file format: {filepath.suffix}")