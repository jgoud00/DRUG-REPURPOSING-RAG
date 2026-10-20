"""Semantic chunking for biomedical text and structured data."""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional

from src.config import get_settings
from src.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class Chunk:
    """A text chunk with metadata."""
    doc_id: str
    source: str  # "pubmed" or "chembl"
    text: str
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "doc_id": self.doc_id,
            "source": self.source,
            "text": self.text,
            "metadata": self.metadata,
        }


# Simple regex-based sentence splitter (no spaCy dependency)
_SENTENCE_END = re.compile(r'(?<=[.!?])\s+(?=[A-Z])')

# Common biomedical abbreviations that should not end sentences
_BIOMEDICAL_ABBREVIATIONS = {
    'fig', 'fig.', 'et al', 'et al.', 'etc', 'etc.', 'vs', 'vs.', 'dr', 'dr.',
    'prof', 'prof.', 'mr', 'mr.', 'ms', 'ms.', 'mrs', 'mrs.', 'jr', 'jr.',
    'sr', 'sr.', 'phd', 'md', 'md.', 'ba', 'bs', 'ms', 'ph.d', 'ph.d.',
    'e.g', 'e.g.', 'i.e', 'i.e.', 'cf', 'cf.', 'vs', 'vs.', 'viz', 'viz.',
    'approx', 'approx.', 'ca', 'ca.', 'esp', 'esp.', 'ie', 'ie.', 'eg', 'eg.',
    'etal', 'et al', 'et al.', 'fig', 'fig.', 'figs', 'figs.', 'tab', 'tab.',
    'tabs', 'tabs.', 'ref', 'ref.', 'refs', 'refs.', 'suppl', 'suppl.',
    'vol', 'vol.', 'vols', 'vols.', 'no', 'no.', 'pp', 'pp.', 'p', 'p.',
    'ch', 'ch.', 'chpt', 'chpt.', 'sec', 'sec.', 'eq', 'eq.', 'eqs', 'eqs.',
    'fig', 'fig.', 'figs', 'figs.', 'table', 'table.', 'tables', 'tables.',
    'appendix', 'append.', 'fig.', 'figs.', 'eq.', 'eqs.', 'sec.', 'secs.',
    'jan', 'feb', 'mar', 'apr', 'jun', 'jul', 'aug', 'sep', 'sept', 'oct', 'nov', 'dec',
    'mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun',
    'mrna', 'dna', 'rna', 'cdna', 'pcr', 'rt-pcr', 'qpcr', 'dna', 'rna',
    'cdna', 'pcr', 'rna', 'dna', 'cdna', 'pcr', 'qpcr', 'rt-pcr',
    'orf', 'utr', 'cds', 'utr', 'snv', 'indel', 'cnv', 'snp', 'gwas',
    'chr', 'chr.', 'kb', 'mb', 'gb', 'ml', 'ul', 'nl', 'mg', 'ug', 'ng',
    'pg', 'mg', 'mcg', 'mm', 'cm', 'm', 'km', 'hz', 'khz', 'mhz', 'ghz',
    'ms', 'us', 'ns', 'min', 'hr', 'hr.', 'sec', 'sec.', 'yr', 'yr.', 'mo', 'wk', 'd',
    'n', 's', 'e', 'w', 'ne', 'nw', 'se', 'sw', 'dr', 'dr.', 'prof', 'prof.',
    'mr', 'mr.', 'ms', 'ms.', 'mrs', 'mrs.', 'jr', 'jr.', 'sr', 'sr.',
    'ph.d', 'm.d', 'b.s', 'm.s', 'b.a', 'm.a', 'ph.d.', 'm.d.', 'b.s.', 'm.s.',
    'b.a.', 'm.a.', 'e.g', 'e.g.', 'i.e', 'i.e.', 'cf', 'cf.', 'viz', 'viz.',
    'approx', 'approx.', 'ca', 'ca.', 'esp', 'esp.', 'et al', 'et al.',
    'fig', 'fig.', 'figs', 'figs.', 'table', 'table.', 'tables', 'tables.',
    'ref', 'ref.', 'refs', 'refs.', 'vol', 'vol.', 'vols', 'vols.',
    'no', 'no.', 'pp', 'pp.', 'p', 'p.', 'ch', 'ch.', 'chpt', 'chpt.',
    'sec', 'sec.', 'eq', 'eq.', 'eqs', 'eqs.', 'fig', 'fig.', 'figs',
    'figs.', 'table', 'table.', 'tables', 'tables.', 'appendix', 'append.',
    'appendices', 'append.', 'eq.', 'eqs.', 'sec.', 'secs.', 'cf.', 'cf',
    'etal', 'et al', 'et al.', 'approx', 'approx.', 'ca', 'ca.',
    'mrna', 'sirna', 'mirna', 'lncrna', 'pirna', 'snrna', 'snrnas',
    'hr', 'hr.', 'min', 'min.', 'sec', 'sec.', 'ms', 'ms.', 'us', 'ns',
    'kv', 'mv', 'ma', 'ma.', 'pa', 'pa.', 'ka', 'ka.', 'da', 'da.',
    'ua', 'ua.', 'ra', 'ra.', 'dec', 'jan', 'feb', 'mar', 'apr', 'jun',
    'jul', 'aug', 'sep', 'sept', 'oct', 'nov', 'dec',
}

# Compile abbreviation patterns for efficient replacement
_ABBREV_PATTERNS = {abbr: abbr.replace('.', '<DOT>') for abbr in _BIOMEDICAL_ABBREVIATIONS if '.' in abbr}
_ABBREV_REGEX = re.compile(r'\b(' + '|'.join(re.escape(abbr) for abbr in _ABBREV_PATTERNS.keys()) + r')\b', re.IGNORECASE)


def split_sentences(text: str) -> list[str]:
    """Split text into sentences using regex (fast, no ML deps)."""
    # Handle common abbreviations to avoid over-splitting
    def replace_abbrev(match):
        return _ABBREV_PATTERNS.get(match.group(0).lower(), match.group(0))

    text = _ABBREV_REGEX.sub(replace_abbrev, text)
    sentences = _SENTENCE_END.split(text)
    # Restore dots in abbreviations
    sentences = [s.replace('<DOT>', '.').strip() for s in sentences if s.strip()]
    return sentences


class SentenceSplitter:
    """Sentence splitter with optional spaCy backend."""

    def __init__(self, model_name: str = "en_core_sci_sm", use_spacy: bool = False):
        self.model_name = model_name
        self.use_spacy = use_spacy
        self._nlp = None

    def split(self, text: str) -> list[str]:
        """Split text into sentences."""
        if self.use_spacy:
            try:
                import spacy
                if self._nlp is None:
                    self._nlp = spacy.load(self.model_name)
                doc = self._nlp(text)
                return [sent.text.strip() for sent in doc.sents if sent.text.strip()]
            except Exception:
                logger.warning("spaCy unavailable, falling back to regex splitter")
                self.use_spacy = False
        return split_sentences(text)


class SemanticChunker:
    """Semantic chunking for PubMed abstracts."""

    def __init__(self, settings=None):
        self.settings = settings or get_settings()
        self.config = self.settings.chunking
        # Use regex splitter by default to avoid spaCy/numpy version conflicts
        self.splitter = SentenceSplitter(self.config.sentence_splitter, use_spacy=False)
        self.chunk_size = self.config.abstract_chunk_size
        self.chunk_overlap = self.config.abstract_chunk_overlap
        self.min_chunk_size = self.config.abstract_min_chunk_size

    def chunk_abstract(self, abstract: dict, base_doc_id: str) -> list[Chunk]:
        """Chunk a PubMed abstract into overlapping semantic windows."""
        text = abstract.get("abstract", "")
        if not text:
            return []

        sentences = self.splitter.split(text)
        if not sentences:
            return []

        # Estimate tokens (rough: 1 token ≈ 4 chars)
        def estimate_tokens(s: str) -> int:
            return len(s) // 4

        chunks = []
        current_chunk = []
        current_tokens = 0
        chunk_idx = 0
        char_offset = 0

        for sent in sentences:
            sent_tokens = estimate_tokens(sent)

            if current_tokens + sent_tokens > self.chunk_size and current_chunk:
                # Create chunk
                chunk_text = " ".join(current_chunk)
                if estimate_tokens(chunk_text) >= self.min_chunk_size:
                    chunks.append(Chunk(
                        doc_id=f"{base_doc_id}_chunk_{chunk_idx}",
                        source="pubmed",
                        text=chunk_text,
                        metadata={
                            "pmid": abstract.get("pmid"),
                            "title": abstract.get("title"),
                            "mesh_terms": self._serialize_list(abstract.get("mesh_terms", [])),
                            "chunk_idx": chunk_idx,
                            "char_start": char_offset,
                            "char_end": char_offset + len(chunk_text),
                            "section": "abstract",
                            "entities": self._serialize_list(self._extract_entities(chunk_text)),
                        }
                    ))
                    chunk_idx += 1
                    char_offset += len(chunk_text) + 1

                # Overlap: keep last sentences that fit in overlap
                overlap_tokens = 0
                overlap_sentences = []
                for s in reversed(current_chunk):
                    t = estimate_tokens(s)
                    if overlap_tokens + t <= self.chunk_overlap:
                        overlap_sentences.insert(0, s)
                        overlap_tokens += t
                    else:
                        break
                current_chunk = overlap_sentences
                current_tokens = overlap_tokens

            current_chunk.append(sent)
            current_tokens += sent_tokens

        # Final chunk
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            if estimate_tokens(chunk_text) >= self.min_chunk_size:
                chunks.append(Chunk(
                    doc_id=f"{base_doc_id}_chunk_{chunk_idx}",
                    source="pubmed",
                    text=chunk_text,
                    metadata={
                        "pmid": abstract.get("pmid"),
                        "title": abstract.get("title"),
                        "mesh_terms": self._serialize_list(abstract.get("mesh_terms", [])),
                        "chunk_idx": chunk_idx,
                        "char_start": char_offset,
                        "char_end": char_offset + len(chunk_text),
                        "section": "abstract",
                        "entities": self._serialize_list(self._extract_entities(chunk_text)),
                    }
                ))

        return chunks

    def _serialize_list(self, items: list) -> str:
        """Serialize list to CSV string for ChromaDB compatibility."""
        if not self.settings.chunking.serialize_lists_as_csv:
            return items
        return ",".join(str(x) for x in items if x)

    def _extract_entities(self, text: str) -> list[str]:
        """Extract biomedical entities (improved regex-based for MVP)."""
        entities = []

        # Gene/protein symbols (e.g., EGFR, TP53, COX-1, mTORC1)
        # Matches: 2-6 uppercase alphanumeric, optionally with hyphen-digit suffix
        gene_pattern = r'\b[A-Z][A-Z0-9]{1,5}(?:-\d+)?\b'
        entities.extend(re.findall(gene_pattern, text))

        # Protein complexes (e.g., mTORC1, mTORC2, NF-kB, NF-κB, IKK, PI3K, AKT, mTOR)
        complex_pattern = r'\b(?:mTORC[12]|NF[-\u03ba]B|NF-kB|I[KK]{1,2}|PI3K|AKT|mTOR|AMPK|MAPK|ERK|JNK|p38|STAT[1-6]|SMAD[1-9]|SOX[0-9]|OCT4|NANOG|KLF4|MYC|TP53|BRCA[12]|PTEN|ATM|ATR|CHEK[12]|CDK[1-9]|CYCLIN[A-E]|RB|E2F|MYC|RAS|RAF|MEK|ERK|JNK|p38|IKK|NFKB|RELA|RELB|REL|IKB|IKBA|IKBB|IKBE|IKBG|NEMO|TRAF|TRADD|FADD|CASP[389]|APAF|BAX|BAK|BCL2|BCLXL|MCL1|BID|BAD|BIM|PUMA|NOXA|BH3|APOPTOSIS|AUTOPHAGY|ULK|ATG|LC3|BECLIN|VPS34|AMBRA|RUBICON|UVRAG)\b'
        entities.extend(re.findall(complex_pattern, text, re.IGNORECASE))

        # ChEMBL IDs
        chembl_pattern = r'CHEMBL\d+'
        entities.extend(re.findall(chembl_pattern, text))

        # UniProt IDs (accession format: P12345, Q9XYZ0, A0A0B4, A0A1B2C3D4)
        uniprot_pattern = r'\b[OPQ][0-9][A-Z0-9]{3}[0-9]\b|\b[A-NR-Z][0-9](?:[A-Z][A-Z0-9]{2}[0-9]){1,2}\b|\bA0A[0-9A-Z]{6}\b'
        entities.extend(re.findall(uniprot_pattern, text))

        # rs identifiers (dbSNP)
        rs_pattern = r'rs\d+'
        entities.extend(re.findall(rs_pattern, text, re.IGNORECASE))

        # miRNAs (e.g., miR-21, miR-155, let-7, miR-146a)
        mirna_pattern = r'\b(?:miR|let|mir|lin|bantam|mirtron|miRNA)-?\d+[a-z]?\b'
        entities.extend(re.findall(mirna_pattern, text, re.IGNORECASE))

        # Common chemical/drug names
        drug_names = [
            'Aspirin', 'Ibuprofen', 'Acetaminophen', 'Paracetamol', 'Metformin',
            'Sildenafil', 'Thalidomide', 'Lenalidomide', 'Pomalidomide',
            'Rofecoxib', 'Celecoxib', 'Warfarin', 'Heparin', 'Insulin',
            'Morphine', 'Fentanyl', 'Codeine', 'Oxycodone', 'Tramadol',
            'Duloxetine', 'Venlafaxine', 'Bupropion', 'Mirtazapine',
            'Quetiapine', 'Olanzapine', 'Risperidone', 'Aripiprazole',
            'Clozapine', 'Haloperidol', 'Lithium', 'Valproate',
            'Carbamazepine', 'Lamotrigine', 'Topiramate', 'Levetiracetam',
            'Phenytoin', 'Valproate', 'Gabapentin', 'Pregabalin',
            'Modafinil', 'Armodafinil', 'Methylphenidate', 'Atomoxetine'
        ]
        for drug in drug_names:
            if re.search(rf'\b{re.escape(drug)}\b', text, re.IGNORECASE):
                entities.append(drug)

        # Gene/protein symbols (e.g., EGFR, TP53, COX-1)
        gene_pattern = r'\b[A-Z][A-Z0-9]{1,5}(?:-\d+)?\b'
        entities.extend(re.findall(gene_pattern, text))

        # ChEMBL IDs
        chembl_pattern = r'CHEMBL\d+'
        entities.extend(re.findall(chembl_pattern, text))

        # UniProt IDs (accession format: P12345, Q9XYZ0, A0A0B4, A0A1B2C3D4)
        uniprot_pattern = r'\b[OPQ][0-9][A-Z0-9]{3}[0-9]\b|\b[A-NR-Z][0-9](?:[A-Z][A-Z0-9]{2}[0-9]){1,2}\b|\bA0A[0-9A-Z]{6}\b'
        entities.extend(re.findall(uniprot_pattern, text))

        # rs identifiers (dbSNP)
        rs_pattern = r'rs\d+'
        entities.extend(re.findall(rs_pattern, text, re.IGNORECASE))

        # miRNAs (e.g., miR-21, miR-155, let-7, miR-146a)
        mirna_pattern = r'\b(?:miR|let|mir|lin|bantam|mirtron|miRNA)-?\d+[a-z]?\b'
        entities.extend(re.findall(mirna_pattern, text, re.IGNORECASE))

        # Common chemical/drug names
        drug_names = [
            'Aspirin', 'Ibuprofen', 'Acetaminophen', 'Paracetamol', 'Metformin',
            'Sildenafil', 'Thalidomide', 'Lenalidomide', 'Pomalidomide',
            'Rofecoxib', 'Celecoxib', 'Warfarin', 'Heparin', 'Insulin',
            'Morphine', 'Fentanyl', 'Codeine', 'Oxycodone', 'Tramadol',
            'Duloxetine', 'Venlafaxine', 'Bupropion', 'Mirtazapine',
            'Quetiapine', 'Olanzapine', 'Risperidone', 'Aripiprazole',
            'Clozapine', 'Haloperidol', 'Lithium', 'Valproate',
            'Carbamazepine', 'Lamotrigine', 'Topiramate', 'Levetiracetam',
            'Phenytoin', 'Valproate', 'Gabapentin', 'Pregabalin',
            'Modafinil', 'Armodafinil', 'Methylphenidate', 'Atomoxetine'
        ]
        for drug in drug_names:
            if re.search(rf'\b{re.escape(drug)}\b', text, re.IGNORECASE):
                entities.append(drug)

        return list(set(entities))


class StructuredChunker:
    """Convert structured drug-target interactions to text chunks."""

    def __init__(self, settings=None):
        self.settings = settings or get_settings()
        self.config = self.settings.chunking
        self.template = self.config.structured_template

    def chunk_interaction(self, interaction: dict) -> list[Chunk]:
        """Convert a DrugTargetInteraction to one or more chunks."""
        chunks = []

        # Row-level chunk (primary)
        if self.config.include_row_level:
            pathway_names = interaction.get("pathway_names", [])
            if isinstance(pathway_names, list):
                pathway_str = ", ".join(pathway_names)
            else:
                pathway_str = str(pathway_names)

            text = self.template.format(
                drug_name=interaction.get("drug_name", ""),
                drug_id=interaction.get("drug_id", ""),
                action_type=interaction.get("action_type", ""),
                target_name=interaction.get("target_name", ""),
                target_uniprot=interaction.get("target_uniprot", ""),
                disease_name=interaction.get("disease_name", ""),
                mechanism=interaction.get("mechanism", ""),
                pathway_names=pathway_str,
            )

            metadata = {
                "drug_id": interaction.get("drug_id"),
                "drug_name": interaction.get("drug_name"),
                "target_id": interaction.get("target_id"),
                "target_name": interaction.get("target_name"),
                "target_uniprot": interaction.get("target_uniprot"),
                "action_type": interaction.get("action_type"),
                "mechanism": interaction.get("mechanism"),
                "disease_id": interaction.get("disease_id"),
                "disease_name": interaction.get("disease_name"),
                "pathway_ids": self._serialize_list(interaction.get("pathway_ids", [])),
                "pathway_names": self._serialize_list(interaction.get("pathway_names", [])),
                "confidence_score": interaction.get("confidence_score", 0),
                "source": interaction.get("source", "chembl"),
            }

            doc_id = f"{interaction.get('source', 'chembl')}_{interaction.get('drug_id')}_{interaction.get('target_id')}"
            chunks.append(Chunk(
                doc_id=doc_id,
                source="chembl",
                text=text,
                metadata=metadata,
            ))

        # Entity-centric summary (optional)
        if self.config.include_entity_summaries:
            # Group by drug for summary (would need all interactions for a drug)
            pass

        return chunks

    def _serialize_list(self, items: list) -> str:
        """Serialize list to CSV string for ChromaDB compatibility."""
        if not self.settings.chunking.serialize_lists_as_csv:
            return items
        return ",".join(str(x) for x in items if x)


def chunk_pubmed_abstracts(abstracts: list[dict], settings=None) -> list[Chunk]:
    """Convenience function to chunk a list of PubMed abstracts."""
    chunker = SemanticChunker(settings)
    all_chunks = []
    for abstract in abstracts:
        base_id = f"pubmed_{abstract.get('pmid', 'unknown')}"
        all_chunks.extend(chunker.chunk_abstract(abstract, base_id))
    return all_chunks


def chunk_chembl_interactions(interactions: list[dict], settings=None) -> list[Chunk]:
    """Convenience function to chunk a list of ChEMBL interactions."""
    chunker = StructuredChunker(settings)
    all_chunks = []
    for interaction in interactions:
        all_chunks.extend(chunker.chunk_interaction(interaction))
    return all_chunks