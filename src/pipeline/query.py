"""End-to-end query pipeline: retrieve -> rerank -> pathway link -> generate."""

import logging
import threading
from dataclasses import dataclass
from typing import List, Optional

from src.config import get_settings
from src.retrieval.hybrid_retriever import HybridRetriever, RetrievalResult
from src.retrieval.reranker import CrossEncoderReranker
from src.retrieval.pathway_linker import PathwayLinker, PathwayLink
from src.generation.answer_synthesizer import AnswerSynthesizer, SynthesizedAnswer

logger = logging.getLogger(__name__)


@dataclass
class QueryResult:
    """Complete query result with all stages."""
    query: str
    retrieved_chunks: List[RetrievalResult]
    reranked_chunks: List[RetrievalResult]
    pathway_candidates: List[PathwayLink]
    synthesized_answer: SynthesizedAnswer


class QueryPipeline:
    """End-to-end query pipeline."""

    def __init__(
        self,
        retriever: Optional[HybridRetriever] = None,
        reranker: Optional[CrossEncoderReranker] = None,
        pathway_linker: Optional[PathwayLinker] = None,
        synthesizer: Optional[AnswerSynthesizer] = None,
        settings=None,
    ):
        self.settings = settings or get_settings()
        self.retriever = retriever
        self.reranker = reranker or CrossEncoderReranker(settings=self.settings)
        self._pathway_linker = pathway_linker
        self._pathway_linker_initialized = False
        self._pathway_linker_lock = threading.Lock()
        self.synthesizer = synthesizer or AnswerSynthesizer(settings=self.settings)

    def _ensure_retriever(self) -> HybridRetriever:
        if self.retriever is None:
            from src.embeddings.hybrid_indexer import HybridIndexer
            from src.embeddings.model_loader import load_embedding_model

            indexer = HybridIndexer(settings=self.settings)
            indexer.initialize()
            indexer.load_bm25_index()

            embedder_model = load_embedding_model(settings=self.settings)
            from src.retrieval.hybrid_retriever import HybridRetriever
            self.retriever = HybridRetriever(
                indexer=indexer,
                embedder_model=embedder_model,
                settings=self.settings,
            )
        return self.retriever

    def _ensure_pathway_linker(self) -> PathwayLinker:
        """Initialize pathway linker once per pipeline instance (thread-safe)."""
        if self._pathway_linker is None:
            with self._pathway_linker_lock:
                if self._pathway_linker is None:
                    linker = PathwayLinker(settings=self.settings)
                    linker.load_pathway_mapping()
                    # Load interactions from corpus once
                    from src.retrieval.pathway_linker import load_pathway_linker_from_corpus
                    linker = load_pathway_linker_from_corpus(
                        self.settings.paths.unified_corpus / "corpus.jsonl",
                        settings=self.settings,
                    )
                    self._pathway_linker = linker
                    self._pathway_linker_initialized = True
        return self._pathway_linker

    def invalidate_pathway_linker(self) -> None:
        """Invalidate cached pathway linker (e.g., after corpus update)."""
        with self._pathway_linker_lock:
            self._pathway_linker = None
            self._pathway_linker_initialized = False

    def query(
        self,
        query: str,
        top_k: Optional[int] = None,
        use_rerank: bool = True,
        use_pathway: bool = True,
        metadata_filter: Optional[dict] = None,
    ) -> QueryResult:
        """Run full query pipeline."""
        top_k = top_k or self.settings.retrieval.top_k_initial
        retriever = self._ensure_retriever()

        # Stage 1: Hybrid retrieval
        logger.info(f"Retrieving for query: {query}")
        retrieved = retriever.retrieve(query, top_k=top_k, metadata_filter=metadata_filter)
        logger.info(f"Retrieved {len(retrieved)} chunks")

        # Stage 2: Reranking
        reranked = retrieved
        if use_rerank and retrieved:
            logger.info("Reranking results")
            reranked = self.reranker.rerank(query, retrieved, top_k=self.settings.retrieval.top_k_final)
            logger.info(f"Reranked to {len(reranked)} chunks")

        # Stage 3: Pathway linking
        pathway_candidates = []
        if use_pathway:
            logger.info("Linking via pathways")
            linker = self._ensure_pathway_linker()
            # Load interactions from retriever's indexer
            from src.retrieval.pathway_linker import load_pathway_linker_from_corpus
            linker = load_pathway_linker_from_corpus(
                self.settings.paths.unified_corpus / "corpus.jsonl",
                settings=self.settings,
            )
            # Extract disease ID from query (simplified - in practice use NER)
            disease_id = self._extract_disease_id(query)
            if disease_id:
                pathway_candidates = linker.rank_repurposing_candidates(
                    disease_id, query, reranked
                )
            logger.info(f"Found {len(pathway_candidates)} pathway candidates")

        # Stage 4: Answer synthesis
        logger.info("Synthesizing answer")
        synthesized = self.synthesizer.synthesize(query, reranked)

        return QueryResult(
            query=query,
            retrieved_chunks=retrieved,
            reranked_chunks=reranked,
            pathway_candidates=pathway_candidates,
            synthesized_answer=synthesized,
        )

    def _extract_disease_id(self, query: str) -> Optional[str]:
        """Extract disease ID from query using synonyms and abbreviations with word boundaries."""
        # Comprehensive disease synonym map: canonical_name -> {efo_id, synonyms}
        # Synonyms include common abbreviations and alternative names
        disease_synonyms = {
            "EFO_0000249": {
                "synonyms": [
                    "alzheimer", "alzheimer's", "alzheimers", "alzheimer disease",
                    "ad", "alz", "dementia of the alzheimer type", "dat"
                ],
                "efo_id": "EFO_0000249"
            },
            "EFO_0001379": {
                "synonyms": [
                    "myocardial infarction", "mi", "heart attack", "acute myocardial infarction",
                    "ami", "stemi", "nstemi", "heart infarction"
                ],
                "efo_id": "EFO_0001379"
            },
            "EFO_0000305": {
                "synonyms": [
                    "multiple myeloma", "mm", "kahler disease", "myeloma",
                    "plasma cell myeloma", "myelomatosis"
                ],
                "efo_id": "EFO_0000305"
            },
            "EFO_0000729": {
                "synonyms": [
                    "diabetes", "diabetes mellitus", "dm", "type 2 diabetes", "t2d",
                    "type 2 diabetes mellitus", "t2dm", "non-insulin dependent diabetes",
                    "niddm", "adult onset diabetes"
                ],
                "efo_id": "EFO_0000729"
            },
            "EFO_0003767": {
                "synonyms": [
                    "parkinson", "parkinson's", "parkinsons", "parkinson disease",
                    "pd", "parkinson's disease", "paralysis agitans", "pd disease"
                ],
                "efo_id": "EFO_0003767"
            },
            "EFO_0000729": {  # Type 1 diabetes
                "synonyms": [
                    "type 1 diabetes", "t1d", "insulin dependent diabetes", "iddm",
                    "juvenile diabetes", "type 1 diabetes mellitus"
                ],
                "efo_id": "EFO_0000407"  # Type 1 diabetes mellitus
            },
            "EFO_0001072": {
                "synonyms": [
                    "breast cancer", "breast carcinoma", "bc", "malignant neoplasm of breast"
                ],
                "efo_id": "EFO_0001072"
            },
            "EFO_0000349": {
                "synonyms": [
                    "lung cancer", "lung carcinoma", "lc", "nsclc", "sclc",
                    "non-small cell lung cancer", "small cell lung cancer"
                ],
                "efo_id": "EFO_0000349"
            },
            "EFO_0001097": {
                "synonyms": [
                    "colorectal cancer", "colon cancer", "rectal cancer", "crc",
                    "colorectal carcinoma", "bowel cancer"
                ],
                "efo_id": "EFO_0001097"
            },
            "EFO_0002690": {
                "synonyms": [
                    "prostate cancer", "prostate carcinoma", "pca", "prostatic adenocarcinoma"
                ],
                "efo_id": "EFO_0002690"
            },
            "EFO_0000618": {
                "synonyms": [
                    "hypertension", "high blood pressure", "htn", "essential hypertension"
                ],
                "efo_id": "EFO_0000618"
            },
            "EFO_0003776": {
                "synonyms": [
                    "heart failure", "chf", "congestive heart failure", "cardiac failure"
                ],
                "efo_id": "EFO_0003776"
            },
            "EFO_0000729": {  # Chronic kidney disease
                "synonyms": [
                    "chronic kidney disease", "ckd", "renal failure", "chronic renal failure",
                    "end stage renal disease", "esrd"
                ],
                "efo_id": "EFO_0000729"
            },
        }

        query_lower = query.lower()
        # Use word boundaries for more precise matching
        import re

        for canonical, info in disease_synonyms.items():
            efo_id = info["efo_id"]
            for synonym in info["synonyms"]:
                # Use word boundary regex for precise matching
                pattern = r'\b' + re.escape(synonym.lower()) + r'\b'
                if re.search(pattern, query_lower):
                    return efo_id

        return None


def run_query(
    query: str,
    top_k: Optional[int] = None,
    use_rerank: bool = True,
    use_pathway: bool = True,
    use_mock: bool = False,
    settings=None,
) -> QueryResult:
    """Convenience function to run query pipeline."""
    if settings is None:
        settings = get_settings()

    pipeline = QueryPipeline(
        settings=settings,
        reranker=CrossEncoderReranker(use_mock=use_mock, settings=settings),
        synthesizer=AnswerSynthesizer(use_mock=use_mock, settings=settings),
    )
    return pipeline.query(query, top_k, use_rerank, use_pathway)