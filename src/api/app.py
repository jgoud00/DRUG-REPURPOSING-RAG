"""FastAPI service for Drug Repurposing RAG Engine."""

import logging
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.config import get_settings
from src.pipeline.query import QueryPipeline, QueryResult, run_query

logger = logging.getLogger(__name__)

# Global pipeline instance
_pipeline: Optional[QueryPipeline] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize pipeline on startup."""
    global _pipeline
    logger.info("Initializing query pipeline...")
    settings = get_settings()
    _pipeline = QueryPipeline(settings=settings)
    logger.info("Query pipeline ready")
    yield
    logger.info("Shutting down...")


app = FastAPI(
    title="Drug Repurposing RAG Engine",
    description="Retrieval-Augmented Generation for Computational Drug Repurposing",
    version="0.1.0",
    lifespan=lifespan,
)


class QueryRequest(BaseModel):
    """Request model for query endpoint."""
    query: str = Field(..., description="Disease or target query string", min_length=3)
    top_k: Optional[int] = Field(None, description="Number of results to retrieve", ge=1, le=100)
    use_rerank: bool = Field(True, description="Enable cross-encoder reranking")
    use_pathway: bool = Field(True, description="Enable pathway-based linking")
    use_mock: bool = Field(False, description="Use mock LLM/embeddings for testing")


class CitationResponse(BaseModel):
    """Citation in response."""
    doc_id: str
    source: str
    text_snippet: str
    retrieval_score: float
    pmid: Optional[str] = None
    drug_id: Optional[str] = None
    target_name: Optional[str] = None
    disease_name: Optional[str] = None
    confidence_score: Optional[float] = None


class DrugCandidateResponse(BaseModel):
    """Drug candidate in response."""
    drug_name: str
    drug_id: str
    target_name: Optional[str] = None
    target_id: Optional[str] = None
    target_uniprot: Optional[str] = None
    action_type: Optional[str] = None
    mechanism: Optional[str] = None
    disease_name: Optional[str] = None
    confidence_score: Optional[float] = None


class PathwayCandidateResponse(BaseModel):
    """Pathway-linked drug candidate."""
    drug_name: str
    drug_id: str
    target_name: str
    target_uniprot: str
    shared_pathways: List[str]
    pathway_names: List[str]
    confidence_score: float
    mechanism: str
    action_type: str


class QueryResponse(BaseModel):
    """Response model for query endpoint."""
    query: str
    answer: str
    citations: List[CitationResponse]
    drug_candidates: List[DrugCandidateResponse]
    pathway_candidates: List[PathwayCandidateResponse]
    hallucination_rate: float
    citation_precision: float
    num_retrieved: int
    num_reranked: int


def _citation_to_response(citation) -> CitationResponse:
    """Convert internal Citation to response model."""
    return CitationResponse(
        doc_id=citation.doc_id,
        source=citation.source,
        text_snippet=citation.text_snippet,
        retrieval_score=citation.retrieval_score,
        pmid=citation.pmid,
        drug_id=citation.drug_id,
        target_name=citation.target_name,
        disease_name=citation.disease_name,
        confidence_score=citation.confidence_score,
    )


def _drug_to_response(drug) -> DrugCandidateResponse:
    """Convert internal drug candidate to response model."""
    return DrugCandidateResponse(
        drug_name=drug.get("drug_name", ""),
        drug_id=drug.get("drug_id", ""),
        target_name=drug.get("target_name"),
        target_id=drug.get("target_id"),
        target_uniprot=drug.get("target_uniprot"),
        action_type=drug.get("action_type"),
        mechanism=drug.get("mechanism"),
        disease_name=drug.get("disease_name"),
        confidence_score=drug.get("confidence_score"),
    )


def _pathway_to_response(link) -> PathwayCandidateResponse:
    """Convert PathwayLink to response model."""
    return PathwayCandidateResponse(
        drug_name=link.drug_name,
        drug_id=link.drug_id,
        target_name=link.target_name,
        target_uniprot=link.target_uniprot,
        shared_pathways=link.shared_pathways,
        pathway_names=link.pathway_names,
        confidence_score=link.confidence_score,
        mechanism=link.mechanism,
        action_type=link.action_type,
    )


@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    """Execute a drug repurposing query."""
    global _pipeline
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    try:
        logger.info(f"Received query: {request.query}")

        result: QueryResult = _pipeline.query(
            query=request.query,
            top_k=request.top_k,
            use_rerank=request.use_rerank,
            use_pathway=request.use_pathway,
        )

        # Convert internal models to response models
        citations = [_citation_to_response(c) for c in result.synthesized_answer.citations]
        drug_candidates = [_drug_to_response(d) for d in result.synthesized_answer.drug_candidates]
        pathway_candidates = [_pathway_to_response(p) for p in result.pathway_candidates]

        return QueryResponse(
            query=request.query,
            answer=result.synthesized_answer.answer,
            citations=citations,
            drug_candidates=drug_candidates,
            pathway_candidates=pathway_candidates,
            hallucination_rate=result.synthesized_answer.hallucination_rate,
            citation_precision=result.synthesized_answer.citation_precision,
            num_retrieved=len(result.retrieved_chunks),
            num_reranked=len(result.reranked_chunks),
        )

    except Exception as e:
        logger.error(f"Query failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "drug-repurposing-rag"}


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "Drug Repurposing RAG Engine",
        "version": "0.1.0",
        "docs": "/docs",
        "query_endpoint": "POST /query",
    }


def create_app(settings=None) -> FastAPI:
    """Create FastAPI app for testing."""
    global _pipeline
    settings = settings or get_settings()
    _pipeline = QueryPipeline(settings=settings)
    return app