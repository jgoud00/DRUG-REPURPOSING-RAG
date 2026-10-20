"""Configuration loader using Pydantic Settings."""

from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class PathsConfig(BaseSettings):
    project_root: Path = Field(default_factory=lambda: Path(__file__).parents[3])
    data_raw: Path = Field(default=Path("data/raw"))
    data_processed: Path = Field(default=Path("data/processed"))
    data_embeddings: Path = Field(default=Path("data/embeddings"))
    pubmed_raw: Path = Field(default=Path("data/raw/pubmed"))
    chembl_raw: Path = Field(default=Path("data/raw/chembl"))
    drugbank_raw: Path = Field(default=Path("data/raw/drugbank"))
    abstracts_processed: Path = Field(default=Path("data/processed/abstracts"))
    interactions_processed: Path = Field(default=Path("data/processed/interactions"))
    unified_corpus: Path = Field(default=Path("data/processed/unified"))
    vector_store: Path = Field(default=Path("data/embeddings/chromadb"))
    logs: Path = Field(default=Path("logs"))
    cache: Path = Field(default=Path(".cache"))

    def __init__(self, **data: Any):
        super().__init__(**data)
        # Resolve all paths relative to project_root
        for field_name, field_value in self.model_dump().items():
            if isinstance(field_value, Path) and not field_value.is_absolute():
                setattr(self, field_name, self.project_root / field_value)


class EmbeddingConfig(BaseSettings):
    model_name: str = "pritamdeka/S-PubMedBert-MS-MARCO"
    model_dim: int = 768
    device: str = "auto"
    batch_size: int = 32
    max_length: int = 512
    normalize_embeddings: bool = True
    use_fp16: bool = True
    cache_dir: str = ".cache/huggingface"
    alternative_models: Dict[str, str] = Field(default_factory=dict)


class VectorStoreConfig(BaseSettings):
    dense: Dict[str, Any] = Field(default_factory=dict)
    sparse: Dict[str, Any] = Field(default_factory=dict)
    hybrid: Dict[str, Any] = Field(default_factory=dict)


class ChunkingConfig(BaseSettings):
    abstract_chunk_size: int = 256
    abstract_chunk_overlap: int = 50
    abstract_min_chunk_size: int = 50
    use_semantic_chunking: bool = True
    sentence_splitter: str = "en_core_sci_sm"
    structured_template: str = (
        "{drug_name} ({drug_id}) {action_type} {target_name} ({target_uniprot}) "
        "implicated in {disease_name} via {mechanism}. Pathways: {pathway_names}."
    )
    include_row_level: bool = True
    include_entity_summaries: bool = True
    serialize_lists_as_csv: bool = True


class RetrievalConfig(BaseSettings):
    hybrid: Dict[str, Any] = Field(default_factory=dict)
    top_k_initial: int = 50
    top_k_final: int = 10
    metadata_filters: Dict[str, Any] = Field(default_factory=dict)


class RerankingConfig(BaseSettings):
    enabled: bool = True
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    batch_size: int = 16
    max_length: int = 512
    top_k: int = 10


class PathwayLinkerConfig(BaseSettings):
    enabled: bool = True
    pathway_sources: list[str] = Field(default_factory=lambda: ["reactome"])
    reactome: Dict[str, Any] = Field(default_factory=dict)
    disease_pathway_map: str = "config/disease_pathway_map.tsv"
    min_shared_pathways: int = 1
    confidence_threshold: float = 0.5


class LLMConfig(BaseSettings):
    provider: str = "ollama"
    model: str = "llama3.1:8b"
    base_url: str = "http://localhost:11434"
    temperature: float = 0.1
    max_tokens: int = 2048
    top_p: float = 0.9
    repeat_penalty: float = 1.1
    openai_fallback: Dict[str, Any] = Field(default_factory=dict)


class PromptConfig(BaseSettings):
    system_template: str = "config/prompts/system.txt"
    user_template: str = "config/prompts/user.txt"
    citation_format: str = "[{doc_id}]"
    require_citations: bool = True
    max_context_chunks: int = 8


class GenerationConfig(BaseSettings):
    llm: LLMConfig = Field(default_factory=LLMConfig)
    prompt: PromptConfig = Field(default_factory=PromptConfig)


class EvaluationConfig(BaseSettings):
    benchmarks: list[str] = Field(default_factory=lambda: ["repo_db", "drugcentral", "custom_cases"])
    metrics: list[str] = Field(default_factory=lambda: ["recall_at_k", "mrr", "ndcg", "citation_precision", "hallucination_rate"])
    k_values: list[int] = Field(default_factory=lambda: [1, 3, 5, 10])
    entailment_model: str = "microsoft/BiomedNLI"


class PubMedIngestionConfig(BaseSettings):
    batch_size: int = 1000
    max_abstracts: Optional[int] = None
    required_fields: list[str] = Field(default_factory=lambda: ["pmid", "title", "abstract"])
    mesh_terms: bool = True


class ChEMBLIngestionConfig(BaseSettings):
    version: str = "33"
    batch_size: int = 5000
    tables: list[str] = Field(default_factory=lambda: [
        "activities", "target_dictionary", "compound_records",
        "mechanism_of_action", "target_components"
    ])
    confidence_threshold: int = 5


class DrugBankIngestionConfig(BaseSettings):
    enabled: bool = False
    version: str = "5.1.11"


class EntityLinkingConfig(BaseSettings):
    enabled: bool = False
    model: str = "en_core_sci_sm"


class IngestionConfig(BaseSettings):
    pubmed: PubMedIngestionConfig = Field(default_factory=PubMedIngestionConfig)
    chembl: ChEMBLIngestionConfig = Field(default_factory=ChEMBLIngestionConfig)
    drugbank: DrugBankIngestionConfig = Field(default_factory=DrugBankIngestionConfig)
    entity_linking: EntityLinkingConfig = Field(default_factory=EntityLinkingConfig)


class LoggingConfig(BaseSettings):
    level: str = "INFO"
    format: str = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    file: str = "logs/rag_engine.log"
    max_bytes: int = 10485760
    backup_count: int = 5


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        nested_model_default_partial_update=True,
    )

    paths: PathsConfig = Field(default_factory=PathsConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    vector_store: VectorStoreConfig = Field(default_factory=VectorStoreConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    reranking: RerankingConfig = Field(default_factory=RerankingConfig)
    pathway_linker: PathwayLinkerConfig = Field(default_factory=PathwayLinkerConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    ingestion: IngestionConfig = Field(default_factory=IngestionConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)


def get_settings(config_path: Optional[str] = None) -> Settings:
    """Load settings from YAML config file and environment variables."""
    if config_path is None:
        config_path = Path(__file__).parents[1] / "config" / "settings.yaml"

    with open(config_path, "r") as f:
        yaml_config = yaml.safe_load(f)

    return Settings(**yaml_config)