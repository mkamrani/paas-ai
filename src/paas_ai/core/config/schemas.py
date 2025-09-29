"""
Configuration schemas for PaaS AI.

Defines configuration data models for all system components.
"""

from enum import Enum
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


# Enums for configuration options
class EmbeddingType(str, Enum):
    OPENAI = "openai"
    SENTENCE_TRANSFORMERS = "sentence_transformers"
    COHERE = "cohere"


class VectorStoreType(str, Enum):
    CHROMA = "chroma"
    FAISS = "faiss"
    PINECONE = "pinecone"


class RetrieverType(str, Enum):
    SIMILARITY = "similarity"
    MMR = "mmr"
    ENSEMBLE = "ensemble"


class LoaderType(str, Enum):
    WEB = "web"
    PDF = "pdf"
    MARKDOWN = "markdown"
    TEXT = "text"
    DIRECTORY = "directory"
    HTML = "html"
    JSON = "json"
    CSV = "csv"
    CONFLUENCE = "confluence"
    NOTION = "notion"
    GITHUB = "github"


class SplitterType(str, Enum):
    RECURSIVE_CHARACTER = "recursive_character"
    CHARACTER = "character"
    MARKDOWN = "markdown"
    TOKEN = "token"
    HTML = "html"
    JSON = "json"
    CODE = "code"


class ResourceType(str, Enum):
    """Type of resource content for RAG knowledge base."""

    DSL = "dsl"
    CONTEXTUAL = "contextual"
    GUIDELINES = "guidelines"
    DOMAIN_RULES = "domain_rules"


class ValidatorType(str, Enum):
    """Type of content validator for processing pipeline."""

    CONTENT = "content"
    QUALITY = "quality"
    RELEVANCE = "relevance"


class ContentValidatorType(str, Enum):
    """Type of content validator for processing pipeline."""

    CONTENT = "content"


class CitationVerbosity(str, Enum):
    """Citation verbosity levels."""

    NONE = "none"  # No citations (current behavior)
    MINIMAL = "minimal"  # Just source name
    STANDARD = "standard"  # Source + location (page/section)
    DETAILED = "detailed"  # Full reference with context
    FORENSIC = "forensic"  # Exact quotes + precise locations


class CitationFormat(str, Enum):
    """Citation output formats."""

    INLINE = "inline"  # [Source, Page 5]
    FOOTNOTE = "footnote"  # Reference number with footnote
    ACADEMIC = "academic"  # Full academic citation
    STRUCTURED = "structured"  # JSON/dict format


# Configuration models
class EmbeddingConfig(BaseModel):
    type: EmbeddingType
    model_name: str = "text-embedding-3-small"
    params: Dict[str, Any] = Field(default_factory=dict)


class LLMConfig(BaseModel):
    """Configuration for Language Model."""

    provider: str = "openai"  # openai, azure, anthropic, etc.
    model_name: str = "gpt-3.5-turbo"
    temperature: float = 0.1
    max_tokens: Optional[int] = None
    api_key_env_var: str = "OPENAI_API_KEY"  # Environment variable name for API key
    params: Dict[str, Any] = Field(default_factory=dict)


class VectorStoreConfig(BaseModel):
    type: VectorStoreType
    persist_directory: Optional[str] = None
    collection_name: str = "paas_ai_knowledge"
    params: Dict[str, Any] = Field(default_factory=dict)


class RetrieverConfig(BaseModel):
    type: RetrieverType
    search_kwargs: Dict[str, Any] = Field(default_factory=lambda: {"k": 4})
    params: Dict[str, Any] = Field(default_factory=dict)


class LoaderConfig(BaseModel):
    type: LoaderType
    params: Dict[str, Any] = Field(default_factory=dict)


class SplitterConfig(BaseModel):
    type: SplitterType
    chunk_size: int = 1000
    chunk_overlap: int = 200
    params: Dict[str, Any] = Field(default_factory=dict)


class ContentValidatorConfig(BaseModel):
    """Configuration for content validators."""

    type: ContentValidatorType
    min_content_length: int = 10
    max_content_length: int = 1000000
    skip_empty: bool = True
    params: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        use_enum_values = True


class CitationConfig(BaseModel):
    """Citation system configuration."""

    enabled: bool = True
    verbosity: CitationVerbosity = CitationVerbosity.STANDARD
    format: CitationFormat = CitationFormat.INLINE

    # Resource-specific verbosity overrides
    resource_overrides: Dict[ResourceType, CitationVerbosity] = Field(default_factory=dict)

    # Citation content preferences
    include_quotes: bool = True
    max_quote_length: int = 150
    include_confidence: bool = False
    generate_deep_links: bool = True

    # Strategy mapping per resource type
    strategies: Dict[ResourceType, str] = Field(
        default_factory=lambda: {
            ResourceType.DSL: "technical_citation",
            ResourceType.CONTEXTUAL: "web_citation",
            ResourceType.GUIDELINES: "policy_citation",
            ResourceType.DOMAIN_RULES: "rule_citation",
        }
    )

    # Base URLs for deep link generation
    base_urls: Dict[str, str] = Field(default_factory=dict)

    class Config:
        use_enum_values = True


class ResourceConfig(BaseModel):
    """Configuration for a RAG resource."""

    url: str
    resource_type: ResourceType
    loader: LoaderConfig
    splitter: SplitterConfig
    tags: List[str] = Field(default_factory=list)
    priority: int = 5
    last_updated: Optional[str] = None


class ValidatorConfig(BaseModel):
    type: ValidatorType
    params: Dict[str, Any] = Field(default_factory=dict)


class PersistenceConfig(BaseModel):
    """Persistence and memory configuration."""

    enabled: bool = True
    checkpointer_type: Literal["memory", "sqlite", "postgres"] = "memory"

    # Checkpointer-specific settings
    sqlite_path: Optional[str] = None
    postgres_url: Optional[str] = None

    # Memory management
    max_history_length: int = 20  # Max conversation turns to keep
    cleanup_interval: int = 3600  # Seconds between cleanup runs


class MultiAgentConfig(BaseModel):
    """Multi-agent system configuration."""

    enabled: bool = True
    mode: Literal["supervisor", "swarm"] = "supervisor"
    default_agent: Literal["designer", "paas_manifest_generator"] = "designer"

    # Token tracking with callback support
    track_tokens: bool = False
    token_callback: Optional[str] = None  # "console", "json_file", "webhook", etc.

    # Verbosity control
    verbose: bool = False  # Controls overall verbosity across the MAS

    # Persistence configuration
    persistence: PersistenceConfig = Field(default_factory=lambda: PersistenceConfig())


class Config(BaseModel):
    """Main configuration for PaaS AI."""

    embedding: EmbeddingConfig
    vectorstore: VectorStoreConfig
    retriever: RetrieverConfig
    llm: LLMConfig = Field(default_factory=lambda: LLMConfig())
    loader: Optional[LoaderConfig] = None
    content_validator: Optional[ContentValidatorConfig] = None
    citation: CitationConfig = Field(default_factory=lambda: CitationConfig())
    batch_size: int = 32
    validate_urls: bool = True
    max_parallel: int = 5
    timeout: int = 30
    log_level: str = "INFO"

    # Multi-agent system configuration
    multi_agent: MultiAgentConfig = Field(default_factory=lambda: MultiAgentConfig())
    agents: Dict[str, Dict[str, Any]] = Field(
        default_factory=lambda: {
            "designer": {"model": "gpt-4o-mini", "temperature": 0.1},
            "paas_manifest_generator": {
                "model": "gpt-4o-mini",
                "temperature": 0.0,  # More deterministic for YAML generation
            },
        }
    )

    class Config:
        use_enum_values = True


class APIConfig(BaseModel):
    """API server configuration."""

    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: List[str] = Field(default_factory=lambda: ["*"])
    rate_limiting: Dict[str, Any] = Field(default_factory=dict)


class AgentConfig(BaseModel):
    """Agent system configuration."""

    max_iterations: int = 10
    memory_type: str = "buffer"
    tools: List[str] = Field(default_factory=list)
    model_provider: str = "openai"


class ConfigFile(BaseModel):
    """Configuration file structure with profile support."""

    current: str = Field(default="default", description="Name of the currently active profile")
    profiles: Dict[str, Config] = Field(
        default_factory=dict, description="Custom profile definitions"
    )

    class Config:
        extra = "forbid"  # Don't allow additional fields


# Predefined profiles
DEFAULT_CONFIG_PROFILES = {
    "default": Config(
        embedding=EmbeddingConfig(type=EmbeddingType.OPENAI, model_name="text-embedding-3-small"),
        vectorstore=VectorStoreConfig(
            type=VectorStoreType.CHROMA,
            persist_directory="rag_data/chroma",
            collection_name="paas_ai_rag",
        ),
        retriever=RetrieverConfig(type=RetrieverType.SIMILARITY, search_kwargs={"k": 4}),
        llm=LLMConfig(provider="openai", model_name="gpt-3.5-turbo", temperature=0.1),
        citation=CitationConfig(
            enabled=True, verbosity=CitationVerbosity.STANDARD, format=CitationFormat.INLINE
        ),
        batch_size=32,
        validate_urls=True,
    ),
    "local": Config(
        embedding=EmbeddingConfig(
            type=EmbeddingType.SENTENCE_TRANSFORMERS, model_name="all-MiniLM-L6-v2"
        ),
        vectorstore=VectorStoreConfig(
            type=VectorStoreType.CHROMA,
            persist_directory="rag_data/chroma_local",
            collection_name="paas_ai_local",
        ),
        retriever=RetrieverConfig(type=RetrieverType.SIMILARITY, search_kwargs={"k": 5}),
        llm=LLMConfig(
            provider="openai",
            model_name="gpt-3.5-turbo",
            temperature=0.0,  # More deterministic for local testing
            max_tokens=1000,
        ),
        citation=CitationConfig(
            enabled=True, verbosity=CitationVerbosity.MINIMAL, format=CitationFormat.INLINE
        ),
        batch_size=16,
        validate_urls=True,
    ),
    "production": Config(
        embedding=EmbeddingConfig(type=EmbeddingType.OPENAI, model_name="text-embedding-3-large"),
        vectorstore=VectorStoreConfig(
            type=VectorStoreType.PINECONE, collection_name="paas-ai-prod"
        ),
        retriever=RetrieverConfig(type=RetrieverType.ENSEMBLE, search_kwargs={"k": 10}),
        llm=LLMConfig(provider="openai", model_name="gpt-4", temperature=0.2, max_tokens=2000),
        citation=CitationConfig(
            enabled=True,
            verbosity=CitationVerbosity.DETAILED,
            format=CitationFormat.INLINE,
            include_confidence=True,
            resource_overrides={
                ResourceType.DSL: CitationVerbosity.FORENSIC,
                ResourceType.GUIDELINES: CitationVerbosity.DETAILED,
            },
        ),
        batch_size=64,
        validate_urls=True,
    ),
    "verbose": Config(
        embedding=EmbeddingConfig(
            type=EmbeddingType.SENTENCE_TRANSFORMERS, model_name="all-MiniLM-L6-v2"
        ),
        vectorstore=VectorStoreConfig(
            type=VectorStoreType.CHROMA,
            persist_directory="rag_data/chroma_local",
            collection_name="paas_ai_local",
        ),
        retriever=RetrieverConfig(type=RetrieverType.SIMILARITY, search_kwargs={"k": 5}),
        llm=LLMConfig(
            provider="openai", model_name="gpt-3.5-turbo", temperature=0.0, max_tokens=1000
        ),
        citation=CitationConfig(
            enabled=True,
            verbosity=CitationVerbosity.FORENSIC,
            format=CitationFormat.STRUCTURED,
            include_quotes=True,
            include_confidence=True,
            max_quote_length=200,
        ),
        batch_size=16,
        validate_urls=True,
        # Enable multi-agent verbose mode and token tracking
        multi_agent=MultiAgentConfig(
            enabled=True,
            mode="supervisor",
            default_agent="designer",
            track_tokens=True,
            token_callback="console",
            verbose=True,
        ),
    ),
}
