"""
Configuration system for RAG components.

Defines configuration schemas for all RAG pipeline components following
LangChain patterns and best practices.
"""

from typing import Dict, Any, Optional, List, Union
from pydantic import BaseModel, Field, validator
from enum import Enum
from pathlib import Path

# Import citation-related enums and configs from main config
try:
    from ..config.schemas import CitationConfig, CitationVerbosity, CitationFormat, ResourceType as MainResourceType
    _CITATION_AVAILABLE = True
except ImportError:
    _CITATION_AVAILABLE = False
    CitationConfig = None
    CitationVerbosity = None 
    CitationFormat = None
    MainResourceType = None


class LoaderType(str, Enum):
    """Supported document loader types."""
    WEB = "web"
    CRAWL4AI_WEB = "crawl4ai_web"
    HTML = "html"
    PDF = "pdf"
    MARKDOWN = "markdown"
    JSON = "json"
    CSV = "csv"
    DIRECTORY = "directory"
    CONFLUENCE = "confluence"
    NOTION = "notion"
    GITHUB = "github"


class SplitterType(str, Enum):
    """Supported text splitter types."""
    CHARACTER = "character"
    RECURSIVE_CHARACTER = "recursive_character"
    SEMANTIC = "semantic"
    MARKDOWN = "markdown"
    HTML = "html"
    CODE = "code"
    JSON = "json"
    TOKEN = "token"


class EmbeddingType(str, Enum):
    """Supported embedding types."""
    OPENAI = "openai"
    HUGGINGFACE = "huggingface"
    SENTENCE_TRANSFORMERS = "sentence_transformers"
    COHERE = "cohere"
    AZURE_OPENAI = "azure_openai"


class VectorStoreType(str, Enum):
    """Supported vector store types."""
    CHROMA = "chroma"
    FAISS = "faiss"
    PINECONE = "pinecone"
    WEAVIATE = "weaviate"
    QDRANT = "qdrant"
    PGVECTOR = "pgvector"


class RetrieverType(str, Enum):
    """Supported retriever types."""
    SIMILARITY = "similarity"
    MMR = "mmr"
    SIMILARITY_SCORE_THRESHOLD = "similarity_score_threshold"
    ENSEMBLE = "ensemble"
    MULTI_QUERY = "multi_query"
    PARENT_DOCUMENT = "parent_document"


class ResourceType(str, Enum):
    """Types of resources in the knowledge base."""
    DSL = "dsl"
    CONTEXTUAL = "contextual"
    GUIDELINES = "guidelines"
    DOMAIN_RULES = "domain_rules"


class LoaderConfig(BaseModel):
    """Configuration for document loaders."""
    type: LoaderType
    params: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        use_enum_values = True


class SplitterConfig(BaseModel):
    """Configuration for text splitters."""
    type: SplitterType
    chunk_size: int = 1000
    chunk_overlap: int = 200
    params: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        use_enum_values = True


class EmbeddingConfig(BaseModel):
    """Configuration for embeddings."""
    type: EmbeddingType
    model_name: str
    params: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        use_enum_values = True


class VectorStoreConfig(BaseModel):
    """Configuration for vector stores."""
    type: VectorStoreType
    persist_directory: Optional[Path] = None
    collection_name: str = "paas_ai_rag"
    params: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        use_enum_values = True


class RetrieverConfig(BaseModel):
    """Configuration for retrievers."""
    type: RetrieverType
    search_kwargs: Dict[str, Any] = Field(default_factory=lambda: {"k": 5})
    params: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        use_enum_values = True


class ResourceConfig(BaseModel):
    """Configuration for a single knowledge base resource."""
    url: str
    resource_type: ResourceType
    loader: LoaderConfig
    splitter: SplitterConfig
    priority: int = Field(default=1, ge=1, le=10)
    tags: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        use_enum_values = True
        
    @validator('url')
    def validate_url(cls, v):
        """Validate URL format."""
        if not v.strip():
            raise ValueError("URL cannot be empty")
        return v.strip()


class Config(BaseModel):
    """Main configuration for the RAG system."""
    embedding: EmbeddingConfig
    vectorstore: VectorStoreConfig
    retriever: RetrieverConfig
    
    # Citation configuration (optional for backward compatibility)
    citation: Optional[CitationConfig] = Field(
        default_factory=lambda: CitationConfig() if _CITATION_AVAILABLE else None
    )
    
    # Pipeline settings
    batch_size: int = Field(default=32, ge=1)
    max_retries: int = Field(default=3, ge=0)
    timeout: int = Field(default=300, ge=1)
    
    # Validation settings
    validate_urls: bool = True
    skip_invalid_docs: bool = True
    
    # Logging settings
    verbose: bool = False
    log_level: str = "INFO"
    
    class ConfigMeta:
        use_enum_values = True


# Default configuration presets
DEFAULT_CONFIGS = {
    "default": Config(
        embedding=EmbeddingConfig(
            type=EmbeddingType.OPENAI,
            model_name="text-embedding-3-small"
        ),
        vectorstore=VectorStoreConfig(
            type=VectorStoreType.CHROMA,
            persist_directory=Path("./rag_data/chroma")
        ),
        retriever=RetrieverConfig(
            type=RetrieverType.SIMILARITY,
            search_kwargs={"k": 5}
        )
    ),
    
    "local": Config(
        embedding=EmbeddingConfig(
            type=EmbeddingType.SENTENCE_TRANSFORMERS,
            model_name="all-MiniLM-L6-v2"
        ),
        vectorstore=VectorStoreConfig(
            type=VectorStoreType.CHROMA,
            persist_directory=Path("./rag_data/chroma_local"),
            collection_name="paas_ai_local"
        ),
        retriever=RetrieverConfig(
            type=RetrieverType.SIMILARITY,
            search_kwargs={"k": 5}
        )
    ),
    
    "production": Config(
        embedding=EmbeddingConfig(
            type=EmbeddingType.OPENAI,
            model_name="text-embedding-3-large"
        ),
        vectorstore=VectorStoreConfig(
            type=VectorStoreType.PINECONE,
            collection_name="paas-ai-production"
        ),
        retriever=RetrieverConfig(
            type=RetrieverType.ENSEMBLE,
            search_kwargs={"k": 10}
        ),
        batch_size=64,
        validate_urls=True
    )
}


def get_default_loader_config(url: str) -> LoaderConfig:
    """Get default loader configuration based on URL."""
    url_lower = url.lower()
    
    if url_lower.endswith('.pdf'):
        return LoaderConfig(type=LoaderType.PDF)
    elif url_lower.endswith('.md') or url_lower.endswith('.markdown'):
        return LoaderConfig(type=LoaderType.MARKDOWN)
    elif url_lower.endswith('.json'):
        return LoaderConfig(type=LoaderType.JSON)
    elif url_lower.endswith('.csv'):
        return LoaderConfig(type=LoaderType.CSV)
    elif 'confluence' in url_lower:
        return LoaderConfig(type=LoaderType.CONFLUENCE)
    elif 'notion' in url_lower:
        return LoaderConfig(type=LoaderType.NOTION)
    elif 'github.com' in url_lower:
        return LoaderConfig(type=LoaderType.GITHUB)
    elif url_lower.startswith(('http://', 'https://')):
        return LoaderConfig(type=LoaderType.CRAWL4AI_WEB)
    else:
        # Assume local directory or file
        return LoaderConfig(type=LoaderType.DIRECTORY)


def get_default_splitter_config(loader_type: LoaderType) -> SplitterConfig:
    """Get default splitter configuration based on loader type."""
    if loader_type == LoaderType.MARKDOWN:
        return SplitterConfig(
            type=SplitterType.MARKDOWN,
            chunk_size=1000,
            chunk_overlap=200
        )
    elif loader_type == LoaderType.HTML:
        return SplitterConfig(
            type=SplitterType.HTML,
            chunk_size=1000,
            chunk_overlap=200
        )
    elif loader_type == LoaderType.JSON:
        return SplitterConfig(
            type=SplitterType.JSON,
            chunk_size=1000,
            chunk_overlap=100
        )
    elif loader_type in [LoaderType.GITHUB, LoaderType.DIRECTORY]:
        return SplitterConfig(
            type=SplitterType.CODE,
            chunk_size=1500,
            chunk_overlap=300
        )
    else:
        return SplitterConfig(
            type=SplitterType.RECURSIVE_CHARACTER,
            chunk_size=1000,
            chunk_overlap=200
        ) 