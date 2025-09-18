"""
Tests for RAG configuration system.
"""

import pytest
from pathlib import Path

from src.paas_ai.core.rag.config import (
    Config,
    ResourceConfig,
    LoaderType,
    SplitterType,
    EmbeddingType,
    VectorStoreType,
    RetrieverType,
    ResourceType,
    DEFAULT_CONFIGS,
    get_default_loader_config,
    get_default_splitter_config,
)


class TestConfig:
    """Test RAG configuration."""
    
    def test_default_config_creation(self):
        """Test creating default RAG config."""
        config = DEFAULT_CONFIGS['default']
        
        assert config.embedding.type == EmbeddingType.OPENAI
        assert config.vectorstore.type == VectorStoreType.CHROMA
        assert config.retriever.type == RetrieverType.SIMILARITY
        assert config.batch_size == 32
        assert config.validate_urls is True
    
    def test_local_config_creation(self):
        """Test creating local RAG config."""
        config = DEFAULT_CONFIGS['local']
        
        assert config.embedding.type == EmbeddingType.SENTENCE_TRANSFORMERS
        assert config.vectorstore.type == VectorStoreType.CHROMA
        assert config.retriever.type == RetrieverType.SIMILARITY


class TestResourceConfig:
    """Test resource configuration."""
    
    def test_resource_config_creation(self):
        """Test creating a resource config."""
        from src.paas_ai.core.rag.config import LoaderConfig, SplitterConfig
        
        loader_config = LoaderConfig(type=LoaderType.WEB)
        splitter_config = SplitterConfig(type=SplitterType.RECURSIVE_CHARACTER)
        
        resource = ResourceConfig(
            url="https://example.com",
            resource_type=ResourceType.DSL,
            loader=loader_config,
            splitter=splitter_config,
            priority=5,
            tags=["test", "example"]
        )
        
        assert resource.url == "https://example.com"
        assert resource.resource_type == ResourceType.DSL
        assert resource.priority == 5
        assert resource.tags == ["test", "example"]
    
    def test_url_validation(self):
        """Test URL validation."""
        from src.paas_ai.core.rag.config import LoaderConfig, SplitterConfig
        
        loader_config = LoaderConfig(type=LoaderType.WEB)
        splitter_config = SplitterConfig(type=SplitterType.RECURSIVE_CHARACTER)
        
        # Empty URL should raise validation error
        with pytest.raises(ValueError, match="URL cannot be empty"):
            ResourceConfig(
                url="",
                resource_type=ResourceType.DSL,
                loader=loader_config,
                splitter=splitter_config
            )


class TestDefaultConfigs:
    """Test default configuration helpers."""
    
    def test_get_default_loader_config_web(self):
        """Test getting default loader config for web URLs."""
        config = get_default_loader_config("https://example.com")
        assert config.type == "crawl4ai_web"
    
    def test_get_default_loader_config_pdf(self):
        """Test getting default loader config for PDF files."""
        config = get_default_loader_config("document.pdf")
        assert config.type == LoaderType.PDF
    
    def test_get_default_loader_config_markdown(self):
        """Test getting default loader config for Markdown files."""
        config = get_default_loader_config("readme.md")
        assert config.type == LoaderType.MARKDOWN
    
    def test_get_default_loader_config_github(self):
        """Test getting default loader config for GitHub URLs."""
        config = get_default_loader_config("https://github.com/user/repo")
        assert config.type == LoaderType.GITHUB
    
    def test_get_default_splitter_config(self):
        """Test getting default splitter config."""
        config = get_default_splitter_config(LoaderType.MARKDOWN)
        assert config.type == SplitterType.MARKDOWN
        assert config.chunk_size == 1000
        assert config.chunk_overlap == 200 