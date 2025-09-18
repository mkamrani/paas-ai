"""
Unit tests for core/config/schemas.py

Tests all configuration schemas including:
- Enums (EmbeddingType, VectorStoreType, etc.)
- Configuration models (EmbeddingConfig, LLMConfig, etc.)
- Main Config class
- ConfigFile class
- Default configuration profiles
"""

import pytest
from pydantic import ValidationError
from typing import Dict, Any, List
from pydantic import ValidationError

from src.paas_ai.core.config.schemas import (
    # Enums
    EmbeddingType,
    VectorStoreType,
    RetrieverType,
    LoaderType,
    SplitterType,
    ResourceType,
    ValidatorType,
    ContentValidatorType,
    CitationVerbosity,
    CitationFormat,
    
    # Configuration models
    EmbeddingConfig,
    LLMConfig,
    VectorStoreConfig,
    RetrieverConfig,
    LoaderConfig,
    SplitterConfig,
    ContentValidatorConfig,
    CitationConfig,
    ResourceConfig,
    ValidatorConfig,
    MultiAgentConfig,
    Config,
    APIConfig,
    AgentConfig,
    ConfigFile,
    
    # Default profiles
    DEFAULT_CONFIG_PROFILES,
)


class TestEnums:
    """Test all enum classes."""
    
    def test_embedding_type_enum(self):
        """Test EmbeddingType enum values."""
        assert EmbeddingType.OPENAI == "openai"
        assert EmbeddingType.SENTENCE_TRANSFORMERS == "sentence_transformers"
        assert EmbeddingType.COHERE == "cohere"
        
        # Test all values
        values = [e.value for e in EmbeddingType]
        assert set(values) == {"openai", "sentence_transformers", "cohere"}
    
    def test_vector_store_type_enum(self):
        """Test VectorStoreType enum values."""
        assert VectorStoreType.CHROMA == "chroma"
        assert VectorStoreType.FAISS == "faiss"
        assert VectorStoreType.PINECONE == "pinecone"
        
        values = [e.value for e in VectorStoreType]
        assert set(values) == {"chroma", "faiss", "pinecone"}
    
    def test_retriever_type_enum(self):
        """Test RetrieverType enum values."""
        assert RetrieverType.SIMILARITY == "similarity"
        assert RetrieverType.MMR == "mmr"
        assert RetrieverType.ENSEMBLE == "ensemble"
        
        values = [e.value for e in RetrieverType]
        assert set(values) == {"similarity", "mmr", "ensemble"}
    
    def test_loader_type_enum(self):
        """Test LoaderType enum values."""
        expected_values = {
            "web", "pdf", "markdown", "text", "directory", 
            "html", "json", "csv", "confluence", "notion", "github"
        }
        values = [e.value for e in LoaderType]
        assert set(values) == expected_values
    
    def test_splitter_type_enum(self):
        """Test SplitterType enum values."""
        expected_values = {
            "recursive_character", "character", "markdown", 
            "token", "html", "json", "code"
        }
        values = [e.value for e in SplitterType]
        assert set(values) == expected_values
    
    def test_resource_type_enum(self):
        """Test ResourceType enum values."""
        expected_values = {"dsl", "contextual", "guidelines", "domain_rules"}
        values = [e.value for e in ResourceType]
        assert set(values) == expected_values
    
    def test_citation_verbosity_enum(self):
        """Test CitationVerbosity enum values."""
        expected_values = {"none", "minimal", "standard", "detailed", "forensic"}
        values = [e.value for e in CitationVerbosity]
        assert set(values) == expected_values
    
    def test_citation_format_enum(self):
        """Test CitationFormat enum values."""
        expected_values = {"inline", "footnote", "academic", "structured"}
        values = [e.value for e in CitationFormat]
        assert set(values) == expected_values


class TestEmbeddingConfig:
    """Test EmbeddingConfig model."""
    
    def test_embedding_config_defaults(self):
        """Test EmbeddingConfig with default values."""
        config = EmbeddingConfig(type=EmbeddingType.OPENAI)
        
        assert config.type == EmbeddingType.OPENAI
        assert config.model_name == "text-embedding-3-small"
    
    def test_embedding_config_custom(self):
        """Test EmbeddingConfig with custom values."""
        config = EmbeddingConfig(
            type=EmbeddingType.SENTENCE_TRANSFORMERS,
            model_name="all-MiniLM-L6-v2",
            params={"device": "cpu", "batch_size": 32}
        )
        
        assert config.type == EmbeddingType.SENTENCE_TRANSFORMERS
        assert config.model_name == "all-MiniLM-L6-v2"
        assert config.params == {"device": "cpu", "batch_size": 32}
    
    def test_embedding_config_validation(self):
        """Test EmbeddingConfig validation."""
        # Valid config
        config = EmbeddingConfig(type="openai")
        assert config.type == EmbeddingType.OPENAI
        
        # Invalid type should raise ValidationError
        with pytest.raises(ValidationError):
            EmbeddingConfig(type="invalid_type")


class TestLLMConfig:
    """Test LLMConfig model."""
    
    def test_llm_config_defaults(self):
        """Test LLMConfig with default values."""
        config = LLMConfig()
        
        assert config.provider == "openai"
        assert config.model_name == "gpt-3.5-turbo"
        assert config.temperature == 0.1
        assert config.max_tokens is None
        assert config.api_key_env_var == "OPENAI_API_KEY"
    
    def test_llm_config_custom(self):
        """Test LLMConfig with custom values."""
        config = LLMConfig(
            provider="anthropic",
            model_name="claude-3-sonnet",
            temperature=0.5,
            max_tokens=2000,
            api_key_env_var="ANTHROPIC_API_KEY",
            params={"top_p": 0.9}
        )
        
        assert config.provider == "anthropic"
        assert config.model_name == "claude-3-sonnet"
        assert config.temperature == 0.5
        assert config.max_tokens == 2000
        assert config.api_key_env_var == "ANTHROPIC_API_KEY"
        assert config.params == {"top_p": 0.9}


class TestVectorStoreConfig:
    """Test VectorStoreConfig model."""
    
    def test_vector_store_config_defaults(self):
        """Test VectorStoreConfig with default values."""
        config = VectorStoreConfig(type=VectorStoreType.CHROMA)
        
        assert config.type == VectorStoreType.CHROMA
        assert config.persist_directory is None
        assert config.collection_name == "paas_ai_knowledge"
    
    def test_vector_store_config_custom(self):
        """Test VectorStoreConfig with custom values."""
        config = VectorStoreConfig(
            type=VectorStoreType.PINECONE,
            persist_directory="/tmp/vectors",
            collection_name="my_collection",
            params={"environment": "us-west1-gcp"}
        )
        
        assert config.type == VectorStoreType.PINECONE
        assert config.persist_directory == "/tmp/vectors"
        assert config.collection_name == "my_collection"
        assert config.params == {"environment": "us-west1-gcp"}


class TestRetrieverConfig:
    """Test RetrieverConfig model."""
    
    def test_retriever_config_defaults(self):
        """Test RetrieverConfig with default values."""
        config = RetrieverConfig(type=RetrieverType.SIMILARITY)
        
        assert config.type == RetrieverType.SIMILARITY
        assert config.search_kwargs == {"k": 4}
    
    def test_retriever_config_custom(self):
        """Test RetrieverConfig with custom values."""
        config = RetrieverConfig(
            type=RetrieverType.MMR,
            search_kwargs={"k": 10, "fetch_k": 20, "lambda_mult": 0.5},
            params={"similarity_threshold": 0.7}
        )
        
        assert config.type == RetrieverType.MMR
        assert config.search_kwargs == {"k": 10, "fetch_k": 20, "lambda_mult": 0.5}
        assert config.params == {"similarity_threshold": 0.7}


class TestLoaderConfig:
    """Test LoaderConfig model."""
    
    def test_loader_config_defaults(self):
        """Test LoaderConfig with default values."""
        config = LoaderConfig(type=LoaderType.WEB)
        
        assert config.type == LoaderType.WEB
    
    def test_loader_config_custom(self):
        """Test LoaderConfig with custom values."""
        config = LoaderConfig(
            type=LoaderType.PDF,
            params={"password": "secret", "extract_images": True}
        )
        
        assert config.type == LoaderType.PDF
        assert config.params == {"password": "secret", "extract_images": True}


class TestSplitterConfig:
    """Test SplitterConfig model."""
    
    def test_splitter_config_defaults(self):
        """Test SplitterConfig with default values."""
        config = SplitterConfig(type=SplitterType.RECURSIVE_CHARACTER)
        
        assert config.type == SplitterType.RECURSIVE_CHARACTER
        assert config.chunk_size == 1000
        assert config.chunk_overlap == 200
    
    def test_splitter_config_custom(self):
        """Test SplitterConfig with custom values."""
        config = SplitterConfig(
            type=SplitterType.TOKEN,
            chunk_size=512,
            chunk_overlap=50,
            params={"model_name": "gpt-3.5-turbo"}
        )
        
        assert config.type == SplitterType.TOKEN
        assert config.chunk_size == 512
        assert config.chunk_overlap == 50
        assert config.params == {"model_name": "gpt-3.5-turbo"}


class TestContentValidatorConfig:
    """Test ContentValidatorConfig model."""
    
    def test_content_validator_config_defaults(self):
        """Test ContentValidatorConfig with default values."""
        config = ContentValidatorConfig(type=ContentValidatorType.CONTENT)
        
        assert config.type == ContentValidatorType.CONTENT
        assert config.min_content_length == 10
        assert config.max_content_length == 1000000
        assert config.skip_empty is True
    
    def test_content_validator_config_custom(self):
        """Test ContentValidatorConfig with custom values."""
        config = ContentValidatorConfig(
            type=ContentValidatorType.CONTENT,
            min_content_length=50,
            max_content_length=500000,
            skip_empty=False,
            params={"language": "en"}
        )
        
        assert config.type == ContentValidatorType.CONTENT
        assert config.min_content_length == 50
        assert config.max_content_length == 500000
        assert config.skip_empty is False
        assert config.params == {"language": "en"}


class TestCitationConfig:
    """Test CitationConfig model."""
    
    def test_citation_config_defaults(self):
        """Test CitationConfig with default values."""
        config = CitationConfig()
        
        assert config.enabled is True
        assert config.verbosity == CitationVerbosity.STANDARD
        assert config.format == CitationFormat.INLINE
        assert config.resource_overrides == {}
        assert config.include_quotes is True
        assert config.max_quote_length == 150
        assert config.include_confidence is False
        assert config.generate_deep_links is True
        assert config.base_urls == {}
        
        # Check default strategies
        expected_strategies = {
            ResourceType.DSL: "technical_citation",
            ResourceType.CONTEXTUAL: "web_citation",
            ResourceType.GUIDELINES: "policy_citation",
            ResourceType.DOMAIN_RULES: "rule_citation"
        }
        assert config.strategies == expected_strategies
    
    def test_citation_config_custom(self):
        """Test CitationConfig with custom values."""
        config = CitationConfig(
            enabled=False,
            verbosity=CitationVerbosity.FORENSIC,
            format=CitationFormat.STRUCTURED,
            resource_overrides={ResourceType.DSL: CitationVerbosity.DETAILED},
            include_quotes=False,
            max_quote_length=300,
            include_confidence=True,
            generate_deep_links=False,
            strategies={ResourceType.DSL: "custom_strategy"},
            base_urls={"docs": "https://docs.example.com"}
        )
        
        assert config.enabled is False
        assert config.verbosity == CitationVerbosity.FORENSIC
        assert config.format == CitationFormat.STRUCTURED
        assert config.resource_overrides == {ResourceType.DSL: CitationVerbosity.DETAILED}
        assert config.include_quotes is False
        assert config.max_quote_length == 300
        assert config.include_confidence is True
        assert config.generate_deep_links is False
        assert config.strategies == {ResourceType.DSL: "custom_strategy"}
        assert config.base_urls == {"docs": "https://docs.example.com"}


class TestResourceConfig:
    """Test ResourceConfig model."""
    
    def test_resource_config_defaults(self):
        """Test ResourceConfig with default values."""
        config = ResourceConfig(
            url="https://example.com",
            resource_type=ResourceType.DSL,
            loader=LoaderConfig(type=LoaderType.WEB),
            splitter=SplitterConfig(type=SplitterType.RECURSIVE_CHARACTER)
        )
        
        assert config.url == "https://example.com"
        assert config.resource_type == ResourceType.DSL
        assert config.loader.type == LoaderType.WEB
        assert config.splitter.type == SplitterType.RECURSIVE_CHARACTER
        assert config.tags == []
        assert config.priority == 5
        assert config.last_updated is None
    
    def test_resource_config_custom(self):
        """Test ResourceConfig with custom values."""
        config = ResourceConfig(
            url="https://docs.example.com",
            resource_type=ResourceType.GUIDELINES,
            loader=LoaderConfig(type=LoaderType.PDF, params={"password": "secret"}),
            splitter=SplitterConfig(type=SplitterType.TOKEN, chunk_size=512),
            tags=["important", "production"],
            priority=10,
            last_updated="2024-01-01T00:00:00Z"
        )
        
        assert config.url == "https://docs.example.com"
        assert config.resource_type == ResourceType.GUIDELINES
        assert config.loader.params == {"password": "secret"}
        assert config.splitter.chunk_size == 512
        assert config.tags == ["important", "production"]
        assert config.priority == 10
        assert config.last_updated == "2024-01-01T00:00:00Z"


class TestMultiAgentConfig:
    """Test MultiAgentConfig model."""
    
    def test_multi_agent_config_defaults(self):
        """Test MultiAgentConfig with default values."""
        config = MultiAgentConfig()
        
        assert config.enabled is True
        assert config.mode == "supervisor"
        assert config.default_agent == "designer"
        assert config.track_tokens is False
        assert config.token_callback is None
        assert config.verbose is False
    
    def test_multi_agent_config_custom(self):
        """Test MultiAgentConfig with custom values."""
        config = MultiAgentConfig(
            enabled=False,
            mode="swarm",
            default_agent="paas_manifest_generator",
            track_tokens=True,
            token_callback="json_file",
            verbose=True
        )
        
        assert config.enabled is False
        assert config.mode == "swarm"
        assert config.default_agent == "paas_manifest_generator"
        assert config.track_tokens is True
        assert config.token_callback == "json_file"
        assert config.verbose is True
    
    def test_multi_agent_config_validation(self):
        """Test MultiAgentConfig validation."""
        # Valid modes
        config1 = MultiAgentConfig(mode="supervisor")
        assert config1.mode == "supervisor"
        
        config2 = MultiAgentConfig(mode="swarm")
        assert config2.mode == "swarm"
        
        # Invalid mode should raise ValidationError
        with pytest.raises(ValidationError):
            MultiAgentConfig(mode="invalid_mode")
        
        # Valid default agents
        config3 = MultiAgentConfig(default_agent="designer")
        assert config3.default_agent == "designer"
        
        config4 = MultiAgentConfig(default_agent="paas_manifest_generator")
        assert config4.default_agent == "paas_manifest_generator"
        
        # Invalid default agent should raise ValidationError
        with pytest.raises(ValidationError):
            MultiAgentConfig(default_agent="invalid_agent")


class TestConfig:
    """Test main Config model."""
    
    def test_config_defaults(self):
        """Test Config with default values."""
        config = Config(
            embedding=EmbeddingConfig(type=EmbeddingType.OPENAI),
            vectorstore=VectorStoreConfig(type=VectorStoreType.CHROMA),
            retriever=RetrieverConfig(type=RetrieverType.SIMILARITY)
        )
        
        # Required fields
        assert config.embedding.type == EmbeddingType.OPENAI
        assert config.vectorstore.type == VectorStoreType.CHROMA
        assert config.retriever.type == RetrieverType.SIMILARITY
        
        # Default fields
        assert config.llm.provider == "openai"
        assert config.llm.model_name == "gpt-3.5-turbo"
        assert config.loader is None
        assert config.content_validator is None
        assert config.citation.enabled is True
        assert config.batch_size == 32
        assert config.validate_urls is True
        assert config.max_parallel == 5
        assert config.timeout == 30
        assert config.log_level == "INFO"
        
        # Multi-agent defaults
        assert config.multi_agent.enabled is True
        assert config.multi_agent.mode == "supervisor"
        assert config.multi_agent.default_agent == "designer"
        
        # Agent configurations
        assert "designer" in config.agents
        assert "paas_manifest_generator" in config.agents
        assert config.agents["designer"]["model"] == "gpt-4o-mini"
        assert config.agents["paas_manifest_generator"]["temperature"] == 0.0
    
    def test_config_custom(self):
        """Test Config with custom values."""
        config = Config(
            embedding=EmbeddingConfig(type=EmbeddingType.SENTENCE_TRANSFORMERS),
            vectorstore=VectorStoreConfig(type=VectorStoreType.PINECONE),
            retriever=RetrieverConfig(type=RetrieverType.ENSEMBLE),
            llm=LLMConfig(provider="anthropic", model_name="claude-3-sonnet"),
            loader=LoaderConfig(type=LoaderType.PDF),
            content_validator=ContentValidatorConfig(type=ContentValidatorType.CONTENT),
            citation=CitationConfig(enabled=False),
            batch_size=64,
            validate_urls=False,
            max_parallel=10,
            timeout=60,
            log_level="DEBUG",
            multi_agent=MultiAgentConfig(enabled=False),
            agents={"custom_agent": {"model": "gpt-4", "temperature": 0.5}}
        )
        
        assert config.embedding.type == EmbeddingType.SENTENCE_TRANSFORMERS
        assert config.vectorstore.type == VectorStoreType.PINECONE
        assert config.retriever.type == RetrieverType.ENSEMBLE
        assert config.llm.provider == "anthropic"
        assert config.llm.model_name == "claude-3-sonnet"
        assert config.loader.type == LoaderType.PDF
        assert config.content_validator.type == ContentValidatorType.CONTENT
        assert config.citation.enabled is False
        assert config.batch_size == 64
        assert config.validate_urls is False
        assert config.max_parallel == 10
        assert config.timeout == 60
        assert config.log_level == "DEBUG"
        assert config.multi_agent.enabled is False
        assert config.agents["custom_agent"]["model"] == "gpt-4"


class TestAPIConfig:
    """Test APIConfig model."""
    
    def test_api_config_defaults(self):
        """Test APIConfig with default values."""
        config = APIConfig()
        
        assert config.host == "0.0.0.0"
        assert config.port == 8000
        assert config.cors_origins == ["*"]
        assert config.rate_limiting == {}
    
    def test_api_config_custom(self):
        """Test APIConfig with custom values."""
        config = APIConfig(
            host="127.0.0.1",
            port=9000,
            cors_origins=["https://example.com", "https://app.example.com"],
            rate_limiting={"requests_per_minute": 100}
        )
        
        assert config.host == "127.0.0.1"
        assert config.port == 9000
        assert config.cors_origins == ["https://example.com", "https://app.example.com"]
        assert config.rate_limiting == {"requests_per_minute": 100}


class TestAgentConfig:
    """Test AgentConfig model."""
    
    def test_agent_config_defaults(self):
        """Test AgentConfig with default values."""
        config = AgentConfig()
        
        assert config.max_iterations == 10
        assert config.memory_type == "buffer"
        assert config.tools == []
        assert config.model_provider == "openai"
    
    def test_agent_config_custom(self):
        """Test AgentConfig with custom values."""
        config = AgentConfig(
            max_iterations=20,
            memory_type="conversation",
            tools=["web_search", "file_operations"],
            model_provider="anthropic"
        )
        
        assert config.max_iterations == 20
        assert config.memory_type == "conversation"
        assert config.tools == ["web_search", "file_operations"]
        assert config.model_provider == "anthropic"


class TestConfigFile:
    """Test ConfigFile model."""
    
    def test_config_file_defaults(self):
        """Test ConfigFile with default values."""
        config_file = ConfigFile()
        
        assert config_file.current == "default"
        assert config_file.profiles == {}
    
    def test_config_file_custom(self):
        """Test ConfigFile with custom values."""
        config = Config(
            embedding=EmbeddingConfig(type=EmbeddingType.OPENAI),
            vectorstore=VectorStoreConfig(type=VectorStoreType.CHROMA),
            retriever=RetrieverConfig(type=RetrieverType.SIMILARITY)
        )
        
        config_file = ConfigFile(
            current="production",
            profiles={"production": config, "staging": config}
        )
        
        assert config_file.current == "production"
        assert len(config_file.profiles) == 2
        assert "production" in config_file.profiles
        assert "staging" in config_file.profiles
    
    def test_config_file_extra_fields_forbidden(self):
        """Test that ConfigFile doesn't allow extra fields."""
        with pytest.raises(ValidationError):
            ConfigFile(extra_field="not_allowed")


class TestDefaultConfigProfiles:
    """Test default configuration profiles."""
    
    def test_default_profiles_exist(self):
        """Test that all expected default profiles exist."""
        expected_profiles = {"default", "local", "production", "verbose"}
        assert set(DEFAULT_CONFIG_PROFILES.keys()) == expected_profiles
    
    def test_default_profile(self):
        """Test default profile configuration."""
        config = DEFAULT_CONFIG_PROFILES["default"]
        
        assert config.embedding.type == EmbeddingType.OPENAI
        assert config.embedding.model_name == "text-embedding-3-small"
        assert config.vectorstore.type == VectorStoreType.CHROMA
        assert config.vectorstore.persist_directory == "rag_data/chroma"
        assert config.vectorstore.collection_name == "paas_ai_rag"
        assert config.retriever.type == RetrieverType.SIMILARITY
        assert config.retriever.search_kwargs == {"k": 4}
        assert config.llm.provider == "openai"
        assert config.llm.model_name == "gpt-3.5-turbo"
        assert config.llm.temperature == 0.1
        assert config.citation.enabled is True
        assert config.citation.verbosity == CitationVerbosity.STANDARD
        assert config.citation.format == CitationFormat.INLINE
        assert config.batch_size == 32
        assert config.validate_urls is True
    
    def test_local_profile(self):
        """Test local profile configuration."""
        config = DEFAULT_CONFIG_PROFILES["local"]
        
        assert config.embedding.type == EmbeddingType.SENTENCE_TRANSFORMERS
        assert config.embedding.model_name == "all-MiniLM-L6-v2"
        assert config.vectorstore.type == VectorStoreType.CHROMA
        assert config.vectorstore.persist_directory == "rag_data/chroma_local"
        assert config.vectorstore.collection_name == "paas_ai_local"
        assert config.retriever.type == RetrieverType.SIMILARITY
        assert config.retriever.search_kwargs == {"k": 5}
        assert config.llm.provider == "openai"
        assert config.llm.model_name == "gpt-3.5-turbo"
        assert config.llm.temperature == 0.0
        assert config.llm.max_tokens == 1000
        assert config.citation.enabled is True
        assert config.citation.verbosity == CitationVerbosity.MINIMAL
        assert config.citation.format == CitationFormat.INLINE
        assert config.batch_size == 16
        assert config.validate_urls is True
    
    def test_production_profile(self):
        """Test production profile configuration."""
        config = DEFAULT_CONFIG_PROFILES["production"]
        
        assert config.embedding.type == EmbeddingType.OPENAI
        assert config.embedding.model_name == "text-embedding-3-large"
        assert config.vectorstore.type == VectorStoreType.PINECONE
        assert config.vectorstore.collection_name == "paas-ai-prod"
        assert config.retriever.type == RetrieverType.ENSEMBLE
        assert config.retriever.search_kwargs == {"k": 10}
        assert config.llm.provider == "openai"
        assert config.llm.model_name == "gpt-4"
        assert config.llm.temperature == 0.2
        assert config.llm.max_tokens == 2000
        assert config.citation.enabled is True
        assert config.citation.verbosity == CitationVerbosity.DETAILED
        assert config.citation.format == CitationFormat.INLINE
        assert config.citation.include_confidence is True
        assert config.citation.resource_overrides[ResourceType.DSL] == CitationVerbosity.FORENSIC
        assert config.citation.resource_overrides[ResourceType.GUIDELINES] == CitationVerbosity.DETAILED
        assert config.batch_size == 64
        assert config.validate_urls is True
    
    def test_verbose_profile(self):
        """Test verbose profile configuration."""
        config = DEFAULT_CONFIG_PROFILES["verbose"]
        
        assert config.embedding.type == EmbeddingType.SENTENCE_TRANSFORMERS
        assert config.embedding.model_name == "all-MiniLM-L6-v2"
        assert config.vectorstore.type == VectorStoreType.CHROMA
        assert config.vectorstore.persist_directory == "rag_data/chroma_local"
        assert config.vectorstore.collection_name == "paas_ai_local"
        assert config.retriever.type == RetrieverType.SIMILARITY
        assert config.retriever.search_kwargs == {"k": 5}
        assert config.llm.provider == "openai"
        assert config.llm.model_name == "gpt-3.5-turbo"
        assert config.llm.temperature == 0.0
        assert config.llm.max_tokens == 1000
        assert config.citation.enabled is True
        assert config.citation.verbosity == CitationVerbosity.FORENSIC
        assert config.citation.format == CitationFormat.STRUCTURED
        assert config.citation.include_quotes is True
        assert config.citation.include_confidence is True
        assert config.citation.max_quote_length == 200
        assert config.batch_size == 16
        assert config.validate_urls is True
        
        # Multi-agent configuration
        assert config.multi_agent.enabled is True
        assert config.multi_agent.mode == "supervisor"
        assert config.multi_agent.default_agent == "designer"
        assert config.multi_agent.track_tokens is True
        assert config.multi_agent.token_callback == "console"
        assert config.multi_agent.verbose is True
    
    def test_all_profiles_valid(self):
        """Test that all default profiles are valid Config objects."""
        for profile_name, config in DEFAULT_CONFIG_PROFILES.items():
            # Should not raise ValidationError
            assert isinstance(config, Config)
            assert config.embedding is not None
            assert config.vectorstore is not None
            assert config.retriever is not None
            assert config.llm is not None
            assert config.citation is not None
            assert config.multi_agent is not None
            assert isinstance(config.agents, dict)


class TestModelSerialization:
    """Test model serialization and deserialization."""
    
    def test_config_serialization(self):
        """Test Config model serialization."""
        config = Config(
            embedding=EmbeddingConfig(type=EmbeddingType.OPENAI),
            vectorstore=VectorStoreConfig(type=VectorStoreType.CHROMA),
            retriever=RetrieverConfig(type=RetrieverType.SIMILARITY)
        )
        
        # Test model_dump
        data = config.model_dump()
        assert isinstance(data, dict)
        assert data["embedding"]["type"] == "openai"
        assert data["vectorstore"]["type"] == "chroma"
        assert data["retriever"]["type"] == "similarity"
        
        # Test model_dump_json
        json_data = config.model_dump_json()
        assert isinstance(json_data, str)
        assert "openai" in json_data
        assert "chroma" in json_data
        assert "similarity" in json_data
    
    def test_config_deserialization(self):
        """Test Config model deserialization."""
        data = {
            "embedding": {"type": "openai", "model_name": "text-embedding-3-small"},
            "vectorstore": {"type": "chroma", "collection_name": "test"},
            "retriever": {"type": "similarity", "search_kwargs": {"k": 5}}
        }
        
        config = Config.model_validate(data)
        assert config.embedding.type == EmbeddingType.OPENAI
        assert config.vectorstore.type == VectorStoreType.CHROMA
        assert config.retriever.type == RetrieverType.SIMILARITY
        assert config.retriever.search_kwargs == {"k": 5}
    
    def test_config_file_serialization(self):
        """Test ConfigFile model serialization."""
        config = Config(
            embedding=EmbeddingConfig(type=EmbeddingType.OPENAI),
            vectorstore=VectorStoreConfig(type=VectorStoreType.CHROMA),
            retriever=RetrieverConfig(type=RetrieverType.SIMILARITY)
        )
        
        config_file = ConfigFile(
            current="test",
            profiles={"test": config}
        )
        
        data = config_file.model_dump()
        assert data["current"] == "test"
        assert "test" in data["profiles"]
        assert data["profiles"]["test"]["embedding"]["type"] == "openai"


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_empty_params_dict(self):
        """Test that empty params dicts are handled correctly."""
        config = EmbeddingConfig(type=EmbeddingType.OPENAI, params={})
        
        # Test with None - Pydantic requires dict type
        with pytest.raises(ValidationError):
            config = EmbeddingConfig(type=EmbeddingType.OPENAI, params=None)
    
    def test_nested_config_validation(self):
        """Test validation of nested configuration objects."""
        # Valid nested config
        config = Config(
            embedding=EmbeddingConfig(type=EmbeddingType.OPENAI),
            vectorstore=VectorStoreConfig(type=VectorStoreType.CHROMA),
            retriever=RetrieverConfig(type=RetrieverType.SIMILARITY),
            citation=CitationConfig(
                resource_overrides={ResourceType.DSL: CitationVerbosity.DETAILED}
            )
        )
        assert config.citation.resource_overrides[ResourceType.DSL] == CitationVerbosity.DETAILED
        
        # Invalid nested config should raise ValidationError
        with pytest.raises(ValidationError):
            Config(
                embedding=EmbeddingConfig(type="invalid_type"),
                vectorstore=VectorStoreConfig(type=VectorStoreType.CHROMA),
                retriever=RetrieverConfig(type=RetrieverType.SIMILARITY)
            )
    
    def test_optional_fields(self):
        """Test optional fields behavior."""
        config = Config(
            embedding=EmbeddingConfig(type=EmbeddingType.OPENAI),
            vectorstore=VectorStoreConfig(type=VectorStoreType.CHROMA),
            retriever=RetrieverConfig(type=RetrieverType.SIMILARITY)
        )
        
        # Optional fields should be None or have defaults
        assert config.loader is None
        assert config.content_validator is None
        assert config.llm.max_tokens is None  # Optional field with None default
    
    def test_enum_string_conversion(self):
        """Test that enum values are properly converted from strings."""
        # Test direct enum assignment
        config = EmbeddingConfig(type=EmbeddingType.OPENAI)
        assert config.type == EmbeddingType.OPENAI
        
        # Test string assignment (should be converted to enum)
        config = EmbeddingConfig(type="openai")
        assert config.type == EmbeddingType.OPENAI
        
        # Test invalid string (should raise ValidationError)
        with pytest.raises(ValidationError):
            EmbeddingConfig(type="invalid_enum_value")
