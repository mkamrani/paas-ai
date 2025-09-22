"""
Unit tests for citation strategies.

Tests all components of the citation strategies including:
- CitationStrategy base class
- DefaultCitationStrategy
- TechnicalCitationStrategy
- WebCitationStrategy
- PolicyCitationStrategy
- CitationStrategyRegistry
- Strategy pattern implementation
"""

import pytest
from abc import ABC
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, Optional

from src.paas_ai.core.rag.citations.strategies import (
    CitationStrategy,
    DefaultCitationStrategy,
    TechnicalCitationStrategy,
    WebCitationStrategy,
    PolicyCitationStrategy,
    CitationStrategyRegistry
)
from src.paas_ai.core.rag.citations.models import SourceReference, CitationMetadata
from src.paas_ai.core.rag.processing.base import ProcessingContext
from src.paas_ai.core.config.schemas import ResourceType, CitationVerbosity
from langchain_core.documents import Document


class TestCitationStrategy:
    """Test the CitationStrategy abstract base class."""
    
    def test_is_abstract_base_class(self):
        """Test that CitationStrategy is an abstract base class."""
        assert issubclass(CitationStrategy, ABC)
    
    def test_cannot_instantiate_directly(self):
        """Test that CitationStrategy cannot be instantiated directly."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            CitationStrategy()
    
    def test_has_required_abstract_methods(self):
        """Test that CitationStrategy has the required abstract methods."""
        abstract_methods = CitationStrategy.__abstractmethods__
        
        assert 'extract_citation_metadata' in abstract_methods
        assert 'format_citation' in abstract_methods
        assert 'generate_citation_link' in abstract_methods
        assert len(abstract_methods) == 3
    
    def test_concrete_implementation_works(self):
        """Test that a concrete implementation of CitationStrategy works."""
        class ConcreteCitationStrategy(CitationStrategy):
            def extract_citation_metadata(self, document: Document, context: ProcessingContext) -> CitationMetadata:
                return CitationMetadata()
            
            def format_citation(self, source_ref: SourceReference, verbosity: CitationVerbosity) -> str:
                return "[Test Citation]"
            
            def generate_citation_link(self, source_ref: SourceReference) -> Optional[str]:
                return "https://example.com"
        
        # Should be able to instantiate
        strategy = ConcreteCitationStrategy()
        assert isinstance(strategy, CitationStrategy)
        
        # Should be able to call methods
        document = Document(page_content="test", metadata={})
        context = Mock(spec=ProcessingContext)
        source_ref = SourceReference(
            source_url="https://example.com",
            resource_type=ResourceType.DSL,
            document_id="doc123",
            chunk_id="chunk456"
        )
        
        metadata = strategy.extract_citation_metadata(document, context)
        assert isinstance(metadata, CitationMetadata)
        
        citation = strategy.format_citation(source_ref, CitationVerbosity.STANDARD)
        assert citation == "[Test Citation]"
        
        link = strategy.generate_citation_link(source_ref)
        assert link == "https://example.com"
    
    def test_incomplete_implementation_fails(self):
        """Test that incomplete implementation fails to instantiate."""
        class IncompleteCitationStrategy(CitationStrategy):
            def extract_citation_metadata(self, document: Document, context: ProcessingContext) -> CitationMetadata:
                return CitationMetadata()
            # Missing format_citation and generate_citation_link methods
        
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            IncompleteCitationStrategy()


class TestDefaultCitationStrategy:
    """Test the DefaultCitationStrategy class."""
    
    def test_init(self):
        """Test strategy initialization."""
        strategy = DefaultCitationStrategy()
        assert strategy is not None
        assert isinstance(strategy, DefaultCitationStrategy)
        assert isinstance(strategy, CitationStrategy)
    
    def test_extract_citation_metadata_basic(self):
        """Test extracting basic citation metadata."""
        strategy = DefaultCitationStrategy()
        document = Document(
            page_content="This is test content with some structure.",
            metadata={"section": "Introduction", "title": "Test Document", "page": 1}
        )
        context = Mock(spec=ProcessingContext)
        
        metadata = strategy.extract_citation_metadata(document, context)
        
        assert isinstance(metadata, CitationMetadata)
        assert metadata.content_type_markers["has_structure"] is True
        assert metadata.content_type_markers["has_page_number"] is True
        assert metadata.content_type_markers["content_length"] == len(document.page_content)
        assert metadata.content_type_markers["word_count"] == len(document.page_content.split())
    
    def test_extract_citation_metadata_no_structure(self):
        """Test extracting citation metadata from document without structure."""
        strategy = DefaultCitationStrategy()
        document = Document(
            page_content="This is test content without structure.",
            metadata={}
        )
        context = Mock(spec=ProcessingContext)
        
        metadata = strategy.extract_citation_metadata(document, context)
        
        assert isinstance(metadata, CitationMetadata)
        assert metadata.content_type_markers["has_structure"] is False
        assert metadata.content_type_markers["has_page_number"] is False
    
    def test_format_citation_none_verbosity(self):
        """Test formatting citation with NONE verbosity."""
        strategy = DefaultCitationStrategy()
        source_ref = SourceReference(
            source_url="https://example.com/doc",
            resource_type=ResourceType.DSL,
            document_id="doc123",
            chunk_id="chunk456"
        )
        
        citation = strategy.format_citation(source_ref, CitationVerbosity.NONE)
        assert citation == ""
    
    def test_format_citation_minimal_verbosity(self):
        """Test formatting citation with MINIMAL verbosity."""
        strategy = DefaultCitationStrategy()
        source_ref = SourceReference(
            source_url="https://example.com/doc",
            resource_type=ResourceType.DSL,
            document_id="doc123",
            chunk_id="chunk456"
        )
        
        citation = strategy.format_citation(source_ref, CitationVerbosity.MINIMAL)
        assert citation == "[example.com]"
    
    def test_format_citation_standard_verbosity_with_page(self):
        """Test formatting citation with STANDARD verbosity and page number."""
        strategy = DefaultCitationStrategy()
        source_ref = SourceReference(
            source_url="https://example.com/doc",
            resource_type=ResourceType.DSL,
            document_id="doc123",
            chunk_id="chunk456",
            page_number=5
        )
        
        citation = strategy.format_citation(source_ref, CitationVerbosity.STANDARD)
        assert citation == "[example.com, p. 5]"
    
    def test_format_citation_standard_verbosity_with_section(self):
        """Test formatting citation with STANDARD verbosity and section title."""
        strategy = DefaultCitationStrategy()
        source_ref = SourceReference(
            source_url="https://example.com/doc",
            resource_type=ResourceType.DSL,
            document_id="doc123",
            chunk_id="chunk456",
            section_title="Introduction"
        )
        
        citation = strategy.format_citation(source_ref, CitationVerbosity.STANDARD)
        assert citation == "[example.com, Introduction]"
    
    def test_format_citation_detailed_verbosity(self):
        """Test formatting citation with DETAILED verbosity."""
        strategy = DefaultCitationStrategy()
        source_ref = SourceReference(
            source_url="https://example.com/doc",
            resource_type=ResourceType.DSL,
            document_id="doc123",
            chunk_id="chunk456",
            page_number=5,
            section_title="Introduction"
        )
        
        citation = strategy.format_citation(source_ref, CitationVerbosity.DETAILED)
        assert citation == "[example.com, Page 5, Section 'Introduction']"
    
    def test_format_citation_forensic_verbosity(self):
        """Test formatting citation with FORENSIC verbosity."""
        strategy = DefaultCitationStrategy()
        source_ref = SourceReference(
            source_url="https://example.com/doc",
            resource_type=ResourceType.DSL,
            document_id="doc123",
            chunk_id="chunk456",
            page_number=5,
            section_title="Introduction",
            exact_quote="This is a test quote"
        )
        
        citation = strategy.format_citation(source_ref, CitationVerbosity.FORENSIC)
        assert citation == "[example.com, Page 5, Section 'Introduction']: \"This is a test quote\""
    
    def test_generate_citation_link(self):
        """Test generating citation link."""
        strategy = DefaultCitationStrategy()
        source_ref = SourceReference(
            source_url="https://example.com/doc",
            resource_type=ResourceType.DSL,
            document_id="doc123",
            chunk_id="chunk456"
        )
        
        link = strategy.generate_citation_link(source_ref)
        assert link == "https://example.com/doc"
    
    def test_get_source_name_http_url(self):
        """Test extracting source name from HTTP URL."""
        strategy = DefaultCitationStrategy()
        
        # Test with www
        url = "https://www.example.com/path/to/doc"
        name = strategy._get_source_name(url)
        assert name == "example.com"
        
        # Test without www
        url = "https://example.com/path/to/doc"
        name = strategy._get_source_name(url)
        assert name == "example.com"
    
    def test_get_source_name_local_file(self):
        """Test extracting source name from local file path."""
        strategy = DefaultCitationStrategy()
        
        url = "/path/to/document.pdf"
        name = strategy._get_source_name(url)
        assert name == "document.pdf"


class TestTechnicalCitationStrategy:
    """Test the TechnicalCitationStrategy class."""
    
    def test_init(self):
        """Test strategy initialization."""
        strategy = TechnicalCitationStrategy()
        assert strategy is not None
        assert isinstance(strategy, TechnicalCitationStrategy)
        assert isinstance(strategy, CitationStrategy)
    
    def test_extract_citation_metadata_technical_content(self):
        """Test extracting metadata from technical content."""
        strategy = TechnicalCitationStrategy()
        document = Document(
            page_content="def function_name(): return 'test'",
            metadata={"section": "API Reference"}
        )
        context = Mock(spec=ProcessingContext)
        
        metadata = strategy.extract_citation_metadata(document, context)
        
        assert isinstance(metadata, CitationMetadata)
        assert metadata.content_type_markers["has_code"] is True
        assert metadata.content_type_markers["has_api_refs"] is False
        assert metadata.content_type_markers["has_examples"] is False
        assert metadata.content_type_markers["has_version_info"] is False
        assert metadata.content_type_markers["has_structure"] is True
        assert len(metadata.headings) == 1
        assert metadata.headings[0]["text"] == "API Reference"
    
    def test_extract_citation_metadata_api_content(self):
        """Test extracting metadata from API documentation content."""
        strategy = TechnicalCitationStrategy()
        document = Document(
            page_content="API endpoint /users with method GET and parameter id",
            metadata={"section": "API Documentation"}
        )
        context = Mock(spec=ProcessingContext)
        
        metadata = strategy.extract_citation_metadata(document, context)
        
        assert metadata.content_type_markers["has_api_refs"] is True
        assert metadata.content_type_markers["has_code"] is False
    
    def test_extract_citation_metadata_example_content(self):
        """Test extracting metadata from content with examples."""
        strategy = TechnicalCitationStrategy()
        document = Document(
            page_content="Here is an example of how to use the API",
            metadata={"section": "Examples"}
        )
        context = Mock(spec=ProcessingContext)
        
        metadata = strategy.extract_citation_metadata(document, context)
        
        assert metadata.content_type_markers["has_examples"] is True
    
    def test_extract_citation_metadata_version_content(self):
        """Test extracting metadata from content with version information."""
        strategy = TechnicalCitationStrategy()
        document = Document(
            page_content="API version v1.2.3 release notes",
            metadata={"section": "Version History"}
        )
        context = Mock(spec=ProcessingContext)
        
        metadata = strategy.extract_citation_metadata(document, context)
        
        assert metadata.content_type_markers["has_version_info"] is True
    
    def test_format_citation_kubernetes_docs(self):
        """Test formatting citation for Kubernetes documentation."""
        strategy = TechnicalCitationStrategy()
        source_ref = SourceReference(
            source_url="https://kubernetes.io/docs/concepts/workloads/pods/",
            resource_type=ResourceType.DSL,
            document_id="doc123",
            chunk_id="chunk456",
            section_title="Pod Configuration"
        )
        
        citation = strategy.format_citation(source_ref, CitationVerbosity.STANDARD)
        assert citation == "[Kubernetes Docs, Pod Configuration]"
    
    def test_format_citation_docker_docs(self):
        """Test formatting citation for Docker documentation."""
        strategy = TechnicalCitationStrategy()
        source_ref = SourceReference(
            source_url="https://docs.docker.com/engine/reference/",
            resource_type=ResourceType.DSL,
            document_id="doc123",
            chunk_id="chunk456",
            section_title="Docker Commands"
        )
        
        citation = strategy.format_citation(source_ref, CitationVerbosity.STANDARD)
        assert citation == "[Docker Docs, Docker Commands]"
    
    def test_format_citation_with_tags(self):
        """Test formatting citation with tags in metadata."""
        strategy = TechnicalCitationStrategy()
        source_ref = SourceReference(
            source_url="https://example.com/api",
            resource_type=ResourceType.DSL,
            document_id="doc123",
            chunk_id="chunk456",
            section_title="API Reference",
            metadata={"tags": ["kubernetes", "api"]}
        )
        
        citation = strategy.format_citation(source_ref, CitationVerbosity.MINIMAL)
        assert citation == "[Kubernetes]"
    
    def test_generate_citation_link_with_section(self):
        """Test generating citation link with section title."""
        strategy = TechnicalCitationStrategy()
        source_ref = SourceReference(
            source_url="https://kubernetes.io/docs/concepts/workloads/pods/",
            resource_type=ResourceType.DSL,
            document_id="doc123",
            chunk_id="chunk456",
            section_title="Pod Configuration"
        )
        
        link = strategy.generate_citation_link(source_ref)
        assert link == "https://kubernetes.io/docs/concepts/workloads/pods/#pod-configuration"
    
    def test_generate_citation_link_without_section(self):
        """Test generating citation link without section title."""
        strategy = TechnicalCitationStrategy()
        source_ref = SourceReference(
            source_url="https://kubernetes.io/docs/concepts/workloads/pods/",
            resource_type=ResourceType.DSL,
            document_id="doc123",
            chunk_id="chunk456"
        )
        
        link = strategy.generate_citation_link(source_ref)
        assert link == "https://kubernetes.io/docs/concepts/workloads/pods/"
    
    def test_get_api_name_kubernetes(self):
        """Test getting API name for Kubernetes URL."""
        strategy = TechnicalCitationStrategy()
        source_ref = SourceReference(
            source_url="https://kubernetes.io/docs/",
            resource_type=ResourceType.DSL,
            document_id="doc123",
            chunk_id="chunk456"
        )
        
        name = strategy._get_api_name(source_ref)
        assert name == "Kubernetes Docs"
    
    def test_get_api_name_docker(self):
        """Test getting API name for Docker URL."""
        strategy = TechnicalCitationStrategy()
        source_ref = SourceReference(
            source_url="https://docs.docker.com/",
            resource_type=ResourceType.DSL,
            document_id="doc123",
            chunk_id="chunk456"
        )
        
        name = strategy._get_api_name(source_ref)
        assert name == "Docker Docs"


class TestWebCitationStrategy:
    """Test the WebCitationStrategy class."""
    
    def test_init(self):
        """Test strategy initialization."""
        strategy = WebCitationStrategy()
        assert strategy is not None
        assert isinstance(strategy, WebCitationStrategy)
        assert isinstance(strategy, DefaultCitationStrategy)
        assert isinstance(strategy, CitationStrategy)
    
    def test_generate_citation_link_with_section(self):
        """Test generating citation link with section title."""
        strategy = WebCitationStrategy()
        source_ref = SourceReference(
            source_url="https://example.com/blog/post",
            resource_type=ResourceType.CONTEXTUAL,
            document_id="doc123",
            chunk_id="chunk456",
            section_title="Introduction"
        )
        
        link = strategy.generate_citation_link(source_ref)
        assert link == "https://example.com/blog/post#introduction"
    
    def test_generate_citation_link_without_section(self):
        """Test generating citation link without section title."""
        strategy = WebCitationStrategy()
        source_ref = SourceReference(
            source_url="https://example.com/blog/post",
            resource_type=ResourceType.CONTEXTUAL,
            document_id="doc123",
            chunk_id="chunk456"
        )
        
        link = strategy.generate_citation_link(source_ref)
        assert link == "https://example.com/blog/post"
    
    def test_create_anchor_simple(self):
        """Test creating anchor from simple section title."""
        strategy = WebCitationStrategy()
        
        anchor = strategy._create_anchor("Introduction")
        assert anchor == "introduction"
    
    def test_create_anchor_with_spaces(self):
        """Test creating anchor from section title with spaces."""
        strategy = WebCitationStrategy()
        
        anchor = strategy._create_anchor("Getting Started Guide")
        assert anchor == "getting-started-guide"
    
    def test_create_anchor_with_special_chars(self):
        """Test creating anchor from section title with special characters."""
        strategy = WebCitationStrategy()
        
        anchor = strategy._create_anchor("API Reference (v2.0)")
        assert anchor == "api-reference-v20"
    
    def test_create_anchor_with_dots(self):
        """Test creating anchor from section title with dots."""
        strategy = WebCitationStrategy()
        
        anchor = strategy._create_anchor("Section 1.2.3")
        assert anchor == "section-123"


class TestPolicyCitationStrategy:
    """Test the PolicyCitationStrategy class."""
    
    def test_init(self):
        """Test strategy initialization."""
        strategy = PolicyCitationStrategy()
        assert strategy is not None
        assert isinstance(strategy, PolicyCitationStrategy)
        assert isinstance(strategy, CitationStrategy)
    
    def test_extract_citation_metadata_policy_content(self):
        """Test extracting metadata from policy content."""
        strategy = PolicyCitationStrategy()
        document = Document(
            page_content="All users must implement MFA. This is required for compliance.",
            metadata={"section": "Access Control"}
        )
        context = Mock(spec=ProcessingContext)
        
        metadata = strategy.extract_citation_metadata(document, context)
        
        assert isinstance(metadata, CitationMetadata)
        assert metadata.content_type_markers["has_requirements"] is True
        assert metadata.content_type_markers["has_sections"] is True
        assert metadata.content_type_markers["has_numbering"] is False
        assert metadata.content_type_markers["has_policy_language"] is True
    
    def test_extract_citation_metadata_numbered_content(self):
        """Test extracting metadata from numbered content."""
        strategy = PolicyCitationStrategy()
        document = Document(
            page_content="1. First requirement\n2. Second requirement\na) Sub-requirement",
            metadata={"section": "Requirements"}
        )
        context = Mock(spec=ProcessingContext)
        
        metadata = strategy.extract_citation_metadata(document, context)
        
        assert metadata.content_type_markers["has_numbering"] is True
    
    def test_format_citation_pdf_document(self):
        """Test formatting citation for PDF policy document."""
        strategy = PolicyCitationStrategy()
        source_ref = SourceReference(
            source_url="https://example.com/policies/security_policy_v3.2.pdf",
            resource_type=ResourceType.GUIDELINES,
            document_id="doc123",
            chunk_id="chunk456",
            section_title="4.3",
            page_number=15
        )
        
        citation = strategy.format_citation(source_ref, CitationVerbosity.DETAILED)
        assert citation == "[Security Policy V3.2, Section 4.3, Page 15]"
    
    def test_format_citation_web_policy(self):
        """Test formatting citation for web-based policy."""
        strategy = PolicyCitationStrategy()
        source_ref = SourceReference(
            source_url="https://company.com/policies/security",
            resource_type=ResourceType.GUIDELINES,
            document_id="doc123",
            chunk_id="chunk456",
            section_title="Access Control"
        )
        
        citation = strategy.format_citation(source_ref, CitationVerbosity.STANDARD)
        assert citation == "[Company Policy, Section Access Control]"
    
    def test_generate_citation_link(self):
        """Test generating citation link."""
        strategy = PolicyCitationStrategy()
        source_ref = SourceReference(
            source_url="https://example.com/policies/security.pdf",
            resource_type=ResourceType.GUIDELINES,
            document_id="doc123",
            chunk_id="chunk456"
        )
        
        link = strategy.generate_citation_link(source_ref)
        assert link == "https://example.com/policies/security.pdf"
    
    def test_get_policy_name_pdf(self):
        """Test getting policy name from PDF URL."""
        strategy = PolicyCitationStrategy()
        source_ref = SourceReference(
            source_url="https://example.com/policies/security_policy_v3.2.pdf",
            resource_type=ResourceType.GUIDELINES,
            document_id="doc123",
            chunk_id="chunk456"
        )
        
        name = strategy._get_policy_name(source_ref)
        assert name == "Security Policy V3.2"
    
    def test_get_policy_name_web(self):
        """Test getting policy name from web URL."""
        strategy = PolicyCitationStrategy()
        source_ref = SourceReference(
            source_url="https://company.com/policies/security",
            resource_type=ResourceType.GUIDELINES,
            document_id="doc123",
            chunk_id="chunk456"
        )
        
        name = strategy._get_policy_name(source_ref)
        assert name == "Company Policy"


class TestCitationStrategyRegistry:
    """Test the CitationStrategyRegistry class."""
    
    def test_init(self):
        """Test registry initialization."""
        registry = CitationStrategyRegistry()
        assert registry is not None
        assert isinstance(registry, CitationStrategyRegistry)
    
    def test_get_strategy_existing(self):
        """Test getting existing strategy."""
        registry = CitationStrategyRegistry()
        
        strategy = registry.get_strategy("default_citation")
        assert isinstance(strategy, DefaultCitationStrategy)
        
        strategy = registry.get_strategy("technical_citation")
        assert isinstance(strategy, TechnicalCitationStrategy)
        
        strategy = registry.get_strategy("web_citation")
        assert isinstance(strategy, WebCitationStrategy)
        
        strategy = registry.get_strategy("policy_citation")
        assert isinstance(strategy, PolicyCitationStrategy)
    
    def test_get_strategy_nonexistent(self):
        """Test getting non-existent strategy returns default."""
        registry = CitationStrategyRegistry()
        
        strategy = registry.get_strategy("nonexistent_strategy")
        assert isinstance(strategy, DefaultCitationStrategy)
    
    def test_register_strategy(self):
        """Test registering a new strategy."""
        registry = CitationStrategyRegistry()
        
        class CustomStrategy(CitationStrategy):
            def extract_citation_metadata(self, document: Document, context: ProcessingContext) -> CitationMetadata:
                return CitationMetadata()
            
            def format_citation(self, source_ref: SourceReference, verbosity: CitationVerbosity) -> str:
                return "[Custom Citation]"
            
            def generate_citation_link(self, source_ref: SourceReference) -> Optional[str]:
                return "https://custom.com"
        
        custom_strategy_instance = CustomStrategy()
        registry.register_strategy("custom_strategy", custom_strategy_instance)
        
        strategy = registry.get_strategy("custom_strategy")
        assert isinstance(strategy, CustomStrategy)
    
    def test_list_strategies(self):
        """Test listing available strategies."""
        registry = CitationStrategyRegistry()
        
        strategies = registry.list_strategies()
        assert isinstance(strategies, list)
        assert "default_citation" in strategies
        assert "technical_citation" in strategies
        assert "web_citation" in strategies
        assert "policy_citation" in strategies
        assert "rule_citation" in strategies


class TestCitationStrategiesEdgeCases:
    """Test edge cases for citation strategies."""
    
    def test_default_strategy_with_empty_content(self):
        """Test default strategy with empty document content."""
        strategy = DefaultCitationStrategy()
        document = Document(page_content="", metadata={})
        context = Mock(spec=ProcessingContext)
        
        metadata = strategy.extract_citation_metadata(document, context)
        
        assert metadata.content_type_markers["content_length"] == 0
        assert metadata.content_type_markers["word_count"] == 0
    
    def test_technical_strategy_with_mixed_content(self):
        """Test technical strategy with mixed content types."""
        strategy = TechnicalCitationStrategy()
        document = Document(
            page_content="def api_function(): # API v1.2 example code",
            metadata={"section": "API Examples"}
        )
        context = Mock(spec=ProcessingContext)
        
        metadata = strategy.extract_citation_metadata(document, context)
        
        assert metadata.content_type_markers["has_code"] is True
        assert metadata.content_type_markers["has_api_refs"] is True
        assert metadata.content_type_markers["has_examples"] is True
        assert metadata.content_type_markers["has_version_info"] is True
    
    def test_web_strategy_with_complex_section_title(self):
        """Test web strategy with complex section title."""
        strategy = WebCitationStrategy()
        
        anchor = strategy._create_anchor("Section 1.2.3 (Advanced Topics)")
        assert anchor == "section-123-advanced-topics"
    
    def test_policy_strategy_with_various_requirement_words(self):
        """Test policy strategy with various requirement words."""
        strategy = PolicyCitationStrategy()
        
        requirement_words = ["must", "shall", "required", "mandatory"]
        for word in requirement_words:
            document = Document(
                page_content=f"Users {word} follow this policy.",
                metadata={"section": "Requirements"}
            )
            context = Mock(spec=ProcessingContext)
            
            metadata = strategy.extract_citation_metadata(document, context)
            assert metadata.content_type_markers["has_requirements"] is True
    
    def test_strategies_with_unicode_content(self):
        """Test strategies with unicode content."""
        strategy = DefaultCitationStrategy()
        document = Document(
            page_content="这是测试内容",
            metadata={"section": "介绍"}
        )
        context = Mock(spec=ProcessingContext)
        
        metadata = strategy.extract_citation_metadata(document, context)
        
        assert metadata.content_type_markers["content_length"] == len("这是测试内容")
        assert metadata.content_type_markers["word_count"] == len("这是测试内容".split())


class TestCitationStrategiesIntegration:
    """Integration tests for citation strategies."""
    
    def test_strategies_work_with_real_documents(self):
        """Test that strategies work with real document structures."""
        # Test default strategy
        default_strategy = DefaultCitationStrategy()
        document = Document(
            page_content="This is a test document with some content.",
            metadata={"section": "Introduction", "page": 1}
        )
        context = Mock(spec=ProcessingContext)
        
        metadata = default_strategy.extract_citation_metadata(document, context)
        assert isinstance(metadata, CitationMetadata)
        
        source_ref = SourceReference(
            source_url="https://example.com/doc",
            resource_type=ResourceType.DSL,
            document_id="doc123",
            chunk_id="chunk456",
            section_title="Introduction",
            page_number=1
        )
        
        citation = default_strategy.format_citation(source_ref, CitationVerbosity.STANDARD)
        assert "[example.com, p. 1]" in citation
        
        link = default_strategy.generate_citation_link(source_ref)
        assert link == "https://example.com/doc"
    
    def test_strategy_registry_workflow(self):
        """Test complete workflow with strategy registry."""
        registry = CitationStrategyRegistry()
        
        # Get different strategies
        default_strategy = registry.get_strategy("default_citation")
        technical_strategy = registry.get_strategy("technical_citation")
        
        # Test they are different instances
        assert default_strategy is not technical_strategy
        assert type(default_strategy) != type(technical_strategy)
        
        # Test they both implement the interface
        document = Document(page_content="test", metadata={})
        context = Mock(spec=ProcessingContext)
        source_ref = SourceReference(
            source_url="https://example.com",
            resource_type=ResourceType.DSL,
            document_id="doc123",
            chunk_id="chunk456"
        )
        
        # Both should work
        default_metadata = default_strategy.extract_citation_metadata(document, context)
        technical_metadata = technical_strategy.extract_citation_metadata(document, context)
        
        assert isinstance(default_metadata, CitationMetadata)
        assert isinstance(technical_metadata, CitationMetadata)
        
        default_citation = default_strategy.format_citation(source_ref, CitationVerbosity.STANDARD)
        technical_citation = technical_strategy.format_citation(source_ref, CitationVerbosity.STANDARD)
        
        assert isinstance(default_citation, str)
        assert isinstance(technical_citation, str)
    
    def test_strategies_with_different_verbosity_levels(self):
        """Test strategies with different verbosity levels."""
        strategy = DefaultCitationStrategy()
        source_ref = SourceReference(
            source_url="https://example.com/doc",
            resource_type=ResourceType.DSL,
            document_id="doc123",
            chunk_id="chunk456",
            page_number=5,
            section_title="Introduction",
            exact_quote="This is a test quote"
        )
        
        verbosity_levels = [
            CitationVerbosity.NONE,
            CitationVerbosity.MINIMAL,
            CitationVerbosity.STANDARD,
            CitationVerbosity.DETAILED,
            CitationVerbosity.FORENSIC
        ]
        
        for verbosity in verbosity_levels:
            citation = strategy.format_citation(source_ref, verbosity)
            assert isinstance(citation, str)
            
            if verbosity == CitationVerbosity.NONE:
                assert citation == ""
            else:
                assert len(citation) > 0
