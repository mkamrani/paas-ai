"""
Integration tests for the citations system.

Tests the complete citations system integration including:
- End-to-end workflows
- Cross-component interactions
- Real citation processing (with mocking)
- Error propagation
- Configuration validation across components
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict, Any

from src.paas_ai.core.rag.citations.enricher import CitationEnricher
from src.paas_ai.core.rag.citations.formatters import CitationFormatter
from src.paas_ai.core.rag.citations.strategies import (
    CitationStrategyRegistry,
    DefaultCitationStrategy,
    TechnicalCitationStrategy,
    WebCitationStrategy,
    PolicyCitationStrategy
)
from src.paas_ai.core.rag.citations.models import SourceReference, CitationMetadata, CitationAwareResult
from src.paas_ai.core.rag.processing.base import ProcessingContext
from src.paas_ai.core.config.schemas import CitationConfig, ResourceType, CitationVerbosity, CitationFormat
from langchain_core.documents import Document


@pytest.fixture(autouse=True)
def clean_registry():
    """Ensure registry is clean before and after each test."""
    # Reset to defaults before each test
    registry = CitationStrategyRegistry()
    # No reset method exists, so we just ensure clean state
    yield
    # Clean up after each test


class TestCitationsSystemIntegration:
    """Integration tests for the complete citations system."""
    
    @pytest.mark.asyncio
    async def test_full_citation_workflow(self):
        """Test complete workflow from document processing to citation formatting."""
        # Setup configuration
        config = CitationConfig(
            enabled=True,
            verbosity=CitationVerbosity.DETAILED,
            include_quotes=True,
            include_confidence=True,
            max_quote_length=100,
            resource_overrides={
                ResourceType.DSL: CitationVerbosity.FORENSIC
            },
            strategies={
                ResourceType.DSL: "technical_citation"
            }
        )
        
        # Create enricher
        enricher = CitationEnricher(config)
        
        # Create formatter
        formatter = CitationFormatter()
        
        # Setup documents
        documents = [
            Document(
                page_content="You can specify resource requests and limits for each container in a pod.",
                metadata={"page": 1, "section": "Resource Management"}
            ),
            Document(
                page_content="Pods are the smallest deployable units in Kubernetes.",
                metadata={"page": 2, "section": "Pod Overview"}
            )
        ]
        
        # Setup context
        context = Mock(spec=ProcessingContext)
        context.resource = Mock()
        context.resource.url = "https://kubernetes.io/docs/concepts/workloads/pods/"
        context.resource.resource_type = ResourceType.DSL
        context.resource.tags = ["kubernetes", "pods"]
        context.resource.priority = 1
        context.pipeline_id = "pipeline123"
        
        # Process documents through enricher
        enriched_documents = await enricher(documents, context)
        
        # Verify enrichment
        assert len(enriched_documents) == 2
        assert "citation_reference" in enriched_documents[0].metadata
        assert "citation_metadata" in enriched_documents[0].metadata
        
        # Extract source references
        source_refs = []
        for doc in enriched_documents:
            citation_ref_data = doc.metadata["citation_reference"]
            source_ref = SourceReference(**citation_ref_data)
            source_refs.append(source_ref)
        
        # Format citations
        citations = formatter.format_multiple_citations(
            source_refs,
            CitationVerbosity.DETAILED,
            CitationFormat.INLINE,
            ["technical_citation"] * len(source_refs)
        )
        
        # Verify citations
        assert len(citations) == 2
        for citation in citations:
            assert isinstance(citation, str)
            assert len(citation) > 0
            assert "[" in citation and "]" in citation
    
    @pytest.mark.asyncio
    async def test_citation_workflow_with_different_formats(self):
        """Test citation workflow with different format styles."""
        config = CitationConfig(enabled=True, verbosity=CitationVerbosity.STANDARD)
        enricher = CitationEnricher(config)
        formatter = CitationFormatter()

        # Setup document
        documents = [Document(
            page_content="This is test content for citation formatting.",
            metadata={"page": 1, "section": "Introduction"}
        )]

        context = Mock(spec=ProcessingContext)
        context.resource = Mock()
        context.resource.url = "https://example.com/doc"
        context.resource.resource_type = ResourceType.CONTEXTUAL
        context.resource.tags = []
        context.resource.priority = 1
        context.pipeline_id = "pipeline123"

        # Process document
        enriched_documents = await enricher(documents, context)
        source_ref_data = enriched_documents[0].metadata["citation_reference"]
        source_ref = SourceReference(**source_ref_data)
        
        # Test different formats
        formats = [
            CitationFormat.INLINE,
            CitationFormat.FOOTNOTE,
            CitationFormat.ACADEMIC,
            CitationFormat.STRUCTURED
        ]
        
        for format_style in formats:
            citation = formatter.format_citation(
                source_ref,
                CitationVerbosity.STANDARD,
                format_style
            )
            
            assert isinstance(citation, str)
            
            if format_style == CitationFormat.INLINE:
                assert "[" in citation and "]" in citation
            elif format_style == CitationFormat.FOOTNOTE:
                assert citation.startswith("[^") and citation.endswith("]")
            elif format_style == CitationFormat.ACADEMIC:
                assert citation.startswith("(") and citation.endswith(")")
            elif format_style == CitationFormat.STRUCTURED:
                assert citation.startswith("{") and citation.endswith("}")
    
    @pytest.mark.asyncio
    async def test_citation_workflow_with_different_verbosity_levels(self):
        """Test citation workflow with different verbosity levels."""
        config = CitationConfig(enabled=True, verbosity=CitationVerbosity.STANDARD)
        enricher = CitationEnricher(config)
        formatter = CitationFormatter()

        # Setup document
        documents = [Document(
            page_content="This is test content with detailed information.",
            metadata={"page": 5, "section": "Detailed Section"}
        )]

        context = Mock(spec=ProcessingContext)
        context.resource = Mock()
        context.resource.url = "https://example.com/doc"
        context.resource.resource_type = ResourceType.DSL
        context.resource.tags = []
        context.resource.priority = 1
        context.pipeline_id = "pipeline123"

        # Process document
        enriched_documents = await enricher(documents, context)
        source_ref_data = enriched_documents[0].metadata["citation_reference"]
        source_ref = SourceReference(**source_ref_data)
        
        # Test different verbosity levels
        verbosity_levels = [
            CitationVerbosity.NONE,
            CitationVerbosity.MINIMAL,
            CitationVerbosity.STANDARD,
            CitationVerbosity.DETAILED,
            CitationVerbosity.FORENSIC
        ]
        
        for verbosity in verbosity_levels:
            citation = formatter.format_citation(
                source_ref,
                verbosity,
                CitationFormat.INLINE
            )
            
            assert isinstance(citation, str)
            
            if verbosity == CitationVerbosity.NONE:
                assert citation == ""
            else:
                assert len(citation) > 0
                
                # Higher verbosity should generally produce longer citations
                if verbosity in [CitationVerbosity.DETAILED, CitationVerbosity.FORENSIC]:
                    assert len(citation) >= len(formatter.format_citation(
                        source_ref, CitationVerbosity.MINIMAL, CitationFormat.INLINE
                    ))
    
    @pytest.mark.asyncio
    async def test_citation_workflow_with_different_strategies(self):
        """Test citation workflow with different citation strategies."""
        config = CitationConfig(enabled=True, verbosity=CitationVerbosity.STANDARD)
        enricher = CitationEnricher(config)
        formatter = CitationFormatter()
        
        # Test different resource types with their strategies
        test_cases = [
            (ResourceType.DSL, "technical_citation"),
            (ResourceType.CONTEXTUAL, "web_citation"),
            (ResourceType.GUIDELINES, "policy_citation"),
            (ResourceType.DOMAIN_RULES, "rule_citation")
        ]
        
        for resource_type, expected_strategy in test_cases:
            documents = [Document(
                page_content="Test content for different strategies.",
                metadata={"page": 1, "section": "Test Section"}
            )]
            
            context = Mock(spec=ProcessingContext)
            context.resource = Mock()
            context.resource.url = "https://example.com/doc"
            context.resource.resource_type = resource_type
            context.resource.tags = []
            context.resource.priority = 1
            context.pipeline_id = "pipeline123"
            
            # Process document
            enriched_documents = await enricher(documents, context)
            source_ref_data = enriched_documents[0].metadata["citation_reference"]
            source_ref = SourceReference(**source_ref_data)
            
            # Format citation
            citation = formatter.format_citation(
                source_ref,
                CitationVerbosity.STANDARD,
                CitationFormat.INLINE,
                expected_strategy
            )
            
            assert isinstance(citation, str)
            assert len(citation) > 0
    
    @pytest.mark.asyncio
    async def test_citation_workflow_with_resource_overrides(self):
        """Test citation workflow with resource-specific verbosity overrides."""
        config = CitationConfig(
            enabled=True,
            verbosity=CitationVerbosity.MINIMAL,
            resource_overrides={
                ResourceType.DSL: CitationVerbosity.FORENSIC,
                ResourceType.CONTEXTUAL: CitationVerbosity.DETAILED,
                ResourceType.GUIDELINES: CitationVerbosity.STANDARD
            }
        )
        enricher = CitationEnricher(config)
        formatter = CitationFormatter()
        
        # Test different resource types
        test_cases = [
            (ResourceType.DSL, CitationVerbosity.FORENSIC),
            (ResourceType.CONTEXTUAL, CitationVerbosity.DETAILED),
            (ResourceType.GUIDELINES, CitationVerbosity.STANDARD),
            (ResourceType.DOMAIN_RULES, CitationVerbosity.MINIMAL)  # Should use default
        ]
        
        for resource_type, expected_verbosity in test_cases:
            documents = [Document(
                page_content="Test content for verbosity overrides.",
                metadata={"page": 1, "section": "Test Section"}
            )]
            
            context = Mock(spec=ProcessingContext)
            context.resource = Mock()
            context.resource.url = "https://example.com/doc"
            context.resource.resource_type = resource_type
            context.resource.tags = []
            context.resource.priority = 1
            context.pipeline_id = "pipeline123"
            
            # Process document
            enriched_documents = await enricher(documents, context)
            
            # Verify verbosity override was applied
            actual_verbosity = enriched_documents[0].metadata["citation_verbosity"]
            assert actual_verbosity == expected_verbosity.value
    
    @pytest.mark.asyncio
    async def test_citation_workflow_with_quotes_and_confidence(self):
        """Test citation workflow with quotes and confidence scoring enabled."""
        config = CitationConfig(
            enabled=True,
            verbosity=CitationVerbosity.DETAILED,
            include_quotes=True,
            include_confidence=True,
            max_quote_length=50
        )
        enricher = CitationEnricher(config)
        formatter = CitationFormatter()
        
        documents = [Document(
            page_content="This is a test sentence with important information. This is another sentence.",
            metadata={"page": 1, "section": "Introduction"}
        )]
        
        context = Mock(spec=ProcessingContext)
        context.resource = Mock()
        context.resource.url = "https://example.com/doc"
        context.resource.resource_type = ResourceType.DSL
        context.resource.tags = []
        context.resource.priority = 1
        context.pipeline_id = "pipeline123"
        
        # Process document
        enriched_documents = await enricher(documents, context)
        source_ref_data = enriched_documents[0].metadata["citation_reference"]
        source_ref = SourceReference(**source_ref_data)
        
        # Verify quotes and confidence are included
        assert source_ref.exact_quote is not None
        assert len(source_ref.exact_quote) <= 50
        assert source_ref.confidence_score is not None
        assert 0.0 <= source_ref.confidence_score <= 1.0
        
        # Format citation
        citation = formatter.format_citation(
            source_ref,
            CitationVerbosity.FORENSIC,
            CitationFormat.INLINE
        )
        
        # Should include quote in forensic verbosity
        assert source_ref.exact_quote in citation
    
    @pytest.mark.asyncio
    async def test_citation_workflow_with_multiple_documents(self):
        """Test citation workflow with multiple documents."""
        config = CitationConfig(enabled=True, verbosity=CitationVerbosity.STANDARD)
        enricher = CitationEnricher(config)
        formatter = CitationFormatter()
        
        documents = [
            Document(
                page_content="First document content.",
                metadata={"page": 1, "section": "Introduction"}
            ),
            Document(
                page_content="Second document content.",
                metadata={"page": 2, "section": "Overview"}
            ),
            Document(
                page_content="Third document content.",
                metadata={"page": 3, "section": "Details"}
            )
        ]
        
        context = Mock(spec=ProcessingContext)
        context.resource = Mock()
        context.resource.url = "https://example.com/doc"
        context.resource.resource_type = ResourceType.DSL
        context.resource.tags = []
        context.resource.priority = 1
        context.pipeline_id = "pipeline123"
        
        # Process documents
        enriched_documents = await enricher(documents, context)
        
        # Verify all documents were processed
        assert len(enriched_documents) == 3
        
        # Extract source references
        source_refs = []
        for doc in enriched_documents:
            citation_ref_data = doc.metadata["citation_reference"]
            source_ref = SourceReference(**citation_ref_data)
            source_refs.append(source_ref)
        
        # Format all citations
        citations = formatter.format_multiple_citations(
            source_refs,
            CitationVerbosity.STANDARD,
            CitationFormat.INLINE
        )
        
        # Verify all citations were formatted
        assert len(citations) == 3
        for i, citation in enumerate(citations):
            assert isinstance(citation, str)
            assert len(citation) > 0
            # Each citation should be unique (different chunk IDs)
            assert source_refs[i].chunk_id != source_refs[(i + 1) % 3].chunk_id


class TestCitationsSystemErrorHandling:
    """Test error handling across the citations system."""
    
    @pytest.mark.asyncio
    async def test_error_handling_in_enricher(self):
        """Test error handling in citation enricher."""
        config = CitationConfig(enabled=True, verbosity=CitationVerbosity.STANDARD)
        enricher = CitationEnricher(config)
        
        documents = [
            Document(page_content="Valid content", metadata={}),
            Document(page_content="Content that causes error", metadata={}),
            Document(page_content="More valid content", metadata={})
        ]
        
        context = Mock(spec=ProcessingContext)
        context.resource = Mock()
        context.resource.url = "https://example.com/doc"
        context.resource.resource_type = ResourceType.DSL
        context.resource.tags = []
        context.resource.priority = 1
        context.pipeline_id = "pipeline123"
        
        with patch.object(enricher.strategy_registry, 'get_strategy') as mock_get_strategy:
            mock_strategy = Mock()
            
            def side_effect(doc, ctx):
                if "error" in doc.page_content:
                    raise Exception("Processing error")
                return CitationMetadata()
            
            mock_strategy.extract_citation_metadata.side_effect = side_effect
            mock_get_strategy.return_value = mock_strategy
            
            # Should not raise exception
            result = await enricher(documents, context)
            
            # Should process all documents
            assert len(result) == 3
            
            # Valid documents should have citation metadata
            assert "citation_reference" in result[0].metadata
            assert "citation_reference" in result[2].metadata
            
            # Error document should not have citation metadata
            assert "citation_reference" not in result[1].metadata
    
    def test_error_handling_in_formatter(self):
        """Test error handling in citation formatter."""
        formatter = CitationFormatter()
        
        # Test with invalid source reference
        with pytest.raises(AttributeError):
            formatter.format_citation(
                None,
                CitationVerbosity.STANDARD,
                CitationFormat.INLINE
            )
        
        # Test with invalid format
        source_ref = SourceReference(
            source_url="https://example.com/doc",
            resource_type=ResourceType.DSL,
            document_id="doc123",
            chunk_id="chunk456"
        )
        
        with patch.object(formatter.strategy_registry, 'get_strategy') as mock_get_strategy:
            mock_strategy = Mock()
            mock_strategy.format_citation.return_value = "[Example Doc]"
            mock_get_strategy.return_value = mock_strategy
            
            # Should handle invalid format gracefully
            citation = formatter.format_citation(
                source_ref,
                CitationVerbosity.STANDARD,
                "invalid_format"
            )
            
            assert citation == "[Example Doc]"
    
    def test_error_handling_in_strategy_registry(self):
        """Test error handling in strategy registry."""
        registry = CitationStrategyRegistry()
        
        # Test getting non-existent strategy
        strategy = registry.get_strategy("nonexistent_strategy")
        assert isinstance(strategy, DefaultCitationStrategy)
        
        # Test registering invalid strategy
        registry.register_strategy("test_strategy", None)
        # Should not raise error, but should store None
    
    @pytest.mark.asyncio
    async def test_error_handling_with_malformed_documents(self):
        """Test error handling with malformed documents."""
        config = CitationConfig(enabled=True, verbosity=CitationVerbosity.STANDARD)
        enricher = CitationEnricher(config)
        
        # Test with empty document
        documents = [Document(page_content="", metadata={})]
        
        context = Mock(spec=ProcessingContext)
        context.resource = Mock()
        context.resource.url = "https://example.com/doc"
        context.resource.resource_type = ResourceType.DSL
        context.resource.tags = []
        context.resource.priority = 1
        context.pipeline_id = "pipeline123"
        
        with patch.object(enricher.strategy_registry, 'get_strategy') as mock_get_strategy:
            mock_strategy = Mock()
            mock_strategy.extract_citation_metadata.return_value = CitationMetadata()
            mock_get_strategy.return_value = mock_strategy
            
            result = await enricher(documents, context)
            
            # Should handle empty document gracefully
            assert len(result) == 1
            assert "citation_reference" in result[0].metadata
    
    @pytest.mark.asyncio
    async def test_error_handling_with_missing_context(self):
        """Test error handling with missing context information."""
        config = CitationConfig(enabled=True, verbosity=CitationVerbosity.STANDARD)
        enricher = CitationEnricher(config)
        
        documents = [Document(page_content="Test content", metadata={})]
        
        # Test with incomplete context
        context = Mock(spec=ProcessingContext)
        context.resource = Mock()
        context.resource.url = "https://example.com/doc"
        context.resource.resource_type = ResourceType.DSL
        context.resource.tags = None  # Missing tags
        context.resource.priority = None  # Missing priority
        context.pipeline_id = "pipeline123"
        
        with patch.object(enricher.strategy_registry, 'get_strategy') as mock_get_strategy:
            mock_strategy = Mock()
            mock_strategy.extract_citation_metadata.return_value = CitationMetadata()
            mock_get_strategy.return_value = mock_strategy
            
            result = await enricher(documents, context)
            
            # Should handle missing context gracefully
            assert len(result) == 1
            citation_ref = result[0].metadata["citation_reference"]
            assert citation_ref["metadata"]["tags"] is None
            assert citation_ref["metadata"]["priority"] is None


class TestCitationsSystemPerformance:
    """Test performance characteristics of the citations system."""
    
    @pytest.mark.asyncio
    async def test_enricher_performance_with_large_document_set(self):
        """Test enricher performance with large document set."""
        config = CitationConfig(enabled=True, verbosity=CitationVerbosity.STANDARD)
        enricher = CitationEnricher(config)
        
        # Create large document set
        documents = []
        for i in range(100):
            documents.append(Document(
                page_content=f"Document {i} content with some text.",
                metadata={"page": i + 1, "section": f"Section {i}"}
            ))
        
        context = Mock(spec=ProcessingContext)
        context.resource = Mock()
        context.resource.url = "https://example.com/doc"
        context.resource.resource_type = ResourceType.DSL
        context.resource.tags = []
        context.resource.priority = 1
        context.pipeline_id = "pipeline123"
        
        with patch.object(enricher.strategy_registry, 'get_strategy') as mock_get_strategy:
            mock_strategy = Mock()
            mock_strategy.extract_citation_metadata.return_value = CitationMetadata()
            mock_get_strategy.return_value = mock_strategy
            
            import time
            start_time = time.time()
            result = await enricher(documents, context)
            end_time = time.time()
            
            # Should process all documents
            assert len(result) == 100
            
            # Should be reasonably fast (less than 1 second for 100 documents)
            assert (end_time - start_time) < 1.0
    
    def test_formatter_performance_with_multiple_citations(self):
        """Test formatter performance with multiple citations."""
        formatter = CitationFormatter()
        
        # Create large set of source references
        source_refs = []
        for i in range(100):
            source_refs.append(SourceReference(
                source_url=f"https://example.com/doc{i}",
                resource_type=ResourceType.DSL,
                document_id=f"doc{i}",
                chunk_id=f"chunk{i}"
            ))
        
        with patch.object(formatter.strategy_registry, 'get_strategy') as mock_get_strategy:
            mock_strategy = Mock()
            mock_strategy.format_citation.return_value = "[Example Doc]"
            mock_get_strategy.return_value = mock_strategy
            
            import time
            start_time = time.time()
            citations = formatter.format_multiple_citations(
                source_refs,
                CitationVerbosity.STANDARD,
                CitationFormat.INLINE
            )
            end_time = time.time()
            
            # Should format all citations
            assert len(citations) == 100
            
            # Should be reasonably fast (less than 0.5 seconds for 100 citations)
            assert (end_time - start_time) < 0.5
    
    def test_strategy_registry_performance(self):
        """Test strategy registry performance."""
        registry = CitationStrategyRegistry()
        
        import time
        start_time = time.time()
        
        # Test multiple strategy lookups
        for _ in range(1000):
            strategy = registry.get_strategy("default_citation")
            assert strategy is not None
        
        end_time = time.time()
        
        # Should be very fast (less than 0.1 seconds for 1000 lookups)
        assert (end_time - start_time) < 0.1


class TestCitationsSystemCompatibility:
    """Test compatibility and interoperability of the citations system."""
    
    def test_config_compatibility(self):
        """Test that configurations work across different components."""
        config = CitationConfig(
            enabled=True,
            verbosity=CitationVerbosity.DETAILED,
            include_quotes=True,
            include_confidence=True,
            max_quote_length=100,
            resource_overrides={
                ResourceType.DSL: CitationVerbosity.FORENSIC
            },
            strategies={
                ResourceType.DSL: "technical_citation"
            }
        )
        
        # Test that config works with enricher
        enricher = CitationEnricher(config)
        assert enricher.config == config
        
        # Test that config works with formatter
        formatter = CitationFormatter()
        # Formatter doesn't directly use config, but should work with any config values
        
        # Test that config works with strategies
        registry = CitationStrategyRegistry()
        strategy = registry.get_strategy(config.strategies[ResourceType.DSL])
        assert strategy is not None
    
    def test_strategy_interface_compatibility(self):
        """Test that all strategies implement the required interface."""
        registry = CitationStrategyRegistry()
        
        strategies = [
            "default_citation",
            "technical_citation",
            "web_citation",
            "policy_citation",
            "rule_citation"
        ]
        
        for strategy_name in strategies:
            strategy = registry.get_strategy(strategy_name)
            
            # Check required methods
            assert hasattr(strategy, 'extract_citation_metadata')
            assert hasattr(strategy, 'format_citation')
            assert hasattr(strategy, 'generate_citation_link')
            assert callable(strategy.extract_citation_metadata)
            assert callable(strategy.format_citation)
            assert callable(strategy.generate_citation_link)
    
    def test_model_serialization_compatibility(self):
        """Test that models can be serialized and deserialized correctly."""
        # Test SourceReference
        original_source_ref = SourceReference(
            source_url="https://example.com/doc",
            resource_type=ResourceType.DSL,
            document_id="doc123",
            chunk_id="chunk456",
            page_number=1,
            section_title="Introduction",
            exact_quote="This is a test quote",
            confidence_score=0.95,
            metadata={"tag": "test"}
        )
        
        # Serialize and deserialize
        data = original_source_ref.model_dump()
        deserialized_source_ref = SourceReference(**data)
        
        # Verify they are equal
        assert deserialized_source_ref == original_source_ref
        
        # Test CitationMetadata
        original_metadata = CitationMetadata(
            document_structure={"sections": 2},
            headings=[{"text": "Introduction", "level": 1}],
            content_type_markers={"has_structure": True}
        )
        
        # Serialize and deserialize
        data = original_metadata.model_dump()
        deserialized_metadata = CitationMetadata(**data)
        
        # Verify they are equal
        assert deserialized_metadata == original_metadata
    
    @pytest.mark.asyncio
    async def test_cross_component_error_handling(self):
        """Test error handling across components."""
        config = CitationConfig(enabled=True, verbosity=CitationVerbosity.STANDARD)
        enricher = CitationEnricher(config)
        formatter = CitationFormatter()
        
        # Test that errors from enricher don't break formatter
        documents = [Document(page_content="Test content", metadata={})]
        context = Mock(spec=ProcessingContext)
        context.resource = Mock()
        context.resource.url = "https://example.com/doc"
        context.resource.resource_type = ResourceType.DSL
        context.resource.tags = []
        context.resource.priority = 1
        context.pipeline_id = "pipeline123"
        
        with patch.object(enricher.strategy_registry, 'get_strategy') as mock_get_strategy:
            mock_strategy = Mock()
            mock_strategy.extract_citation_metadata.side_effect = Exception("Enricher error")
            mock_get_strategy.return_value = mock_strategy
            
            # Should not raise exception
            result = await enricher(documents, context)
            
            # Should return document without citation metadata
            assert len(result) == 1
            assert "citation_reference" not in result[0].metadata
        
        # Test that formatter can still work with valid source references
        source_ref = SourceReference(
            source_url="https://example.com/doc",
            resource_type=ResourceType.DSL,
            document_id="doc123",
            chunk_id="chunk456"
        )
        
        citation = formatter.format_citation(
            source_ref,
            CitationVerbosity.STANDARD,
            CitationFormat.INLINE
        )
        
        assert isinstance(citation, str)
        assert len(citation) > 0
