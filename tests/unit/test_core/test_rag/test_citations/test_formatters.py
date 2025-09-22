"""
Unit tests for citation formatters.

Tests all components of the citation formatters including:
- CitationFormatter class
- Different format styles (INLINE, FOOTNOTE, ACADEMIC, STRUCTURED)
- Multiple citation formatting
- Error handling and edge cases
- Integration with strategies
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import List

from src.paas_ai.core.rag.citations.formatters import CitationFormatter
from src.paas_ai.core.rag.citations.models import SourceReference
from src.paas_ai.core.rag.citations.strategies import CitationStrategyRegistry
from src.paas_ai.core.config.schemas import CitationFormat, CitationVerbosity, ResourceType


class TestCitationFormatter:
    """Test the CitationFormatter class."""
    
    def test_init(self):
        """Test formatter initialization."""
        formatter = CitationFormatter()
        assert formatter is not None
        assert isinstance(formatter, CitationFormatter)
        assert isinstance(formatter.strategy_registry, CitationStrategyRegistry)
    
    def test_format_citation_inline_format(self):
        """Test formatting citation with INLINE format."""
        formatter = CitationFormatter()
        source_ref = SourceReference(
            source_url="https://example.com/doc",
            resource_type=ResourceType.DSL,
            document_id="doc123",
            chunk_id="chunk456",
            section_title="Introduction"
        )
        
        with patch.object(formatter.strategy_registry, 'get_strategy') as mock_get_strategy:
            mock_strategy = Mock()
            mock_strategy.format_citation.return_value = "[Example Doc, Introduction]"
            mock_get_strategy.return_value = mock_strategy
            
            citation = formatter.format_citation(
                source_ref, 
                CitationVerbosity.STANDARD, 
                CitationFormat.INLINE
            )
            
            assert citation == "[Example Doc, Introduction]"
            mock_strategy.format_citation.assert_called_once_with(source_ref, CitationVerbosity.STANDARD)
    
    def test_format_citation_footnote_format(self):
        """Test formatting citation with FOOTNOTE format."""
        formatter = CitationFormatter()
        source_ref = SourceReference(
            source_url="https://example.com/doc",
            resource_type=ResourceType.DSL,
            document_id="doc123",
            chunk_id="chunk456",
            section_title="Introduction"
        )
        
        with patch.object(formatter.strategy_registry, 'get_strategy') as mock_get_strategy:
            mock_strategy = Mock()
            mock_strategy.format_citation.return_value = "[Example Doc, Introduction]"
            mock_get_strategy.return_value = mock_strategy
            
            citation = formatter.format_citation(
                source_ref, 
                CitationVerbosity.STANDARD, 
                CitationFormat.FOOTNOTE
            )
            
            # Should return footnote reference format
            assert citation.startswith("[^")
            assert citation.endswith("]")
            assert len(citation) > 3  # Should have some content between [^ and ]
    
    def test_format_citation_academic_format(self):
        """Test formatting citation with ACADEMIC format."""
        formatter = CitationFormatter()
        source_ref = SourceReference(
            source_url="https://example.com/doc.pdf",
            resource_type=ResourceType.DSL,
            document_id="doc123",
            chunk_id="chunk456",
            page_number=5,
            section_title="Introduction",
            metadata={"processed_at": "2024-01-01"}
        )
        
        citation = formatter.format_citation(
            source_ref, 
            CitationVerbosity.STANDARD, 
            CitationFormat.ACADEMIC
        )
        
        # Should be in academic format with parentheses
        assert citation.startswith("(")
        assert citation.endswith(")")
        assert "doc" in citation
        assert "5" in citation
        assert "https://example.com/doc.pdf" in citation
    
    def test_format_citation_structured_format(self):
        """Test formatting citation with STRUCTURED format."""
        formatter = CitationFormatter()
        source_ref = SourceReference(
            source_url="https://example.com/doc",
            resource_type=ResourceType.DSL,
            document_id="doc123",
            chunk_id="chunk456",
            page_number=5,
            section_title="Introduction"
        )
        
        citation = formatter.format_citation(
            source_ref, 
            CitationVerbosity.STANDARD, 
            CitationFormat.STRUCTURED
        )
        
        # Should be in structured format with curly braces
        assert citation.startswith("{")
        assert citation.endswith("}")
        assert "source:" in citation
        assert "url:" in citation
        assert "type:" in citation
        assert "page:" in citation
        assert "section:" in citation
    
    def test_format_citation_none_verbosity(self):
        """Test formatting citation with NONE verbosity."""
        formatter = CitationFormatter()
        source_ref = SourceReference(
            source_url="https://example.com/doc",
            resource_type=ResourceType.DSL,
            document_id="doc123",
            chunk_id="chunk456"
        )
        
        citation = formatter.format_citation(
            source_ref, 
            CitationVerbosity.NONE, 
            CitationFormat.INLINE
        )
        
        assert citation == ""
    
    def test_format_citation_with_custom_strategy(self):
        """Test formatting citation with custom strategy."""
        formatter = CitationFormatter()
        source_ref = SourceReference(
            source_url="https://example.com/doc",
            resource_type=ResourceType.DSL,
            document_id="doc123",
            chunk_id="chunk456"
        )
        
        with patch.object(formatter.strategy_registry, 'get_strategy') as mock_get_strategy:
            mock_strategy = Mock()
            mock_strategy.format_citation.return_value = "[Custom Strategy Citation]"
            mock_get_strategy.return_value = mock_strategy
            
            citation = formatter.format_citation(
                source_ref, 
                CitationVerbosity.STANDARD, 
                CitationFormat.INLINE,
                "custom_strategy"
            )
            
            assert citation == "[Custom Strategy Citation]"
            mock_get_strategy.assert_called_once_with("custom_strategy")
    
    def test_format_multiple_citations(self):
        """Test formatting multiple citations."""
        formatter = CitationFormatter()
        
        source_refs = [
            SourceReference(
                source_url="https://example.com/doc1",
                resource_type=ResourceType.DSL,
                document_id="doc123",
                chunk_id="chunk456"
            ),
            SourceReference(
                source_url="https://example.com/doc2",
                resource_type=ResourceType.CONTEXTUAL,
                document_id="doc456",
                chunk_id="chunk789"
            )
        ]
        
        with patch.object(formatter.strategy_registry, 'get_strategy') as mock_get_strategy:
            mock_strategy = Mock()
            mock_strategy.format_citation.side_effect = ["[Doc 1]", "[Doc 2]"]
            mock_get_strategy.return_value = mock_strategy
            
            citations = formatter.format_multiple_citations(
                source_refs, 
                CitationVerbosity.STANDARD, 
                CitationFormat.INLINE
            )
            
            assert len(citations) == 2
            assert citations[0] == "[Doc 1]"
            assert citations[1] == "[Doc 2]"
            assert mock_strategy.format_citation.call_count == 2
    
    def test_format_multiple_citations_with_strategy_names(self):
        """Test formatting multiple citations with specific strategy names."""
        formatter = CitationFormatter()
        
        source_refs = [
            SourceReference(
                source_url="https://example.com/doc1",
                resource_type=ResourceType.DSL,
                document_id="doc123",
                chunk_id="chunk456"
            ),
            SourceReference(
                source_url="https://example.com/doc2",
                resource_type=ResourceType.CONTEXTUAL,
                document_id="doc456",
                chunk_id="chunk789"
            )
        ]
        
        strategy_names = ["technical_citation", "web_citation"]
        
        with patch.object(formatter.strategy_registry, 'get_strategy') as mock_get_strategy:
            mock_strategy1 = Mock()
            mock_strategy1.format_citation.return_value = "[Technical Doc 1]"
            mock_strategy2 = Mock()
            mock_strategy2.format_citation.return_value = "[Web Doc 2]"
            mock_get_strategy.side_effect = [mock_strategy1, mock_strategy2]
            
            citations = formatter.format_multiple_citations(
                source_refs, 
                CitationVerbosity.STANDARD, 
                CitationFormat.INLINE,
                strategy_names
            )
            
            assert len(citations) == 2
            assert citations[0] == "[Technical Doc 1]"
            assert citations[1] == "[Web Doc 2]"
            mock_get_strategy.assert_any_call("technical_citation")
            mock_get_strategy.assert_any_call("web_citation")
    
    def test_format_multiple_citations_filters_empty(self):
        """Test that format_multiple_citations filters out empty citations."""
        formatter = CitationFormatter()
        
        source_refs = [
            SourceReference(
                source_url="https://example.com/doc1",
                resource_type=ResourceType.DSL,
                document_id="doc123",
                chunk_id="chunk456"
            ),
            SourceReference(
                source_url="https://example.com/doc2",
                resource_type=ResourceType.CONTEXTUAL,
                document_id="doc456",
                chunk_id="chunk789"
            )
        ]
        
        with patch.object(formatter.strategy_registry, 'get_strategy') as mock_get_strategy:
            mock_strategy = Mock()
            mock_strategy.format_citation.side_effect = ["[Doc 1]", ""]  # Second citation is empty
            mock_get_strategy.return_value = mock_strategy
            
            citations = formatter.format_multiple_citations(
                source_refs, 
                CitationVerbosity.STANDARD, 
                CitationFormat.INLINE
            )
            
            assert len(citations) == 1
            assert citations[0] == "[Doc 1]"


class TestCitationFormatterFormatMethods:
    """Test individual format methods of CitationFormatter."""
    
    def test_format_footnote(self):
        """Test _format_footnote method."""
        formatter = CitationFormatter()
        source_ref = SourceReference(
            source_url="https://example.com/doc",
            resource_type=ResourceType.DSL,
            document_id="doc123",
            chunk_id="chunk456"
        )
        
        footnote = formatter._format_footnote("[Example Doc, Introduction]", source_ref)
        
        assert footnote.startswith("[^")
        assert footnote.endswith("]")
        # Should contain a hash-based number
        assert len(footnote) > 3
    
    def test_format_academic_pdf_document(self):
        """Test _format_academic method with PDF document."""
        formatter = CitationFormatter()
        source_ref = SourceReference(
            source_url="https://example.com/document.pdf",
            resource_type=ResourceType.DSL,
            document_id="doc123",
            chunk_id="chunk456",
            page_number=5,
            section_title="Introduction",
            metadata={"processed_at": "2024-01-01"}
        )
        
        academic_citation = formatter._format_academic(source_ref, CitationVerbosity.STANDARD)
        
        assert academic_citation.startswith("(")
        assert academic_citation.endswith(")")
        assert "document" in academic_citation
        assert "5" in academic_citation
        # Note: section title is not included when page number is present (elif logic)
        assert "accessed 2024-01-01" in academic_citation
        assert "https://example.com/document.pdf" in academic_citation
    
    def test_format_academic_web_document(self):
        """Test _format_academic method with web document."""
        formatter = CitationFormatter()
        source_ref = SourceReference(
            source_url="https://example.com/blog/post",
            resource_type=ResourceType.CONTEXTUAL,
            document_id="doc123",
            chunk_id="chunk456",
            section_title="Introduction",
            metadata={"processed_at": "2024-01-01"}
        )
        
        academic_citation = formatter._format_academic(source_ref, CitationVerbosity.STANDARD)
        
        assert academic_citation.startswith("(")
        assert academic_citation.endswith(")")
        assert "example.com" in academic_citation
        assert "Introduction" in academic_citation
        assert "accessed 2024-01-01" in academic_citation
        assert "https://example.com/blog/post" in academic_citation
    
    def test_format_structured_basic(self):
        """Test _format_structured method with basic fields."""
        formatter = CitationFormatter()
        source_ref = SourceReference(
            source_url="https://example.com/doc",
            resource_type=ResourceType.DSL,
            document_id="doc123",
            chunk_id="chunk456"
        )
        
        structured_citation = formatter._format_structured(source_ref, CitationVerbosity.STANDARD)
        
        assert structured_citation.startswith("{")
        assert structured_citation.endswith("}")
        assert "source: example.com" in structured_citation
        assert "url: https://example.com/doc" in structured_citation
        assert "type: dsl" in structured_citation
    
    def test_format_structured_detailed(self):
        """Test _format_structured method with detailed fields."""
        formatter = CitationFormatter()
        source_ref = SourceReference(
            source_url="https://example.com/doc",
            resource_type=ResourceType.DSL,
            document_id="doc123",
            chunk_id="chunk456",
            page_number=5,
            section_title="Introduction",
            exact_quote="This is a test quote",
            confidence_score=0.9
        )
        
        structured_citation = formatter._format_structured(source_ref, CitationVerbosity.DETAILED)
        
        assert structured_citation.startswith("{")
        assert structured_citation.endswith("}")
        assert "source: example.com" in structured_citation
        assert "url: https://example.com/doc" in structured_citation
        assert "type: dsl" in structured_citation
        assert "page: 5" in structured_citation
        assert "section: Introduction" in structured_citation
        assert "quote: This is a test quote" in structured_citation
        assert "confidence: 0.9" in structured_citation
    
    def test_format_structured_forensic(self):
        """Test _format_structured method with forensic verbosity."""
        formatter = CitationFormatter()
        source_ref = SourceReference(
            source_url="https://example.com/doc",
            resource_type=ResourceType.DSL,
            document_id="doc123",
            chunk_id="chunk456",
            exact_quote="This is a test quote",
            confidence_score=0.8
        )
        
        structured_citation = formatter._format_structured(source_ref, CitationVerbosity.FORENSIC)
        
        assert "quote: This is a test quote" in structured_citation
        assert "confidence: 0.8" in structured_citation
    
    def test_extract_domain_http_url(self):
        """Test _extract_domain method with HTTP URL."""
        formatter = CitationFormatter()
        
        # Test with www
        domain = formatter._extract_domain("https://www.example.com/path/to/doc")
        assert domain == "example.com"
        
        # Test without www
        domain = formatter._extract_domain("https://example.com/path/to/doc")
        assert domain == "example.com"
        
        # Test with subdomain
        domain = formatter._extract_domain("https://docs.example.com/path/to/doc")
        assert domain == "docs.example.com"
    
    def test_extract_domain_http_url_with_port(self):
        """Test _extract_domain method with URL containing port."""
        formatter = CitationFormatter()
        
        domain = formatter._extract_domain("https://example.com:8080/path/to/doc")
        assert domain == "example.com:8080"
    
    def test_extract_domain_local_file(self):
        """Test _extract_domain method with local file path."""
        formatter = CitationFormatter()
        
        domain = formatter._extract_domain("/path/to/document.pdf")
        assert domain == "document.pdf"
        
        domain = formatter._extract_domain("document.pdf")
        assert domain == "document.pdf"


class TestCitationFormatterEdgeCases:
    """Test edge cases for CitationFormatter."""
    
    def test_format_citation_with_none_source_ref(self):
        """Test formatting citation with None source reference."""
        formatter = CitationFormatter()
        
        with pytest.raises(AttributeError):
            formatter.format_citation(
                None, 
                CitationVerbosity.STANDARD, 
                CitationFormat.INLINE
            )
    
    def test_format_citation_with_invalid_format(self):
        """Test formatting citation with invalid format."""
        formatter = CitationFormatter()
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
            
            # Test with invalid format (should fall back to base citation)
            citation = formatter.format_citation(
                source_ref, 
                CitationVerbosity.STANDARD, 
                "invalid_format"
            )
            
            assert citation == "[Example Doc]"
    
    def test_format_multiple_citations_empty_list(self):
        """Test formatting multiple citations with empty list."""
        formatter = CitationFormatter()
        
        citations = formatter.format_multiple_citations(
            [], 
            CitationVerbosity.STANDARD, 
            CitationFormat.INLINE
        )
        
        assert citations == []
    
    def test_format_multiple_citations_mismatched_strategy_names(self):
        """Test formatting multiple citations with mismatched strategy names."""
        formatter = CitationFormatter()
        
        source_refs = [
            SourceReference(
                source_url="https://example.com/doc1",
                resource_type=ResourceType.DSL,
                document_id="doc123",
                chunk_id="chunk456"
            ),
            SourceReference(
                source_url="https://example.com/doc2",
                resource_type=ResourceType.CONTEXTUAL,
                document_id="doc456",
                chunk_id="chunk789"
            )
        ]
        
        # Only one strategy name for two source refs
        strategy_names = ["technical_citation"]
        
        with patch.object(formatter.strategy_registry, 'get_strategy') as mock_get_strategy:
            mock_strategy = Mock()
            mock_strategy.format_citation.side_effect = ["[Doc 1]", "[Doc 2]"]
            mock_get_strategy.return_value = mock_strategy
            
            citations = formatter.format_multiple_citations(
                source_refs, 
                CitationVerbosity.STANDARD, 
                CitationFormat.INLINE,
                strategy_names
            )
            
            assert len(citations) == 2
            # Second citation should use default strategy
            mock_get_strategy.assert_any_call("default_citation")
    
    def test_format_academic_with_missing_metadata(self):
        """Test _format_academic with missing metadata."""
        formatter = CitationFormatter()
        source_ref = SourceReference(
            source_url="https://example.com/doc.pdf",
            resource_type=ResourceType.DSL,
            document_id="doc123",
            chunk_id="chunk456",
            page_number=5
            # No processed_at in metadata
        )
        
        academic_citation = formatter._format_academic(source_ref, CitationVerbosity.STANDARD)
        
        assert academic_citation.startswith("(")
        assert academic_citation.endswith(")")
        assert "doc" in academic_citation
        assert "5" in academic_citation
        assert "accessed" not in academic_citation  # Should not include access date
    
    def test_format_structured_with_confidence_1_0(self):
        """Test _format_structured with confidence score of 1.0."""
        formatter = CitationFormatter()
        source_ref = SourceReference(
            source_url="https://example.com/doc",
            resource_type=ResourceType.DSL,
            document_id="doc123",
            chunk_id="chunk456",
            confidence_score=1.0
        )
        
        structured_citation = formatter._format_structured(source_ref, CitationVerbosity.DETAILED)
        
        # Should not include confidence when it's 1.0
        assert "confidence:" not in structured_citation
    
    def test_extract_domain_with_invalid_url(self):
        """Test _extract_domain with invalid URL format."""
        formatter = CitationFormatter()
        
        # Test with malformed URL
        domain = formatter._extract_domain("not-a-url")
        assert domain == "not-a-url"
        
        # Test with empty string
        domain = formatter._extract_domain("")
        assert domain == ""
    
    def test_format_footnote_with_special_characters(self):
        """Test _format_footnote with special characters in citation."""
        formatter = CitationFormatter()
        source_ref = SourceReference(
            source_url="https://example.com/doc",
            resource_type=ResourceType.DSL,
            document_id="doc123",
            chunk_id="chunk456"
        )
        
        footnote = formatter._format_footnote("[Example Doc, Section (1.2.3)]", source_ref)
        
        assert footnote.startswith("[^")
        assert footnote.endswith("]")
        # Should still be valid footnote format


class TestCitationFormatterIntegration:
    """Integration tests for CitationFormatter."""
    
    def test_formatter_with_real_strategies(self):
        """Test formatter with real strategy implementations."""
        formatter = CitationFormatter()
        source_ref = SourceReference(
            source_url="https://kubernetes.io/docs/concepts/workloads/pods/",
            resource_type=ResourceType.DSL,
            document_id="doc123",
            chunk_id="chunk456",
            section_title="Pod Configuration"
        )
        
        # Test with technical strategy
        citation = formatter.format_citation(
            source_ref, 
            CitationVerbosity.STANDARD, 
            CitationFormat.INLINE,
            "technical_citation"
        )
        
        assert isinstance(citation, str)
        assert len(citation) > 0
    
    def test_formatter_workflow_with_multiple_formats(self):
        """Test complete workflow with multiple format styles."""
        formatter = CitationFormatter()
        source_ref = SourceReference(
            source_url="https://example.com/doc.pdf",
            resource_type=ResourceType.DSL,
            document_id="doc123",
            chunk_id="chunk456",
            page_number=5,
            section_title="Introduction",
            exact_quote="This is a test quote",
            confidence_score=0.9,
            metadata={"processed_at": "2024-01-01"}
        )
        
        formats = [
            CitationFormat.INLINE,
            CitationFormat.FOOTNOTE,
            CitationFormat.ACADEMIC,
            CitationFormat.STRUCTURED
        ]
        
        for format_style in formats:
            citation = formatter.format_citation(
                source_ref, 
                CitationVerbosity.DETAILED, 
                format_style
            )
            
            assert isinstance(citation, str)
            
            if format_style == CitationFormat.INLINE:
                # Should be basic citation format
                assert len(citation) > 0
            elif format_style == CitationFormat.FOOTNOTE:
                # Should be footnote format
                assert citation.startswith("[^")
                assert citation.endswith("]")
            elif format_style == CitationFormat.ACADEMIC:
                # Should be academic format
                assert citation.startswith("(")
                assert citation.endswith(")")
            elif format_style == CitationFormat.STRUCTURED:
                # Should be structured format
                assert citation.startswith("{")
                assert citation.endswith("}")
    
    def test_formatter_with_different_verbosity_levels(self):
        """Test formatter with different verbosity levels."""
        formatter = CitationFormatter()
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
    
    def test_formatter_with_different_resource_types(self):
        """Test formatter with different resource types."""
        formatter = CitationFormatter()
        
        resource_types = [
            ResourceType.DSL,
            ResourceType.CONTEXTUAL,
            ResourceType.GUIDELINES,
            ResourceType.DOMAIN_RULES
        ]
        
        for resource_type in resource_types:
            source_ref = SourceReference(
                source_url="https://example.com/doc",
                resource_type=resource_type,
                document_id="doc123",
                chunk_id="chunk456"
            )
            
            citation = formatter.format_citation(
                source_ref, 
                CitationVerbosity.STANDARD, 
                CitationFormat.STRUCTURED
            )
            
            assert isinstance(citation, str)
            assert "type: " + resource_type.value in citation
