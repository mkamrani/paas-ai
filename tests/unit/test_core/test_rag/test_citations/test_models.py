"""
Unit tests for citation data models.

Tests all components of the citation models including:
- SourceReference model
- CitationAwareResult model
- CitationMetadata model
- Model validation and serialization
- Edge cases and error handling
"""

import pytest
from pydantic import ValidationError
from typing import Dict, Any, List

from src.paas_ai.core.rag.citations.models import (
    SourceReference,
    CitationAwareResult,
    CitationMetadata
)
from src.paas_ai.core.config.schemas import ResourceType


class TestSourceReference:
    """Test the SourceReference model."""
    
    def test_create_source_reference_basic(self):
        """Test creating a basic source reference."""
        source_ref = SourceReference(
            source_url="https://example.com/doc",
            resource_type=ResourceType.DSL,
            document_id="doc123",
            chunk_id="chunk456"
        )
        
        assert source_ref.source_url == "https://example.com/doc"
        assert source_ref.resource_type == ResourceType.DSL
        assert source_ref.document_id == "doc123"
        assert source_ref.chunk_id == "chunk456"
        assert source_ref.confidence_score == 1.0
        assert source_ref.metadata == {}
    
    def test_create_source_reference_with_all_fields(self):
        """Test creating a source reference with all fields."""
        metadata = {"tag": "test", "priority": 1}
        
        source_ref = SourceReference(
            source_url="https://example.com/doc",
            resource_type=ResourceType.CONTEXTUAL,
            document_id="doc123",
            page_number=5,
            section_title="Introduction",
            subsection_title="Overview",
            paragraph_number=2,
            line_number=10,
            exact_quote="This is a test quote",
            start_char=100,
            end_char=120,
            surrounding_context="Some context around the quote",
            chunk_id="chunk456",
            confidence_score=0.95,
            metadata=metadata
        )
        
        assert source_ref.source_url == "https://example.com/doc"
        assert source_ref.resource_type == ResourceType.CONTEXTUAL
        assert source_ref.document_id == "doc123"
        assert source_ref.page_number == 5
        assert source_ref.section_title == "Introduction"
        assert source_ref.subsection_title == "Overview"
        assert source_ref.paragraph_number == 2
        assert source_ref.line_number == 10
        assert source_ref.exact_quote == "This is a test quote"
        assert source_ref.start_char == 100
        assert source_ref.end_char == 120
        assert source_ref.surrounding_context == "Some context around the quote"
        assert source_ref.chunk_id == "chunk456"
        assert source_ref.confidence_score == 0.95
        assert source_ref.metadata == metadata
    
    def test_create_source_reference_with_different_resource_types(self):
        """Test creating source references with different resource types."""
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
            assert source_ref.resource_type == resource_type
    
    def test_source_reference_default_values(self):
        """Test default values for optional fields."""
        source_ref = SourceReference(
            source_url="https://example.com/doc",
            resource_type=ResourceType.DSL,
            document_id="doc123",
            chunk_id="chunk456"
        )
        
        assert source_ref.page_number is None
        assert source_ref.section_title is None
        assert source_ref.subsection_title is None
        assert source_ref.paragraph_number is None
        assert source_ref.line_number is None
        assert source_ref.exact_quote is None
        assert source_ref.start_char is None
        assert source_ref.end_char is None
        assert source_ref.surrounding_context is None
        assert source_ref.confidence_score == 1.0
        assert source_ref.metadata == {}
    
    def test_source_reference_validation_required_fields(self):
        """Test validation of required fields."""
        # Missing source_url
        with pytest.raises(ValidationError):
            SourceReference(
                resource_type=ResourceType.DSL,
                document_id="doc123",
                chunk_id="chunk456"
            )
        
        # Missing resource_type
        with pytest.raises(ValidationError):
            SourceReference(
                source_url="https://example.com/doc",
                document_id="doc123",
                chunk_id="chunk456"
            )
        
        # Missing document_id
        with pytest.raises(ValidationError):
            SourceReference(
                source_url="https://example.com/doc",
                resource_type=ResourceType.DSL,
                chunk_id="chunk456"
            )
        
        # Missing chunk_id
        with pytest.raises(ValidationError):
            SourceReference(
                source_url="https://example.com/doc",
                resource_type=ResourceType.DSL,
                document_id="doc123"
            )
    
    def test_source_reference_validation_field_types(self):
        """Test validation of field types."""
        # Invalid resource_type
        with pytest.raises(ValidationError):
            SourceReference(
                source_url="https://example.com/doc",
                resource_type="invalid_type",
                document_id="doc123",
                chunk_id="chunk456"
            )
        
        # Invalid confidence_score
        with pytest.raises(ValidationError):
            SourceReference(
                source_url="https://example.com/doc",
                resource_type=ResourceType.DSL,
                document_id="doc123",
                chunk_id="chunk456",
                confidence_score="invalid"
            )
    
    def test_source_reference_serialization(self):
        """Test serialization to dict."""
        source_ref = SourceReference(
            source_url="https://example.com/doc",
            resource_type=ResourceType.DSL,
            document_id="doc123",
            chunk_id="chunk456",
            page_number=5,
            section_title="Test Section",
            confidence_score=0.9,
            metadata={"tag": "test"}
        )
        
        data = source_ref.model_dump()
        
        assert data["source_url"] == "https://example.com/doc"
        assert data["resource_type"] == "dsl"
        assert data["document_id"] == "doc123"
        assert data["chunk_id"] == "chunk456"
        assert data["page_number"] == 5
        assert data["section_title"] == "Test Section"
        assert data["confidence_score"] == 0.9
        assert data["metadata"] == {"tag": "test"}
    
    def test_source_reference_deserialization(self):
        """Test deserialization from dict."""
        data = {
            "source_url": "https://example.com/doc",
            "resource_type": "dsl",
            "document_id": "doc123",
            "chunk_id": "chunk456",
            "page_number": 5,
            "section_title": "Test Section",
            "confidence_score": 0.9,
            "metadata": {"tag": "test"}
        }
        
        source_ref = SourceReference(**data)
        
        assert source_ref.source_url == "https://example.com/doc"
        assert source_ref.resource_type == ResourceType.DSL
        assert source_ref.document_id == "doc123"
        assert source_ref.chunk_id == "chunk456"
        assert source_ref.page_number == 5
        assert source_ref.section_title == "Test Section"
        assert source_ref.confidence_score == 0.9
        assert source_ref.metadata == {"tag": "test"}


class TestCitationAwareResult:
    """Test the CitationAwareResult model."""
    
    def test_create_citation_aware_result_basic(self):
        """Test creating a basic citation-aware result."""
        source_ref = SourceReference(
            source_url="https://example.com/doc",
            resource_type=ResourceType.DSL,
            document_id="doc123",
            chunk_id="chunk456"
        )
        
        result = CitationAwareResult(
            content="This is the content",
            score=0.95,
            source_reference=source_ref,
            formatted_citation="[Example Doc]"
        )
        
        assert result.content == "This is the content"
        assert result.score == 0.95
        assert result.source_reference == source_ref
        assert result.formatted_citation == "[Example Doc]"
        assert result.citation_link is None
        assert result.supporting_quotes == []
        assert result.confidence_indicators == {}
    
    def test_create_citation_aware_result_with_all_fields(self):
        """Test creating a citation-aware result with all fields."""
        source_ref = SourceReference(
            source_url="https://example.com/doc",
            resource_type=ResourceType.DSL,
            document_id="doc123",
            chunk_id="chunk456"
        )
        
        supporting_quotes = ["Quote 1", "Quote 2"]
        confidence_indicators = {"relevance": 0.9, "accuracy": 0.8}
        
        result = CitationAwareResult(
            content="This is the content",
            score=0.95,
            source_reference=source_ref,
            formatted_citation="[Example Doc, Section 1]",
            citation_link="https://example.com/doc#section1",
            supporting_quotes=supporting_quotes,
            confidence_indicators=confidence_indicators
        )
        
        assert result.content == "This is the content"
        assert result.score == 0.95
        assert result.source_reference == source_ref
        assert result.formatted_citation == "[Example Doc, Section 1]"
        assert result.citation_link == "https://example.com/doc#section1"
        assert result.supporting_quotes == supporting_quotes
        assert result.confidence_indicators == confidence_indicators
    
    def test_citation_aware_result_validation_required_fields(self):
        """Test validation of required fields."""
        source_ref = SourceReference(
            source_url="https://example.com/doc",
            resource_type=ResourceType.DSL,
            document_id="doc123",
            chunk_id="chunk456"
        )
        
        # Missing content
        with pytest.raises(ValidationError):
            CitationAwareResult(
                score=0.95,
                source_reference=source_ref,
                formatted_citation="[Example Doc]"
            )
        
        # Missing score
        with pytest.raises(ValidationError):
            CitationAwareResult(
                content="This is the content",
                source_reference=source_ref,
                formatted_citation="[Example Doc]"
            )
        
        # Missing source_reference
        with pytest.raises(ValidationError):
            CitationAwareResult(
                content="This is the content",
                score=0.95,
                formatted_citation="[Example Doc]"
            )
        
        # Missing formatted_citation
        with pytest.raises(ValidationError):
            CitationAwareResult(
                content="This is the content",
                score=0.95,
                source_reference=source_ref
            )
    
    def test_citation_aware_result_validation_field_types(self):
        """Test validation of field types."""
        source_ref = SourceReference(
            source_url="https://example.com/doc",
            resource_type=ResourceType.DSL,
            document_id="doc123",
            chunk_id="chunk456"
        )
        
        # Invalid score type
        with pytest.raises(ValidationError):
            CitationAwareResult(
                content="This is the content",
                score="invalid",
                source_reference=source_ref,
                formatted_citation="[Example Doc]"
            )
        
        # Invalid supporting_quotes type
        with pytest.raises(ValidationError):
            CitationAwareResult(
                content="This is the content",
                score=0.95,
                source_reference=source_ref,
                formatted_citation="[Example Doc]",
                supporting_quotes="not a list"
            )
    
    def test_citation_aware_result_serialization(self):
        """Test serialization to dict."""
        source_ref = SourceReference(
            source_url="https://example.com/doc",
            resource_type=ResourceType.DSL,
            document_id="doc123",
            chunk_id="chunk456"
        )
        
        result = CitationAwareResult(
            content="This is the content",
            score=0.95,
            source_reference=source_ref,
            formatted_citation="[Example Doc]",
            citation_link="https://example.com/doc#section1",
            supporting_quotes=["Quote 1"],
            confidence_indicators={"relevance": 0.9}
        )
        
        data = result.model_dump()
        
        assert data["content"] == "This is the content"
        assert data["score"] == 0.95
        assert data["formatted_citation"] == "[Example Doc]"
        assert data["citation_link"] == "https://example.com/doc#section1"
        assert data["supporting_quotes"] == ["Quote 1"]
        assert data["confidence_indicators"] == {"relevance": 0.9}
        assert "source_reference" in data


class TestCitationMetadata:
    """Test the CitationMetadata model."""
    
    def test_create_citation_metadata_basic(self):
        """Test creating basic citation metadata."""
        metadata = CitationMetadata()
        
        assert metadata.document_structure == {}
        assert metadata.headings == []
        assert metadata.page_breaks == []
        assert metadata.section_markers == []
        assert metadata.content_type_markers == {}
        assert metadata.anchor_points == []
    
    def test_create_citation_metadata_with_all_fields(self):
        """Test creating citation metadata with all fields."""
        document_structure = {"sections": 3, "pages": 5}
        headings = [{"text": "Introduction", "level": 1}, {"text": "Overview", "level": 2}]
        page_breaks = [1, 2, 3, 4, 5]
        section_markers = [{"type": "heading", "position": 0}, {"type": "paragraph", "position": 100}]
        content_type_markers = {"has_code": True, "has_tables": False}
        anchor_points = [{"id": "intro", "position": 0}, {"id": "overview", "position": 200}]
        
        metadata = CitationMetadata(
            document_structure=document_structure,
            headings=headings,
            page_breaks=page_breaks,
            section_markers=section_markers,
            content_type_markers=content_type_markers,
            anchor_points=anchor_points
        )
        
        assert metadata.document_structure == document_structure
        assert metadata.headings == headings
        assert metadata.page_breaks == page_breaks
        assert metadata.section_markers == section_markers
        assert metadata.content_type_markers == content_type_markers
        assert metadata.anchor_points == anchor_points
    
    def test_citation_metadata_validation_field_types(self):
        """Test validation of field types."""
        # Invalid headings type
        with pytest.raises(ValidationError):
            CitationMetadata(headings="not a list")
        
        # Invalid page_breaks type
        with pytest.raises(ValidationError):
            CitationMetadata(page_breaks="not a list")
        
        # Invalid content_type_markers type
        with pytest.raises(ValidationError):
            CitationMetadata(content_type_markers="not a dict")
    
    def test_citation_metadata_serialization(self):
        """Test serialization to dict."""
        metadata = CitationMetadata(
            document_structure={"sections": 3},
            headings=[{"text": "Introduction", "level": 1}],
            page_breaks=[1, 2, 3],
            section_markers=[{"type": "heading", "position": 0}],
            content_type_markers={"has_code": True},
            anchor_points=[{"id": "intro", "position": 0}]
        )
        
        data = metadata.model_dump()
        
        assert data["document_structure"] == {"sections": 3}
        assert data["headings"] == [{"text": "Introduction", "level": 1}]
        assert data["page_breaks"] == [1, 2, 3]
        assert data["section_markers"] == [{"type": "heading", "position": 0}]
        assert data["content_type_markers"] == {"has_code": True}
        assert data["anchor_points"] == [{"id": "intro", "position": 0}]


class TestCitationModelsEdgeCases:
    """Test edge cases for citation models."""
    
    def test_source_reference_with_empty_strings(self):
        """Test source reference with empty string values."""
        source_ref = SourceReference(
            source_url="",
            resource_type=ResourceType.DSL,
            document_id="",
            chunk_id="",
            section_title="",
            exact_quote=""
        )
        
        assert source_ref.source_url == ""
        assert source_ref.document_id == ""
        assert source_ref.chunk_id == ""
        assert source_ref.section_title == ""
        assert source_ref.exact_quote == ""
    
    def test_source_reference_with_unicode_content(self):
        """Test source reference with unicode content."""
        source_ref = SourceReference(
            source_url="https://example.com/文档",
            resource_type=ResourceType.DSL,
            document_id="文档123",
            chunk_id="chunk456",
            section_title="介绍",
            exact_quote="这是一个测试引用"
        )
        
        assert source_ref.source_url == "https://example.com/文档"
        assert source_ref.document_id == "文档123"
        assert source_ref.section_title == "介绍"
        assert source_ref.exact_quote == "这是一个测试引用"
    
    def test_source_reference_with_very_long_content(self):
        """Test source reference with very long content."""
        long_url = "https://example.com/" + "a" * 1000
        long_quote = "This is a very long quote. " * 100
        
        source_ref = SourceReference(
            source_url=long_url,
            resource_type=ResourceType.DSL,
            document_id="doc123",
            chunk_id="chunk456",
            exact_quote=long_quote
        )
        
        assert source_ref.source_url == long_url
        assert source_ref.exact_quote == long_quote
    
    def test_citation_aware_result_with_edge_case_scores(self):
        """Test citation aware result with edge case scores."""
        source_ref = SourceReference(
            source_url="https://example.com/doc",
            resource_type=ResourceType.DSL,
            document_id="doc123",
            chunk_id="chunk456"
        )
        
        # Test with score 0
        result_zero = CitationAwareResult(
            content="Content",
            score=0.0,
            source_reference=source_ref,
            formatted_citation="[Example Doc]"
        )
        assert result_zero.score == 0.0
        
        # Test with score 1
        result_one = CitationAwareResult(
            content="Content",
            score=1.0,
            source_reference=source_ref,
            formatted_citation="[Example Doc]"
        )
        assert result_one.score == 1.0
    
    def test_citation_metadata_with_complex_structures(self):
        """Test citation metadata with complex nested structures."""
        complex_structure = {
            "sections": [
                {"title": "Section 1", "subsections": [{"title": "Subsection 1.1"}]},
                {"title": "Section 2", "subsections": []}
            ],
            "metadata": {"author": "Test Author", "version": "1.0"}
        }
        
        complex_headings = [
            {"text": "Introduction", "level": 1, "position": 0, "style": "h1"},
            {"text": "Overview", "level": 2, "position": 100, "style": "h2"},
            {"text": "Details", "level": 3, "position": 200, "style": "h3"}
        ]
        
        metadata = CitationMetadata(
            document_structure=complex_structure,
            headings=complex_headings
        )
        
        assert metadata.document_structure == complex_structure
        assert metadata.headings == complex_headings
    
    def test_models_with_none_values(self):
        """Test models with None values where allowed."""
        source_ref = SourceReference(
            source_url="https://example.com/doc",
            resource_type=ResourceType.DSL,
            document_id="doc123",
            chunk_id="chunk456",
            page_number=None,
            section_title=None,
            exact_quote=None
        )
        
        assert source_ref.page_number is None
        assert source_ref.section_title is None
        assert source_ref.exact_quote is None
        
        result = CitationAwareResult(
            content="Content",
            score=0.95,
            source_reference=source_ref,
            formatted_citation="[Example Doc]",
            citation_link=None
        )
        
        assert result.citation_link is None


class TestCitationModelsIntegration:
    """Integration tests for citation models."""
    
    def test_models_work_together(self):
        """Test that all models work together properly."""
        # Create citation metadata
        citation_metadata = CitationMetadata(
            document_structure={"sections": 2},
            headings=[{"text": "Introduction", "level": 1}],
            content_type_markers={"has_code": True}
        )
        
        # Create source reference
        source_ref = SourceReference(
            source_url="https://example.com/doc",
            resource_type=ResourceType.DSL,
            document_id="doc123",
            chunk_id="chunk456",
            page_number=1,
            section_title="Introduction",
            exact_quote="This is a test quote",
            confidence_score=0.95,
            metadata={"citation_metadata": citation_metadata.model_dump()}
        )
        
        # Create citation-aware result
        result = CitationAwareResult(
            content="This is the search result content",
            score=0.95,
            source_reference=source_ref,
            formatted_citation="[Example Doc, Introduction]",
            citation_link="https://example.com/doc#introduction",
            supporting_quotes=["This is a test quote"],
            confidence_indicators={"relevance": 0.9, "accuracy": 0.95}
        )
        
        # Verify all models work together
        assert result.source_reference.section_title == "Introduction"
        assert result.source_reference.metadata["citation_metadata"]["headings"][0]["text"] == "Introduction"
        assert result.supporting_quotes[0] == result.source_reference.exact_quote
    
    def test_serialization_round_trip(self):
        """Test that models can be serialized and deserialized correctly."""
        # Create original model
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
        assert deserialized_source_ref.model_dump() == original_source_ref.model_dump()
    
    def test_models_with_different_resource_types(self):
        """Test models with different resource types."""
        resource_types = [ResourceType.DSL, ResourceType.CONTEXTUAL, ResourceType.GUIDELINES, ResourceType.DOMAIN_RULES]
        
        for resource_type in resource_types:
            source_ref = SourceReference(
                source_url="https://example.com/doc",
                resource_type=resource_type,
                document_id="doc123",
                chunk_id="chunk456"
            )
            
            result = CitationAwareResult(
                content="Content",
                score=0.95,
                source_reference=source_ref,
                formatted_citation=f"[Example Doc - {resource_type.value}]"
            )
            
            assert result.source_reference.resource_type == resource_type
            assert resource_type.value in result.formatted_citation
