"""
Unit tests for processing stages.

Tests all processing stage implementations including:
- LoadStage
- ValidateStage
- SplitStage
- EnrichStage
- FilterStage
- Error handling and edge cases
- Integration with dependencies
"""

import pytest
import time
from unittest.mock import Mock, patch, MagicMock, AsyncMock

from src.paas_ai.core.rag.processing.stages import (
    LoadStage,
    ValidateStage,
    SplitStage,
    EnrichStage,
    FilterStage
)
from src.paas_ai.core.rag.processing.base import ProcessingContext
from src.paas_ai.core.rag.config import (
    ResourceConfig,
    LoaderConfig,
    LoaderType,
    SplitterConfig,
    SplitterType,
    ResourceType
)
from src.paas_ai.core.config.schemas import (
    ContentValidatorConfig,
    ContentValidatorType
)
from langchain_core.documents import Document


def create_test_resource_config(url: str = "http://test.com") -> ResourceConfig:
    """Create a valid ResourceConfig for testing."""
    return ResourceConfig(
        url=url,
        resource_type=ResourceType.DSL,
        loader=LoaderConfig(type=LoaderType.WEB),
        splitter=SplitterConfig(type=SplitterType.RECURSIVE_CHARACTER)
    )


class TestLoadStage:
    """Test the LoadStage class."""
    
    def test_init(self):
        """Test LoadStage initialization."""
        stage = LoadStage()
        assert stage.name == "load"
    
    def test_init_with_custom_name(self):
        """Test LoadStage initialization with custom name."""
        stage = LoadStage(name="custom_load")
        assert stage.name == "custom_load"
    
    @pytest.mark.asyncio
    async def test_process_success(self):
        """Test successful document loading."""
        # Mock documents
        mock_docs = [
            Document(page_content="Test content 1", metadata={"source": "test1"}),
            Document(page_content="Test content 2", metadata={"source": "test2"})
        ]
        
        # Mock loader
        mock_loader = Mock()
        mock_loader.load.return_value = mock_docs
        
        # Mock factory
        with patch('src.paas_ai.core.rag.processing.stages.load.DocumentLoaderFactory') as mock_factory:
            mock_factory.create_loader.return_value = mock_loader
            
            stage = LoadStage()
            resource = create_test_resource_config()
            context = ProcessingContext(resource=resource)
            
            result = await stage.process(context)
            
            # Verify factory was called correctly
            mock_factory.create_loader.assert_called_once_with(
                resource.loader, resource.url
            )
            
            # Verify loader was called
            mock_loader.load.assert_called_once()
            
            # Verify context was updated
            assert result.documents == mock_docs
            assert result.metadata['source_loader'] == LoaderType.WEB
            assert result.metadata['document_count'] == 2
    
    @pytest.mark.asyncio
    async def test_process_no_documents(self):
        """Test loading when no documents are returned."""
        # Mock empty loader
        mock_loader = Mock()
        mock_loader.load.return_value = []
        
        with patch('src.paas_ai.core.rag.processing.stages.load.DocumentLoaderFactory') as mock_factory:
            mock_factory.create_loader.return_value = mock_loader
            
            stage = LoadStage()
            resource = create_test_resource_config()
            context = ProcessingContext(resource=resource)
            
            with pytest.raises(ValueError, match="No documents loaded from"):
                await stage.process(context)
    
    @pytest.mark.asyncio
    async def test_process_loader_exception(self):
        """Test loading when loader raises exception."""
        # Mock loader that raises exception
        mock_loader = Mock()
        mock_loader.load.side_effect = Exception("Loader error")
        
        with patch('src.paas_ai.core.rag.processing.stages.load.DocumentLoaderFactory') as mock_factory:
            mock_factory.create_loader.return_value = mock_loader
            
            stage = LoadStage()
            resource = create_test_resource_config()
            context = ProcessingContext(resource=resource)
            
            with pytest.raises(Exception, match="Loader error"):
                await stage.process(context)
    
    @pytest.mark.asyncio
    async def test_process_factory_exception(self):
        """Test loading when factory raises exception."""
        with patch('src.paas_ai.core.rag.processing.stages.load.DocumentLoaderFactory') as mock_factory:
            mock_factory.create_loader.side_effect = Exception("Factory error")
            
            stage = LoadStage()
            resource = create_test_resource_config()
            context = ProcessingContext(resource=resource)
            
            with pytest.raises(Exception, match="Factory error"):
                await stage.process(context)


class TestValidateStage:
    """Test the ValidateStage class."""
    
    def test_init_default(self):
        """Test ValidateStage initialization with defaults."""
        stage = ValidateStage()
        assert stage.name == "validate"
        assert stage.validator_config is not None
        assert stage.validator_config.type == ContentValidatorType.CONTENT
        assert stage.content_validator is not None
    
    def test_init_with_custom_name(self):
        """Test ValidateStage initialization with custom name."""
        stage = ValidateStage(name="custom_validate")
        assert stage.name == "custom_validate"
    
    def test_init_with_custom_config(self):
        """Test ValidateStage initialization with custom config."""
        custom_config = ContentValidatorConfig(
            type=ContentValidatorType.CONTENT,
            min_content_length=50,
            max_content_length=5000,
            skip_empty=False
        )
        
        stage = ValidateStage(validator_config=custom_config)
        assert stage.validator_config == custom_config
    
    @pytest.mark.asyncio
    async def test_process_success(self):
        """Test successful document validation."""
        # Mock validator
        mock_validator = Mock()
        mock_validator.validate_documents.return_value = [
            Document(page_content="Valid content", metadata={})
        ]
        
        with patch('src.paas_ai.core.rag.processing.stages.validate.ContentValidatorFactory') as mock_factory:
            mock_factory.create_content_validator.return_value = mock_validator
            
            stage = ValidateStage()
            resource = create_test_resource_config()
            documents = [
                Document(page_content="Valid content", metadata={}),
                Document(page_content="", metadata={})  # Empty document
            ]
            context = ProcessingContext(resource=resource, documents=documents)
            
            result = await stage.process(context)
            
            # Verify validator was called
            mock_validator.validate_documents.assert_called_once()
            
            # Verify context was updated
            assert len(result.documents) == 1
            assert result.metadata['validated_count'] == 1
            assert result.metadata['rejected_count'] == 1
            assert 'validation_rules' in result.metadata
    
    @pytest.mark.asyncio
    async def test_process_no_documents(self):
        """Test validation with no documents."""
        stage = ValidateStage()
        resource = create_test_resource_config()
        context = ProcessingContext(resource=resource)
        
        result = await stage.process(context)
        
        assert result.documents == []
        assert result.metadata['validated_count'] == 0
        assert result.metadata['rejected_count'] == 0
    
    @pytest.mark.asyncio
    async def test_process_all_documents_rejected(self):
        """Test validation when all documents are rejected."""
        # Mock validator that rejects all documents
        mock_validator = Mock()
        mock_validator.validate_documents.return_value = []
        
        with patch('src.paas_ai.core.rag.processing.stages.validate.ContentValidatorFactory') as mock_factory:
            mock_factory.create_content_validator.return_value = mock_validator
            
            stage = ValidateStage()
            resource = create_test_resource_config()
            documents = [
                Document(page_content="Invalid content", metadata={}),
                Document(page_content="Another invalid", metadata={})
            ]
            context = ProcessingContext(resource=resource, documents=documents)
            
            result = await stage.process(context)
            
            assert len(result.documents) == 0
            assert result.metadata['validated_count'] == 0
            assert result.metadata['rejected_count'] == 2


class TestSplitStage:
    """Test the SplitStage class."""
    
    def test_init(self):
        """Test SplitStage initialization."""
        stage = SplitStage()
        assert stage.name == "split"
    
    def test_init_with_custom_name(self):
        """Test SplitStage initialization with custom name."""
        stage = SplitStage(name="custom_split")
        assert stage.name == "custom_split"
    
    @pytest.mark.asyncio
    async def test_process_with_split_documents_method(self):
        """Test splitting with splitter that has split_documents method."""
        # Mock splitter with split_documents method
        mock_splitter = Mock()
        mock_splitter.split_documents.return_value = [
            Document(page_content="Chunk 1", metadata={"chunk": 1}),
            Document(page_content="Chunk 2", metadata={"chunk": 2})
        ]
        
        with patch('src.paas_ai.core.rag.processing.stages.split.TextSplitterFactory') as mock_factory:
            mock_factory.create_splitter.return_value = mock_splitter
            
            stage = SplitStage()
            resource = create_test_resource_config()
            documents = [Document(page_content="Original content", metadata={"source": "test"})]
            context = ProcessingContext(resource=resource, documents=documents)
            
            result = await stage.process(context)
            
            # Verify factory was called
            mock_factory.create_splitter.assert_called_once_with(resource.splitter)
            
            # Verify splitter was called
            mock_splitter.split_documents.assert_called_once_with(documents)
            
            # Verify context was updated
            assert len(result.documents) == 2
            assert result.metadata['splitter_type'] == SplitterType.RECURSIVE_CHARACTER
            assert result.metadata['chunk_count'] == 2
    
    @pytest.mark.asyncio
    async def test_process_with_split_text_method_returns_documents(self):
        """Test splitting with splitter that has split_text method returning Documents."""
        # Mock splitter with split_text method returning Document objects
        mock_splitter = Mock()
        mock_splitter.split_text.return_value = [
            Document(page_content="Chunk 1", metadata={"chunk": 1}),
            Document(page_content="Chunk 2", metadata={"chunk": 2})
        ]
        
        with patch('src.paas_ai.core.rag.processing.stages.split.TextSplitterFactory') as mock_factory:
            mock_factory.create_splitter.return_value = mock_splitter
            
            stage = SplitStage()
            resource = create_test_resource_config()
            original_doc = Document(page_content="Original content", metadata={"source": "test"})
            documents = [original_doc]
            context = ProcessingContext(resource=resource, documents=documents)
            
            result = await stage.process(context)
            
            # Verify splitter was called for each document
            mock_splitter.split_text.assert_called_once_with("Original content")
            
            # Verify context was updated
            assert len(result.documents) == 2
            assert result.metadata['chunk_count'] == 2
    
    @pytest.mark.asyncio
    async def test_process_with_split_text_method_returns_strings(self):
        """Test splitting with splitter that has split_text method returning strings."""
        # Mock splitter with split_text method returning strings
        mock_splitter = Mock()
        mock_splitter.split_text.return_value = ["Chunk 1", "Chunk 2"]
        
        with patch('src.paas_ai.core.rag.processing.stages.split.TextSplitterFactory') as mock_factory:
            mock_factory.create_splitter.return_value = mock_splitter
            
            stage = SplitStage()
            resource = create_test_resource_config()
            original_doc = Document(page_content="Original content", metadata={"source": "test"})
            documents = [original_doc]
            context = ProcessingContext(resource=resource, documents=documents)
            
            result = await stage.process(context)
            
            # Verify splitter was called
            mock_splitter.split_text.assert_called_once_with("Original content")
            
            # Verify context was updated with new Document objects
            assert len(result.documents) == 2
            assert result.documents[0].page_content == "Chunk 1"
            assert result.documents[1].page_content == "Chunk 2"
            assert result.documents[0].metadata["source"] == "test"
            assert result.documents[1].metadata["source"] == "test"
    
    @pytest.mark.asyncio
    async def test_process_no_documents(self):
        """Test splitting with no documents."""
        stage = SplitStage()
        resource = create_test_resource_config()
        context = ProcessingContext(resource=resource)
        
        result = await stage.process(context)
        
        assert result.documents == []
        assert result.metadata['chunk_count'] == 0
    
    @pytest.mark.asyncio
    async def test_process_splitter_exception(self):
        """Test splitting when splitter raises exception."""
        # Mock splitter that raises exception
        mock_splitter = Mock()
        mock_splitter.split_documents.side_effect = Exception("Splitter error")
        
        with patch('src.paas_ai.core.rag.processing.stages.split.TextSplitterFactory') as mock_factory:
            mock_factory.create_splitter.return_value = mock_splitter
            
            stage = SplitStage()
            resource = create_test_resource_config()
            documents = [Document(page_content="Test content", metadata={})]
            context = ProcessingContext(resource=resource, documents=documents)
            
            with pytest.raises(Exception, match="Splitter error"):
                await stage.process(context)


class TestEnrichStage:
    """Test the EnrichStage class."""
    
    def test_init_default(self):
        """Test EnrichStage initialization with defaults."""
        stage = EnrichStage()
        assert stage.name == "enrich"
        assert stage.custom_enrichers == []
    
    def test_init_with_custom_name(self):
        """Test EnrichStage initialization with custom name."""
        stage = EnrichStage(name="custom_enrich")
        assert stage.name == "custom_enrich"
    
    def test_init_with_custom_enrichers(self):
        """Test EnrichStage initialization with custom enrichers."""
        custom_enrichers = [Mock(), Mock()]
        stage = EnrichStage(custom_enrichers=custom_enrichers)
        assert stage.custom_enrichers == custom_enrichers
    
    @pytest.mark.asyncio
    async def test_process_success(self):
        """Test successful document enrichment."""
        stage = EnrichStage()
        resource = ResourceConfig(
            url="http://test.com",
            resource_type=ResourceType.DSL,
            loader=LoaderConfig(type=LoaderType.WEB),
            splitter=SplitterConfig(type=SplitterType.RECURSIVE_CHARACTER),
            priority=1,
            tags=["test"],
            metadata={"custom": "value"}
        )
        documents = [Document(page_content="Test content", metadata={})]
        context = ProcessingContext(resource=resource, documents=documents)
        
        result = await stage.process(context)
        
        # Verify documents were enriched
        doc = result.documents[0]
        assert doc.metadata['source_url'] == "http://test.com"
        assert doc.metadata['resource_type'] == ResourceType.DSL
        assert doc.metadata['priority'] == 1
        assert doc.metadata['tags'] == ["test"]
        assert doc.metadata['custom'] == "value"
        assert doc.metadata['content_length'] == len("Test content")
        assert doc.metadata['word_count'] == 2
        assert doc.metadata['char_count'] == 12
        assert 'processed_at' in doc.metadata
        assert 'pipeline_id' in doc.metadata
        
        # Verify context metadata
        assert 'enriched_at' in result.metadata
        assert result.metadata['enricher_count'] == 0
    
    @pytest.mark.asyncio
    async def test_process_with_custom_enrichers(self):
        """Test enrichment with custom enrichers."""
        # Mock custom enricher
        mock_enricher = AsyncMock()
        mock_enricher.return_value = [
            Document(page_content="Enriched content", metadata={"enriched": True})
        ]
        
        stage = EnrichStage(custom_enrichers=[mock_enricher])
        resource = create_test_resource_config()
        documents = [Document(page_content="Test content", metadata={})]
        context = ProcessingContext(resource=resource, documents=documents)
        
        result = await stage.process(context)
        
        # Verify custom enricher was called
        mock_enricher.assert_called_once()
        
        # Verify context metadata
        assert result.metadata['enricher_count'] == 1
    
    @pytest.mark.asyncio
    async def test_process_with_citation_enricher(self):
        """Test enrichment with citation enricher."""
        # Mock citation enricher
        mock_citation_enricher = AsyncMock()
        mock_citation_enricher.return_value = [
            Document(page_content="Cited content", metadata={"cited": True})
        ]
        
        stage = EnrichStage()
        resource = create_test_resource_config()
        documents = [Document(page_content="Test content", metadata={})]
        context = ProcessingContext(resource=resource, documents=documents)
        context.citation_enricher = mock_citation_enricher
        
        result = await stage.process(context)
        
        # Verify citation enricher was called
        mock_citation_enricher.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_process_no_documents(self):
        """Test enrichment with no documents."""
        stage = EnrichStage()
        resource = create_test_resource_config()
        context = ProcessingContext(resource=resource)
        
        result = await stage.process(context)
        
        assert result.documents == []
        assert 'enriched_at' in result.metadata
    
    @pytest.mark.asyncio
    async def test_process_multiple_documents(self):
        """Test enrichment with multiple documents."""
        stage = EnrichStage()
        resource = create_test_resource_config()
        documents = [
            Document(page_content="Content 1", metadata={}),
            Document(page_content="Content 2", metadata={})
        ]
        context = ProcessingContext(resource=resource, documents=documents)
        
        result = await stage.process(context)
        
        assert len(result.documents) == 2
        for doc in result.documents:
            assert doc.metadata['source_url'] == "http://test.com"
            assert 'content_length' in doc.metadata


class TestFilterStage:
    """Test the FilterStage class."""
    
    def test_init_default(self):
        """Test FilterStage initialization with defaults."""
        stage = FilterStage()
        assert stage.name == "filter"
        assert stage.content_filters == []
        assert stage.metadata_filters == []
    
    def test_init_with_custom_name(self):
        """Test FilterStage initialization with custom name."""
        stage = FilterStage(name="custom_filter")
        assert stage.name == "custom_filter"
    
    def test_init_with_custom_filters(self):
        """Test FilterStage initialization with custom filters."""
        content_filters = [lambda x: len(x) > 10]
        metadata_filters = [lambda x: "valid" in x]
        
        stage = FilterStage(
            content_filters=content_filters,
            metadata_filters=metadata_filters
        )
        assert stage.content_filters == content_filters
        assert stage.metadata_filters == metadata_filters
    
    @pytest.mark.asyncio
    async def test_process_success_all_passed(self):
        """Test filtering when all documents pass filters."""
        # Create filters that all documents pass
        content_filters = [lambda x: len(x) > 5]
        metadata_filters = [lambda x: "valid" in x]
        
        stage = FilterStage(
            content_filters=content_filters,
            metadata_filters=metadata_filters
        )
        
        documents = [
            Document(page_content="Valid content", metadata={"valid": True}),
            Document(page_content="Another valid", metadata={"valid": True})
        ]
        context = ProcessingContext(resource=create_test_resource_config(), documents=documents)
        
        result = await stage.process(context)
        
        assert len(result.documents) == 2
        assert result.metadata['filtered_count'] == 2
        assert result.metadata['removed_count'] == 0
        assert result.metadata['filter_rules_count'] == 2
    
    @pytest.mark.asyncio
    async def test_process_some_filtered(self):
        """Test filtering when some documents are filtered out."""
        # Create filters that some documents fail
        content_filters = [lambda x: len(x) > 10]
        metadata_filters = [lambda x: "valid" in x]
        
        stage = FilterStage(
            content_filters=content_filters,
            metadata_filters=metadata_filters
        )
        
        documents = [
            Document(page_content="Valid content", metadata={"valid": True}),  # Passes
            Document(page_content="Short", metadata={"valid": True}),  # Fails content filter
            Document(page_content="Valid content", metadata={"invalid": True})  # Fails metadata filter
        ]
        context = ProcessingContext(resource=create_test_resource_config(), documents=documents)
        
        result = await stage.process(context)
        
        assert len(result.documents) == 1
        assert result.documents[0].page_content == "Valid content"
        assert result.metadata['filtered_count'] == 1
        assert result.metadata['removed_count'] == 2
    
    @pytest.mark.asyncio
    async def test_process_all_filtered(self):
        """Test filtering when all documents are filtered out."""
        # Create filters that all documents fail
        content_filters = [lambda x: len(x) > 100]
        
        stage = FilterStage(content_filters=content_filters)
        
        documents = [
            Document(page_content="Short", metadata={}),
            Document(page_content="Also short", metadata={})
        ]
        context = ProcessingContext(resource=create_test_resource_config(), documents=documents)
        
        result = await stage.process(context)
        
        assert len(result.documents) == 0
        assert result.metadata['filtered_count'] == 0
        assert result.metadata['removed_count'] == 2
    
    @pytest.mark.asyncio
    async def test_process_no_documents(self):
        """Test filtering with no documents."""
        stage = FilterStage()
        context = ProcessingContext(resource=create_test_resource_config())
        
        result = await stage.process(context)
        
        assert result.documents == []
        assert result.metadata['filtered_count'] == 0
        assert result.metadata['removed_count'] == 0
    
    @pytest.mark.asyncio
    async def test_process_no_filters(self):
        """Test filtering with no filters (all documents pass)."""
        stage = FilterStage()
        documents = [
            Document(page_content="Any content", metadata={}),
            Document(page_content="Another content", metadata={})
        ]
        context = ProcessingContext(resource=create_test_resource_config(), documents=documents)
        
        result = await stage.process(context)
        
        assert len(result.documents) == 2
        assert result.metadata['filtered_count'] == 2
        assert result.metadata['removed_count'] == 0
        assert result.metadata['filter_rules_count'] == 0


class TestFilterFunctions:
    """Test filter helper functions."""
    
    def test_no_duplicate_content_filter(self):
        """Test no_duplicate_content_filter function."""
        from src.paas_ai.core.rag.processing.stages.filter import no_duplicate_content_filter
        
        seen_content = set()
        filter_func = no_duplicate_content_filter(seen_content)
        
        # First call should pass
        assert filter_func("unique content") is True
        assert len(seen_content) == 1
        
        # Duplicate should fail
        assert filter_func("unique content") is False
        assert len(seen_content) == 1
        
        # Different content should pass
        assert filter_func("different content") is True
        assert len(seen_content) == 2
    
    def test_min_word_count_filter(self):
        """Test min_word_count_filter function."""
        from src.paas_ai.core.rag.processing.stages.filter import min_word_count_filter
        
        filter_func = min_word_count_filter(3)
        
        assert filter_func("This has enough words") is True
        assert filter_func("Not enough") is False
        assert filter_func("") is False
        assert filter_func("One") is False
        assert filter_func("Two words") is False
    
    def test_language_filter(self):
        """Test language_filter function."""
        from src.paas_ai.core.rag.processing.stages.filter import language_filter
        
        filter_func = language_filter(["en", "es"])
        
        assert filter_func({"language": "en"}) is True
        assert filter_func({"language": "es"}) is True
        assert filter_func({"language": "fr"}) is False
        assert filter_func({"language": "unknown"}) is False
        assert filter_func({}) is False  # No language key
    
    def test_exclude_pattern_filter(self):
        """Test exclude_pattern_filter function."""
        from src.paas_ai.core.rag.processing.stages.filter import exclude_pattern_filter
        
        filter_func = exclude_pattern_filter(["spam", "advertisement"])
        
        assert filter_func("This is normal content") is True
        assert filter_func("This contains spam content") is False
        assert filter_func("This is an advertisement") is False
        assert filter_func("This has SPAM in caps") is False  # Case insensitive
        assert filter_func("This has advertisement and spam") is False


class TestEnrichmentClasses:
    """Test enrichment helper classes."""
    
    @pytest.mark.asyncio
    async def test_content_enricher(self):
        """Test ContentEnricher class."""
        from src.paas_ai.core.rag.processing.stages.enrich import ContentEnricher
        
        enricher = ContentEnricher()
        documents = [
            Document(page_content="This has ```code``` and http://example.com", metadata={}),
            Document(page_content="Contact me at test@example.com", metadata={})
        ]
        context = ProcessingContext(resource=create_test_resource_config())
        
        result = await enricher(documents, context)
        
        # Check first document
        doc1 = result[0]
        assert doc1.metadata['has_code'] is True
        assert doc1.metadata['has_urls'] is True
        assert doc1.metadata['has_emails'] is False
        assert doc1.metadata['paragraph_count'] == 1
        assert doc1.metadata['sentence_count'] > 0
        
        # Check second document
        doc2 = result[1]
        assert doc2.metadata['has_code'] is False
        assert doc2.metadata['has_urls'] is False
        assert doc2.metadata['has_emails'] is True
    
    @pytest.mark.asyncio
    async def test_language_enricher_available(self):
        """Test LanguageEnricher when langdetect is available."""
        with patch('langdetect.detect') as mock_detect:
            mock_detect.return_value = "en"
            
            from src.paas_ai.core.rag.processing.stages.enrich import LanguageEnricher
            
            enricher = LanguageEnricher()
            documents = [Document(page_content="This is English text", metadata={})]
            context = ProcessingContext(resource=create_test_resource_config())
            
            result = await enricher(documents, context)
            
            assert result[0].metadata['language'] == "en"
            mock_detect.assert_called_once_with("This is English text")
    
    @pytest.mark.asyncio
    async def test_language_enricher_not_available(self):
        """Test LanguageEnricher when langdetect is not available."""
        from src.paas_ai.core.rag.processing.stages.enrich import LanguageEnricher
        
        # Create a mock enricher that simulates unavailable state
        enricher = LanguageEnricher()
        enricher.available = False  # Manually set to unavailable
        
        documents = [Document(page_content="This is English text", metadata={})]
        context = ProcessingContext(resource=create_test_resource_config())
        
        result = await enricher(documents, context)
        
        # Should return documents unchanged
        assert result == documents
        assert 'language' not in result[0].metadata
    
    @pytest.mark.asyncio
    async def test_language_enricher_detection_error(self):
        """Test LanguageEnricher when detection fails."""
        with patch('langdetect.detect') as mock_detect:
            mock_detect.side_effect = Exception("Detection error")
            
            from src.paas_ai.core.rag.processing.stages.enrich import LanguageEnricher
            
            enricher = LanguageEnricher()
            documents = [Document(page_content="Some text", metadata={})]
            context = ProcessingContext(resource=create_test_resource_config())
            
            result = await enricher(documents, context)
            
            assert result[0].metadata['language'] == 'unknown'
