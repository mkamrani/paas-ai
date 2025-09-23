"""
Integration tests for the processing system.

Tests the complete processing system integration including:
- End-to-end workflows
- Cross-component interactions
- Real processing scenarios (with mocking)
- Error propagation
- Configuration validation across components
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, MagicMock, AsyncMock

from src.paas_ai.core.rag.processing.pipeline import ProcessingPipeline
from src.paas_ai.core.rag.processing.stages import (
    LoadStage,
    ValidateStage,
    SplitStage,
    EnrichStage,
    FilterStage
)
from src.paas_ai.core.rag.processing.base import ProcessingContext, ProcessingResult, ProcessingStage
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


class TestProcessingSystemIntegration:
    """Integration tests for the complete processing system."""
    
    @pytest.mark.asyncio
    async def test_full_pipeline_workflow_integration(self):
        """Test complete pipeline workflow integration."""
        # Mock all the dependencies
        mock_docs = [
            Document(page_content="Original content to be processed", metadata={"source": "test"})
        ]
        
        mock_loader = Mock()
        mock_loader.load.return_value = mock_docs
        
        mock_validator = Mock()
        mock_validator.validate_documents.return_value = mock_docs
        
        mock_splitter = Mock()
        mock_splitter.split_documents.return_value = [
            Document(page_content="Chunk 1", metadata={"chunk": 1}),
            Document(page_content="Chunk 2", metadata={"chunk": 2})
        ]
        
        with patch('src.paas_ai.core.rag.processing.stages.load.DocumentLoaderFactory') as mock_loader_factory, \
             patch('src.paas_ai.core.rag.processing.stages.validate.ContentValidatorFactory') as mock_validator_factory, \
             patch('src.paas_ai.core.rag.processing.stages.split.TextSplitterFactory') as mock_splitter_factory:
            
            mock_loader_factory.create_loader.return_value = mock_loader
            mock_validator_factory.create_content_validator.return_value = mock_validator
            mock_splitter_factory.create_splitter.return_value = mock_splitter
            
            # Create pipeline
            pipeline = (
                LoadStage() |
                ValidateStage() |
                SplitStage() |
                EnrichStage() |
                FilterStage()
            )
            
            resource = create_test_resource_config()
            
            with patch('builtins.print'):
                result = await pipeline.process(resource)
            
            # Verify the complete workflow
            assert result.success is True
            assert len(result.documents) == 2  # Split into 2 chunks
            assert result.stage_count == 5
            
            # Verify all factories were called
            mock_loader_factory.create_loader.assert_called_once()
            mock_validator_factory.create_content_validator.assert_called_once()
            mock_splitter_factory.create_splitter.assert_called_once()
            
            # Verify documents were enriched
            for doc in result.documents:
                assert doc.metadata['source_url'] == "http://test.com"
                assert 'processed_at' in doc.metadata
                assert 'content_length' in doc.metadata
    
    @pytest.mark.asyncio
    async def test_pipeline_with_custom_enrichers_integration(self):
        """Test pipeline integration with custom enrichers."""
        # Mock custom enricher
        mock_enricher = AsyncMock()
        mock_enricher.return_value = [
            Document(page_content="Enriched content", metadata={"custom_enriched": True})
        ]
        
        # Mock basic dependencies
        mock_docs = [Document(page_content="Test content", metadata={})]
        mock_loader = Mock()
        mock_loader.load.return_value = mock_docs
        
        mock_validator = Mock()
        mock_validator.validate_documents.return_value = mock_docs
        
        with patch('src.paas_ai.core.rag.processing.stages.load.DocumentLoaderFactory') as mock_loader_factory, \
             patch('src.paas_ai.core.rag.processing.stages.validate.ContentValidatorFactory') as mock_validator_factory:
            
            mock_loader_factory.create_loader.return_value = mock_loader
            mock_validator_factory.create_content_validator.return_value = mock_validator
            
            # Create pipeline with custom enricher
            pipeline = (
                LoadStage() |
                ValidateStage() |
                EnrichStage(custom_enrichers=[mock_enricher])
            )
            
            resource = create_test_resource_config()
            
            with patch('builtins.print'):
                result = await pipeline.process(resource)
            
            # Verify custom enricher was called
            mock_enricher.assert_called_once()
            
            # Verify result
            assert result.success is True
            assert len(result.documents) == 1
            assert result.documents[0].metadata['custom_enriched'] is True
    
    @pytest.mark.asyncio
    async def test_pipeline_with_filters_integration(self):
        """Test pipeline integration with content and metadata filters."""
        from src.paas_ai.core.rag.processing.stages.filter import min_word_count_filter, language_filter
        
        # Mock dependencies
        mock_docs = [
            Document(page_content="This is a long enough content", metadata={"language": "en"}),
            Document(page_content="Short", metadata={"language": "en"}),  # Will be filtered
            Document(page_content="This is also long enough", metadata={"language": "fr"})  # Will be filtered
        ]
        
        mock_loader = Mock()
        mock_loader.load.return_value = mock_docs
        
        mock_validator = Mock()
        mock_validator.validate_documents.return_value = mock_docs
        
        with patch('src.paas_ai.core.rag.processing.stages.load.DocumentLoaderFactory') as mock_loader_factory, \
             patch('src.paas_ai.core.rag.processing.stages.validate.ContentValidatorFactory') as mock_validator_factory:
            
            mock_loader_factory.create_loader.return_value = mock_loader
            mock_validator_factory.create_content_validator.return_value = mock_validator
            
            # Create pipeline with filters
            pipeline = (
                LoadStage() |
                ValidateStage() |
                FilterStage(
                    content_filters=[min_word_count_filter(5)],
                    metadata_filters=[language_filter(["en"])]
                )
            )
            
            resource = create_test_resource_config()
            
            with patch('builtins.print'):
                result = await pipeline.process(resource)
            
            # Verify filtering worked
            assert result.success is True
            assert len(result.documents) == 1  # Only one document should pass both filters
            assert result.documents[0].page_content == "This is a long enough content"
            assert result.context.metadata['filtered_count'] == 1
            assert result.context.metadata['removed_count'] == 2
    
    @pytest.mark.asyncio
    async def test_pipeline_error_propagation_integration(self):
        """Test error propagation through the pipeline."""
        # Mock loader that raises an error
        mock_loader = Mock()
        mock_loader.load.side_effect = Exception("Loader failed")
        
        with patch('src.paas_ai.core.rag.processing.stages.load.DocumentLoaderFactory') as mock_loader_factory:
            mock_loader_factory.create_loader.return_value = mock_loader
            
            # Create pipeline
            pipeline = LoadStage() | ValidateStage() | SplitStage()
            
            resource = create_test_resource_config()
            
            with patch('builtins.print'):
                result = await pipeline.process(resource)
            
            # Verify error was propagated
            assert result.success is False
            assert "Loader failed" in result.error
            assert len(result.documents) == 0
    
    @pytest.mark.asyncio
    async def test_pipeline_batch_processing_integration(self):
        """Test batch processing integration."""
        # Mock dependencies
        mock_docs = [Document(page_content="Test content", metadata={})]
        mock_loader = Mock()
        mock_loader.load.return_value = mock_docs
        
        mock_validator = Mock()
        mock_validator.validate_documents.return_value = mock_docs
        
        with patch('src.paas_ai.core.rag.processing.stages.load.DocumentLoaderFactory') as mock_loader_factory, \
             patch('src.paas_ai.core.rag.processing.stages.validate.ContentValidatorFactory') as mock_validator_factory:
            
            mock_loader_factory.create_loader.return_value = mock_loader
            mock_validator_factory.create_content_validator.return_value = mock_validator
            
            # Create pipeline
            pipeline = LoadStage() | ValidateStage() | EnrichStage()
            
            # Create multiple resources
            resources = [
                create_test_resource_config("http://test1.com"),
                create_test_resource_config("http://test2.com"),
                create_test_resource_config("http://test3.com")
            ]
            
            with patch('builtins.print'):
                results = await pipeline.process_batch(resources)
            
            # Verify batch processing
            assert len(results) == 3
            for result in results:
                assert result.success is True
                assert len(result.documents) == 1
                assert result.documents[0].metadata['source_url'] in ["http://test1.com", "http://test2.com", "http://test3.com"]
    
    @pytest.mark.asyncio
    async def test_pipeline_with_mixed_success_failure_integration(self):
        """Test pipeline with mixed success/failure scenarios."""
        # Create a conditional loader that fails for specific URLs
        class ConditionalLoader:
            def load(self):
                if "fail" in self.url:
                    raise Exception("Conditional failure")
                return [Document(page_content="Success content", metadata={})]
        
        # Mock factory to return conditional loader
        def create_conditional_loader(loader_config, url):
            loader = ConditionalLoader()
            loader.url = url
            return loader
        
        mock_validator = Mock()
        mock_validator.validate_documents.return_value = [Document(page_content="Success content", metadata={})]
        
        with patch('src.paas_ai.core.rag.processing.stages.load.DocumentLoaderFactory') as mock_loader_factory, \
             patch('src.paas_ai.core.rag.processing.stages.validate.ContentValidatorFactory') as mock_validator_factory:
            
            mock_loader_factory.create_loader.side_effect = create_conditional_loader
            mock_validator_factory.create_content_validator.return_value = mock_validator
            
            # Create pipeline
            pipeline = LoadStage() | ValidateStage() | EnrichStage()
            
            # Create resources with mixed success/failure
            resources = [
                create_test_resource_config("http://success1.com"),
                create_test_resource_config("http://fail.com"),
                create_test_resource_config("http://success2.com")
            ]
            
            with patch('builtins.print'):
                results = await pipeline.process_batch(resources)
            
            # Verify mixed results
            assert len(results) == 3
            assert results[0].success is True
            assert results[1].success is False
            assert results[2].success is True
            assert "Conditional failure" in results[1].error


class TestProcessingSystemErrorHandling:
    """Test error handling across the processing system."""
    
    @pytest.mark.asyncio
    async def test_loader_factory_error_handling(self):
        """Test error handling for loader factory failures."""
        with patch('src.paas_ai.core.rag.processing.stages.load.DocumentLoaderFactory') as mock_factory:
            mock_factory.create_loader.side_effect = Exception("Factory error")
            
            pipeline = ProcessingPipeline([LoadStage()])
            resource = create_test_resource_config()
            
            with patch('builtins.print'):
                result = await pipeline.process(resource)
            
            assert result.success is False
            assert "Factory error" in result.error
    
    @pytest.mark.asyncio
    async def test_validator_factory_error_handling(self):
        """Test error handling for validator factory failures."""
        mock_docs = [Document(page_content="Test", metadata={})]
        mock_loader = Mock()
        mock_loader.load.return_value = mock_docs
        
        with patch('src.paas_ai.core.rag.processing.stages.load.DocumentLoaderFactory') as mock_loader_factory, \
             patch('src.paas_ai.core.rag.processing.stages.validate.ContentValidatorFactory') as mock_validator_factory:
            
            mock_loader_factory.create_loader.return_value = mock_loader
            mock_validator_factory.create_content_validator.side_effect = Exception("Validator factory error")
            
            # Create stages after setting up the mock
            load_stage = LoadStage()
            try:
                validate_stage = ValidateStage()
                # If we get here, the test should fail because the exception wasn't raised
                assert False, "Expected ValidateStage creation to raise an exception"
            except Exception as e:
                assert "Validator factory error" in str(e)
    
    @pytest.mark.asyncio
    async def test_splitter_factory_error_handling(self):
        """Test error handling for splitter factory failures."""
        mock_docs = [Document(page_content="Test", metadata={})]
        mock_loader = Mock()
        mock_loader.load.return_value = mock_docs
        
        mock_validator = Mock()
        mock_validator.validate_documents.return_value = mock_docs
        
        with patch('src.paas_ai.core.rag.processing.stages.load.DocumentLoaderFactory') as mock_loader_factory, \
             patch('src.paas_ai.core.rag.processing.stages.validate.ContentValidatorFactory') as mock_validator_factory, \
             patch('src.paas_ai.core.rag.processing.stages.split.TextSplitterFactory') as mock_splitter_factory:
            
            mock_loader_factory.create_loader.return_value = mock_loader
            mock_validator_factory.create_content_validator.return_value = mock_validator
            mock_splitter_factory.create_splitter.side_effect = Exception("Splitter factory error")
            
            pipeline = LoadStage() | ValidateStage() | SplitStage()
            resource = create_test_resource_config()
            
            with patch('builtins.print'):
                result = await pipeline.process(resource)
            
            assert result.success is False
            assert "Splitter factory error" in result.error
    
    @pytest.mark.asyncio
    async def test_custom_enricher_error_handling(self):
        """Test error handling for custom enricher failures."""
        # Mock enricher that raises an error
        mock_enricher = AsyncMock()
        mock_enricher.side_effect = Exception("Enricher error")
        
        mock_docs = [Document(page_content="Test", metadata={})]
        mock_loader = Mock()
        mock_loader.load.return_value = mock_docs
        
        mock_validator = Mock()
        mock_validator.validate_documents.return_value = mock_docs
        
        with patch('src.paas_ai.core.rag.processing.stages.load.DocumentLoaderFactory') as mock_loader_factory, \
             patch('src.paas_ai.core.rag.processing.stages.validate.ContentValidatorFactory') as mock_validator_factory:
            
            mock_loader_factory.create_loader.return_value = mock_loader
            mock_validator_factory.create_content_validator.return_value = mock_validator
            
            pipeline = LoadStage() | ValidateStage() | EnrichStage(custom_enrichers=[mock_enricher])
            resource = create_test_resource_config()
            
            with patch('builtins.print'):
                result = await pipeline.process(resource)
            
            assert result.success is False
            assert "Enricher error" in result.error


class TestProcessingSystemPerformance:
    """Test performance characteristics of the processing system."""
    
    @pytest.mark.asyncio
    async def test_pipeline_stage_isolation(self):
        """Test that pipeline stages are isolated and don't interfere."""
        # Create stages that modify context in different ways
        class IsolatedStage1(ProcessingStage):
            def __init__(self):
                super().__init__("isolated1")
            
            async def process(self, context: ProcessingContext) -> ProcessingContext:
                context.metadata["stage1"] = "processed"
                return context
        
        class IsolatedStage2(ProcessingStage):
            def __init__(self):
                super().__init__("isolated2")
            
            async def process(self, context: ProcessingContext) -> ProcessingContext:
                context.metadata["stage2"] = "processed"
                return context
        
        pipeline = ProcessingPipeline([IsolatedStage1(), IsolatedStage2()])
        resource = create_test_resource_config()
        
        with patch('builtins.print'):
            result = await pipeline.process(resource)
        
        # Verify both stages were processed
        assert result.success is True
        assert result.context.metadata["stage1"] == "processed"
        assert result.context.metadata["stage2"] == "processed"
    
    async def _mock_stage_call(self, context: ProcessingContext) -> ProcessingContext:
        """Helper method to mock stage calls."""
        return context
    
    @pytest.mark.asyncio
    async def test_pipeline_metrics_accuracy(self):
        """Test that pipeline metrics are accurate."""
        mock_docs = [Document(page_content="Test content", metadata={})]
        mock_loader = Mock()
        mock_loader.load.return_value = mock_docs
        
        mock_validator = Mock()
        mock_validator.validate_documents.return_value = mock_docs
        
        with patch('src.paas_ai.core.rag.processing.stages.load.DocumentLoaderFactory') as mock_loader_factory, \
             patch('src.paas_ai.core.rag.processing.stages.validate.ContentValidatorFactory') as mock_validator_factory:
            
            mock_loader_factory.create_loader.return_value = mock_loader
            mock_validator_factory.create_content_validator.return_value = mock_validator
            
            pipeline = LoadStage() | ValidateStage() | EnrichStage()
            resource = create_test_resource_config()
            
            with patch('builtins.print'):
                result = await pipeline.process(resource)
            
            # Verify metrics accuracy
            assert result.stage_count == 3
            assert result.total_duration > 0
            
            # Check individual stage metrics
            metrics = result.context.metrics
            assert len(metrics) == 3
            assert metrics[0].stage_name == "load"
            assert metrics[1].stage_name == "validate"
            assert metrics[2].stage_name == "enrich"
            
            # All stages should be completed
            for metric in metrics:
                assert metric.status.value == "completed"
    
    @pytest.mark.asyncio
    async def test_pipeline_memory_efficiency(self):
        """Test that pipeline doesn't leak memory or accumulate state."""
        mock_docs = [Document(page_content="Test content", metadata={})]
        mock_loader = Mock()
        mock_loader.load.return_value = mock_docs
        
        mock_validator = Mock()
        mock_validator.validate_documents.return_value = mock_docs
        
        with patch('src.paas_ai.core.rag.processing.stages.load.DocumentLoaderFactory') as mock_loader_factory, \
             patch('src.paas_ai.core.rag.processing.stages.validate.ContentValidatorFactory') as mock_validator_factory:
            
            mock_loader_factory.create_loader.return_value = mock_loader
            mock_validator_factory.create_content_validator.return_value = mock_validator
            
            pipeline = LoadStage() | ValidateStage() | EnrichStage()
            
            # Process multiple resources
            resources = [
                create_test_resource_config(f"http://test{i}.com")
                for i in range(5)
            ]
            
            with patch('builtins.print'):
                results = await pipeline.process_batch(resources)
            
            # Verify all results are independent
            assert len(results) == 5
            for i, result in enumerate(results):
                assert result.success is True
                assert result.context.resource.url == f"http://test{i}.com"
                assert len(result.context.metrics) == 3  # Each should have its own metrics


class TestProcessingSystemCompatibility:
    """Test compatibility and interoperability of the processing system."""
    
    @pytest.mark.asyncio
    async def test_pipeline_stage_composition_compatibility(self):
        """Test that pipeline stage composition works correctly."""
        # Test different composition patterns
        stage1 = LoadStage(name="load1")
        stage2 = ValidateStage(name="validate1")
        stage3 = EnrichStage(name="enrich1")
        
        # Test direct composition
        pipeline1 = stage1 | stage2 | stage3
        assert len(pipeline1.stages) == 3
        assert pipeline1.name == "load1_to_validate1_to_enrich1"
        
        # Test pipeline composition
        pipeline2 = ProcessingPipeline([stage1], name="pipeline2")
        pipeline3 = pipeline2 | stage2
        assert len(pipeline3.stages) == 2
        assert pipeline3.name == "pipeline2_to_validate1"
        
        # Test mixed composition
        pipeline4 = stage1 | ProcessingPipeline([stage2, stage3], name="sub_pipeline")
        assert len(pipeline4.stages) == 3
        assert pipeline4.name == "load1_to_sub_pipeline"
    
    @pytest.mark.asyncio
    async def test_pipeline_context_preservation_compatibility(self):
        """Test that pipeline preserves context correctly across stages."""
        # Create stages that add to context
        class ContextAddingStage(ProcessingStage):
            def __init__(self, name: str, key: str):
                super().__init__(name)
                self.key = key
            
            async def process(self, context: ProcessingContext) -> ProcessingContext:
                context.metadata[self.key] = f"added_by_{self.name}"
                return context
        
        stage1 = ContextAddingStage("stage1", "key1")
        stage2 = ContextAddingStage("stage2", "key2")
        
        pipeline = ProcessingPipeline([stage1, stage2])
        resource = create_test_resource_config()
        
        with patch('builtins.print'):
            result = await pipeline.process(resource)
        
        # Verify context was preserved and modified correctly
        assert result.success is True
        assert result.context.metadata["key1"] == "added_by_stage1"
        assert result.context.metadata["key2"] == "added_by_stage2"
        assert result.context.resource.url == "http://test.com"
    
    @pytest.mark.asyncio
    async def test_pipeline_error_handling_compatibility(self):
        """Test that pipeline error handling is compatible across components."""
        # Test that errors from one stage don't affect others in batch processing
        class ConditionalErrorStage(ProcessingStage):
            def __init__(self, name: str, fail_urls: list):
                super().__init__(name)
                self.fail_urls = fail_urls
            
            async def process(self, context: ProcessingContext) -> ProcessingContext:
                if any(fail_url in context.resource.url for fail_url in self.fail_urls):
                    raise Exception(f"Error in {self.name}")
                context.metadata[f"{self.name}_success"] = True
                return context
        
        stage = ConditionalErrorStage("conditional", ["fail"])
        pipeline = ProcessingPipeline([stage])
        
        resources = [
            create_test_resource_config("http://success1.com"),
            create_test_resource_config("http://fail.com"),
            create_test_resource_config("http://success2.com")
        ]
        
        with patch('builtins.print'):
            results = await pipeline.process_batch(resources)
        
        # Verify error handling compatibility
        assert len(results) == 3
        assert results[0].success is True
        assert results[1].success is False
        assert results[2].success is True
        
        # Verify successful results have correct metadata
        assert results[0].context.metadata["conditional_success"] is True
        assert results[2].context.metadata["conditional_success"] is True
    
    @pytest.mark.asyncio
    async def test_pipeline_configuration_compatibility(self):
        """Test that pipeline works with different configuration types."""
        # Test with minimal configuration
        minimal_resource = create_test_resource_config()
        
        # Test with full configuration
        full_resource = ResourceConfig(
            url="http://test.com",
            resource_type=ResourceType.DSL,
            priority=1,
            tags=["test"],
            metadata={"custom": "value"},
            loader=LoaderConfig(type=LoaderType.WEB),
            splitter=SplitterConfig(type=SplitterType.RECURSIVE_CHARACTER)
        )
        
        # Both should work with appropriate stages
        mock_docs = [Document(page_content="Test", metadata={})]
        mock_loader = Mock()
        mock_loader.load.return_value = mock_docs
        
        with patch('src.paas_ai.core.rag.processing.stages.load.DocumentLoaderFactory') as mock_factory:
            mock_factory.create_loader.return_value = mock_loader
            
            pipeline = LoadStage() | EnrichStage()
            
            with patch('builtins.print'):
                result1 = await pipeline.process(minimal_resource)
                result2 = await pipeline.process(full_resource)
            
            # Both should succeed
            assert result1.success is True
            assert result2.success is True
            
            # Both should have enriched documents
            assert len(result1.documents) == 1
            assert len(result2.documents) == 1
            
            # Full resource should have more metadata
            assert result2.documents[0].metadata.get("custom") == "value"
            assert result2.documents[0].metadata.get("tags") == ["test"]
