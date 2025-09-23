"""
Unit tests for processing pipeline.

Tests all components of the ProcessingPipeline including:
- ProcessingPipeline class
- Pipeline composition and execution
- Error handling and edge cases
- Integration with stages
- Batch processing
- Helper functions
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, MagicMock, AsyncMock

from src.paas_ai.core.rag.processing.pipeline import (
    ProcessingPipeline,
    load_stage,
    validate_stage,
    split_stage,
    enrich_stage,
    filter_stage
)
from src.paas_ai.core.rag.processing.base import (
    ProcessingStage,
    ProcessingContext,
    ProcessingResult,
    ProcessingMetrics
)
from src.paas_ai.core.rag.config import ResourceConfig, LoaderConfig, LoaderType, SplitterConfig, SplitterType, ResourceType
from langchain_core.documents import Document


def create_test_resource_config(url: str = "http://test.com") -> ResourceConfig:
    """Create a valid ResourceConfig for testing."""
    return ResourceConfig(
        url=url,
        resource_type=ResourceType.DSL,
        loader=LoaderConfig(type=LoaderType.WEB),
        splitter=SplitterConfig(type=SplitterType.RECURSIVE_CHARACTER)
    )


class MockProcessingStage(ProcessingStage):
    """Mock processing stage for testing."""
    
    def __init__(self, name: str, should_raise_error=False, modify_documents=True):
        super().__init__(name)
        self.should_raise_error = should_raise_error
        self.modify_documents = modify_documents
        self.process_calls = []
    
    async def process(self, context: ProcessingContext) -> ProcessingContext:
        """Mock process method."""
        self.process_calls.append(context)
        
        if self.should_raise_error:
            raise ValueError(f"Mock error in {self.name}")
        
        if self.modify_documents:
            # Add a document to simulate processing
            new_doc = Document(
                page_content=f"Processed by {self.name}",
                metadata={"stage": self.name}
            )
            context.documents.append(new_doc)
            context.metadata[f"{self.name}_processed"] = True
        
        return context


class TestProcessingPipeline:
    """Test the ProcessingPipeline class."""
    
    def test_init_with_stages(self):
        """Test pipeline initialization with stages."""
        stage1 = MockProcessingStage("stage1")
        stage2 = MockProcessingStage("stage2")
        stages = [stage1, stage2]
        
        pipeline = ProcessingPipeline(stages, name="test_pipeline")
        
        assert pipeline.stages == stages
        assert pipeline.name == "test_pipeline"
    
    def test_init_with_default_name(self):
        """Test pipeline initialization with default name."""
        stage = MockProcessingStage("stage1")
        pipeline = ProcessingPipeline([stage])
        
        assert pipeline.name == "pipeline"
    
    def test_init_empty_stages(self):
        """Test pipeline initialization with empty stages."""
        pipeline = ProcessingPipeline([])
        
        assert pipeline.stages == []
        assert pipeline.name == "pipeline"
    
    def test_or_operator_with_pipeline(self):
        """Test | operator with another pipeline."""
        stage1 = MockProcessingStage("stage1")
        stage2 = MockProcessingStage("stage2")
        stage3 = MockProcessingStage("stage3")
        
        pipeline1 = ProcessingPipeline([stage1], name="pipeline1")
        pipeline2 = ProcessingPipeline([stage2, stage3], name="pipeline2")
        
        combined = pipeline1 | pipeline2
        
        assert len(combined.stages) == 3
        assert combined.stages[0] == stage1
        assert combined.stages[1] == stage2
        assert combined.stages[2] == stage3
        assert combined.name == "pipeline1_to_pipeline2"
    
    def test_or_operator_with_stage(self):
        """Test | operator with a stage."""
        stage1 = MockProcessingStage("stage1")
        stage2 = MockProcessingStage("stage2")
        
        pipeline = ProcessingPipeline([stage1], name="pipeline1")
        combined = pipeline | stage2
        
        assert len(combined.stages) == 2
        assert combined.stages[0] == stage1
        assert combined.stages[1] == stage2
        assert combined.name == "pipeline1_to_stage2"
    
    def test_or_operator_not_implemented(self):
        """Test | operator with unsupported type."""
        stage = MockProcessingStage("stage1")
        pipeline = ProcessingPipeline([stage])
        
        with pytest.raises(TypeError, match="unsupported operand type"):
            pipeline | "not_a_stage_or_pipeline"
    
    @pytest.mark.asyncio
    async def test_process_success(self):
        """Test successful pipeline processing."""
        stage1 = MockProcessingStage("stage1")
        stage2 = MockProcessingStage("stage2")
        pipeline = ProcessingPipeline([stage1, stage2])
        
        resource = create_test_resource_config()
        
        with patch('builtins.print') as mock_print:
            result = await pipeline.process(resource)
        
        assert result.success is True
        assert result.error is None
        assert len(result.documents) == 2  # Each stage adds one document
        assert result.documents[0].metadata["stage"] == "stage1"
        assert result.documents[1].metadata["stage"] == "stage2"
        
        # Check that stages were called
        assert len(stage1.process_calls) == 1
        assert len(stage2.process_calls) == 1
        
        # Check that metrics were created
        assert len(result.context.metrics) == 2
        assert result.context.metrics[0].stage_name == "stage1"
        assert result.context.metrics[1].stage_name == "stage2"
    
    @pytest.mark.asyncio
    async def test_process_with_context_success(self):
        """Test successful pipeline processing with existing context."""
        stage1 = MockProcessingStage("stage1")
        stage2 = MockProcessingStage("stage2")
        pipeline = ProcessingPipeline([stage1, stage2])
        
        resource = create_test_resource_config()
        context = ProcessingContext(resource=resource)
        
        with patch('builtins.print') as mock_print:
            result = await pipeline.process_with_context(context)
        
        assert result.success is True
        assert result.error is None
        assert len(result.documents) == 2
        assert result.context == context
    
    @pytest.mark.asyncio
    async def test_process_with_exception(self):
        """Test pipeline processing with exception."""
        stage1 = MockProcessingStage("stage1")
        stage2 = MockProcessingStage("stage2", should_raise_error=True)
        pipeline = ProcessingPipeline([stage1, stage2])
        
        resource = create_test_resource_config()
        
        with patch('builtins.print') as mock_print:
            result = await pipeline.process(resource)
        
        assert result.success is False
        assert "Mock error in stage2" in result.error
        assert len(result.documents) == 1  # Only stage1 succeeded
    
    @pytest.mark.asyncio
    async def test_process_empty_pipeline(self):
        """Test processing with empty pipeline."""
        pipeline = ProcessingPipeline([])
        resource = create_test_resource_config()
        
        with patch('builtins.print') as mock_print:
            result = await pipeline.process(resource)
        
        assert result.success is True
        assert len(result.documents) == 0
        assert len(result.context.metrics) == 0
    
    @pytest.mark.asyncio
    async def test_process_batch_success(self):
        """Test successful batch processing."""
        stage = MockProcessingStage("stage1")
        pipeline = ProcessingPipeline([stage])
        
        resources = [
            create_test_resource_config("http://test1.com"),
            create_test_resource_config("http://test2.com"),
            create_test_resource_config("http://test3.com")
        ]
        
        with patch('builtins.print') as mock_print:
            results = await pipeline.process_batch(resources)
        
        assert len(results) == 3
        for result in results:
            assert result.success is True
            assert len(result.documents) == 1
    
    @pytest.mark.asyncio
    async def test_process_batch_with_exceptions(self):
        """Test batch processing with some exceptions."""
        stage1 = MockProcessingStage("stage1")
        stage2 = MockProcessingStage("stage2", should_raise_error=True)
        pipeline = ProcessingPipeline([stage1, stage2])
        
        resources = [
            create_test_resource_config("http://test1.com"),
            create_test_resource_config("http://test2.com")
        ]
        
        with patch('builtins.print') as mock_print:
            results = await pipeline.process_batch(resources)
        
        assert len(results) == 2
        for result in results:
            assert result.success is False
            assert "Mock error in stage2" in result.error
    
    @pytest.mark.asyncio
    async def test_process_batch_empty_list(self):
        """Test batch processing with empty resource list."""
        stage = MockProcessingStage("stage1")
        pipeline = ProcessingPipeline([stage])
        
        with patch('builtins.print') as mock_print:
            results = await pipeline.process_batch([])
        
        assert len(results) == 0
    
    @pytest.mark.asyncio
    async def test_process_batch_mixed_results(self):
        """Test batch processing with mixed success/failure results."""
        # Create a stage that fails for specific URLs
        class ConditionalErrorStage(ProcessingStage):
            async def process(self, context: ProcessingContext) -> ProcessingContext:
                if "fail" in context.resource.url:
                    raise ValueError("Conditional error")
                return context
        
        stage = ConditionalErrorStage("conditional_stage")
        pipeline = ProcessingPipeline([stage])
        
        resources = [
            create_test_resource_config("http://success1.com"),
            create_test_resource_config("http://fail.com"),
            create_test_resource_config("http://success2.com")
        ]
        
        with patch('builtins.print') as mock_print:
            results = await pipeline.process_batch(resources)
        
        assert len(results) == 3
        assert results[0].success is True
        assert results[1].success is False
        assert results[2].success is True


class TestProcessingPipelineEdgeCases:
    """Test edge cases for ProcessingPipeline."""
    
    def test_pipeline_with_none_stages(self):
        """Test pipeline with None stages."""
        with pytest.raises(TypeError):
            ProcessingPipeline(None)
    
    def test_pipeline_with_mixed_stage_types(self):
        """Test pipeline with mixed stage types."""
        stage1 = MockProcessingStage("stage1")
        stage2 = MockProcessingStage("stage2")
        
        # This should work - pipeline accepts any ProcessingStage
        pipeline = ProcessingPipeline([stage1, stage2])
        assert len(pipeline.stages) == 2
    
    def test_pipeline_name_with_special_characters(self):
        """Test pipeline with special characters in name."""
        stage = MockProcessingStage("stage1")
        pipeline = ProcessingPipeline([stage], name="pipeline-with_special.chars")
        
        assert pipeline.name == "pipeline-with_special.chars"
    
    def test_pipeline_name_with_unicode(self):
        """Test pipeline with unicode characters in name."""
        stage = MockProcessingStage("stage1")
        pipeline = ProcessingPipeline([stage], name="pipeline_测试")
        
        assert pipeline.name == "pipeline_测试"
    
    @pytest.mark.asyncio
    async def test_pipeline_with_stage_that_returns_none(self):
        """Test pipeline with stage that returns None."""
        class NoneReturningStage(ProcessingStage):
            async def process(self, context: ProcessingContext) -> ProcessingContext:
                return None  # This should cause an error
        
        stage = NoneReturningStage("none_stage")
        pipeline = ProcessingPipeline([stage])
        resource = create_test_resource_config()
        
        with patch('builtins.print') as mock_print:
            result = await pipeline.process(resource)
        
        # Should handle the None return gracefully
        assert result.success is False
    
    @pytest.mark.asyncio
    async def test_pipeline_with_stage_that_modifies_context_reference(self):
        """Test pipeline with stage that modifies context reference."""
        class ContextModifyingStage(ProcessingStage):
            async def process(self, context: ProcessingContext) -> ProcessingContext:
                # Create new context instead of modifying existing one
                new_context = ProcessingContext(
                    resource=context.resource,
                    documents=context.documents.copy(),
                    metadata=context.metadata.copy(),
                    metrics=context.metrics.copy(),
                    pipeline_id=context.pipeline_id
                )
                new_context.metadata["modified"] = True
                return new_context
        
        stage = ContextModifyingStage("modifying_stage")
        pipeline = ProcessingPipeline([stage])
        resource = create_test_resource_config()
        
        with patch('builtins.print') as mock_print:
            result = await pipeline.process(resource)
        
        assert result.success is True
        assert result.context.metadata["modified"] is True
    
    @pytest.mark.asyncio
    async def test_pipeline_metrics_accumulation(self):
        """Test that metrics accumulate correctly across stages."""
        stage1 = MockProcessingStage("stage1")
        stage2 = MockProcessingStage("stage2")
        pipeline = ProcessingPipeline([stage1, stage2])
        resource = create_test_resource_config()
        
        with patch('builtins.print') as mock_print:
            result = await pipeline.process(resource)
        
        assert len(result.context.metrics) == 2
        assert result.context.metrics[0].stage_name == "stage1"
        assert result.context.metrics[1].stage_name == "stage2"
        assert result.stage_count == 2
    
    @pytest.mark.asyncio
    async def test_pipeline_duration_calculation(self):
        """Test that pipeline duration is calculated correctly."""
        import time
        
        class SlowStage(ProcessingStage):
            async def process(self, context: ProcessingContext) -> ProcessingContext:
                await asyncio.sleep(0.1)  # Simulate processing time
                return context
        
        stage = SlowStage("slow_stage")
        pipeline = ProcessingPipeline([stage])
        resource = create_test_resource_config()
        
        with patch('builtins.print') as mock_print:
            result = await pipeline.process(resource)
        
        assert result.total_duration > 0.1
        assert result.total_duration < 0.2  # Should be close to 0.1


class TestProcessingPipelineHelperFunctions:
    """Test helper functions for creating stages."""
    
    def test_load_stage_helper(self):
        """Test load_stage helper function."""
        with patch('src.paas_ai.core.rag.processing.stages.LoadStage') as mock_load_stage:
            mock_stage = Mock()
            mock_load_stage.return_value = mock_stage
            
            result = load_stage(custom_param="value")
            
            mock_load_stage.assert_called_once_with(custom_param="value")
            assert result == mock_stage
    
    def test_validate_stage_helper(self):
        """Test validate_stage helper function."""
        with patch('src.paas_ai.core.rag.processing.stages.ValidateStage') as mock_validate_stage:
            mock_stage = Mock()
            mock_validate_stage.return_value = mock_stage
            
            result = validate_stage(validator_config="config")
            
            mock_validate_stage.assert_called_once_with(validator_config="config")
            assert result == mock_stage
    
    def test_split_stage_helper(self):
        """Test split_stage helper function."""
        with patch('src.paas_ai.core.rag.processing.stages.SplitStage') as mock_split_stage:
            mock_stage = Mock()
            mock_split_stage.return_value = mock_stage
            
            result = split_stage(custom_param="value")
            
            mock_split_stage.assert_called_once_with(custom_param="value")
            assert result == mock_stage
    
    def test_enrich_stage_helper(self):
        """Test enrich_stage helper function."""
        with patch('src.paas_ai.core.rag.processing.stages.EnrichStage') as mock_enrich_stage:
            mock_stage = Mock()
            mock_enrich_stage.return_value = mock_stage
            
            result = enrich_stage(custom_enrichers=["enricher1"])
            
            mock_enrich_stage.assert_called_once_with(custom_enrichers=["enricher1"])
            assert result == mock_stage
    
    def test_filter_stage_helper(self):
        """Test filter_stage helper function."""
        with patch('src.paas_ai.core.rag.processing.stages.FilterStage') as mock_filter_stage:
            mock_stage = Mock()
            mock_filter_stage.return_value = mock_stage
            
            result = filter_stage(content_filters=["filter1"])
            
            mock_filter_stage.assert_called_once_with(content_filters=["filter1"])
            assert result == mock_stage
    
    def test_helper_functions_with_no_params(self):
        """Test helper functions with no parameters."""
        with patch('src.paas_ai.core.rag.processing.stages.LoadStage') as mock_load_stage, \
             patch('src.paas_ai.core.rag.processing.stages.ValidateStage') as mock_validate_stage, \
             patch('src.paas_ai.core.rag.processing.stages.SplitStage') as mock_split_stage, \
             patch('src.paas_ai.core.rag.processing.stages.EnrichStage') as mock_enrich_stage, \
             patch('src.paas_ai.core.rag.processing.stages.FilterStage') as mock_filter_stage:
            
            load_stage()
            validate_stage()
            split_stage()
            enrich_stage()
            filter_stage()
            
            mock_load_stage.assert_called_once_with()
            mock_validate_stage.assert_called_once_with()
            mock_split_stage.assert_called_once_with()
            mock_enrich_stage.assert_called_once_with()
            mock_filter_stage.assert_called_once_with()


class TestProcessingPipelineIntegration:
    """Integration tests for ProcessingPipeline."""
    
    @pytest.mark.asyncio
    async def test_full_pipeline_workflow(self):
        """Test complete pipeline workflow."""
        # Create a realistic pipeline
        stage1 = MockProcessingStage("load", modify_documents=True)
        stage2 = MockProcessingStage("validate", modify_documents=False)
        stage3 = MockProcessingStage("split", modify_documents=True)
        
        pipeline = ProcessingPipeline([stage1, stage2, stage3], name="full_pipeline")
        resource = create_test_resource_config()
        
        with patch('builtins.print') as mock_print:
            result = await pipeline.process(resource)
        
        assert result.success is True
        assert len(result.documents) == 2  # load and split add documents
        assert result.stage_count == 3
        
        # Check that all stages were called
        assert len(stage1.process_calls) == 1
        assert len(stage2.process_calls) == 1
        assert len(stage3.process_calls) == 1
    
    @pytest.mark.asyncio
    async def test_pipeline_composition_workflow(self):
        """Test pipeline composition workflow."""
        # Create individual pipelines
        load_pipeline = ProcessingPipeline([MockProcessingStage("load")], name="load_pipeline")
        process_pipeline = ProcessingPipeline([MockProcessingStage("process")], name="process_pipeline")
        
        # Compose them
        full_pipeline = load_pipeline | process_pipeline
        
        assert full_pipeline.name == "load_pipeline_to_process_pipeline"
        assert len(full_pipeline.stages) == 2
        
        resource = create_test_resource_config()
        
        with patch('builtins.print') as mock_print:
            result = await full_pipeline.process(resource)
        
        assert result.success is True
        assert result.stage_count == 2
    
    @pytest.mark.asyncio
    async def test_pipeline_with_context_preservation(self):
        """Test that pipeline preserves context across stages."""
        class ContextPreservingStage(ProcessingStage):
            def __init__(self, name: str, add_metadata: str):
                super().__init__(name)
                self.add_metadata = add_metadata
            
            async def process(self, context: ProcessingContext) -> ProcessingContext:
                context.metadata[self.add_metadata] = True
                return context
        
        stage1 = ContextPreservingStage("stage1", "stage1_done")
        stage2 = ContextPreservingStage("stage2", "stage2_done")
        pipeline = ProcessingPipeline([stage1, stage2])
        
        resource = create_test_resource_config()
        
        with patch('builtins.print') as mock_print:
            result = await pipeline.process(resource)
        
        assert result.success is True
        assert result.context.metadata["stage1_done"] is True
        assert result.context.metadata["stage2_done"] is True
    
    @pytest.mark.asyncio
    async def test_pipeline_error_recovery(self):
        """Test pipeline behavior with error recovery scenarios."""
        class RecoverableStage(ProcessingStage):
            def __init__(self, name: str, fail_first_time=True):
                super().__init__(name)
                self.fail_first_time = fail_first_time
                self.call_count = 0
            
            async def process(self, context: ProcessingContext) -> ProcessingContext:
                self.call_count += 1
                if self.fail_first_time and self.call_count == 1:
                    raise ValueError("First attempt failed")
                context.metadata[f"{self.name}_success"] = True
                return context
        
        stage = RecoverableStage("recoverable")
        pipeline = ProcessingPipeline([stage])
        resource = create_test_resource_config()
        
        with patch('builtins.print') as mock_print:
            result = await pipeline.process(resource)
        
        # Should fail on first attempt
        assert result.success is False
        assert "First attempt failed" in result.error
        assert stage.call_count == 1
    
    @pytest.mark.asyncio
    async def test_pipeline_performance_characteristics(self):
        """Test pipeline performance characteristics."""
        import time
        
        class PerformanceStage(ProcessingStage):
            async def process(self, context: ProcessingContext) -> ProcessingContext:
                # Simulate some processing
                await asyncio.sleep(0.01)
                return context
        
        # Test with multiple stages
        stages = [PerformanceStage(f"stage_{i}") for i in range(5)]
        pipeline = ProcessingPipeline(stages)
        resource = create_test_resource_config()
        
        start_time = time.time()
        with patch('builtins.print') as mock_print:
            result = await pipeline.process(resource)
        end_time = time.time()
        
        assert result.success is True
        assert result.stage_count == 5
        # Should take at least 0.05 seconds (5 * 0.01)
        assert (end_time - start_time) >= 0.05
