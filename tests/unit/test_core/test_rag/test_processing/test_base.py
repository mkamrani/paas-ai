"""
Unit tests for base processing interfaces.

Tests the abstract ProcessingStage base class and related classes including:
- ProcessingStage abstract base class
- ProcessingContext data class
- ProcessingResult data class
- ProcessingMetrics data class
- ProcessingStatus enum
- Error handling and edge cases
- Type checking and validation
"""

import pytest
import time
import uuid
from abc import ABC
from unittest.mock import Mock, patch, MagicMock
from dataclasses import dataclass

from src.paas_ai.core.rag.processing.base import (
    ProcessingStage,
    ProcessingContext,
    ProcessingResult,
    ProcessingMetrics,
    ProcessingStatus
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


class TestProcessingStatus:
    """Test the ProcessingStatus enum."""
    
    def test_enum_values(self):
        """Test that ProcessingStatus has correct values."""
        assert ProcessingStatus.PENDING == "pending"
        assert ProcessingStatus.RUNNING == "running"
        assert ProcessingStatus.COMPLETED == "completed"
        assert ProcessingStatus.FAILED == "failed"
        assert ProcessingStatus.SKIPPED == "skipped"
    
    def test_enum_inheritance(self):
        """Test that ProcessingStatus inherits from str and Enum."""
        assert issubclass(ProcessingStatus, str)
        assert issubclass(ProcessingStatus, type(ProcessingStatus.PENDING).__bases__[0])
    
    def test_enum_membership(self):
        """Test enum membership operations."""
        assert "pending" in ProcessingStatus
        assert "running" in ProcessingStatus
        assert "completed" in ProcessingStatus
        assert "failed" in ProcessingStatus
        assert "skipped" in ProcessingStatus
        assert "invalid" not in ProcessingStatus


class TestProcessingMetrics:
    """Test the ProcessingMetrics data class."""
    
    def test_init_defaults(self):
        """Test ProcessingMetrics initialization with defaults."""
        metric = ProcessingMetrics(stage_name="test_stage")
        
        assert metric.stage_name == "test_stage"
        assert metric.start_time is not None
        assert metric.end_time is None
        assert metric.input_count == 0
        assert metric.output_count == 0
        assert metric.errors == []
    
    def test_init_with_values(self):
        """Test ProcessingMetrics initialization with custom values."""
        start_time = time.time()
        metric = ProcessingMetrics(
            stage_name="test_stage",
            start_time=start_time,
            end_time=start_time + 1.0,
            input_count=10,
            output_count=8,
            errors=["error1", "error2"]
        )
        
        assert metric.stage_name == "test_stage"
        assert metric.start_time == start_time
        assert metric.end_time == start_time + 1.0
        assert metric.input_count == 10
        assert metric.output_count == 8
        assert metric.errors == ["error1", "error2"]
    
    def test_duration_property(self):
        """Test the duration property."""
        start_time = time.time()
        metric = ProcessingMetrics(
            stage_name="test_stage",
            start_time=start_time,
            end_time=start_time + 2.5
        )
        
        assert metric.duration == 2.5
    
    def test_duration_property_none_end_time(self):
        """Test duration property when end_time is None."""
        metric = ProcessingMetrics(stage_name="test_stage")
        
        assert metric.duration is None
    
    def test_status_property_with_errors(self):
        """Test status property when there are errors."""
        metric = ProcessingMetrics(
            stage_name="test_stage",
            errors=["some error"]
        )
        
        assert metric.status == ProcessingStatus.FAILED
    
    def test_status_property_completed(self):
        """Test status property when completed."""
        start_time = time.time()
        metric = ProcessingMetrics(
            stage_name="test_stage",
            start_time=start_time,
            end_time=start_time + 1.0
        )
        
        assert metric.status == ProcessingStatus.COMPLETED
    
    def test_status_property_running(self):
        """Test status property when running."""
        metric = ProcessingMetrics(stage_name="test_stage")
        
        assert metric.status == ProcessingStatus.RUNNING


class TestProcessingContext:
    """Test the ProcessingContext data class."""
    
    def test_init_defaults(self):
        """Test ProcessingContext initialization with defaults."""
        resource = create_test_resource_config()
        context = ProcessingContext(resource=resource)
        
        assert context.resource == resource
        assert context.documents == []
        assert context.metadata == {}
        assert context.metrics == []
        assert isinstance(context.pipeline_id, str)
        assert len(context.pipeline_id) > 0
    
    def test_init_with_values(self):
        """Test ProcessingContext initialization with custom values."""
        resource = create_test_resource_config()
        documents = [Document(page_content="test", metadata={})]
        metadata = {"key": "value"}
        metrics = [ProcessingMetrics(stage_name="test")]
        pipeline_id = "test-pipeline-id"
        
        context = ProcessingContext(
            resource=resource,
            documents=documents,
            metadata=metadata,
            metrics=metrics,
            pipeline_id=pipeline_id
        )
        
        assert context.resource == resource
        assert context.documents == documents
        assert context.metadata == metadata
        assert context.metrics == metrics
        assert context.pipeline_id == pipeline_id
    
    def test_add_metric(self):
        """Test adding metrics to context."""
        resource = create_test_resource_config()
        context = ProcessingContext(resource=resource)
        metric = ProcessingMetrics(stage_name="test_stage")
        
        context.add_metric(metric)
        
        assert len(context.metrics) == 1
        assert context.metrics[0] == metric
    
    def test_add_multiple_metrics(self):
        """Test adding multiple metrics to context."""
        resource = create_test_resource_config()
        context = ProcessingContext(resource=resource)
        metric1 = ProcessingMetrics(stage_name="stage1")
        metric2 = ProcessingMetrics(stage_name="stage2")
        
        context.add_metric(metric1)
        context.add_metric(metric2)
        
        assert len(context.metrics) == 2
        assert context.metrics[0] == metric1
        assert context.metrics[1] == metric2
    
    def test_get_metric_existing(self):
        """Test getting existing metric from context."""
        resource = create_test_resource_config()
        context = ProcessingContext(resource=resource)
        metric = ProcessingMetrics(stage_name="test_stage")
        context.add_metric(metric)
        
        result = context.get_metric("test_stage")
        
        assert result == metric
    
    def test_get_metric_nonexistent(self):
        """Test getting non-existent metric from context."""
        resource = create_test_resource_config()
        context = ProcessingContext(resource=resource)
        
        result = context.get_metric("nonexistent_stage")
        
        assert result is None
    
    def test_get_metric_multiple_metrics(self):
        """Test getting metric when multiple metrics exist."""
        resource = create_test_resource_config()
        context = ProcessingContext(resource=resource)
        metric1 = ProcessingMetrics(stage_name="stage1")
        metric2 = ProcessingMetrics(stage_name="stage2")
        context.add_metric(metric1)
        context.add_metric(metric2)
        
        result = context.get_metric("stage2")
        
        assert result == metric2


class TestProcessingResult:
    """Test the ProcessingResult data class."""
    
    def test_init_success(self):
        """Test ProcessingResult initialization for success."""
        resource = create_test_resource_config()
        context = ProcessingContext(resource=resource)
        result = ProcessingResult(context=context, success=True)
        
        assert result.context == context
        assert result.success is True
        assert result.error is None
    
    def test_init_failure(self):
        """Test ProcessingResult initialization for failure."""
        resource = create_test_resource_config()
        context = ProcessingContext(resource=resource)
        error = "Test error"
        result = ProcessingResult(context=context, success=False, error=error)
        
        assert result.context == context
        assert result.success is False
        assert result.error == error
    
    def test_documents_property(self):
        """Test the documents property."""
        resource = create_test_resource_config()
        documents = [Document(page_content="test", metadata={})]
        context = ProcessingContext(resource=resource, documents=documents)
        result = ProcessingResult(context=context, success=True)
        
        assert result.documents == documents
    
    def test_total_duration_property(self):
        """Test the total_duration property."""
        resource = create_test_resource_config()
        context = ProcessingContext(resource=resource)
        
        # Add metrics with durations
        start_time = time.time()
        metric1 = ProcessingMetrics(
            stage_name="stage1",
            start_time=start_time,
            end_time=start_time + 1.0
        )
        metric2 = ProcessingMetrics(
            stage_name="stage2",
            start_time=start_time + 1.0,
            end_time=start_time + 2.5
        )
        context.add_metric(metric1)
        context.add_metric(metric2)
        
        result = ProcessingResult(context=context, success=True)
        
        assert result.total_duration == 2.5
    
    def test_total_duration_property_with_none_duration(self):
        """Test total_duration property with metrics that have None duration."""
        resource = create_test_resource_config()
        context = ProcessingContext(resource=resource)
        
        # Add metric without end_time
        metric = ProcessingMetrics(stage_name="stage1")
        context.add_metric(metric)
        
        result = ProcessingResult(context=context, success=True)
        
        assert result.total_duration == 0.0
    
    def test_stage_count_property(self):
        """Test the stage_count property."""
        resource = create_test_resource_config()
        context = ProcessingContext(resource=resource)
        
        # Add multiple metrics
        metric1 = ProcessingMetrics(stage_name="stage1")
        metric2 = ProcessingMetrics(stage_name="stage2")
        context.add_metric(metric1)
        context.add_metric(metric2)
        
        result = ProcessingResult(context=context, success=True)
        
        assert result.stage_count == 2


class TestProcessingStage:
    """Test the ProcessingStage abstract base class."""
    
    def test_is_abstract_base_class(self):
        """Test that ProcessingStage is an abstract base class."""
        assert issubclass(ProcessingStage, ABC)
    
    def test_cannot_instantiate_directly(self):
        """Test that ProcessingStage cannot be instantiated directly."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            ProcessingStage("test")
    
    def test_has_required_abstract_methods(self):
        """Test that ProcessingStage has the required abstract methods."""
        abstract_methods = ProcessingStage.__abstractmethods__
        
        assert 'process' in abstract_methods
        assert len(abstract_methods) == 1
    
    def test_concrete_implementation_works(self):
        """Test that a concrete implementation of ProcessingStage works."""
        class ConcreteProcessingStage(ProcessingStage):
            async def process(self, context: ProcessingContext) -> ProcessingContext:
                return context
        
        # Should be able to instantiate
        stage = ConcreteProcessingStage("test_stage")
        assert isinstance(stage, ProcessingStage)
        assert stage.name == "test_stage"
    
    def test_incomplete_implementation_fails(self):
        """Test that incomplete implementation fails to instantiate."""
        class IncompleteProcessingStage(ProcessingStage):
            pass  # Missing process method
        
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            IncompleteProcessingStage("test_stage")
    
    def test_create_metric(self):
        """Test creating processing metric."""
        class ConcreteProcessingStage(ProcessingStage):
            async def process(self, context: ProcessingContext) -> ProcessingContext:
                return context
        
        stage = ConcreteProcessingStage("test_stage")
        resource = create_test_resource_config()
        documents = [Document(page_content="test", metadata={})]
        context = ProcessingContext(resource=resource, documents=documents)
        
        metric = stage.create_metric(context)
        
        assert metric.stage_name == "test_stage"
        assert metric.input_count == 1
        assert metric.start_time is not None
        assert metric.end_time is None
        assert metric.output_count == 0
        assert metric.errors == []
    
    def test_finish_metric_success(self):
        """Test finishing metric for successful processing."""
        class ConcreteProcessingStage(ProcessingStage):
            async def process(self, context: ProcessingContext) -> ProcessingContext:
                return context
        
        stage = ConcreteProcessingStage("test_stage")
        resource = create_test_resource_config()
        documents = [Document(page_content="test", metadata={})]
        context = ProcessingContext(resource=resource, documents=documents)
        metric = ProcessingMetrics(stage_name="test_stage")
        
        stage.finish_metric(metric, context)
        
        assert metric.end_time is not None
        assert metric.output_count == 1
        assert metric.errors == []
    
    def test_finish_metric_with_error(self):
        """Test finishing metric with error."""
        class ConcreteProcessingStage(ProcessingStage):
            async def process(self, context: ProcessingContext) -> ProcessingContext:
                return context
        
        stage = ConcreteProcessingStage("test_stage")
        resource = create_test_resource_config()
        context = ProcessingContext(resource=resource)
        metric = ProcessingMetrics(stage_name="test_stage")
        error = "Test error"
        
        stage.finish_metric(metric, context, error)
        
        assert metric.end_time is not None
        assert metric.output_count == 0
        assert metric.errors == [error]
    
    @pytest.mark.asyncio
    async def test_call_success(self):
        """Test calling stage successfully."""
        class ConcreteProcessingStage(ProcessingStage):
            async def process(self, context: ProcessingContext) -> ProcessingContext:
                # Modify context
                context.metadata["processed"] = True
                return context
        
        stage = ConcreteProcessingStage("test_stage")
        resource = create_test_resource_config()
        context = ProcessingContext(resource=resource)
        
        result = await stage(context)
        
        assert result.metadata["processed"] is True
        assert len(result.metrics) == 1
        metric = result.metrics[0]
        assert metric.stage_name == "test_stage"
        assert metric.status == ProcessingStatus.COMPLETED
    
    @pytest.mark.asyncio
    async def test_call_with_exception(self):
        """Test calling stage with exception."""
        class ConcreteProcessingStage(ProcessingStage):
            async def process(self, context: ProcessingContext) -> ProcessingContext:
                raise ValueError("Test error")
        
        stage = ConcreteProcessingStage("test_stage")
        resource = create_test_resource_config()
        context = ProcessingContext(resource=resource)
        
        with pytest.raises(ValueError, match="Test error"):
            await stage(context)
        
        # Check that metric was still created and finished with error
        assert len(context.metrics) == 1
        metric = context.metrics[0]
        assert metric.stage_name == "test_stage"
        assert metric.status == ProcessingStatus.FAILED
        assert "Test error" in metric.errors
    
    def test_or_operator_with_stage(self):
        """Test | operator with another stage."""
        class ConcreteProcessingStage(ProcessingStage):
            async def process(self, context: ProcessingContext) -> ProcessingContext:
                return context
        
        stage1 = ConcreteProcessingStage("stage1")
        stage2 = ConcreteProcessingStage("stage2")
        
        pipeline = stage1 | stage2
        
        assert hasattr(pipeline, 'stages')
        assert len(pipeline.stages) == 2
        assert pipeline.stages[0] == stage1
        assert pipeline.stages[1] == stage2
        assert pipeline.name == "stage1_to_stage2"
    
    def test_or_operator_with_pipeline(self):
        """Test | operator with existing pipeline."""
        class ConcreteProcessingStage(ProcessingStage):
            async def process(self, context: ProcessingContext) -> ProcessingContext:
                return context
        
        stage1 = ConcreteProcessingStage("stage1")
        stage2 = ConcreteProcessingStage("stage2")
        stage3 = ConcreteProcessingStage("stage3")
        
        pipeline1 = stage1 | stage2
        pipeline2 = pipeline1 | stage3
        
        assert len(pipeline2.stages) == 3
        assert pipeline2.stages[0] == stage1
        assert pipeline2.stages[1] == stage2
        assert pipeline2.stages[2] == stage3
        assert pipeline2.name == "stage1_to_stage2_to_stage3"
    
    def test_ror_operator_with_stage(self):
        """Test right-side | operator with another stage."""
        class ConcreteProcessingStage(ProcessingStage):
            async def process(self, context: ProcessingContext) -> ProcessingContext:
                return context
        
        stage1 = ConcreteProcessingStage("stage1")
        stage2 = ConcreteProcessingStage("stage2")
        
        pipeline = stage2.__ror__(stage1)
        
        assert hasattr(pipeline, 'stages')
        assert len(pipeline.stages) == 2
        assert pipeline.stages[0] == stage1
        assert pipeline.stages[1] == stage2
        assert pipeline.name == "stage1_to_stage2"
    
    def test_ror_operator_with_pipeline(self):
        """Test right-side | operator with existing pipeline."""
        class ConcreteProcessingStage(ProcessingStage):
            async def process(self, context: ProcessingContext) -> ProcessingContext:
                return context
        
        stage1 = ConcreteProcessingStage("stage1")
        stage2 = ConcreteProcessingStage("stage2")
        stage3 = ConcreteProcessingStage("stage3")
        
        pipeline1 = stage1 | stage2
        pipeline2 = stage3.__ror__(pipeline1)
        
        assert len(pipeline2.stages) == 3
        assert pipeline2.stages[0] == stage1
        assert pipeline2.stages[1] == stage2
        assert pipeline2.stages[2] == stage3
        assert pipeline2.name == "stage1_to_stage2_to_stage3"
    
    def test_or_operator_not_implemented(self):
        """Test | operator with unsupported type."""
        class ConcreteProcessingStage(ProcessingStage):
            async def process(self, context: ProcessingContext) -> ProcessingContext:
                return context
        
        stage = ConcreteProcessingStage("test_stage")
        
        # The | operator should raise TypeError for unsupported types
        with pytest.raises(TypeError):
            stage | "not_a_stage"


class TestProcessingStageEdgeCases:
    """Test edge cases for ProcessingStage."""
    
    def test_stage_with_properties(self):
        """Test ProcessingStage with properties."""
        class StageWithProperties(ProcessingStage):
            def __init__(self, name: str):
                super().__init__(name)
                self._config = {}
            
            @property
            def config(self):
                return self._config
            
            @config.setter
            def config(self, value):
                self._config = value
            
            async def process(self, context: ProcessingContext) -> ProcessingContext:
                return context
        
        stage = StageWithProperties("test_stage")
        stage.config = {"key": "value"}
        
        assert stage.config == {"key": "value"}
    
    def test_stage_with_class_methods(self):
        """Test ProcessingStage with class methods."""
        class StageWithClassMethods(ProcessingStage):
            @classmethod
            def get_supported_types(cls):
                return ["type1", "type2"]
            
            async def process(self, context: ProcessingContext) -> ProcessingContext:
                return context
        
        stage = StageWithClassMethods("test_stage")
        types = StageWithClassMethods.get_supported_types()
        
        assert types == ["type1", "type2"]
    
    def test_stage_with_static_methods(self):
        """Test ProcessingStage with static methods."""
        class StageWithStaticMethods(ProcessingStage):
            @staticmethod
            def validate_config(config: dict) -> bool:
                return "url" in config
            
            async def process(self, context: ProcessingContext) -> ProcessingContext:
                return context
        
        stage = StageWithStaticMethods("test_stage")
        assert StageWithStaticMethods.validate_config({"url": "test"}) is True
        assert StageWithStaticMethods.validate_config({}) is False
    
    def test_multiple_inheritance(self):
        """Test ProcessingStage with multiple inheritance."""
        class Mixin:
            def extra_method(self):
                return "mixin"
        
        class MultiInheritanceStage(ProcessingStage, Mixin):
            async def process(self, context: ProcessingContext) -> ProcessingContext:
                return context
        
        stage = MultiInheritanceStage("test_stage")
        assert isinstance(stage, ProcessingStage)
        assert isinstance(stage, Mixin)
        assert stage.extra_method() == "mixin"
    
    @pytest.mark.asyncio
    async def test_stage_with_async_context_manager(self):
        """Test ProcessingStage with async context manager behavior."""
        class AsyncContextStage(ProcessingStage):
            async def process(self, context: ProcessingContext) -> ProcessingContext:
                context.metadata["async_processed"] = True
                return context
        
        stage = AsyncContextStage("test_stage")
        resource = create_test_resource_config()
        context = ProcessingContext(resource=resource)
        
        result = await stage(context)
        
        assert result.metadata["async_processed"] is True
    
    def test_stage_name_validation(self):
        """Test that stage name is properly set and accessible."""
        class ConcreteProcessingStage(ProcessingStage):
            async def process(self, context: ProcessingContext) -> ProcessingContext:
                return context
        
        stage = ConcreteProcessingStage("custom_stage_name")
        
        assert stage.name == "custom_stage_name"
        assert hasattr(stage, 'name')
    
    def test_stage_metric_creation_with_empty_context(self):
        """Test metric creation with empty context."""
        class ConcreteProcessingStage(ProcessingStage):
            async def process(self, context: ProcessingContext) -> ProcessingContext:
                return context
        
        stage = ConcreteProcessingStage("test_stage")
        resource = create_test_resource_config()
        context = ProcessingContext(resource=resource)  # Empty documents
        
        metric = stage.create_metric(context)
        
        assert metric.input_count == 0
        assert metric.stage_name == "test_stage"
