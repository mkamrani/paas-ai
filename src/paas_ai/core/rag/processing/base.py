"""
Base interfaces for document processing pipeline.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
import time
import uuid

from langchain_core.documents import Document

from ..config import ResourceConfig


class ProcessingStatus(str, Enum):
    """Status of processing stages."""
    PENDING = "pending"
    RUNNING = "running" 
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class ProcessingMetrics:
    """Metrics for processing stages."""
    stage_name: str
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    input_count: int = 0
    output_count: int = 0
    errors: List[str] = field(default_factory=list)
    
    @property
    def duration(self) -> Optional[float]:
        """Get processing duration in seconds."""
        if self.end_time:
            return self.end_time - self.start_time
        return None
    
    @property 
    def status(self) -> ProcessingStatus:
        """Get current processing status."""
        if self.errors:
            return ProcessingStatus.FAILED
        elif self.end_time:
            return ProcessingStatus.COMPLETED
        else:
            return ProcessingStatus.RUNNING


@dataclass
class ProcessingContext:
    """Context passed between processing stages."""
    resource: ResourceConfig
    documents: List[Document] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    metrics: List[ProcessingMetrics] = field(default_factory=list)
    pipeline_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    def add_metric(self, metric: ProcessingMetrics) -> None:
        """Add processing metric."""
        self.metrics.append(metric)
    
    def get_metric(self, stage_name: str) -> Optional[ProcessingMetrics]:
        """Get metric for specific stage."""
        for metric in self.metrics:
            if metric.stage_name == stage_name:
                return metric
        return None


@dataclass
class ProcessingResult:
    """Result of pipeline processing."""
    context: ProcessingContext
    success: bool
    error: Optional[str] = None
    
    @property
    def documents(self) -> List[Document]:
        """Get final processed documents."""
        return self.context.documents
    
    @property
    def total_duration(self) -> float:
        """Get total processing duration."""
        return sum(m.duration or 0 for m in self.context.metrics)
    
    @property
    def stage_count(self) -> int:
        """Get number of processing stages."""
        return len(self.context.metrics)


class ProcessingStage(ABC):
    """Base class for pipeline processing stages."""
    
    def __init__(self, name: str):
        self.name = name
    
    @abstractmethod
    async def process(self, context: ProcessingContext) -> ProcessingContext:
        """Process the context and return modified context."""
        pass
    
    def create_metric(self, context: ProcessingContext) -> ProcessingMetrics:
        """Create processing metric for this stage."""
        return ProcessingMetrics(
            stage_name=self.name,
            input_count=len(context.documents)
        )
    
    def finish_metric(self, metric: ProcessingMetrics, context: ProcessingContext, error: Optional[str] = None) -> None:
        """Finish processing metric."""
        metric.end_time = time.time()
        metric.output_count = len(context.documents)
        if error:
            metric.errors.append(error)
    
    async def __call__(self, context: ProcessingContext) -> ProcessingContext:
        """Execute the processing stage with metrics."""
        metric = self.create_metric(context)
        context.add_metric(metric)
        
        try:
            result = await self.process(context)
            self.finish_metric(metric, result)
            return result
        except Exception as e:
            self.finish_metric(metric, context, str(e))
            raise
    
    def __or__(self, other: 'ProcessingStage') -> 'ProcessingPipeline':
        """Compose stages using the | operator."""
        from .pipeline import ProcessingPipeline
        
        if isinstance(other, ProcessingPipeline):
            # Stage | Pipeline -> Pipeline with stage prepended
            return ProcessingPipeline([self] + other.stages, name=f"{self.name}_to_{other.name}")
        elif isinstance(other, ProcessingStage):
            # Stage | Stage -> Pipeline with both stages
            return ProcessingPipeline([self, other], name=f"{self.name}_to_{other.name}")
        else:
            return NotImplemented
    
    def __ror__(self, other: 'ProcessingStage') -> 'ProcessingPipeline':
        """Right-side composition (should rarely be called due to left-to-right evaluation)."""
        from .pipeline import ProcessingPipeline
        
        if isinstance(other, ProcessingPipeline):
            # Pipeline | Stage -> Pipeline with stage appended
            return ProcessingPipeline(other.stages + [self], name=f"{other.name}_to_{self.name}")
        elif isinstance(other, ProcessingStage):
            # Stage | Stage -> Pipeline with both stages
            return ProcessingPipeline([other, self], name=f"{other.name}_to_{self.name}")
        else:
            return NotImplemented 