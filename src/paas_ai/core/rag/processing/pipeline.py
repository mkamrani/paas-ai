"""
Document processing pipeline engine.
"""

import asyncio
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from .base import ProcessingStage, ProcessingContext, ProcessingResult
from ..config import ResourceConfig


class ProcessingPipeline:
    """Pipeline for composing and executing processing stages."""
    
    def __init__(self, stages: List[ProcessingStage], name: str = "pipeline"):
        if stages is None:
            raise TypeError("stages cannot be None")
        self.stages = stages
        self.name = name
    
    def __or__(self, other) -> 'ProcessingPipeline':
        """Compose pipelines and stages using the | operator."""
        if isinstance(other, ProcessingPipeline):
            # Pipeline | Pipeline -> Combined pipeline
            return ProcessingPipeline(
                self.stages + other.stages, 
                name=f"{self.name}_to_{other.name}"
            )
        elif isinstance(other, ProcessingStage):
            # Pipeline | Stage -> Pipeline with stage appended
            return ProcessingPipeline(
                self.stages + [other], 
                name=f"{self.name}_to_{other.name}"
            )
        else:
            raise TypeError(f"unsupported operand type(s) for |: '{type(self).__name__}' and '{type(other).__name__}'")
    
    async def process_with_context(self, context: ProcessingContext) -> ProcessingResult:
        """Process using an existing context (allows for pre-injection of enrichers)."""
        
        print(f"Starting pipeline '{self.name}' for resource: {context.resource.url}")
        
        try:
            # Execute each stage in sequence
            for stage in self.stages:
                print(f"Executing stage: {stage.name}")
                context = await stage(context)
                
                # Log progress
                metric = context.get_metric(stage.name)
                if metric:
                    print(f"Stage '{stage.name}' completed: "
                          f"{metric.input_count} → {metric.output_count} documents "
                          f"in {metric.duration:.2f}s")
            
            total_duration = sum(m.duration or 0 for m in context.metrics)
            print(f"Pipeline completed: {len(context.documents)} documents processed "
                  f"in {total_duration:.2f}s")
            
            return ProcessingResult(context=context, success=True)
            
        except Exception as e:
            print(f"Pipeline failed: {e}")
            return ProcessingResult(
                context=context, 
                success=False, 
                error=str(e)
            )
    
    async def process(self, resource: ResourceConfig) -> ProcessingResult:
        """Process a single resource through the pipeline."""
        context = ProcessingContext(resource=resource)
        
        print(f"Starting pipeline '{self.name}' for resource: {resource.url}")
        
        try:
            # Execute each stage in sequence
            for stage in self.stages:
                print(f"Executing stage: {stage.name}")
                context = await stage(context)
                
                # Log progress
                metric = context.get_metric(stage.name)
                if metric:
                    print(f"Stage '{stage.name}' completed: "
                          f"{metric.input_count} → {metric.output_count} documents "
                          f"in {metric.duration:.2f}s")
            
            total_duration = sum(m.duration or 0 for m in context.metrics)
            print(f"Pipeline completed: {len(context.documents)} documents processed "
                  f"in {total_duration:.2f}s")
            
            return ProcessingResult(context=context, success=True)
            
        except Exception as e:
            print(f"Pipeline failed: {e}")
            return ProcessingResult(
                context=context, 
                success=False, 
                error=str(e)
            )
    
    async def process_batch(self, resources: List[ResourceConfig]) -> List[ProcessingResult]:
        """Process multiple resources in parallel."""
        print(f"Processing batch of {len(resources)} resources")
        
        # Create tasks for parallel processing
        tasks = [self.process(resource) for resource in resources]
        
        # Execute in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append(ProcessingResult(
                    context=ProcessingContext(resource=resources[i]),
                    success=False,
                    error=str(result)
                ))
            else:
                processed_results.append(result)
        
        successful = sum(1 for r in processed_results if r.success)
        print(f"Batch processing completed: {successful}/{len(resources)} successful")
        
        return processed_results


# Optional helper functions for convenient stage creation
def load_stage(**kwargs):
    """Create a LoadStage with parameters."""
    from .stages import LoadStage
    return LoadStage(**kwargs)


def validate_stage(**kwargs):
    """Create a ValidateStage with parameters."""
    from .stages import ValidateStage
    return ValidateStage(**kwargs)


def split_stage(**kwargs):
    """Create a SplitStage with parameters."""
    from .stages import SplitStage
    return SplitStage(**kwargs)


def enrich_stage(**kwargs):
    """Create an EnrichStage with parameters."""
    from .stages import EnrichStage
    return EnrichStage(**kwargs)


def filter_stage(**kwargs):
    """Create a FilterStage with parameters."""
    from .stages import FilterStage
    return FilterStage(**kwargs) 