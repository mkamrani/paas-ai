"""
Document processing pipeline for RAG system.

Provides a pipeline pattern for document ingestion with composable stages:
- Load: Fetch documents from various sources  
- Validate: Check document validity and requirements
- Split: Break documents into chunks
- Enrich: Add metadata and transformations
- Filter: Remove unwanted content
- Store: Persist to vector stores

Pipeline can be composed and configured for different workflows.

Unix-pipe style composition:
    pipeline = LoadStage() | ValidateStage() | SplitStage() | EnrichStage()

Or use helper functions for parameter passing:
    pipeline = load_stage() | validate_stage() | split_stage() | enrich_stage(custom_enrichers=[...])
"""

from .pipeline import (
    ProcessingPipeline,
    load_stage,
    validate_stage,
    split_stage, 
    enrich_stage,
    filter_stage
)
from .base import ProcessingStage, ProcessingContext, ProcessingResult
from .stages import (
    LoadStage,
    ValidateStage, 
    SplitStage,
    EnrichStage,
    FilterStage
)

__all__ = [
    'ProcessingPipeline',
    'ProcessingStage',
    'ProcessingContext', 
    'ProcessingResult',
    'LoadStage',
    'ValidateStage',
    'SplitStage', 
    'EnrichStage',
    'FilterStage',
    'load_stage',
    'validate_stage',
    'split_stage',
    'enrich_stage', 
    'filter_stage'
] 