"""
Document loading stage for processing pipeline.
"""

from typing import List
from langchain_core.documents import Document

from ..base import ProcessingStage, ProcessingContext
from ...loaders import DocumentLoaderFactory


class LoadStage(ProcessingStage):
    """Stage for loading documents from various sources."""
    
    def __init__(self, name: str = "load"):
        super().__init__(name)
    
    async def process(self, context: ProcessingContext) -> ProcessingContext:
        """Load documents from the resource."""
        resource = context.resource
        
        # Create loader using factory
        loader = DocumentLoaderFactory.create_loader(resource.loader, resource.url)
        
        # Load documents
        documents = loader.load()
        
        if not documents:
            raise ValueError(f"No documents loaded from {resource.url}")
        
        # Update context
        context.documents = documents
        context.metadata.update({
            'source_loader': resource.loader.type,
            'document_count': len(documents)
        })
        
        return context 