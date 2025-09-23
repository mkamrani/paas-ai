"""
Document splitting stage for processing pipeline.
"""

import time
from typing import List
from langchain_core.documents import Document

from ..base import ProcessingStage, ProcessingContext
from ...splitters import TextSplitterFactory


class SplitStage(ProcessingStage):
    """Stage for splitting documents into chunks."""
    
    def __init__(self, name: str = "split"):
        super().__init__(name)
    
    async def process(self, context: ProcessingContext) -> ProcessingContext:
        """Split documents into chunks."""
        resource = context.resource
        documents = context.documents
        
        # Initialize split_docs as empty list
        split_docs = []
        
        if documents:
            # Create splitter using factory
            splitter = TextSplitterFactory.create_splitter(resource.splitter)
            
            # Handle different splitter types
            if hasattr(splitter, 'split_documents') and callable(getattr(splitter, 'split_documents')):
                split_docs = splitter.split_documents(documents)
                # If split_documents returns a Mock or non-list, fall back to split_text
                if not isinstance(split_docs, list):
                    split_docs = []
                    for doc in documents:
                        split_results = splitter.split_text(doc.page_content)
                        
                        # Check if returns Document objects or strings
                        if split_results and hasattr(split_results[0], 'page_content'):
                            # Returns Document objects
                            for split_doc in split_results:
                                # Merge metadata
                                merged_metadata = doc.metadata.copy()
                                merged_metadata.update(split_doc.metadata)
                                split_doc.metadata = merged_metadata
                                split_docs.append(split_doc)
                        else:
                            # Returns strings
                            for text in split_results:
                                split_doc = Document(
                                    page_content=text,
                                    metadata=doc.metadata.copy()
                                )
                                split_docs.append(split_doc)
            else:
                # For splitters that only have split_text method
                for doc in documents:
                    split_results = splitter.split_text(doc.page_content)
                    
                    # Check if returns Document objects or strings
                    if split_results and hasattr(split_results[0], 'page_content'):
                        # Returns Document objects
                        for split_doc in split_results:
                            # Merge metadata
                            merged_metadata = doc.metadata.copy()
                            merged_metadata.update(split_doc.metadata)
                            split_doc.metadata = merged_metadata
                            split_docs.append(split_doc)
                    else:
                        # Returns strings
                        for text in split_results:
                            split_doc = Document(
                                page_content=text,
                                metadata=doc.metadata.copy()
                            )
                            split_docs.append(split_doc)
        
        # Update context
        context.documents = split_docs
        context.metadata.update({
            'splitter_type': resource.splitter.type,
            'chunk_count': len(split_docs),
            'chunk_size': resource.splitter.chunk_size,
            'chunk_overlap': resource.splitter.chunk_overlap
        })
        
        return context 